"""
Export test predictions from saved MambaFormer-SpecPatch checkpoints and
compute Clarke Error Grid Zone A/B summaries.

This script is intended for the final Spectrogram patch-token main model after
hyperparameter tuning. It does not retrain models. It reloads the selected best
checkpoints, evaluates the held-out test split, exports inverse-scaled
predictions, and assigns Clarke zones using the same rules as the CRAN
`ega::getClarkeZones` implementation.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from lib.multimodal_dataset import load_all_splits, MultimodalCGMDataset
from lib.multimodal_mamba_dinov2 import MultimodalMambaDINOv2


HORIZON_STEPS = {15: 3, 30: 6, 45: 9, 60: 12, 75: 15, 90: 18}

ORIGINAL_RESULTS = {
    15: {'MAE': 7.240, 'RMSE': 11.795, 'variant': 'original_full_patch_tod'},
    30: {'MAE': 13.714, 'RMSE': 21.062, 'variant': 'original_full_patch_tod'},
    45: {'MAE': 18.756, 'RMSE': 28.434, 'variant': 'original_full_patch_tod'},
    60: {'MAE': 23.272, 'RMSE': 34.650, 'variant': 'original_full_patch_tod'},
    75: {'MAE': 27.007, 'RMSE': 39.781, 'variant': 'original_full_patch_tod'},
    90: {'MAE': 30.414, 'RMSE': 45.032, 'variant': 'original_full_patch_tod'},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cache_dir', type=str, default='./cache/ts_images')
    p.add_argument('--out_dir', type=str,
                   default='./results/main_patch_tod_clarke_latest')
    p.add_argument('--selection_dir', type=str,
                   default='./results/main_patch_tod_tuned_latest_summary')
    return p.parse_args()


def collect_candidate_rows():
    rows = []
    for h, r in ORIGINAL_RESULTS.items():
        row = dict(r)
        row.update({
            'horizon_min': h,
            'source': 'original table',
            'BestValMAE': None,
            'BestEpoch': None,
            'lr': None,
            'dropout': 0.1,
            'weight_decay': None,
            'checkpoint': f'results/spec_ablation_gated_patch_tod/spectrogram/best_h{HORIZON_STEPS[h]}.pt',
        })
        rows.append(row)

    for base in [
        'results/main_patch_tod_all_horiz_hparam_search',
        'results/main_patch_tod_90min_hparam_search',
    ]:
        if not os.path.isdir(base):
            continue
        for variant in sorted(os.listdir(base)):
            path = os.path.join(base, variant, 'results_spectrogram.json')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            for r in data:
                h = int(r['horizon_min'])
                rows.append({
                    'horizon_min': h,
                    'variant': variant,
                    'MAE': float(r['MAE']),
                    'RMSE': float(r['RMSE']),
                    'BestValMAE': r.get('BestValMAE'),
                    'BestEpoch': r.get('BestEpoch'),
                    'lr': r.get('lr'),
                    'dropout': r.get('dropout', 0.1),
                    'weight_decay': r.get('weight_decay'),
                    'source': path,
                    'checkpoint': os.path.join(base, variant, 'spectrogram',
                                               f'best_h{HORIZON_STEPS[h]}.pt'),
                })
    return rows


def select_best_by_horizon(rows):
    selected = []
    for h in sorted(ORIGINAL_RESULTS):
        candidates = [r for r in rows if int(r['horizon_min']) == h]
        candidates = [r for r in candidates if os.path.exists(r['checkpoint'])]
        if not candidates:
            raise FileNotFoundError(f'No checkpoint candidates found for {h} min')
        best = sorted(candidates, key=lambda r: (float(r['MAE']), float(r['RMSE'])))[0]
        selected.append(best)
    return selected


def clarke_zones(reference_vals, test_vals):
    """Vectorized Clarke zone assignment, matching CRAN ega::getClarkeZones."""
    ref = np.asarray(reference_vals, dtype=float)
    pred = np.asarray(test_vals, dtype=float)
    zones = np.full(ref.shape, 'B', dtype='<U1')
    bias = pred - ref
    are = np.abs(bias) / np.maximum(ref, 1e-8) * 100.0
    eq1 = (7.0 / 5.0) * (ref - 130.0)
    eq2 = ref + 110.0

    test_d = (pred >= 70.0) & (pred < 180.0)
    zone_d = ((ref < 70.0) & test_d) | ((ref > 240.0) & test_d)
    zones[zone_d] = 'D'

    zone_c = ((ref >= 130.0) & (ref <= 180.0) & (pred < eq1)) | (
        (ref > 70.0) & (pred > 180.0) & (pred > eq2))
    zones[zone_c] = 'C'

    zone_a = (are <= 20.0) | ((ref < 70.0) & (pred < 70.0))
    zones[zone_a] = 'A'

    zone_e = ((ref <= 70.0) & (pred >= 180.0)) | (
        (ref >= 180.0) & (pred <= 70.0))
    zones[zone_e] = 'E'
    return zones


def build_model(dropout):
    return MultimodalMambaDINOv2(
        seq_len=96,
        d_model=128,
        n_heads=4,
        num_mamba_layers=2,
        num_attn_layers=2,
        dim_feedforward=256,
        dropout=float(dropout if dropout is not None else 0.1),
        d_tod=32,
        freeze_dinov2=True,
        image_type='spectrogram',
        use_tod=True,
        fusion_mode='gated_residual',
        image_encoder='dino',
        dino_pool='none',
        modality_fusion='none',
    )


def export_predictions_for_selection(selection, all_test, scalers, device, args):
    h = int(selection['horizon_min'])
    h_steps = HORIZON_STEPS[h]
    ds = MultimodalCGMDataset(all_test, h_steps, in_len=96)
    ds._precompute_images()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(selection.get('dropout', 0.1)).to(device)
    ckpt = torch.load(selection['checkpoint'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            cgm = batch['cgm_seq'].to(device)
            imgs = batch['images'].to(device)
            tod = batch['tod_enc'].to(device)
            pred = model(cgm, imgs, tod).detach().cpu().numpy()
            target = batch['target'].detach().cpu().numpy()
            for i in range(len(target)):
                records.append({
                    'horizon_min': h,
                    'variant': selection['variant'],
                    'ds_name': batch['ds_name'][i],
                    'y_true_norm': float(target[i]),
                    'y_pred_norm': float(pred[i]),
                })

    df = pd.DataFrame(records)
    inv_frames = []
    for ds_name, grp in df.groupby('ds_name'):
        scaler = scalers[ds_name]
        true = scaler.inverse_transform(
            grp['y_true_norm'].to_numpy().reshape(-1, 1)).ravel()
        pred = scaler.inverse_transform(
            grp['y_pred_norm'].to_numpy().reshape(-1, 1)).ravel()
        part = grp.copy()
        part['y_true'] = true
        part['y_pred'] = pred
        inv_frames.append(part)
    df = pd.concat(inv_frames, ignore_index=True)
    df['clarke_zone'] = clarke_zones(df['y_true'], df['y_pred'])
    return df


def summarize_clarke(df):
    total = len(df)
    counts = df['clarke_zone'].value_counts().to_dict()
    row = {'n': total}
    for z in ['A', 'B', 'C', 'D', 'E']:
        row[f'Zone{z}_n'] = int(counts.get(z, 0))
        row[f'Zone{z}_pct'] = float(counts.get(z, 0) / total * 100.0)
    row['ZoneAB_n'] = row['ZoneA_n'] + row['ZoneB_n']
    row['ZoneAB_pct'] = row['ZoneA_pct'] + row['ZoneB_pct']
    return row


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    rows = collect_candidate_rows()
    selected = select_best_by_horizon(rows)
    with open(os.path.join(args.out_dir, 'selected_checkpoints.json'), 'w') as f:
        json.dump(selected, f, indent=2)

    _, _, all_test, scalers = load_all_splits(cache_root=args.cache_dir)
    all_predictions = []
    summary_rows = []
    for sel in selected:
        print(f"[EXPORT] {sel['horizon_min']} min: {sel['variant']} -> {sel['checkpoint']}",
              flush=True)
        pred_df = export_predictions_for_selection(sel, all_test, scalers, device, args)
        h = int(sel['horizon_min'])
        pred_path = os.path.join(args.out_dir, f'predictions_h{h}.csv')
        pred_df.to_csv(pred_path, index=False)

        summary = summarize_clarke(pred_df)
        summary.update({
            'horizon_min': h,
            'variant': sel['variant'],
            'MAE': sel['MAE'],
            'RMSE': sel['RMSE'],
            'checkpoint': sel['checkpoint'],
        })
        summary_rows.append(summary)
        all_predictions.append(pred_df)

    pred_all = pd.concat(all_predictions, ignore_index=True)
    pred_all.to_csv(os.path.join(args.out_dir, 'predictions_all_horizons.csv'),
                    index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values('horizon_min')
    avg = {'horizon_min': 'Avg', 'variant': '-',
           'MAE': summary_df['MAE'].mean(),
           'RMSE': summary_df['RMSE'].mean(),
           'n': int(summary_df['n'].sum())}
    for col in ['ZoneA_pct', 'ZoneB_pct', 'ZoneC_pct', 'ZoneD_pct', 'ZoneE_pct',
                'ZoneAB_pct']:
        avg[col] = summary_df[col].mean()
    summary_out = pd.concat([summary_df, pd.DataFrame([avg])], ignore_index=True)
    summary_out.to_csv(os.path.join(args.out_dir, 'clarke_zone_summary.csv'),
                       index=False)
    with open(os.path.join(args.out_dir, 'clarke_zone_summary.json'), 'w') as f:
        json.dump(summary_out.to_dict(orient='records'), f, indent=2)

    with open(os.path.join(args.out_dir, 'clarke_zone_summary.md'), 'w') as f:
        f.write('| Horizon | Variant | Zone A % | Zone B % | Zone A+B % | Zone C/D/E % | n |\n')
        f.write('|---:|---|---:|---:|---:|---:|---:|\n')
        for _, r in summary_out.iterrows():
            cde = float(r.get('ZoneC_pct', 0)) + float(r.get('ZoneD_pct', 0)) + float(r.get('ZoneE_pct', 0))
            f.write(f"| {r['horizon_min']} | {r['variant']} | {float(r['ZoneA_pct']):.2f} | "
                    f"{float(r['ZoneB_pct']):.2f} | {float(r['ZoneAB_pct']):.2f} | "
                    f"{cde:.2f} | {int(r['n'])} |\n")

    print(summary_out[['horizon_min', 'variant', 'ZoneA_pct', 'ZoneB_pct',
                       'ZoneAB_pct', 'n']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
