"""
Train/evaluate the sequence-only MambaFormer-96 baseline and compute Clarke
Error Grid Zone A/B summaries.

The baseline follows the original GlucoBench setup:
  - input window: 96 CGM steps
  - pure CGM sequence model, no image branch and no time-of-day branch
  - MambaFormer architecture from lib.mambaformer_model
  - pred_len=12 for 15/30/45/60 min
  - pred_len=18 for 75/90 min

For clinical comparison with the final Spectrogram patch-token model, test
predictions are exported on the same horizon-specific sliding windows used by
the multimodal experiments.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(__file__))
from lib.mambaformer_model import MambaFormer
from lib.multimodal_dataset import load_all_splits


HORIZON_STEPS = {15: 3, 30: 6, 45: 9, 60: 12, 75: 15, 90: 18}
BASELINE_TABLE = {
    15: {'MAE': 9.33, 'RMSE': 13.66},
    30: {'MAE': 15.28, 'RMSE': 21.83},
    45: {'MAE': 19.75, 'RMSE': 29.40},
    60: {'MAE': 24.11, 'RMSE': 35.68},
    75: {'MAE': 28.67, 'RMSE': 41.88},
    90: {'MAE': 31.67, 'RMSE': 46.41},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--in_len', type=int, default=96)
    p.add_argument('--stride', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cache_dir', type=str, default='./cache/ts_images')
    p.add_argument('--out_dir', type=str,
                   default='./results/mambaformer96_baseline_clarke')
    p.add_argument('--force_train', action='store_true')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiStepSequenceDataset(Dataset):
    def __init__(self, segs, in_len, out_len, stride):
        self.windows = []
        for seg_id, entry in segs.items():
            series = entry['series'].astype(np.float32)
            max_start = len(series) - in_len - out_len
            for start in range(0, max_start + 1, stride):
                self.windows.append((series, start))
        self.in_len = in_len
        self.out_len = out_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        series, start = self.windows[idx]
        x = series[start:start + self.in_len]
        y = series[start + self.in_len:start + self.in_len + self.out_len]
        return torch.from_numpy(x), torch.from_numpy(y)


class HorizonEvalDataset(Dataset):
    def __init__(self, segs, in_len, horizon, stride):
        self.windows = []
        for seg_id, entry in segs.items():
            series = entry['series'].astype(np.float32)
            max_start = len(series) - in_len - horizon
            for start in range(0, max_start + 1, stride):
                self.windows.append((entry, seg_id, start))
        self.in_len = in_len
        self.horizon = horizon

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        entry, seg_id, start = self.windows[idx]
        series = entry['series'].astype(np.float32)
        x = series[start:start + self.in_len]
        target = series[start + self.in_len + self.horizon - 1]
        return {
            'cgm_seq': torch.from_numpy(x),
            'target': torch.tensor(target, dtype=torch.float32),
            'ds_name': entry['ds_name'],
            'seg_id': seg_id,
            'start': start,
        }


def build_model(in_len, out_len):
    return MambaFormer(
        seq_len=in_len,
        pred_len=out_len,
        d_model=128,
        n_heads=4,
        num_mamba_layers=2,
        num_attn_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
    )


def train_or_load(out_len, all_train, all_val, args, device):
    ckpt_dir = os.path.join(args.out_dir, f'baseline_predlen{out_len}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')

    model = build_model(args.in_len, out_len).to(device)
    if os.path.exists(ckpt_path) and not args.force_train:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        return model, ckpt_path, ckpt

    train_ds = MultiStepSequenceDataset(all_train, args.in_len, out_len, args.stride)
    val_ds = MultiStepSequenceDataset(all_val, args.in_len, out_len, args.stride)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    best_val, best_epoch, patience_cnt = float('inf'), 0, 0

    print(f'[TRAIN] MambaFormer-96 pred_len={out_len} '
          f'train={len(train_ds)} val={len(val_ds)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            n_train += x.shape[0]
        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                loss = criterion(model(x), y)
                val_loss += loss.item() * x.shape[0]
                n_val += x.shape[0]
        train_loss /= max(1, n_train)
        val_loss /= max(1, n_val)
        print(f'[EPOCH {epoch:02d}] pred_len={out_len} '
              f'train_loss={train_loss:.6f} val_loss={val_loss:.6f}',
              flush=True)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'best_val_loss': best_val,
                'state_dict': model.state_dict(),
                'args': vars(args),
                'pred_len': out_len,
            }, ckpt_path)
            print(f'[SAVE] {ckpt_path}', flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'[EARLY_STOP] pred_len={out_len} best_epoch={best_epoch} '
                      f'best_val_loss={best_val:.6f}', flush=True)
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt_path, ckpt


def inverse_by_dataset(df, scalers):
    frames = []
    for ds_name, grp in df.groupby('ds_name'):
        scaler = scalers[ds_name]
        true = scaler.inverse_transform(
            grp['y_true_norm'].to_numpy().reshape(-1, 1)).ravel()
        pred = scaler.inverse_transform(
            grp['y_pred_norm'].to_numpy().reshape(-1, 1)).ravel()
        part = grp.copy()
        part['y_true'] = true
        part['y_pred'] = pred
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def compute_metrics(y_true, y_pred):
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    mask = t != 0
    mape = float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100.0)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


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


def summarize_clarke(df):
    total = len(df)
    counts = df['clarke_zone'].value_counts().to_dict()
    row = {'n': total}
    for zone in ['A', 'B', 'C', 'D', 'E']:
        row[f'Zone{zone}_n'] = int(counts.get(zone, 0))
        row[f'Zone{zone}_pct'] = float(counts.get(zone, 0) / total * 100.0)
    row['ZoneAB_n'] = row['ZoneA_n'] + row['ZoneB_n']
    row['ZoneAB_pct'] = row['ZoneA_pct'] + row['ZoneB_pct']
    return row


def export_horizon(model, pred_len, checkpoint, horizon_min, all_test, scalers,
                   args, device):
    horizon = HORIZON_STEPS[horizon_min]
    ds = HorizonEvalDataset(all_test, args.in_len, horizon, args.stride)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            x = batch['cgm_seq'].to(device)
            pred_seq = model(x).detach().cpu().numpy()
            pred = pred_seq[:, horizon - 1]
            target = batch['target'].detach().cpu().numpy()
            for i in range(len(target)):
                records.append({
                    'horizon_min': horizon_min,
                    'variant': 'MambaFormer-96',
                    'pred_len': pred_len,
                    'checkpoint': checkpoint,
                    'ds_name': batch['ds_name'][i],
                    'seg_id': batch['seg_id'][i],
                    'start': int(batch['start'][i]),
                    'y_true_norm': float(target[i]),
                    'y_pred_norm': float(pred[i]),
                })
    df = inverse_by_dataset(pd.DataFrame(records), scalers)
    df['clarke_zone'] = clarke_zones(df['y_true'], df['y_pred'])
    return df


def load_main_clarke(out_dir):
    path = './results/main_patch_tod_clarke_final/clarke_zone_summary.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df['horizon_min'].astype(str) != 'Avg'].copy()
    df['horizon_min'] = df['horizon_min'].astype(int)
    return df


def write_markdown(summary_out, metrics_out, args):
    summary_path = os.path.join(args.out_dir, 'clarke_zone_summary.md')
    with open(summary_path, 'w') as f:
        f.write('| Horizon | Variant | Zone A % | Zone B % | Zone A+B % | Zone C/D/E % | n |\n')
        f.write('|---:|---|---:|---:|---:|---:|---:|\n')
        for _, r in summary_out.iterrows():
            cde = float(r.get('ZoneC_pct', 0)) + float(r.get('ZoneD_pct', 0)) + float(r.get('ZoneE_pct', 0))
            f.write(f"| {r['horizon_min']} | {r['variant']} | {float(r['ZoneA_pct']):.2f} | "
                    f"{float(r['ZoneB_pct']):.2f} | {float(r['ZoneAB_pct']):.2f} | "
                    f"{cde:.2f} | {int(r['n'])} |\n")

    metrics_path = os.path.join(args.out_dir, 'baseline_metrics.md')
    with open(metrics_path, 'w') as f:
        f.write('| Horizon | MAE | RMSE | MAPE % | R2 | Table MAE | Table RMSE |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
        for _, r in metrics_out.iterrows():
            h = int(r['horizon_min'])
            f.write(f"| {h} | {float(r['MAE']):.3f} | {float(r['RMSE']):.3f} | "
                    f"{float(r['MAPE']):.2f} | {float(r['R2']):.4f} | "
                    f"{BASELINE_TABLE[h]['MAE']:.2f} | {BASELINE_TABLE[h]['RMSE']:.2f} |\n")

    main_df = load_main_clarke(args.out_dir)
    if main_df is None:
        return
    comp = summary_out[summary_out['horizon_min'].astype(str) != 'Avg'].copy()
    comp['horizon_min'] = comp['horizon_min'].astype(int)
    comp = comp.merge(main_df, on='horizon_min', suffixes=('_baseline', '_main'))
    comp_path = os.path.join(args.out_dir, 'baseline_vs_specpatch_clarke.md')
    with open(comp_path, 'w') as f:
        f.write('| Horizon | Baseline Zone A % | SpecPatch Zone A % | Delta A | '
                'Baseline A+B % | SpecPatch A+B % | Delta A+B |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
        for _, r in comp.iterrows():
            da = float(r['ZoneA_pct_main']) - float(r['ZoneA_pct_baseline'])
            dab = float(r['ZoneAB_pct_main']) - float(r['ZoneAB_pct_baseline'])
            f.write(f"| {int(r['horizon_min'])} | {float(r['ZoneA_pct_baseline']):.2f} | "
                    f"{float(r['ZoneA_pct_main']):.2f} | {da:+.2f} | "
                    f"{float(r['ZoneAB_pct_baseline']):.2f} | "
                    f"{float(r['ZoneAB_pct_main']):.2f} | {dab:+.2f} |\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device} out_dir={args.out_dir}', flush=True)

    all_train, all_val, all_test, scalers = load_all_splits(cache_root=args.cache_dir)
    models = {}
    checkpoints = {}
    ckpt_meta = {}
    for pred_len in [12, 18]:
        model, ckpt_path, meta = train_or_load(pred_len, all_train, all_val, args, device)
        models[pred_len] = model
        checkpoints[pred_len] = ckpt_path
        ckpt_meta[pred_len] = {
            'checkpoint': ckpt_path,
            'epoch': int(meta.get('epoch', -1)),
            'best_val_loss': float(meta.get('best_val_loss', np.nan)),
        }

    with open(os.path.join(args.out_dir, 'selected_checkpoints.json'), 'w') as f:
        json.dump(ckpt_meta, f, indent=2)

    all_predictions = []
    summary_rows = []
    metric_rows = []
    pred_len_for_horizon = {15: 12, 30: 12, 45: 12, 60: 12, 75: 18, 90: 18}
    for horizon_min in [15, 30, 45, 60, 75, 90]:
        pred_len = pred_len_for_horizon[horizon_min]
        print(f'[EXPORT] horizon={horizon_min}min pred_len={pred_len}', flush=True)
        pred_df = export_horizon(
            models[pred_len], pred_len, checkpoints[pred_len], horizon_min,
            all_test, scalers, args, device)
        pred_df.to_csv(os.path.join(args.out_dir, f'predictions_h{horizon_min}.csv'),
                       index=False)
        metrics = compute_metrics(pred_df['y_true'], pred_df['y_pred'])
        metrics.update({
            'horizon_min': horizon_min,
            'variant': 'MambaFormer-96',
            'pred_len': pred_len,
            'checkpoint': checkpoints[pred_len],
        })
        metric_rows.append(metrics)

        summary = summarize_clarke(pred_df)
        summary.update({
            'horizon_min': horizon_min,
            'variant': 'MambaFormer-96',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'pred_len': pred_len,
            'checkpoint': checkpoints[pred_len],
        })
        summary_rows.append(summary)
        all_predictions.append(pred_df)

    pred_all = pd.concat(all_predictions, ignore_index=True)
    pred_all.to_csv(os.path.join(args.out_dir, 'predictions_all_horizons.csv'),
                    index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values('horizon_min')
    avg = {
        'horizon_min': 'Avg',
        'variant': 'MambaFormer-96',
        'MAE': summary_df['MAE'].mean(),
        'RMSE': summary_df['RMSE'].mean(),
        'n': int(summary_df['n'].sum()),
    }
    for col in ['ZoneA_pct', 'ZoneB_pct', 'ZoneC_pct', 'ZoneD_pct', 'ZoneE_pct',
                'ZoneAB_pct']:
        avg[col] = summary_df[col].mean()
    summary_out = pd.concat([summary_df, pd.DataFrame([avg])], ignore_index=True)
    summary_out.to_csv(os.path.join(args.out_dir, 'clarke_zone_summary.csv'),
                       index=False)
    with open(os.path.join(args.out_dir, 'clarke_zone_summary.json'), 'w') as f:
        json.dump(summary_out.to_dict(orient='records'), f, indent=2)

    metrics_out = pd.DataFrame(metric_rows).sort_values('horizon_min')
    metrics_out.to_csv(os.path.join(args.out_dir, 'baseline_metrics.csv'), index=False)
    with open(os.path.join(args.out_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(metrics_out.to_dict(orient='records'), f, indent=2)

    write_markdown(summary_out, metrics_out, args)
    print(summary_out[['horizon_min', 'variant', 'ZoneA_pct', 'ZoneB_pct',
                       'ZoneAB_pct', 'n']].to_string(index=False), flush=True)
    print(f'[DONE] saved to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
