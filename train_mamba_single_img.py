"""
MambaFormer-96 multimodal: single image type per run.

Usage
-----
  python train_mamba_single_img.py --image_type rp --gpu 5
  python train_mamba_single_img.py --image_type spectrogram --gpu 5
  python train_mamba_single_img.py --image_type gaf --gpu 5
  python train_mamba_single_img.py --image_type mtf --gpu 5

image_type : rp | spectrogram | gaf | mtf | all
  Dataset order: [RP(0), Spectrogram(1), GAF(2), MTF(3)]
"""
import sys, os, json, argparse, datetime, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from lib.multimodal_dataset import load_all_splits, MultimodalCGMDataset
from lib.multimodal_mamba_dinov2 import MultimodalMambaDINOv2

HORIZON_MAP = {
    '15': 3, '30': 6, '45': 9, '60': 12, '75': 15, '90': 18,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_type',  type=str,  default='rp',
                   choices=['rp', 'spectrogram', 'gaf', 'mtf', 'all'])
    p.add_argument('--in_len',      type=int,  default=96)
    p.add_argument('--gpu',         type=int,  default=5)
    p.add_argument('--batch_size',  type=int,  default=8)
    p.add_argument('--grad_accum',  type=int,  default=4)
    p.add_argument('--epochs',      type=int,  default=20)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--patience',    type=int,  default=5)
    p.add_argument('--seed',        type=int,  default=0)
    p.add_argument('--num_workers', type=int,  default=4)
    p.add_argument('--d_model',     type=int,  default=128)
    p.add_argument('--n_heads',     type=int,  default=4)
    p.add_argument('--fusion_mode', type=str,  default='cross_attn',
                   choices=['cross_attn', 'gated_residual', 'simple_concat'])
    p.add_argument('--image_encoder', type=str, default='dino',
                   choices=['dino', 'cnn'])
    p.add_argument('--dino_pool',   type=str,  default='none',
                   choices=['none', 'mean', 'cls'])
    p.add_argument('--modality_fusion', type=str, default='none',
                   choices=['none', 'attention', 'uniform'])
    p.add_argument('--use_tod', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Enable/disable cyclic time-of-day feature.')
    p.add_argument('--horizons',    type=str,  default='15,30,45,60,75,90',
                   help='Comma-separated horizon minutes, e.g. 45,60,75,90')
    p.add_argument('--results_dir', type=str,  default='./results/mamba_single_img')
    p.add_argument('--cache_dir',   type=str,  default='./cache/ts_images')
    args = p.parse_args()
    args.horizon_steps = []
    for item in args.horizons.split(','):
        key = item.strip()
        if key not in HORIZON_MAP:
            raise ValueError(f'Unsupported horizon minute: {key}')
        args.horizon_steps.append(HORIZON_MAP[key])
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae  = float(np.mean(np.abs(t - p)))
    mask = t != 0
    mape = float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def _unpack_model_output(output):
    if isinstance(output, tuple):
        return output
    return output, {}


def run_epoch(model, loader, optimizer, device, grad_accum, is_train):
    model.train(is_train)
    crit = nn.MSELoss()
    total_loss, n_samples = 0.0, 0
    if is_train:
        optimizer.zero_grad()
    for step, batch in enumerate(loader):
        cgm_seq = batch['cgm_seq'].to(device)
        images  = batch['images'].to(device)
        tod_enc = batch['tod_enc'].to(device)
        target  = batch['target'].to(device)
        if is_train:
            pred = model(cgm_seq, images, tod_enc)
            loss = crit(pred, target) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                pred = model(cgm_seq, images, tod_enc)
                loss = crit(pred, target)
        total_loss += loss.item() * target.shape[0] * (grad_accum if is_train else 1)
        n_samples  += target.shape[0]
        if step % 100 == 0:
            mode = 'TRAIN' if is_train else 'VAL'
            print(f'  [{mode}] step {step:4d}/{len(loader):4d}  '
                  f'loss={total_loss/max(1,n_samples):.6f}', flush=True)
    return total_loss / max(1, n_samples)


def evaluate(model, loader, scalers, device):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            cgm  = batch['cgm_seq'].to(device)
            imgs = batch['images'].to(device)
            tod  = batch['tod_enc'].to(device)
            tgt  = batch['target']
            pred, aux = _unpack_model_output(model(cgm, imgs, tod, return_aux=True))
            pred = pred.cpu()
            alpha = aux.get('modality_alpha')
            alpha = alpha.cpu() if alpha is not None else None
            for i in range(len(tgt)):
                rec = {'y_true_norm': float(tgt[i]),
                       'y_pred_norm': float(pred[i]),
                       'ds_name':     batch['ds_name'][i]}
                if alpha is not None:
                    for j in range(alpha.shape[1]):
                        rec[f'alpha_{j}'] = float(alpha[i, j])
                records.append(rec)
    if not records:
        return {}
    df = pd.DataFrame(records)
    all_true, all_pred = [], []
    for ds, grp in df.groupby('ds_name'):
        sk = scalers[ds]
        t = sk.inverse_transform(grp['y_true_norm'].values.reshape(-1, 1)).flatten()
        p = sk.inverse_transform(grp['y_pred_norm'].values.reshape(-1, 1)).flatten()
        all_true.extend(t.tolist())
        all_pred.extend(p.tolist())
    metrics = compute_metrics(all_true, all_pred)
    alpha_cols = [c for c in df.columns if c.startswith('alpha_')]
    if alpha_cols:
        names = ['RP', 'SPEC', 'GAF', 'MTF']
        for idx, col in enumerate(alpha_cols):
            label = names[idx] if idx < len(names) else str(idx)
            metrics[f'Alpha{label}'] = float(df[col].mean())
    return metrics


def train_one_horizon(h_steps, args, all_train, all_val, all_test, scalers, device, out_dir):
    hmin = h_steps * 5
    print(f'\n{"="*60}', flush=True)
    print(f'[image={args.image_type}  horizon={hmin}min  IN_LEN={args.in_len}  '
          f'fusion={args.fusion_mode}  image_encoder={args.image_encoder}  '
          f'dino_pool={args.dino_pool}  '
          f'modality_fusion={args.modality_fusion}  '
          f'use_tod={args.use_tod}]', flush=True)

    train_ds = MultimodalCGMDataset(all_train, h_steps, in_len=args.in_len)
    val_ds   = MultimodalCGMDataset(all_val,   h_steps, in_len=args.in_len)
    test_ds  = MultimodalCGMDataset(all_test,  h_steps, in_len=args.in_len)

    print('[INFO] Pre-generating TS images ...', flush=True)
    for ds in [train_ds, val_ds, test_ds]:
        ds._precompute_images()

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    model = MultimodalMambaDINOv2(
        seq_len=args.in_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_mamba_layers=2,
        num_attn_layers=2,
        dim_feedforward=256,
        dropout=args.dropout,
        d_tod=32,
        freeze_dinov2=True,
        image_type=args.image_type,
        use_tod=args.use_tod,
        fusion_mode=args.fusion_mode,
        image_encoder=args.image_encoder,
        dino_pool=args.dino_pool,
        modality_fusion=args.modality_fusion,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_mae, best_epoch, patience_cnt = float('inf'), 0, 0
    ckpt_path = os.path.join(out_dir, f'best_h{h_steps}.pt')

    for epoch in range(1, args.epochs + 1):
        ts = datetime.datetime.now()
        train_loss = run_epoch(model, train_loader, optimizer, device, args.grad_accum, True)
        val_m = evaluate(model, val_loader, scalers, device)
        val_mae = val_m.get('MAE', float('nan'))
        print(f'  [Ep {epoch}] loss={train_loss:.6f}  Val MAE={val_mae:.3f}  '
              f'RMSE={val_m.get("RMSE", float("nan")):.3f}  '
              f'[{datetime.datetime.now()-ts}]', flush=True)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae; best_epoch = epoch; patience_cnt = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, ckpt_path)
            print(f'  → Saved (val_mae={val_mae:.3f})', flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'  Early stop (best ep={best_epoch}, val_mae={best_val_mae:.3f})', flush=True)
                break

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    test_m = evaluate(model, test_loader, scalers, device)
    test_m['BestValMAE'] = float(best_val_mae)
    test_m['BestEpoch'] = int(best_epoch)

    print(f'[RESULT] image={args.image_type} horizon={hmin}min  '
          f'RMSE={test_m["RMSE"]:.3f}  MAE={test_m["MAE"]:.3f}  '
          f'MAPE={test_m["MAPE"]:.2f}%  R2={test_m["R2"]:.4f}', flush=True)
    alpha_keys = [k for k in ['AlphaRP', 'AlphaSPEC', 'AlphaGAF', 'AlphaMTF'] if k in test_m]
    if alpha_keys:
        print('[AUX] ' + '  '.join(f'{k}={test_m[k]:.4f}' for k in alpha_keys), flush=True)
    return test_m


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] image_type={args.image_type}  IN_LEN={args.in_len}  '
          f'fusion={args.fusion_mode}  image_encoder={args.image_encoder}  '
          f'dino_pool={args.dino_pool}  '
          f'modality_fusion={args.modality_fusion}  '
          f'use_tod={args.use_tod}  '
          f'seed={args.seed}  '
          f'lr={args.lr}  weight_decay={args.weight_decay}  '
          f'dropout={args.dropout}  '
          f'horizons={args.horizons}  device={device}', flush=True)

    out_dir = os.path.join(args.results_dir, args.image_type)
    os.makedirs(out_dir, exist_ok=True)

    all_train, all_val, all_test, scalers = load_all_splits(cache_root=args.cache_dir)

    summary = []
    for h in args.horizon_steps:
        m = train_one_horizon(h, args, all_train, all_val, all_test, scalers, device, out_dir)
        summary.append({'image_type': args.image_type, 'horizon_min': h * 5,
                        'fusion_mode': args.fusion_mode,
                        'image_encoder': args.image_encoder,
                        'dino_pool': args.dino_pool,
                        'modality_fusion': args.modality_fusion,
                        'use_tod': args.use_tod,
                        'seed': args.seed,
                        'lr': args.lr,
                        'weight_decay': args.weight_decay,
                        'dropout': args.dropout, **m})
        result_path = os.path.join(args.results_dir, f'results_{args.image_type}.json')
        with open(result_path, 'w') as f:
            json.dump(summary, f, indent=2)

    result_path = os.path.join(args.results_dir, f'results_{args.image_type}.json')
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[INFO] Saved to {result_path}', flush=True)


if __name__ == '__main__':
    main()
