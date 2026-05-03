"""
Multimodal CGM Dataset for MambaFormer + DINOv2 fusion experiments.

Each sample returns:
  cgm_seq       : (IN_LEN,) float32 – normalized CGM history window
  images        : (4, 3, 224, 224) float32 – [RP, Spectrogram, GAF, MTF]
  tod_enc       : (2,) float32 – [sin, cos] cyclic time-of-day encoding
  target        : scalar float32 – normalized glucose at the forecast horizon
  ds_name       : str – dataset label (for correct inverse-transform)

Subject-independent train/val/test split mirrors the existing MambaFormer baseline:
  - DataFormatter config with random_state=0 (default in all yaml configs)
  - 15% of train id_segment keys held out as validation
"""
import os
import sys
import yaml
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import DataFormatter
from lib.ts_image_gen import generate_ts_images, load_ts_images_as_tensors

DATASETS = ['weinstock', 'colas', 'dubosson', 'hall', 'iglu']
IN_LEN   = 96    # 8 hours of history at 5-min intervals
STRIDE   = 12    # slide window by 12 steps (60 min) for efficiency

# Standard ImageNet normalization expected by DINOv2
_DINO_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── data loading helpers ────────────────────────────────────────────────────

def _load_dataset_splits(ds_name, cache_root='./cache/ts_images'):
    """
    Load one CGM dataset via DataFormatter and return segment series dicts.

    Returns
    -------
    train_segs, val_segs, test_segs : dict
        {seg_id: {'series': np.ndarray, 'hours': np.ndarray,
                  'minutes': np.ndarray, 'scaler': sklearn.scaler,
                  'ds_name': str, 'cache_root': str}}
    scaler : sklearn MinMaxScaler (per-target, fitted on train)
    """
    config_path = f'./config/{ds_name}.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['scaling_params']['scaler'] = 'MinMaxScaler'

    fmt = DataFormatter(config)
    tc     = fmt.get_column('target')[0]
    sid_c  = fmt.get_column('sid')       # 'id_segment' – unique continuous segment ID
    scaler = fmt.scalers[tc]

    def _df_to_segs(df):
        segs = {}
        for seg_id, grp in df.groupby(sid_c):
            grp = grp.sort_values(fmt.get_column('time'))
            key = f'{ds_name}_{seg_id}'
            segs[key] = {
                'series':  grp[tc].values.astype(np.float32),
                'hours':   grp['time_hour'].values.astype(np.float32),
                'minutes': grp['time_minute'].values.astype(np.float32),
                'scaler':  scaler,
                'ds_name': ds_name,
                'cache_root': os.path.join(cache_root, ds_name, str(seg_id)),
            }
        return segs

    all_train_segs = _df_to_segs(fmt.train_data)

    # Replicate the 85/15 train-val subject split from combined_mambaformer.py
    seg_ids = list(all_train_segs.keys())
    n_val   = max(1, int(0.15 * len(seg_ids)))
    val_ids = set(seg_ids[-n_val:])

    train_segs = {k: v for k, v in all_train_segs.items() if k not in val_ids}
    val_segs   = {k: v for k, v in all_train_segs.items() if k     in val_ids}
    test_segs  = _df_to_segs(fmt.test_data)

    return train_segs, val_segs, test_segs, scaler


def load_all_splits(cache_root='./cache/ts_images'):
    """Load all 5 datasets and merge train/val/test dicts."""
    all_train, all_val, all_test, scalers = {}, {}, {}, {}
    for ds in DATASETS:
        tr, va, te, sc = _load_dataset_splits(ds, cache_root)
        all_train.update(tr)
        all_val.update(va)
        all_test.update(te)
        scalers[ds] = sc
    return all_train, all_val, all_test, scalers


# ─── sliding window index builder ────────────────────────────────────────────

def _build_windows(segs_dict, horizon, in_len=IN_LEN, stride=STRIDE):
    """
    Build a list of (seg_entry, window_start) pairs for all segments.

    horizon : int – number of 5-min steps ahead to predict
    """
    windows = []
    for seg_id, entry in segs_dict.items():
        series = entry['series']
        max_start = len(series) - in_len - horizon  # inclusive upper bound
        for start in range(0, max_start + 1, stride):
            windows.append((entry, seg_id, start))
    return windows


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class MultimodalCGMDataset(Dataset):
    """
    Parameters
    ----------
    segs_dict  : {seg_id -> segment entry dict}  (from load_all_splits)
    horizon    : prediction horizon in 5-min steps (9 → 45 min, 12 → 60 min)
    in_len     : input window length in steps (default: IN_LEN=96)
    image_size : side length for PNG images (224 for DINOv2)
    precompute : if True, pre-generate all missing images at init time
    """

    def __init__(self, segs_dict, horizon, in_len=IN_LEN,
                 image_size=224, precompute=False):
        self.horizon    = horizon
        self.in_len     = in_len
        self.image_size = image_size
        self.transform  = _DINO_TRANSFORM
        self.windows    = _build_windows(segs_dict, horizon, in_len=in_len)

        if precompute:
            self._precompute_images()

    def _img_dir(self, entry, start):
        """Cache path includes in_len so 48-step and 96-step images don't collide."""
        return os.path.join(entry['cache_root'], f'len{self.in_len}', str(start))

    def _precompute_images(self):
        """Generate and cache all images (skip if already on disk)."""
        try:
            from tqdm import tqdm
            it = tqdm(self.windows, desc='Generating TS images', ncols=80)
        except ImportError:
            it = self.windows

        for entry, seg_id, start in it:
            x = entry['series'][start: start + self.in_len]
            generate_ts_images(x, self._img_dir(entry, start), size=self.image_size)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        entry, seg_id, start = self.windows[idx]
        series  = entry['series']
        hours   = entry['hours']
        minutes = entry['minutes']

        # ── CGM sequence ──
        cgm_seq = torch.from_numpy(series[start: start + self.in_len])

        # ── Target: single glucose value horizon steps ahead ──
        target = torch.tensor(series[start + self.in_len + self.horizon - 1],
                              dtype=torch.float32)

        # ── Time-of-day cyclic encoding at last input step ──
        tod_idx  = start + self.in_len - 1
        minofday = float(hours[tod_idx]) * 60.0 + float(minutes[tod_idx])
        angle    = 2.0 * math.pi * minofday / 1440.0
        tod_enc  = torch.tensor([math.sin(angle), math.cos(angle)],
                                dtype=torch.float32)

        # ── Images (lazy-generate if missing, then load) ──
        img_dir = self._img_dir(entry, start)
        generate_ts_images(series[start: start + self.in_len],
                           img_dir, size=self.image_size)
        images = load_ts_images_as_tensors(img_dir, self.transform)  # (4, 3, 224, 224)

        return {
            'cgm_seq':  cgm_seq,          # (in_len,)
            'images':   images,           # (4, 3, H, W)
            'tod_enc':  tod_enc,          # (2,)
            'target':   target,           # scalar
            'ds_name':  entry['ds_name'], # str  (for inverse-transform)
        }
