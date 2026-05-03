"""
Time-series image representation generator for CGM windows.
Supports: Recurrence Plot (RP), Spectrogram, Gramian Angular Field (GAF),
          Markov Transition Field (MTF).
Images are cached to disk as RGB PNG files (224x224) for DINOv2 compatibility.
"""
import os
import numpy as np
from PIL import Image
import scipy.signal


# ─── core representation functions ──────────────────────────────────────────

def _norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def make_rp(x, eps_pct=0.15):
    """Recurrence Plot: binary distance matrix thresholded at eps_pct of range."""
    x = _norm01(x.astype(np.float32))
    D = np.abs(x[:, None] - x[None, :])
    eps = eps_pct * (x.max() - x.min() + 1e-8)
    rp = (D <= eps).astype(np.float32)
    return rp  # (L, L), values in {0, 1}


def make_spectrogram(x, nperseg=16, noverlap=12):
    """Log-magnitude spectrogram via STFT."""
    x = x.astype(np.float64)
    _, _, Sxx = scipy.signal.spectrogram(x, fs=1.0, nperseg=min(nperseg, len(x)),
                                         noverlap=min(noverlap, min(nperseg, len(x)) - 1),
                                         scaling='spectrum')
    Sxx = np.log1p(np.abs(Sxx)).astype(np.float32)
    return Sxx  # (freq_bins, time_bins)


def make_gaf(x):
    """Gramian Angular Summation Field."""
    x = x.astype(np.float32)
    x = _norm01(x) * 2 - 1          # → [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    phi = np.arccos(x)               # (L,)
    gaf = np.cos(phi[:, None] + phi[None, :])  # (L, L), values in [-1, 1]
    return ((gaf + 1) / 2).astype(np.float32)  # → [0, 1]


def make_mtf(x, n_bins=8):
    """Markov Transition Field."""
    x = x.astype(np.float32)
    x_n = _norm01(x)
    bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    q = (np.digitize(x_n, bins) - 1).clip(0, n_bins - 1)  # (L,) quantized indices

    # Transition probability matrix
    M = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(len(q) - 1):
        M[q[i], q[i + 1]] += 1.0
    M /= (M.sum(axis=1, keepdims=True) + 1e-8)

    # MTF: M[q[i], q[j]] for all i, j  (vectorised)
    mtf = M[q[:, None], q[None, :]]  # (L, L)
    return mtf


# ─── image I/O helpers ───────────────────────────────────────────────────────

def _arr_to_rgb_pil(arr, size=224):
    """Convert 2-D float32 array → RGB PIL Image of given size."""
    arr = _norm01(arr.astype(np.float32))
    arr_u8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode='L').convert('RGB')
    return img.resize((size, size), Image.BILINEAR)


def generate_ts_images(x, cache_dir, size=224, force=False):
    """
    Generate (or load from cache) the 4 TS images for CGM window x.

    Parameters
    ----------
    x         : 1-D float array, length L (normalized CGM window)
    cache_dir : directory where *.png files are stored
    size      : output image side length (pixels)
    force     : regenerate even if cache exists

    Returns
    -------
    paths : dict with keys 'rp', 'spectrogram', 'gaf', 'mtf'
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths = {
        'rp':          os.path.join(cache_dir, 'rp.png'),
        'spectrogram': os.path.join(cache_dir, 'spectrogram.png'),
        'gaf':         os.path.join(cache_dir, 'gaf.png'),
        'mtf':         os.path.join(cache_dir, 'mtf.png'),
    }
    if not force and all(os.path.exists(p) for p in paths.values()):
        return paths

    import tempfile
    x = np.asarray(x, dtype=np.float32)
    imgs = [
        ('rp',          make_rp(x)),
        ('spectrogram', make_spectrogram(x)),
        ('gaf',         make_gaf(x)),
        ('mtf',         make_mtf(x)),
    ]
    for key, arr in imgs:
        dest = paths[key]
        if force or not os.path.exists(dest):
            pil_img = _arr_to_rgb_pil(arr, size)
            # Atomic write: write to temp then rename to avoid race conditions
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix='.png.tmp')
            os.close(fd)
            try:
                pil_img.save(tmp_path, 'PNG')
                os.replace(tmp_path, dest)
            except Exception:
                os.unlink(tmp_path)
                raise
    return paths


def load_ts_images_as_tensors(cache_dir, transform):
    """
    Load cached PNG images and apply transform.

    Returns
    -------
    Tensor of shape (4, C, H, W) in image order: [rp, spectrogram, gaf, mtf]
    """
    import torch
    keys = ['rp', 'spectrogram', 'gaf', 'mtf']
    tensors = []
    for k in keys:
        path = os.path.join(cache_dir, f'{k}.png')
        img = Image.open(path).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)  # (4, C, H, W)
