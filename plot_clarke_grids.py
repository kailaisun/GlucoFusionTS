"""
Generate publication-quality Clarke Error Grid figures from exported prediction
CSVs for the sequence-only MambaFormer baseline and the final Spectrogram
patch-token model.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


HORIZONS = [15, 30, 45, 60, 75, 90]
ZONE_ORDER = ['A', 'B', 'C', 'D', 'E']
ZONE_TO_ID = {z: i for i, z in enumerate(ZONE_ORDER)}
ZONE_COLORS = {
    'A': '#e8f4ea',
    'B': '#eef2fb',
    'C': '#fff4d8',
    'D': '#fdebd3',
    'E': '#f8d7da',
}
MODEL_COLORS = {
    'baseline': '#6b7280',
    'main': '#1f77b4',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_dir', type=str,
                   default='./results/mambaformer96_baseline_clarke')
    p.add_argument('--main_dir', type=str,
                   default='./results/main_patch_tod_clarke_final')
    p.add_argument('--out_dir', type=str,
                   default='./figures/clarke_grid')
    p.add_argument('--sample_max', type=int, default=6000,
                   help='Maximum points per panel. Use <=0 to plot all points.')
    p.add_argument('--seed', type=int, default=20260505)
    p.add_argument('--dpi', type=int, default=300)
    return p.parse_args()


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


def load_predictions(model_key, horizon, args):
    base_dir = args.baseline_dir if model_key == 'baseline' else args.main_dir
    df = pd.read_csv(os.path.join(base_dir, f'predictions_h{horizon}.csv'))
    return df


def zone_pct(df, zone='A'):
    return 100.0 * float((df['clarke_zone'] == zone).mean())


def zone_ab_pct(df):
    return 100.0 * float(df['clarke_zone'].isin(['A', 'B']).mean())


def subsample(df, sample_max, rng):
    if sample_max <= 0 or len(df) <= sample_max:
        return df
    idx = rng.choice(len(df), size=sample_max, replace=False)
    return df.iloc[np.sort(idx)]


def draw_zone_background(ax, axis_max):
    grid = np.linspace(0.0, axis_max, 500)
    xx, yy = np.meshgrid(grid, grid)
    zones = clarke_zones(xx.ravel(), yy.ravel()).reshape(xx.shape)
    zone_ids = np.vectorize(ZONE_TO_ID.get)(zones)
    cmap = ListedColormap([ZONE_COLORS[z] for z in ZONE_ORDER])
    norm = BoundaryNorm(np.arange(len(ZONE_ORDER) + 1) - 0.5, cmap.N)
    ax.imshow(zone_ids, extent=[0, axis_max, 0, axis_max], origin='lower',
              cmap=cmap, norm=norm, interpolation='nearest', alpha=0.72,
              aspect='equal', zorder=0)

    xs = np.linspace(0, axis_max, 300)
    ax.plot(xs, xs, color='black', lw=1.0, zorder=2)
    ax.plot(xs, 1.2 * xs, color='black', lw=0.65, ls='--', alpha=0.55, zorder=2)
    ax.plot(xs, 0.8 * xs, color='black', lw=0.65, ls='--', alpha=0.55, zorder=2)
    ax.axvline(70, color='black', lw=0.55, alpha=0.5, zorder=2)
    ax.axhline(70, color='black', lw=0.55, alpha=0.5, zorder=2)
    ax.axvline(180, color='black', lw=0.45, alpha=0.35, zorder=2)
    ax.axhline(180, color='black', lw=0.45, alpha=0.35, zorder=2)
    ax.axvline(240, color='black', lw=0.45, alpha=0.35, zorder=2)

    label_pos = {
        'A': (45, 34),
        'B': (310, 250),
        'C': (145, 25),
        'D': (325, 105),
        'E': (45, 300),
    }
    for z, (x, y) in label_pos.items():
        if x < axis_max and y < axis_max:
            ax.text(x, y, z, fontsize=11, fontweight='bold',
                    color='#333333', ha='center', va='center', alpha=0.75,
                    zorder=3)


def setup_axis(ax, axis_max):
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Reference glucose (mg/dL)')
    ax.set_ylabel('Predicted glucose (mg/dL)')
    ticks = np.arange(0, axis_max + 1, 100)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_single(df, model_label, model_key, horizon, out_base, args, rng):
    axis_max = choose_axis_max([df])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    draw_zone_background(ax, axis_max)
    d = subsample(df, args.sample_max, rng)
    ax.scatter(d['y_true'], d['y_pred'], s=5.0, alpha=0.28,
               c=MODEL_COLORS[model_key], edgecolors='none', rasterized=True,
               zorder=4)
    setup_axis(ax, axis_max)
    ax.set_title(
        f'{model_label}, {horizon} min\n'
        f'Zone A={zone_pct(df):.2f}%, Zone A+B={zone_ab_pct(df):.2f}%',
        fontsize=9)
    save_all(fig, out_base, args)
    plt.close(fig)


def plot_pair(baseline, main, horizon, out_base, args, rng):
    axis_max = choose_axis_max([baseline, main])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharex=True, sharey=True)
    panels = [
        (axes[0], baseline, 'MambaFormer-96', 'baseline'),
        (axes[1], main, 'Spectrogram patch-token', 'main'),
    ]
    for ax, df, title, key in panels:
        draw_zone_background(ax, axis_max)
        d = subsample(df, args.sample_max, rng)
        ax.scatter(d['y_true'], d['y_pred'], s=5.0, alpha=0.28,
                   c=MODEL_COLORS[key], edgecolors='none', rasterized=True,
                   zorder=4)
        setup_axis(ax, axis_max)
        ax.set_title(f'{title}\nZone A={zone_pct(df):.2f}%, A+B={zone_ab_pct(df):.2f}%',
                     fontsize=9)
    fig.suptitle(f'Clarke Error Grid at {horizon} min', fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_all(fig, out_base, args)
    plt.close(fig)


def plot_main_all_horizons(args, rng):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), sharex=True, sharey=True)
    dfs = [load_predictions('main', h, args) for h in HORIZONS]
    axis_max = choose_axis_max(dfs)
    for ax, h, df in zip(axes.ravel(), HORIZONS, dfs):
        draw_zone_background(ax, axis_max)
        d = subsample(df, args.sample_max, rng)
        ax.scatter(d['y_true'], d['y_pred'], s=4.0, alpha=0.22,
                   c=MODEL_COLORS['main'], edgecolors='none', rasterized=True,
                   zorder=4)
        setup_axis(ax, axis_max)
        ax.set_title(f'{h} min: Zone A={zone_pct(df):.2f}%, A+B={zone_ab_pct(df):.2f}%',
                     fontsize=9)
    fig.suptitle('Final Spectrogram Patch-Token Model: Clarke Error Grid',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_all(fig, os.path.join(args.out_dir, 'clarke_main_all_horizons'), args)
    plt.close(fig)


def choose_axis_max(dfs):
    max_val = 0.0
    for df in dfs:
        max_val = max(max_val, float(df['y_true'].max()), float(df['y_pred'].max()))
    return int(max(400, np.ceil((max_val + 10) / 50.0) * 50))


def save_all(fig, out_base, args):
    fig.savefig(f'{out_base}.pdf')
    fig.savefig(f'{out_base}.png', dpi=args.dpi)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


def write_latex_snippets(args):
    path = os.path.join(args.out_dir, 'latex_include_clarke_grid.tex')
    with open(path, 'w') as f:
        f.write('% Clarke Error Grid figures\n')
        f.write('\\begin{figure}[t]\n')
        f.write('    \\centering\n')
        f.write('    \\includegraphics[width=0.95\\textwidth]{figures/clarke_grid/clarke_pair_90min.pdf}\n')
        f.write('    \\caption{Clarke Error Grid comparison at the 90-min prediction horizon. '\
                'The proposed Spectrogram patch-token model increases Zone A predictions '\
                'relative to the sequence-only MambaFormer-96 baseline.}\n')
        f.write('    \\label{fig:clarke_grid_90min}\n')
        f.write('\\end{figure}\n\n')
        f.write('\\begin{figure*}[t]\n')
        f.write('    \\centering\n')
        f.write('    \\includegraphics[width=0.95\\textwidth]{figures/clarke_grid/clarke_main_all_horizons.pdf}\n')
        f.write('    \\caption{Clarke Error Grid analysis of the final Spectrogram patch-token '\
                'model across all prediction horizons.}\n')
        f.write('    \\label{fig:clarke_grid_all_horizons}\n')
        f.write('\\end{figure*}\n')
    print(f'Saved: {path}')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 8,
        'savefig.dpi': args.dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    rng = np.random.default_rng(args.seed)

    # Main-paper figures.
    baseline_90 = load_predictions('baseline', 90, args)
    main_90 = load_predictions('main', 90, args)
    plot_pair(baseline_90, main_90, 90,
              os.path.join(args.out_dir, 'clarke_pair_90min'), args, rng)
    plot_single(main_90, 'Spectrogram patch-token', 'main', 90,
                os.path.join(args.out_dir, 'clarke_main_90min'), args, rng)
    plot_single(baseline_90, 'MambaFormer-96', 'baseline', 90,
                os.path.join(args.out_dir, 'clarke_baseline_90min'), args, rng)
    plot_main_all_horizons(args, rng)

    # Complete per-horizon supplementary figures.
    for h in HORIZONS:
        baseline = load_predictions('baseline', h, args)
        main = load_predictions('main', h, args)
        plot_pair(baseline, main, h,
                  os.path.join(args.out_dir, f'clarke_pair_{h}min'), args, rng)
        plot_single(main, 'Spectrogram patch-token', 'main', h,
                    os.path.join(args.out_dir, f'clarke_main_{h}min'), args, rng)

    write_latex_snippets(args)


if __name__ == '__main__':
    main()
