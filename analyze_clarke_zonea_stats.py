"""
Paired statistical analysis for Clarke Error Grid Zone A.

Outputs:
  1. Bootstrap 95% confidence intervals for Zone A rates of the
     MambaFormer-96 baseline and the final Spectrogram patch-token model.
  2. Paired bootstrap 95% confidence intervals for the Zone A improvement.
  3. McNemar tests for paired Zone A membership.

The analysis assumes that baseline and main prediction CSVs use the same
horizon-specific test windows in the same order. The script verifies this by
checking sample count, dataset labels, and inverse-scaled ground truth values.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd


HORIZONS = [15, 30, 45, 60, 75, 90]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_dir', type=str,
                   default='./results/mambaformer96_baseline_clarke')
    p.add_argument('--main_dir', type=str,
                   default='./results/main_patch_tod_clarke_final')
    p.add_argument('--out_dir', type=str,
                   default='./results/clarke_zonea_stats')
    p.add_argument('--n_boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260505)
    p.add_argument('--batch_size', type=int, default=500)
    return p.parse_args()


def exact_mcnemar_pvalue(b, c):
    """Two-sided exact McNemar p-value via binomial test."""
    n = int(b + c)
    if n == 0:
        return 1.0
    k = int(min(b, c))
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n=n, p=0.5, alternative='two-sided').pvalue)
    except Exception:
        # Numerically stable exact binomial CDF for p=0.5.
        log_half_n = -n * math.log(2.0)
        probs = [
            math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                     - math.lgamma(n - i + 1) + log_half_n)
            for i in range(k + 1)
        ]
        return float(min(1.0, 2.0 * math.fsum(probs)))


def bootstrap_mean_ci(values, rng, n_boot=10000, batch_size=500):
    """Bootstrap percentile 95% CI for the mean of a 1D numeric array."""
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    stats = np.empty(n_boot, dtype=np.float32)
    out = 0
    while out < n_boot:
        b = min(batch_size, n_boot - out)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        stats[out:out + b] = values[idx].mean(axis=1)
        out += b
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def load_pair(horizon, args):
    b = pd.read_csv(os.path.join(args.baseline_dir, f'predictions_h{horizon}.csv'))
    m = pd.read_csv(os.path.join(args.main_dir, f'predictions_h{horizon}.csv'))
    if len(b) != len(m):
        raise ValueError(f'H{horizon}: sample count mismatch {len(b)} vs {len(m)}')
    if not (b['ds_name'].to_numpy() == m['ds_name'].to_numpy()).all():
        raise ValueError(f'H{horizon}: ds_name order mismatch')
    max_true_diff = float(np.max(np.abs(b['y_true'].to_numpy() - m['y_true'].to_numpy())))
    if max_true_diff > 1e-5:
        raise ValueError(f'H{horizon}: y_true mismatch, max diff={max_true_diff}')
    return b, m


def analyze_one(label, base_zone_a, main_zone_a, rng, args):
    base_float = base_zone_a.astype(np.float32)
    main_float = main_zone_a.astype(np.float32)
    delta_float = main_float - base_float

    base_mean, base_lo, base_hi = bootstrap_mean_ci(
        base_float, rng, args.n_boot, args.batch_size)
    main_mean, main_lo, main_hi = bootstrap_mean_ci(
        main_float, rng, args.n_boot, args.batch_size)
    delta_mean, delta_lo, delta_hi = bootstrap_mean_ci(
        delta_float, rng, args.n_boot, args.batch_size)

    both_a = int(np.sum(base_zone_a & main_zone_a))
    base_only = int(np.sum(base_zone_a & ~main_zone_a))
    main_only = int(np.sum(~base_zone_a & main_zone_a))
    neither = int(np.sum(~base_zone_a & ~main_zone_a))
    p_value = exact_mcnemar_pvalue(base_only, main_only)

    return {
        'horizon_min': label,
        'n': int(len(base_zone_a)),
        'baseline_zoneA_pct': base_mean * 100.0,
        'baseline_zoneA_ci_low': base_lo * 100.0,
        'baseline_zoneA_ci_high': base_hi * 100.0,
        'main_zoneA_pct': main_mean * 100.0,
        'main_zoneA_ci_low': main_lo * 100.0,
        'main_zoneA_ci_high': main_hi * 100.0,
        'delta_zoneA_pp': delta_mean * 100.0,
        'delta_zoneA_ci_low': delta_lo * 100.0,
        'delta_zoneA_ci_high': delta_hi * 100.0,
        'both_zoneA_n': both_a,
        'baseline_only_zoneA_n': base_only,
        'main_only_zoneA_n': main_only,
        'neither_zoneA_n': neither,
        'mcnemar_p': p_value,
        'discordant_n': int(base_only + main_only),
        'main_to_baseline_discordant_ratio': (
            float(main_only / base_only) if base_only > 0 else float('inf')
        ),
    }


def write_markdown(rows, out_path):
    with open(out_path, 'w') as f:
        f.write('# Clarke Zone A Bootstrap CI and McNemar Test\n\n')
        f.write('Bootstrap confidence intervals use paired test samples and '
                '10,000 bootstrap resamples by default. McNemar tests compare '
                'Zone A membership between MambaFormer-96 and the final '
                'Spectrogram patch-token model on the same samples.\n\n')

        f.write('| Horizon | Baseline Zone A % (95% CI) | Main Zone A % (95% CI) | '
                'Delta pp (95% CI) | baseline A / main non-A | '
                'baseline non-A / main A | McNemar p |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            p = '<1e-6' if r['mcnemar_p'] < 1e-6 else f"{r['mcnemar_p']:.3g}"
            f.write(
                f"| {r['horizon_min']} | "
                f"{r['baseline_zoneA_pct']:.2f} "
                f"[{r['baseline_zoneA_ci_low']:.2f}, {r['baseline_zoneA_ci_high']:.2f}] | "
                f"{r['main_zoneA_pct']:.2f} "
                f"[{r['main_zoneA_ci_low']:.2f}, {r['main_zoneA_ci_high']:.2f}] | "
                f"{r['delta_zoneA_pp']:+.2f} "
                f"[{r['delta_zoneA_ci_low']:+.2f}, {r['delta_zoneA_ci_high']:+.2f}] | "
                f"{r['baseline_only_zoneA_n']} | "
                f"{r['main_only_zoneA_n']} | {p} |\n"
            )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    pooled_base = []
    pooled_main = []
    for horizon in HORIZONS:
        baseline, main = load_pair(horizon, args)
        base_zone_a = baseline['clarke_zone'].to_numpy() == 'A'
        main_zone_a = main['clarke_zone'].to_numpy() == 'A'
        rows.append(analyze_one(horizon, base_zone_a, main_zone_a, rng, args))
        pooled_base.append(base_zone_a)
        pooled_main.append(main_zone_a)

    rows.append(analyze_one(
        'Pooled',
        np.concatenate(pooled_base),
        np.concatenate(pooled_main),
        rng,
        args,
    ))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.out_dir, 'zonea_bootstrap_mcnemar.csv'),
                  index=False)
    with open(os.path.join(args.out_dir, 'zonea_bootstrap_mcnemar.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    write_markdown(rows, os.path.join(args.out_dir, 'zonea_bootstrap_mcnemar.md'))
    print(out_df.to_string(index=False), flush=True)
    print(f'[DONE] saved to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
