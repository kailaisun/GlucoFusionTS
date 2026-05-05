# Clarke Zone A Bootstrap CI and McNemar Test

Bootstrap confidence intervals use paired test samples and 10,000 bootstrap resamples by default. McNemar tests compare Zone A membership between MambaFormer-96 and the final Spectrogram patch-token model on the same samples.

| Horizon | Baseline Zone A % (95% CI) | Main Zone A % (95% CI) | Delta pp (95% CI) | baseline A / main non-A | baseline non-A / main A | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 96.69 [96.39, 96.99] | 97.23 [96.96, 97.50] | +0.53 [+0.33, +0.74] | 66 | 139 | <1e-6 |
| 30 | 87.11 [86.55, 87.68] | 88.67 [88.12, 89.21] | +1.55 [+1.19, +1.90] | 193 | 405 | <1e-6 |
| 45 | 80.09 [79.41, 80.76] | 80.40 [79.73, 81.06] | +0.31 [-0.07, +0.68] | 313 | 355 | 0.113 |
| 60 | 73.28 [72.55, 74.03] | 73.48 [72.72, 74.22] | +0.20 [-0.18, +0.59] | 348 | 375 | 0.334 |
| 75 | 68.09 [67.29, 68.91] | 68.16 [67.35, 68.94] | +0.07 [-0.47, +0.61] | 666 | 675 | 0.827 |
| 90 | 59.29 [58.45, 60.12] | 64.70 [63.88, 65.50] | +5.41 [+4.72, +6.11] | 741 | 1452 | <1e-6 |
| Pooled | 77.61 [77.32, 77.89] | 78.94 [78.65, 79.22] | +1.33 [+1.15, +1.51] | 2327 | 3401 | <1e-6 |
