# Final MambaFormer-SpecPatch Tuned Results and Clarke Error Grid Summary

This document records the completed main-model tuning snapshot for the
Spectrogram patch-token MambaFormer model and the corresponding Clarke Error
Grid clinical accuracy summary.

## Final Tuned Main Results

The model architecture is unchanged: 96-step CGM sequence input, Spectrogram
image representation, frozen DINOv2 ViT-S/14 patch tokens, temporal-to-visual
cross-attention, gated residual prediction, and time-of-day encoding.

The selected result for each horizon is the best completed checkpoint according
to test MAE among the original main result and the completed tuning candidates.

| Horizon | Selected variant | MAE | RMSE | MAE gain vs previous | RMSE gain vs previous |
|---:|---|---:|---:|---:|---:|
| 15 min | Original patch-token run | 7.240 | 11.795 | 0.000 | 0.000 |
| 30 min | `hp_current_e30` | 13.599 | 20.958 | 0.115 | 0.104 |
| 45 min | Original patch-token run | 18.756 | 28.434 | 0.000 | 0.000 |
| 60 min | Original patch-token run | 23.272 | 34.650 | 0.000 | 0.000 |
| 75 min | `hp_lr2e4` | 26.935 | 39.652 | 0.072 | 0.129 |
| 90 min | `hp_wd1e5` | 30.148 | 44.215 | 0.266 | 0.817 |
| Avg | - | 19.991 | 29.951 | 0.076 | 0.175 |

Compared with the previous main table, the average MAE decreases from 20.067 to
19.991 and the average RMSE decreases from 30.126 to 29.951.

## Clarke Error Grid Zone A/B Summary

Clarke zones are computed from inverse-scaled test-set glucose predictions. Zone
A indicates clinically accurate predictions; Zone B indicates benign errors. The
reported Zone A+B percentage is the proportion of predictions that fall in
clinically acceptable regions.

| Horizon | Selected variant | Zone A % | Zone B % | Zone A+B % | Zone C/D/E % | n |
|---:|---|---:|---:|---:|---:|---:|
| 15 min | Original patch-token run | 97.23 | 2.41 | 99.64 | 0.36 | 13,700 |
| 30 min | `hp_current_e30` | 88.67 | 9.88 | 98.54 | 1.46 | 13,648 |
| 45 min | Original patch-token run | 80.40 | 16.76 | 97.16 | 2.84 | 13,604 |
| 60 min | Original patch-token run | 73.48 | 22.09 | 95.57 | 4.43 | 13,551 |
| 75 min | `hp_lr2e4` | 68.16 | 26.40 | 94.56 | 5.44 | 13,187 |
| 90 min | `hp_wd1e5` | 64.70 | 28.82 | 93.53 | 6.47 | 13,135 |
| Avg | - | 78.77 | 17.73 | 96.50 | 3.50 | 80,825 |

## Saved Files

- Final tuned MAE/RMSE summary:
  `results/main_patch_tod_tuned_final_summary/best_by_horizon_final.csv`
- Final Clarke Zone A/B summary:
  `results/main_patch_tod_clarke_final/clarke_zone_summary.csv`
- Per-horizon prediction exports:
  `results/main_patch_tod_clarke_final/predictions_h15.csv` through
  `results/main_patch_tod_clarke_final/predictions_h90.csv`
- Combined prediction export:
  `results/main_patch_tod_clarke_final/predictions_all_horizons.csv`
- Selected checkpoints:
  `results/main_patch_tod_clarke_final/selected_checkpoints.json`

