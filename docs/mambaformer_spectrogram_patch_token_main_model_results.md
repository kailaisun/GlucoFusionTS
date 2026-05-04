# MambaFormer-Spectrogram Patch-Token Main Model

## 1. Model Identity

This document defines the updated main model for the blood glucose time-series image-fusion study.

**Final model name**: MambaFormer-SpecPatch

**Full configuration**:

- Temporal backbone: MambaFormer
- Input window: 96 CGM steps
- Forecast horizons: 15, 30, 45, 60, 75, and 90 minutes
- Image representation: Spectrogram
- Image encoder: frozen DINOv2 ViT-S/14
- DINOv2 image token strategy: patch tokens, no global pooling
- Fusion module: cross-attention followed by gated residual prediction
- Auxiliary temporal feature: cyclic time-of-day encoding
- Metrics: MAE and RMSE after inverse scaling to glucose values

The model is intended to replace the previous mean-pooled Spectrogram model as the primary Spectrogram-based method because it achieves the best average MAE and RMSE among the tested DINO token strategies.

## 2. Architecture

The model contains three branches.

### 2.1 CGM Sequence Branch

The normalized 96-step CGM history is passed through a MambaFormer encoder:

```text
x_cgm in R^{B x 96}
H_seq = MambaFormer(x_cgm) in R^{B x 96 x d}
```

where `d = 128` in the current implementation.

### 2.2 Spectrogram Image Branch

Each 96-step CGM window is converted into a spectrogram image and encoded by frozen DINOv2 ViT-S/14. Unlike the previous mean-pooling model, all DINO patch tokens are retained:

```text
H_img = DINOv2(Spectrogram(x_cgm)) in R^{B x 256 x 384}
Z_img = Linear(H_img) in R^{B x 256 x d}
```

For a 224 x 224 image with ViT-S/14, DINOv2 produces a 16 x 16 grid, i.e., 256 patch tokens.

### 2.3 Time-of-Day Branch

The time-of-day at the last input step is represented by sine/cosine cyclic encoding and projected by a small MLP:

```text
h_tod = MLP([sin(t), cos(t)]) in R^{B x d_tod}
```

where `d_tod = 32`.

## 3. Patch-Token Fusion Mechanism

The patch-token model uses cross-attention from temporal tokens to local spectrogram patch tokens:

```text
Q = H_seq      in R^{B x 96 x d}
K = Z_img      in R^{B x 256 x d}
V = Z_img      in R^{B x 256 x d}
```

The cross-attention output is:

```text
H_fused = CrossAttention(Q=H_seq, K=Z_img, V=Z_img)
```

This allows each temporal token to attend to local time-frequency regions in the spectrogram.

The prediction head uses gated residual fusion:

```text
h_seq   = mean(H_seq)
h_fused = mean(H_fused)

y_base  = Head([h_seq, h_tod])
delta   = Head([h_fused, h_tod])
gate    = sigmoid(Gate([h_seq, h_fused, h_tod]))

y_hat   = y_base + gate * delta
```

This design keeps the sequence-only forecast as the base prediction and lets the image branch make a gated correction.

## 4. Difference from Mean Pooling and CLS Token

The three DINO token strategies differ in the image information passed into cross-attention.

| Strategy | Image tokens passed to fusion | Shape after projection | Interpretation |
|---|---:|---:|---|
| CLS token | DINO global CLS token | B x 1 x d | A learned global image summary |
| Mean pooling | Average over all patch tokens | B x 1 x d | A compact global spectrogram summary |
| Patch tokens | All local patch tokens | B x 256 x d | Local time-frequency spectrogram regions |

Mean pooling compresses the spectrogram into a single global vector, while patch-token fusion preserves local spectrogram structure and allows temporal tokens to selectively attend to relevant time-frequency regions.

## 5. Main Results

### 5.1 Comparison with Sequence-Only Baseline

| Model | 15 min MAE/RMSE | 30 min MAE/RMSE | 45 min MAE/RMSE | 60 min MAE/RMSE | 75 min MAE/RMSE | 90 min MAE/RMSE | Avg MAE | Avg RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MambaFormer-96 baseline | 9.330 / 13.660 | 15.280 / 21.830 | 19.750 / 29.400 | 24.110 / 35.680 | 28.670 / 41.880 | 31.670 / 46.410 | 21.468 | 31.477 |
| MambaFormer-SpecPatch | **7.240 / 11.795** | **13.714 / 21.062** | **18.756 / 28.434** | **23.272 / 34.650** | **27.007 / 39.781** | **30.414 / 45.032** | **20.067** | **30.125** |

Relative to the sequence-only MambaFormer baseline, MambaFormer-SpecPatch improves the average MAE by **6.53%** and the average RMSE by **4.29%**.

### 5.2 Horizon-Wise Improvement over MambaFormer-96

| Horizon | MAE improvement | RMSE improvement |
|---:|---:|---:|
| 15 min | +22.4% | +13.7% |
| 30 min | +10.2% | +3.5% |
| 45 min | +5.0% | +3.3% |
| 60 min | +3.5% | +2.9% |
| 75 min | +5.8% | +5.0% |
| 90 min | +4.0% | +3.0% |

Positive values indicate lower error than the sequence-only baseline.

### 5.3 Latest Hyperparameter-Tuned Snapshot

After observing that the original 90-minute run was weaker than several ablation variants, the main MambaFormer-SpecPatch model was re-tuned at the training-configuration level while keeping the model architecture fixed. The tuning variables were learning rate, weight decay, dropout, maximum epochs, and early-stopping patience. Candidate selection used validation MAE, and the table below reports the latest completed snapshot. Some full-horizon tuning jobs may still be running, so this section should be treated as the latest completed result set rather than a final frozen camera-ready table.

| Horizon | Selected variant | MAE | RMSE | Change in MAE | Change in RMSE |
|---:|---|---:|---:|---:|---:|
| 15 min | Original patch-token run | 7.240 | 11.795 | 0.000 | 0.000 |
| 30 min | `hp_current_e30` | **13.599** | **20.958** | -0.115 | -0.104 |
| 45 min | Original patch-token run | 18.756 | 28.434 | 0.000 | 0.000 |
| 60 min | Original patch-token run | 23.272 | 34.650 | 0.000 | 0.000 |
| 75 min | Original patch-token run | 27.007 | 39.781 | 0.000 | 0.000 |
| 90 min | `hp_wd1e5` | **30.148** | **44.215** | -0.266 | -0.817 |
| Avg | - | **20.004** | **29.972** | -0.064 | -0.154 |

The largest gain is at the 90-minute horizon, where retuning reduces RMSE from 45.032 to 44.215. This addresses the earlier concern that the full patch-token model was underperforming at the longest horizon relative to some ablation variants.

The latest completed tuning snapshot is saved in:

- `results/main_patch_tod_tuned_latest_summary/best_by_horizon_latest.csv`
- `results/main_patch_tod_tuned_latest_summary/best_by_horizon_latest.json`
- `results/main_patch_tod_tuned_latest_summary/all_completed_candidates.csv`
- `results/main_patch_tod_tuned_latest_summary/summary_latest.json`

## 6. DINO Token Strategy Ablation

| Variant | 15 min MAE/RMSE | 30 min MAE/RMSE | 45 min MAE/RMSE | 60 min MAE/RMSE | 75 min MAE/RMSE | 90 min MAE/RMSE | Avg MAE | Avg RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO CLS token | 7.319 / 12.017 | 13.643 / 21.039 | 18.809 / 28.728 | 23.427 / 34.875 | 26.912 / 40.077 | 30.513 / 44.656 | 20.104 | 30.232 |
| DINO mean pooling | 7.306 / 11.977 | 13.979 / 21.110 | 18.976 / 28.963 | 23.324 / 35.167 | 26.919 / 40.118 | **30.144 / 44.365** | 20.108 | 30.283 |
| DINO patch tokens | **7.240 / 11.795** | 13.714 / 21.062 | **18.756 / 28.434** | **23.272 / 34.650** | 27.007 / **39.781** | 30.414 / 45.032 | **20.067** | **30.125** |

Patch tokens obtain the best average MAE and RMSE. Mean pooling remains slightly more stable at the 90-minute horizon, but its average performance is lower than patch-token fusion.

## 7. Interpretation

The results support three observations.

First, spectrogram image features improve the MambaFormer baseline across all horizons. The gain is largest at 15 minutes and remains positive for long-horizon forecasting.

Second, preserving DINO patch tokens is more effective than compressing the spectrogram into a single global token. This suggests that local time-frequency structure in the spectrogram contains useful predictive information for glucose forecasting.

Third, the patch-token variant is not uniformly dominant at every horizon. In particular, mean pooling gives a lower RMSE at 90 minutes. For the paper, this should be reported as a trade-off: patch tokens provide the best average performance, while global pooling can be more stable at the longest horizon.

## 8. Plotting Data

The plotting-ready files are saved at:

- Long-format CSV: `results/spec_patch_token_main_model/plot_data_long.csv`
- Long-format JSON: `results/spec_patch_token_main_model/plot_data_long.json`
- Wide-format CSV: `results/spec_patch_token_main_model/summary_wide.csv`
- Wide-format JSON: `results/spec_patch_token_main_model/summary_wide.json`

Recommended plots:

1. Line plot of MAE vs forecasting horizon for baseline, mean pooling, CLS token, and patch tokens.
2. Line plot of RMSE vs forecasting horizon for the same models.
3. Bar plot of average MAE and average RMSE across models.
4. Optional grouped bar chart for DINO token strategy ablation.

## 9. Source Result Files

The raw experiment outputs are:

- Patch-token full model: `results/spec_ablation_gated_patch_tod/results_spectrogram.json`
- Mean-pooling ablation: `results/mambaformer_win96_spec_gated_pooled_h15_30/results_spectrogram.json`
- Mean-pooling ablation: `results/mambaformer_win96_spec_gated_pooled/results_spectrogram.json`
- CLS-token ablation: `results/spec_ablation_gated_cls_tod/results_spectrogram.json`
- Combined ablation summary: `results/spec_mambaformer_ablation_summary/summary_spec_ablation.csv`

## 10. Suggested Paper Description

The final model can be described as follows:

> We use a MambaFormer encoder to model the 96-step CGM sequence and transform the same window into a spectrogram image. The spectrogram is encoded by a frozen DINOv2 ViT-S/14 encoder. Instead of reducing the visual representation to a single global embedding, we preserve all DINO patch tokens and use them as key-value tokens in a cross-attention layer queried by temporal MambaFormer tokens. The fused representation is then used in a gated residual prediction head, allowing spectrogram-derived features to adaptively correct the sequence-only forecast. This patch-token fusion design achieves the best average MAE and RMSE among the evaluated DINO token aggregation strategies.
