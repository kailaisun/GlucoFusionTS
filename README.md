# GlucoFusionTS

Image-fused blood glucose time-series forecasting with MambaFormer and DINOv2.

This repository contains the experimental code and saved results for a new blood glucose forecasting study that transforms continuous glucose monitoring (CGM) windows into time-series image representations and fuses them with temporal sequence models.

## Main Results

All experiments use a 96-step CGM input window and evaluate 15, 30, 45, 60, 75, and 90 minute forecasting horizons. MAE and RMSE are computed after inverse scaling back to glucose values.

### Final Tuned Main Spectrogram Patch-Token Results

The current main model is MambaFormer-SpecPatch: a 96-step MambaFormer sequence encoder fused with frozen DINOv2 spectrogram patch tokens through cross-attention, gated residual prediction, and time-of-day encoding. The table below reports the latest completed hyperparameter-tuned snapshot. Some longer-running tuning jobs may still be active, so the raw candidate files are also included for auditability.

| Horizon | Selected Variant | MAE | RMSE | Delta MAE vs Previous | Delta RMSE vs Previous |
|---:|---|---:|---:|---:|---:|
| 15 min | Previous main result | 7.240 | 11.795 | 0.000 | 0.000 |
| 30 min | `hp_current_e30` | **13.599** | **20.958** | 0.115 | 0.104 |
| 45 min | Previous main result | 18.756 | 28.434 | 0.000 | 0.000 |
| 60 min | Previous main result | 23.272 | 34.650 | 0.000 | 0.000 |
| 75 min | `hp_lr2e4` | **26.935** | **39.652** | 0.072 | 0.129 |
| 90 min | `hp_wd1e5` | **30.148** | **44.215** | 0.266 | 0.817 |
| Avg | - | **19.991** | **29.951** | 0.076 | 0.175 |

### Clarke Error Grid Clinical Accuracy

Clarke zones are computed from inverse-scaled held-out test predictions. Zone A is clinically accurate; Zone B is a benign error region. Zone A+B is commonly used as the clinically acceptable proportion.

| Horizon | Selected Variant | Zone A % | Zone B % | Zone A+B % | Zone C/D/E % |
|---:|---|---:|---:|---:|---:|
| 15 min | Previous main result | 97.23 | 2.41 | 99.64 | 0.36 |
| 30 min | `hp_current_e30` | 88.67 | 9.88 | 98.54 | 1.46 |
| 45 min | Previous main result | 80.40 | 16.76 | 97.16 | 2.84 |
| 60 min | Previous main result | 73.48 | 22.09 | 95.57 | 4.43 |
| 75 min | `hp_lr2e4` | 68.16 | 26.40 | 94.56 | 5.44 |
| 90 min | `hp_wd1e5` | 64.70 | 28.82 | 93.53 | 6.47 |
| Avg | - | **78.77** | **17.73** | **96.50** | 3.50 |

Latest tuned summary files:

- Comprehensive GlucoImg variant ablation table: `results/glucoimg_variant_ablation/glucoimg_variant_ablation_results.csv`
- Best-by-horizon table: `results/main_patch_tod_tuned_final_summary/best_by_horizon_final.csv`
- Clarke Zone A/B table: `results/main_patch_tod_clarke_final/clarke_zone_summary.csv`
- All completed tuning candidates: `results/main_patch_tod_tuned_final_summary/all_completed_candidates.csv`
- Machine-readable summary: `results/main_patch_tod_tuned_final_summary/summary_final.json`
- Patch-token single-image summary: `results/mambaformer_win96_single_img_patch_tod_selected/summary_patch_single_img_with_spec.csv`
- Clarke Zone A bootstrap and McNemar statistics: `results/clarke_zonea_stats/zonea_bootstrap_mcnemar.md`
- Clarke grid figures: `figures/clarke_grid/`

### Single-Image Representation Results

The following table reports the latest patch-token version of the single-image experiments. All structured image variants use the same MambaFormer-96 backbone, frozen DINOv2 patch tokens, gated residual fusion, and time-of-day features. `RawCurve` is included as a control image representation.

| Model | Metric | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MambaFormer-96 | MAE | 9.330 | 15.280 | 19.750 | 24.110 | 28.670 | 31.670 | 21.468 |
| MambaFormer-96 | RMSE | 13.660 | 21.830 | 29.400 | 35.680 | 41.880 | 46.410 | 31.477 |
| + RawCurve | MAE | 10.890 | 24.660 | 30.670 | 33.440 | 35.190 | 37.480 | 28.722 |
| + RawCurve | RMSE | 17.450 | 41.360 | 49.950 | 52.250 | 52.830 | 56.360 | 45.033 |
| + Spectrogram | MAE | **7.240** | **13.600** | **18.760** | **23.270** | **26.940** | **30.150** | **19.993** |
| + Spectrogram | RMSE | **11.800** | **20.960** | 28.430 | 34.650 | **39.650** | **44.220** | **29.952** |
| + RP | MAE | 7.540 | 13.970 | **18.920** | 23.460 | 27.190 | 30.430 | 20.252 |
| + RP | RMSE | 12.180 | 21.320 | **28.410** | 35.060 | 40.140 | 44.360 | 30.245 |
| + GAF | MAE | 7.680 | 13.630 | 19.090 | 23.520 | 27.180 | 30.360 | 20.243 |
| + GAF | RMSE | 12.400 | 21.030 | 28.860 | 35.100 | 40.100 | 44.570 | 30.343 |
| + MTF | MAE | 7.470 | 13.640 | 19.090 | 23.560 | 27.160 | 30.330 | 20.208 |
| + MTF | RMSE | 12.190 | 21.050 | 28.850 | **34.610** | 40.010 | 44.570 | 30.213 |

The older mean-pooling single-image results are kept below for historical comparison.

| Image | 15 min MAE/RMSE | 30 min MAE/RMSE | 45 min MAE/RMSE | 60 min MAE/RMSE | 75 min MAE/RMSE | 90 min MAE/RMSE |
|---|---:|---:|---:|---:|---:|---:|
| RP | **7.230 / 11.828** | 13.872 / 21.369 | 19.052 / 28.762 | **23.256 / 34.653** | 27.322 / 40.339 | 30.305 / 44.725 |
| Spectrogram | 7.306 / 11.977 | 13.979 / 21.110 | 18.976 / 28.963 | 23.324 / 35.167 | **26.919 / 40.118** | **30.144 / 44.365** |
| GAF | 7.344 / 11.915 | 14.309 / 22.232 | 19.346 / 29.326 | 23.290 / 35.037 | 27.370 / 40.658 | 30.200 / **44.185** |
| MTF | 7.780 / 12.170 | **13.616 / 20.999** | **18.790 / 28.624** | 23.415 / 34.986 | 27.064 / 40.160 | 30.267 / 44.636 |

### Four-Image Adaptive Fusion Results

The four-image fusion models use RP, Spectrogram, GAF, and MTF together with DINOv2 patch tokens. Three fusion variants were evaluated: equal averaging, adaptive attention, and direct concatenation. None outperformed the single-spectrogram GlucoImg model on average.

| Model | Metric | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Multi Equal | MAE | 7.640 | 13.820 | 19.620 | 23.270 | 27.320 | 30.650 | 20.387 |
| Multi Equal | RMSE | 12.250 | 21.200 | 29.320 | 34.750 | 40.410 | 44.980 | 30.485 |
| Multi Attn | MAE | 7.350 | 14.110 | 19.120 | 24.080 | 27.140 | 31.260 | 20.510 |
| Multi Attn | RMSE | 11.850 | 21.570 | 28.710 | 35.660 | 40.160 | 45.480 | 30.572 |
| Multi Concat | MAE | 7.890 | 14.110 | 19.290 | 23.380 | 27.150 | 30.330 | 20.358 |
| Multi Concat | RMSE | 12.620 | 21.290 | 29.000 | 34.890 | 40.190 | 44.560 | 30.425 |
| Spectrogram only | MAE | **7.240** | **13.600** | **18.760** | **23.270** | **26.940** | **30.150** | **19.993** |
| Spectrogram only | RMSE | **11.800** | **20.960** | **28.430** | **34.650** | **39.650** | **44.220** | **29.952** |

## Method Summary

The model combines three complementary information sources:

- CGM sequence branch: a MambaFormer encoder processes the 96-step glucose history.
- Image branch: each CGM window is converted into RP, Spectrogram, GAF, or MTF images and encoded with frozen DINOv2.
- Time branch: the last input timestamp is represented by cyclic time-of-day encoding.

For single-image experiments, one image representation is fused with the MambaFormer sequence feature through gated residual fusion. For four-image fusion, a modality-attention module first selects among RP, Spectrogram, GAF, and MTF before gated residual fusion.

## Result Files

- Single-image summary: `results/mambaformer_win96_all_single_img_gated_pooled/summary_all_images.csv`
- Best single-image result by horizon: `results/mambaformer_win96_all_single_img_gated_pooled/best_by_horizon.csv`
- Four-image adaptive fusion summary: `results/mambaformer_win96_all4_modality_attention/summary_all4_attention.csv`
- Full four-image fusion JSON: `results/mambaformer_win96_all4_modality_attention/results_all.json`
- Final tuned MambaFormer-SpecPatch summary: `results/main_patch_tod_tuned_final_summary/best_by_horizon_final.csv`
- Final Clarke Error Grid summary: `results/main_patch_tod_clarke_final/clarke_zone_summary.csv`
- Patch-token single-image summary: `results/mambaformer_win96_single_img_patch_tod_selected/summary_patch_single_img_with_spec.md`
- Baseline Clarke comparison: `results/mambaformer96_baseline_clarke/baseline_vs_specpatch_clarke.md`
- Clarke statistical comparison: `results/clarke_zonea_stats/zonea_bootstrap_mcnemar.md`

Model checkpoint files are not committed because they are large. The committed result files contain the reported MAE, RMSE, MAPE, R2, and learned modality-attention weights.

## Reproducing the Experiments

Run the single-image experiments:

```bash
python train_mamba_single_img.py \
  --image_type spectrogram \
  --in_len 96 \
  --gpu 0 \
  --fusion_mode gated_residual \
  --image_encoder dino \
  --dino_pool none \
  --modality_fusion none \
  --use_tod \
  --horizons 15,30,45,60,75,90 \
  --results_dir results/mambaformer_win96_single_img_patch_tod_selected/spectrogram
```

Run the four-image adaptive fusion experiment:

```bash
python -u train_mamba_single_img.py \
  --image_type all \
  --in_len 96 \
  --gpu 0 \
  --fusion_mode gated_residual \
  --dino_pool mean \
  --modality_fusion attention \
  --horizons 15,30,45,60,75,90 \
  --results_dir results/mambaformer_win96_all4_modality_attention
```

## Data

The experiments use five public CGM datasets: Broll, Colas, Dubosson, Hall, and Weinstock. Data preprocessing follows the existing CGM formatter structure in this repository.
