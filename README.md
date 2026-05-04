# GlucoFusionTS

Image-fused blood glucose time-series forecasting with MambaFormer and DINOv2.

This repository contains the experimental code and saved results for a new blood glucose forecasting study that transforms continuous glucose monitoring (CGM) windows into time-series image representations and fuses them with temporal sequence models.

## Main Results

All experiments use a 96-step CGM input window and evaluate 15, 30, 45, 60, 75, and 90 minute forecasting horizons. MAE and RMSE are computed after inverse scaling back to glucose values.

### Latest Tuned Main Spectrogram Patch-Token Results

The current main model is MambaFormer-SpecPatch: a 96-step MambaFormer sequence encoder fused with frozen DINOv2 spectrogram patch tokens through cross-attention, gated residual prediction, and time-of-day encoding. The table below reports the latest completed hyperparameter-tuned snapshot. Some longer-running tuning jobs may still be active, so the raw candidate files are also included for auditability.

| Horizon | Selected Variant | MAE | RMSE | Delta MAE vs Previous | Delta RMSE vs Previous |
|---:|---|---:|---:|---:|---:|
| 15 min | Previous main result | 7.240 | 11.795 | 0.000 | 0.000 |
| 30 min | `hp_current_e30` | **13.599** | **20.958** | 0.115 | 0.104 |
| 45 min | Previous main result | 18.756 | 28.434 | 0.000 | 0.000 |
| 60 min | Previous main result | 23.272 | 34.650 | 0.000 | 0.000 |
| 75 min | Previous main result | 27.007 | 39.781 | 0.000 | 0.000 |
| 90 min | `hp_wd1e5` | **30.148** | **44.215** | 0.266 | 0.817 |
| Avg | - | **20.004** | **29.972** | 0.064 | 0.154 |

Latest tuned summary files:

- Best-by-horizon table: `results/main_patch_tod_tuned_latest_summary/best_by_horizon_latest.csv`
- All completed tuning candidates: `results/main_patch_tod_tuned_latest_summary/all_completed_candidates.csv`
- Machine-readable summary: `results/main_patch_tod_tuned_latest_summary/summary_latest.json`

### Single-Image Representation Results

| Image | 15 min MAE/RMSE | 30 min MAE/RMSE | 45 min MAE/RMSE | 60 min MAE/RMSE | 75 min MAE/RMSE | 90 min MAE/RMSE |
|---|---:|---:|---:|---:|---:|---:|
| RP | **7.230 / 11.828** | 13.872 / 21.369 | 19.052 / 28.762 | **23.256 / 34.653** | 27.322 / 40.339 | 30.305 / 44.725 |
| Spectrogram | 7.306 / 11.977 | 13.979 / 21.110 | 18.976 / 28.963 | 23.324 / 35.167 | **26.919 / 40.118** | **30.144 / 44.365** |
| GAF | 7.344 / 11.915 | 14.309 / 22.232 | 19.346 / 29.326 | 23.290 / 35.037 | 27.370 / 40.658 | 30.200 / **44.185** |
| MTF | 7.780 / 12.170 | **13.616 / 20.999** | **18.790 / 28.624** | 23.415 / 34.986 | 27.064 / 40.160 | 30.267 / 44.636 |

### Four-Image Adaptive Fusion Results

The four-image fusion model uses RP, Spectrogram, GAF, and MTF together. Each representation is encoded by frozen DINOv2 ViT-S/14 with mean pooling. A modality-attention layer learns image weights conditioned on the temporal representation and time-of-day feature, followed by gated residual fusion.

| Horizon | All-4 MAE | All-4 RMSE | Alpha RP | Alpha SPEC | Alpha GAF | Alpha MTF | Best Single Image |
|---:|---:|---:|---:|---:|---:|---:|---|
| 15 min | 7.612 | 12.214 | 0.132 | 0.065 | 0.532 | 0.271 | RP: 7.230 / 11.828 |
| 30 min | 13.902 | 21.338 | 0.160 | 0.010 | 0.751 | 0.078 | MTF: 13.616 / 20.999 |
| 45 min | 20.033 | 30.424 | 0.095 | 0.019 | 0.711 | 0.175 | MTF: 18.790 / 28.624 |
| 60 min | 23.343 | 35.206 | 0.094 | 0.382 | 0.397 | 0.127 | RP: 23.256 / 34.653 |
| 75 min | 26.933 | 40.320 | 0.028 | 0.117 | 0.763 | 0.092 | Spectrogram: 26.919 / 40.118 |
| 90 min | 30.484 | **44.302** | 0.027 | 0.017 | 0.918 | 0.038 | Spectrogram: 30.144 / 44.365 |

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
- Latest tuned MambaFormer-SpecPatch summary: `results/main_patch_tod_tuned_latest_summary/best_by_horizon_latest.csv`

Model checkpoint files are not committed because they are large. The committed result files contain the reported MAE, RMSE, MAPE, R2, and learned modality-attention weights.

## Reproducing the Experiments

Run the single-image experiments:

```bash
GPU=0 PYTHON=python bin/run_mambaformer_single_image_gated_pooled.sh
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
