# MambaFormer-96 Single-Image Representation Results

This directory summarizes the MambaFormer-96 single-image fusion experiments
using a 96-step CGM input window and forecast horizons of 15, 30, 45, 60, 75,
and 90 minutes.

Configuration:

- Sequence baseline: MambaFormer
- Image encoder: frozen DINOv2 ViT-S/14
- DINO pooling: mean pooling
- Fusion: gated residual fusion
- Image representations: RP, Spectrogram, GAF, and MTF
- Metrics: MAE and RMSE on the inverse-scaled glucose values

## Full Results

| Image | Horizon (min) | MAE | RMSE |
|---|---:|---:|---:|
| GAF | 15 | 7.344 | 11.915 |
| GAF | 30 | 14.309 | 22.232 |
| GAF | 45 | 19.346 | 29.326 |
| GAF | 60 | 23.290 | 35.037 |
| GAF | 75 | 27.370 | 40.658 |
| GAF | 90 | 30.200 | 44.185 |
| MTF | 15 | 7.780 | 12.170 |
| MTF | 30 | 13.616 | 20.999 |
| MTF | 45 | 18.790 | 28.624 |
| MTF | 60 | 23.415 | 34.986 |
| MTF | 75 | 27.064 | 40.160 |
| MTF | 90 | 30.267 | 44.636 |
| RP | 15 | 7.230 | 11.828 |
| RP | 30 | 13.872 | 21.369 |
| RP | 45 | 19.052 | 28.762 |
| RP | 60 | 23.256 | 34.653 |
| RP | 75 | 27.322 | 40.339 |
| RP | 90 | 30.305 | 44.725 |
| Spectrogram | 15 | 7.306 | 11.977 |
| Spectrogram | 30 | 13.979 | 21.110 |
| Spectrogram | 45 | 18.976 | 28.963 |
| Spectrogram | 60 | 23.324 | 35.167 |
| Spectrogram | 75 | 26.919 | 40.118 |
| Spectrogram | 90 | 30.144 | 44.365 |

## Best Representation by Horizon

| Horizon (min) | Best Image | MAE | RMSE |
|---:|---|---:|---:|
| 15 | RP | 7.230 | 11.828 |
| 30 | MTF | 13.616 | 20.999 |
| 45 | MTF | 18.790 | 28.624 |
| 60 | RP | 23.256 | 34.653 |
| 75 | Spectrogram | 26.919 | 40.118 |
| 90 | Spectrogram | 30.144 | 44.365 |
