# Blood Glucose Time-Series Image Fusion Results

This directory contains the four-image adaptive fusion experiment for blood glucose time-series forecasting.

Configuration:

- Input window: 96 CGM steps
- Forecast horizons: 15, 30, 45, 60, 75, and 90 minutes
- Sequence model: MambaFormer
- Image encoder: frozen DINOv2 ViT-S/14 with mean pooling
- Image representations: RP, Spectrogram, GAF, and MTF
- Fusion: adaptive modality attention followed by gated residual fusion
- Metrics: MAE and RMSE on inverse-scaled glucose values

## All-4 Adaptive Fusion Results

| Horizon (min) | MAE | RMSE | Alpha RP | Alpha SPEC | Alpha GAF | Alpha MTF | Best Single Image | Best Single MAE | Best Single RMSE |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 15 | 7.612 | 12.214 | 0.132 | 0.065 | 0.532 | 0.271 | RP | 7.230 | 11.828 |
| 30 | 13.902 | 21.338 | 0.160 | 0.010 | 0.751 | 0.078 | MTF | 13.616 | 20.999 |
| 45 | 20.033 | 30.424 | 0.095 | 0.019 | 0.711 | 0.175 | MTF | 18.790 | 28.624 |
| 60 | 23.343 | 35.206 | 0.094 | 0.382 | 0.397 | 0.127 | RP | 23.256 | 34.653 |
| 75 | 26.933 | 40.320 | 0.028 | 0.117 | 0.763 | 0.092 | Spectrogram | 26.919 | 40.118 |
| 90 | 30.484 | 44.302 | 0.027 | 0.017 | 0.918 | 0.038 | Spectrogram | 30.144 | 44.365 |

The learned modality attention weights are averaged over the test set. The modality order is RP, Spectrogram, GAF, and MTF.
