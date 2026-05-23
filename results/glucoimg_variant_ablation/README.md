# GlucoImg Variant Ablation Results

This folder contains the updated paper-facing variant ablation table for the GlucoImg study.

The table compares:

- `SeqOnly`: MambaFormer sequence-only baseline.
- `GlucoImg-RawCurve`: raw CGM curve image control.
- `GlucoImg-SPEC`: final spectrogram-based GlucoImg model.
- `GlucoImg-RP`, `GlucoImg-GAF`, `GlucoImg-MTF`: single-image representation variants.
- `GlucoImg-Multi (Equal)`, `GlucoImg-Multi (Attn)`, `GlucoImg-Multi (Concat)`: four-representation fusion variants.
- `GlucoImg-GRU`, `GlucoImg-PatchTST`: alternative temporal backbone variants with spectrogram image fusion.

Values are reported as MAE and RMSE in mg/dL across 15, 30, 45, 60, 75, and 90 minute forecasting horizons. Relative improvements in the manuscript table are computed against `SeqOnly`; positive values indicate lower error.

Files:

- `glucoimg_variant_ablation_results.csv`: machine-readable table values.
- `paper_drafts/overleaf_glucoimg_v1/tables/glucoimg_variant_ablation.tex`: LaTeX table used by the Overleaf manuscript draft.
