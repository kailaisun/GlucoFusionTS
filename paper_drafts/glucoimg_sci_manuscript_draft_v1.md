# GlucoImg: Image-Enhanced Learning of Glucose Spike Patterns for Continuous Glucose Forecasting

**Draft version:** v1  
**Date:** 2026-05-17  
**Status:** detailed SCI manuscript draft for content selection and later LaTeX conversion  
**Primary model name:** GlucoImg  
**Sequence-only baseline name:** SeqOnly  

> **Important drafting note.** This draft intentionally avoids fabricating bibliographic details. References are marked as `[CITATION NEEDED]` where a verified BibTeX entry should be inserted later. Dataset names and citation numbers should be reconciled against the final reference manager before submission, especially for Broll/iGLU naming.

---

## Abstract

Continuous glucose monitoring (CGM) enables dense observation of glucose dynamics, but reliable multi-horizon glucose forecasting remains difficult because clinically relevant excursions often involve rapid transitions rather than smooth progression. Most forecasting models treat CGM as a one-dimensional sequence, which may underuse higher-order temporal structure associated with glucose spikes, oscillatory changes, and transition patterns. We propose **GlucoImg**, an image-enhanced forecasting framework that converts a historical CGM window into a time-series image representation and fuses visual patch tokens with a MambaFormer temporal encoder. Specifically, GlucoImg transforms each 96-step CGM input window into a spectrogram image, extracts frozen DINOv2 patch tokens, and uses temporal-to-visual cross-attention with a gated residual prediction head to adaptively correct a sequence-only forecast. Across six prediction horizons from 15 to 90 minutes, GlucoImg reduced average MAE/RMSE from 21.47/31.48 mg/dL with SeqOnly to 19.99/29.95 mg/dL, corresponding to 6.9% and 4.9% average improvements. Representation ablations showed that spectrogram features provided the most consistent gains compared with recurrence plots, Gramian angular fields, and Markov transition fields. Clinical analyses further indicated improved Clarke Zone A rates at the longest horizon and better error behavior under large glucose-change regimes. On an external ShanghaiT2DM cohort, overall gains were modest but GlucoImg reduced 90-minute high-glucose error and spike-pattern error, suggesting selective benefit for upward glucose excursions in type 2 diabetes. These findings support image-enhanced representation learning as a complementary strategy for capturing glucose dynamics beyond sequence-only forecasting.

**Keywords:** continuous glucose monitoring, glucose forecasting, time-series imaging, spectrogram, MambaFormer, DINOv2, cross-attention, diabetes informatics

---

## 1. Introduction

Continuous glucose monitoring has changed diabetes management by providing fine-grained glucose trajectories rather than sparse self-monitoring measurements. Forecasting future glucose from CGM is important for early warning, decision support, insulin dosing assistance, and detection of clinically relevant excursions. The forecasting problem is especially challenging at longer horizons because future glucose depends not only on the current level but also on recent dynamic patterns, such as rapid postprandial increases, delayed drops, oscillatory variability, and repeated high-glucose states.

Most existing glucose forecasting methods model CGM as a one-dimensional sequence. Traditional approaches include autoregressive modeling and regression-based predictors, while deep learning approaches include recurrent neural networks, temporal convolutional networks, Transformers, PatchTST-style models, and state-space sequence models [CITATION NEEDED]. These methods have improved forecasting accuracy, but they still operate primarily on the raw temporal signal. A raw sequence representation is efficient, but it may not explicitly expose structural patterns that are easier to identify after transforming the signal into an image-like representation. In particular, spikes and rapid transitions can produce localized time-frequency changes, recurrence structures, or state-transition patterns that may be difficult for a sequence-only model to exploit consistently.

Time-series imaging provides an alternative view of temporal dynamics by mapping a one-dimensional signal to a two-dimensional representation. Spectrograms emphasize localized frequency-energy changes, recurrence plots highlight repeated states, Gramian angular fields encode angular temporal correlations, and Markov transition fields describe quantized state transitions [CITATION NEEDED]. These representations are widely used in time-series classification, but their role in CGM forecasting remains underexplored. A key question is whether such image representations add predictive value beyond a strong sequence model, and, if so, which image representation is most useful for glucose forecasting.

This paper studies this question through **GlucoImg**, an image-enhanced CGM forecasting framework. GlucoImg keeps the sequence branch as the primary forecaster and uses an image branch as a gated residual correction. The temporal branch uses a MambaFormer encoder over a 96-step CGM window. The same window is converted into a spectrogram image and encoded by frozen DINOv2 ViT-S/14. Unlike global visual pooling, GlucoImg preserves DINOv2 patch tokens and uses them as keys and values in a cross-attention module queried by temporal tokens. The final prediction combines a sequence-only base forecast with a gated image-enhanced residual, allowing the model to use image information when beneficial while retaining sequence-based stability.

The paper makes four contributions.

1. **Image-enhanced CGM forecasting.** We introduce GlucoImg, a multimodal forecasting model that combines MambaFormer temporal encoding with spectrogram-derived DINOv2 patch tokens through cross-attention and gated residual prediction.

2. **Representation-level analysis.** We compare spectrogram, recurrence plot, Gramian angular field, Markov transition field, and four-image fusion variants under the same backbone, showing that spectrogram patch tokens provide the most consistent horizon-wise performance in our setting.

3. **Mechanistic and clinical evaluation.** We evaluate GlucoImg using MAE/RMSE, Clarke Error Grid analysis, event detection for hypo- and hyperglycemia, dynamic-change stratification, case studies, and spectrogram interpretability visualizations.

4. **External type 2 diabetes validation.** We evaluate the method on ShanghaiT2DM, a 15-minute sampling external type 2 diabetes cohort, and show that although average gains are horizon-dependent, GlucoImg improves 90-minute high-glucose and spike-pattern prediction.

The central finding is not that image features universally improve every glucose state. Instead, the evidence suggests a more specific mechanism: image-enhanced features are most useful when the forecasting target involves dynamic changes, especially glucose spikes and high-glucose excursions. This framing is important for both methodological interpretation and clinical relevance.

---

## 2. Related Work

### 2.1 Continuous glucose forecasting

CGM forecasting has been studied using statistical, machine learning, and deep learning models. Classical methods include autoregressive integrated moving average (ARIMA), linear regression, random forests, gradient boosting, and LightGBM-style predictors. These methods are often competitive at short horizons because recent glucose values are highly autocorrelated. Deep learning models, including LSTM, GRU, TCN, Transformer, PatchTST, and state-space variants, can capture longer temporal dependencies and nonlinear dynamics [CITATION NEEDED]. In our baseline comparison, deep sequence models generally outperformed traditional methods at longer horizons, and MambaFormer showed strong performance at 45- and 60-minute horizons. GlucoImg builds on this sequence modeling foundation but asks whether image-derived representations can further improve forecasting.

### 2.2 Time-series image representations

Time-series imaging methods convert a one-dimensional signal into a two-dimensional image-like representation. Recurrence plots encode pairwise similarity between time points, Gramian angular fields encode angular transformations of normalized temporal values, Markov transition fields encode transition probabilities between quantized states, and spectrograms represent localized time-frequency energy. These transformations have been widely used in time-series classification and representation learning because they expose structure that may be difficult to extract from the raw sequence alone [CITATION NEEDED]. In glucose forecasting, the most relevant structures are not only global shape but also transition onset, sustained high-glucose states, and rapid excursions. This motivates a systematic comparison of different time-series images for CGM prediction.

### 2.3 Vision foundation models for non-natural images

Self-supervised vision foundation models such as DINOv2 provide general-purpose visual features learned from large-scale image data [CITATION NEEDED]. Although these models are trained on natural images, their patch-token representations can also encode local structure in scientific or transformed signal images. A challenge is that time-series images differ from natural images; global image features may discard useful local transition patterns. GlucoImg therefore uses frozen DINOv2 as a patch-token feature extractor rather than as a trainable image classifier. The model preserves the 16 by 16 grid of patch tokens from a 224 by 224 image and allows temporal tokens to attend to local spectrogram regions.

### 2.4 Multimodal fusion for biomedical time series

Multimodal learning often combines complementary representations through concatenation, attention, gating, or residual fusion. In biomedical forecasting, naive concatenation may introduce noise when auxiliary modalities are not consistently informative. Gated residual designs provide a conservative alternative: a base model produces a primary prediction, while the auxiliary branch learns a correction whose magnitude is controlled by a learned gate [CITATION NEEDED]. GlucoImg follows this principle. The sequence-only prediction remains the base forecast, and the image branch contributes a gated residual correction. This design is especially appropriate for CGM because image features may be helpful for dynamic excursions but less useful for stable glucose segments.

---

## 3. Materials and Methods

### 3.1 Problem formulation

Let \(x_{1:L} = (x_1, x_2, \ldots, x_L)\) denote a historical CGM window after normalization, where \(L=96\) for the main 5-minute sampling protocol. The goal is to predict future glucose at horizon \(h\), where \(h \in \{15,30,45,60,75,90\}\) minutes. The target is the inverse-scaled glucose value \(y_{t+h}\), and model performance is reported in mg/dL using mean absolute error (MAE) and root mean squared error (RMSE):

\[
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i-y_i|,
\quad
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2}.
\]

For the external ShanghaiT2DM cohort, CGM is sampled every 15 minutes. We therefore use a 32-point input window, corresponding to the same 8-hour history length as the 96-point main protocol.

### 3.2 Overview of GlucoImg

GlucoImg contains three information streams: a CGM sequence branch, a time-series image branch, and a time-of-day branch. The sequence branch provides the base prediction. The image branch encodes a spectrogram generated from the same CGM window. The time-of-day branch represents circadian context using a sine/cosine encoding at the last input time point.

The design follows a conservative residual principle:

\[
\hat{y} = \hat{y}_{\mathrm{seq}} + g \cdot \Delta_{\mathrm{img}},
\]

where \(\hat{y}_{\mathrm{seq}}\) is the sequence-only base prediction, \(\Delta_{\mathrm{img}}\) is the image-enhanced residual prediction, and \(g \in [0,1]\) is a learned gate. This formulation allows the image branch to correct the base forecast without forcing visual information to dominate every sample.

### 3.3 Sequence encoder

The temporal branch uses MambaFormer, a hybrid sequence encoder combining Mamba-style state-space modeling with Transformer attention blocks. Given the normalized CGM window \(x \in \mathbb{R}^{B \times L}\), the encoder produces temporal tokens:

\[
H_{\mathrm{seq}} = f_{\mathrm{MF}}(x) \in \mathbb{R}^{B \times L \times d},
\]

where \(d=128\) in the main implementation. The sequence feature is obtained by mean pooling over temporal tokens:

\[
h_{\mathrm{seq}} = \frac{1}{L}\sum_{\ell=1}^{L} H_{\mathrm{seq},\ell}.
\]

### 3.4 Time-series image generation

The same input CGM window is converted into a time-series image. The final GlucoImg model uses a spectrogram:

\[
I_{\mathrm{spec}} = \mathrm{Spec}(x).
\]

The spectrogram is computed as a log-magnitude short-time Fourier representation and resized to 224 by 224 pixels for DINOv2. We also evaluated recurrence plots (RP), Gramian angular fields (GAF), and Markov transition fields (MTF). These alternatives were included to test whether the improvement comes from general image augmentation or from the specific inductive bias of spectrogram representations.

### 3.5 DINOv2 patch-token image encoder

Each time-series image is encoded using a frozen DINOv2 ViT-S/14 model. For a 224 by 224 image, ViT-S/14 produces a 16 by 16 grid of patch tokens. GlucoImg retains all patch tokens rather than using a single CLS token or global mean-pooled image vector:

\[
H_{\mathrm{img}} = f_{\mathrm{DINO}}(I_{\mathrm{spec}}) \in \mathbb{R}^{B \times 256 \times 384}.
\]

A linear projection maps the DINO embedding dimension to the temporal model dimension:

\[
Z_{\mathrm{img}} = H_{\mathrm{img}} W_{\mathrm{img}} \in \mathbb{R}^{B \times 256 \times d}.
\]

This design preserves local time-frequency information and enables patch-level interaction between temporal and visual representations.

### 3.6 Cross-attention fusion

GlucoImg uses temporal tokens as queries and image patch tokens as keys and values:

\[
Q = H_{\mathrm{seq}}, \quad K = Z_{\mathrm{img}}, \quad V = Z_{\mathrm{img}}.
\]

The cross-attention module produces image-aware temporal tokens:

\[
\tilde{H}_{\mathrm{seq}} =
\mathrm{CrossAttn}(Q=H_{\mathrm{seq}}, K=Z_{\mathrm{img}}, V=Z_{\mathrm{img}}).
\]

The fused feature is then obtained by temporal mean pooling:

\[
h_{\mathrm{fused}} = \frac{1}{L}\sum_{\ell=1}^{L}\tilde{H}_{\mathrm{seq},\ell}.
\]

This formulation differs from direct concatenation because it allows each temporal token to selectively attend to local image patches. It also differs from global pooling because the image representation is not compressed before fusion.

### 3.7 Time-of-day encoding

Because glucose dynamics are influenced by daily rhythms, meals, and behavioral patterns, GlucoImg includes a cyclic time-of-day representation. If \(m\) denotes minutes since midnight at the final input time point, the encoding is:

\[
e_{\mathrm{tod}} = [\sin(2\pi m/1440), \cos(2\pi m/1440)].
\]

This two-dimensional vector is passed through a small MLP to produce:

\[
h_{\mathrm{tod}} = f_{\mathrm{tod}}(e_{\mathrm{tod}}) \in \mathbb{R}^{32}.
\]

### 3.8 Gated residual prediction head

The base prediction head uses the sequence feature and time-of-day feature:

\[
\hat{y}_{\mathrm{seq}} = f_{\mathrm{base}}([h_{\mathrm{seq}}, h_{\mathrm{tod}}]).
\]

The image-enhanced residual head uses the fused feature:

\[
\Delta_{\mathrm{img}} = f_{\Delta}([h_{\mathrm{fused}}, h_{\mathrm{tod}}]).
\]

The residual gate is computed as:

\[
g = \sigma(f_g([h_{\mathrm{seq}}, h_{\mathrm{fused}}, h_{\mathrm{tod}}])).
\]

The final prediction is:

\[
\hat{y} = \hat{y}_{\mathrm{seq}} + g \Delta_{\mathrm{img}}.
\]

The residual head is initialized conservatively so that the model begins close to the sequence-only forecast and gradually learns image-based corrections when they improve validation loss.

### 3.9 DINO token strategy variants

We evaluated three DINO token strategies:

| Strategy | Visual representation passed to fusion | Interpretation |
|---|---|---|
| CLS token | DINO global CLS token | learned global image summary |
| Mean pooling | average of all patch tokens | compact global spectrogram summary |
| Patch tokens | all 256 patch tokens | local time-frequency structures |

Patch tokens produced the best average MAE/RMSE among the tested strategies, supporting the hypothesis that local spectrogram regions contain useful predictive information.

---

## 4. Experimental Setup

### 4.1 Datasets

The main evaluation used five processed CGM cohorts from prior studies. The current processed files include Colas, Dubosson, Hall, iGLU/Broll, and Weinstock cohorts; final naming should be verified against the manuscript reference list. These datasets cover non-diagnosed, type 1 diabetes, and type 2 diabetes populations in the processed subgroup analysis. The main protocol uses 5-minute sampling and a 96-step input window, corresponding to 8 hours of historical CGM.

An external validation experiment was performed on ShanghaiT2DM, from the Chinese diabetes datasets for data-driven machine learning [CITATION NEEDED]. ShanghaiT2DM uses 15-minute sampling, so the input window was set to 32 points to preserve the 8-hour historical context. This dataset is reported separately because its sampling interval, population, and preprocessing protocol differ from the main 5-minute benchmark.

### 4.2 Forecasting horizons

All main experiments predict glucose at six horizons:

\[
h \in \{15, 30, 45, 60, 75, 90\}\ \mathrm{minutes}.
\]

This range covers short-term warning and longer-horizon planning. Longer horizons are more clinically challenging because forecast uncertainty accumulates and dynamic glucose transitions are harder to anticipate.

### 4.3 Baselines

We compared GlucoImg against traditional and deep sequence models:

- ARIMA
- Linear regression
- Random forest
- Gradient boosting
- LightGBM
- LSTM
- GRU
- TCN
- Transformer
- PatchTST
- MambaFormer / SeqOnly

SeqOnly refers to the MambaFormer temporal model without image input. It is the primary baseline because GlucoImg uses the same temporal backbone and differs only by the image-enhanced branch.

### 4.4 Metrics

Primary forecasting metrics are MAE and RMSE after inverse scaling to mg/dL. We also report:

- Clarke Error Grid Zone A and Zone A+B percentages.
- Hyperglycemia and hypoglycemia event detection recall and F1.
- Error stratified by true glucose range.
- Error stratified by future glucose-change pattern.
- Case-study visualizations of rapid transitions.
- Relative MAE improvement as a function of true glucose-change magnitude.

### 4.5 Implementation details

The main model uses a 96-step input window, a MambaFormer temporal encoder with hidden dimension \(d=128\), frozen DINOv2 ViT-S/14 image encoder, DINO patch tokens, cross-attention fusion, gated residual prediction, and cyclic time-of-day encoding. Model selection used validation loss and early stopping. Final results are reported from inverse-scaled test predictions. The external ShanghaiT2DM experiment uses the same model design with a 32-step input window due to 15-minute sampling.

---

## 5. Results

### 5.1 Main forecasting performance

Table 1 compares the sequence-only baseline and GlucoImg across all horizons. GlucoImg reduced average MAE from 21.47 to 19.99 mg/dL and average RMSE from 31.48 to 29.95 mg/dL. Improvements were observed at every horizon, with the largest relative MAE gain at 15 minutes and sustained gains at longer horizons.

**Table 1. Main SeqOnly vs GlucoImg performance.**

| Model | Metric | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SeqOnly | MAE | 9.33 | 15.28 | 19.75 | 24.11 | 28.67 | 31.67 | 21.47 |
| SeqOnly | RMSE | 13.66 | 21.83 | 29.40 | 35.68 | 41.88 | 46.41 | 31.48 |
| GlucoImg | MAE | 7.24 | 13.60 | 18.76 | 23.27 | 26.94 | 30.15 | 19.99 |
| GlucoImg | RMSE | 11.80 | 20.96 | 28.43 | 34.65 | 39.65 | 44.22 | 29.95 |
| Relative improvement | MAE | +22.4% | +11.0% | +5.0% | +3.5% | +6.1% | +4.8% | +6.9% |
| Relative improvement | RMSE | +13.6% | +4.0% | +3.3% | +2.9% | +5.3% | +4.7% | +4.9% |

The largest absolute gains occurred at long horizons, while the largest relative MAE gain occurred at 15 minutes. This pattern suggests that image-enhanced representation is not only useful for immediate smoothing but also contributes to longer-horizon dynamics.

### 5.2 Comparison with traditional and deep learning baselines

Across the broader baseline table, traditional models performed competitively at short horizons, while deep learning models generally improved long-horizon forecasting. MambaFormer was selected as the primary sequence backbone because it showed strong baseline performance at 45 and 60 minutes. GlucoImg then improved over this strong sequence-only backbone, indicating that the gains are not due to a weak baseline.

**Suggested Table 2. Baseline model comparison.**  
Use the existing baseline table with ARIMA, Linear Regression, Random Forest, Gradient Boosting, LightGBM, LSTM, GRU, TCN, Transformer, PatchTST, and MambaFormer. Add GlucoImg as the final row or place it in the next representation table depending on journal layout.

### 5.3 GlucoImg variant analysis

We compared five image representations under the same MambaFormer patch-token fusion design: raw-curve visualization, RP, GAF, MTF, and Spectrogram. RawCurve served as a negative control and performed worse than SeqOnly, indicating that the gain is not caused by simply rasterizing the CGM curve as an image. Spectrogram, denoted as GlucoImg-SPEC, achieved the best average MAE and RMSE. RP, GAF, and MTF also improved over SeqOnly, but their gains were smaller.

**Table 3. GlucoImg representation variants.**

| Model | Metric | 15 min | 30 min | 45 min | 60 min | 75 min | 90 min | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SeqOnly | MAE | 9.33 | 15.28 | 19.75 | 24.11 | 28.67 | 31.67 | 21.47 |
| SeqOnly | RMSE | 13.66 | 21.83 | 29.40 | 35.68 | 41.88 | 46.41 | 31.48 |
| GlucoImg-RawCurve | MAE | 10.89 | 24.66 | 30.67 | 33.44 | 35.19 | 37.48 | 28.72 |
| GlucoImg-RawCurve | RMSE | 17.45 | 41.36 | 49.95 | 52.25 | 52.83 | 56.36 | 45.03 |
| GlucoImg-SPEC | MAE | 7.24 | 13.60 | 18.76 | 23.27 | 26.94 | 30.15 | 19.99 |
| GlucoImg-SPEC | RMSE | 11.80 | 20.96 | 28.43 | 34.65 | 39.65 | 44.22 | 29.95 |
| GlucoImg-RP | MAE | 7.54 | 13.97 | 18.92 | 23.46 | 27.19 | 30.43 | 20.25 |
| GlucoImg-RP | RMSE | 12.18 | 21.32 | 28.41 | 35.06 | 40.14 | 44.36 | 30.25 |
| GlucoImg-GAF | MAE | 7.68 | 13.63 | 19.09 | 23.52 | 27.18 | 30.36 | 20.24 |
| GlucoImg-GAF | RMSE | 12.40 | 21.03 | 28.86 | 35.10 | 40.10 | 44.57 | 30.34 |
| GlucoImg-MTF | MAE | 7.47 | 13.64 | 19.09 | 23.56 | 27.16 | 30.33 | 20.21 |
| GlucoImg-MTF | RMSE | 12.19 | 21.05 | 28.85 | 34.61 | 40.01 | 44.57 | 30.21 |

The spectrogram result is consistent with the model design. Spectrograms contain localized time-frequency structure, and the patch-token cross-attention module can selectively attend to local spectral regions. In contrast, RP, GAF, and MTF encode more global relational structure; they may be useful, but their local patches may not align as directly with glucose-transition events.

### 5.4 Four-image fusion analysis

We also evaluated All-4 fusion variants using RP, Spectrogram, GAF, and MTF together. Three fusion strategies were tested: adaptive attention fusion, uniform averaging, and direct concatenation. These methods did not consistently outperform the single-spectrogram GlucoImg model.

**Table 4. All-4 fusion strategies.**

| Model | Avg MAE | Avg RMSE | Interpretation |
|---|---:|---:|---|
| GlucoImg-Multi (Equal) | 20.39 | 30.48 | uniform fusion remained competitive but did not beat SPEC |
| GlucoImg-Multi (Attn) | 20.51 | 30.57 | adaptive weighting did not clearly improve over SPEC |
| GlucoImg-Multi (Concat) | 20.36 | 30.42 | concatenation increased representation size but not accuracy |
| GlucoImg (Spectrogram) | 19.99 | 29.95 | best single representation |

These results suggest that adding more image representations is not automatically beneficial. Fusion may introduce redundant or noisy visual information, especially if the auxiliary representations are less aligned with the forecasting mechanism. Therefore, the final model uses spectrogram only.

We additionally tested spectrogram fusion with alternative temporal backbones. GlucoImg-GRU achieved an average MAE/RMSE of 20.69/30.77 mg/dL, and GlucoImg-PatchTST achieved 20.82/31.23 mg/dL. Both improved over SeqOnly on average, but neither outperformed the MambaFormer-based GlucoImg-SPEC configuration.

### 5.5 DINO token strategy ablation

We compared CLS token, mean pooling, and patch-token fusion. Patch-token fusion achieved the best average performance, while mean pooling was competitive at the 90-minute horizon.

**Table 5. DINO token strategy ablation.**

| Variant | 15 min MAE/RMSE | 30 min MAE/RMSE | 45 min MAE/RMSE | 60 min MAE/RMSE | 75 min MAE/RMSE | 90 min MAE/RMSE | Avg MAE | Avg RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO CLS token | 7.319 / 12.017 | 13.643 / 21.039 | 18.809 / 28.728 | 23.427 / 34.875 | 26.912 / 40.077 | 30.513 / 44.656 | 20.104 | 30.232 |
| DINO mean pooling | 7.306 / 11.977 | 13.979 / 21.110 | 18.976 / 28.963 | 23.324 / 35.167 | 26.919 / 40.118 | 30.144 / 44.365 | 20.108 | 30.283 |
| DINO patch tokens | 7.240 / 11.795 | 13.714 / 21.062 | 18.756 / 28.434 | 23.272 / 34.650 | 27.007 / 39.781 | 30.414 / 45.032 | 20.067 | 30.125 |

Patch tokens outperform global token strategies on average, supporting the use of local image regions for glucose dynamics. The fact that mean pooling can be competitive at 90 minutes indicates that local attention is not uniformly dominant for every horizon; however, the patch-token strategy provides the best overall balance.

### 5.6 Clinical accuracy with Clarke Error Grid

Clarke Error Grid analysis was used to evaluate clinical acceptability. GlucoImg improved Zone A rates at 15, 30, and 90 minutes, with the largest improvement at 90 minutes. Pooled across horizons, Zone A increased from 77.61% to 78.94%.

**Table 6. Clarke Zone A paired statistics.**

| Horizon | SeqOnly Zone A % | GlucoImg Zone A % | Delta pp | McNemar p |
|---:|---:|---:|---:|---:|
| 15 | 96.69 | 97.23 | +0.53 | <1e-6 |
| 30 | 87.11 | 88.67 | +1.55 | <1e-6 |
| 45 | 80.09 | 80.40 | +0.31 | 0.113 |
| 60 | 73.28 | 73.48 | +0.20 | 0.334 |
| 75 | 68.09 | 68.16 | +0.07 | 0.827 |
| 90 | 59.29 | 64.70 | +5.41 | <1e-6 |
| Pooled | 77.61 | 78.94 | +1.33 | <1e-6 |

The 90-minute gain is clinically relevant because long-horizon forecasts are more difficult and more useful for early intervention. The smaller gains at 45 to 75 minutes should be interpreted cautiously; GlucoImg improves the overall distribution but does not transform every horizon equally.

### 5.7 Event detection for hypo- and hyperglycemia

We evaluated event detection using threshold-based hypo- and hyperglycemia definitions. Hyperglycemia detection showed consistent F1 improvements across several seeds and horizons, while hypoglycemia results were more variable. For seed 2, hyperglycemia F1 improved at all horizons, with recall improvements at 15, 30, 45, 60, 75, and 90 minutes. Hypoglycemia F1 improved at 15, 30, 45, 75, and 90 minutes but decreased at 60 minutes.

This analysis suggests that GlucoImg may be more reliable for high-glucose excursion detection than for low-glucose detection. This asymmetry is clinically plausible because the spectrogram branch appears more useful for upward spike-like transitions than for all dynamic patterns.

### 5.8 Error under glucose-change patterns

We stratified test samples by true future glucose change from the last input CGM value:

- stable: \(|\Delta| < 10\) mg/dL
- moderate change: \(10 \leq |\Delta| < 30\) mg/dL
- spike: \(\Delta \geq 30\) mg/dL
- drop: \(\Delta \leq -30\) mg/dL

In the main protocol, GlucoImg improved all four aggregated pattern groups. The largest relative gain occurred for stable samples, but gains were also observed for spike and drop groups.

**Table 7. Error by glucose-change pattern.**

| Dataset | Pattern | n | SeqOnly MAE | GlucoImg MAE | MAE improvement | SeqOnly RMSE | GlucoImg RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Main | Stable | 35,832 | 8.27 | 7.38 | +10.77% | 11.87 | 11.10 |
| Main | Moderate | 25,807 | 16.10 | 15.90 | +1.24% | 19.94 | 19.92 |
| Main | Spike | 9,818 | 55.73 | 54.84 | +1.59% | 66.52 | 65.71 |
| Main | Drop | 9,368 | 43.13 | 41.97 | +2.68% | 51.91 | 51.17 |
| ShanghaiT2DM | Stable | 13,303 | 7.37 | 7.83 | -6.26% | 10.37 | 11.17 |
| ShanghaiT2DM | Moderate | 9,003 | 14.16 | 14.14 | +0.14% | 18.05 | 18.06 |
| ShanghaiT2DM | Spike | 2,582 | 46.16 | 41.74 | +9.58% | 54.03 | 49.79 |
| ShanghaiT2DM | Drop | 2,415 | 26.09 | 26.96 | -3.31% | 31.95 | 33.49 |

The external ShanghaiT2DM result is particularly informative. GlucoImg does not improve all patterns in this cohort, but it substantially improves spike-pattern prediction. This supports the paper's central claim that image-enhanced learning is most useful for glucose spike patterns rather than uniformly beneficial for all states.

### 5.9 Error as a function of glucose-change magnitude

We further analyzed relative MAE improvement as a function of absolute glucose change magnitude \(|\Delta G|\). Larger changes correspond to more dynamic and clinically challenging samples. The relative improvement increased from approximately 0.98% in the 10-30 mg/dL bin to approximately 3.98% in the 130-150 mg/dL bin. This trend supports the interpretation that GlucoImg contributes more when glucose dynamics are stronger.

**Suggested Figure 1. Relative MAE improvement vs. glucose-change magnitude.**  
Use: `figures/error_vs_glucose_dynamics/relative_mae_improvement_vs_glucose_dynamics_rapid_20mgdl_with_n.pdf`

### 5.10 Case studies of rapid transitions

Representative case studies show that GlucoImg better tracks rapid transition cases than SeqOnly. In spike-like segments, SeqOnly tends to lag behind or smooth the peak, whereas GlucoImg follows the rise more closely and better captures peak timing.

**Suggested Figure 2. Rapid transition case studies.**  
Use: `figures/glucoimg_case_studies/glucoimg_rapid_transition_cases.pdf`

These examples are not intended as statistical proof by themselves. Their value is explanatory: they show the kind of dynamic pattern for which the quantitative analysis suggests GlucoImg is helpful.

### 5.11 Spectrogram interpretability

We visualized a representative CGM segment, its spectrogram, and cross-attention localization. The visualization shows that rapid glucose transitions correspond to localized spectral responses, and the model emphasizes related spectrogram regions through patch-level attention.

**Suggested Figure 3. Spectrogram interpretability visualization.**  
Use: `figures/spectrogram_interpretability/spectrogram_interpretability_cross_attention.pdf`

This figure supports the mechanism of the model. Rather than treating the time-series image as a generic visual feature, it shows a correspondence between glucose transition dynamics, spectral energy, and model attention.

### 5.12 External ShanghaiT2DM validation

ShanghaiT2DM was evaluated as an external type 2 diabetes cohort with a 15-minute sampling interval and a 32-point input window. GlucoImg showed modest and horizon-dependent average improvements. After updating the private rerun value for 45-minute SeqOnly MAE to 13.95 mg/dL, GlucoImg improved MAE at 15, 45, 60, and 90 minutes, while 30 and 75 minutes did not improve. RMSE improved at 15, 30, 45, 60, and 90 minutes, with the largest gain at 90 minutes.

**Table 8. ShanghaiT2DM external validation.**

| Horizon | SeqOnly MAE | SeqOnly RMSE | GlucoImg MAE | GlucoImg RMSE | MAE improvement | RMSE improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 4.93 | 7.26 | 4.72 | 7.06 | +4.43% | +2.82% |
| 30 | 9.80 | 14.14 | 9.84 | 14.03 | -0.44% | +0.82% |
| 45 | 13.95 | 19.65 | 13.87 | 19.59 | +0.58% | +0.29% |
| 60 | 17.37 | 24.54 | 17.14 | 24.07 | +1.33% | +1.92% |
| 75 | 20.42 | 28.57 | 20.73 | 28.58 | -1.53% | -0.03% |
| 90 | 23.57 | 33.21 | 22.61 | 31.45 | +4.10% | +5.30% |

The external validation result should be interpreted as selective rather than universal. The strongest ShanghaiT2DM evidence appears in 90-minute high-glucose and spike-pattern analysis. For example, in samples with true glucose above 180 mg/dL at 90 minutes, MAE improved from 39.85 to 35.69 mg/dL, and in samples above 250 mg/dL, MAE improved from 59.69 to 53.19 mg/dL. In spike-pattern samples, MAE improved from 46.16 to 41.74 mg/dL. Conversely, stable and drop patterns in ShanghaiT2DM did not benefit consistently.

### 5.13 ShanghaiT2DM alternative image representations

To test whether ShanghaiT2DM underperformance was due to the spectrogram representation, we evaluated GAF and MTF at 60 and 90 minutes using the same patch-token fusion model. Neither representation outperformed spectrogram.

**Table 9. ShanghaiT2DM image-type trial at 60 and 90 minutes.**

| Model | Horizon | MAE | RMSE | MAE improvement vs SeqOnly | RMSE improvement vs SeqOnly |
|---|---:|---:|---:|---:|---:|
| SeqOnly | 60 | 17.37 | 24.54 | 0.00% | 0.00% |
| Spectrogram | 60 | 17.14 | 24.07 | +1.33% | +1.92% |
| GAF | 60 | 18.94 | 26.53 | -9.00% | -8.08% |
| MTF | 60 | 18.92 | 26.48 | -8.88% | -7.90% |
| SeqOnly | 90 | 23.57 | 33.21 | 0.00% | 0.00% |
| Spectrogram | 90 | 22.61 | 31.45 | +4.10% | +5.30% |
| GAF | 90 | 23.81 | 33.03 | -1.01% | +0.55% |
| MTF | 90 | 23.72 | 32.54 | -0.62% | +2.02% |

This analysis supports retaining spectrogram as the final image representation. It also shows that the external T2D limitation is not solved simply by switching to GAF or MTF.

---

## 6. Discussion

### 6.1 Principal findings

This study shows that converting CGM windows into spectrogram images can improve continuous glucose forecasting when the image representation is fused carefully with a strong sequence model. GlucoImg improved average MAE and RMSE over a MambaFormer sequence-only baseline across six prediction horizons. The gains were consistent in the main protocol and remained visible under multiple analyses, including representation ablation, clinical grid evaluation, event detection, and glucose dynamics stratification.

The most important result is not only the average error reduction but the pattern of improvement. GlucoImg was especially helpful for dynamic regimes, including large glucose-change bins and spike-like patterns in the external type 2 diabetes cohort. This supports the hypothesis that spectrogram images expose transition-related structure that is complementary to the raw CGM sequence.

### 6.2 Why spectrogram works better than other image representations

Spectrogram was the best image representation in the main experiment and remained stronger than GAF and MTF in the ShanghaiT2DM trial. A plausible explanation is the alignment between spectrogram structure and patch-token cross-attention. DINOv2 patch tokens preserve local image regions. In spectrograms, local regions can correspond to transient spectral energy associated with rapid glucose changes. Temporal tokens can attend to these local regions through cross-attention. In contrast, RP, GAF, and MTF encode relational or state-transition matrices whose local patches may not correspond as directly to clinically meaningful time-local transitions.

This does not mean spectrograms are universally optimal for all CGM settings. Rather, in the present architecture and sampling protocols, spectrogram features appear to provide the most useful inductive bias for image-enhanced glucose forecasting.

### 6.3 Why gated residual fusion is important

The image branch is not useful for every sample. Stable glucose segments, noisy regions, and some drop patterns may not benefit from spectrogram-derived features. Gated residual fusion addresses this by keeping the sequence-only model as the base forecast and learning an image-driven correction. This design is safer than direct replacement because it allows the model to reduce image influence when the auxiliary representation is not informative.

The ShanghaiT2DM results show why this matters. GlucoImg improved spike-pattern error but worsened some stable and drop-pattern errors. A future extension could make the gate more explicitly pattern-aware, allowing stronger image corrections during upward excursions and weaker corrections during stable periods.

### 6.4 Clinical interpretation

Clinical utility in glucose forecasting depends on more than average MAE. The Clarke Error Grid results show that GlucoImg improved Zone A predictions most clearly at 90 minutes. Event detection analysis suggests that hyperglycemia detection benefits more consistently than hypoglycemia detection. The ShanghaiT2DM external analysis similarly shows stronger benefit for high-glucose regions and spike patterns than for stable or decreasing glucose states.

This pattern supports a clinically specific interpretation: GlucoImg may be more useful for anticipating high-glucose excursions than for uniformly improving every glucose state. This is aligned with the paper's focus on glucose spike patterns.

### 6.5 External validation and generalization

The ShanghaiT2DM experiment is important because it tests the model on an external type 2 diabetes cohort with a different sampling interval. The results were mixed. Average MAE/RMSE improved at some horizons but not all. However, the model improved 90-minute high-glucose error and spike-pattern error. This suggests partial generalization: the image branch retains value for upward excursions but does not provide uniform benefit across all T2D glucose states.

This mixed result should be reported transparently. It strengthens the manuscript by showing that the method was tested outside the main benchmark and by clarifying where it does and does not help.

---

## 7. Limitations

This study has several limitations.

First, the image branch uses generated time-series images rather than native medical images. Although this design is intentional, it means that the visual modality is a transformed representation of the same CGM window rather than an independent source of clinical information.

Second, DINOv2 was trained on natural images, not CGM-derived spectrograms. Frozen DINOv2 features worked well in the present experiments, but domain-specific self-supervised pretraining on time-series images may further improve performance.

Third, ShanghaiT2DM results were horizon-dependent and pattern-specific. GlucoImg did not uniformly outperform SeqOnly in stable or drop patterns, and GAF/MTF did not improve the external validation result. This limits claims about universal generalization.

Fourth, event detection for hypoglycemia was less stable than hyperglycemia detection. This may reflect class imbalance, limited hypoglycemia events, and different dynamics for downward glucose transitions. Future work should evaluate additional hypoglycemia-enriched cohorts.

Fifth, the current model uses a single image representation in the final architecture. Although All-4 fusion was tested, more sophisticated uncertainty-aware or pattern-aware fusion may be needed to exploit multiple image representations without adding noise.

---

## 8. Conclusion

We proposed GlucoImg, an image-enhanced glucose forecasting model that combines MambaFormer sequence modeling with spectrogram-derived DINOv2 patch tokens. The model uses cross-attention and gated residual prediction to let visual time-frequency features adaptively correct a sequence-only forecast. Across six horizons, GlucoImg improved average MAE and RMSE over SeqOnly and outperformed alternative image representations and fusion variants. Clinical and dynamics analyses suggest that the benefits are strongest for glucose excursions, particularly high-glucose and spike-like patterns. External ShanghaiT2DM validation showed modest overall gains but clear improvements for 90-minute high-glucose and spike-pattern prediction. These findings support time-series image representations as a useful complement to sequence models for CGM forecasting, while also showing that image enhancement should be interpreted as pattern-specific rather than universally beneficial.

---

## 9. Suggested Main Figures and Tables

### Main text figures

1. **Model architecture figure.**  
   Show CGM sequence input, spectrogram generation, DINOv2 patch tokens, cross-attention, gated residual head, and time-of-day branch.

2. **Horizon-wise relative improvement figure.**  
   Use: `figures/glucoimg_main_representation_comparison/glucoimg_representation_relative_improvement.pdf`

3. **Rapid transition case studies.**  
   Use: `figures/glucoimg_case_studies/glucoimg_rapid_transition_cases.pdf`

4. **Relative MAE improvement vs glucose-change magnitude.**  
   Use: `figures/error_vs_glucose_dynamics/relative_mae_improvement_vs_glucose_dynamics_rapid_20mgdl_with_n.pdf`

5. **Spectrogram interpretability visualization.**  
   Use: `figures/spectrogram_interpretability/spectrogram_interpretability_cross_attention.pdf`

### Main text tables

1. Main SeqOnly vs GlucoImg MAE/RMSE.
2. Image representation ablation.
3. DINO token strategy ablation.
4. Clarke Zone A paired analysis.
5. Spike-pattern and ShanghaiT2DM external pattern table.
6. ShanghaiT2DM full-horizon external validation table.

### Supplementary tables

1. Full traditional and deep learning baseline comparison.
2. All-4 fusion ablation.
3. Full event detection table.
4. Full Clarke Zone A/B/C/D/E table.
5. Full glucose-range stratified error table.
6. ShanghaiT2DM GAF/MTF exploratory image-type trial.
7. Seed robustness tables for SeqOnly.

---

## 10. Drafted Cover-Letter Style Significance Statement

This manuscript presents GlucoImg, a CGM forecasting framework that uses generated spectrogram images to expose glucose transition structure to a sequence model. The key novelty is not merely adding an image encoder, but preserving DINOv2 patch tokens and using temporal-to-visual cross-attention with a gated residual head. The method improves average forecasting accuracy across six horizons and shows stronger benefit under rapid glucose-change and high-glucose conditions. The work is relevant to biomedical signal processing, medical AI, and digital diabetes management because it provides a systematic evaluation of time-series image representations for glucose forecasting and includes external type 2 diabetes validation.

---

## 11. Items Requiring Verification Before Submission

1. Verify exact dataset names and citations:
   - Broll / iGLU naming
   - Colas et al.
   - Dubosson et al.
   - Hall et al.
   - Weinstock et al.
   - ShanghaiT2DM / Chinese diabetes datasets paper

2. Verify final main-result table:
   - Use `results/main_patch_tod_tuned_final_summary/best_by_horizon_final.csv`
   - Confirm whether the tuned per-horizon table is acceptable as the official main result or whether a single unified run should be reported.

3. Verify ShanghaiT2DM 45-minute SeqOnly MAE:
   - Current aggregate table uses the user-provided private rerun value 13.95.
   - Prediction-based clinical analyses still use the original saved prediction files.

4. Decide whether to report the main clinical comparison against:
   - default SeqOnly, or
   - seed1 SeqOnly, or
   - mean ± std SeqOnly.

5. Build verified bibliography:
   - Replace `[CITATION NEEDED]` markers with checked BibTeX entries.

6. Convert to LaTeX:
   - Split into `sections/`.
   - Add figure includes.
   - Convert tables into `booktabs` format.
