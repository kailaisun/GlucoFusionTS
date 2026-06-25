# GlucoImg

### Image-Enhanced Learning for Continuous Glucose Forecasting

Official research code for **GlucoImg**, an image-enhanced learning framework that combines continuous glucose monitoring (CGM) sequences with visual representations derived from the same glucose trajectories.

GlucoImg investigates whether recurrence plots (RP), Gramian angular fields (GAF), Markov transition fields (MTF), and spectrograms provide complementary information beyond conventional one-dimensional sequence modeling. The framework integrates a MambaFormer-based temporal encoder, a frozen DINOv2 image encoder, cyclic time-of-day features, cross-attention, and gated residual fusion for glucose forecasting at horizons from 15 to 90 minutes.

## Research Scope

The accompanying study evaluates GlucoImg across:

- five benchmark CGM cohorts curated under the GLUCOBENCH framework;
- non-diagnosed, type 1 diabetes, and type 2 diabetes populations;
- forecasting horizons of 15, 30, 45, 60, 75, and 90 minutes;
- multiple time-series image representations;
- an independent T2D cohort with 15-minute CGM sampling
- predictive accuracy, clinical reliability, event detection, and interpretability analyses.

## Method Overview

GlucoImg contains three complementary branches:

1. **CGM sequence branch** — a MambaFormer encoder models temporal dependencies in an eight-hour glucose history.
2. **Time-series image branch** — the same CGM window is converted into RP, GAF, MTF, or spectrogram images and encoded using a frozen DINOv2 backbone.
3. **Time-of-day branch** — cyclic sine/cosine features provide circadian context.

Image and sequence representations are integrated through cross-attention. A learned gate controls the contribution of the image-enhanced residual prediction:

```text
prediction = sequence-only prediction + gate × image-enhanced residual
```

Separate models are trained for each forecasting horizon.

## Repository Structure

```text
GlucoFusionTS/
├── bin/                     Shell scripts for running experiments
├── config/                  Dataset and preprocessing configurations
├── data_formatter/          Data loading, segmentation, and formatting
├── exploratory_analysis/    Dataset exploration notebooks
├── lib/                     Models, image generation, and core methods
├── utils/                   Shared training and evaluation utilities
├── export_clarke_specpatch.py
│                            Clarke Error Grid export and evaluation
├── train_mamba_single_img.py
│                            Main GlucoImg training entry point
├── requirements.txt         Python dependencies
└── README.md                Project documentation
```


## Installation

The experiments reported in the paper were conducted on Ubuntu 24.04 with Python 3.12 and NVIDIA GPUs. A CUDA-capable GPU is strongly recommended.

```bash
git clone https://github.com/kailaisun/GlucoFusionTS.git
cd GlucoFusionTS

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The image encoder is loaded through `torch.hub` from the official DINOv2 repository on first use. An internet connection is therefore required for the initial model download.


## Running GlucoImg

### Main spectrogram model

The following command trains the spectrogram-based GlucoImg model with patch-token DINOv2 features, time-of-day encoding, cross-attention, and gated residual fusion:

```bash
python -u train_mamba_single_img.py \
  --image_type spectrogram \
  --in_len 96 \
  --gpu 0 \
  --fusion_mode gated_residual \
  --dino_pool none \
  --use_tod \
  --horizons 15,30,45,60,75,90 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --seed 0 \
```

For the five-minute benchmark datasets, `--in_len 96` corresponds to an eight-hour input window. One model is fitted for each requested forecasting horizon.

### Compare individual image representations

Set `--image_type` to one of:

```text
rp | gaf | mtf | spectrogram
```

For example:

```bash
python -u train_mamba_single_img.py \
  --image_type rp \
  --gpu 0 \
  --fusion_mode gated_residual \
  --dino_pool none \
  --horizons 15,30,45,60,75,90 \
```

A batch script for pooled single-representation experiments is also provided:

```bash
GPU=0 bash bin/run_mambaformer_single_image_gated_pooled.sh
```


## Evaluation

The paper evaluates forecasting performance after inverse transformation to the original mg/dL scale. The evaluation framework includes:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- Clarke Error Grid Zone A and Zone A+B agreement;
- hyperglycemia and hypoglycemia recall and F1 score;
- population-stratified and glucose-change-stratified analyses; and
- spectrogram correspondence and region-masking analyses.

Clarke Error Grid outputs can be generated with:

```bash
python export_clarke_specpatch.py --help
```

## Reproducibility Notes

- Benchmark CGM signals are sampled at five-minute intervals.
- The default benchmark input contains 96 observations (eight hours).
- Forecasting horizons are 15–90 minutes in 15-minute increments.
- Splits are subject-independent to prevent participant-level data leakage.
- The default random seed is `0`.
- Models are optimized with AdamW and selected using validation MAE.
- DINOv2 parameters are frozen in the main image branch.
- Hardware, driver, and library differences may introduce small numerical variation.

## Citation

If you use this code, please cite the accompanying manuscript. The citation will be updated when a DOI and final publication details become available.

```bibtex
@unpublished{cui_glucoimg,
  title  = {GlucoImg: Image-Enhanced Learning for Continuous Glucose Forecasting},
  author = {Cui, Yue and Sun, Kailai},
  note   = {Manuscript under review}
}
```

## Authors and Contact

- **Yue Cui** — Department of Dermatology, Chongqing General Hospital, Chongqing University<br>
  ORCID: [0009-0006-0836-8362](https://orcid.org/0009-0006-0836-8362) · Email: `cuiyue_medicine@cqu.edu.cn`
- **Kailai Sun** — Singapore-MIT Alliance for Research and Technology Centre and Massachusetts Institute of Technology<br>
  ORCID: [0000-0003-1648-3409](https://orcid.org/0000-0003-1648-3409) · Email: `skl24@mit.edu`

## Responsible Use

This software is provided for research and educational purposes only. It is not a medical device and is not intended for diagnosis, treatment selection, real-time clinical decision-making, or patient management. Independent validation and appropriate regulatory review are required before any clinical deployment.

## License

No software license is currently included in this repository. Until a license is added, reuse and redistribution are not automatically permitted. Please contact the authors regarding permissions.
