#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-results/mambaformer_win96_single_img_gated_pooled}"
HORIZONS="${HORIZONS:-15,30,45,60,75,90}"

mkdir -p "$RESULTS_DIR" logs

for IMAGE_TYPE in rp spectrogram gaf mtf; do
  "$PYTHON" -u train_mamba_single_img.py \
    --image_type "$IMAGE_TYPE" \
    --in_len 96 \
    --gpu "$GPU" \
    --fusion_mode gated_residual \
    --dino_pool mean \
    --horizons "$HORIZONS" \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "logs/mambaformer_win96_${IMAGE_TYPE}_gated_pooled.log"
done
