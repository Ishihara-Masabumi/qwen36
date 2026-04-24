#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3.6-35B-A3B-FP8}"
OUTPUT_DIR="${2:-results/mgsm_ja_vllm}"

mkdir -p "${OUTPUT_DIR}"

conda run -n qwen36 python scripts/eval_mgsm_ja_vllm.py \
  --model "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"

