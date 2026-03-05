#!/usr/bin/env bash
set -euo pipefail

# One-shot training + evaluation for auxiliary DETR branch.
# Usage:
#   bash qwen3_vl_det/run_experiment_aux.sh
# Optional overrides:
#   MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
#   DATASET_NAME=foye501/VLM-Counting-dataset-qwenvl-sharegpt
#   DATASET_FROM_DISK=qwen3_vl_det/data_synth_v2/train
#   OUTPUT_DIR=qwen3_vl_det/checkpoints_aux_exp1
#   EPOCHS=3
#   NUM_QUERIES=100
#   BATCH_SIZE=4
#   GRAD_ACCUM=8
#   LR=2e-5
#   DET_WEIGHT=0.3

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_NAME="${DATASET_NAME:-foye501/VLM-Counting-dataset-qwenvl-sharegpt}"
DATASET_FROM_DISK="${DATASET_FROM_DISK:-}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen3_vl_det/checkpoints_aux_exp1}"
EPOCHS="${EPOCHS:-3}"
NUM_QUERIES="${NUM_QUERIES:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
DET_WEIGHT="${DET_WEIGHT:-0.3}"
BOX_MODE="${BOX_MODE:-norm1000}"
BOX_ORDER="${BOX_ORDER:-yxyx}"
OBJ_THRESHOLD="${OBJ_THRESHOLD:-0.35}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS:-1}"
START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_INDEX="${SAMPLE_INDEX:-100}"

mkdir -p "${OUTPUT_DIR}"

train_cmd=(
  python -m qwen3_vl_det.train_sharegpt_aux
  --model-name "${MODEL_NAME}"
  --train-split "${SPLIT}"
  --require-vision
  --box-supervision-source all
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --num-queries "${NUM_QUERIES}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --det-weight "${DET_WEIGHT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DATASET_FROM_DISK}" ]]; then
  train_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  train_cmd+=(--dataset-name "${DATASET_NAME}")
fi
if [[ "${USE_LORA}" == "1" ]]; then
  train_cmd+=(--use-lora --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}")
fi
if [[ "${ASSISTANT_ONLY_LOSS}" == "1" ]]; then
  train_cmd+=(--assistant-only-loss)
else
  train_cmd+=(--full-seq-loss)
fi

echo "== Training =="
printf '%q ' "${train_cmd[@]}"
echo
"${train_cmd[@]}"

eval_json="${OUTPUT_DIR}/eval_split.json"
overlay_dir="${OUTPUT_DIR}/eval_overlays"

eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${OUTPUT_DIR}/last"
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source all
  --require-vision
  --output-json "${eval_json}"
  --save-overlays
  --overlay-dir "${overlay_dir}"
)

if [[ -n "${DATASET_FROM_DISK}" ]]; then
  eval_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  eval_cmd+=(--dataset-name "${DATASET_NAME}")
fi

echo "== Split Evaluation =="
printf '%q ' "${eval_cmd[@]}"
echo
"${eval_cmd[@]}"

one_cmd=(
  python -m qwen3_vl_det.eval_one_aux
  --checkpoint-dir "${OUTPUT_DIR}/last"
  --split "${SPLIT}"
  --sample-index "${SAMPLE_INDEX}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --require-vision
  --output-image "${OUTPUT_DIR}/eval_one_${SAMPLE_INDEX}.png"
)

if [[ -n "${DATASET_FROM_DISK}" ]]; then
  one_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  one_cmd+=(--dataset-name "${DATASET_NAME}")
fi

echo "== One-Sample Debug =="
printf '%q ' "${one_cmd[@]}"
echo
"${one_cmd[@]}"

echo "Done."
echo "Checkpoint: ${OUTPUT_DIR}/last"
echo "Eval JSON:  ${eval_json}"
echo "Overlay dir:${overlay_dir}"
