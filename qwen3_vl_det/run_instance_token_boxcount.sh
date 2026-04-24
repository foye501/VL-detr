#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Stronger method branch:
# - fuse Qwen visual memory with DINO features
# - train DETR query decoder with Hungarian supervision
# - select top-K query outputs by objectness as object-level instance tokens
# - inject those instance tokens into the LM for count+box generation

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_NAME="${DATASET_NAME:-foye501/VLM-Counting-dataset-qwenvl-sharegpt}"
DATASET_FROM_DISK="${DATASET_FROM_DISK:-}"
SPLIT="${SPLIT:-train}"

OUTPUT_DIR="${OUTPUT_DIR:-qwen3_vl_det/checkpoints_instance_token_boxcount}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_QUERIES="${NUM_QUERIES:-100}"
INSTANCE_LM_NUM_TOKENS="${INSTANCE_LM_NUM_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-1e-4}"
ADAPTER_LR="${ADAPTER_LR:-2e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

BOX_MODE="${BOX_MODE:-norm1000}"
BOX_ORDER="${BOX_ORDER:-yxyx}"
LM_BOX_OUTPUT_MODE="${LM_BOX_OUTPUT_MODE:-norm1000}"
LM_BOX_OUTPUT_ORDER="${LM_BOX_OUTPUT_ORDER:-yxyx}"
LM_BOX_SOURCE="${LM_BOX_SOURCE:-target}"
DETR_BOX_SOURCE="${DETR_BOX_SOURCE:-all}"
OBJ_THRESHOLD="${OBJ_THRESHOLD:-0.15}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
START_INDEX="${START_INDEX:-3000}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_INDEX="${SAMPLE_INDEX:-3200}"
LM_MAX_NEW_TOKENS="${LM_MAX_NEW_TOKENS:-768}"

USE_LORA="${USE_LORA:-1}"
ENABLE_VISION_LORA="${ENABLE_VISION_LORA:-1}"
FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE:-0}"
STRICT_VISION_MEMORY="${STRICT_VISION_MEMORY:-1}"
STRICT_LM_FUSION="${STRICT_LM_FUSION:-1}"
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS:-1}"
MERGE_LORA_ON_SAVE="${MERGE_LORA_ON_SAVE:-1}"

USE_DINO_FUSION="${USE_DINO_FUSION:-1}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-facebook/dinov2-base}"
DINO_TRAINABLE="${DINO_TRAINABLE:-0}"

DET_WEIGHT="${DET_WEIGHT:-0.3}"
NO_OBJECT_WEIGHT="${NO_OBJECT_WEIGHT:-0.1}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.05}"
OBJ_BIAS_INIT="${OBJ_BIAS_INIT:--2.0}"

LOG_LM_GENERATE_EVERY="${LOG_LM_GENERATE_EVERY:-10}"
LOG_LM_GENERATE_MAX_NEW_TOKENS="${LOG_LM_GENERATE_MAX_NEW_TOKENS:-128}"
DEBUG_FIRST_BATCH="${DEBUG_FIRST_BATCH:-0}"

mkdir -p "${OUTPUT_DIR}"

train_cmd=(
  python -m qwen3_vl_det.train_sharegpt_aux
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --train-split "${SPLIT}"
  --require-vision
  --num-queries "${NUM_QUERIES}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --epochs "${EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --adapter-lr "${ADAPTER_LR}"
  --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
  --warmup-steps "${WARMUP_STEPS}"
  --max-length "${MAX_LENGTH}"
  --lm-weight 1.0
  --det-weight "${DET_WEIGHT}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-target-mode box_count
  --lm-box-source "${LM_BOX_SOURCE}"
  --lm-box-output-mode "${LM_BOX_OUTPUT_MODE}"
  --lm-box-output-order "${LM_BOX_OUTPUT_ORDER}"
  --lm-count-first
  --lm-append-box-instruction
  --inject-instance-tokens-to-lm
  --instance-lm-num-tokens "${INSTANCE_LM_NUM_TOKENS}"
  --no-object-weight "${NO_OBJECT_WEIGHT}"
  --count-loss-weight "${COUNT_LOSS_WEIGHT}"
  --obj-bias-init "${OBJ_BIAS_INIT}"
  --log-lm-generate-every "${LOG_LM_GENERATE_EVERY}"
  --log-lm-generate-max-new-tokens "${LOG_LM_GENERATE_MAX_NEW_TOKENS}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  train_cmd+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${DATASET_FROM_DISK}" ]]; then
  train_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  train_cmd+=(--dataset-name "${DATASET_NAME}")
fi
if [[ "${USE_LORA}" == "1" ]]; then
  train_cmd+=(--use-lora)
fi
if [[ "${ENABLE_VISION_LORA}" == "1" ]]; then
  train_cmd+=(--enable-vision-lora)
fi
if [[ "${FREEZE_VISION_BACKBONE}" == "1" ]]; then
  train_cmd+=(--freeze-vision-backbone)
fi
if [[ "${STRICT_VISION_MEMORY}" == "1" ]]; then
  train_cmd+=(--strict-vision-memory)
else
  train_cmd+=(--allow-token-fallback-memory)
fi
if [[ "${STRICT_LM_FUSION}" == "1" ]]; then
  train_cmd+=(--strict-lm-fusion)
fi
if [[ "${ASSISTANT_ONLY_LOSS}" == "1" ]]; then
  train_cmd+=(--assistant-only-loss)
else
  train_cmd+=(--full-seq-loss)
fi
if [[ "${MERGE_LORA_ON_SAVE}" == "1" ]]; then
  train_cmd+=(--merge-lora-on-save)
else
  train_cmd+=(--no-merge-lora-on-save)
fi
if [[ "${USE_DINO_FUSION}" == "1" ]]; then
  train_cmd+=(--use-dino-fusion --dino-model-name "${DINO_MODEL_NAME}")
fi
if [[ "${DINO_TRAINABLE}" == "1" ]]; then
  train_cmd+=(--dino-trainable)
fi
if [[ "${DEBUG_FIRST_BATCH}" == "1" ]]; then
  train_cmd+=(--debug-first-batch)
fi

printf '%q ' "${train_cmd[@]}"; echo
"${train_cmd[@]}"

eval_json="${OUTPUT_DIR}/eval_split_both.json"
eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${OUTPUT_DIR}/last"
  --model-name "${MODEL_NAME}"
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --iou-threshold "${IOU_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-box-supervision-source "${LM_BOX_SOURCE}"
  --lm-max-new-tokens "${LM_MAX_NEW_TOKENS}"
  --require-vision
  --output-json "${eval_json}"
)
if [[ -n "${DATASET_FROM_DISK}" ]]; then
  eval_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  eval_cmd+=(--dataset-name "${DATASET_NAME}")
fi
printf '%q ' "${eval_cmd[@]}"; echo
"${eval_cmd[@]}"

one_cmd=(
  python -m qwen3_vl_det.eval_one_aux
  --checkpoint-dir "${OUTPUT_DIR}/last"
  --model-name "${MODEL_NAME}"
  --split "${SPLIT}"
  --sample-index "${SAMPLE_INDEX}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-box-supervision-source "${LM_BOX_SOURCE}"
  --require-vision
  --lm-max-new-tokens "${LM_MAX_NEW_TOKENS}"
  --output-image "${OUTPUT_DIR}/eval_one_${SAMPLE_INDEX}.png"
)
if [[ -n "${DATASET_FROM_DISK}" ]]; then
  one_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  one_cmd+=(--dataset-name "${DATASET_NAME}")
fi
printf '%q ' "${one_cmd[@]}"; echo
"${one_cmd[@]}"
