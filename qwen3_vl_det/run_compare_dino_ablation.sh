#!/usr/bin/env bash
set -euo pipefail

# Compare:
# 1. LM-only DINO fusion (no DETR supervision, query injection on)
# 2. DINO fusion + auxiliary DETR two-stage training
#
# Usage:
#   bash qwen3_vl_det/run_compare_dino_ablation.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_NAME="${DATASET_NAME:-foye501/VLM-Counting-dataset-qwenvl-sharegpt}"
DATASET_FROM_DISK="${DATASET_FROM_DISK:-}"
SPLIT="${SPLIT:-train}"

EXP_ROOT="${EXP_ROOT:-qwen3_vl_det/checkpoints_dino_compare}"
BASELINE_DIR="${BASELINE_DIR:-${EXP_ROOT}/lm_only_dino}"
MAIN_ROOT="${MAIN_ROOT:-${EXP_ROOT}/dino_plus_aux_detr}"
MAIN_STAGE2_DIR="${MAIN_STAGE2_DIR:-${MAIN_ROOT}/stage2_joint}"

NUM_QUERIES="${NUM_QUERIES:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_BASELINE="${LR_BASELINE:-1e-5}"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-3}"
LR_STAGE1="${LR_STAGE1:-1e-5}"
LR_STAGE2="${LR_STAGE2:-1e-5}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-1}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_STEPS_BASELINE="${MAX_STEPS_BASELINE:-0}"
MAX_STEPS_STAGE1="${MAX_STEPS_STAGE1:-0}"
MAX_STEPS_STAGE2="${MAX_STEPS_STAGE2:-0}"

BOX_MODE="${BOX_MODE:-norm1000}"
BOX_ORDER="${BOX_ORDER:-yxyx}"
OBJ_THRESHOLD="${OBJ_THRESHOLD:-0.15}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
START_INDEX="${START_INDEX:-3000}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_INDEX="${SAMPLE_INDEX:-3200}"

LM_TARGET_MODE="${LM_TARGET_MODE:-box_count}"
LM_BOX_SOURCE="${LM_BOX_SOURCE:-target}"
DETR_BOX_SOURCE="${DETR_BOX_SOURCE:-target}"
LM_BOX_OUTPUT_MODE="${LM_BOX_OUTPUT_MODE:-norm1000}"
LM_BOX_OUTPUT_ORDER="${LM_BOX_OUTPUT_ORDER:-yxyx}"
LM_COUNT_FIRST="${LM_COUNT_FIRST:-1}"
LM_APPEND_BOX_INSTRUCTION="${LM_APPEND_BOX_INSTRUCTION:-1}"
LM_MAX_NEW_TOKENS="${LM_MAX_NEW_TOKENS:-768}"

USE_LORA="${USE_LORA:-1}"
ENABLE_VISION_LORA="${ENABLE_VISION_LORA:-1}"
FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE:-0}"
STRICT_VISION_MEMORY="${STRICT_VISION_MEMORY:-1}"
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS:-1}"
MERGE_LORA_ON_SAVE="${MERGE_LORA_ON_SAVE:-1}"

USE_DINO_FUSION="${USE_DINO_FUSION:-1}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-facebook/dinov2-base}"
DINO_TRAINABLE="${DINO_TRAINABLE:-0}"

NO_OBJECT_WEIGHT="${NO_OBJECT_WEIGHT:-0.1}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.05}"
OBJ_BIAS_INIT="${OBJ_BIAS_INIT:--2.0}"

mkdir -p "${BASELINE_DIR}" "${MAIN_ROOT}"

common_train_args=(
  --train-split "${SPLIT}"
  --require-vision
  --num-queries "${NUM_QUERIES}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --max-length "${MAX_LENGTH}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-target-mode "${LM_TARGET_MODE}"
  --lm-box-source "${LM_BOX_SOURCE}"
  --lm-box-output-mode "${LM_BOX_OUTPUT_MODE}"
  --lm-box-output-order "${LM_BOX_OUTPUT_ORDER}"
  --no-object-weight "${NO_OBJECT_WEIGHT}"
  --count-loss-weight "${COUNT_LOSS_WEIGHT}"
  --obj-bias-init "${OBJ_BIAS_INIT}"
)

if [[ -n "${DATASET_FROM_DISK}" ]]; then
  common_train_args+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  common_train_args+=(--dataset-name "${DATASET_NAME}")
fi
if [[ "${USE_LORA}" == "1" ]]; then
  common_train_args+=(--use-lora)
fi
if [[ "${ENABLE_VISION_LORA}" == "1" ]]; then
  common_train_args+=(--enable-vision-lora)
fi
if [[ "${FREEZE_VISION_BACKBONE}" == "1" ]]; then
  common_train_args+=(--freeze-vision-backbone)
fi
if [[ "${STRICT_VISION_MEMORY}" == "1" ]]; then
  common_train_args+=(--strict-vision-memory)
else
  common_train_args+=(--allow-token-fallback-memory)
fi
if [[ "${ASSISTANT_ONLY_LOSS}" == "1" ]]; then
  common_train_args+=(--assistant-only-loss)
else
  common_train_args+=(--full-seq-loss)
fi
if [[ "${LM_COUNT_FIRST}" == "1" ]]; then
  common_train_args+=(--lm-count-first)
else
  common_train_args+=(--lm-count-last)
fi
if [[ "${LM_APPEND_BOX_INSTRUCTION}" == "1" ]]; then
  common_train_args+=(--lm-append-box-instruction)
else
  common_train_args+=(--no-lm-append-box-instruction)
fi
if [[ "${MERGE_LORA_ON_SAVE}" == "1" ]]; then
  common_train_args+=(--merge-lora-on-save)
else
  common_train_args+=(--no-merge-lora-on-save)
fi
if [[ "${USE_DINO_FUSION}" == "1" ]]; then
  common_train_args+=(--use-dino-fusion --dino-model-name "${DINO_MODEL_NAME}")
  if [[ "${DINO_TRAINABLE}" == "1" ]]; then
    common_train_args+=(--dino-trainable)
  fi
fi

echo "== Baseline: LM-only DINO fusion =="
baseline_train_cmd=(
  python -m qwen3_vl_det.train_sharegpt_aux
  --model-name "${MODEL_NAME}"
  --output-dir "${BASELINE_DIR}"
  --epochs "${EPOCHS_BASELINE}"
  --max-steps "${MAX_STEPS_BASELINE}"
  --lr "${LR_BASELINE}"
  --lm-weight 1.0
  --det-weight 0.0
  --inject-det-queries-to-lm
  "${common_train_args[@]}"
)
printf '%q ' "${baseline_train_cmd[@]}"; echo
"${baseline_train_cmd[@]}"

baseline_eval_json="${BASELINE_DIR}/eval_split_both.json"
baseline_eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${BASELINE_DIR}/last"
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --iou-threshold "${IOU_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-max-new-tokens "${LM_MAX_NEW_TOKENS}"
  --require-vision
  --output-json "${baseline_eval_json}"
)
if [[ -n "${DATASET_FROM_DISK}" ]]; then
  baseline_eval_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  baseline_eval_cmd+=(--dataset-name "${DATASET_NAME}")
fi
printf '%q ' "${baseline_eval_cmd[@]}"; echo
"${baseline_eval_cmd[@]}"

echo "== Main: DINO fusion + auxiliary DETR =="
main_cmd=(
  bash qwen3_vl_det/run_two_stage_aux.sh
)
printf '%q ' \
  "MODEL_NAME=${MODEL_NAME}" \
  "DATASET_NAME=${DATASET_NAME}" \
  "DATASET_FROM_DISK=${DATASET_FROM_DISK}" \
  "SPLIT=${SPLIT}" \
  "EXP_ROOT=${MAIN_ROOT}" \
  "NUM_QUERIES=${NUM_QUERIES}" \
  "BATCH_SIZE=${BATCH_SIZE}" \
  "GRAD_ACCUM=${GRAD_ACCUM}" \
  "LR_STAGE1=${LR_STAGE1}" \
  "LR_STAGE2=${LR_STAGE2}" \
  "EPOCHS_STAGE1=${EPOCHS_STAGE1}" \
  "EPOCHS_STAGE2=${EPOCHS_STAGE2}" \
  "MAX_STEPS_STAGE1=${MAX_STEPS_STAGE1}" \
  "MAX_STEPS_STAGE2=${MAX_STEPS_STAGE2}" \
  "MAX_LENGTH=${MAX_LENGTH}" \
  "BOX_MODE=${BOX_MODE}" \
  "BOX_ORDER=${BOX_ORDER}" \
  "OBJ_THRESHOLD=${OBJ_THRESHOLD}" \
  "IOU_THRESHOLD=${IOU_THRESHOLD}" \
  "START_INDEX=${START_INDEX}" \
  "MAX_SAMPLES=${MAX_SAMPLES}" \
  "SAMPLE_INDEX=${SAMPLE_INDEX}" \
  "LM_TARGET_MODE=${LM_TARGET_MODE}" \
  "LM_BOX_SOURCE=${LM_BOX_SOURCE}" \
  "DETR_BOX_SOURCE=${DETR_BOX_SOURCE}" \
  "LM_BOX_OUTPUT_MODE=${LM_BOX_OUTPUT_MODE}" \
  "LM_BOX_OUTPUT_ORDER=${LM_BOX_OUTPUT_ORDER}" \
  "LM_COUNT_FIRST=${LM_COUNT_FIRST}" \
  "LM_APPEND_BOX_INSTRUCTION=${LM_APPEND_BOX_INSTRUCTION}" \
  "LM_MAX_NEW_TOKENS=${LM_MAX_NEW_TOKENS}" \
  "USE_LORA=${USE_LORA}" \
  "ENABLE_VISION_LORA=${ENABLE_VISION_LORA}" \
  "FREEZE_VISION_BACKBONE=${FREEZE_VISION_BACKBONE}" \
  "STRICT_VISION_MEMORY=${STRICT_VISION_MEMORY}" \
  "ASSISTANT_ONLY_LOSS=${ASSISTANT_ONLY_LOSS}" \
  "MERGE_LORA_ON_SAVE=${MERGE_LORA_ON_SAVE}" \
  "USE_DINO_FUSION=${USE_DINO_FUSION}" \
  "DINO_MODEL_NAME=${DINO_MODEL_NAME}" \
  "DINO_TRAINABLE=${DINO_TRAINABLE}" \
  "NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT}" \
  "COUNT_LOSS_WEIGHT=${COUNT_LOSS_WEIGHT}" \
  "OBJ_BIAS_INIT=${OBJ_BIAS_INIT}" \
  "INJECT_DET_TO_LM_STAGE1=0" \
  "INJECT_DET_TO_LM_STAGE2=1"; echo
MODEL_NAME="${MODEL_NAME}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_FROM_DISK="${DATASET_FROM_DISK}" \
SPLIT="${SPLIT}" \
EXP_ROOT="${MAIN_ROOT}" \
NUM_QUERIES="${NUM_QUERIES}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
LR_STAGE1="${LR_STAGE1}" \
LR_STAGE2="${LR_STAGE2}" \
EPOCHS_STAGE1="${EPOCHS_STAGE1}" \
EPOCHS_STAGE2="${EPOCHS_STAGE2}" \
MAX_STEPS_STAGE1="${MAX_STEPS_STAGE1}" \
MAX_STEPS_STAGE2="${MAX_STEPS_STAGE2}" \
MAX_LENGTH="${MAX_LENGTH}" \
BOX_MODE="${BOX_MODE}" \
BOX_ORDER="${BOX_ORDER}" \
OBJ_THRESHOLD="${OBJ_THRESHOLD}" \
IOU_THRESHOLD="${IOU_THRESHOLD}" \
START_INDEX="${START_INDEX}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
SAMPLE_INDEX="${SAMPLE_INDEX}" \
LM_TARGET_MODE="${LM_TARGET_MODE}" \
LM_BOX_SOURCE="${LM_BOX_SOURCE}" \
DETR_BOX_SOURCE="${DETR_BOX_SOURCE}" \
LM_BOX_OUTPUT_MODE="${LM_BOX_OUTPUT_MODE}" \
LM_BOX_OUTPUT_ORDER="${LM_BOX_OUTPUT_ORDER}" \
LM_COUNT_FIRST="${LM_COUNT_FIRST}" \
LM_APPEND_BOX_INSTRUCTION="${LM_APPEND_BOX_INSTRUCTION}" \
LM_MAX_NEW_TOKENS="${LM_MAX_NEW_TOKENS}" \
USE_LORA="${USE_LORA}" \
ENABLE_VISION_LORA="${ENABLE_VISION_LORA}" \
FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE}" \
STRICT_VISION_MEMORY="${STRICT_VISION_MEMORY}" \
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS}" \
MERGE_LORA_ON_SAVE="${MERGE_LORA_ON_SAVE}" \
USE_DINO_FUSION="${USE_DINO_FUSION}" \
DINO_MODEL_NAME="${DINO_MODEL_NAME}" \
DINO_TRAINABLE="${DINO_TRAINABLE}" \
NO_OBJECT_WEIGHT="${NO_OBJECT_WEIGHT}" \
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT}" \
OBJ_BIAS_INIT="${OBJ_BIAS_INIT}" \
INJECT_DET_TO_LM_STAGE1=0 \
INJECT_DET_TO_LM_STAGE2=1 \
"${main_cmd[@]}"

echo "== Comparison plots =="
compare_plot_cmd=(
  python -m qwen3_vl_det.plot_aux_eval
  --input-json "${baseline_eval_json}"
  --input-json "${MAIN_STAGE2_DIR}/eval_split_both.json"
  --labels "lm_only_dino,dino_plus_aux_detr"
  --output-dir "${EXP_ROOT}/compare_plots"
  --title "DINO Ablation"
)
printf '%q ' "${compare_plot_cmd[@]}"; echo
"${compare_plot_cmd[@]}"

echo "Done."
echo "Baseline eval: ${baseline_eval_json}"
echo "Main eval:     ${MAIN_STAGE2_DIR}/eval_split_both.json"
echo "Compare plots: ${EXP_ROOT}/compare_plots"
