#!/usr/bin/env bash
set -euo pipefail

# Two-stage training + evaluation pipeline for auxiliary DETR on Qwen3-VL.
#
# Stage 1 (DETR warmup):
#   - lm_weight=0.0
#   - det_weight=1.0
# Stage 2 (Joint):
#   - lm_weight=1.0
#   - det_weight configurable (default 0.3)
#
# Usage:
#   bash qwen3_vl_det/run_two_stage_aux.sh
#
# Important defaults are tuned for your recent setup:
#   - strict vision-only DETR memory
#   - vision LoRA enabled
#   - LM target mode = box_count (grounded count supervision)
#   - target-only counting objective
#
# Optional key overrides:
#   DATASET_FROM_DISK=qwen3_vl_det/data_synth_v2/train
#   EXP_ROOT=qwen3_vl_det/checkpoints_aux_two_stage
#   EPOCHS_STAGE1=1
#   EPOCHS_STAGE2=3
#   START_INDEX=3000
#   MAX_SAMPLES=1000

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_NAME="${DATASET_NAME:-foye501/VLM-Counting-dataset-qwenvl-sharegpt}"
DATASET_FROM_DISK="${DATASET_FROM_DISK:-}"
SPLIT="${SPLIT:-train}"

EXP_ROOT="${EXP_ROOT:-qwen3_vl_det/checkpoints_aux_two_stage}"
STAGE1_DIR="${STAGE1_DIR:-${EXP_ROOT}/stage1_warmup}"
STAGE2_DIR="${STAGE2_DIR:-${EXP_ROOT}/stage2_joint}"

NUM_QUERIES="${NUM_QUERIES:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_STAGE1="${LR_STAGE1:-1e-5}"
LR_STAGE2="${LR_STAGE2:-1e-5}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-1}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-3}"
MAX_STEPS_STAGE1="${MAX_STEPS_STAGE1:-0}"
MAX_STEPS_STAGE2="${MAX_STEPS_STAGE2:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
START_INDEX="${START_INDEX:-3000}"
SAMPLE_INDEX="${SAMPLE_INDEX:-3200}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

BOX_MODE="${BOX_MODE:-norm1000}"
BOX_ORDER="${BOX_ORDER:-yxyx}"
OBJ_THRESHOLD="${OBJ_THRESHOLD:-0.15}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"

# DETR/LM loss config
LM_WEIGHT_STAGE1="${LM_WEIGHT_STAGE1:-0.0}"
DET_WEIGHT_STAGE1="${DET_WEIGHT_STAGE1:-1.0}"
LM_WEIGHT_STAGE2="${LM_WEIGHT_STAGE2:-1.0}"
DET_WEIGHT_STAGE2="${DET_WEIGHT_STAGE2:-0.3}"
NO_OBJECT_WEIGHT="${NO_OBJECT_WEIGHT:-0.1}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.05}"
OBJ_BIAS_INIT="${OBJ_BIAS_INIT:--2.0}"

# Target alignment defaults:
# - LM counts target objects
# - DETR also supervised on target boxes by default
#   (set DETR_BOX_SOURCE=all + COUNT_LOSS_WEIGHT=0.0 if you want DETR over all objects)
LM_TARGET_MODE="${LM_TARGET_MODE:-box_count}"      # dataset|count_only|box_count|mixed
LM_BOX_SOURCE="${LM_BOX_SOURCE:-target}"           # target|all
DETR_BOX_SOURCE="${DETR_BOX_SOURCE:-target}"       # target|all
LM_BOX_RATIO="${LM_BOX_RATIO:-0.5}"                # for mixed only
LM_BOX_OUTPUT_MODE="${LM_BOX_OUTPUT_MODE:-norm1000}"  # absolute|norm1000|norm01
LM_BOX_OUTPUT_ORDER="${LM_BOX_OUTPUT_ORDER:-yxyx}"    # xyxy|yxyx
LM_COUNT_FIRST="${LM_COUNT_FIRST:-1}"
LM_APPEND_BOX_INSTRUCTION="${LM_APPEND_BOX_INSTRUCTION:-1}"
LM_MAX_NEW_TOKENS="${LM_MAX_NEW_TOKENS:-768}"

# LoRA/vision config
USE_LORA="${USE_LORA:-1}"
ENABLE_VISION_LORA="${ENABLE_VISION_LORA:-1}"
FREEZE_VISION_BACKBONE="${FREEZE_VISION_BACKBONE:-0}"
STRICT_VISION_MEMORY="${STRICT_VISION_MEMORY:-1}"
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS:-1}"
MERGE_LORA_ON_SAVE="${MERGE_LORA_ON_SAVE:-1}"
INJECT_DET_TO_LM_STAGE1="${INJECT_DET_TO_LM_STAGE1:-0}"
INJECT_DET_TO_LM_STAGE2="${INJECT_DET_TO_LM_STAGE2:-1}"
DETACH_DET_QUERIES_FOR_LM="${DETACH_DET_QUERIES_FOR_LM:-0}"
USE_DINO_FUSION="${USE_DINO_FUSION:-0}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-facebook/dinov2-base}"
DINO_TRAINABLE="${DINO_TRAINABLE:-0}"
DINO_DROP_CLS_TOKEN="${DINO_DROP_CLS_TOKEN:-1}"
DINO_CROSS_ATTN_HEADS="${DINO_CROSS_ATTN_HEADS:-8}"
DINO_CROSS_ATTN_DROPOUT="${DINO_CROSS_ATTN_DROPOUT:-0.0}"
DINO_GATE_INIT="${DINO_GATE_INIT:-0.1}"

mkdir -p "${STAGE1_DIR}" "${STAGE2_DIR}"

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
  --lm-box-ratio "${LM_BOX_RATIO}"
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
  common_train_args+=(
    --use-dino-fusion
    --dino-model-name "${DINO_MODEL_NAME}"
    --dino-cross-attn-heads "${DINO_CROSS_ATTN_HEADS}"
    --dino-cross-attn-dropout "${DINO_CROSS_ATTN_DROPOUT}"
    --dino-gate-init "${DINO_GATE_INIT}"
  )
  if [[ "${DINO_TRAINABLE}" == "1" ]]; then
    common_train_args+=(--dino-trainable)
  fi
  if [[ "${DINO_DROP_CLS_TOKEN}" == "1" ]]; then
    common_train_args+=(--dino-drop-cls-token)
  else
    common_train_args+=(--keep-dino-cls-token)
  fi
fi

stage1_fusion_args=()
if [[ "${INJECT_DET_TO_LM_STAGE1}" == "1" ]]; then
  stage1_fusion_args+=(--inject-det-queries-to-lm)
  if [[ "${DETACH_DET_QUERIES_FOR_LM}" == "1" ]]; then
    stage1_fusion_args+=(--detach-det-queries-for-lm)
  fi
fi

stage2_fusion_args=()
if [[ "${INJECT_DET_TO_LM_STAGE2}" == "1" ]]; then
  stage2_fusion_args+=(--inject-det-queries-to-lm)
  if [[ "${DETACH_DET_QUERIES_FOR_LM}" == "1" ]]; then
    stage2_fusion_args+=(--detach-det-queries-for-lm)
  fi
fi

echo "== Stage 1: DETR Warmup =="
stage1_cmd=(
  python -m qwen3_vl_det.train_sharegpt_aux
  --model-name "${MODEL_NAME}"
  --output-dir "${STAGE1_DIR}"
  --epochs "${EPOCHS_STAGE1}"
  --max-steps "${MAX_STEPS_STAGE1}"
  --lr "${LR_STAGE1}"
  --lm-weight "${LM_WEIGHT_STAGE1}"
  --det-weight "${DET_WEIGHT_STAGE1}"
  "${stage1_fusion_args[@]}"
  "${common_train_args[@]}"
)
printf '%q ' "${stage1_cmd[@]}"; echo
"${stage1_cmd[@]}"

echo "== Stage 2: Joint LM + DETR =="
stage1_ckpt="${STAGE1_DIR}/last"
stage2_cmd=(
  python -m qwen3_vl_det.train_sharegpt_aux
  --model-name "${stage1_ckpt}"
  --output-dir "${STAGE2_DIR}"
  --epochs "${EPOCHS_STAGE2}"
  --max-steps "${MAX_STEPS_STAGE2}"
  --lr "${LR_STAGE2}"
  --lm-weight "${LM_WEIGHT_STAGE2}"
  --det-weight "${DET_WEIGHT_STAGE2}"
  "${stage2_fusion_args[@]}"
  "${common_train_args[@]}"
)
printf '%q ' "${stage2_cmd[@]}"; echo
"${stage2_cmd[@]}"

echo "== Split Eval: Stage 2 =="
eval_json="${STAGE2_DIR}/eval_split_both.json"
overlay_dir="${STAGE2_DIR}/eval_overlays"
eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${STAGE2_DIR}/last"
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
  --output-json "${eval_json}"
  --save-overlays
  --overlay-dir "${overlay_dir}"
)
if [[ -n "${DATASET_FROM_DISK}" ]]; then
  eval_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  eval_cmd+=(--dataset-name "${DATASET_NAME}")
fi
printf '%q ' "${eval_cmd[@]}"; echo
"${eval_cmd[@]}"

echo "== One Sample Debug (Stage 2) =="
one_cmd=(
  python -m qwen3_vl_det.eval_one_aux
  --checkpoint-dir "${STAGE2_DIR}/last"
  --split "${SPLIT}"
  --sample-index "${SAMPLE_INDEX}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --box-coord-mode "${BOX_MODE}"
  --box-coord-order "${BOX_ORDER}"
  --box-supervision-source "${DETR_BOX_SOURCE}"
  --lm-max-new-tokens "${LM_MAX_NEW_TOKENS}"
  --require-vision
  --annotate-scores
  --output-image "${STAGE2_DIR}/eval_one_${SAMPLE_INDEX}.png"
)
if [[ -n "${DATASET_FROM_DISK}" ]]; then
  one_cmd+=(--dataset-from-disk "${DATASET_FROM_DISK}")
else
  one_cmd+=(--dataset-name "${DATASET_NAME}")
fi
printf '%q ' "${one_cmd[@]}"; echo
"${one_cmd[@]}"

echo "== Plot Analysis =="
plot_cmd=(
  python -m qwen3_vl_det.plot_aux_eval
  --input-json "${eval_json}"
  --output-dir "${STAGE2_DIR}/analysis_plots"
  --title "Two-Stage Aux (Stage2)"
)
printf '%q ' "${plot_cmd[@]}"; echo
"${plot_cmd[@]}"

echo "Done."
echo "Stage1 ckpt: ${STAGE1_DIR}/last"
echo "Stage2 ckpt: ${STAGE2_DIR}/last"
echo "Eval JSON:   ${eval_json}"
echo "Overlays:    ${overlay_dir}"
