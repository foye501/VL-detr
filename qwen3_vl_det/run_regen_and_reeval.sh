#!/usr/bin/env bash
set -euo pipefail

DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-qwen3_vl_det/data_synth_regen_eval50}"
SPLIT_NAME="${SPLIT_NAME:-eval}"
SAMPLES_PER_LEVEL="${SAMPLES_PER_LEVEL:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
SEED="${SEED:-17}"
EXPORT_IMAGES="${EXPORT_IMAGES:-1}"

LORA_DIR="${LORA_DIR:-qwen3_vl_det/checkpoints_lora_boxcount_baseline_v2}"
DINO_DIR="${DINO_DIR:-qwen3_vl_det/checkpoints_dino_fused_visual_boxcount_v2_t64}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"

START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
NUM_QUERIES="${NUM_QUERIES:-100}"
OBJ_THRESHOLD="${OBJ_THRESHOLD:-0.15}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
LM_MAX_NEW_TOKENS="${LM_MAX_NEW_TOKENS:-2048}"
REQUIRE_VISION="${REQUIRE_VISION:-1}"

SAMPLE_INDEX="${SAMPLE_INDEX:-73}"
ANALYSIS_DIR="${ANALYSIS_DIR:-qwen3_vl_det/analysis_regen_eval50_lora_vs_dino}"

DATASET_FROM_DISK="${DATASET_FROM_DISK:-${DATA_OUTPUT_DIR}/${SPLIT_NAME}}"
LORA_JSON="${LORA_JSON:-${LORA_DIR}/eval_regen_split.json}"
DINO_JSON="${DINO_JSON:-${DINO_DIR}/eval_regen_split.json}"

gen_cmd=(
  python -m qwen3_vl_det.generate_synth_counting_dataset
  --output-dir "${DATA_OUTPUT_DIR}"
  --split-name "${SPLIT_NAME}"
  --samples-per-level "${SAMPLES_PER_LEVEL}"
  --image-size "${IMAGE_SIZE}"
  --seed "${SEED}"
)
if [[ "${EXPORT_IMAGES}" == "1" ]]; then
  gen_cmd+=(--export-images)
fi

eval_common=(
  --dataset-from-disk "${DATASET_FROM_DISK}"
  --split "${SPLIT_NAME}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
  --num-queries "${NUM_QUERIES}"
  --obj-threshold "${OBJ_THRESHOLD}"
  --iou-threshold "${IOU_THRESHOLD}"
  --box-coord-mode norm1000
  --box-coord-order yxyx
  --box-supervision-source target
  --lm-max-new-tokens "${LM_MAX_NEW_TOKENS}"
)
if [[ "${REQUIRE_VISION}" == "1" ]]; then
  eval_common+=(--require-vision)
fi

lora_eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${LORA_DIR}/last"
  --model-name "${MODEL_NAME}"
  "${eval_common[@]}"
  --output-json "${LORA_JSON}"
)

dino_eval_cmd=(
  python -m qwen3_vl_det.eval_split_aux
  --checkpoint-dir "${DINO_DIR}/last"
  --model-name "${MODEL_NAME}"
  "${eval_common[@]}"
  --output-json "${DINO_JSON}"
)

lora_one_cmd=(
  python -m qwen3_vl_det.eval_one_aux
  --checkpoint-dir "${LORA_DIR}/last"
  --model-name "${MODEL_NAME}"
  --dataset-from-disk "${DATASET_FROM_DISK}"
  --split "${SPLIT_NAME}"
  --sample-index "${SAMPLE_INDEX}"
  --lm-max-new-tokens 4096
  --output-image "${LORA_DIR}/eval_regen_one_${SAMPLE_INDEX}.png"
)
if [[ "${REQUIRE_VISION}" == "1" ]]; then
  lora_one_cmd+=(--require-vision)
fi

dino_one_cmd=(
  python -m qwen3_vl_det.eval_one_aux
  --checkpoint-dir "${DINO_DIR}/last"
  --model-name "${MODEL_NAME}"
  --dataset-from-disk "${DATASET_FROM_DISK}"
  --split "${SPLIT_NAME}"
  --sample-index "${SAMPLE_INDEX}"
  --lm-max-new-tokens 4096
  --output-image "${DINO_DIR}/eval_regen_one_${SAMPLE_INDEX}.png"
)
if [[ "${REQUIRE_VISION}" == "1" ]]; then
  dino_one_cmd+=(--require-vision)
fi

compare_cmd=(
  python -m qwen3_vl_det.plot_aux_eval
  --input-json "${LORA_JSON}"
  --input-json "${DINO_JSON}"
  --labels "lora_regen,dino_fused_t64_regen"
  --output-dir "${ANALYSIS_DIR}"
  --title "Regen Synthetic Eval: LoRA-only vs DINO fused visual t64"
)

printf '%q ' "${gen_cmd[@]}"; echo
"${gen_cmd[@]}"

printf '%q ' "${lora_eval_cmd[@]}"; echo
"${lora_eval_cmd[@]}"

printf '%q ' "${dino_eval_cmd[@]}"; echo
"${dino_eval_cmd[@]}"

printf '%q ' "${lora_one_cmd[@]}"; echo
"${lora_one_cmd[@]}"

printf '%q ' "${dino_one_cmd[@]}"; echo
"${dino_one_cmd[@]}"

printf '%q ' "${compare_cmd[@]}"; echo
"${compare_cmd[@]}"

echo "Fresh dataset: ${DATASET_FROM_DISK}"
echo "LoRA eval json: ${LORA_JSON}"
echo "DINO eval json: ${DINO_JSON}"
echo "Comparison plots: ${ANALYSIS_DIR}"
