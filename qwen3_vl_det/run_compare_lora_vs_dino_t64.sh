#!/usr/bin/env bash
set -euo pipefail

LORA_DIR="${LORA_DIR:-qwen3_vl_det/checkpoints_lora_boxcount_baseline_v2}"
DINO_DIR="${DINO_DIR:-qwen3_vl_det/checkpoints_dino_fused_visual_boxcount_v2_t64}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen3_vl_det/analysis_lora_vs_dino_t64}"

LORA_JSON="${LORA_JSON:-${LORA_DIR}/eval_split_both.json}"
DINO_JSON="${DINO_JSON:-${DINO_DIR}/eval_split_both.json}"

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python -m qwen3_vl_det.plot_aux_eval
  --input-json "${LORA_JSON}"
  --input-json "${DINO_JSON}"
  --labels "lora_boxcount_v2,dino_fused_t64"
  --output-dir "${OUTPUT_DIR}"
  --title "LoRA-only vs DINO fused visual t64"
)

printf '%q ' "${cmd[@]}"; echo
"${cmd[@]}"

echo "LoRA eval:  ${LORA_JSON}"
echo "DINO eval:  ${DINO_JSON}"
echo "Plots:      ${OUTPUT_DIR}"
