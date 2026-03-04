# Qwen3-VL + DETR-Style Hungarian Loss (Adapter)

This folder gives you a practical integration path for your current Qwen3-VL research:
- keep standard LM loss
- add fixed query slots
- supervise slots with one-to-one Hungarian matching on GT boxes

It is designed as a patch layer for your existing training code.

## Auxiliary DETR Framework (No Query Prompt Tokens)

If you want DETR supervision to improve the shared vision encoder but avoid any
query-token overhead at inference, use:

- `Qwen3VLAuxDetrAdapter` in `modeling.py`
- `train_sharegpt_aux.py` for training
- `eval_one_aux.py` for one-sample debug

This setup:
- keeps the normal Qwen3-VL generation input unchanged
- runs a DETR branch on visual token features during training
- allows dropping DETR at inference (`det_enabled=False`)

Quick start:

```bash
python -m qwen3_vl_det.train_sharegpt_aux \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --train-split train \
  --require-vision \
  --num-queries 100 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --epochs 3 \
  --lr 2e-5 \
  --lm-weight 1.0 \
  --det-weight 0.3 \
  --box-coord-mode norm1000 \
  --box-coord-order yxyx \
  --output-dir qwen3_vl_det/checkpoints_aux_run1
```

```bash
python -m qwen3_vl_det.eval_one_aux \
  --checkpoint-dir qwen3_vl_det/checkpoints_aux_run1/last \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --sample-index 100 \
  --box-coord-mode norm1000 \
  --box-coord-order yxyx \
  --obj-threshold 0.15 \
  --output-image qwen3_vl_det/eval_aux_sample100.png
```

## Dataset Normalization (Recommended)

Your original dataset is VLM-chat style. To avoid repeated regex parsing and
coordinate ambiguity, normalize once to explicit fields:

```bash
python -m qwen3_vl_det.normalize_dataset \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --split train \
  --box-coord-mode norm1000 \
  --box-coord-order yxyx \
  --output-dir qwen3_vl_det/data_normalized/train
```

Then train/evaluate directly from disk:

```bash
python -m qwen3_vl_det.train_sharegpt_aux \
  --dataset-from-disk qwen3_vl_det/data_normalized/train \
  --train-split train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --require-vision \
  --num-queries 100 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --epochs 3 \
  --lr 2e-5 \
  --det-weight 0.3 \
  --output-dir qwen3_vl_det/checkpoints_aux_run1
```

## What This Adds

- `modeling.py`: wrapper that adds:
  - objectness head `Linear(H -> 1)`
  - box head `Linear(H -> 4)` with sigmoid output (`cx, cy, w, h`)
- `hungarian.py`: DETR-style matching + losses:
  - matching cost = class + L1 + GIoU
  - training loss = objectness BCE + matched box L1 + matched box GIoU
- `train_stub.py`: minimal loop showing how to pass:
  - `query_positions` (`[B, Q]`)
  - `gt_boxes` (`list[tensor[Gi,4]]`)
- `train_sharegpt.py`: end-to-end finetune script for ShareGPT-style datasets with `<box> [...] </box>` annotations

## Integration Steps In Your Real Trainer

1. Insert query markers in your prompt (e.g. `<|det_query|>` repeated `Q` times).
2. After tokenization, build `query_positions` from marker token ids.
3. Keep your normal `labels` for LM loss.
4. Pass `query_positions` and `gt_boxes` to `Qwen3VLDetrAdapter.forward(...)`.
5. Optimize `out["loss"]`, which combines LM and DETR-style loss.

## Recommended Initial Hyperparameters

- `num_queries = 32` (increase for dense scenes)
- `lm_weight = 1.0`, `det_weight = 1.0`
- Hungarian cost: `class=1.0`, `bbox=5.0`, `giou=2.0`
- No-object weight: `0.5`
- Count regularization: `count_loss_weight=0.5`
- Objectness prior bias: `obj_bias_init=-2.0`

## Important Notes

- `query_positions` must be exact token positions for query markers in each sample.
- `gt_boxes` must be normalized to `[0,1]` in `cx, cy, w, h`.
- If your Qwen3-VL checkpoint requires a model class different from `AutoModelForCausalLM`,
  keep your existing model loader and pass the already-loaded model to `Qwen3VLDetrAdapter(...)`.

## Finetuning On Your Dataset

For dataset `foye501/VLM-Counting-dataset-qwenvl-sharegpt`, run:

```bash
python -m qwen3_vl_det.train_sharegpt \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --train-split train \
  --require-vision \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --num-queries 32 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max-length 2048 \
  --lm-weight 1.0 \
  --det-weight 0.7 \
  --box-coord-mode auto \
  --box-coord-order auto \
  --no-object-weight 0.5 \
  --count-loss-weight 0.5 \
  --obj-bias-init -2.0 \
  --output-dir qwen3_vl_det/checkpoints
```

Quick smoke run:

```bash
python -m qwen3_vl_det.train_sharegpt \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --require-vision \
  --max-samples 32 \
  --max-steps 20 \
  --num-queries 16 \
  --batch-size 1 \
  --grad-accum-steps 4
```

The script automatically:
- loads ShareGPT `conversations/messages`
- removes `<image>` marker from user text
- parses assistant boxes from `<box> [x1, y1, x2, y2] </box>`
- normalizes boxes to `cx, cy, w, h` in `[0,1]`
- injects `DET_QUERY` tokens and computes `query_positions`

Before training, inspect coordinate scale once:

```bash
python -m qwen3_vl_det.inspect_boxes \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --split train \
  --max-samples 500
```

If it reports `suggested_box_coord_mode=norm1000`, train/eval with:
- `--box-coord-mode norm1000`
If it reports `suggested_box_coord_order=yxyx`, train/eval with:
- `--box-coord-order yxyx`

## One-Sample Evaluation + Plot

After training, test one sample and save overlay image:

```bash
python -m qwen3_vl_det.eval_one \
  --checkpoint-dir qwen3_vl_det/checkpoints_run1/last \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --sample-index 0 \
  --num-queries 32 \
  --require-vision \
  --box-coord-mode auto \
  --box-coord-order auto \
  --obj-threshold 0.5 \
  --output-image qwen3_vl_det/eval_sample0.png
```

Output image:
- green boxes = GT
- red boxes = prediction
- `eval_one` also prints `Pred soft count (sum probs)` to compare against thresholded count.

If you see `vision tensors are disabled in this fallback`, do not trust the result.
Install vision dependencies and force strict mode with `--require-vision`.

To debug annotation interpretation, you can dump GT-only overlays for all mode/order combinations:

```bash
python -m qwen3_vl_det.eval_one \
  --checkpoint-dir qwen3_vl_det/checkpoints_run2_fix/last \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --sample-index 4000 \
  --num-queries 100 \
  --obj-threshold 1.1 \
  --debug-all-gt-parses \
  --debug-prefix qwen3_vl_det/gt_parse_4000
```

To test whether predictions are x/y-transposed, enable:
- `--debug-pred-transpose-check`

This prints normal-vs-swapped detection metrics and writes an extra overlay with swapped predictions.

If GT itself looks transposed, force coordinate order explicitly:
- `--box-coord-order yxyx`

Recommendation: use the same `box_coord_mode/order` in both training and evaluation.

## Split Evaluation (ShareGPT Dataset)

```bash
python -m qwen3_vl_det.eval_split \
  --checkpoint-dir qwen3_vl_det/checkpoints_run2/last \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --split train \
  --start-index 9000 \
  --max-samples 500 \
  --num-queries 32 \
  --box-coord-mode auto \
  --box-coord-order auto \
  --obj-threshold 0.5 \
  --iou-threshold 0.5 \
  --easy-max 5 \
  --medium-max 20 \
  --hard-max 50 \
  --require-vision \
  --output-json qwen3_vl_det/eval_sharegpt_run2.json \
  --save-overlays \
  --overlay-dir qwen3_vl_det/eval_sharegpt_overlays
```

`eval_split` now reports both overall metrics and bucket metrics (`easy`, `medium`, `hard`, `extreme`)
based on GT count ranges configured by the threshold flags above.
It includes both `count_mae` (thresholded hard count) and `count_soft_mae` (sum of objectness probabilities).

## Real Data Check (COCO Subset)

Requires local COCO files (`instances_val2017.json` + `val2017/` images):

```bash
python -m qwen3_vl_det.eval_coco_subset \
  --checkpoint-dir qwen3_vl_det/checkpoints_run2/last \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --coco-ann /data/coco/annotations/instances_val2017.json \
  --coco-img-dir /data/coco/val2017 \
  --category person \
  --max-images 200 \
  --num-queries 32 \
  --obj-threshold 0.5 \
  --iou-threshold 0.5 \
  --require-vision \
  --output-json qwen3_vl_det/eval_coco_person.json \
  --save-overlays \
  --overlay-dir qwen3_vl_det/eval_coco_overlays
```

## Why This Helps Your Paper

- Forces one-to-one instance assignment (reduces duplicate counting).
- Turns count into verifiable set prediction.
- Lets you report VLM-only vs hybrid set-supervised improvements under the same protocol.
