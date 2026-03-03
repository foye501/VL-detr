# Qwen3-VL + DETR-Style Hungarian Loss (Adapter)

This folder gives you a practical integration path for your current Qwen3-VL research:
- keep standard LM loss
- add fixed query slots
- supervise slots with one-to-one Hungarian matching on GT boxes

It is designed as a patch layer for your existing training code.

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
- No-object weight: `0.1`

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
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --num-queries 32 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max-length 2048 \
  --lm-weight 1.0 \
  --det-weight 0.7 \
  --output-dir qwen3_vl_det/checkpoints
```

Quick smoke run:

```bash
python -m qwen3_vl_det.train_sharegpt \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
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

## One-Sample Evaluation + Plot

After training, test one sample and save overlay image:

```bash
python -m qwen3_vl_det.eval_one \
  --checkpoint-dir qwen3_vl_det/checkpoints_run1/last \
  --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
  --sample-index 0 \
  --num-queries 32 \
  --obj-threshold 0.5 \
  --output-image qwen3_vl_det/eval_sample0.png
```

Output image:
- green boxes = GT
- red boxes = prediction

## Why This Helps Your Paper

- Forces one-to-one instance assignment (reduces duplicate counting).
- Turns count into verifiable set prediction.
- Lets you report VLM-only vs hybrid set-supervised improvements under the same protocol.
