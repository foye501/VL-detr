"""Evaluate a checkpoint on ShareGPT-style counting data.

Reports:
- count MAE
- count exact accuracy
- detection precision/recall/F1 at IoU threshold

Example:
  python -m qwen3_vl_det.eval_split \
    --checkpoint-dir qwen3_vl_det/checkpoints_run2/last \
    --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
    --split train \
    --start-index 9000 \
    --max-samples 500 \
    --num-queries 32 \
    --obj-threshold 0.5 \
    --require-vision \
    --output-json qwen3_vl_det/eval_sharegpt_run2.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import load_dataset
from PIL import ImageDraw
from transformers import AutoTokenizer

from qwen3_vl_det.modeling import Qwen3VLDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    DET_QUERY_TOKEN,
    _as_messages,
    _extract_image,
    _extract_user_assistant,
    find_query_positions,
    load_processor,
    parse_boxes_from_text,
)


@dataclass
class EvalRow:
    index: int
    gt_count: int
    pred_count: int
    abs_error: int
    precision: float
    recall: float
    f1: float
    bucket: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ShareGPT-style counting dataset.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--split", default="train")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--obj-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--box-coord-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--easy-max", type=int, default=5)
    p.add_argument("--medium-max", type=int, default=20)
    p.add_argument("--hard-max", type=int, default=50)
    p.add_argument("--model-name", default="")
    p.add_argument("--hf-token", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-dir", default="qwen3_vl_det/eval_overlays")
    p.add_argument("--overlay-every", type=int, default=50)
    p.add_argument("--output-json", default="qwen3_vl_det/eval_sharegpt.json")
    return p.parse_args()


def bucket_from_gt_count(gt_count: int, easy_max: int, medium_max: int, hard_max: int) -> str:
    if gt_count <= easy_max:
        return "easy"
    if gt_count <= medium_max:
        return "medium"
    if gt_count <= hard_max:
        return "hard"
    return "extreme"


def aggregate_metrics(rows: list[EvalRow]) -> dict[str, float]:
    if not rows:
        return {
            "count_mae": 0.0,
            "count_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    count_mae = sum(r.abs_error for r in rows) / len(rows)
    count_acc = sum(1.0 for r in rows if r.abs_error == 0) / len(rows)
    precision = sum(r.precision for r in rows) / len(rows)
    recall = sum(r.recall for r in rows) / len(rows)
    f1 = sum(r.f1 for r in rows) / len(rows)
    return {
        "count_mae": count_mae,
        "count_accuracy": count_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cxcywh_to_xyxy_abs(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height - 1)
    return out


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-8)


def detection_prf(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, iou_thr: float) -> tuple[float, float, float]:
    if pred_xyxy.numel() == 0 and gt_xyxy.numel() == 0:
        return 1.0, 1.0, 1.0
    if pred_xyxy.numel() == 0:
        return 0.0, 0.0, 0.0
    if gt_xyxy.numel() == 0:
        return 0.0, 0.0, 0.0

    iou = pairwise_iou_xyxy(pred_xyxy, gt_xyxy)
    used_gt = set()
    tp = 0
    for i in range(pred_xyxy.shape[0]):
        best_j = int(torch.argmax(iou[i]).item())
        best_iou = float(iou[i, best_j].item())
        if best_iou >= iou_thr and best_j not in used_gt:
            used_gt.add(best_j)
            tp += 1
    fp = pred_xyxy.shape[0] - tp
    fn = gt_xyxy.shape[0] - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return float(precision), float(recall), float(f1)


def draw_boxes(img, boxes_xyxy: torch.Tensor, color: str, width: int = 3) -> None:
    draw = ImageDraw.Draw(img)
    for b in boxes_xyxy.tolist():
        draw.rectangle(b, outline=color, width=width)


def build_model_and_processor(args: argparse.Namespace):
    ckpt_dir = args.checkpoint_dir
    train_args_path = os.path.join(ckpt_dir, "train_args.json")
    train_args: dict[str, Any] = {}
    if os.path.exists(train_args_path):
        with open(train_args_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)

    tokenizer_path = ckpt_dir if os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json")) else args.model_name
    if not tokenizer_path:
        raise ValueError("Tokenizer not found in checkpoint; provide --model-name.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN) < 0:
        tokenizer.add_special_tokens({"additional_special_tokens": [DET_QUERY_TOKEN]})

    if args.model_name:
        processor_model = args.model_name
    elif os.path.exists(os.path.join(ckpt_dir, "preprocessor_config.json")):
        processor_model = ckpt_dir
    elif "model_name" in train_args and train_args["model_name"]:
        processor_model = str(train_args["model_name"])
    else:
        processor_model = ckpt_dir
    processor, use_vision = load_processor(processor_model, tokenizer)
    if args.require_vision and not use_vision:
        raise RuntimeError("Vision processor unavailable. Fix env and rerun with --require-vision.")

    model = Qwen3VLDetrAdapter.from_pretrained(
        ckpt_dir,
        num_queries=args.num_queries,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    adapter_path = os.path.join(ckpt_dir, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu")
        model.load_state_dict(state["adapter_state_dict"], strict=False)
    model.to(args.device)
    model.eval()
    return model, tokenizer, processor, use_vision


def main() -> None:
    args = parse_args()
    if not (0 <= args.easy_max < args.medium_max < args.hard_max):
        raise ValueError("Require thresholds: 0 <= easy-max < medium-max < hard-max.")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    if args.save_overlays:
        os.makedirs(args.overlay_dir, exist_ok=True)

    token = args.hf_token if args.hf_token else True
    ds = load_dataset(args.dataset_name, split=args.split, token=token)
    start = max(0, int(args.start_index))
    end = min(len(ds), start + int(args.max_samples))
    print(f"Evaluating samples [{start}, {end}) from {args.dataset_name}/{args.split}")

    model, tokenizer, processor, use_vision = build_model_and_processor(args)
    if not use_vision:
        print("WARNING: running without vision tensors; metrics are not valid for real detection.")

    query_token_id = int(tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))

    rows: list[EvalRow] = []
    for idx in range(start, end):
        ex = ds[idx]
        messages = _as_messages(ex)
        user_text, assistant_text = _extract_user_assistant(messages)
        img = _extract_image(ex).convert("RGB")
        w, h = img.size

        user_text = user_text.replace("<image>", "").strip()
        query_text = " ".join([DET_QUERY_TOKEN] * args.num_queries)
        user_text = f"{user_text}\n{query_text}"
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if use_vision:
            inputs = processor(
                text=[chat_text],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(
                text=[chat_text],
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        query_positions = find_query_positions(inputs["input_ids"], query_token_id, args.num_queries)

        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                query_positions=query_positions,
            )
            obj_prob = out["det_obj_logits"].sigmoid()[0]
            box_pred = out["det_boxes"][0]

        keep = obj_prob >= args.obj_threshold
        pred_boxes = box_pred[keep].detach().cpu()
        pred_xyxy = cxcywh_to_xyxy_abs(pred_boxes, width=w, height=h)

        gt_boxes = parse_boxes_from_text(
            assistant_text,
            width=w,
            height=h,
            coord_mode=args.box_coord_mode,
        )
        gt_xyxy = cxcywh_to_xyxy_abs(gt_boxes, width=w, height=h)

        precision, recall, f1 = detection_prf(pred_xyxy, gt_xyxy, iou_thr=args.iou_threshold)
        gt_count = int(gt_boxes.shape[0])
        pred_count = int(pred_boxes.shape[0])
        row = EvalRow(
            index=idx,
            gt_count=gt_count,
            pred_count=pred_count,
            abs_error=abs(pred_count - gt_count),
            precision=precision,
            recall=recall,
            f1=f1,
            bucket=bucket_from_gt_count(
                gt_count=gt_count,
                easy_max=args.easy_max,
                medium_max=args.medium_max,
                hard_max=args.hard_max,
            ),
        )
        rows.append(row)

        if args.save_overlays and ((idx - start) % max(args.overlay_every, 1) == 0):
            vis = img.copy()
            draw_boxes(vis, gt_xyxy, color="lime", width=3)
            draw_boxes(vis, pred_xyxy, color="red", width=2)
            vis.save(os.path.join(args.overlay_dir, f"sample_{idx}.png"))

        if (idx - start + 1) % 50 == 0:
            recent = rows[-50:]
            mae = sum(r.abs_error for r in recent) / len(recent)
            print(f"processed={idx-start+1}/{end-start}, recent_mae={mae:.3f}")

    if not rows:
        raise RuntimeError("No samples evaluated.")

    metrics = aggregate_metrics(rows)

    bucket_ranges = {
        "easy": f"<= {args.easy_max}",
        "medium": f"{args.easy_max + 1}-{args.medium_max}",
        "hard": f"{args.medium_max + 1}-{args.hard_max}",
        "extreme": f">= {args.hard_max + 1}",
    }
    bucket_order = ("easy", "medium", "hard", "extreme")
    bucket_metrics: dict[str, dict[str, float | int | str]] = {}
    for bucket in bucket_order:
        b_rows = [r for r in rows if r.bucket == bucket]
        b_metrics = aggregate_metrics(b_rows)
        bucket_metrics[bucket] = {
            "range": bucket_ranges[bucket],
            "num_samples": len(b_rows),
            **b_metrics,
        }

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "start_index": start,
        "num_samples": len(rows),
        "obj_threshold": args.obj_threshold,
        "iou_threshold": args.iou_threshold,
        "box_coord_mode": args.box_coord_mode,
        "bucket_thresholds": {
            "easy_max": args.easy_max,
            "medium_max": args.medium_max,
            "hard_max": args.hard_max,
        },
        "vision_enabled": bool(use_vision),
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "rows": [asdict(r) for r in rows],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print("Bucket metrics:")
    for bucket in bucket_order:
        bm = bucket_metrics[bucket]
        print(
            f"  {bucket:<7} range={bm['range']:<8} n={bm['num_samples']:<4} "
            f"mae={bm['count_mae']:.3f} acc={bm['count_accuracy']:.3f} "
            f"p={bm['precision']:.3f} r={bm['recall']:.3f} f1={bm['f1']:.3f}"
        )
    print(f"Saved eval json: {args.output_json}")


if __name__ == "__main__":
    main()
