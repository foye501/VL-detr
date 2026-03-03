"""Evaluate counting+detection on a local COCO subset for one category.

Requires local COCO files:
- annotation json (instances_val2017.json)
- images directory (val2017/)

Example:
  python -m qwen3_vl_det.eval_coco_subset \
    --checkpoint-dir qwen3_vl_det/checkpoints_run2/last \
    --model-name Qwen/Qwen3-VL-2B-Instruct \
    --coco-ann /data/coco/annotations/instances_val2017.json \
    --coco-img-dir /data/coco/val2017 \
    --category person \
    --max-images 200 \
    --num-queries 32 \
    --obj-threshold 0.5 \
    --require-vision \
    --output-json qwen3_vl_det/eval_coco_person.json \
    --save-overlays
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from transformers import AutoTokenizer

from qwen3_vl_det.modeling import Qwen3VLDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    DET_QUERY_TOKEN,
    find_query_positions,
    load_processor,
)


@dataclass
class EvalRow:
    image_id: int
    gt_count: int
    pred_count: int
    abs_error: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Qwen checkpoint on local COCO subset.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--coco-ann", required=True)
    p.add_argument("--coco-img-dir", required=True)
    p.add_argument("--category", required=True, help="COCO category name, e.g. person")
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--obj-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-dir", default="qwen3_vl_det/eval_coco_overlays")
    p.add_argument("--overlay-every", type=int, default=20)
    p.add_argument("--output-json", default="qwen3_vl_det/eval_coco.json")
    return p.parse_args()


def xywh_abs_to_cxcywh_norm(boxes_xywh: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes_xywh.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x, y, w, h = boxes_xywh.unbind(-1)
    cx = (x + 0.5 * w) / width
    cy = (y + 0.5 * h) / height
    ww = w / width
    hh = h / height
    out = torch.stack([cx, cy, ww, hh], dim=-1)
    return out.clamp(0.0, 1.0)


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
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    if args.save_overlays:
        os.makedirs(args.overlay_dir, exist_ok=True)

    coco = COCO(args.coco_ann)
    cat_ids = coco.getCatIds(catNms=[args.category])
    if not cat_ids:
        raise ValueError(f"Category not found in COCO: {args.category}")
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    img_ids = img_ids[: args.max_images]
    print(f"Evaluating COCO category={args.category}, images={len(img_ids)}")

    model, tokenizer, processor, use_vision = build_model_and_processor(args)
    if not use_vision:
        print("WARNING: running without vision tensors; metrics are not valid for real detection.")
    query_token_id = int(tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))

    rows: list[EvalRow] = []
    for n, img_id in enumerate(img_ids, start=1):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.coco_img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        gt_xywh = []
        for ann in anns:
            x, y, ww, hh = ann["bbox"]
            if ww <= 0 or hh <= 0:
                continue
            gt_xywh.append([x, y, ww, hh])
        gt_xywh_t = (
            torch.tensor(gt_xywh, dtype=torch.float32)
            if gt_xywh
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        gt_boxes = xywh_abs_to_cxcywh_norm(gt_xywh_t, width=w, height=h)
        gt_xyxy = cxcywh_to_xyxy_abs(gt_boxes, width=w, height=h)

        prompt = (
            f"Count the number of {args.category} objects in this image. "
            "Please annotate the location of each object, and then state the total count."
        )
        query_text = " ".join([DET_QUERY_TOKEN] * args.num_queries)
        prompt = f"{prompt}\n{query_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages,
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
        precision, recall, f1 = detection_prf(pred_xyxy, gt_xyxy, iou_thr=args.iou_threshold)

        gt_count = int(gt_boxes.shape[0])
        pred_count = int(pred_boxes.shape[0])
        rows.append(
            EvalRow(
                image_id=int(img_id),
                gt_count=gt_count,
                pred_count=pred_count,
                abs_error=abs(pred_count - gt_count),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

        if args.save_overlays and ((n - 1) % max(args.overlay_every, 1) == 0):
            vis = img.copy()
            draw_boxes(vis, gt_xyxy, color="lime", width=3)
            draw_boxes(vis, pred_xyxy, color="red", width=2)
            vis.save(os.path.join(args.overlay_dir, f"coco_{img_id}.png"))

        if n % 50 == 0:
            recent = rows[-50:]
            mae = sum(r.abs_error for r in recent) / len(recent)
            print(f"processed={n}/{len(img_ids)}, recent_mae={mae:.3f}")

    if not rows:
        raise RuntimeError("No COCO samples evaluated.")

    count_mae = sum(r.abs_error for r in rows) / len(rows)
    count_acc = sum(1.0 for r in rows if r.abs_error == 0) / len(rows)
    precision = sum(r.precision for r in rows) / len(rows)
    recall = sum(r.recall for r in rows) / len(rows)
    f1 = sum(r.f1 for r in rows) / len(rows)

    summary = {
        "dataset": "coco",
        "category": args.category,
        "num_samples": len(rows),
        "obj_threshold": args.obj_threshold,
        "iou_threshold": args.iou_threshold,
        "vision_enabled": bool(use_vision),
        "metrics": {
            "count_mae": count_mae,
            "count_accuracy": count_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "rows": [asdict(r) for r in rows],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved eval json: {args.output_json}")


if __name__ == "__main__":
    main()
