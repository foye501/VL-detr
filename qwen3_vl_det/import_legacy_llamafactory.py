"""Import legacy LLaMA-Factory synthetic counting JSON into current HF save_to_disk format.

Expected legacy item format:
{
  "messages": [
    {"role": "user", "content": "<image>Count the number of squares. Please count the objects."},
    {"role": "assistant", "content": "Total count: 3"}
  ],
  "images": ["data/images/abc.png"],
  "gt_boxes": [[y1, x1, y2, x2], ...]
}

This script produces a dataset consumable by train/eval scripts in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from datasets import Dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import legacy LLaMA-Factory synthetic counting JSON.")
    p.add_argument("--input-json", required=True, help="Path to legacy vlm_counting_train.json")
    p.add_argument(
        "--images-root",
        default="",
        help="Optional root to resolve relative image paths. Defaults to the input-json directory.",
    )
    p.add_argument("--output-dir", required=True, help="Output save_to_disk directory")
    p.add_argument("--easy-max", type=int, default=5)
    p.add_argument("--medium-max", type=int, default=20)
    p.add_argument("--hard-max", type=int, default=50)
    return p.parse_args()


def _bucket_from_count(count: int, easy_max: int, medium_max: int, hard_max: int) -> str:
    if count <= easy_max:
        return "easy"
    if count <= medium_max:
        return "medium"
    if count <= hard_max:
        return "hard"
    return "extreme"


def _strip_image_tokens(text: str) -> str:
    return str(text).replace("<image>", "").replace("<|image_pad|>", "").strip()


def _extract_count(text: str) -> int | None:
    m = re.search(r"total\s*count\s*[:=]\s*(-?\d+)", str(text), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(?<![\d.])-?\d+(?![\d.])", str(text))
    if len(nums) == 1:
        return int(nums[0])
    return None


def _resolve_image_path(raw_path: str, images_root: str, json_dir: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    if images_root:
        cand = os.path.join(images_root, raw_path)
        if os.path.exists(cand):
            return cand
    cand = os.path.join(json_dir, raw_path)
    if os.path.exists(cand):
        return cand
    return raw_path


def _yxyx_1000_to_cxcywh_norm(boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    out: list[list[float]] = []
    for box in boxes:
        if len(box) != 4:
            continue
        y1, x1, y2, x2 = [float(v) for v in box]
        x1a = (x1 / 1000.0) * width
        y1a = (y1 / 1000.0) * height
        x2a = (x2 / 1000.0) * width
        y2a = (y2 / 1000.0) * height
        w = max(0.0, x2a - x1a)
        h = max(0.0, y2a - y1a)
        if w <= 0 or h <= 0:
            continue
        cx = ((x1a + x2a) / 2.0) / float(width)
        cy = ((y1a + y2a) / 2.0) / float(height)
        out.append([cx, cy, w / float(width), h / float(height)])
    return out


def _infer_id(item: dict[str, Any], source_index: int, image_path: str) -> str:
    if "id" in item and str(item["id"]).strip():
        return str(item["id"])
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return stem or f"legacy_{source_index:06d}"


def main() -> None:
    args = parse_args()
    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("Legacy input JSON must be a list of items.")

    json_dir = os.path.dirname(os.path.abspath(args.input_json))
    images_root = args.images_root.strip() or json_dir

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        messages = item.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Item {idx} does not contain a 2-turn messages list.")
        user_text = _strip_image_tokens(messages[0].get("content", ""))
        assistant_text = str(messages[1].get("content", "")).strip()
        image_list = item.get("images", [])
        if not image_list:
            raise ValueError(f"Item {idx} has no images field.")
        image_path = _resolve_image_path(str(image_list[0]), images_root=images_root, json_dir=json_dir)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image for item {idx} not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        gt_boxes = item.get("gt_boxes", []) or []
        gt_boxes = [[float(v) for v in box] for box in gt_boxes if isinstance(box, list) and len(box) == 4]
        gt_count = _extract_count(assistant_text)
        if gt_count is None:
            gt_count = len(gt_boxes)

        difficulty_bucket = _bucket_from_count(
            gt_count,
            easy_max=args.easy_max,
            medium_max=args.medium_max,
            hard_max=args.hard_max,
        )
        image_id = _infer_id(item, idx, image_path)
        target_cxcywh = _yxyx_1000_to_cxcywh_norm(gt_boxes, width=w, height=h)

        rows.append(
            {
                "id": image_id,
                "image": img,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "messages": [
                    {"role": "user", "content": "<image>" + user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "image_width": int(w),
                "image_height": int(h),
                "difficulty_bucket": difficulty_bucket,
                "gt_count": int(gt_count),
                "gt_count_all": int(gt_count),
                "num_distractors": 0,
                "has_overlap": None,
                "target_shape": None,
                "target_color": None,
                "target_boxes_yxyx_1000": gt_boxes,
                "distractor_boxes_yxyx_1000": [],
                "all_boxes_yxyx_1000": gt_boxes,
                "target_boxes_cxcywh_norm": target_cxcywh,
                "distractor_boxes_cxcywh_norm": [],
                "all_boxes_cxcywh_norm": target_cxcywh,
                "boxes_yxyx_1000": gt_boxes,
                "boxes_cxcywh_norm": target_cxcywh,
                "gt_boxes": gt_boxes,
                "source_index": idx,
                "image_path": image_path,
            }
        )
        if (idx + 1) % 500 == 0:
            print(f"processed={idx+1}/{len(items)}")

    os.makedirs(args.output_dir, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.output_dir)
    print(f"Saved imported dataset: {args.output_dir}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {ds.column_names}")


if __name__ == "__main__":
    main()
