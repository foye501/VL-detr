"""Normalize ShareGPT-style counting dataset into explicit detection fields.

Output sample fields:
- image
- user_text
- assistant_text
- boxes_cxcywh_norm
- boxes_xyxy_abs
- image_width
- image_height
- gt_count
- difficulty_bucket
- source_index

This avoids regex parsing at train/eval time and keeps box conventions explicit.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import Dataset, load_dataset

from qwen3_vl_det.train_sharegpt import (
    _extract_image,
    cxcywh_norm_to_xyxy_abs,
    extract_gt_boxes_from_example,
    extract_user_assistant_from_example,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize ShareGPT-style VLM counting dataset.")
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--split", default="train")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--box-coord-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--box-coord-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument("--easy-max", type=int, default=5)
    p.add_argument("--medium-max", type=int, default=20)
    p.add_argument("--hard-max", type=int, default=50)
    p.add_argument("--output-dir", default="qwen3_vl_det/data_normalized")
    p.add_argument("--hf-token", default="")
    return p.parse_args()


def bucket_from_count(count: int, easy_max: int, medium_max: int, hard_max: int) -> str:
    if count <= easy_max:
        return "easy"
    if count <= medium_max:
        return "medium"
    if count <= hard_max:
        return "hard"
    return "extreme"


def main() -> None:
    args = parse_args()
    if not (0 <= args.easy_max < args.medium_max < args.hard_max):
        raise ValueError("Require thresholds: 0 <= easy-max < medium-max < hard-max.")

    token = args.hf_token if args.hf_token else True
    ds = load_dataset(args.dataset_name, split=args.split, token=token)
    start = max(0, int(args.start_index))
    end = len(ds) if args.max_samples <= 0 else min(len(ds), start + int(args.max_samples))
    if end <= start:
        raise ValueError("No samples selected for normalization.")

    rows: list[dict[str, Any]] = []
    for i in range(start, end):
        ex = ds[i]
        img = _extract_image(ex).convert("RGB")
        w, h = img.size
        user_text, assistant_text = extract_user_assistant_from_example(ex)
        boxes_cxcywh = extract_gt_boxes_from_example(
            ex,
            width=w,
            height=h,
            assistant_text=assistant_text,
            coord_mode=args.box_coord_mode,
            coord_order=args.box_coord_order,
        )
        boxes_xyxy = cxcywh_norm_to_xyxy_abs(boxes_cxcywh, width=w, height=h)
        gt_count = int(boxes_cxcywh.shape[0])
        bucket = bucket_from_count(
            gt_count,
            easy_max=args.easy_max,
            medium_max=args.medium_max,
            hard_max=args.hard_max,
        )

        rows.append(
            {
                "image": img,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "boxes_cxcywh_norm": boxes_cxcywh.tolist(),
                "boxes_xyxy_abs": boxes_xyxy.tolist(),
                "image_width": int(w),
                "image_height": int(h),
                "gt_count": gt_count,
                "difficulty_bucket": bucket,
                "source_index": int(i),
            }
        )
        if (i - start + 1) % 500 == 0:
            print(f"processed={i-start+1}/{end-start}")

    out = Dataset.from_list(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out.save_to_disk(args.output_dir)
    print(f"Saved normalized dataset: {args.output_dir}")
    print(f"Samples: {len(out)}")
    print(f"Columns: {out.column_names}")


if __name__ == "__main__":
    main()
