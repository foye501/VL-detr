"""Inspect dataset box coordinate scale to detect format mismatches.

Example:
  python -m qwen3_vl_det.inspect_boxes \
    --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
    --split train \
    --max-samples 500
"""

from __future__ import annotations

import argparse
from collections import Counter

from datasets import load_dataset

from qwen3_vl_det.train_sharegpt import (
    _as_messages,
    _extract_image,
    _extract_user_assistant,
    _infer_box_coord_mode,
    _infer_box_coord_order,
    extract_boxes_raw,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect box coordinate scale in ShareGPT-style dataset.")
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--split", default="train")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--hf-token", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.hf_token if args.hf_token else True
    ds = load_dataset(args.dataset_name, split=args.split, token=token)
    start = max(0, args.start_index)
    end = min(len(ds), start + args.max_samples)

    mode_counter = Counter()
    order_counter = Counter()
    total_boxes = 0
    over_dim_boxes = 0
    max_coords = []

    for i in range(start, end):
        ex = ds[i]
        messages = _as_messages(ex)
        _, assistant = _extract_user_assistant(messages)
        img = _extract_image(ex)
        w, h = img.size
        raw_boxes = extract_boxes_raw(assistant)
        mode = _infer_box_coord_mode(raw_boxes, width=w, height=h)
        order = _infer_box_coord_order(raw_boxes, width=w, height=h, mode=mode)
        mode_counter[mode] += 1
        order_counter[order] += 1
        for b in raw_boxes:
            total_boxes += 1
            m = max(b)
            max_coords.append(m)
            if m > max(w, h):
                over_dim_boxes += 1

    if end - start == 0:
        raise RuntimeError("No samples inspected.")

    print(f"inspected_samples={end-start}")
    print("inferred_mode_counts:", dict(mode_counter))
    print("inferred_order_counts:", dict(order_counter))
    print(f"total_boxes={total_boxes}")
    if total_boxes > 0:
        print(f"boxes_with_coord_gt_image_dim={over_dim_boxes} ({over_dim_boxes/total_boxes:.2%})")
        max_coords_sorted = sorted(max_coords)
        p50 = max_coords_sorted[len(max_coords_sorted) // 2]
        p90 = max_coords_sorted[int(len(max_coords_sorted) * 0.9)]
        p99 = max_coords_sorted[int(len(max_coords_sorted) * 0.99)]
        print(f"max_coord_p50={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}")
    if mode_counter:
        suggested = mode_counter.most_common(1)[0][0]
        print(f"suggested_box_coord_mode={suggested}")
    if order_counter:
        suggested_order = order_counter.most_common(1)[0][0]
        print(f"suggested_box_coord_order={suggested_order}")


if __name__ == "__main__":
    main()
