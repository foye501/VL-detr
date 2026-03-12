"""Generate one controlled synthetic counting probe image with distractors."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from PIL import Image, ImageDraw

from qwen3_vl_det.generate_synth_counting_dataset import (
    BACKGROUND_COLORS,
    DISTRACTOR_COLORS,
    SHAPES,
    TARGET_COLORS,
    check_overlap,
    draw_shape,
    xyxy_abs_to_cxcywh_norm,
    xyxy_abs_to_yxyx_1000,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one synthetic probe image with target+distractor objects.")
    p.add_argument("--output-image", default="qwen3_vl_det/probe_mix.png")
    p.add_argument("--output-json", default="")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target-shape", choices=SHAPES, default="star")
    p.add_argument("--target-color", default="orange")
    p.add_argument("--target-count", type=int, default=12)
    p.add_argument("--distractor-count", type=int, default=18)
    p.add_argument("--distractor-shapes", default="circle,square,triangle,diamond")
    p.add_argument("--background", choices=sorted(BACKGROUND_COLORS.keys()), default="mint")
    p.add_argument("--size-min", type=int, default=12)
    p.add_argument("--size-max", type=int, default=24)
    p.add_argument("--allow-overlap", action="store_true")
    return p.parse_args()


def _resolve_color(name: str) -> tuple[str, tuple[int, int, int]]:
    lookup = {n.lower(): (n, rgb) for n, rgb in TARGET_COLORS}
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(f"Unknown color '{name}'. Choose one of: {', '.join(sorted(lookup))}")
    return lookup[key]


def _random_point(image_size: int, size: int, margin: int = 10) -> tuple[int, int]:
    return (
        random.randint(margin, image_size - margin - size * 2),
        random.randint(margin, image_size - margin - size * 2),
    )


def _choose_distractor(
    target_shape: str,
    target_color_name: str,
    allowed_shapes: list[str],
) -> tuple[str, tuple[int, int, int], str]:
    shape = random.choice(allowed_shapes)
    color_name, color_rgb = random.choice(TARGET_COLORS)
    if shape == target_shape and color_name == target_color_name:
        candidates = [s for s in allowed_shapes if s != target_shape]
        if candidates:
            shape = random.choice(candidates)
        else:
            candidates = [n for n, _ in TARGET_COLORS if n != target_color_name]
            color_name, color_rgb = _resolve_color(random.choice(candidates))
    return shape, color_rgb, color_name


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_json = args.output_json.strip() or (args.output_image + ".json")
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    target_color_name, target_color_rgb = _resolve_color(args.target_color)
    bg = BACKGROUND_COLORS[args.background]
    img = Image.new("RGB", (args.image_size, args.image_size), bg)
    draw = ImageDraw.Draw(img)

    distractor_shapes = [s.strip() for s in args.distractor_shapes.split(",") if s.strip()]
    distractor_shapes = [s for s in distractor_shapes if s in SHAPES]
    if not distractor_shapes:
        distractor_shapes = [s for s in SHAPES if s != args.target_shape]

    planned: list[dict[str, Any]] = []
    for _ in range(int(args.target_count)):
        planned.append(
            {
                "shape": args.target_shape,
                "color_name": target_color_name,
                "color_rgb": target_color_rgb,
                "is_target": True,
            }
        )
    for _ in range(int(args.distractor_count)):
        shape, color_rgb, color_name = _choose_distractor(
            target_shape=args.target_shape,
            target_color_name=target_color_name,
            allowed_shapes=distractor_shapes,
        )
        planned.append(
            {
                "shape": shape,
                "color_name": color_name,
                "color_rgb": color_rgb,
                "is_target": False,
            }
        )
    random.shuffle(planned)

    placed_boxes: list[tuple[int, int, int, int]] = []
    target_boxes_abs: list[tuple[int, int, int, int]] = []
    distractor_boxes_abs: list[tuple[int, int, int, int]] = []
    placed_objects: list[dict[str, Any]] = []

    for spec in planned:
        size = random.randint(int(args.size_min), int(args.size_max))
        placed = False
        attempts = 0
        while not placed and attempts < 200:
            x, y = _random_point(args.image_size, size)
            new_box = (x, y, x + size * 2, y + size * 2)
            overlaps = any(check_overlap(new_box, b) for b in placed_boxes)
            if overlaps and not args.allow_overlap and attempts < 180:
                attempts += 1
                continue
            placed_box = draw_shape(draw, spec["shape"], spec["color_rgb"], x, y, size)
            placed_boxes.append(placed_box)
            placed_objects.append(
                {
                    "shape": spec["shape"],
                    "color_name": spec["color_name"],
                    "is_target": bool(spec["is_target"]),
                    "xyxy_abs": list(placed_box),
                }
            )
            if spec["is_target"]:
                target_boxes_abs.append(placed_box)
            else:
                distractor_boxes_abs.append(placed_box)
            placed = True
            attempts += 1

    all_boxes_abs = target_boxes_abs + distractor_boxes_abs
    question = f"Count the number of {target_color_name} {args.target_shape}s. Please count the objects."

    img.save(args.output_image)

    payload = {
        "image_path": args.output_image,
        "image_size": args.image_size,
        "seed": args.seed,
        "question": question,
        "target_shape": args.target_shape,
        "target_color": target_color_name,
        "gt_count": len(target_boxes_abs),
        "gt_count_all": len(all_boxes_abs),
        "target_boxes_yxyx_1000": xyxy_abs_to_yxyx_1000(target_boxes_abs, image_size=args.image_size),
        "distractor_boxes_yxyx_1000": xyxy_abs_to_yxyx_1000(distractor_boxes_abs, image_size=args.image_size),
        "all_boxes_yxyx_1000": xyxy_abs_to_yxyx_1000(all_boxes_abs, image_size=args.image_size),
        "target_boxes_cxcywh_norm": xyxy_abs_to_cxcywh_norm(target_boxes_abs, image_size=args.image_size),
        "distractor_boxes_cxcywh_norm": xyxy_abs_to_cxcywh_norm(distractor_boxes_abs, image_size=args.image_size),
        "all_boxes_cxcywh_norm": xyxy_abs_to_cxcywh_norm(all_boxes_abs, image_size=args.image_size),
        "objects": placed_objects,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved image: {args.output_image}")
    print(f"Saved json: {output_json}")
    print(f"Question: {question}")
    print(f"GT target count: {len(target_boxes_abs)}")
    print(f"GT all-object count: {len(all_boxes_abs)}")


if __name__ == "__main__":
    main()
