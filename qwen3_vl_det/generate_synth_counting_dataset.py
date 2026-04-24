"""Generate a synthetic counting dataset with explicit target and distractor boxes.

The saved HF dataset (save_to_disk) includes:
- image (PIL)
- user_text / assistant_text
- target_boxes_* / distractor_boxes_* / all_boxes_*
- gt_count (targets) and gt_count_all (targets + distractors)

This format is directly consumable by train_sharegpt.py / train_sharegpt_aux.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import uuid
from typing import Any

from datasets import Dataset
from PIL import Image, ImageDraw


SHAPES = ["circle", "square", "triangle", "star", "diamond", "pentagon", "hexagon"]
TARGET_COLORS = [
    ("red", (220, 60, 60)),
    ("blue", (70, 120, 220)),
    ("green", (60, 180, 80)),
    ("yellow", (230, 205, 60)),
    ("purple", (145, 95, 195)),
    ("orange", (230, 150, 55)),
]
DISTRACTOR_COLORS = [c for _, c in TARGET_COLORS]
BACKGROUND_COLORS = {
    "white": (245, 245, 245),
    "mint": (205, 232, 210),
    "pink": (232, 205, 223),
    "sky": (210, 228, 240),
    "sand": (232, 224, 205),
}
QUESTION_TEMPLATES = [
    "Count the number of {color} {shape}s.",
    "How many {color} {shape}s are in this image?",
    "Please count all {color} {shape}s.",
    "What is the total number of {color} {shape}s?",
]
QUESTION_TEMPLATES_NO_COLOR = [
    "Count the number of {shape}s.",
    "How many {shape}s are shown?",
]

DIFFICULTY_LEVELS: dict[str, dict[str, Any]] = {
    "easy": {
        "count_range": (1, 5),
        "size_range": (18, 34),
        "max_distractors": 0,
        "allow_overlap": False,
        "backgrounds": ["white", "mint", "pink", "sky", "sand"],
    },
    "medium": {
        "count_range": (6, 20),
        "size_range": (14, 30),
        "max_distractors": 5,
        "allow_overlap": True,
        "backgrounds": ["mint", "pink", "sky", "sand"],
    },
    "hard": {
        "count_range": (21, 50),
        "size_range": (12, 26),
        "max_distractors": 15,
        "allow_overlap": True,
        "backgrounds": ["mint", "pink", "sky", "sand"],
    },
    "extreme": {
        "count_range": (51, 100),
        "size_range": (10, 22),
        "max_distractors": 25,
        "allow_overlap": True,
        "backgrounds": ["mint", "pink", "sky", "sand"],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic counting dataset with target+distractor boxes.")
    p.add_argument("--output-dir", default="qwen3_vl_det/data_synth")
    p.add_argument("--split-name", default="train")
    p.add_argument("--samples-per-level", type=int, default=2500)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--export-images", action="store_true")
    p.add_argument(
        "--distractor-policy",
        choices=["shape_only", "mixed", "language_hard"],
        default="shape_only",
        help=(
            "shape_only preserves the original generator. mixed adds some same-shape or same-color "
            "distractors. language_hard makes most distractors share either the target shape or target color."
        ),
    )
    return p.parse_args()


def random_point(margin: int, size: int, image_size: int) -> tuple[int, int]:
    return (
        random.randint(margin, image_size - margin - size * 2),
        random.randint(margin, image_size - margin - size * 2),
    )


def check_overlap(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape_type: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
    size: int,
) -> tuple[int, int, int, int]:
    bbox = (x, y, x + size * 2, y + size * 2)
    if shape_type == "circle":
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "square":
        draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "triangle":
        points = [(x + size, y), (x, y + size * 2), (x + size * 2, y + size * 2)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "star":
        cx, cy = x + size, y + size
        r_outer = size
        r_inner = max(1.0, size * 0.4)
        points = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = r_outer if i % 2 == 0 else r_inner
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "diamond":
        points = [(x + size, y), (x, y + size), (x + size, y + size * 2), (x + size * 2, y + size)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "pentagon":
        cx, cy = x + size, y + size
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            points.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "hexagon":
        cx, cy = x + size, y + size
        points = []
        for i in range(6):
            angle = i * 2 * math.pi / 6 - math.pi / 2
            points.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    else:
        draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=2)
    return bbox


def xyxy_abs_to_yxyx_1000(boxes: list[tuple[int, int, int, int]], image_size: int) -> list[list[int]]:
    out: list[list[int]] = []
    for x1, y1, x2, y2 in boxes:
        ny1 = max(0, min(1000, int((y1 / image_size) * 1000)))
        nx1 = max(0, min(1000, int((x1 / image_size) * 1000)))
        ny2 = max(0, min(1000, int((y2 / image_size) * 1000)))
        nx2 = max(0, min(1000, int((x2 / image_size) * 1000)))
        out.append([ny1, nx1, ny2, nx2])
    return out


def xyxy_abs_to_cxcywh_norm(boxes: list[tuple[int, int, int, int]], image_size: int) -> list[list[float]]:
    out: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        if w <= 0 or h <= 0:
            continue
        cx = ((float(x1) + float(x2)) / 2.0) / float(image_size)
        cy = ((float(y1) + float(y2)) / 2.0) / float(image_size)
        ww = w / float(image_size)
        hh = h / float(image_size)
        out.append([cx, cy, ww, hh])
    return out


def synth_background(name: str, image_size: int) -> tuple[int, int, int]:
    if name in BACKGROUND_COLORS:
        return BACKGROUND_COLORS[name]
    return (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255),
    )


def build_question(target_shape: str, target_color_name: str, num_distractors: int) -> str:
    if num_distractors > 0:
        tpl = random.choice(QUESTION_TEMPLATES)
        return tpl.format(color=target_color_name, shape=target_shape)
    tpl = random.choice(QUESTION_TEMPLATES_NO_COLOR + QUESTION_TEMPLATES)
    return tpl.format(color=target_color_name, shape=target_shape)


def sample_distractor_spec(
    target_shape: str,
    target_color_rgb: tuple[int, int, int],
    policy: str,
) -> tuple[str, tuple[int, int, int]]:
    other_shapes = [s for s in SHAPES if s != target_shape]
    other_colors = [c for _, c in TARGET_COLORS if c != target_color_rgb]
    if not other_shapes or not other_colors:
        return random.choice(SHAPES), random.choice(DISTRACTOR_COLORS)

    if policy == "shape_only":
        return random.choice(other_shapes), random.choice(DISTRACTOR_COLORS)

    if policy == "language_hard":
        r = random.random()
        if r < 0.45:
            return target_shape, random.choice(other_colors)
        if r < 0.90:
            return random.choice(other_shapes), target_color_rgb
        return random.choice(other_shapes), random.choice(other_colors)

    r = random.random()
    if r < 0.33:
        return target_shape, random.choice(other_colors)
    if r < 0.66:
        return random.choice(other_shapes), target_color_rgb
    return random.choice(other_shapes), random.choice(other_colors)


def generate_one(
    difficulty: str,
    sample_index: int,
    image_size: int,
    export_images: bool,
    image_dir: str,
    distractor_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = DIFFICULTY_LEVELS[difficulty]
    target_count = random.randint(*params["count_range"])
    target_shape = random.choice(SHAPES)
    target_color_name, target_color_rgb = random.choice(TARGET_COLORS)
    bg_name = random.choice(params["backgrounds"])
    bg_color = synth_background(bg_name, image_size=image_size)

    img = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(img)

    num_distractors = 0
    distractor_specs: list[tuple[str, tuple[int, int, int]]] = []
    if params["max_distractors"] > 0:
        num_distractors = random.randint(0, min(params["max_distractors"], target_count * 2))
        for _ in range(num_distractors):
            d_shape, d_color = sample_distractor_spec(
                target_shape=target_shape,
                target_color_rgb=target_color_rgb,
                policy=distractor_policy,
            )
            distractor_specs.append((d_shape, d_color))

    planned = [(target_shape, target_color_rgb, True)] * target_count
    planned += [(s, c, False) for s, c in distractor_specs]
    random.shuffle(planned)

    size_min, size_max = params["size_range"]
    placed_boxes: list[tuple[int, int, int, int]] = []
    target_boxes_abs: list[tuple[int, int, int, int]] = []
    distractor_boxes_abs: list[tuple[int, int, int, int]] = []
    has_overlap = False

    for shape, color, is_target in planned:
        size = random.randint(size_min, size_max)
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            x, y = random_point(margin=10, size=size, image_size=image_size)
            new_box = (x, y, x + size * 2, y + size * 2)
            overlaps_existing = any(check_overlap(new_box, b) for b in placed_boxes)
            conflict = False
            if not params["allow_overlap"]:
                conflict = overlaps_existing
            else:
                # Try non-overlap first, then allow overlap in dense scenes.
                if overlaps_existing and attempts <= 50:
                    conflict = True
                else:
                    conflict = False
                    if overlaps_existing:
                        has_overlap = True

            if not conflict:
                placed_box = draw_shape(draw, shape, color, x, y, size)
                placed_boxes.append(placed_box)
                if is_target:
                    target_boxes_abs.append(placed_box)
                else:
                    distractor_boxes_abs.append(placed_box)
                placed = True
            attempts += 1

    all_boxes_abs = target_boxes_abs + distractor_boxes_abs
    actual_targets = len(target_boxes_abs)
    actual_distractors = len(distractor_boxes_abs)

    image_id = f"{difficulty}_{sample_index:05d}_{uuid.uuid4().hex[:6]}"
    image_filename = f"{image_id}.png"
    image_relpath = os.path.join("images", image_filename)
    image_abspath = os.path.join(image_dir, image_filename)
    if export_images:
        os.makedirs(image_dir, exist_ok=True)
        img.save(image_abspath)

    question = build_question(
        target_shape=target_shape,
        target_color_name=target_color_name,
        num_distractors=actual_distractors,
    )
    user_text = f"{question} Please count the objects."
    assistant_text = f"Total count: {actual_targets}"

    target_yxyx_1000 = xyxy_abs_to_yxyx_1000(target_boxes_abs, image_size=image_size)
    distractor_yxyx_1000 = xyxy_abs_to_yxyx_1000(distractor_boxes_abs, image_size=image_size)
    all_yxyx_1000 = xyxy_abs_to_yxyx_1000(all_boxes_abs, image_size=image_size)

    target_cxcywh_norm = xyxy_abs_to_cxcywh_norm(target_boxes_abs, image_size=image_size)
    distractor_cxcywh_norm = xyxy_abs_to_cxcywh_norm(distractor_boxes_abs, image_size=image_size)
    all_cxcywh_norm = xyxy_abs_to_cxcywh_norm(all_boxes_abs, image_size=image_size)

    row = {
        "id": image_id,
        "image": img,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "messages": [
            {"role": "user", "content": "<image>" + user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "target_shape": target_shape,
        "target_color": target_color_name,
        "difficulty_bucket": difficulty,
        "has_overlap": bool(has_overlap),
        "image_width": int(image_size),
        "image_height": int(image_size),
        "gt_count": int(actual_targets),
        "gt_count_all": int(len(all_boxes_abs)),
        "num_distractors": int(actual_distractors),
        # Preferred explicit fields for supervision.
        "target_boxes_yxyx_1000": target_yxyx_1000,
        "distractor_boxes_yxyx_1000": distractor_yxyx_1000,
        "all_boxes_yxyx_1000": all_yxyx_1000,
        "target_boxes_cxcywh_norm": target_cxcywh_norm,
        "distractor_boxes_cxcywh_norm": distractor_cxcywh_norm,
        "all_boxes_cxcywh_norm": all_cxcywh_norm,
        # Backward-compatible aliases.
        "boxes_yxyx_1000": target_yxyx_1000,
        "boxes_cxcywh_norm": target_cxcywh_norm,
        "gt_boxes": target_yxyx_1000,
        "source_index": int(sample_index),
    }
    if export_images:
        row["image_path"] = image_relpath

    eval_row = {
        "id": image_id,
        "image_path": image_relpath if export_images else "",
        "question": question,
        "answer": actual_targets,
        "difficulty": difficulty,
        "target_shape": target_shape,
        "target_color": target_color_name,
        "num_distractors": actual_distractors,
        "has_overlap": bool(has_overlap),
        "target_boxes_yxyx_1000": target_yxyx_1000,
        "distractor_boxes_yxyx_1000": distractor_yxyx_1000,
    }
    return row, eval_row


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")

    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    ordered_levels = ["easy", "medium", "hard", "extreme"]
    sample_index = 0
    for level in ordered_levels:
        print(f"Generating {args.samples_per_level} samples for '{level}'...")
        for _ in range(args.samples_per_level):
            row, eval_row = generate_one(
                difficulty=level,
                sample_index=sample_index,
                image_size=args.image_size,
                export_images=args.export_images,
                image_dir=image_dir,
                distractor_policy=args.distractor_policy,
            )
            rows.append(row)
            eval_rows.append(eval_row)
            sample_index += 1

    split_dir = os.path.join(args.output_dir, args.split_name)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(split_dir)
    print(f"Saved HF dataset to: {split_dir}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    eval_json = os.path.join(args.output_dir, f"{args.split_name}_eval_annotations.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)
    print(f"Saved eval annotations to: {eval_json}")


if __name__ == "__main__":
    main()
