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
from PIL import Image, ImageDraw, ImageFilter


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
    p.add_argument(
        "--zero-count-prob",
        type=float,
        default=0.0,
        help="Probability that a sample asks for a valid target with zero target instances.",
    )
    p.add_argument(
        "--min-distractors",
        type=int,
        default=0,
        help="Minimum distractors per image. Useful for language-hard easy and zero-count samples.",
    )
    p.add_argument(
        "--guarantee-language-distractors",
        action="store_true",
        help="When possible, force both same-shape/wrong-color and same-color/wrong-shape distractors.",
    )
    p.add_argument(
        "--background-style",
        choices=["solid", "gradient", "noisy", "grid", "mixed"],
        default="solid",
        help="Background renderer. mixed samples one style per image.",
    )
    p.add_argument(
        "--visual-style",
        choices=["clean", "varied"],
        default="clean",
        help="varied enables per-object rotation, shadows, outline variation, and optional corruptions.",
    )
    p.add_argument("--rotation-max-deg", type=float, default=0.0)
    p.add_argument("--color-jitter", type=int, default=0)
    p.add_argument("--occlusion-prob", type=float, default=0.0)
    p.add_argument("--noise-prob", type=float, default=0.0)
    p.add_argument("--blur-prob", type=float, default=0.0)
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


def clamp_u8(v: int) -> int:
    return max(0, min(255, int(v)))


def jitter_color(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    if amount <= 0:
        return color
    return tuple(clamp_u8(c + random.randint(-amount, amount)) for c in color)


def rotate_points(
    points: list[tuple[float, float]],
    center: tuple[float, float],
    angle_deg: float,
) -> list[tuple[float, float]]:
    if abs(angle_deg) < 1e-6:
        return points
    cx, cy = center
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    out: list[tuple[float, float]] = []
    for px, py in points:
        dx = px - cx
        dy = py - cy
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def offset_points(points: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in points]


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape_type: str,
    color: tuple[int, int, int],
    x: int,
    y: int,
    size: int,
    rotation_deg: float = 0.0,
    outline_width: int = 2,
    shadow: bool = False,
    canvas_size: int | None = None,
) -> tuple[int, int, int, int]:
    bbox = (x, y, x + size * 2, y + size * 2)
    cx, cy = x + size, y + size
    outline = (0, 0, 0)
    shadow_color = (70, 70, 70)

    def clamp_xy(v: float, upper: int | None) -> int:
        value = int(v)
        if upper is None:
            return max(0, value)
        return max(0, min(upper, value))

    def draw_poly(points: list[tuple[float, float]]) -> tuple[int, int, int, int]:
        rotated = rotate_points(points, center=(cx, cy), angle_deg=rotation_deg)
        if shadow:
            draw.polygon(offset_points(rotated, 3, 3), fill=shadow_color)
        draw.polygon(rotated, fill=color, outline=outline, width=outline_width)
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        return (
            clamp_xy(math.floor(min(xs)), canvas_size),
            clamp_xy(math.floor(min(ys)), canvas_size),
            clamp_xy(math.ceil(max(xs)), canvas_size),
            clamp_xy(math.ceil(max(ys)), canvas_size),
        )

    if shape_type == "circle":
        if shadow:
            draw.ellipse((bbox[0] + 3, bbox[1] + 3, bbox[2] + 3, bbox[3] + 3), fill=shadow_color)
        draw.ellipse(bbox, fill=color, outline=outline, width=outline_width)
    elif shape_type == "square":
        return draw_poly([(x, y), (x + size * 2, y), (x + size * 2, y + size * 2), (x, y + size * 2)])
    elif shape_type == "triangle":
        return draw_poly([(x + size, y), (x, y + size * 2), (x + size * 2, y + size * 2)])
    elif shape_type == "star":
        r_outer = size
        r_inner = max(1.0, size * 0.4)
        points = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = r_outer if i % 2 == 0 else r_inner
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return draw_poly(points)
    elif shape_type == "diamond":
        return draw_poly([(x + size, y), (x, y + size), (x + size, y + size * 2), (x + size * 2, y + size)])
    elif shape_type == "pentagon":
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            points.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
        return draw_poly(points)
    elif shape_type == "hexagon":
        points = []
        for i in range(6):
            angle = i * 2 * math.pi / 6 - math.pi / 2
            points.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
        return draw_poly(points)
    else:
        if shadow:
            draw.rectangle((bbox[0] + 3, bbox[1] + 3, bbox[2] + 3, bbox[3] + 3), fill=shadow_color)
        draw.rectangle(bbox, fill=color, outline=outline, width=outline_width)
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


def make_background(
    name: str,
    image_size: int,
    style: str,
) -> tuple[Image.Image, str]:
    bg_style = random.choice(["solid", "gradient", "noisy", "grid"]) if style == "mixed" else style
    base = synth_background(name, image_size=image_size)
    img = Image.new("RGB", (image_size, image_size), base)
    if bg_style == "solid":
        return img, bg_style

    draw = ImageDraw.Draw(img)
    if bg_style == "gradient":
        other = jitter_color(base, 35)
        for y in range(image_size):
            t = y / max(1, image_size - 1)
            row = tuple(clamp_u8(round(base[i] * (1.0 - t) + other[i] * t)) for i in range(3))
            draw.line([(0, y), (image_size, y)], fill=row)
    elif bg_style == "noisy":
        dots = max(20, int(image_size * image_size * 0.015))
        for _ in range(dots):
            x = random.randrange(image_size)
            y = random.randrange(image_size)
            draw.point((x, y), fill=jitter_color(base, 45))
    elif bg_style == "grid":
        step = random.choice([24, 32, 40, 48])
        line_color = jitter_color(base, 25)
        for p in range(0, image_size, step):
            draw.line([(p, 0), (p, image_size)], fill=line_color)
            draw.line([(0, p), (image_size, p)], fill=line_color)
    return img, bg_style


def apply_occlusion(
    img: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    image_size: int,
    prob: float,
) -> tuple[Image.Image, bool]:
    if prob <= 0.0 or not boxes or random.random() >= prob:
        return img, False
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1, 3)):
        x1, y1, x2, y2 = random.choice(boxes)
        bw = max(4, x2 - x1)
        bh = max(4, y2 - y1)
        occ_w = random.randint(max(3, bw // 5), max(4, bw // 2))
        occ_h = random.randint(max(3, bh // 5), max(4, bh // 2))
        ox1 = max(0, min(image_size - 1, random.randint(x1, max(x1, x2 - occ_w))))
        oy1 = max(0, min(image_size - 1, random.randint(y1, max(y1, y2 - occ_h))))
        ox2 = min(image_size, ox1 + occ_w)
        oy2 = min(image_size, oy1 + occ_h)
        fill = (*random.choice(list(BACKGROUND_COLORS.values())), random.randint(150, 230))
        if random.random() < 0.5:
            draw.rectangle((ox1, oy1, ox2, oy2), fill=fill)
        else:
            draw.ellipse((ox1, oy1, ox2, oy2), fill=fill)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), True


def apply_corruptions(
    img: Image.Image,
    image_size: int,
    noise_prob: float,
    blur_prob: float,
) -> tuple[Image.Image, bool, bool]:
    has_noise = False
    has_blur = False
    if noise_prob > 0.0 and random.random() < noise_prob:
        draw = ImageDraw.Draw(img)
        dots = max(20, int(image_size * image_size * 0.004))
        for _ in range(dots):
            x = random.randrange(image_size)
            y = random.randrange(image_size)
            v = random.randint(0, 255)
            draw.point((x, y), fill=(v, v, v))
        has_noise = True
    if blur_prob > 0.0 and random.random() < blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))
        has_blur = True
    return img, has_noise, has_blur


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
) -> tuple[str, tuple[int, int, int], str]:
    other_shapes = [s for s in SHAPES if s != target_shape]
    other_colors = [c for _, c in TARGET_COLORS if c != target_color_rgb]
    if not other_shapes or not other_colors:
        return random.choice(SHAPES), random.choice(DISTRACTOR_COLORS), "other"

    if policy == "shape_only":
        d_shape = random.choice(other_shapes)
        d_color = random.choice(DISTRACTOR_COLORS)
        relation = "same_color_wrong_shape" if d_color == target_color_rgb else "other"
        return d_shape, d_color, relation

    if policy == "language_hard":
        r = random.random()
        if r < 0.45:
            return target_shape, random.choice(other_colors), "same_shape_wrong_color"
        if r < 0.90:
            return random.choice(other_shapes), target_color_rgb, "same_color_wrong_shape"
        return random.choice(other_shapes), random.choice(other_colors), "other"

    r = random.random()
    if r < 0.33:
        return target_shape, random.choice(other_colors), "same_shape_wrong_color"
    if r < 0.66:
        return random.choice(other_shapes), target_color_rgb, "same_color_wrong_shape"
    return random.choice(other_shapes), random.choice(other_colors), "other"


def generate_one(
    difficulty: str,
    sample_index: int,
    image_size: int,
    export_images: bool,
    image_dir: str,
    distractor_policy: str,
    zero_count_prob: float,
    min_distractors: int,
    guarantee_language_distractors: bool,
    background_style: str,
    visual_style: str,
    rotation_max_deg: float,
    color_jitter: int,
    occlusion_prob: float,
    noise_prob: float,
    blur_prob: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = DIFFICULTY_LEVELS[difficulty]
    target_count = 0 if random.random() < max(0.0, min(1.0, zero_count_prob)) else random.randint(*params["count_range"])
    target_shape = random.choice(SHAPES)
    target_color_name, target_color_rgb = random.choice(TARGET_COLORS)
    bg_name = random.choice(params["backgrounds"])

    img, actual_background_style = make_background(bg_name, image_size=image_size, style=background_style)
    draw = ImageDraw.Draw(img)

    num_distractors = 0
    distractor_specs: list[tuple[str, tuple[int, int, int], str]] = []
    max_distractors = int(params["max_distractors"])
    min_distractors = max(0, int(min_distractors))
    if min_distractors > 0:
        max_distractors = max(max_distractors, min_distractors)
    if target_count > 0:
        max_distractors = min(max_distractors, max(min_distractors, target_count * 2))
    if max_distractors > 0:
        num_distractors = random.randint(min(min_distractors, max_distractors), max_distractors)
        other_shapes = [s for s in SHAPES if s != target_shape]
        other_colors = [c for _, c in TARGET_COLORS if c != target_color_rgb]
        if guarantee_language_distractors and distractor_policy in {"mixed", "language_hard"} and num_distractors > 0:
            if other_colors:
                distractor_specs.append((target_shape, random.choice(other_colors), "same_shape_wrong_color"))
            if len(distractor_specs) < num_distractors and other_shapes:
                distractor_specs.append((random.choice(other_shapes), target_color_rgb, "same_color_wrong_shape"))
        for _ in range(num_distractors):
            if len(distractor_specs) >= num_distractors:
                break
            d_shape, d_color, relation = sample_distractor_spec(
                target_shape=target_shape,
                target_color_rgb=target_color_rgb,
                policy=distractor_policy,
            )
            distractor_specs.append((d_shape, d_color, relation))

    planned = [(target_shape, target_color_rgb, True, "target")] * target_count
    planned += [(s, c, False, relation) for s, c, relation in distractor_specs]
    random.shuffle(planned)

    size_min, size_max = params["size_range"]
    placed_boxes: list[tuple[int, int, int, int]] = []
    target_boxes_abs: list[tuple[int, int, int, int]] = []
    distractor_boxes_abs: list[tuple[int, int, int, int]] = []
    placed_distractor_relations = {
        "same_shape_wrong_color": 0,
        "same_color_wrong_shape": 0,
        "other": 0,
    }
    has_overlap = False

    for shape, color, is_target, relation in planned:
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
                if visual_style == "varied":
                    rotation = random.uniform(-abs(rotation_max_deg), abs(rotation_max_deg))
                    outline_width = random.randint(1, 4)
                    shadow = random.random() < 0.35
                else:
                    rotation = 0.0
                    outline_width = 2
                    shadow = False
                placed_box = draw_shape(
                    draw,
                    shape,
                    jitter_color(color, color_jitter),
                    x,
                    y,
                    size,
                    rotation_deg=rotation,
                    outline_width=outline_width,
                    shadow=shadow,
                    canvas_size=image_size,
                )
                placed_boxes.append(placed_box)
                if is_target:
                    target_boxes_abs.append(placed_box)
                else:
                    distractor_boxes_abs.append(placed_box)
                    if relation in placed_distractor_relations:
                        placed_distractor_relations[relation] += 1
                placed = True
            attempts += 1

    all_boxes_abs = target_boxes_abs + distractor_boxes_abs
    actual_targets = len(target_boxes_abs)
    actual_distractors = len(distractor_boxes_abs)
    img, has_occlusion = apply_occlusion(
        img,
        boxes=all_boxes_abs,
        image_size=image_size,
        prob=occlusion_prob if visual_style == "varied" else 0.0,
    )
    img, has_noise, has_blur = apply_corruptions(
        img,
        image_size=image_size,
        noise_prob=noise_prob if visual_style == "varied" else 0.0,
        blur_prob=blur_prob if visual_style == "varied" else 0.0,
    )

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
        "has_occlusion": bool(has_occlusion),
        "has_noise": bool(has_noise),
        "has_blur": bool(has_blur),
        "is_zero_count": bool(actual_targets == 0),
        "distractor_policy": distractor_policy,
        "distractor_relation_counts": placed_distractor_relations,
        "background_name": bg_name,
        "background_style": actual_background_style,
        "visual_style": visual_style,
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
        "has_occlusion": bool(has_occlusion),
        "has_noise": bool(has_noise),
        "has_blur": bool(has_blur),
        "is_zero_count": bool(actual_targets == 0),
        "distractor_policy": distractor_policy,
        "distractor_relation_counts": placed_distractor_relations,
        "background_name": bg_name,
        "background_style": actual_background_style,
        "visual_style": visual_style,
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
                zero_count_prob=args.zero_count_prob,
                min_distractors=args.min_distractors,
                guarantee_language_distractors=args.guarantee_language_distractors,
                background_style=args.background_style,
                visual_style=args.visual_style,
                rotation_max_deg=args.rotation_max_deg,
                color_jitter=args.color_jitter,
                occlusion_prob=args.occlusion_prob,
                noise_prob=args.noise_prob,
                blur_prob=args.blur_prob,
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
