"""Toy dataset for DETR-style set prediction experiments.

The dataset is synthetic and generated on the fly:
- grayscale images
- 1..max_objects randomly placed shapes per image
- normalized [cx, cy, w, h] boxes in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToySample:
    image: np.ndarray  # [H, W], float32 in [0, 1]
    boxes: np.ndarray  # [N, 4], float32 normalized xywh


def _draw_rectangle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, intensity: float) -> None:
    img[y1:y2, x1:x2] = np.maximum(img[y1:y2, x1:x2], intensity)


def _draw_circle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, intensity: float) -> None:
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ry = max(h / 2.0, 1e-6)
    rx = max(w / 2.0, 1e-6)
    mask = ((yy - cy) ** 2) / (ry**2) + ((xx - cx) ** 2) / (rx**2) <= 1.0
    patch = img[y1:y2, x1:x2]
    patch[mask] = np.maximum(patch[mask], intensity)


def _random_box(
    rng: np.random.Generator,
    image_size: int,
    min_frac: float = 0.14,
    max_frac: float = 0.35,
) -> tuple[int, int, int, int]:
    min_side = max(3, int(image_size * min_frac))
    max_side = max(min_side + 1, int(image_size * max_frac))
    w = int(rng.integers(min_side, max_side + 1))
    h = int(rng.integers(min_side, max_side + 1))
    x1 = int(rng.integers(0, image_size - w + 1))
    y1 = int(rng.integers(0, image_size - h + 1))
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def _box_xyxy_to_xywh_norm(x1: int, y1: int, x2: int, y2: int, image_size: int) -> np.ndarray:
    cx = (x1 + x2) / 2.0 / image_size
    cy = (y1 + y2) / 2.0 / image_size
    w = (x2 - x1) / image_size
    h = (y2 - y1) / image_size
    return np.array([cx, cy, w, h], dtype=np.float32)


def generate_toy_scene(
    rng: np.random.Generator,
    image_size: int = 32,
    min_objects: int = 1,
    max_objects: int = 4,
) -> ToySample:
    """Generate one synthetic detection scene."""
    img = rng.uniform(0.0, 0.06, size=(image_size, image_size)).astype(np.float32)
    n_obj = int(rng.integers(min_objects, max_objects + 1))
    boxes = []

    for _ in range(n_obj):
        x1, y1, x2, y2 = _random_box(rng, image_size)
        intensity = float(rng.uniform(0.50, 1.0))
        if rng.random() < 0.5:
            _draw_rectangle(img, x1, y1, x2, y2, intensity)
        else:
            _draw_circle(img, x1, y1, x2, y2, intensity)
        boxes.append(_box_xyxy_to_xywh_norm(x1, y1, x2, y2, image_size))

    # Add light noise to avoid overfitting to crisp edges.
    img = np.clip(img + rng.normal(0.0, 0.015, size=img.shape).astype(np.float32), 0.0, 1.0)
    if not boxes:
        arr = np.zeros((0, 4), dtype=np.float32)
    else:
        arr = np.stack(boxes, axis=0).astype(np.float32)
    return ToySample(image=img, boxes=arr)


class ToyDetectionDataset:
    """In-memory synthetic dataset."""

    def __init__(
        self,
        n_samples: int,
        image_size: int = 32,
        min_objects: int = 1,
        max_objects: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.seed = seed

        rng = np.random.default_rng(seed)
        images = []
        targets = []
        for _ in range(n_samples):
            sample = generate_toy_scene(
                rng=rng,
                image_size=image_size,
                min_objects=min_objects,
                max_objects=max_objects,
            )
            images.append(sample.image)
            targets.append(sample.boxes)
        self.images = np.stack(images, axis=0).astype(np.float32)
        self.targets = targets

    def flattened_images(self) -> np.ndarray:
        """Return [N, H*W] float32 image features."""
        return self.images.reshape(self.n_samples, -1).astype(np.float32)
