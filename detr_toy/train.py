"""Train a DETR-style toy detector on synthetic data.

Usage:
  python3 detr_toy/train.py --epochs 20 --n-train 500 --n-val 120
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.path.dirname(__file__), ".cache"))

import matplotlib.pyplot as plt
import numpy as np

try:
    from .matcher import canonical_targets, detection_prf, matched_targets, sigmoid
    from .model import ToySetDetector
    from .toy_dataset import ToyDetectionDataset
except ImportError:
    from matcher import canonical_targets, detection_prf, matched_targets, sigmoid
    from model import ToySetDetector
    from toy_dataset import ToyDetectionDataset


@dataclass
class EvalMetrics:
    count_mae: float
    count_acc: float
    precision: float
    recall: float
    f1: float


def evaluate(
    model: ToySetDetector,
    x: np.ndarray,
    targets: list[np.ndarray],
    obj_threshold: float,
    iou_threshold: float,
) -> EvalMetrics:
    obj_logits, boxes = model.predict_raw(x)
    obj_prob = sigmoid(obj_logits)

    abs_errors = []
    exact = []
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(x)):
        gt = targets[i]
        keep = obj_prob[i] >= obj_threshold
        pred = boxes[i][keep]

        abs_errors.append(abs(len(pred) - len(gt)))
        exact.append(1.0 if len(pred) == len(gt) else 0.0)
        p, r, f = detection_prf(pred, gt, iou_threshold=iou_threshold)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return EvalMetrics(
        count_mae=float(np.mean(abs_errors)),
        count_acc=float(np.mean(exact)),
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
    )


def draw_box(ax, box_xywh: np.ndarray, image_size: int, color: str, linewidth: float = 1.6) -> None:
    cx, cy, w, h = box_xywh
    x1 = (cx - w / 2.0) * image_size
    y1 = (cy - h / 2.0) * image_size
    ww = w * image_size
    hh = h * image_size
    rect = plt.Rectangle((x1, y1), ww, hh, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)


def save_qualitative_examples(
    model: ToySetDetector,
    images: np.ndarray,
    x: np.ndarray,
    targets: list[np.ndarray],
    out_dir: str,
    obj_threshold: float,
    max_examples: int = 6,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    obj_logits, boxes = model.predict_raw(x)
    obj_prob = sigmoid(obj_logits)
    n = min(max_examples, len(images))

    for i in range(n):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(images[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        image_size = images[i].shape[0]

        for b in targets[i]:
            draw_box(ax, b, image_size=image_size, color="lime", linewidth=1.8)

        keep = obj_prob[i] >= obj_threshold
        for b in boxes[i][keep]:
            draw_box(ax, b, image_size=image_size, color="red", linewidth=1.2)

        ax.set_title(
            f"GT={len(targets[i])}, Pred={int(np.sum(keep))}",
            fontsize=10,
        )
        out_path = os.path.join(out_dir, f"pred_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


def build_initial_targets(targets: list[np.ndarray], num_queries: int) -> np.ndarray:
    y = np.zeros((len(targets), num_queries, 5), dtype=np.float32)
    for i, gt in enumerate(targets):
        y[i] = canonical_targets(gt, num_queries=num_queries)
    return y


def build_hungarian_targets(
    obj_logits: np.ndarray,
    boxes: np.ndarray,
    targets: list[np.ndarray],
    num_queries: int,
    class_cost: float,
    bbox_cost: float,
) -> np.ndarray:
    y = np.zeros((len(targets), num_queries, 5), dtype=np.float32)
    for i, gt in enumerate(targets):
        y[i] = matched_targets(
            pred_obj_logits=obj_logits[i],
            pred_boxes=boxes[i],
            gt_boxes=gt,
            num_queries=num_queries,
            class_cost=class_cost,
            bbox_cost=bbox_cost,
        )
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a toy DETR-style set detector.")
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--match-epochs", type=int, default=10)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--min-objects", type=int, default=1)
    p.add_argument("--max-objects", type=int, default=2)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--class-cost", type=float, default=0.2)
    p.add_argument("--bbox-cost", type=float, default=8.0)
    p.add_argument("--obj-threshold", type=float, default=0.6)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=str, default="detr_toy/artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("Creating toy datasets...")
    train_ds = ToyDetectionDataset(
        n_samples=args.n_train,
        image_size=args.image_size,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed,
    )
    val_ds = ToyDetectionDataset(
        n_samples=args.n_val,
        image_size=args.image_size,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed + 1,
    )
    x_train = train_ds.flattened_images()
    x_val = val_ds.flattened_images()

    model = ToySetDetector(
        num_queries=args.num_queries,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Canonical slot supervision warmup before self-aligned Hungarian steps.
    y_boot = build_initial_targets(train_ds.targets, num_queries=args.num_queries)

    history = []
    print("Training...")
    for epoch in range(1, args.warmup_epochs + 1):
        model.fit_step(x_train, y_boot)
        train_metrics = evaluate(
            model=model,
            x=x_train,
            targets=train_ds.targets,
            obj_threshold=args.obj_threshold,
            iou_threshold=args.iou_threshold,
        )
        val_metrics = evaluate(
            model=model,
            x=x_val,
            targets=val_ds.targets,
            obj_threshold=args.obj_threshold,
            iou_threshold=args.iou_threshold,
        )
        row = {
            "stage": "warmup",
            "epoch": epoch,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(row)
        print(
            f"[Warmup {epoch:02d}/{args.warmup_epochs:02d}] "
            f"Train MAE {train_metrics.count_mae:.3f}, F1 {train_metrics.f1:.3f} | "
            f"Val MAE {val_metrics.count_mae:.3f}, F1 {val_metrics.f1:.3f}"
        )

    for epoch in range(1, args.match_epochs + 1):
        pred_logits, pred_boxes = model.predict_raw(x_train)
        y_train = build_hungarian_targets(
            obj_logits=pred_logits,
            boxes=pred_boxes,
            targets=train_ds.targets,
            num_queries=args.num_queries,
            class_cost=args.class_cost,
            bbox_cost=args.bbox_cost,
        )
        model.fit_step(x_train, y_train)

        train_metrics = evaluate(
            model=model,
            x=x_train,
            targets=train_ds.targets,
            obj_threshold=args.obj_threshold,
            iou_threshold=args.iou_threshold,
        )
        val_metrics = evaluate(
            model=model,
            x=x_val,
            targets=val_ds.targets,
            obj_threshold=args.obj_threshold,
            iou_threshold=args.iou_threshold,
        )
        row = {
            "stage": "hungarian",
            "epoch": epoch,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(row)
        print(
            f"[Hungarian {epoch:02d}/{args.match_epochs:02d}] "
            f"Train MAE {train_metrics.count_mae:.3f}, F1 {train_metrics.f1:.3f} | "
            f"Val MAE {val_metrics.count_mae:.3f}, F1 {val_metrics.f1:.3f}"
        )

    model_path = os.path.join(args.out_dir, "toy_set_detector.pkl")
    hist_path = os.path.join(args.out_dir, "history.json")
    cfg_path = os.path.join(args.out_dir, "config.json")
    model.save(model_path)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    save_qualitative_examples(
        model=model,
        images=val_ds.images,
        x=x_val,
        targets=val_ds.targets,
        out_dir=os.path.join(args.out_dir, "qualitative"),
        obj_threshold=args.obj_threshold,
        max_examples=6,
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {hist_path}")
    print(f"Saved qualitative samples: {os.path.join(args.out_dir, 'qualitative')}")


if __name__ == "__main__":
    main()
