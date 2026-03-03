"""Hungarian matching and evaluation helpers for toy DETR."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes.T
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    out = np.stack([x1, y1, x2, y2], axis=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pairwise_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU matrix for normalized xywh boxes."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    a = xywh_to_xyxy(boxes1)
    b = xywh_to_xyxy(boxes2)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    area_a = np.clip((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 0.0, None)
    area_b = np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-8, None)).astype(np.float32)


def hungarian_assign(
    pred_obj_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match GT boxes to query slots with Hungarian assignment.

    Returns:
      gt_indices, query_indices such that each pair is a match.
    """
    n_gt = len(gt_boxes)
    n_q = len(pred_boxes)
    if n_gt == 0 or n_q == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    obj_prob = sigmoid(pred_obj_logits)  # [Q]
    cls = 1.0 - obj_prob[None, :]  # [GT, Q]
    l1 = np.abs(gt_boxes[:, None, :] - pred_boxes[None, :, :]).sum(axis=-1)  # [GT, Q]
    cost = class_cost * cls + bbox_cost * l1
    gt_idx, q_idx = linear_sum_assignment(cost)
    return gt_idx.astype(np.int64), q_idx.astype(np.int64)


def canonical_targets(gt_boxes: np.ndarray, num_queries: int) -> np.ndarray:
    """Deterministic bootstrap targets for the first optimization step."""
    y = np.zeros((num_queries, 5), dtype=np.float32)
    if len(gt_boxes) == 0:
        return y
    order = np.argsort(gt_boxes[:, 0] + 0.01 * gt_boxes[:, 1])
    gt_sorted = gt_boxes[order]
    n = min(len(gt_sorted), num_queries)
    y[:n, 0] = 1.0
    y[:n, 1:] = gt_sorted[:n]
    return y


def matched_targets(
    pred_obj_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    num_queries: int,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
) -> np.ndarray:
    """Build slot targets using Hungarian matching from current predictions."""
    y = np.zeros((num_queries, 5), dtype=np.float32)
    if len(gt_boxes) == 0:
        return y
    gt_idx, q_idx = hungarian_assign(
        pred_obj_logits=pred_obj_logits,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        class_cost=class_cost,
        bbox_cost=bbox_cost,
    )
    for gi, qi in zip(gt_idx, q_idx):
        y[qi, 0] = 1.0
        y[qi, 1:] = gt_boxes[gi]
    return y


def detection_prf(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Precision / recall / F1 using greedy IoU matching."""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0

    ious = pairwise_iou_xywh(pred_boxes, gt_boxes)
    used_gt = set()
    tp = 0
    for pi in range(len(pred_boxes)):
        best_j = int(np.argmax(ious[pi]))
        best_iou = float(ious[pi, best_j])
        if best_iou >= iou_threshold and best_j not in used_gt:
            used_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)
