"""Hungarian matching and DETR-style box/objectness losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass
class HungarianLossConfig:
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    no_object_weight: float = 0.1
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    obj_loss_weight: float = 1.0
    count_loss_weight: float = 0.5


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0.0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    lt_inter = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0.0)
    inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
    union = area1[:, None] + area2 - inter

    return iou - (area_c - union) / (area_c + 1e-8)


def hungarian_match_single(
    pred_obj_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    cfg: HungarianLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match one sample: returns matched prediction/gt indices."""
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_boxes.device)
        return empty, empty

    with torch.no_grad():
        pred_prob = pred_obj_logits.sigmoid()  # [Q]
        class_cost = (1.0 - pred_prob)[None, :].expand(num_gt, -1)  # [G, Q]
        bbox_l1 = torch.cdist(gt_boxes, pred_boxes, p=1)  # [G, Q]

        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        giou = generalized_iou(gt_xyxy, pred_xyxy)  # [G, Q]
        giou_cost = 1.0 - giou

        total_cost = cfg.class_cost * class_cost + cfg.bbox_cost * bbox_l1 + cfg.giou_cost * giou_cost
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())

    gt_idx = torch.as_tensor(row_ind, dtype=torch.long, device=pred_boxes.device)
    pred_idx = torch.as_tensor(col_ind, dtype=torch.long, device=pred_boxes.device)
    return pred_idx, gt_idx


def detr_hungarian_loss(
    pred_obj_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: list[torch.Tensor],
    cfg: Optional[HungarianLossConfig] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DETR-style loss for a batch.

    Args:
      pred_obj_logits: [B, Q] logits for objectness.
      pred_boxes: [B, Q, 4] normalized cxcywh in [0,1].
      gt_boxes: length-B list, each tensor [Gi, 4] normalized cxcywh.
    """
    if cfg is None:
        cfg = HungarianLossConfig()

    bsz, num_queries, _ = pred_boxes.shape
    device = pred_boxes.device

    obj_targets = torch.zeros((bsz, num_queries), device=device, dtype=pred_obj_logits.dtype)
    all_box_l1 = []
    all_box_giou = []
    matched_count = 0

    for b in range(bsz):
        gt = gt_boxes[b].to(device=device, dtype=pred_boxes.dtype)
        if gt.numel() == 0:
            continue
        pred_idx, gt_idx = hungarian_match_single(
            pred_obj_logits=pred_obj_logits[b],
            pred_boxes=pred_boxes[b],
            gt_boxes=gt,
            cfg=cfg,
        )
        if pred_idx.numel() == 0:
            continue
        obj_targets[b, pred_idx] = 1.0
        matched_count += int(pred_idx.numel())

        matched_pred = pred_boxes[b, pred_idx]
        matched_gt = gt[gt_idx]
        all_box_l1.append(F.l1_loss(matched_pred, matched_gt, reduction="none").sum(dim=-1))

        pred_xyxy = box_cxcywh_to_xyxy(matched_pred)
        gt_xyxy = box_cxcywh_to_xyxy(matched_gt)
        diag_giou = generalized_iou(pred_xyxy, gt_xyxy).diag()
        all_box_giou.append(1.0 - diag_giou)

    # Objectness: normalize positives/negatives separately so positive signal
    # is not washed out when using many queries (e.g. Q=100).
    bce = F.binary_cross_entropy_with_logits(pred_obj_logits, obj_targets, reduction="none")
    pos_mask = obj_targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.any():
        obj_pos_loss = bce[pos_mask].mean()
    else:
        obj_pos_loss = torch.zeros((), device=device, dtype=bce.dtype)
    if neg_mask.any():
        obj_neg_loss = bce[neg_mask].mean()
    else:
        obj_neg_loss = torch.zeros((), device=device, dtype=bce.dtype)
    obj_loss = obj_pos_loss + cfg.no_object_weight * obj_neg_loss

    # Penalize global count mismatch to reduce "all slots active" collapse.
    pred_count = pred_obj_logits.sigmoid().sum(dim=1)
    gt_count = torch.tensor(
        [float(gt.shape[0]) for gt in gt_boxes],
        device=device,
        dtype=pred_obj_logits.dtype,
    )
    count_loss = F.l1_loss(
        pred_count / float(num_queries),
        gt_count / float(num_queries),
    )

    if all_box_l1:
        box_l1 = torch.cat(all_box_l1).mean()
        box_giou = torch.cat(all_box_giou).mean()
    else:
        box_l1 = torch.zeros((), device=device)
        box_giou = torch.zeros((), device=device)

    loss = (
        cfg.obj_loss_weight * obj_loss
        + cfg.bbox_loss_weight * box_l1
        + cfg.giou_loss_weight * box_giou
        + cfg.count_loss_weight * count_loss
    )

    stats = {
        "det_total_loss": float(loss.detach().cpu()),
        "det_obj_loss": float(obj_loss.detach().cpu()),
        "det_obj_pos_loss": float(obj_pos_loss.detach().cpu()),
        "det_obj_neg_loss": float(obj_neg_loss.detach().cpu()),
        "det_l1_loss": float(box_l1.detach().cpu()),
        "det_giou_loss": float(box_giou.detach().cpu()),
        "det_count_loss": float(count_loss.detach().cpu()),
        "det_matches": float(matched_count),
        "det_pos_slots": float(pos_mask.sum().detach().cpu()),
        "det_neg_slots": float(neg_mask.sum().detach().cpu()),
        "pred_count_mean": float(pred_count.detach().mean().cpu()),
        "gt_count_mean": float(gt_count.detach().mean().cpu()),
    }
    return loss, stats
