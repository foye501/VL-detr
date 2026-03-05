"""Train Qwen3-VL with DETR-style Hungarian supervision on ShareGPT-style data.

Expected sample format (one of the supported variants):
- {"image": <PIL or image path>, "conversations": [{"from":"human","value":"<image>..."}, ...]}
- {"images": [<image>], "messages": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import transformers
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from qwen3_vl_det.hungarian import HungarianLossConfig
from qwen3_vl_det.modeling import AdapterLossConfig, Qwen3VLDetrAdapter

def _optional_transformers_class(*names: str):
    for name in names:
        try:
            cls = getattr(transformers, name)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


Qwen3VLProcessor = _optional_transformers_class("Qwen3VLProcessor", "Qwen3_VLProcessor")
Qwen2_5_VLProcessor = _optional_transformers_class("Qwen2_5_VLProcessor")
Qwen2VLProcessor = _optional_transformers_class("Qwen2VLProcessor")

DET_QUERY_TOKEN = "<|det_query|>"
BOX_PATTERN = re.compile(
    r"<box>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</box>",
    flags=re.IGNORECASE,
)
BOX_SUPERVISION_CHOICES = ("target", "all")


@dataclass
class TrainArgs:
    dataset_name: str = "foye501/VLM-Counting-dataset-qwenvl-sharegpt"
    dataset_from_disk: str = ""
    train_split: str = "train"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "qwen3_vl_det/checkpoints"
    num_queries: int = 32
    batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 1
    lr: float = 2e-5
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    no_object_weight: float = 0.5
    count_loss_weight: float = 0.5
    obj_bias_init: float = -2.0
    grad_clip_norm: float = 1.0
    max_length: int = 2048
    max_steps: int = 0
    max_samples: int = 0
    log_every: int = 10
    lm_weight: float = 1.0
    det_weight: float = 0.7
    train_heads_only: bool = False
    require_vision: bool = False
    box_coord_mode: str = "auto"  # auto | absolute | norm1000 | norm01
    box_coord_order: str = "auto"  # auto | xyxy | yxyx
    box_supervision_source: str = "target"  # target | all
    seed: int = 7
    hf_token: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="Qwen3-VL + DETR Hungarian finetuning on ShareGPT-style data.")
    p.add_argument("--dataset-name", default=TrainArgs.dataset_name)
    p.add_argument("--dataset-from-disk", default=TrainArgs.dataset_from_disk)
    p.add_argument("--train-split", default=TrainArgs.train_split)
    p.add_argument("--model-name", default=TrainArgs.model_name)
    p.add_argument("--output-dir", default=TrainArgs.output_dir)
    p.add_argument("--num-queries", type=int, default=TrainArgs.num_queries)
    p.add_argument("--batch-size", type=int, default=TrainArgs.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=TrainArgs.grad_accum_steps)
    p.add_argument("--epochs", type=int, default=TrainArgs.epochs)
    p.add_argument("--lr", type=float, default=TrainArgs.lr)
    p.add_argument("--class-cost", type=float, default=TrainArgs.class_cost)
    p.add_argument("--bbox-cost", type=float, default=TrainArgs.bbox_cost)
    p.add_argument("--giou-cost", type=float, default=TrainArgs.giou_cost)
    p.add_argument("--no-object-weight", type=float, default=TrainArgs.no_object_weight)
    p.add_argument("--count-loss-weight", type=float, default=TrainArgs.count_loss_weight)
    p.add_argument("--obj-bias-init", type=float, default=TrainArgs.obj_bias_init)
    p.add_argument("--grad-clip-norm", type=float, default=TrainArgs.grad_clip_norm)
    p.add_argument("--max-length", type=int, default=TrainArgs.max_length)
    p.add_argument("--max-steps", type=int, default=TrainArgs.max_steps)
    p.add_argument("--max-samples", type=int, default=TrainArgs.max_samples)
    p.add_argument("--log-every", type=int, default=TrainArgs.log_every)
    p.add_argument("--lm-weight", type=float, default=TrainArgs.lm_weight)
    p.add_argument("--det-weight", type=float, default=TrainArgs.det_weight)
    p.add_argument("--train-heads-only", action="store_true")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument(
        "--box-coord-mode",
        choices=["auto", "absolute", "norm1000", "norm01"],
        default=TrainArgs.box_coord_mode,
    )
    p.add_argument(
        "--box-coord-order",
        choices=["auto", "xyxy", "yxyx"],
        default=TrainArgs.box_coord_order,
    )
    p.add_argument(
        "--box-supervision-source",
        choices=list(BOX_SUPERVISION_CHOICES),
        default=TrainArgs.box_supervision_source,
    )
    p.add_argument("--seed", type=int, default=TrainArgs.seed)
    p.add_argument("--hf-token", default=TrainArgs.hf_token)
    p.add_argument("--device", default=TrainArgs.device)
    ns = p.parse_args()
    return TrainArgs(**vars(ns))


def load_split_dataset(
    dataset_name: str,
    split: str,
    token: Any,
    dataset_from_disk: str = "",
):
    """Load either HF hub dataset or local `save_to_disk` dataset."""
    if dataset_from_disk:
        ds = load_from_disk(dataset_from_disk)
        if hasattr(ds, "keys"):  # DatasetDict
            if split not in ds:
                raise KeyError(f"Split '{split}' not found in dataset_from_disk: {dataset_from_disk}")
            return ds[split]
        return ds
    return load_dataset(dataset_name, split=split, token=token)


def _as_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("conversations", "messages"):
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, str):
                return json.loads(value)
            return value
    raise KeyError("Example has neither `conversations` nor `messages`.")


def _extract_user_assistant(messages: list[dict[str, Any]]) -> tuple[str, str]:
    def _to_text(value: Any) -> str:
        if isinstance(value, list):
            # Some schemas store content as typed chunks.
            chunks = []
            for c in value:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        chunks.append(str(c.get("text", "")))
                elif isinstance(c, str):
                    chunks.append(c)
            return "\n".join(chunks)
        return str(value)

    parsed: list[tuple[str, str]] = []
    for m in messages:
        role = (m.get("from") or m.get("role") or "").lower()
        text = _to_text(m.get("value", m.get("content", "")))
        parsed.append((role, text))

    # Prefer assistant turn that actually contains <box> annotations and pair it
    # with the closest preceding user turn to avoid multi-turn misalignment.
    assistant_candidates: list[tuple[int, str, int]] = []
    for i, (role, text) in enumerate(parsed):
        if role in ("gpt", "assistant"):
            num_boxes = len(BOX_PATTERN.findall(text))
            if num_boxes > 0:
                assistant_candidates.append((i, text, num_boxes))

    if assistant_candidates:
        # In multi-turn chats, the image annotation usually corresponds to the
        # latest assistant turn with boxes, not the one with max box count.
        best_i, assistant_text, _ = assistant_candidates[-1]
        user_text = ""
        for j in range(best_i - 1, -1, -1):
            role, text = parsed[j]
            if role in ("human", "user"):
                user_text = text
                break
        if not user_text:
            # Fallback: use latest user turn if no preceding one is found.
            for role, text in reversed(parsed):
                if role in ("human", "user"):
                    user_text = text
                    break
        if user_text:
            return user_text, assistant_text

    # Fallback for datasets without explicit <box> annotations in assistant text.
    user_text = ""
    assistant_text = ""
    for role, text in parsed:
        if role in ("human", "user"):
            user_text = text
        elif role in ("gpt", "assistant"):
            assistant_text = text
    if not user_text or not assistant_text:
        raise ValueError("Could not extract user/assistant turns from sample.")
    return user_text, assistant_text


def _extract_image(example: dict[str, Any]):
    for key in ("image", "images", "img"):
        if key not in example or example[key] is None:
            continue
        val = example[key]
        if isinstance(val, list):
            if not val:
                continue
            return val[0]
        return val
    raise KeyError("No image field found. Expected one of: image/images/img.")


def extract_user_assistant_from_example(example: dict[str, Any]) -> tuple[str, str]:
    """Extract user/assistant text from either normalized or ShareGPT-style sample."""
    user_text = example.get("user_text", "")
    assistant_text = example.get("assistant_text", "")
    if isinstance(user_text, str) and isinstance(assistant_text, str):
        if user_text.strip() and assistant_text.strip():
            return user_text, assistant_text
    messages = _as_messages(example)
    return _extract_user_assistant(messages)


def extract_boxes_raw(text: str) -> list[list[float]]:
    boxes = []
    for m in BOX_PATTERN.finditer(text):
        x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
        boxes.append([x1, y1, x2, y2])
    return boxes


def _infer_box_coord_mode(raw_boxes: list[list[float]], width: int, height: int) -> str:
    if not raw_boxes:
        return "absolute"
    max_coord = max(max(b) for b in raw_boxes)
    max_dim = max(width, height)
    if max_coord <= 1.5:
        return "norm01"
    # Common for Qwen grounding outputs: 0..1000 normalized coordinates.
    if max_coord <= 1005:
        for x1, y1, x2, y2 in raw_boxes:
            if x1 > width * 1.02 or x2 > width * 1.02 or y1 > height * 1.02 or y2 > height * 1.02:
                return "norm1000"
        if max_coord > (max_dim * 1.05):
            return "norm1000"
    return "absolute"


def _convert_raw_box_to_pixels(
    box: list[float],
    width: int,
    height: int,
    mode: str,
    order: str,
) -> tuple[float, float, float, float]:
    if order == "yxyx":
        y1, x1, y2, x2 = box
    else:
        x1, y1, x2, y2 = box
    if mode == "norm1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    elif mode == "norm01":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    # absolute mode keeps values unchanged.
    return x1, y1, x2, y2


def _coord_order_score(
    raw_boxes: list[list[float]],
    width: int,
    height: int,
    mode: str,
    order: str,
) -> tuple[float, int]:
    score = 0.0
    valid = 0
    for raw in raw_boxes:
        x1, y1, x2, y2 = _convert_raw_box_to_pixels(
            raw,
            width=width,
            height=height,
            mode=mode,
            order=order,
        )
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        valid += 1
        in_bounds = (
            0.0 <= x1 <= float(width)
            and 0.0 <= x2 <= float(width)
            and 0.0 <= y1 <= float(height)
            and 0.0 <= y2 <= float(height)
        )
        score += 2.0 if in_bounds else 1.0
        area = (w * h) / max(float(width * height), 1.0)
        if 1e-5 <= area <= 0.8:
            score += 0.5
    return score, valid


def _infer_box_coord_order(raw_boxes: list[list[float]], width: int, height: int, mode: str) -> str:
    if not raw_boxes:
        return "xyxy"
    xy_score, xy_valid = _coord_order_score(raw_boxes, width, height, mode, "xyxy")
    yx_score, yx_valid = _coord_order_score(raw_boxes, width, height, mode, "yxyx")
    if yx_valid > xy_valid:
        return "yxyx"
    if xy_valid > yx_valid:
        return "xyxy"
    return "yxyx" if yx_score > xy_score else "xyxy"


def parse_boxes_from_text(
    text: str,
    width: int,
    height: int,
    coord_mode: str = "auto",
    coord_order: str = "auto",
) -> torch.Tensor:
    raw_boxes = extract_boxes_raw(text)
    mode = _infer_box_coord_mode(raw_boxes, width=width, height=height) if coord_mode == "auto" else coord_mode
    order = (
        _infer_box_coord_order(raw_boxes, width=width, height=height, mode=mode)
        if coord_order == "auto"
        else coord_order
    )
    boxes = []
    for raw in raw_boxes:
        x1, y1, x2, y2 = _convert_raw_box_to_pixels(
            raw,
            width=width,
            height=height,
            mode=mode,
            order=order,
        )
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / float(width)
        cy = ((y1 + y2) / 2.0) / float(height)
        w = (x2 - x1) / float(width)
        h = (y2 - y1) / float(height)
        boxes.append([cx, cy, w, h])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height)
    return out


def _boxes_xyxy_abs_to_cxcywh_norm(boxes_xyxy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    x1 = x1.clamp(0, width)
    x2 = x2.clamp(0, width)
    y1 = y1.clamp(0, height)
    y2 = y2.clamp(0, height)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    valid = (w > 0) & (h > 0)
    if valid.sum() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    x1 = x1[valid]
    x2 = x2[valid]
    y1 = y1[valid]
    y2 = y2[valid]
    cx = ((x1 + x2) / 2.0) / float(width)
    cy = ((y1 + y2) / 2.0) / float(height)
    ww = (x2 - x1) / float(width)
    hh = (y2 - y1) / float(height)
    out = torch.stack([cx, cy, ww, hh], dim=-1)
    return out.clamp(0, 1)


def _as_float_box_tensor(value: Any) -> torch.Tensor:
    if value is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    if t.dim() == 1:
        if int(t.shape[0]) != 4:
            raise ValueError(f"Expected box vector of length 4, got shape {tuple(t.shape)}")
        t = t.unsqueeze(0)
    if t.dim() != 2 or int(t.shape[1]) != 4:
        raise ValueError(f"Expected box tensor shape [N,4], got {tuple(t.shape)}")
    return t


def _sanitize_cxcywh_norm(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = boxes.clamp(0, 1)
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    return boxes[valid]


def _raw_boxes_to_cxcywh_norm(
    raw_boxes: list[list[float]],
    width: int,
    height: int,
    coord_mode: str = "auto",
    coord_order: str = "auto",
) -> torch.Tensor:
    mode = _infer_box_coord_mode(raw_boxes, width=width, height=height) if coord_mode == "auto" else coord_mode
    order = (
        _infer_box_coord_order(raw_boxes, width=width, height=height, mode=mode)
        if coord_order == "auto"
        else coord_order
    )
    boxes = []
    for raw in raw_boxes:
        x1, y1, x2, y2 = _convert_raw_box_to_pixels(
            raw,
            width=width,
            height=height,
            mode=mode,
            order=order,
        )
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / float(width)
        cy = ((y1 + y2) / 2.0) / float(height)
        w = (x2 - x1) / float(width)
        h = (y2 - y1) / float(height)
        boxes.append([cx, cy, w, h])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return _sanitize_cxcywh_norm(torch.tensor(boxes, dtype=torch.float32))


def _extract_boxes_by_field(
    example: dict[str, Any],
    field_name: str,
    width: int,
    height: int,
) -> Optional[torch.Tensor]:
    if field_name not in example or example[field_name] is None:
        return None
    try:
        t = _as_float_box_tensor(example[field_name])
    except Exception:
        return None
    if t.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    if field_name.endswith("_cxcywh_norm"):
        return _sanitize_cxcywh_norm(t)
    if field_name.endswith("_xyxy_abs"):
        return _boxes_xyxy_abs_to_cxcywh_norm(t, width=width, height=height)
    if field_name.endswith("_xyxy_1000"):
        t = t.clone()
        t[:, [0, 2]] = t[:, [0, 2]] / 1000.0 * float(width)
        t[:, [1, 3]] = t[:, [1, 3]] / 1000.0 * float(height)
        return _boxes_xyxy_abs_to_cxcywh_norm(t, width=width, height=height)
    if field_name.endswith("_yxyx_1000"):
        y1 = t[:, 0] / 1000.0 * float(height)
        x1 = t[:, 1] / 1000.0 * float(width)
        y2 = t[:, 2] / 1000.0 * float(height)
        x2 = t[:, 3] / 1000.0 * float(width)
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        return _boxes_xyxy_abs_to_cxcywh_norm(xyxy, width=width, height=height)
    # Generic raw format fallback (e.g., gt_boxes / all_boxes / target_boxes / distractor_boxes).
    return _raw_boxes_to_cxcywh_norm(
        t.tolist(),
        width=width,
        height=height,
        coord_mode="auto",
        coord_order="auto",
    )


def _extract_from_candidates(
    example: dict[str, Any],
    width: int,
    height: int,
    field_names: list[str],
) -> Optional[torch.Tensor]:
    for name in field_names:
        boxes = _extract_boxes_by_field(example, name, width=width, height=height)
        if boxes is not None:
            return boxes
    return None


def extract_gt_boxes_from_example(
    example: dict[str, Any],
    width: int,
    height: int,
    assistant_text: str,
    coord_mode: str = "auto",
    coord_order: str = "auto",
    box_supervision_source: str = "target",
) -> torch.Tensor:
    """Extract GT boxes with support for normalized dataset fields and ShareGPT text boxes."""
    source = str(box_supervision_source).strip().lower()
    if source not in BOX_SUPERVISION_CHOICES:
        raise ValueError(
            f"Unsupported box_supervision_source={box_supervision_source!r}. "
            f"Choose one of: {BOX_SUPERVISION_CHOICES}"
        )

    target_fields = [
        "target_boxes_cxcywh_norm",
        "boxes_cxcywh_norm",
        "target_boxes_xyxy_abs",
        "boxes_xyxy_abs",
        "target_boxes_xyxy_1000",
        "boxes_xyxy_1000",
        "target_boxes_yxyx_1000",
        "boxes_yxyx_1000",
        "target_boxes",
        "gt_boxes",
    ]
    distractor_fields = [
        "distractor_boxes_cxcywh_norm",
        "distractor_boxes_xyxy_abs",
        "distractor_boxes_xyxy_1000",
        "distractor_boxes_yxyx_1000",
        "distractor_boxes",
    ]
    all_fields = [
        "all_boxes_cxcywh_norm",
        "all_boxes_xyxy_abs",
        "all_boxes_xyxy_1000",
        "all_boxes_yxyx_1000",
        "all_boxes",
    ]

    if source == "all":
        all_boxes = _extract_from_candidates(
            example,
            width=width,
            height=height,
            field_names=all_fields,
        )
        if all_boxes is not None:
            return all_boxes

        target_boxes = _extract_from_candidates(
            example,
            width=width,
            height=height,
            field_names=target_fields,
        )
        distractor_boxes = _extract_from_candidates(
            example,
            width=width,
            height=height,
            field_names=distractor_fields,
        )
        if target_boxes is not None and distractor_boxes is not None:
            if target_boxes.numel() == 0:
                return distractor_boxes
            if distractor_boxes.numel() == 0:
                return target_boxes
            return torch.cat([target_boxes, distractor_boxes], dim=0)

        if target_boxes is not None:
            return target_boxes

    target_boxes = _extract_from_candidates(
        example,
        width=width,
        height=height,
        field_names=target_fields,
    )
    if target_boxes is not None:
        return target_boxes

    if source == "target":
        all_boxes = _extract_from_candidates(
            example,
            width=width,
            height=height,
            field_names=all_fields,
        )
        if all_boxes is not None:
            return all_boxes

    # Fallback: parse from assistant text.
    return parse_boxes_from_text(
        assistant_text,
        width=width,
        height=height,
        coord_mode=coord_mode,
        coord_order=coord_order,
    )


def find_query_positions(input_ids: torch.Tensor, query_token_id: int, num_queries: int) -> torch.Tensor:
    bsz = input_ids.shape[0]
    out = torch.full((bsz, num_queries), -1, dtype=torch.long, device=input_ids.device)
    for b in range(bsz):
        pos = torch.where(input_ids[b] == query_token_id)[0]
        if pos.numel() < num_queries:
            raise ValueError(
                f"Sample {b} has {pos.numel()} query tokens; expected {num_queries}. "
                f"Increase max_length or check template."
            )
        out[b] = pos[:num_queries]
    return out


class ShareGptCollator:
    def __init__(
        self,
        processor,
        num_queries: int,
        max_length: int,
        use_vision: bool,
        box_coord_mode: str,
        box_coord_order: str,
        box_supervision_source: str,
    ) -> None:
        self.processor = processor
        self.num_queries = num_queries
        self.max_length = max_length
        self.use_vision = use_vision
        self.box_coord_mode = box_coord_mode
        self.box_coord_order = box_coord_order
        self.box_supervision_source = box_supervision_source
        self.query_token_id = int(processor.tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))
        if self.query_token_id < 0:
            raise ValueError(f"{DET_QUERY_TOKEN} token id not found in tokenizer.")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        gt_boxes: list[torch.Tensor] = []

        for ex in batch:
            user_text, assistant_text = extract_user_assistant_from_example(ex)
            img = _extract_image(ex)
            width, height = img.size

            user_text = user_text.replace("<image>", "").strip()
            query_text = " ".join([DET_QUERY_TOKEN] * self.num_queries)
            user_text = f"{user_text}\n{query_text}"

            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]

            chat_text = self.processor.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(chat_text)
            images.append(img)
            gt_boxes.append(
                extract_gt_boxes_from_example(
                    ex,
                    width=width,
                    height=height,
                    assistant_text=assistant_text,
                    coord_mode=self.box_coord_mode,
                    coord_order=self.box_coord_order,
                    box_supervision_source=self.box_supervision_source,
                )
            )

        if self.use_vision:
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        labels[input_ids == self.query_token_id] = -100

        query_positions = find_query_positions(
            input_ids=input_ids,
            query_token_id=self.query_token_id,
            num_queries=self.num_queries,
        )

        out = dict(inputs)
        out["labels"] = labels
        out["query_positions"] = query_positions
        out["gt_boxes"] = gt_boxes
        return out


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if k == "gt_boxes":
            moved[k] = [x.to(device) for x in v]
        elif torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


class TokenizerOnlyProcessor:
    """Fallback when multimodal processor fails in current transformers build."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text, **kwargs):
        # Drop image/video kwargs in tokenizer-only fallback.
        kwargs.pop("images", None)
        kwargs.pop("videos", None)
        return self.tokenizer(text, **kwargs)


def load_processor(model_name: str, tokenizer):
    errors = []
    if Qwen3VLProcessor is not None:
        try:
            p = Qwen3VLProcessor.from_pretrained(model_name, trust_remote_code=True)
            p.tokenizer = tokenizer
            return p, True
        except Exception as e:
            errors.append(f"Qwen3VLProcessor: {type(e).__name__}: {e}")
    if Qwen2_5_VLProcessor is not None:
        try:
            p = Qwen2_5_VLProcessor.from_pretrained(model_name, trust_remote_code=True)
            p.tokenizer = tokenizer
            return p, True
        except Exception as e:
            errors.append(f"Qwen2_5_VLProcessor: {type(e).__name__}: {e}")
    if Qwen2VLProcessor is not None:
        try:
            p = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)
            p.tokenizer = tokenizer
            return p, True
        except Exception as e:
            errors.append(f"Qwen2VLProcessor: {type(e).__name__}: {e}")
    try:
        p = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        p.tokenizer = tokenizer
        return p, True
    except Exception as e:
        errors.append(f"AutoProcessor: {type(e).__name__}: {e}")

    print("WARNING: Could not load VL processor; falling back to tokenizer-only mode.")
    for err in errors:
        print("  -", err)
    print("Training will run, but vision tensors are disabled in this fallback.")
    return TokenizerOnlyProcessor(tokenizer), False


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    token = args.hf_token if args.hf_token else True
    ds = load_split_dataset(
        dataset_name=args.dataset_name,
        split=args.train_split,
        token=token,
        dataset_from_disk=args.dataset_from_disk,
    )
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Loaded dataset: {args.dataset_name} split={args.train_split} size={len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [DET_QUERY_TOKEN]})
    processor, use_vision = load_processor(args.model_name, tokenizer)
    if args.require_vision and not use_vision:
        raise RuntimeError(
            "Vision processor failed to load. Install compatible vision deps (e.g., torchvision) "
            "or adjust transformers version, then rerun."
        )

    model = Qwen3VLDetrAdapter.from_pretrained(
        args.model_name,
        num_queries=args.num_queries,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))
    model.hungarian_cfg = HungarianLossConfig(
        class_cost=args.class_cost,
        bbox_cost=args.bbox_cost,
        giou_cost=args.giou_cost,
        no_object_weight=args.no_object_weight,
        count_loss_weight=args.count_loss_weight,
    )
    model.loss_cfg = AdapterLossConfig(lm_weight=args.lm_weight, det_weight=args.det_weight)
    model.set_objectness_bias(args.obj_bias_init)

    if args.train_heads_only:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("Training detection heads only.")

    model.to(args.device)
    model.train()

    collator = ShareGptCollator(
        processor=processor,
        num_queries=args.num_queries,
        max_length=args.max_length,
        use_vision=use_vision,
        box_coord_mode=args.box_coord_mode,
        box_coord_order=args.box_coord_order,
        box_supervision_source=args.box_supervision_source,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            global_step += 1
            if args.max_steps > 0 and global_step > args.max_steps:
                break

            batch = to_device(batch, args.device)
            model_inputs = {
                "input_ids": batch["input_ids"],
                "labels": batch["labels"],
                "query_positions": batch["query_positions"],
                "gt_boxes": batch["gt_boxes"],
            }
            for k in ("attention_mask", "pixel_values", "image_grid_thw"):
                if k in batch and batch[k] is not None:
                    model_inputs[k] = batch[k]

            out = model(**model_inputs)
            loss = out["loss"] / max(args.grad_accum_steps, 1)
            loss.backward()

            if global_step % args.grad_accum_steps == 0:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                lm = out.get("lm_loss")
                det = out.get("det_loss")
                det_stats = out.get("det_stats", {})
                pred_count = det_stats.get("pred_count_mean", -1.0)
                gt_count = det_stats.get("gt_count_mean", -1.0)
                obj_pos = det_stats.get("det_obj_pos_loss", -1.0)
                obj_neg = det_stats.get("det_obj_neg_loss", -1.0)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"total={out['loss'].detach().item():.4f} "
                    f"lm={(lm.detach().item() if lm is not None else -1):.4f} "
                    f"det={(det.detach().item() if det is not None else -1):.4f} "
                    f"pred_count={pred_count:.2f} gt_count={gt_count:.2f} "
                    f"obj_pos={obj_pos:.3f} obj_neg={obj_neg:.3f}"
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_dir = os.path.join(args.output_dir, "last")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.base_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    if use_vision:
        try:
            processor.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"WARNING: failed to save processor: {type(e).__name__}: {e}")
    torch.save(
        {
            "adapter_state_dict": model.state_dict(),
            "num_queries": args.num_queries,
            "hungarian_cfg": asdict(model.hungarian_cfg),
            "loss_cfg": asdict(model.loss_cfg),
        },
        os.path.join(ckpt_dir, "adapter.pt"),
    )
    with open(os.path.join(ckpt_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)
    print(f"Saved checkpoint to: {ckpt_dir}")


if __name__ == "__main__":
    main()
