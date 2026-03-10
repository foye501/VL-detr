"""Evaluate auxiliary-branch checkpoints on ShareGPT-style counting data.

Reports:
- count MAE
- count exact accuracy
- detection precision/recall/F1 at IoU threshold

Unlike eval_split.py (query-token branch), this uses Qwen3VLAuxDetrAdapter and
does not inject DET query tokens into the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import ImageDraw
from transformers import AutoImageProcessor, AutoTokenizer

from qwen3_vl_det.modeling import AuxDetrBranchConfig, Qwen3VLAuxDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    BOX_SUPERVISION_CHOICES,
    DET_QUERY_TOKEN,
    DINO_PATCH_TOKEN,
    _extract_image,
    extract_gt_boxes_from_example,
    extract_user_assistant_from_example,
    parse_boxes_from_text,
    load_split_dataset,
    load_processor,
)


@dataclass
class EvalRow:
    index: int
    gt_count: int
    pred_count_detr: int | None
    pred_count_soft_detr: float | None
    abs_error_detr: int | None
    precision_detr: float | None
    recall_detr: float | None
    f1_detr: float | None
    pred_count_lm_text: int | None
    abs_error_lm_text: int | None
    pred_count_lm_boxes: int
    abs_error_lm_boxes: int
    precision_lm_boxes: float
    recall_lm_boxes: float
    f1_lm_boxes: float
    bucket: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate auxiliary DETR checkpoint on ShareGPT-style data.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--dataset-from-disk", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--obj-threshold", type=float, default=0.35)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--box-coord-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--box-coord-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument(
        "--box-supervision-source",
        choices=list(BOX_SUPERVISION_CHOICES),
        default="all",
    )
    p.add_argument("--easy-max", type=int, default=5)
    p.add_argument("--medium-max", type=int, default=20)
    p.add_argument("--hard-max", type=int, default=50)
    p.add_argument("--model-name", default="")
    p.add_argument("--hf-token", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-dir", default="qwen3_vl_det/eval_aux_overlays")
    p.add_argument("--overlay-every", type=int, default=50)
    p.add_argument("--lm-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-lm-generation", dest="eval_lm_generation", action="store_true")
    p.add_argument("--no-eval-lm-generation", dest="eval_lm_generation", action="store_false")
    p.add_argument("--lm-box-parse-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--lm-box-parse-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument("--output-json", default="qwen3_vl_det/eval_sharegpt_aux.json")
    p.set_defaults(eval_lm_generation=True)
    return p.parse_args()


def bucket_from_gt_count(gt_count: int, easy_max: int, medium_max: int, hard_max: int) -> str:
    if gt_count <= easy_max:
        return "easy"
    if gt_count <= medium_max:
        return "medium"
    if gt_count <= hard_max:
        return "hard"
    return "extreme"


def _na_det_metrics() -> dict[str, float | None]:
    return {
        "count_mae": None,
        "count_soft_mae": None,
        "count_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def aggregate_det_metrics(rows: list[EvalRow], det_active: bool) -> dict[str, float | None]:
    if not det_active:
        return _na_det_metrics()
    valid = [r for r in rows if r.abs_error_detr is not None and r.pred_count_soft_detr is not None]
    if not valid:
        return _na_det_metrics()
    if not rows:
        return {
            "count_mae": 0.0,
            "count_soft_mae": 0.0,
            "count_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    count_mae = sum(int(r.abs_error_detr) for r in valid) / len(valid)
    count_soft_mae = sum(abs(float(r.pred_count_soft_detr) - r.gt_count) for r in valid) / len(valid)
    count_acc = sum(1.0 for r in valid if r.abs_error_detr == 0) / len(valid)
    precision = sum(float(r.precision_detr) for r in valid) / len(valid)
    recall = sum(float(r.recall_detr) for r in valid) / len(valid)
    f1 = sum(float(r.f1_detr) for r in valid) / len(valid)
    return {
        "count_mae": count_mae,
        "count_soft_mae": count_soft_mae,
        "count_accuracy": count_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_lm_text_metrics(rows: list[EvalRow]) -> dict[str, float | int]:
    total = len(rows)
    valid = [r for r in rows if r.abs_error_lm_text is not None]
    if total == 0:
        return {
            "count_mae": 0.0,
            "count_accuracy": 0.0,
            "parse_rate": 0.0,
            "num_parsed": 0,
        }
    if not valid:
        return {
            "count_mae": 0.0,
            "count_accuracy": 0.0,
            "parse_rate": 0.0,
            "num_parsed": 0,
        }
    count_mae = sum(int(r.abs_error_lm_text) for r in valid if r.abs_error_lm_text is not None) / len(valid)
    count_acc = sum(1.0 for r in valid if r.abs_error_lm_text == 0) / len(valid)
    return {
        "count_mae": count_mae,
        "count_accuracy": count_acc,
        "parse_rate": len(valid) / total,
        "num_parsed": len(valid),
    }


def aggregate_lm_box_metrics(rows: list[EvalRow]) -> dict[str, float]:
    if not rows:
        return {
            "count_mae": 0.0,
            "count_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    count_mae = sum(r.abs_error_lm_boxes for r in rows) / len(rows)
    count_acc = sum(1.0 for r in rows if r.abs_error_lm_boxes == 0) / len(rows)
    precision = sum(r.precision_lm_boxes for r in rows) / len(rows)
    recall = sum(r.recall_lm_boxes for r in rows) / len(rows)
    f1 = sum(r.f1_lm_boxes for r in rows) / len(rows)
    return {
        "count_mae": count_mae,
        "count_accuracy": count_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def extract_count_from_text(text: str) -> int | None:
    patterns = [
        r"total\s*count\s*[:=]\s*(-?\d+)",
        r"count\s*[:=]\s*(-?\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    nums = re.findall(r"(?<![\d.])-?\d+(?![\d.])", text)
    if len(nums) == 1:
        try:
            return int(nums[0])
        except Exception:
            return None
    return None


def _box_format_instruction(lm_box_output_mode: str, lm_box_output_order: str) -> str:
    coord_desc = {
        "absolute": "absolute pixel coordinates",
        "norm1000": "0-1000 normalized coordinates",
        "norm01": "0-1 normalized coordinates",
    }.get(lm_box_output_mode, lm_box_output_mode)
    order_desc = " [y1, x1, y2, x2]" if lm_box_output_order == "yxyx" else " [x1, y1, x2, y2]"
    return (
        "Please output one target box per line in the format "
        f"<box>{order_desc}</box> using {coord_desc}, and include 'Total count: N'."
    )


def cxcywh_to_xyxy_abs(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height - 1)
    return out


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0.0) * (a[:, 3] - a[:, 1]).clamp(min=0.0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-8)


def detection_prf(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, iou_thr: float) -> tuple[float, float, float]:
    if pred_xyxy.numel() == 0 and gt_xyxy.numel() == 0:
        return 1.0, 1.0, 1.0
    if pred_xyxy.numel() == 0:
        return 0.0, 0.0, 0.0
    if gt_xyxy.numel() == 0:
        return 0.0, 0.0, 0.0

    iou = pairwise_iou_xyxy(pred_xyxy, gt_xyxy)
    used_gt = set()
    tp = 0
    for i in range(pred_xyxy.shape[0]):
        best_j = int(torch.argmax(iou[i]).item())
        best_iou = float(iou[i, best_j].item())
        if best_iou >= iou_thr and best_j not in used_gt:
            used_gt.add(best_j)
            tp += 1
    fp = pred_xyxy.shape[0] - tp
    fn = gt_xyxy.shape[0] - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return float(precision), float(recall), float(f1)


def draw_boxes(img, boxes_xyxy: torch.Tensor, color: str, width: int = 3) -> None:
    draw = ImageDraw.Draw(img)
    for b in boxes_xyxy.tolist():
        draw.rectangle(b, outline=color, width=width)


def _checkpoint_uses_detr(train_args: dict[str, Any], branch_cfg: AuxDetrBranchConfig) -> bool:
    try:
        det_weight = float(train_args.get("det_weight", 0.0))
    except Exception:
        det_weight = 0.0
    return det_weight > 0.0 or bool(getattr(branch_cfg, "inject_det_queries_to_lm", False))


def build_model_and_processor(args: argparse.Namespace):
    ckpt_dir = args.checkpoint_dir
    train_args_path = os.path.join(ckpt_dir, "train_args.json")
    train_args: dict[str, Any] = {}
    if os.path.exists(train_args_path):
        with open(train_args_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)

    tokenizer_path = ckpt_dir if os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json")) else args.model_name
    if not tokenizer_path:
        raise ValueError("Tokenizer not found in checkpoint; provide --model-name.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if args.model_name:
        processor_model = args.model_name
    elif os.path.exists(os.path.join(ckpt_dir, "preprocessor_config.json")):
        processor_model = ckpt_dir
    elif "model_name" in train_args and train_args["model_name"]:
        processor_model = str(train_args["model_name"])
    else:
        processor_model = ckpt_dir
    processor, use_vision = load_processor(processor_model, tokenizer)
    if args.require_vision and not use_vision:
        raise RuntimeError("Vision processor unavailable. Fix env and rerun with --require-vision.")

    adapter_path = os.path.join(ckpt_dir, "adapter.pt")
    adapter_state: dict[str, Any] | None = None
    ckpt_num_queries = int(args.num_queries)
    branch_cfg = AuxDetrBranchConfig()
    if os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        if "num_queries" in adapter_state:
            ckpt_num_queries = int(adapter_state["num_queries"])
        if "branch_cfg" in adapter_state and isinstance(adapter_state["branch_cfg"], dict):
            branch_cfg = AuxDetrBranchConfig(**adapter_state["branch_cfg"])
    if ckpt_num_queries != int(args.num_queries):
        print(f"INFO: overriding --num-queries {args.num_queries} with checkpoint value {ckpt_num_queries}")
    args.num_queries = ckpt_num_queries

    model = Qwen3VLAuxDetrAdapter.from_pretrained(
        ckpt_dir,
        num_queries=args.num_queries,
        branch_cfg=branch_cfg,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    if adapter_state is not None:
        model.load_state_dict(adapter_state["adapter_state_dict"], strict=False)
    model.to(args.device)
    model.eval()
    if bool(getattr(branch_cfg, "inject_det_queries_to_lm", False)):
        print(
            "LM fusion mode: enabled "
            f"(det_query_token_id={getattr(branch_cfg, 'det_query_token_id', None)})"
        )
    if bool(getattr(branch_cfg, "inject_dino_tokens_to_lm", False)):
        print(
            "LM fusion mode: direct DINO "
            f"(dino_lm_token_id={getattr(branch_cfg, 'dino_lm_token_id', None)}, "
            f"num_tokens={getattr(branch_cfg, 'dino_lm_num_tokens', None)})"
        )
    if bool(getattr(branch_cfg, "inject_fused_visual_tokens_to_lm", False)):
        print(
            "LM fusion mode: fused visual "
            f"(dino_lm_token_id={getattr(branch_cfg, 'dino_lm_token_id', None)}, "
            f"num_tokens={getattr(branch_cfg, 'dino_lm_num_tokens', None)})"
        )
    dino_processor = None
    if bool(getattr(branch_cfg, "use_dino_fusion", False)):
        dino_model_name = str(getattr(branch_cfg, "dino_model_name", "")).strip()
        if not dino_model_name:
            raise RuntimeError("Checkpoint enables DINO fusion but has empty dino_model_name.")
        dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        print(f"DINO fusion active: {dino_model_name}")
    return model, tokenizer, processor, dino_processor, use_vision, train_args


def main() -> None:
    args = parse_args()
    if not (0 <= args.easy_max < args.medium_max < args.hard_max):
        raise ValueError("Require thresholds: 0 <= easy-max < medium-max < hard-max.")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    if args.save_overlays:
        os.makedirs(args.overlay_dir, exist_ok=True)

    token = args.hf_token if args.hf_token else True
    ds = load_split_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        token=token,
        dataset_from_disk=args.dataset_from_disk,
    )
    start = max(0, int(args.start_index))
    end = min(len(ds), start + int(args.max_samples))
    print(f"Evaluating samples [{start}, {end}) from {args.dataset_name}/{args.split}")

    model, tokenizer, processor, dino_processor, use_vision, train_args = build_model_and_processor(args)
    if not use_vision:
        print("WARNING: running without vision tensors; metrics are not valid for real detection.")
    det_active = _checkpoint_uses_detr(train_args, model.branch_cfg)

    ckpt_mode = str(train_args.get("box_coord_mode", "")).lower()
    ckpt_order = str(train_args.get("box_coord_order", "")).lower()
    ckpt_source = str(train_args.get("box_supervision_source", "")).lower()
    ckpt_lm_mode = str(train_args.get("lm_box_output_mode", "")).lower()
    ckpt_lm_order = str(train_args.get("lm_box_output_order", "")).lower()
    ckpt_lm_target_mode = str(train_args.get("lm_target_mode", "")).lower()
    ckpt_lm_append_box_instruction = bool(train_args.get("lm_append_box_instruction", False))
    effective_mode = args.box_coord_mode
    effective_order = args.box_coord_order
    effective_source = args.box_supervision_source
    if effective_mode == "auto" and ckpt_mode in ("absolute", "norm1000", "norm01"):
        effective_mode = ckpt_mode
    if effective_order == "auto" and ckpt_order in ("xyxy", "yxyx"):
        effective_order = ckpt_order
    if args.box_supervision_source == "all" and ckpt_source in BOX_SUPERVISION_CHOICES:
        # Mirror train-time default unless user explicitly overrides.
        effective_source = ckpt_source
    effective_lm_parse_mode = args.lm_box_parse_mode
    effective_lm_parse_order = args.lm_box_parse_order
    if effective_lm_parse_mode == "auto" and ckpt_lm_mode in ("absolute", "norm1000", "norm01"):
        effective_lm_parse_mode = ckpt_lm_mode
    if effective_lm_parse_order == "auto" and ckpt_lm_order in ("xyxy", "yxyx"):
        effective_lm_parse_order = ckpt_lm_order
    print(
        f"Eval coord mode/order/source: requested=({args.box_coord_mode},{args.box_coord_order},{args.box_supervision_source}) "
        f"checkpoint=({ckpt_mode or 'n/a'},{ckpt_order or 'n/a'},{ckpt_source or 'n/a'}) "
        f"used=({effective_mode},{effective_order},{effective_source})"
    )
    print(
        f"Eval LM-box parse mode/order: requested=({args.lm_box_parse_mode},{args.lm_box_parse_order}) "
        f"checkpoint=({ckpt_lm_mode or 'n/a'},{ckpt_lm_order or 'n/a'}) "
        f"used=({effective_lm_parse_mode},{effective_lm_parse_order})"
    )

    rows: list[EvalRow] = []
    lm_text_preview: list[dict[str, Any]] = []
    for idx in range(start, end):
        ex = ds[idx]
        user_text, assistant_text = extract_user_assistant_from_example(ex)
        img = _extract_image(ex).convert("RGB")
        w, h = img.size

        user_text = user_text.replace("<image>", "").strip()
        if bool(getattr(model.branch_cfg, "inject_det_queries_to_lm", False)) and (
            getattr(model.branch_cfg, "det_query_token_id", None) is not None
        ):
            query_text = " ".join([DET_QUERY_TOKEN] * max(int(args.num_queries), 1))
            user_text = f"{user_text}\n{query_text}"
        if (
            bool(getattr(model.branch_cfg, "inject_dino_tokens_to_lm", False))
            or bool(getattr(model.branch_cfg, "inject_fused_visual_tokens_to_lm", False))
        ) and (
            getattr(model.branch_cfg, "dino_lm_token_id", None) is not None
        ):
            dino_text = " ".join(
                [DINO_PATCH_TOKEN] * max(int(getattr(model.branch_cfg, "dino_lm_num_tokens", 16)), 1)
            )
            user_text = f"{user_text}\n{dino_text}"
        if ckpt_lm_target_mode == "box_count" and ckpt_lm_append_box_instruction:
            user_text = (
                f"{user_text}\n"
                f"{_box_format_instruction(ckpt_lm_mode or 'norm1000', ckpt_lm_order or 'yxyx')}"
            )
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if use_vision:
            inputs = processor(
                text=[chat_text],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = processor(
                text=[chat_text],
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if dino_processor is not None:
            dino_inputs = dino_processor(images=[img], return_tensors="pt")
            if "pixel_values" not in dino_inputs:
                raise RuntimeError("DINO processor did not return pixel_values at eval.")
            inputs["dino_pixel_values"] = dino_inputs["pixel_values"].to(args.device)

        if det_active:
            with torch.no_grad():
                out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    dino_pixel_values=inputs.get("dino_pixel_values"),
                    det_enabled=True,
                    return_det=True,
                )
                obj_prob = out["det_obj_logits"].sigmoid()[0]
                box_pred = out["det_boxes"][0]
            keep = obj_prob >= args.obj_threshold
            pred_boxes = box_pred[keep].detach().cpu()
            pred_xyxy = cxcywh_to_xyxy_abs(pred_boxes, width=w, height=h)
            pred_count_soft = float(obj_prob.sum().detach().cpu().item())
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            pred_count_soft = None

        gt_boxes = extract_gt_boxes_from_example(
            ex,
            width=w,
            height=h,
            assistant_text=assistant_text,
            coord_mode=effective_mode,
            coord_order=effective_order,
            box_supervision_source=effective_source,
        )
        gt_xyxy = cxcywh_to_xyxy_abs(gt_boxes, width=w, height=h)

        if det_active:
            precision, recall, f1 = detection_prf(pred_xyxy, gt_xyxy, iou_thr=args.iou_threshold)
        else:
            precision, recall, f1 = None, None, None
        lm_pred_count_text: int | None = None
        lm_abs_error_text: int | None = None
        lm_pred_count_boxes = 0
        lm_abs_error_boxes = 0
        lm_precision = 0.0
        lm_recall = 0.0
        lm_f1 = 0.0
        lm_text = ""
        lm_xyxy = torch.zeros((0, 4), dtype=torch.float32)
        if args.eval_lm_generation:
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": int(args.lm_max_new_tokens),
                "do_sample": False,
            }
            if "pixel_values" in inputs:
                gen_kwargs["pixel_values"] = inputs.get("pixel_values")
            if "image_grid_thw" in inputs:
                gen_kwargs["image_grid_thw"] = inputs.get("image_grid_thw")
            if "dino_pixel_values" in inputs:
                gen_kwargs["dino_pixel_values"] = inputs.get("dino_pixel_values")
            with torch.no_grad():
                gen_ids = model.generate_with_visual_injection(**gen_kwargs)
            gen_trimmed = [
                (out_ids[len(in_ids) :] if out_ids.shape[0] > in_ids.shape[0] else out_ids)
                for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)
            ]
            lm_text = processor.batch_decode(
                gen_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            lm_pred_count_text = extract_count_from_text(lm_text)
            lm_boxes = parse_boxes_from_text(
                lm_text,
                width=w,
                height=h,
                coord_mode=effective_lm_parse_mode,
                coord_order=effective_lm_parse_order,
            )
            lm_xyxy = cxcywh_to_xyxy_abs(lm_boxes, width=w, height=h)
            lm_pred_count_boxes = int(lm_boxes.shape[0])
            lm_abs_error_boxes = abs(lm_pred_count_boxes - int(gt_boxes.shape[0]))
            lm_precision, lm_recall, lm_f1 = detection_prf(lm_xyxy, gt_xyxy, iou_thr=args.iou_threshold)
            if lm_pred_count_text is not None:
                lm_abs_error_text = abs(int(lm_pred_count_text) - int(gt_boxes.shape[0]))
            if len(lm_text_preview) < 20:
                lm_text_preview.append(
                    {
                        "index": idx,
                        "gt_count": int(gt_boxes.shape[0]),
                        "lm_count_text": lm_pred_count_text,
                        "lm_count_boxes": lm_pred_count_boxes,
                        "text": lm_text,
                    }
                )

        gt_count = int(gt_boxes.shape[0])
        pred_count = (int(pred_boxes.shape[0]) if det_active else None)
        row = EvalRow(
            index=idx,
            gt_count=gt_count,
            pred_count_detr=pred_count,
            pred_count_soft_detr=pred_count_soft,
            abs_error_detr=(abs(int(pred_count) - gt_count) if pred_count is not None else None),
            precision_detr=precision,
            recall_detr=recall,
            f1_detr=f1,
            pred_count_lm_text=lm_pred_count_text,
            abs_error_lm_text=lm_abs_error_text,
            pred_count_lm_boxes=lm_pred_count_boxes,
            abs_error_lm_boxes=lm_abs_error_boxes,
            precision_lm_boxes=lm_precision,
            recall_lm_boxes=lm_recall,
            f1_lm_boxes=lm_f1,
            bucket=bucket_from_gt_count(
                gt_count=gt_count,
                easy_max=args.easy_max,
                medium_max=args.medium_max,
                hard_max=args.hard_max,
            ),
        )
        rows.append(row)

        if args.save_overlays and ((idx - start) % max(args.overlay_every, 1) == 0):
            vis = img.copy()
            draw_boxes(vis, gt_xyxy, color="lime", width=3)
            if det_active:
                draw_boxes(vis, pred_xyxy, color="red", width=2)
            if args.eval_lm_generation:
                draw_boxes(vis, lm_xyxy, color="dodgerblue", width=2)
            vis.save(os.path.join(args.overlay_dir, f"sample_{idx}.png"))

        if (idx - start + 1) % 50 == 0:
            recent = rows[-50:]
            mae = (
                sum(int(r.abs_error_detr) for r in recent if r.abs_error_detr is not None) / len(recent)
                if det_active
                else None
            )
            lm_mae = sum(r.abs_error_lm_boxes for r in recent) / len(recent)
            if det_active and mae is not None:
                print(
                    f"processed={idx-start+1}/{end-start}, recent_mae_detr={mae:.3f}, "
                    f"recent_mae_lm_boxes={lm_mae:.3f}"
                )
            else:
                print(
                    f"processed={idx-start+1}/{end-start}, recent_mae_detr=n/a, "
                    f"recent_mae_lm_boxes={lm_mae:.3f}"
                )

    if not rows:
        raise RuntimeError("No samples evaluated.")

    metrics_det = aggregate_det_metrics(rows, det_active=det_active)
    metrics_lm_text = aggregate_lm_text_metrics(rows)
    metrics_lm_boxes = aggregate_lm_box_metrics(rows)

    bucket_ranges = {
        "easy": f"<= {args.easy_max}",
        "medium": f"{args.easy_max + 1}-{args.medium_max}",
        "hard": f"{args.medium_max + 1}-{args.hard_max}",
        "extreme": f">= {args.hard_max + 1}",
    }
    bucket_order = ("easy", "medium", "hard", "extreme")
    bucket_metrics: dict[str, dict[str, float | int | str]] = {}
    for bucket in bucket_order:
        b_rows = [r for r in rows if r.bucket == bucket]
        b_metrics_det = aggregate_det_metrics(b_rows, det_active=det_active)
        b_metrics_lm_text = aggregate_lm_text_metrics(b_rows)
        b_metrics_lm_boxes = aggregate_lm_box_metrics(b_rows)
        bucket_metrics[bucket] = {
            "range": bucket_ranges[bucket],
            "num_samples": len(b_rows),
            **b_metrics_det,
            "lm_text": b_metrics_lm_text,
            "lm_boxes": b_metrics_lm_boxes,
        }

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "start_index": start,
        "num_samples": len(rows),
        "obj_threshold": args.obj_threshold,
        "iou_threshold": args.iou_threshold,
        "box_coord_mode": args.box_coord_mode,
        "box_coord_order": args.box_coord_order,
        "box_supervision_source": args.box_supervision_source,
        "box_coord_mode_used": effective_mode,
        "box_coord_order_used": effective_order,
        "box_supervision_source_used": effective_source,
        "lm_box_parse_mode_used": effective_lm_parse_mode,
        "lm_box_parse_order_used": effective_lm_parse_order,
        "bucket_thresholds": {
            "easy_max": args.easy_max,
            "medium_max": args.medium_max,
            "hard_max": args.hard_max,
        },
        "vision_enabled": bool(use_vision),
        "detr_active": bool(det_active),
        "eval_lm_generation": bool(args.eval_lm_generation),
        "metrics": metrics_det,
        "metrics_detr": metrics_det,
        "metrics_lm_text": metrics_lm_text,
        "metrics_lm_boxes": metrics_lm_boxes,
        "bucket_metrics": bucket_metrics,
        "lm_text_preview": lm_text_preview,
        "rows": [asdict(r) for r in rows],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if det_active:
        print("DETR metrics:")
        print(json.dumps(summary["metrics_detr"], indent=2))
    else:
        print("DETR metrics: n/a (DETR inactive for this checkpoint)")
    print("LM text-count metrics:")
    print(json.dumps(summary["metrics_lm_text"], indent=2))
    print("LM generated-box metrics:")
    print(json.dumps(summary["metrics_lm_boxes"], indent=2))
    print("Bucket metrics:")
    for bucket in bucket_order:
        bm = bucket_metrics[bucket]
        if det_active:
            print(
                f"  {bucket:<7} range={bm['range']:<8} n={bm['num_samples']:<4} "
                f"mae={bm['count_mae']:.3f} soft_mae={bm['count_soft_mae']:.3f} acc={bm['count_accuracy']:.3f} "
                f"p={bm['precision']:.3f} r={bm['recall']:.3f} f1={bm['f1']:.3f}"
            )
        else:
            print(
                f"  {bucket:<7} range={bm['range']:<8} n={bm['num_samples']:<4} "
                "mae=n/a soft_mae=n/a acc=n/a p=n/a r=n/a f1=n/a"
            )
        lm_t = bm["lm_text"]
        lm_b = bm["lm_boxes"]
        print(
            f"           lm_text: mae={float(lm_t['count_mae']):.3f} acc={float(lm_t['count_accuracy']):.3f} "
            f"parse_rate={float(lm_t['parse_rate']):.3f} parsed={int(lm_t['num_parsed'])}"
        )
        print(
            f"           lm_boxes: mae={float(lm_b['count_mae']):.3f} acc={float(lm_b['count_accuracy']):.3f} "
            f"p={float(lm_b['precision']):.3f} r={float(lm_b['recall']):.3f} f1={float(lm_b['f1']):.3f}"
        )
    print(f"Saved eval json: {args.output_json}")


if __name__ == "__main__":
    main()
