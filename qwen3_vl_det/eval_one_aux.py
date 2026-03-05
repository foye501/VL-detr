"""Evaluate one sample for the auxiliary-branch model (no query prompt tokens)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from qwen3_vl_det.modeling import AuxDetrBranchConfig, Qwen3VLAuxDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    BOX_SUPERVISION_CHOICES,
    _extract_image,
    extract_gt_boxes_from_example,
    extract_user_assistant_from_example,
    _infer_box_coord_mode,
    _infer_box_coord_order,
    extract_boxes_raw,
    load_split_dataset,
    load_processor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one sample with auxiliary DETR branch.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--dataset-from-disk", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--obj-threshold", type=float, default=0.5)
    p.add_argument("--box-coord-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--box-coord-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument(
        "--box-supervision-source",
        choices=list(BOX_SUPERVISION_CHOICES),
        default="all",
    )
    p.add_argument("--model-name", default="")
    p.add_argument("--hf-token", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--output-image", default="qwen3_vl_det/eval_aux_overlay.png")
    p.add_argument("--output-json", default="")
    p.add_argument("--print-topk", type=int, default=20)
    p.add_argument("--annotate-scores", action="store_true")
    return p.parse_args()


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


def draw_boxes(img, boxes_xyxy: torch.Tensor, color: str, width: int = 3) -> None:
    draw = ImageDraw.Draw(img)
    for b in boxes_xyxy.tolist():
        draw.rectangle(b, outline=color, width=width)


def draw_labeled_boxes(
    img,
    boxes_xyxy: torch.Tensor,
    labels: list[str],
    color: str,
    width: int = 2,
) -> None:
    draw = ImageDraw.Draw(img)
    for b, lab in zip(boxes_xyxy.tolist(), labels):
        draw.rectangle(b, outline=color, width=width)
        x1, y1, _, _ = b
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 10)
        draw.text((tx, ty), lab, fill=color)


def _round4(v: float) -> float:
    return round(float(v), 4)


def _derive_path(base_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".png"
    return f"{root}.{suffix}{ext}"


def build_comparison_strip(
    gt_img: Image.Image,
    pred_img: Image.Image,
    overlay_img: Image.Image,
) -> Image.Image:
    w, h = gt_img.size
    out = Image.new("RGB", (w * 3, h + 22), color=(255, 255, 255))
    out.paste(gt_img, (0, 22))
    out.paste(pred_img, (w, 22))
    out.paste(overlay_img, (2 * w, 22))
    draw = ImageDraw.Draw(out)
    draw.text((8, 4), "GT", fill="black")
    draw.text((w + 8, 4), "Pred", fill="black")
    draw.text((2 * w + 8, 4), "Overlay", fill="black")
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)

    token = args.hf_token if args.hf_token else True
    ds = load_split_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        token=token,
        dataset_from_disk=args.dataset_from_disk,
    )
    ex = ds[int(args.sample_index)]

    user_text, assistant_text = extract_user_assistant_from_example(ex)
    img = _extract_image(ex).convert("RGB")
    w, h = img.size

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
        raise RuntimeError("Vision processor unavailable. Install compatible transformers/torchvision and rerun.")

    adapter_state = None
    branch_cfg = AuxDetrBranchConfig()
    adapter_path = os.path.join(ckpt_dir, "adapter.pt")
    ckpt_num_queries = int(args.num_queries)
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
    else:
        print("WARNING: adapter.pt not found; using randomly initialized adapter heads.")
    model.to(args.device)
    model.eval()

    user_text = user_text.replace("<image>", "").replace("<|image_pad|>", "").strip()
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
    if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
        print(f"Processor pixel_values shape: {tuple(inputs['pixel_values'].shape)}")
    if "image_grid_thw" in inputs and torch.is_tensor(inputs["image_grid_thw"]):
        print(f"Processor image_grid_thw: {inputs['image_grid_thw'].detach().cpu().tolist()}")

    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            det_enabled=True,
            return_det=True,
        )
        obj_prob = out["det_obj_logits"].sigmoid()[0]
        box_pred = out["det_boxes"][0]

    keep = obj_prob >= args.obj_threshold
    pred_boxes = box_pred[keep].detach().cpu()
    pred_scores = obj_prob[keep].detach().cpu()
    pred_xyxy = cxcywh_to_xyxy_abs(pred_boxes, width=w, height=h)
    soft_count = float(obj_prob.sum().detach().cpu().item())

    k = max(1, int(args.print_topk))
    topk = min(k, int(obj_prob.shape[0]))
    top_scores, top_idx = torch.topk(obj_prob.detach().cpu(), k=topk, largest=True)
    top_boxes = box_pred.detach().cpu()[top_idx]
    top_xyxy = cxcywh_to_xyxy_abs(top_boxes, width=w, height=h)

    raw_boxes = extract_boxes_raw(assistant_text)
    inferred_mode = _infer_box_coord_mode(raw_boxes, width=w, height=h)
    inferred_order = _infer_box_coord_order(raw_boxes, width=w, height=h, mode=inferred_mode)
    ckpt_mode = str(train_args.get("box_coord_mode", "")).lower()
    ckpt_order = str(train_args.get("box_coord_order", "")).lower()
    ckpt_source = str(train_args.get("box_supervision_source", "")).lower()
    effective_mode = args.box_coord_mode
    effective_order = args.box_coord_order
    effective_source = args.box_supervision_source
    if effective_mode == "auto" and ckpt_mode in ("absolute", "norm1000", "norm01"):
        effective_mode = ckpt_mode
    if effective_order == "auto" and ckpt_order in ("xyxy", "yxyx"):
        effective_order = ckpt_order
    if args.box_supervision_source == "all" and ckpt_source in BOX_SUPERVISION_CHOICES:
        effective_source = ckpt_source
    used_mode = inferred_mode if effective_mode == "auto" else effective_mode
    used_order = inferred_order if effective_order == "auto" else effective_order
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

    vis_gt = img.copy()
    draw_boxes(vis_gt, gt_xyxy, color="lime", width=3)  # GT

    vis_pred = img.copy()
    if args.annotate_scores and pred_xyxy.numel() > 0:
        labels = [f"{_round4(s)}" for s in pred_scores.tolist()]
        draw_labeled_boxes(vis_pred, pred_xyxy, labels=labels, color="red", width=2)
    else:
        draw_boxes(vis_pred, pred_xyxy, color="red", width=2)  # Pred

    vis_overlay = img.copy()
    draw_boxes(vis_overlay, gt_xyxy, color="lime", width=3)  # GT
    if args.annotate_scores and pred_xyxy.numel() > 0:
        labels = [f"{_round4(s)}" for s in pred_scores.tolist()]
        draw_labeled_boxes(vis_overlay, pred_xyxy, labels=labels, color="red", width=2)
    else:
        draw_boxes(vis_overlay, pred_xyxy, color="red", width=2)  # Pred

    gt_image_path = _derive_path(args.output_image, "gt")
    pred_image_path = _derive_path(args.output_image, "pred")
    compare_image_path = _derive_path(args.output_image, "compare")
    vis_gt.save(gt_image_path)
    vis_pred.save(pred_image_path)
    vis_overlay.save(args.output_image)
    compare = build_comparison_strip(vis_gt, vis_pred, vis_overlay)
    compare.save(compare_image_path)

    pred_items = []
    for i in range(int(pred_boxes.shape[0])):
        pred_items.append(
            {
                "rank": i,
                "score": _round4(pred_scores[i].item()),
                "cxcywh_norm": [_round4(v) for v in pred_boxes[i].tolist()],
                "xyxy_abs": [_round4(v) for v in pred_xyxy[i].tolist()],
            }
        )
    top_items = []
    for i in range(topk):
        top_items.append(
            {
                "query_idx": int(top_idx[i].item()),
                "score": _round4(top_scores[i].item()),
                "cxcywh_norm": [_round4(v) for v in top_boxes[i].tolist()],
                "xyxy_abs": [_round4(v) for v in top_xyxy[i].tolist()],
            }
        )
    gt_items = []
    for i in range(int(gt_boxes.shape[0])):
        gt_items.append(
            {
                "rank": i,
                "cxcywh_norm": [_round4(v) for v in gt_boxes[i].tolist()],
                "xyxy_abs": [_round4(v) for v in gt_xyxy[i].tolist()],
            }
        )

    out_json = args.output_json.strip()
    if not out_json:
        out_json = args.output_image + ".json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    sidecar = {
        "sample_index": int(args.sample_index),
        "split": args.split,
        "checkpoint_dir": args.checkpoint_dir,
        "image_size": {"width": int(w), "height": int(h)},
        "prompt_user_text": user_text,
        "assistant_text": assistant_text,
        "threshold": float(args.obj_threshold),
        "counts": {
            "gt_count": int(gt_boxes.shape[0]),
            "pred_count_thresholded": int(pred_boxes.shape[0]),
            "pred_soft_count": _round4(soft_count),
        },
        "gt_boxes": gt_items,
        "pred_boxes_thresholded": pred_items,
        "pred_topk_queries": top_items,
        "settings": {
            "box_coord_mode_used": used_mode,
            "box_coord_order_used": used_order,
            "box_supervision_source_used": effective_source,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved overlay: {args.output_image}")
    print(f"Saved GT-only: {gt_image_path}")
    print(f"Saved Pred-only: {pred_image_path}")
    print(f"Saved side-by-side compare: {compare_image_path}")
    print(f"Saved prediction json: {out_json}")
    print(f"GT count: {gt_boxes.shape[0]}")
    print(f"Pred count (@{args.obj_threshold:.2f}): {pred_boxes.shape[0]}")
    print(f"Pred soft count (sum probs): {soft_count:.2f}")
    print(f"Image size: {w}x{h}")
    print(
        f"Box coord mode: requested={args.box_coord_mode}, checkpoint={ckpt_mode or 'n/a'}, "
        f"inferred={inferred_mode}, used={used_mode}"
    )
    print(
        f"Box coord order: requested={args.box_coord_order}, checkpoint={ckpt_order or 'n/a'}, "
        f"inferred={inferred_order}, used={used_order}"
    )
    print(
        f"Box supervision source: requested={args.box_supervision_source}, "
        f"checkpoint={ckpt_source or 'n/a'}, used={effective_source}"
    )
    if gt_boxes.numel() > 0:
        print(
            "GT box stats (norm cxcywh):",
            {
                "cx_mean": round(float(gt_boxes[:, 0].mean()), 4),
                "cy_mean": round(float(gt_boxes[:, 1].mean()), 4),
                "w_mean": round(float(gt_boxes[:, 2].mean()), 4),
                "h_mean": round(float(gt_boxes[:, 3].mean()), 4),
            },
        )
    if pred_boxes.numel() > 0:
        print(
            "Pred box stats (norm cxcywh):",
            {
                "cx_mean": round(float(pred_boxes[:, 0].mean()), 4),
                "cy_mean": round(float(pred_boxes[:, 1].mean()), 4),
                "w_mean": round(float(pred_boxes[:, 2].mean()), 4),
                "h_mean": round(float(pred_boxes[:, 3].mean()), 4),
            },
        )
    print("Pred objectness (first 10):", [round(float(x), 4) for x in obj_prob[:10].detach().cpu()])
    print(f"Top-{topk} queries by objectness:")
    for item in top_items:
        print(
            f"  q={item['query_idx']:>3} score={item['score']:.4f} "
            f"cxcywh={item['cxcywh_norm']} xyxy={item['xyxy_abs']}"
        )
    if "det_stats" in out:
        print("Det stats:", json.dumps(out["det_stats"], indent=2))
    if "lm_loss" in out:
        print(f"LM loss (for this sample prompt): {float(out['lm_loss'].detach().cpu()):.4f}")
    print("Branch cfg:", json.dumps(asdict(branch_cfg), indent=2))


if __name__ == "__main__":
    main()
