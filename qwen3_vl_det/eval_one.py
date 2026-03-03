"""Run one-sample evaluation and save an overlay image.

Example:
  python -m qwen3_vl_det.eval_one \
    --checkpoint-dir qwen3_vl_det/checkpoints_run1/last \
    --dataset-name foye501/VLM-Counting-dataset-qwenvl-sharegpt \
    --sample-index 0 \
    --output-image qwen3_vl_det/eval_sample0.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from datasets import load_dataset
from PIL import ImageDraw
from transformers import AutoTokenizer

from qwen3_vl_det.modeling import Qwen3VLDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    DET_QUERY_TOKEN,
    _as_messages,
    _extract_image,
    _extract_user_assistant,
    find_query_positions,
    _infer_box_coord_order,
    load_processor,
    _infer_box_coord_mode,
    extract_boxes_raw,
    parse_boxes_from_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one sample and save detection overlay.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-name", default="foye501/VLM-Counting-dataset-qwenvl-sharegpt")
    p.add_argument("--split", default="train")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--obj-threshold", type=float, default=0.5)
    p.add_argument("--box-coord-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--box-coord-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument("--model-name", default="")
    p.add_argument("--hf-token", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--output-image", default="qwen3_vl_det/eval_overlay.png")
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


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)

    token = args.hf_token if args.hf_token else True
    ds = load_dataset(args.dataset_name, split=args.split, token=token)
    ex = ds[int(args.sample_index)]

    messages = _as_messages(ex)
    user_text, assistant_text = _extract_user_assistant(messages)
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
    if tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN) < 0:
        tokenizer.add_special_tokens({"additional_special_tokens": [DET_QUERY_TOKEN]})

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
    if not use_vision:
        print("WARNING: running without vision tensors; predictions are not valid for real detection quality.")

    model = Qwen3VLDetrAdapter.from_pretrained(
        ckpt_dir,
        num_queries=args.num_queries,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    adapter_path = os.path.join(ckpt_dir, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu")
        model.load_state_dict(state["adapter_state_dict"], strict=False)
        if "num_queries" in state:
            args.num_queries = int(state["num_queries"])
    else:
        print("WARNING: adapter.pt not found; using randomly initialized adapter heads.")

    model.to(args.device)
    model.eval()

    user_text = user_text.replace("<image>", "").strip()
    query_text = " ".join([DET_QUERY_TOKEN] * args.num_queries)
    user_text = f"{user_text}\n{query_text}"
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
    query_token_id = int(tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))
    query_positions = find_query_positions(inputs["input_ids"], query_token_id, args.num_queries)

    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            query_positions=query_positions,
        )
        obj_prob = out["det_obj_logits"].sigmoid()[0]
        box_pred = out["det_boxes"][0]

    keep = obj_prob >= args.obj_threshold
    pred_boxes = box_pred[keep].detach().cpu()
    pred_xyxy = cxcywh_to_xyxy_abs(pred_boxes, width=w, height=h)

    raw_boxes = extract_boxes_raw(assistant_text)
    inferred_mode = _infer_box_coord_mode(raw_boxes, width=w, height=h)
    inferred_order = _infer_box_coord_order(raw_boxes, width=w, height=h, mode=inferred_mode)
    used_mode = inferred_mode if args.box_coord_mode == "auto" else args.box_coord_mode
    used_order = inferred_order if args.box_coord_order == "auto" else args.box_coord_order
    gt_boxes = parse_boxes_from_text(
        assistant_text,
        width=w,
        height=h,
        coord_mode=args.box_coord_mode,
        coord_order=args.box_coord_order,
    )
    gt_xyxy = cxcywh_to_xyxy_abs(gt_boxes, width=w, height=h)

    vis = img.copy()
    draw_boxes(vis, gt_xyxy, color="lime", width=3)
    draw_boxes(vis, pred_xyxy, color="red", width=2)
    vis.save(args.output_image)

    print(f"Saved overlay: {args.output_image}")
    print(f"GT count: {gt_boxes.shape[0]}")
    print(f"Pred count (@{args.obj_threshold:.2f}): {pred_boxes.shape[0]}")
    print(f"Image size: {w}x{h}")
    print(f"Box coord mode: requested={args.box_coord_mode}, inferred={inferred_mode}, used={used_mode}")
    print(f"Box coord order: requested={args.box_coord_order}, inferred={inferred_order}, used={used_order}")
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


if __name__ == "__main__":
    main()
