"""Run a trained aux/fused-visual checkpoint on an arbitrary real image."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from qwen3_vl_det.eval_one_aux import (
    _box_format_instruction,
    _checkpoint_uses_detr,
    _derive_path,
    _indexed_labels,
    _round4,
    _trim_generated_ids_by_prompt,
    build_comparison_strip,
    build_four_panel_strip,
    cxcywh_to_xyxy_abs,
    draw_boxes,
    draw_labeled_boxes,
    extract_count_from_text,
    load_plain_base_model,
)
from qwen3_vl_det.modeling import AuxDetrBranchConfig, Qwen3VLAuxDetrAdapter
from qwen3_vl_det.train_sharegpt import (
    DET_QUERY_TOKEN,
    DINO_PATCH_TOKEN,
    extract_boxes_raw,
    load_processor,
    parse_boxes_from_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer count/boxes on an arbitrary real image.")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--image-path", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--target-object", default="")
    p.add_argument("--base-model-only", action="store_true")
    p.add_argument("--model-name", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--obj-threshold", type=float, default=0.15)
    p.add_argument("--lm-max-new-tokens", type=int, default=1024)
    p.add_argument("--lm-box-parse-mode", choices=["auto", "absolute", "norm1000", "norm01"], default="auto")
    p.add_argument("--lm-box-parse-order", choices=["auto", "xyxy", "yxyx"], default="auto")
    p.add_argument("--output-image", default="qwen3_vl_det/real_image_infer.png")
    p.add_argument("--output-json", default="")
    return p.parse_args()


def _default_prompt(target_object: str) -> str:
    target = target_object.strip()
    if target:
        return f"Count the number of {target} objects in this image. Please count the objects."
    return "Count the relevant objects in this image. Please count the objects."


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)

    img = Image.open(args.image_path).convert("RGB")
    w, h = img.size

    ckpt_dir = args.checkpoint_dir
    is_local_ckpt = os.path.isdir(ckpt_dir)
    train_args_path = os.path.join(ckpt_dir, "train_args.json")
    train_args: dict[str, Any] = {}
    if is_local_ckpt and os.path.exists(train_args_path):
        with open(train_args_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
    ckpt_lm_mode = str(train_args.get("lm_box_output_mode", "")).lower()
    ckpt_lm_order = str(train_args.get("lm_box_output_order", "")).lower()
    ckpt_lm_target_mode = str(train_args.get("lm_target_mode", "")).lower()
    ckpt_lm_append_box_instruction = bool(train_args.get("lm_append_box_instruction", False))

    effective_lm_parse_mode = args.lm_box_parse_mode
    effective_lm_parse_order = args.lm_box_parse_order
    if effective_lm_parse_mode == "auto" and ckpt_lm_mode in ("absolute", "norm1000", "norm01"):
        effective_lm_parse_mode = ckpt_lm_mode
    if effective_lm_parse_order == "auto" and ckpt_lm_order in ("xyxy", "yxyx"):
        effective_lm_parse_order = ckpt_lm_order

    tokenizer_path = (
        ckpt_dir
        if is_local_ckpt and os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json"))
        else args.model_name
    )
    if args.base_model_only and not tokenizer_path:
        tokenizer_path = ckpt_dir
    if not tokenizer_path:
        raise ValueError("Tokenizer not found in checkpoint; provide --model-name.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if args.model_name:
        processor_model = args.model_name
    elif is_local_ckpt and os.path.exists(os.path.join(ckpt_dir, "preprocessor_config.json")):
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
    if (not args.base_model_only) and is_local_ckpt and os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        if "num_queries" in adapter_state:
            ckpt_num_queries = int(adapter_state["num_queries"])
        if "branch_cfg" in adapter_state and isinstance(adapter_state["branch_cfg"], dict):
            branch_cfg = AuxDetrBranchConfig(**adapter_state["branch_cfg"])
    if (not args.base_model_only) and ckpt_num_queries != int(args.num_queries):
        print(f"INFO: overriding --num-queries {args.num_queries} with checkpoint value {ckpt_num_queries}")
    args.num_queries = ckpt_num_queries
    det_active = (not args.base_model_only) and _checkpoint_uses_detr(train_args, branch_cfg)

    load_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base_model_load_source = args.model_name or ckpt_dir
    if args.base_model_only:
        model, loaded_with = load_plain_base_model(
            base_model_load_source,
            dtype=load_dtype,
            device=args.device,
        )
        print(f"Base-model-only mode: loaded with {loaded_with} from {base_model_load_source}")
    else:
        model = Qwen3VLAuxDetrAdapter.from_pretrained(
            ckpt_dir,
            num_queries=args.num_queries,
            branch_cfg=branch_cfg,
            trust_remote_code=True,
            dtype=load_dtype,
        )
        if adapter_state is not None:
            model.load_state_dict(adapter_state["adapter_state_dict"], strict=False)
        else:
            print("WARNING: adapter.pt not found; using randomly initialized adapter heads.")
        model.to(args.device)
        model.eval()

    dino_processor = None
    if (not args.base_model_only) and bool(getattr(branch_cfg, "use_dino_fusion", False)):
        dino_model_name = str(getattr(branch_cfg, "dino_model_name", "")).strip()
        if not dino_model_name:
            raise RuntimeError("Checkpoint enables DINO fusion but has empty dino_model_name.")
        dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        print(f"DINO fusion active: {dino_model_name}")
    if (not args.base_model_only) and bool(getattr(branch_cfg, "inject_fused_visual_tokens_to_lm", False)):
        print(
            "LM fusion mode: fused visual "
            f"(dino_lm_token_id={getattr(branch_cfg, 'dino_lm_token_id', None)}, "
            f"num_tokens={getattr(branch_cfg, 'dino_lm_num_tokens', None)})"
        )
    elif (not args.base_model_only) and bool(getattr(branch_cfg, "inject_dino_tokens_to_lm", False)):
        print(
            "LM fusion mode: direct DINO "
            f"(dino_lm_token_id={getattr(branch_cfg, 'dino_lm_token_id', None)}, "
            f"num_tokens={getattr(branch_cfg, 'dino_lm_num_tokens', None)})"
        )

    user_text = args.prompt.strip() or _default_prompt(args.target_object)
    if (not args.base_model_only) and bool(getattr(branch_cfg, "inject_det_queries_to_lm", False)) and (
        getattr(branch_cfg, "det_query_token_id", None) is not None
    ):
        query_text = " ".join([DET_QUERY_TOKEN] * max(int(args.num_queries), 1))
        user_text = f"{user_text}\n{query_text}"
    if (
        (not args.base_model_only)
        and (
            bool(getattr(branch_cfg, "inject_dino_tokens_to_lm", False))
            or bool(getattr(branch_cfg, "inject_fused_visual_tokens_to_lm", False))
        )
        and getattr(branch_cfg, "dino_lm_token_id", None) is not None
    ):
        dino_text = " ".join([DINO_PATCH_TOKEN] * max(int(getattr(branch_cfg, "dino_lm_num_tokens", 16)), 1))
        user_text = f"{user_text}\n{dino_text}"
    if ckpt_lm_target_mode == "box_count" and ckpt_lm_append_box_instruction:
        user_text = f"{user_text}\n{_box_format_instruction(ckpt_lm_mode or 'norm1000', ckpt_lm_order or 'yxyx')}"

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
            raise RuntimeError("DINO processor did not return pixel_values.")
        inputs["dino_pixel_values"] = dino_inputs["pixel_values"].to(args.device)

    out: dict[str, Any] = {}
    if args.base_model_only or not det_active:
        obj_prob = torch.zeros((0,), dtype=torch.float32)
        box_pred = torch.zeros((0, 4), dtype=torch.float32)
    else:
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
        if args.base_model_only:
            gen_ids = model.generate(**gen_kwargs)
        else:
            gen_ids = model.generate_with_visual_injection(**gen_kwargs)
    gen_trimmed = []
    trim_meta = []
    for in_ids, out_ids in zip(inputs["input_ids"], gen_ids):
        trimmed_ids, meta = _trim_generated_ids_by_prompt(in_ids, out_ids)
        gen_trimmed.append(trimmed_ids)
        trim_meta.append(meta)
    lm_generated_text = processor.batch_decode(
        gen_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    lm_count_text = extract_count_from_text(lm_generated_text)
    lm_raw_boxes = extract_boxes_raw(lm_generated_text)
    lm_boxes = parse_boxes_from_text(
        lm_generated_text,
        width=w,
        height=h,
        coord_mode=effective_lm_parse_mode,
        coord_order=effective_lm_parse_order,
    )
    lm_xyxy = cxcywh_to_xyxy_abs(lm_boxes, width=w, height=h)

    if det_active and obj_prob.numel() > 0:
        keep = obj_prob >= args.obj_threshold
        pred_boxes = box_pred[keep].detach().cpu()
        pred_scores = obj_prob[keep].detach().cpu()
        pred_xyxy = cxcywh_to_xyxy_abs(pred_boxes, width=w, height=h)
    else:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
        pred_xyxy = torch.zeros((0, 4), dtype=torch.float32)

    pred_labels = _indexed_labels("d", int(pred_xyxy.shape[0]))
    lm_labels = _indexed_labels("p", int(lm_xyxy.shape[0]))

    vis_pred = img.copy()
    if det_active:
        draw_labeled_boxes(vis_pred, pred_xyxy, labels=pred_labels, color="red", width=2)
    else:
        draw_boxes(vis_pred, pred_xyxy, color="red", width=2)

    vis_lm = img.copy()
    draw_labeled_boxes(vis_lm, lm_xyxy, labels=lm_labels, color="dodgerblue", width=2)

    vis_overlay = img.copy()
    if det_active:
        draw_labeled_boxes(vis_overlay, pred_xyxy, labels=pred_labels, color="red", width=2)
    draw_labeled_boxes(vis_overlay, lm_xyxy, labels=lm_labels, color="dodgerblue", width=2)

    pred_image_path = _derive_path(args.output_image, "pred")
    lm_image_path = _derive_path(args.output_image, "lm")
    compare_image_path = _derive_path(args.output_image, "compare")
    compare4_image_path = _derive_path(args.output_image, "compare4")
    vis_pred.save(pred_image_path)
    vis_lm.save(lm_image_path)
    vis_overlay.save(args.output_image)
    compare = build_comparison_strip(
        img.copy(),
        vis_lm if not det_active else vis_pred,
        vis_overlay,
        pred_title=("LM boxes" if not det_active else "Pred"),
    )
    compare.save(compare_image_path)
    compare4 = build_four_panel_strip(
        img.copy(),
        vis_pred,
        vis_lm,
        vis_overlay,
        det_title=("DETR n/a" if not det_active else "DETR"),
    )
    compare4.save(compare4_image_path)

    out_json = args.output_json.strip() or (args.output_image + ".json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    sidecar = {
        "image_path": args.image_path,
        "checkpoint_dir": args.checkpoint_dir,
        "base_model_only": bool(args.base_model_only),
        "detr_active": bool(det_active),
        "image_size": {"width": int(w), "height": int(h)},
        "prompt_user_text": user_text,
        "threshold": float(args.obj_threshold),
        "counts": {
            "pred_count_thresholded_detr": (int(pred_boxes.shape[0]) if det_active else None),
            "pred_count_lm_text": int(lm_count_text) if lm_count_text is not None else None,
            "pred_count_lm_boxes": int(lm_boxes.shape[0]),
        },
        "pred_boxes_thresholded_detr": [
            {
                "rank": i,
                "score": _round4(pred_scores[i].item()),
                "cxcywh_norm": [_round4(v) for v in pred_boxes[i].tolist()],
                "xyxy_abs": [_round4(v) for v in pred_xyxy[i].tolist()],
            }
            for i in range(int(pred_boxes.shape[0]))
        ],
        "pred_boxes_lm_generation": [
            {
                "rank": i,
                "raw_box_text_order": ([float(v) for v in lm_raw_boxes[i]] if i < len(lm_raw_boxes) else None),
                "cxcywh_norm": [_round4(v) for v in lm_boxes[i].tolist()],
                "xyxy_abs": [_round4(v) for v in lm_xyxy[i].tolist()],
            }
            for i in range(int(lm_boxes.shape[0]))
        ],
        "lm_generation": {
            "max_new_tokens": int(args.lm_max_new_tokens),
            "box_parse_mode_used": effective_lm_parse_mode,
            "box_parse_order_used": effective_lm_parse_order,
            "trim": (trim_meta[0] if trim_meta else None),
            "text": lm_generated_text,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved overlay: {args.output_image}")
    print(f"Saved Pred-only: {pred_image_path}")
    print(f"Saved LM-only: {lm_image_path}")
    print(f"Saved side-by-side compare: {compare_image_path}")
    print(f"Saved 4-panel compare: {compare4_image_path}")
    print(f"Saved prediction json: {out_json}")
    if det_active:
        print(f"Pred count DETR (@{args.obj_threshold:.2f}): {pred_boxes.shape[0]}")
    else:
        print("Pred count DETR: n/a (DETR inactive for this checkpoint)")
    print(f"Pred count LM text: {lm_count_text if lm_count_text is not None else 'n/a'}")
    print(f"Pred count LM parsed boxes: {int(lm_boxes.shape[0])}")
    if trim_meta:
        print("LM generation trim:", trim_meta[0])
    if lm_raw_boxes:
        print("LM raw boxes (text order):")
        for i, raw in enumerate(lm_raw_boxes[:20]):
            xyxy = ([_round4(v) for v in lm_xyxy[i].tolist()] if i < int(lm_xyxy.shape[0]) else None)
            print(f"  p{i+1}: raw={raw} -> xyxy_abs={xyxy}")
    print("LM generated text:")
    print(lm_generated_text)
    if args.base_model_only:
        print("Branch cfg: n/a (base-model-only mode)")
    else:
        print("Branch cfg:", json.dumps(asdict(branch_cfg), indent=2))


if __name__ == "__main__":
    main()
