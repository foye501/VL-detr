"""Train Qwen3-VL with auxiliary DETR supervision on visual tokens (no query prompt tokens).

This keeps generation inputs unchanged and applies Hungarian detection loss only as
an auxiliary branch during training. At inference, the DETR branch can be disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from qwen3_vl_det.hungarian import HungarianLossConfig
from qwen3_vl_det.modeling import (
    AdapterLossConfig,
    AuxDetrBranchConfig,
    Qwen3VLAuxDetrAdapter,
)
from qwen3_vl_det.train_sharegpt import (
    BOX_SUPERVISION_CHOICES,
    DET_QUERY_TOKEN,
    _extract_image,
    cxcywh_norm_to_xyxy_abs,
    extract_gt_boxes_from_example,
    extract_user_assistant_from_example,
    load_split_dataset,
    load_processor,
    to_device,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass
class TrainAuxArgs:
    dataset_name: str = "foye501/VLM-Counting-dataset-qwenvl-sharegpt"
    dataset_from_disk: str = ""
    train_split: str = "train"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "qwen3_vl_det/checkpoints_aux"
    num_queries: int = 100
    image_token_id: int = -1  # set >=0 to bypass auto inference
    batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 1
    lr: float = 2e-5
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    no_object_weight: float = 0.5
    count_loss_weight: float = 0.5
    count_loss_normalize_by_queries: bool = False
    obj_bias_init: float = -2.0
    grad_clip_norm: float = 1.0
    max_length: int = 2048
    max_steps: int = 0
    max_samples: int = 0
    log_every: int = 10
    lm_weight: float = 1.0
    det_weight: float = 0.3
    assistant_only_loss: bool = True
    train_heads_only: bool = False
    require_vision: bool = False
    use_lora: bool = False
    enable_vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_modules_to_save: str = ""
    merge_lora_on_save: bool = True
    freeze_vision_backbone: bool = False
    strict_vision_memory: bool = True
    box_coord_mode: str = "auto"  # auto | absolute | norm1000 | norm01
    box_coord_order: str = "auto"  # auto | xyxy | yxyx
    box_supervision_source: str = "all"  # target | all
    lm_target_mode: str = "dataset"  # dataset | count_only | box_count | mixed
    lm_box_source: str = "target"  # target | all
    lm_box_ratio: float = 0.5  # used only for mixed
    lm_box_output_mode: str = "norm1000"  # absolute | norm1000 | norm01
    lm_box_output_order: str = "yxyx"  # xyxy | yxyx
    lm_count_first: bool = True
    lm_append_box_instruction: bool = True
    inject_det_queries_to_lm: bool = False
    detach_det_queries_for_lm: bool = False
    use_dino_fusion: bool = False
    dino_model_name: str = "facebook/dinov2-base"
    dino_trainable: bool = False
    dino_drop_cls_token: bool = True
    dino_cross_attn_heads: int = 8
    dino_cross_attn_dropout: float = 0.0
    dino_gate_init: float = 0.1
    seed: int = 7
    hf_token: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainAuxArgs:
    p = argparse.ArgumentParser(description="Qwen3-VL + auxiliary DETR finetuning on ShareGPT-style data.")
    p.add_argument("--dataset-name", default=TrainAuxArgs.dataset_name)
    p.add_argument("--dataset-from-disk", default=TrainAuxArgs.dataset_from_disk)
    p.add_argument("--train-split", default=TrainAuxArgs.train_split)
    p.add_argument("--model-name", default=TrainAuxArgs.model_name)
    p.add_argument("--output-dir", default=TrainAuxArgs.output_dir)
    p.add_argument("--num-queries", type=int, default=TrainAuxArgs.num_queries)
    p.add_argument("--image-token-id", type=int, default=TrainAuxArgs.image_token_id)
    p.add_argument("--batch-size", type=int, default=TrainAuxArgs.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=TrainAuxArgs.grad_accum_steps)
    p.add_argument("--epochs", type=int, default=TrainAuxArgs.epochs)
    p.add_argument("--lr", type=float, default=TrainAuxArgs.lr)
    p.add_argument("--class-cost", type=float, default=TrainAuxArgs.class_cost)
    p.add_argument("--bbox-cost", type=float, default=TrainAuxArgs.bbox_cost)
    p.add_argument("--giou-cost", type=float, default=TrainAuxArgs.giou_cost)
    p.add_argument("--no-object-weight", type=float, default=TrainAuxArgs.no_object_weight)
    p.add_argument("--count-loss-weight", type=float, default=TrainAuxArgs.count_loss_weight)
    p.add_argument(
        "--count-loss-normalize-by-queries",
        action="store_true",
        default=TrainAuxArgs.count_loss_normalize_by_queries,
    )
    p.add_argument("--obj-bias-init", type=float, default=TrainAuxArgs.obj_bias_init)
    p.add_argument("--grad-clip-norm", type=float, default=TrainAuxArgs.grad_clip_norm)
    p.add_argument("--max-length", type=int, default=TrainAuxArgs.max_length)
    p.add_argument("--max-steps", type=int, default=TrainAuxArgs.max_steps)
    p.add_argument("--max-samples", type=int, default=TrainAuxArgs.max_samples)
    p.add_argument("--log-every", type=int, default=TrainAuxArgs.log_every)
    p.add_argument("--lm-weight", type=float, default=TrainAuxArgs.lm_weight)
    p.add_argument("--det-weight", type=float, default=TrainAuxArgs.det_weight)
    p.add_argument("--assistant-only-loss", dest="assistant_only_loss", action="store_true")
    p.add_argument("--full-seq-loss", dest="assistant_only_loss", action="store_false")
    p.add_argument("--train-heads-only", action="store_true")
    p.add_argument("--require-vision", action="store_true")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--enable-vision-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=TrainAuxArgs.lora_r)
    p.add_argument("--lora-alpha", type=int, default=TrainAuxArgs.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=TrainAuxArgs.lora_dropout)
    p.add_argument("--lora-target-modules", default=TrainAuxArgs.lora_target_modules)
    p.add_argument("--lora-modules-to-save", default=TrainAuxArgs.lora_modules_to_save)
    p.add_argument("--merge-lora-on-save", dest="merge_lora_on_save", action="store_true")
    p.add_argument("--no-merge-lora-on-save", dest="merge_lora_on_save", action="store_false")
    p.add_argument("--freeze-vision-backbone", action="store_true")
    p.add_argument("--strict-vision-memory", dest="strict_vision_memory", action="store_true")
    p.add_argument("--allow-token-fallback-memory", dest="strict_vision_memory", action="store_false")
    p.add_argument(
        "--box-coord-mode",
        choices=["auto", "absolute", "norm1000", "norm01"],
        default=TrainAuxArgs.box_coord_mode,
    )
    p.add_argument(
        "--box-coord-order",
        choices=["auto", "xyxy", "yxyx"],
        default=TrainAuxArgs.box_coord_order,
    )
    p.add_argument(
        "--box-supervision-source",
        choices=list(BOX_SUPERVISION_CHOICES),
        default=TrainAuxArgs.box_supervision_source,
    )
    p.add_argument(
        "--lm-target-mode",
        choices=["dataset", "count_only", "box_count", "mixed"],
        default=TrainAuxArgs.lm_target_mode,
    )
    p.add_argument(
        "--lm-box-source",
        choices=list(BOX_SUPERVISION_CHOICES),
        default=TrainAuxArgs.lm_box_source,
    )
    p.add_argument("--lm-box-ratio", type=float, default=TrainAuxArgs.lm_box_ratio)
    p.add_argument(
        "--lm-box-output-mode",
        choices=["absolute", "norm1000", "norm01"],
        default=TrainAuxArgs.lm_box_output_mode,
    )
    p.add_argument(
        "--lm-box-output-order",
        choices=["xyxy", "yxyx"],
        default=TrainAuxArgs.lm_box_output_order,
    )
    p.add_argument("--lm-count-first", dest="lm_count_first", action="store_true")
    p.add_argument("--lm-count-last", dest="lm_count_first", action="store_false")
    p.add_argument("--lm-append-box-instruction", dest="lm_append_box_instruction", action="store_true")
    p.add_argument("--no-lm-append-box-instruction", dest="lm_append_box_instruction", action="store_false")
    p.add_argument("--inject-det-queries-to-lm", action="store_true")
    p.add_argument("--detach-det-queries-for-lm", action="store_true")
    p.add_argument("--use-dino-fusion", action="store_true")
    p.add_argument("--dino-model-name", default=TrainAuxArgs.dino_model_name)
    p.add_argument("--dino-trainable", action="store_true")
    p.add_argument("--dino-drop-cls-token", dest="dino_drop_cls_token", action="store_true")
    p.add_argument("--keep-dino-cls-token", dest="dino_drop_cls_token", action="store_false")
    p.add_argument("--dino-cross-attn-heads", type=int, default=TrainAuxArgs.dino_cross_attn_heads)
    p.add_argument("--dino-cross-attn-dropout", type=float, default=TrainAuxArgs.dino_cross_attn_dropout)
    p.add_argument("--dino-gate-init", type=float, default=TrainAuxArgs.dino_gate_init)
    p.add_argument("--seed", type=int, default=TrainAuxArgs.seed)
    p.add_argument("--hf-token", default=TrainAuxArgs.hf_token)
    p.add_argument("--device", default=TrainAuxArgs.device)
    p.set_defaults(
        assistant_only_loss=TrainAuxArgs.assistant_only_loss,
        merge_lora_on_save=TrainAuxArgs.merge_lora_on_save,
        strict_vision_memory=TrainAuxArgs.strict_vision_memory,
        lm_count_first=TrainAuxArgs.lm_count_first,
        lm_append_box_instruction=TrainAuxArgs.lm_append_box_instruction,
        dino_drop_cls_token=TrainAuxArgs.dino_drop_cls_token,
    )
    ns = p.parse_args()
    return TrainAuxArgs(**vars(ns))


class ShareGptAuxCollator:
    def __init__(
        self,
        processor,
        dino_processor,
        max_length: int,
        use_vision: bool,
        use_dino_fusion: bool,
        box_coord_mode: str,
        box_coord_order: str,
        box_supervision_source: str,
        lm_target_mode: str,
        lm_box_source: str,
        lm_box_ratio: float,
        lm_box_output_mode: str,
        lm_box_output_order: str,
        lm_count_first: bool,
        lm_append_box_instruction: bool,
        inject_det_queries_to_lm: bool,
        num_queries: int,
        assistant_only_loss: bool,
    ) -> None:
        self.processor = processor
        self.dino_processor = dino_processor
        self.max_length = max_length
        self.use_vision = use_vision
        self.use_dino_fusion = bool(use_dino_fusion)
        self.box_coord_mode = box_coord_mode
        self.box_coord_order = box_coord_order
        self.box_supervision_source = box_supervision_source
        self.lm_target_mode = lm_target_mode
        self.lm_box_source = lm_box_source
        self.lm_box_ratio = max(0.0, min(1.0, float(lm_box_ratio)))
        self.lm_box_output_mode = lm_box_output_mode
        self.lm_box_output_order = lm_box_output_order
        self.lm_count_first = bool(lm_count_first)
        self.lm_append_box_instruction = bool(lm_append_box_instruction)
        self.inject_det_queries_to_lm = bool(inject_det_queries_to_lm)
        self.num_queries = int(num_queries)
        self.assistant_only_loss = assistant_only_loss
        self.query_token_id = -1
        if self.inject_det_queries_to_lm:
            vocab = self.processor.tokenizer.get_vocab()
            if DET_QUERY_TOKEN not in vocab:
                raise ValueError(
                    f"{DET_QUERY_TOKEN} is missing in tokenizer vocab while "
                    "--inject-det-queries-to-lm is enabled."
                )
            self.query_token_id = int(self.processor.tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))

    def _format_box_for_lm(self, xyxy_abs: list[float], width: int, height: int) -> list[float | int]:
        x1, y1, x2, y2 = [float(v) for v in xyxy_abs]
        if self.lm_box_output_mode == "absolute":
            vals_xyxy = [x1, y1, x2, y2]
            vals_xyxy = [int(round(v)) for v in vals_xyxy]
        elif self.lm_box_output_mode == "norm01":
            vals_xyxy = [
                x1 / max(float(width), 1.0),
                y1 / max(float(height), 1.0),
                x2 / max(float(width), 1.0),
                y2 / max(float(height), 1.0),
            ]
            vals_xyxy = [round(v, 4) for v in vals_xyxy]
        else:  # norm1000
            vals_xyxy = [
                x1 / max(float(width), 1.0) * 1000.0,
                y1 / max(float(height), 1.0) * 1000.0,
                x2 / max(float(width), 1.0) * 1000.0,
                y2 / max(float(height), 1.0) * 1000.0,
            ]
            vals_xyxy = [int(round(v)) for v in vals_xyxy]
        if self.lm_box_output_order == "yxyx":
            return [vals_xyxy[1], vals_xyxy[0], vals_xyxy[3], vals_xyxy[2]]
        return vals_xyxy

    def _assistant_text_from_mode(
        self,
        dataset_assistant_text: str,
        lm_target_boxes: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[str, str]:
        mode = self.lm_target_mode
        if mode == "mixed":
            mode = "box_count" if random.random() < self.lm_box_ratio else "count_only"
        if mode == "dataset":
            return dataset_assistant_text, mode

        count = int(lm_target_boxes.shape[0])
        if mode == "count_only":
            return f"Total count: {count}", mode

        # box_count
        if count <= 0:
            return "Total count: 0", mode
        xyxy = cxcywh_norm_to_xyxy_abs(lm_target_boxes, width=width, height=height)
        lines: list[str] = []
        if self.lm_count_first:
            lines.append(f"Total count: {count}")
        for b in xyxy.tolist():
            vals = self._format_box_for_lm(b, width=width, height=height)
            lines.append(f"<box> [{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}] </box>")
        if not self.lm_count_first:
            lines.append(f"Total count: {count}")
        return "\n".join(lines), mode

    def _box_format_instruction(self) -> str:
        coord_desc = {
            "absolute": "absolute pixel coordinates",
            "norm1000": "0-1000 normalized coordinates",
            "norm01": "0-1 normalized coordinates",
        }.get(self.lm_box_output_mode, self.lm_box_output_mode)
        order_desc = " [y1, x1, y2, x2]" if self.lm_box_output_order == "yxyx" else " [x1, y1, x2, y2]"
        return (
            "Please output one target box per line in the format "
            f"<box>{order_desc}</box> using {coord_desc}, and include 'Total count: N'."
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        prompt_texts = []
        images = []
        gt_boxes: list[torch.Tensor] = []

        for ex in batch:
            user_text, assistant_text_dataset = extract_user_assistant_from_example(ex)
            img = _extract_image(ex)
            width, height = img.size

            lm_target_boxes = extract_gt_boxes_from_example(
                ex,
                width=width,
                height=height,
                assistant_text=assistant_text_dataset,
                coord_mode=self.box_coord_mode,
                coord_order=self.box_coord_order,
                box_supervision_source=self.lm_box_source,
            )
            assistant_text, lm_mode_used = self._assistant_text_from_mode(
                dataset_assistant_text=assistant_text_dataset,
                lm_target_boxes=lm_target_boxes,
                width=width,
                height=height,
            )

            user_text = user_text.replace("<image>", "").strip()
            if self.inject_det_queries_to_lm:
                query_text = " ".join([DET_QUERY_TOKEN] * max(self.num_queries, 1))
                user_text = f"{user_text}\n{query_text}"
            if self.lm_append_box_instruction and lm_mode_used == "box_count":
                user_text = f"{user_text}\n{self._box_format_instruction()}"
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
            chat_messages = [
                user_msg,
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
            prompt_text = self.processor.apply_chat_template(
                [user_msg],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text)
            images.append(img)
            gt_boxes.append(
                extract_gt_boxes_from_example(
                    ex,
                    width=width,
                    height=height,
                    assistant_text=assistant_text_dataset,
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

        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        if self.inject_det_queries_to_lm and self.query_token_id >= 0:
            labels[inputs["input_ids"] == self.query_token_id] = -100
        if self.assistant_only_loss:
            if self.use_vision:
                prompt_inputs = self.processor(
                    text=prompt_texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                prompt_inputs = self.processor(
                    text=prompt_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            if "attention_mask" in prompt_inputs:
                prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            else:
                prompt_lens = [prompt_inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
            for b, p_len in enumerate(prompt_lens):
                p_len_i = int(max(0, min(int(p_len), int(labels.shape[1]))))
                if p_len_i > 0:
                    labels[b, :p_len_i] = -100

        out = dict(inputs)
        if self.use_dino_fusion:
            if self.dino_processor is None:
                raise RuntimeError("DINO fusion enabled but dino_processor is not available.")
            dino_inputs = self.dino_processor(images=images, return_tensors="pt")
            if "pixel_values" not in dino_inputs:
                raise RuntimeError("DINO processor did not return pixel_values.")
            out["dino_pixel_values"] = dino_inputs["pixel_values"]
        out["labels"] = labels
        out["gt_boxes"] = gt_boxes
        return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    processor, use_vision = load_processor(args.model_name, tokenizer)
    if args.require_vision and not use_vision:
        raise RuntimeError(
            "Vision processor failed to load. Install compatible vision deps (e.g., torchvision) "
            "or adjust transformers version, then rerun."
        )
    dino_processor = None
    if args.use_dino_fusion:
        dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_name)
        print(f"DINO fusion enabled with model: {args.dino_model_name}")

    det_query_token_id = None
    added_det_query_tokens = 0
    if args.inject_det_queries_to_lm:
        proc_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else tokenizer
        vocab = proc_tokenizer.get_vocab()
        if DET_QUERY_TOKEN not in vocab:
            added_det_query_tokens = int(
                proc_tokenizer.add_special_tokens({"additional_special_tokens": [DET_QUERY_TOKEN]})
            )
        det_query_token_id = int(proc_tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))
        tokenizer = proc_tokenizer
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = proc_tokenizer
        print(
            f"DETR query injection enabled: token={DET_QUERY_TOKEN} "
            f"id={det_query_token_id} added={added_det_query_tokens}"
        )

    image_token_id = args.image_token_id if args.image_token_id >= 0 else None
    branch_cfg = AuxDetrBranchConfig(
        image_token_id=image_token_id,
        strict_vision_memory=bool(args.strict_vision_memory),
        inject_det_queries_to_lm=bool(args.inject_det_queries_to_lm),
        det_query_token_id=det_query_token_id,
        detach_det_queries_for_lm=bool(args.detach_det_queries_for_lm),
        use_dino_fusion=bool(args.use_dino_fusion),
        dino_model_name=str(args.dino_model_name),
        dino_trainable=bool(args.dino_trainable),
        dino_drop_cls_token=bool(args.dino_drop_cls_token),
        dino_cross_attn_heads=int(args.dino_cross_attn_heads),
        dino_cross_attn_dropout=float(args.dino_cross_attn_dropout),
        dino_gate_init=float(args.dino_gate_init),
    )
    model = Qwen3VLAuxDetrAdapter.from_pretrained(
        args.model_name,
        num_queries=args.num_queries,
        branch_cfg=branch_cfg,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    if added_det_query_tokens > 0:
        model.base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings to {len(tokenizer)}")
    model.hungarian_cfg = HungarianLossConfig(
        class_cost=args.class_cost,
        bbox_cost=args.bbox_cost,
        giou_cost=args.giou_cost,
        no_object_weight=args.no_object_weight,
        count_loss_weight=args.count_loss_weight,
        count_loss_normalize_by_queries=args.count_loss_normalize_by_queries,
    )
    model.loss_cfg = AdapterLossConfig(lm_weight=args.lm_weight, det_weight=args.det_weight)
    model.set_objectness_bias(args.obj_bias_init)

    if args.use_lora:
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is not installed. Install `peft` to use --use-lora.")
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        if args.enable_vision_lora:
            vision_linear_targets: list[str] = []
            for mod_name, mod in model.base_model.named_modules():
                lname = mod_name.lower()
                if (("visual" in lname) or ("vision" in lname)) and isinstance(mod, torch.nn.Linear):
                    vision_linear_targets.append(mod_name)
            vision_linear_targets = sorted(set(vision_linear_targets))
            if vision_linear_targets:
                target_modules = sorted(set(target_modules + vision_linear_targets))
                print(
                    f"Vision LoRA enabled: added {len(vision_linear_targets)} "
                    "vision linear modules to LoRA targets."
                )
            else:
                print(
                    "WARNING: --enable-vision-lora set, but no vision linear modules were found. "
                    "No extra vision LoRA targets added."
                )
        modules_to_save = [x.strip() for x in args.lora_modules_to_save.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save if modules_to_save else None,
        )
        model.base_model = get_peft_model(model.base_model, lora_cfg)
        if hasattr(model.base_model, "print_trainable_parameters"):
            model.base_model.print_trainable_parameters()
        print("LoRA enabled for base model.")

    if args.train_heads_only:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("Training auxiliary DETR heads only.")
    elif args.freeze_vision_backbone:
        frozen = 0
        for n, p in model.base_model.named_parameters():
            lname = n.lower()
            if ("visual" in lname) or ("vision" in lname):
                if args.enable_vision_lora and ("lora_" in lname):
                    continue
                p.requires_grad = False
                frozen += 1
        print(f"Froze vision/backbone params: {frozen}")

    base_total = 0
    base_trainable = 0
    vision_total = 0
    vision_trainable = 0
    for n, p in model.base_model.named_parameters():
        n_params = int(p.numel())
        base_total += n_params
        if p.requires_grad:
            base_trainable += n_params
        lname = n.lower()
        if ("visual" in lname) or ("vision" in lname):
            vision_total += n_params
            if p.requires_grad:
                vision_trainable += n_params
    dino_total = 0
    dino_trainable = 0
    if getattr(model, "dino_model", None) is not None:
        for p in model.dino_model.parameters():
            n_params = int(p.numel())
            dino_total += n_params
            if p.requires_grad:
                dino_trainable += n_params
    print(
        "Base model trainability: "
        f"trainable={base_trainable}/{base_total} "
        f"({(100.0 * base_trainable / max(base_total, 1)):.4f}%) "
        f"vision_trainable={vision_trainable}/{vision_total} "
        f"({(100.0 * vision_trainable / max(vision_total, 1)):.4f}%) "
        f"dino_trainable={dino_trainable}/{dino_total} "
        f"({(100.0 * dino_trainable / max(dino_total, 1)):.4f}%) "
        f"strict_vision_memory={bool(args.strict_vision_memory)}"
    )
    if bool(args.strict_vision_memory) and vision_trainable == 0:
        print(
            "WARNING: strict_vision_memory is enabled, but no vision parameters are trainable. "
            "DETR loss cannot improve the vision encoder in this configuration."
        )

    model.to(args.device)
    model.train()

    collator = ShareGptAuxCollator(
        processor=processor,
        dino_processor=dino_processor,
        max_length=args.max_length,
        use_vision=use_vision,
        use_dino_fusion=bool(args.use_dino_fusion),
        box_coord_mode=args.box_coord_mode,
        box_coord_order=args.box_coord_order,
        box_supervision_source=args.box_supervision_source,
        lm_target_mode=args.lm_target_mode,
        lm_box_source=args.lm_box_source,
        lm_box_ratio=args.lm_box_ratio,
        lm_box_output_mode=args.lm_box_output_mode,
        lm_box_output_order=args.lm_box_output_order,
        lm_count_first=args.lm_count_first,
        lm_append_box_instruction=args.lm_append_box_instruction,
        inject_det_queries_to_lm=args.inject_det_queries_to_lm,
        num_queries=args.num_queries,
        assistant_only_loss=args.assistant_only_loss,
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
        for _, batch in enumerate(loader):
            global_step += 1
            if args.max_steps > 0 and global_step > args.max_steps:
                break

            batch = to_device(batch, args.device)
            model_inputs = {
                "input_ids": batch["input_ids"],
                "labels": batch["labels"],
                "gt_boxes": batch["gt_boxes"],
                "det_enabled": True,
                "return_det": True,
            }
            for k in ("attention_mask", "pixel_values", "image_grid_thw", "dino_pixel_values"):
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
                print(
                    f"epoch={epoch} step={global_step} "
                    f"total={out['loss'].detach().item():.4f} "
                    f"lm={(lm.detach().item() if lm is not None else -1):.4f} "
                    f"det={(det.detach().item() if det is not None else -1):.4f} "
                    f"pred_count={pred_count:.2f} gt_count={gt_count:.2f}"
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_dir = os.path.join(args.output_dir, "last")
    os.makedirs(ckpt_dir, exist_ok=True)
    base_to_save = model.base_model
    if args.use_lora and args.merge_lora_on_save and hasattr(model.base_model, "merge_and_unload"):
        print("Merging LoRA weights into base model for checkpoint export...")
        base_to_save = model.base_model.merge_and_unload()
        model.base_model = base_to_save
    base_to_save.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    if use_vision:
        try:
            processor.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"WARNING: failed to save processor: {type(e).__name__}: {e}")
    aux_only_state = {
        k: v
        for k, v in model.state_dict().items()
        if (not k.startswith("base_model.")) and (not k.startswith("dino_model."))
    }
    if bool(args.use_dino_fusion) and bool(args.dino_trainable):
        print(
            "WARNING: DINO was trainable but dino_model.* weights are excluded from adapter.pt. "
            "Use frozen DINO for reproducible reloads, or add explicit DINO checkpoint saving."
        )
    torch.save(
        {
            "adapter_type": "aux_visual",
            "adapter_state_dict": aux_only_state,
            "num_queries": args.num_queries,
            "branch_cfg": asdict(model.branch_cfg),
            "hungarian_cfg": asdict(model.hungarian_cfg),
            "loss_cfg": asdict(model.loss_cfg),
            "assistant_only_loss": bool(args.assistant_only_loss),
            "use_lora": bool(args.use_lora),
        },
        os.path.join(ckpt_dir, "adapter.pt"),
    )
    with open(os.path.join(ckpt_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)
    print(f"Saved checkpoint to: {ckpt_dir}")


if __name__ == "__main__":
    main()
