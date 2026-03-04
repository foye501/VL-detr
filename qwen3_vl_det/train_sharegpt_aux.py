"""Train Qwen3-VL with auxiliary DETR supervision on visual tokens (no query prompt tokens).

This keeps generation inputs unchanged and applies Hungarian detection loss only as
an auxiliary branch during training. At inference, the DETR branch can be disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qwen3_vl_det.hungarian import HungarianLossConfig
from qwen3_vl_det.modeling import (
    AdapterLossConfig,
    AuxDetrBranchConfig,
    Qwen3VLAuxDetrAdapter,
)
from qwen3_vl_det.train_sharegpt import (
    _as_messages,
    _extract_image,
    _extract_user_assistant,
    load_processor,
    parse_boxes_from_text,
    to_device,
)


@dataclass
class TrainAuxArgs:
    dataset_name: str = "foye501/VLM-Counting-dataset-qwenvl-sharegpt"
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
    obj_bias_init: float = -2.0
    grad_clip_norm: float = 1.0
    max_length: int = 2048
    max_steps: int = 0
    max_samples: int = 0
    log_every: int = 10
    lm_weight: float = 1.0
    det_weight: float = 0.3
    train_heads_only: bool = False
    require_vision: bool = False
    box_coord_mode: str = "auto"  # auto | absolute | norm1000 | norm01
    box_coord_order: str = "auto"  # auto | xyxy | yxyx
    seed: int = 7
    hf_token: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainAuxArgs:
    p = argparse.ArgumentParser(description="Qwen3-VL + auxiliary DETR finetuning on ShareGPT-style data.")
    p.add_argument("--dataset-name", default=TrainAuxArgs.dataset_name)
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
    p.add_argument("--obj-bias-init", type=float, default=TrainAuxArgs.obj_bias_init)
    p.add_argument("--grad-clip-norm", type=float, default=TrainAuxArgs.grad_clip_norm)
    p.add_argument("--max-length", type=int, default=TrainAuxArgs.max_length)
    p.add_argument("--max-steps", type=int, default=TrainAuxArgs.max_steps)
    p.add_argument("--max-samples", type=int, default=TrainAuxArgs.max_samples)
    p.add_argument("--log-every", type=int, default=TrainAuxArgs.log_every)
    p.add_argument("--lm-weight", type=float, default=TrainAuxArgs.lm_weight)
    p.add_argument("--det-weight", type=float, default=TrainAuxArgs.det_weight)
    p.add_argument("--train-heads-only", action="store_true")
    p.add_argument("--require-vision", action="store_true")
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
    p.add_argument("--seed", type=int, default=TrainAuxArgs.seed)
    p.add_argument("--hf-token", default=TrainAuxArgs.hf_token)
    p.add_argument("--device", default=TrainAuxArgs.device)
    ns = p.parse_args()
    return TrainAuxArgs(**vars(ns))


class ShareGptAuxCollator:
    def __init__(
        self,
        processor,
        max_length: int,
        use_vision: bool,
        box_coord_mode: str,
        box_coord_order: str,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.use_vision = use_vision
        self.box_coord_mode = box_coord_mode
        self.box_coord_order = box_coord_order

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        gt_boxes: list[torch.Tensor] = []

        for ex in batch:
            messages = _as_messages(ex)
            user_text, assistant_text = _extract_user_assistant(messages)
            img = _extract_image(ex)
            width, height = img.size

            user_text = user_text.replace("<image>", "").strip()
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
                parse_boxes_from_text(
                    assistant_text,
                    width=width,
                    height=height,
                    coord_mode=self.box_coord_mode,
                    coord_order=self.box_coord_order,
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

        out = dict(inputs)
        out["labels"] = labels
        out["gt_boxes"] = gt_boxes
        return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    token = args.hf_token if args.hf_token else True
    ds = load_dataset(args.dataset_name, split=args.train_split, token=token)
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

    image_token_id = args.image_token_id if args.image_token_id >= 0 else None
    branch_cfg = AuxDetrBranchConfig(image_token_id=image_token_id)
    model = Qwen3VLAuxDetrAdapter.from_pretrained(
        args.model_name,
        num_queries=args.num_queries,
        branch_cfg=branch_cfg,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
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
        print("Training auxiliary DETR heads only.")

    model.to(args.device)
    model.train()

    collator = ShareGptAuxCollator(
        processor=processor,
        max_length=args.max_length,
        use_vision=use_vision,
        box_coord_mode=args.box_coord_mode,
        box_coord_order=args.box_coord_order,
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
    model.base_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    if use_vision:
        try:
            processor.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"WARNING: failed to save processor: {type(e).__name__}: {e}")
    torch.save(
        {
            "adapter_type": "aux_visual",
            "adapter_state_dict": model.state_dict(),
            "num_queries": args.num_queries,
            "branch_cfg": asdict(model.branch_cfg),
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
