"""Train Qwen2.5-VL with DETR-style Hungarian supervision on ShareGPT-style data.

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
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from qwen3_vl_det.hungarian import HungarianLossConfig
from qwen3_vl_det.modeling import AdapterLossConfig, Qwen3VLDetrAdapter

try:
    from transformers import Qwen2_5_VLProcessor
except Exception:  # pragma: no cover
    Qwen2_5_VLProcessor = None

try:
    from transformers import Qwen2VLProcessor
except Exception:  # pragma: no cover
    Qwen2VLProcessor = None

DET_QUERY_TOKEN = "<|det_query|>"
BOX_PATTERN = re.compile(
    r"<box>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</box>",
    flags=re.IGNORECASE,
)


@dataclass
class TrainArgs:
    dataset_name: str = "foye501/VLM-Counting-dataset-qwenvl-sharegpt"
    train_split: str = "train"
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir: str = "qwen3_vl_det/checkpoints"
    num_queries: int = 32
    batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 1
    lr: float = 2e-5
    max_length: int = 2048
    max_steps: int = 0
    max_samples: int = 0
    log_every: int = 10
    lm_weight: float = 1.0
    det_weight: float = 0.7
    train_heads_only: bool = False
    seed: int = 7
    hf_token: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="Qwen2.5-VL + DETR Hungarian finetuning on ShareGPT-style data.")
    p.add_argument("--dataset-name", default=TrainArgs.dataset_name)
    p.add_argument("--train-split", default=TrainArgs.train_split)
    p.add_argument("--model-name", default=TrainArgs.model_name)
    p.add_argument("--output-dir", default=TrainArgs.output_dir)
    p.add_argument("--num-queries", type=int, default=TrainArgs.num_queries)
    p.add_argument("--batch-size", type=int, default=TrainArgs.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=TrainArgs.grad_accum_steps)
    p.add_argument("--epochs", type=int, default=TrainArgs.epochs)
    p.add_argument("--lr", type=float, default=TrainArgs.lr)
    p.add_argument("--max-length", type=int, default=TrainArgs.max_length)
    p.add_argument("--max-steps", type=int, default=TrainArgs.max_steps)
    p.add_argument("--max-samples", type=int, default=TrainArgs.max_samples)
    p.add_argument("--log-every", type=int, default=TrainArgs.log_every)
    p.add_argument("--lm-weight", type=float, default=TrainArgs.lm_weight)
    p.add_argument("--det-weight", type=float, default=TrainArgs.det_weight)
    p.add_argument("--train-heads-only", action="store_true")
    p.add_argument("--seed", type=int, default=TrainArgs.seed)
    p.add_argument("--hf-token", default=TrainArgs.hf_token)
    p.add_argument("--device", default=TrainArgs.device)
    ns = p.parse_args()
    return TrainArgs(**vars(ns))


def _as_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("conversations", "messages"):
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, str):
                return json.loads(value)
            return value
    raise KeyError("Example has neither `conversations` nor `messages`.")


def _extract_user_assistant(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_text = ""
    assistant_text = ""
    for m in messages:
        role = (m.get("from") or m.get("role") or "").lower()
        value = m.get("value", m.get("content", ""))
        if isinstance(value, list):
            # Some schemas store content as typed chunks.
            chunks = []
            for c in value:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        chunks.append(c.get("text", ""))
                elif isinstance(c, str):
                    chunks.append(c)
            value = "\n".join(chunks)
        if role in ("human", "user"):
            user_text = str(value)
        elif role in ("gpt", "assistant"):
            assistant_text = str(value)
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


def parse_boxes_from_text(text: str, width: int, height: int) -> torch.Tensor:
    boxes = []
    for m in BOX_PATTERN.finditer(text):
        x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
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
    def __init__(self, processor, num_queries: int, max_length: int, use_vision: bool) -> None:
        self.processor = processor
        self.num_queries = num_queries
        self.max_length = max_length
        self.use_vision = use_vision
        self.query_token_id = int(processor.tokenizer.convert_tokens_to_ids(DET_QUERY_TOKEN))
        if self.query_token_id < 0:
            raise ValueError(f"{DET_QUERY_TOKEN} token id not found in tokenizer.")

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
            gt_boxes.append(parse_boxes_from_text(assistant_text, width=width, height=height))

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
    ds = load_dataset(args.dataset_name, split=args.train_split, token=token)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Loaded dataset: {args.dataset_name} split={args.train_split} size={len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [DET_QUERY_TOKEN]})
    processor, use_vision = load_processor(args.model_name, tokenizer)

    model = Qwen3VLDetrAdapter.from_pretrained(
        args.model_name,
        num_queries=args.num_queries,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))
    model.hungarian_cfg = HungarianLossConfig(
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        no_object_weight=0.1,
    )
    model.loss_cfg = AdapterLossConfig(lm_weight=args.lm_weight, det_weight=args.det_weight)

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
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                lm = out.get("lm_loss")
                det = out.get("det_loss")
                print(
                    f"epoch={epoch} step={global_step} "
                    f"total={out['loss'].detach().item():.4f} "
                    f"lm={(lm.detach().item() if lm is not None else -1):.4f} "
                    f"det={(det.detach().item() if det is not None else -1):.4f}"
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ckpt_dir = os.path.join(args.output_dir, "last")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.base_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
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
