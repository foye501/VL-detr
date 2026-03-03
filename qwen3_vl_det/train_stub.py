"""Minimal training stub for Qwen3-VL + DETR-style Hungarian loss.

This is a patch template for your existing trainer, not a full dataset pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from qwen3_vl_det.hungarian import HungarianLossConfig
from qwen3_vl_det.modeling import AdapterLossConfig, Qwen3VLDetrAdapter


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    num_queries: int = 16
    lr: float = 2e-5
    epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 64


def find_query_positions(input_ids: torch.Tensor, query_token_id: int, num_queries: int) -> torch.Tensor:
    """Find query marker positions for each sample."""
    bsz = input_ids.shape[0]
    out = torch.full((bsz, num_queries), -1, dtype=torch.long, device=input_ids.device)
    for b in range(bsz):
        pos = torch.where(input_ids[b] == query_token_id)[0]
        if pos.numel() < num_queries:
            raise ValueError(
                f"Sample {b} has {pos.numel()} query markers but expected {num_queries}."
            )
        out[b] = pos[:num_queries]
    return out


def fake_batch(tokenizer, num_queries: int, device: str) -> dict[str, Any]:
    """Toy batch for wiring check only.

    Replace this with your real collator that returns:
      input_ids, attention_mask, pixel_values..., labels, gt_boxes(list), query_positions.
    """
    prompt = "Count objects and ground them. " + " ".join(["<|det_query|>"] * num_queries)
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    query_positions = find_query_positions(
        input_ids=enc["input_ids"],
        query_token_id=tokenizer.convert_tokens_to_ids("<|det_query|>"),
        num_queries=num_queries,
    )
    gt_boxes = [torch.tensor([[0.4, 0.4, 0.2, 0.2]], dtype=torch.float32, device=device)]

    batch = {
        **enc,
        "labels": labels,
        "query_positions": query_positions,
        "gt_boxes": gt_boxes,
    }
    return batch


def main() -> None:
    cfg = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    # Add a query marker token; you can also reserve multiple query tokens if preferred.
    special_tokens = {"additional_special_tokens": ["<|det_query|>"]}
    tokenizer.add_special_tokens(special_tokens)

    model = Qwen3VLDetrAdapter.from_pretrained(
        cfg.model_name,
        num_queries=cfg.num_queries,
        trust_remote_code=True,
        dtype=torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))
    model.hungarian_cfg = HungarianLossConfig(
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        no_object_weight=0.1,
    )
    model.loss_cfg = AdapterLossConfig(lm_weight=1.0, det_weight=1.0)
    model.to(cfg.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Replace this loop with your real dataloader.
    for step in range(5):
        batch = fake_batch(tokenizer, num_queries=cfg.num_queries, device=cfg.device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            query_positions=batch["query_positions"],
            gt_boxes=batch["gt_boxes"],
        )
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        det_loss = out.get("det_loss")
        lm_loss = out.get("lm_loss")
        print(
            f"step={step} total={float(loss):.4f} "
            f"lm={float(lm_loss) if lm_loss is not None else -1:.4f} "
            f"det={float(det_loss) if det_loss is not None else -1:.4f}"
        )


if __name__ == "__main__":
    main()
