"""Qwen3-VL wrapper with DETR-style query-slot head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional in some transformers versions
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - optional in some transformers versions
    AutoModelForVision2Seq = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - optional in some transformers versions
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:  # pragma: no cover - optional in some transformers versions
    Qwen2VLForConditionalGeneration = None

from .hungarian import HungarianLossConfig, detr_hungarian_loss


@dataclass
class AdapterLossConfig:
    lm_weight: float = 1.0
    det_weight: float = 1.0


class Qwen3VLDetrAdapter(nn.Module):
    """Add fixed query slots + Hungarian box loss on top of a language model.

    Expected batch fields:
      - query_positions: LongTensor [B, Q], token positions of query markers.
      - gt_boxes: list[Tensor [Gi, 4]] in normalized cxcywh format.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_queries: int = 16,
        hungarian_cfg: Optional[HungarianLossConfig] = None,
        loss_cfg: Optional[AdapterLossConfig] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.obj_head = nn.Linear(hidden_size, 1)
        self.box_head = nn.Linear(hidden_size, 4)
        self.hungarian_cfg = hungarian_cfg or HungarianLossConfig()
        self.loss_cfg = loss_cfg or AdapterLossConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_queries: int = 16,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "Qwen3VLDetrAdapter":
        """Load a causal LM and attach DETR-style heads.

        Notes:
          - For some Qwen3-VL checkpoints you may need a different AutoModel class.
          - If your current code already loads the model, pass it to __init__ directly.
        """
        loaders = []
        if Qwen2_5_VLForConditionalGeneration is not None:
            loaders.append(Qwen2_5_VLForConditionalGeneration)
        if Qwen2VLForConditionalGeneration is not None:
            loaders.append(Qwen2VLForConditionalGeneration)
        if AutoModelForImageTextToText is not None:
            loaders.append(AutoModelForImageTextToText)
        if AutoModelForVision2Seq is not None:
            loaders.append(AutoModelForVision2Seq)
        loaders.append(AutoModelForCausalLM)

        base = None
        load_errors: list[str] = []
        for loader in loaders:
            try:
                base = loader.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on runtime env/model
                load_errors.append(f"{loader.__name__}: {exc}")

        if base is None:
            raise ValueError(
                "Could not load model with any supported loader. Errors:\n"
                + "\n".join(load_errors)
            )

        hidden_size = cls._extract_hidden_size(base.config)
        return cls(base_model=base, hidden_size=hidden_size, num_queries=num_queries)

    @staticmethod
    def _extract_hidden_size(config: Any) -> int:
        for key in ("hidden_size", "d_model"):
            if hasattr(config, key):
                return int(getattr(config, key))

        for sub_name in ("text_config", "language_config", "llm_config"):
            sub_cfg = getattr(config, sub_name, None)
            if sub_cfg is None:
                continue
            for key in ("hidden_size", "d_model"):
                if hasattr(sub_cfg, key):
                    return int(getattr(sub_cfg, key))

        raise ValueError("Unable to infer hidden size from model config.")

    def _gather_query_states(
        self,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [B, T, H], query_positions: [B, Q]
        if query_positions.dim() != 2:
            raise ValueError("query_positions must be [B, Q].")
        if query_positions.shape[1] != self.num_queries:
            raise ValueError(
                f"query_positions second dim ({query_positions.shape[1]}) "
                f"must equal num_queries ({self.num_queries})."
            )
        bsz, _, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            raise ValueError(f"Hidden size mismatch: got {hidden}, expected {self.hidden_size}")
        gather_idx = query_positions.unsqueeze(-1).expand(bsz, self.num_queries, hidden)
        return hidden_states.gather(dim=1, index=gather_idx)

    def forward(
        self,
        query_positions: Optional[torch.Tensor] = None,
        gt_boxes: Optional[list[torch.Tensor]] = None,
        **base_inputs: Any,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
          query_positions: [B, Q] token positions for query slots.
          gt_boxes: list of length B, each [Gi,4] normalized cxcywh.
          **base_inputs: forwarded to base model (input_ids, labels, pixel_values, ...).
        """
        outputs = self.base_model(
            output_hidden_states=True,
            return_dict=True,
            **base_inputs,
        )

        result: dict[str, Any] = {
            "base_outputs": outputs,
        }
        if not hasattr(outputs, "logits"):
            raise ValueError(
                "Base model output has no `logits`. Use a conditional-generation "
                "Qwen2.5-VL model class."
            )
        result["logits"] = outputs.logits
        lm_loss = getattr(outputs, "loss", None)
        if lm_loss is not None:
            result["lm_loss"] = lm_loss

        det_loss = None
        if query_positions is not None:
            hidden = outputs.hidden_states[-1]
            query_states = self._gather_query_states(hidden_states=hidden, query_positions=query_positions)
            # Keep detection heads numerically stable by running them in their own dtype.
            # This avoids bf16/fp32 matmul mismatches when base model uses mixed precision.
            head_dtype = self.obj_head.weight.dtype
            if query_states.dtype != head_dtype:
                query_states = query_states.to(head_dtype)
            obj_logits = self.obj_head(query_states).squeeze(-1)  # [B, Q]
            box_pred = torch.sigmoid(self.box_head(query_states))  # [B, Q, 4]
            result["det_obj_logits"] = obj_logits
            result["det_boxes"] = box_pred

            if gt_boxes is not None:
                det_loss, det_stats = detr_hungarian_loss(
                    pred_obj_logits=obj_logits,
                    pred_boxes=box_pred,
                    gt_boxes=gt_boxes,
                    cfg=self.hungarian_cfg,
                )
                result["det_loss"] = det_loss
                result["det_stats"] = det_stats

        total_loss = None
        if lm_loss is not None and det_loss is not None:
            total_loss = self.loss_cfg.lm_weight * lm_loss + self.loss_cfg.det_weight * det_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif det_loss is not None:
            total_loss = det_loss

        if total_loss is not None:
            result["loss"] = total_loss
        return result
