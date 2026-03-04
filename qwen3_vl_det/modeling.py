"""Qwen3-VL wrapper with DETR-style query-slot head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional in some transformers versions
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - optional in some transformers versions
    AutoModelForVision2Seq = None

def _optional_transformers_class(*names: str):
    for name in names:
        try:
            cls = getattr(transformers, name)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


Qwen3VLForConditionalGeneration = _optional_transformers_class(
    "Qwen3VLForConditionalGeneration",
    "Qwen3_VLForConditionalGeneration",
)
Qwen2_5_VLForConditionalGeneration = _optional_transformers_class("Qwen2_5_VLForConditionalGeneration")
Qwen2VLForConditionalGeneration = _optional_transformers_class("Qwen2VLForConditionalGeneration")

from .hungarian import HungarianLossConfig, detr_hungarian_loss


@dataclass
class AdapterLossConfig:
    lm_weight: float = 1.0
    det_weight: float = 1.0


@dataclass
class AuxDetrBranchConfig:
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_ffn_dim: int = 2048
    dropout: float = 0.1
    image_token_id: Optional[int] = None


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
        if self.obj_head.bias is not None:
            # Start conservative so unmatched slots don't all activate early.
            nn.init.constant_(self.obj_head.bias, -2.0)
        self.hungarian_cfg = hungarian_cfg or HungarianLossConfig()
        self.loss_cfg = loss_cfg or AdapterLossConfig()

    def set_objectness_bias(self, bias: float) -> None:
        if self.obj_head.bias is not None:
            nn.init.constant_(self.obj_head.bias, bias)

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
        if Qwen3VLForConditionalGeneration is not None:
            loaders.append(Qwen3VLForConditionalGeneration)
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


class Qwen3VLAuxDetrAdapter(nn.Module):
    """Auxiliary DETR branch on visual tokens, without prompt query tokens.

    This class keeps Qwen3-VL generation path unchanged while adding a train-time
    DETR-style supervision branch on visual representations. At inference, you can
    disable the DETR branch (`det_enabled=False`) and run as plain VLM.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_queries: int = 100,
        branch_cfg: Optional[AuxDetrBranchConfig] = None,
        hungarian_cfg: Optional[HungarianLossConfig] = None,
        loss_cfg: Optional[AdapterLossConfig] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.branch_cfg = branch_cfg or AuxDetrBranchConfig()

        self.det_query_embed = nn.Embedding(num_queries, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=self.branch_cfg.decoder_heads,
            dim_feedforward=self.branch_cfg.decoder_ffn_dim,
            dropout=self.branch_cfg.dropout,
            batch_first=True,
        )
        self.det_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.branch_cfg.decoder_layers)
        self.det_norm = nn.LayerNorm(hidden_size)

        self.obj_head = nn.Linear(hidden_size, 1)
        self.box_head = nn.Linear(hidden_size, 4)
        if self.obj_head.bias is not None:
            nn.init.constant_(self.obj_head.bias, -2.0)
        self.hungarian_cfg = hungarian_cfg or HungarianLossConfig()
        self.loss_cfg = loss_cfg or AdapterLossConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_queries: int = 100,
        branch_cfg: Optional[AuxDetrBranchConfig] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "Qwen3VLAuxDetrAdapter":
        loaders = []
        if Qwen3VLForConditionalGeneration is not None:
            loaders.append(Qwen3VLForConditionalGeneration)
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

        hidden_size = Qwen3VLDetrAdapter._extract_hidden_size(base.config)
        return cls(
            base_model=base,
            hidden_size=hidden_size,
            num_queries=num_queries,
            branch_cfg=branch_cfg,
        )

    def set_objectness_bias(self, bias: float) -> None:
        if self.obj_head.bias is not None:
            nn.init.constant_(self.obj_head.bias, bias)

    def _infer_image_token_id(self, input_ids: torch.Tensor) -> int:
        if self.branch_cfg.image_token_id is not None:
            return int(self.branch_cfg.image_token_id)

        candidate_attrs = (
            "image_token_id",
            "vision_token_id",
            "image_token_index",
            "img_token_id",
        )
        cfgs = [
            getattr(self.base_model, "config", None),
            getattr(getattr(self.base_model, "config", None), "text_config", None),
            getattr(getattr(self.base_model, "config", None), "vision_config", None),
        ]
        for cfg in cfgs:
            if cfg is None:
                continue
            for attr in candidate_attrs:
                if hasattr(cfg, attr):
                    v = getattr(cfg, attr)
                    if v is not None:
                        return int(v)

        # Fallback heuristic: pick most frequent non-pad token id.
        # This is less reliable; pass `image_token_id` in branch_cfg when possible.
        flat = input_ids.reshape(-1)
        uniq, counts = torch.unique(flat, return_counts=True)
        if uniq.numel() == 0:
            raise ValueError("Cannot infer image token id from empty input_ids.")
        best_idx = int(torch.argmax(counts).item())
        return int(uniq[best_idx].item())

    def _extract_visual_memory(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract per-sample visual token states using the image token id mask."""
        bsz, _, hidden = hidden_states.shape
        image_token_id = self._infer_image_token_id(input_ids)

        seqs: list[torch.Tensor] = []
        max_len = 0
        for b in range(bsz):
            idx = torch.where(input_ids[b] == image_token_id)[0]
            if idx.numel() == 0:
                raise ValueError(
                    "No image tokens found in input_ids for DETR aux branch. "
                    "Ensure processor inserts image tokens and `image_token_id` is correct."
                )
            states = hidden_states[b, idx]  # [Vb, H]
            seqs.append(states)
            if int(states.shape[0]) > max_len:
                max_len = int(states.shape[0])

        memory = hidden_states.new_zeros((bsz, max_len, hidden))
        mask = torch.zeros((bsz, max_len), device=hidden_states.device, dtype=torch.bool)
        for b, states in enumerate(seqs):
            v = states.shape[0]
            memory[b, :v] = states
            mask[b, :v] = True
        return memory, mask

    def _decode_queries(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        bsz = memory.shape[0]
        queries = self.det_query_embed.weight.unsqueeze(0).expand(bsz, -1, -1)  # [B, Q, H]
        det_dtype = self.obj_head.weight.dtype
        if memory.dtype != det_dtype:
            memory = memory.to(det_dtype)
        if queries.dtype != det_dtype:
            queries = queries.to(det_dtype)
        query_states = self.det_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~memory_mask,
        )
        return self.det_norm(query_states)

    def forward(
        self,
        gt_boxes: Optional[list[torch.Tensor]] = None,
        det_enabled: bool = True,
        return_det: bool = True,
        **base_inputs: Any,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
          gt_boxes: list of length B, each [Gi,4] normalized cxcywh.
          det_enabled: if False, skip DETR branch entirely.
          return_det: if False and gt_boxes is None, DETR outputs are omitted.
          **base_inputs: input_ids, labels, pixel_values, image_grid_thw, ...
        """
        outputs = self.base_model(
            output_hidden_states=True,
            return_dict=True,
            **base_inputs,
        )

        result: dict[str, Any] = {"base_outputs": outputs}
        if hasattr(outputs, "logits"):
            result["logits"] = outputs.logits
        lm_loss = getattr(outputs, "loss", None)
        if lm_loss is not None:
            result["lm_loss"] = lm_loss

        det_loss = None
        need_det = det_enabled and (gt_boxes is not None or return_det)
        if need_det:
            input_ids = base_inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("input_ids is required for auxiliary DETR visual-token extraction.")

            hidden = outputs.hidden_states[-1]
            memory, memory_mask = self._extract_visual_memory(hidden_states=hidden, input_ids=input_ids)
            query_states = self._decode_queries(memory=memory, memory_mask=memory_mask)
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
