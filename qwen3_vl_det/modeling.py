"""Qwen3-VL wrapper with DETR-style query-slot head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel
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
Dinov2Model = _optional_transformers_class("Dinov2Model")

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
    use_grid_pos: bool = True
    prefer_output_vision_states: bool = True
    strict_vision_memory: bool = False
    inject_det_queries_to_lm: bool = False
    det_query_token_id: Optional[int] = None
    detach_det_queries_for_lm: bool = False
    inject_dino_tokens_to_lm: bool = False
    dino_lm_token_id: Optional[int] = None
    dino_lm_num_tokens: int = 16
    detach_dino_tokens_for_lm: bool = False
    inject_fused_visual_tokens_to_lm: bool = False
    detach_fused_visual_tokens_for_lm: bool = False
    use_dino_fusion: bool = False
    dino_model_name: str = "facebook/dinov2-base"
    dino_drop_cls_token: bool = True
    dino_trainable: bool = False
    dino_cross_attn_heads: int = 8
    dino_cross_attn_dropout: float = 0.0
    dino_gate_init: float = 0.1


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
        self.grid_pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.obj_head = nn.Linear(hidden_size, 1)
        self.box_head = nn.Linear(hidden_size, 4)
        if self.obj_head.bias is not None:
            nn.init.constant_(self.obj_head.bias, -2.0)

        # Optional DINOv2 branch for instance-aware fusion into Qwen visual memory.
        self.dino_model: Optional[nn.Module] = None
        self.dino_proj: Optional[nn.Linear] = None
        self.dino_q_ln: Optional[nn.LayerNorm] = None
        self.dino_kv_ln: Optional[nn.LayerNorm] = None
        self.dino_q_to_d_attn: Optional[nn.MultiheadAttention] = None
        self.dino_gate_mlp: Optional[nn.Sequential] = None
        self.dino_fuse_ln: Optional[nn.LayerNorm] = None
        self.dino_fuse_ffn_ln: Optional[nn.LayerNorm] = None
        self.dino_fuse_ffn: Optional[nn.Sequential] = None
        self.dino_lm_ln: Optional[nn.LayerNorm] = None
        self.dino_alpha = nn.Parameter(torch.tensor(float(self.branch_cfg.dino_gate_init)))
        if self.branch_cfg.use_dino_fusion:
            self._init_dino_fusion()

        self.hungarian_cfg = hungarian_cfg or HungarianLossConfig()
        self.loss_cfg = loss_cfg or AdapterLossConfig()
        self._warned_visual_fallback = False
        self._warned_lm_fusion_fail = False
        self._warned_dino_missing_inputs = False

    def _init_dino_fusion(self) -> None:
        model_name = str(self.branch_cfg.dino_model_name).strip()
        if not model_name:
            raise ValueError("branch_cfg.use_dino_fusion=True requires branch_cfg.dino_model_name.")

        dino = None
        load_errors: list[str] = []
        loaders: list[Any] = []
        if Dinov2Model is not None:
            loaders.append(Dinov2Model)
        loaders.append(AutoModel)
        for loader in loaders:
            try:
                dino = loader.from_pretrained(model_name)
                break
            except Exception as exc:  # pragma: no cover - model/env dependent
                load_errors.append(f"{getattr(loader, '__name__', str(loader))}: {exc}")
        if dino is None:
            raise RuntimeError(
                "Failed to load DINO model for fusion. "
                f"model_name={model_name}. Errors:\n" + "\n".join(load_errors)
            )

        dino_hidden = int(getattr(getattr(dino, "config", None), "hidden_size", 0))
        if dino_hidden <= 0:
            raise ValueError("Could not infer DINO hidden size from config.hidden_size.")

        self.dino_model = dino
        self.dino_proj = nn.Linear(dino_hidden, self.hidden_size)
        self.dino_q_ln = nn.LayerNorm(self.hidden_size)
        self.dino_kv_ln = nn.LayerNorm(self.hidden_size)
        self.dino_q_to_d_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=int(self.branch_cfg.dino_cross_attn_heads),
            dropout=float(self.branch_cfg.dino_cross_attn_dropout),
            batch_first=True,
        )
        self.dino_gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.dino_fuse_ln = nn.LayerNorm(self.hidden_size)
        self.dino_fuse_ffn_ln = nn.LayerNorm(self.hidden_size)
        self.dino_fuse_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.dino_lm_ln = nn.LayerNorm(self.hidden_size)
        self.set_dino_trainable(bool(self.branch_cfg.dino_trainable))

    def set_dino_trainable(self, trainable: bool) -> None:
        if self.dino_model is None:
            return
        for p in self.dino_model.parameters():
            p.requires_grad = bool(trainable)

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

    @staticmethod
    def _grid_counts_from_thw(image_grid_thw: Optional[torch.Tensor], bsz: int) -> Optional[list[int]]:
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw):
            return None
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] < 3:
            return None
        if image_grid_thw.shape[0] != bsz:
            return None
        counts: list[int] = []
        for b in range(bsz):
            t, h, w = [int(x) for x in image_grid_thw[b, :3].detach().cpu().tolist()]
            counts.append(max(0, t) * max(0, h) * max(0, w))
        return counts

    def _extract_visual_memory_from_outputs(
        self,
        outputs: Any,
        bsz: int,
        hidden: int,
        counts_from_thw: Optional[list[int]],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not self.branch_cfg.prefer_output_vision_states:
            return None
        candidate_attrs = (
            "image_hidden_states",
            "vision_hidden_states",
            "visual_hidden_states",
        )
        states = None
        for attr in candidate_attrs:
            v = getattr(outputs, attr, None)
            if torch.is_tensor(v):
                states = v
                break
            if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[-1]):
                states = v[-1]
                break
        if states is None:
            return None

        if states.dim() == 3 and states.shape[0] == bsz and states.shape[-1] == hidden:
            memory = states
            mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
            return memory, mask

        if states.dim() == 2 and states.shape[-1] == hidden and counts_from_thw is not None:
            total_needed = int(sum(counts_from_thw))
            if total_needed <= 0 or states.shape[0] < total_needed:
                return None
            max_len = max(counts_from_thw) if counts_from_thw else 0
            memory = states.new_zeros((bsz, max_len, hidden))
            mask = torch.zeros((bsz, max_len), device=states.device, dtype=torch.bool)
            offset = 0
            for b, n in enumerate(counts_from_thw):
                if n <= 0:
                    continue
                memory[b, :n] = states[offset : offset + n]
                mask[b, :n] = True
                offset += n
            return memory, mask

        return None

    def _extract_visual_memory_from_image_features(
        self,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        hidden: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Extract visual memory via model.get_image_features() if available.

        This keeps DETR supervision on vision-side outputs even when the model's
        forward output does not expose `image_hidden_states`.
        """
        if pixel_values is None or not torch.is_tensor(pixel_values):
            return None
        candidate_models: list[Any] = []
        visited: set[int] = set()
        queue: list[Any] = [self.base_model]
        while queue:
            cur = queue.pop(0)
            if cur is None:
                continue
            cur_id = id(cur)
            if cur_id in visited:
                continue
            visited.add(cur_id)
            if hasattr(cur, "get_image_features") and callable(getattr(cur, "get_image_features")):
                candidate_models.append(cur)
            next_objs = []
            if hasattr(cur, "get_base_model"):
                try:
                    next_objs.append(cur.get_base_model())
                except Exception:
                    pass
            for attr in ("model", "base_model"):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    next_objs.append(nxt)
            queue.extend(next_objs)

        if not candidate_models:
            return None

        image_outputs = None
        for m in candidate_models:
            try:
                image_outputs = m.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    return_dict=True,
                )
                break
            except Exception:
                continue
        if image_outputs is None:
            return None

        def _pack_feature_list(feats_list: list[torch.Tensor]) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
            if not feats_list:
                return None
            max_len = max(int(x.shape[0]) for x in feats_list)
            device = feats_list[0].device
            dtype = feats_list[0].dtype
            memory = torch.zeros((len(feats_list), max_len, hidden), device=device, dtype=dtype)
            mask = torch.zeros((len(feats_list), max_len), device=device, dtype=torch.bool)
            for b, x in enumerate(feats_list):
                n = int(x.shape[0])
                if n > 0:
                    memory[b, :n] = x
                    mask[b, :n] = True
            return memory, mask

        def _as_sequence_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
            if not torch.is_tensor(t):
                return None
            if t.dim() == 4 and int(t.shape[-1]) == hidden:
                return t.reshape(int(t.shape[0]), -1, hidden)
            if t.dim() == 3 and int(t.shape[-1]) == hidden:
                return t
            return None

        candidates: list[Any] = []
        if torch.is_tensor(image_outputs):
            candidates.append(image_outputs)
        else:
            for attr in ("last_hidden_state", "pooler_output", "deepstack_features", "hidden_states"):
                v = getattr(image_outputs, attr, None)
                if v is not None:
                    candidates.append(v)

        for cand in candidates:
            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                if torch.is_tensor(cand[-1]) and torch.is_tensor(cand[0]):
                    # tuple(hidden_states) or list(deepstack_features): use the last/highest-level map
                    seq = _as_sequence_tensor(cand[-1])
                    if seq is not None:
                        return seq, torch.ones(seq.shape[:2], device=seq.device, dtype=torch.bool)
                feats_list = []
                valid = True
                for t in cand:
                    if not torch.is_tensor(t):
                        valid = False
                        break
                    if t.dim() == 1 and int(t.shape[0]) == hidden:
                        t = t.unsqueeze(0)
                    elif t.dim() == 3 and int(t.shape[-1]) == hidden and int(t.shape[0]) == 1:
                        t = t.squeeze(0)
                    if t.dim() != 2 or int(t.shape[-1]) != hidden:
                        valid = False
                        break
                    feats_list.append(t)
                if valid:
                    packed = _pack_feature_list(feats_list)
                    if packed is not None:
                        return packed
                continue

            if torch.is_tensor(cand):
                seq = _as_sequence_tensor(cand)
                if seq is not None:
                    return seq, torch.ones(seq.shape[:2], device=seq.device, dtype=torch.bool)
                if cand.dim() == 2 and int(cand.shape[-1]) == hidden:
                    image_token_id = self._infer_image_token_id(input_ids)
                    counts = []
                    for b in range(int(input_ids.shape[0])):
                        counts.append(int((input_ids[b] == image_token_id).sum().item()))
                    total = sum(counts)
                    if total > 0 and int(cand.shape[0]) >= total:
                        max_len = max(counts) if counts else 0
                        memory = cand.new_zeros((int(input_ids.shape[0]), max_len, hidden))
                        mask = torch.zeros((int(input_ids.shape[0]), max_len), device=cand.device, dtype=torch.bool)
                        offset = 0
                        for b, n in enumerate(counts):
                            if n > 0:
                                memory[b, :n] = cand[offset : offset + n]
                                mask[b, :n] = True
                                offset += n
                        return memory, mask

        return None

    def _extract_visual_memory(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        dino_pixel_values: Optional[torch.Tensor] = None,
        outputs: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract visual memory for DETR.

        Preferred path is model-provided vision states. Fallback path (image-token
        masking over language hidden states) is disabled when strict_vision_memory is
        enabled.
        """
        bsz, _, hidden = hidden_states.shape
        counts_from_thw = self._grid_counts_from_thw(image_grid_thw=image_grid_thw, bsz=bsz)

        # Prefer explicit vision states when available from the model output.
        from_outputs = self._extract_visual_memory_from_outputs(
            outputs=outputs,
            bsz=bsz,
            hidden=hidden,
            counts_from_thw=counts_from_thw,
        )
        if from_outputs is not None:
            return from_outputs

        # For Qwen3-VL, explicit image hidden states may not be exposed in outputs;
        # query vision-side features directly from get_image_features().
        from_image_features = self._extract_visual_memory_from_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            hidden=hidden,
        )
        if from_image_features is not None:
            return from_image_features

        from_dino = self._extract_visual_memory_from_dino(
            dino_pixel_values=dino_pixel_values,
        )
        if from_dino is not None:
            return from_dino

        if self.branch_cfg.strict_vision_memory:
            raise RuntimeError(
                "strict_vision_memory=True but no model-provided vision states were found "
                "(expected one of: image_hidden_states / vision_hidden_states / visual_hidden_states "
                "or get_image_features outputs). DETR memory fallback to language hidden states is disabled."
            )

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
            # If grid metadata is available, keep only expected number of visual tokens.
            if counts_from_thw is not None:
                expected = int(counts_from_thw[b])
                if expected > 0 and states.shape[0] >= expected:
                    states = states[:expected]
            seqs.append(states)
            if int(states.shape[0]) > max_len:
                max_len = int(states.shape[0])

        if not self._warned_visual_fallback:
            print(
                "WARNING: using image-token-id visual extraction fallback. "
                "Consider setting image_token_id explicitly or using model-provided vision states."
            )
            self._warned_visual_fallback = True

        memory = hidden_states.new_zeros((bsz, max_len, hidden))
        mask = torch.zeros((bsz, max_len), device=hidden_states.device, dtype=torch.bool)
        for b, states in enumerate(seqs):
            v = states.shape[0]
            memory[b, :v] = states
            mask[b, :v] = True
        return memory, mask

    def _build_grid_coord_tensor(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.branch_cfg.use_grid_pos:
            return None
        bsz, max_len, _ = memory.shape
        counts = self._grid_counts_from_thw(image_grid_thw=image_grid_thw, bsz=bsz)
        if counts is None:
            return None

        coords = memory.new_zeros((bsz, max_len, 3))
        for b in range(bsz):
            t, h, w = [int(x) for x in image_grid_thw[b, :3].detach().cpu().tolist()]
            if t <= 0 or h <= 0 or w <= 0:
                continue
            tt = torch.arange(t, device=memory.device, dtype=memory.dtype)
            yy = torch.arange(h, device=memory.device, dtype=memory.dtype)
            xx = torch.arange(w, device=memory.device, dtype=memory.dtype)
            tg, yg, xg = torch.meshgrid(tt, yy, xx, indexing="ij")
            sample_coords = torch.stack(
                [
                    (tg + 0.5) / float(t),
                    (yg + 0.5) / float(h),
                    (xg + 0.5) / float(w),
                ],
                dim=-1,
            ).reshape(-1, 3)
            n = min(int(sample_coords.shape[0]), int(memory_mask[b].sum().item()))
            if n > 0:
                coords[b, :n] = sample_coords[:n]
        return coords

    def _extract_dino_tokens(self, dino_pixel_values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.dino_model is None or dino_pixel_values is None or not torch.is_tensor(dino_pixel_values):
            return None
        outputs = self.dino_model(
            pixel_values=dino_pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        states = getattr(outputs, "last_hidden_state", None)
        if not torch.is_tensor(states) or states.dim() != 3:
            return None
        if bool(self.branch_cfg.dino_drop_cls_token) and states.shape[1] > 1:
            states = states[:, 1:, :]
        return states

    def _extract_visual_memory_from_dino(
        self,
        dino_pixel_values: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not bool(self.branch_cfg.use_dino_fusion):
            return None
        dino_tokens = self._extract_dino_tokens(dino_pixel_values=dino_pixel_values)
        if dino_tokens is None or self.dino_proj is None:
            return None
        proj_dtype = self.dino_proj.weight.dtype
        memory = self.dino_proj(dino_tokens.to(dtype=proj_dtype))
        mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        return memory, mask

    def _pool_dino_tokens_for_lm(
        self,
        dino_pixel_values: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not bool(self.branch_cfg.inject_dino_tokens_to_lm):
            return None
        dino_tokens = self._extract_dino_tokens(dino_pixel_values=dino_pixel_values)
        if dino_tokens is None or self.dino_proj is None:
            return None
        num_tokens = max(int(self.branch_cfg.dino_lm_num_tokens), 1)
        proj_dtype = self.dino_proj.weight.dtype
        dino_tokens = self.dino_proj(dino_tokens.to(dtype=proj_dtype))
        if dino_tokens.shape[1] != num_tokens:
            pooled = F.adaptive_avg_pool1d(
                dino_tokens.transpose(1, 2),
                output_size=num_tokens,
            ).transpose(1, 2)
        else:
            pooled = dino_tokens
        if self.dino_lm_ln is not None:
            ln_dtype = self.dino_lm_ln.weight.dtype
            pooled_in = pooled if pooled.dtype == ln_dtype else pooled.to(ln_dtype)
            pooled = self.dino_lm_ln(pooled_in)
        return pooled

    def _pool_memory_tokens_for_lm(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        pooled_rows: list[torch.Tensor] = []
        target_tokens = max(int(num_tokens), 1)
        for b in range(int(memory.shape[0])):
            valid = memory[b, memory_mask[b]]
            if valid.numel() == 0:
                valid = memory[b, :1]
            if valid.shape[0] != target_tokens:
                pooled = F.adaptive_avg_pool1d(
                    valid.transpose(0, 1).unsqueeze(0),
                    output_size=target_tokens,
                ).squeeze(0).transpose(0, 1)
            else:
                pooled = valid
            pooled_rows.append(pooled)
        pooled = torch.stack(pooled_rows, dim=0)
        if self.dino_lm_ln is not None:
            ln_dtype = self.dino_lm_ln.weight.dtype
            pooled_in = pooled if pooled.dtype == ln_dtype else pooled.to(ln_dtype)
            pooled = self.dino_lm_ln(pooled_in)
        return pooled

    def _fuse_qwen_memory_with_dino(
        self,
        memory: torch.Tensor,
        dino_pixel_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.dino_model is None or not bool(self.branch_cfg.use_dino_fusion):
            return memory
        dino_tokens = self._extract_dino_tokens(dino_pixel_values=dino_pixel_values)
        if dino_tokens is None:
            if not self._warned_dino_missing_inputs:
                print(
                    "WARNING: DINO fusion is enabled but dino_pixel_values are missing/invalid; "
                    "running DETR branch with Qwen visual memory only."
                )
                self._warned_dino_missing_inputs = True
            return memory
        if dino_tokens.shape[0] != memory.shape[0]:
            if not self._warned_dino_missing_inputs:
                print(
                    "WARNING: DINO fusion batch mismatch "
                    f"(dino={int(dino_tokens.shape[0])}, memory={int(memory.shape[0])}); "
                    "running DETR branch with Qwen visual memory only."
                )
                self._warned_dino_missing_inputs = True
            return memory

        if self.dino_proj is None:
            return memory
        proj_dtype = self.dino_proj.weight.dtype
        dino_tokens = self.dino_proj(dino_tokens.to(dtype=proj_dtype))

        # Run fusion layers in their parameter dtype, then cast back to memory dtype.
        if self.dino_q_to_d_attn is None or self.dino_gate_mlp is None:
            return memory
        attn_dtype = self.dino_q_to_d_attn.in_proj_weight.dtype
        q = memory.to(dtype=attn_dtype)
        d = dino_tokens.to(dtype=attn_dtype)
        if self.dino_q_ln is not None:
            q = self.dino_q_ln(q)
        if self.dino_kv_ln is not None:
            d = self.dino_kv_ln(d)
        d2q, _ = self.dino_q_to_d_attn(query=q, key=d, value=d, need_weights=False)

        gate_in = torch.cat([q, d2q], dim=-1)
        gate = torch.sigmoid(self.dino_gate_mlp(gate_in))
        fused = q + (self.dino_alpha.to(dtype=q.dtype) * gate * d2q)
        if self.dino_fuse_ln is not None:
            fused = self.dino_fuse_ln(fused)
        if self.dino_fuse_ffn is not None:
            ff = fused
            if self.dino_fuse_ffn_ln is not None:
                ff = self.dino_fuse_ffn_ln(ff)
            fused = fused + self.dino_fuse_ffn(ff)

        if fused.dtype != memory.dtype:
            fused = fused.to(memory.dtype)
        return fused

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

    def _inject_token_features_into_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        token_states: torch.Tensor,
        token_id: Optional[int],
        detach_features: bool = False,
    ) -> tuple[Optional[torch.Tensor], int]:
        if input_ids is None or not torch.is_tensor(input_ids):
            return None, 0
        if token_id is None:
            return None, 0
        embed_layer = self.base_model.get_input_embeddings()
        if embed_layer is None:
            return None, 0

        # Avoid in-place writes on a grad-tracked leaf tensor when replacing
        # DET query token slots with visual query states.
        inputs_embeds = embed_layer(input_ids).clone()
        injected = 0
        for b in range(int(input_ids.shape[0])):
            pos = torch.where(input_ids[b] == int(token_id))[0]
            if pos.numel() == 0:
                continue
            n = min(int(pos.numel()), int(token_states.shape[1]))
            if n <= 0:
                continue
            feats = token_states[b, :n]
            if detach_features:
                feats = feats.detach()
            if feats.dtype != inputs_embeds.dtype:
                feats = feats.to(inputs_embeds.dtype)
            inputs_embeds[b, pos[:n]] = feats
            injected += n
        if injected <= 0:
            return None, 0
        return inputs_embeds, injected

    def _inject_det_queries_into_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        query_states: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], int]:
        return self._inject_token_features_into_inputs_embeds(
            input_ids=input_ids,
            token_states=query_states,
            token_id=self.branch_cfg.det_query_token_id,
            detach_features=bool(self.branch_cfg.detach_det_queries_for_lm),
        )

    def _inject_dino_tokens_into_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        dino_pixel_values: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], int]:
        pooled = self._pool_dino_tokens_for_lm(dino_pixel_values=dino_pixel_values)
        if pooled is None:
            return None, 0
        return self._inject_token_features_into_inputs_embeds(
            input_ids=input_ids,
            token_states=pooled,
            token_id=self.branch_cfg.dino_lm_token_id,
            detach_features=bool(self.branch_cfg.detach_dino_tokens_for_lm),
        )

    def _build_fused_visual_tokens_for_lm(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        dino_pixel_values: Optional[torch.Tensor],
        outputs: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if not bool(self.branch_cfg.inject_fused_visual_tokens_to_lm):
            return None
        memory, memory_mask = self._extract_visual_memory(
            hidden_states=hidden_states,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dino_pixel_values=dino_pixel_values,
            outputs=outputs,
        )
        grid_coords = self._build_grid_coord_tensor(
            memory=memory,
            memory_mask=memory_mask,
            image_grid_thw=image_grid_thw,
        )
        if grid_coords is not None:
            mlp_dtype = next(self.grid_pos_mlp.parameters()).dtype
            pos = self.grid_pos_mlp(grid_coords.to(dtype=mlp_dtype))
            if pos.dtype != memory.dtype:
                pos = pos.to(memory.dtype)
            memory = memory + pos
        memory = self._fuse_qwen_memory_with_dino(
            memory=memory,
            dino_pixel_values=dino_pixel_values,
        )
        return self._pool_memory_tokens_for_lm(
            memory=memory,
            memory_mask=memory_mask,
            num_tokens=int(self.branch_cfg.dino_lm_num_tokens),
        )

    def _inject_fused_visual_tokens_into_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        dino_pixel_values: Optional[torch.Tensor],
        outputs: Optional[Any],
    ) -> tuple[Optional[torch.Tensor], int]:
        if input_ids is None or not torch.is_tensor(input_ids):
            return None, 0
        fused_tokens = self._build_fused_visual_tokens_for_lm(
            hidden_states=hidden_states,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dino_pixel_values=dino_pixel_values,
            outputs=outputs,
        )
        if fused_tokens is None:
            return None, 0
        return self._inject_token_features_into_inputs_embeds(
            input_ids=input_ids,
            token_states=fused_tokens,
            token_id=self.branch_cfg.dino_lm_token_id,
            detach_features=bool(self.branch_cfg.detach_fused_visual_tokens_for_lm),
        )

    def _run_base_with_inputs_embeds(
        self,
        base_inputs: dict[str, Any],
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool,
    ) -> Any:
        fused_inputs = dict(base_inputs)
        fused_inputs.pop("input_ids", None)
        fused_inputs["inputs_embeds"] = inputs_embeds
        fused_inputs["output_hidden_states"] = output_hidden_states
        fused_inputs["return_dict"] = True
        fused_inputs["use_cache"] = False
        try:
            return self.base_model(**fused_inputs)
        except Exception as exc:
            fused_inputs.pop("pixel_values", None)
            fused_inputs.pop("image_grid_thw", None)
            if not self._warned_lm_fusion_fail:
                print(
                    "WARNING: visual-token LM fusion with pixel inputs failed; "
                    f"retrying without pixel tensors. error={type(exc).__name__}: {exc}"
                )
                self._warned_lm_fusion_fail = True
            return self.base_model(**fused_inputs)

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
        first_inputs = dict(base_inputs)
        dino_pixel_values = first_inputs.pop("dino_pixel_values", None)
        need_det = det_enabled and (gt_boxes is not None or return_det)
        use_det_lm_fusion = bool(self.branch_cfg.inject_det_queries_to_lm)
        use_dino_lm_fusion = bool(self.branch_cfg.inject_dino_tokens_to_lm) and not use_det_lm_fusion
        use_fused_visual_lm_fusion = (
            bool(self.branch_cfg.inject_fused_visual_tokens_to_lm)
            and not use_det_lm_fusion
            and not use_dino_lm_fusion
        )
        need_visual_memory = bool(need_det or use_fused_visual_lm_fusion)
        if use_det_lm_fusion:
            # LM loss will be computed from a fused second pass after DET query injection.
            first_inputs.pop("labels", None)
        if use_fused_visual_lm_fusion:
            first_inputs.pop("labels", None)
        if use_dino_lm_fusion:
            dino_inputs_embeds, dino_injected = self._inject_dino_tokens_into_inputs_embeds(
                input_ids=base_inputs.get("input_ids"),
                dino_pixel_values=dino_pixel_values,
            )
            result_dino = {
                "lm_dino_token_injected": bool(dino_injected > 0),
                "lm_dino_token_injected_count": int(dino_injected),
            }
            if dino_inputs_embeds is not None:
                first_inputs["inputs_embeds"] = dino_inputs_embeds
        else:
            result_dino = {}
        first_inputs["use_cache"] = False

        if use_dino_lm_fusion and first_inputs.get("inputs_embeds") is not None:
            dino_inputs_embeds = first_inputs.pop("inputs_embeds")
            outputs = self._run_base_with_inputs_embeds(
                base_inputs={k: v for k, v in first_inputs.items() if k != "dino_pixel_values"},
                inputs_embeds=dino_inputs_embeds,
                output_hidden_states=bool(need_visual_memory),
            )
        else:
            outputs = self.base_model(
                output_hidden_states=bool(need_visual_memory),
                return_dict=True,
                **first_inputs,
            )

        result: dict[str, Any] = dict(result_dino)
        lm_loss = getattr(outputs, "loss", None)
        if lm_loss is not None:
            result["lm_loss"] = lm_loss
        hidden = outputs.hidden_states[-1] if need_visual_memory else None
        vision_outputs = outputs if need_visual_memory else None
        del outputs

        det_loss = None
        query_states = None
        if need_det:
            input_ids = base_inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("input_ids is required for auxiliary DETR visual-token extraction.")

            memory, memory_mask = self._extract_visual_memory(
                hidden_states=hidden,
                input_ids=input_ids,
                pixel_values=base_inputs.get("pixel_values"),
                image_grid_thw=base_inputs.get("image_grid_thw"),
                dino_pixel_values=dino_pixel_values,
                outputs=vision_outputs,
            )
            grid_coords = self._build_grid_coord_tensor(
                memory=memory,
                memory_mask=memory_mask,
                image_grid_thw=base_inputs.get("image_grid_thw"),
            )
            if grid_coords is not None:
                # Keep grid-pos MLP math in module dtype (often fp32), then cast
                # back to memory dtype (often bf16) before residual add.
                mlp_dtype = next(self.grid_pos_mlp.parameters()).dtype
                pos = self.grid_pos_mlp(grid_coords.to(dtype=mlp_dtype))
                if pos.dtype != memory.dtype:
                    pos = pos.to(memory.dtype)
                memory = memory + pos
            memory = self._fuse_qwen_memory_with_dino(
                memory=memory,
                dino_pixel_values=dino_pixel_values,
            )
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

        if (
            use_fused_visual_lm_fusion
            and hidden is not None
            and ("labels" in base_inputs)
            and (base_inputs.get("labels") is not None)
        ):
            input_ids = base_inputs.get("input_ids")
            inputs_embeds, injected = self._inject_fused_visual_tokens_into_inputs_embeds(
                input_ids=input_ids,
                hidden_states=hidden,
                pixel_values=base_inputs.get("pixel_values"),
                image_grid_thw=base_inputs.get("image_grid_thw"),
                dino_pixel_values=dino_pixel_values,
                outputs=vision_outputs,
            )
            result["lm_fused_visual_injected"] = bool(injected > 0)
            result["lm_fused_visual_injected_count"] = int(injected)
            if inputs_embeds is not None:
                try:
                    fused_outputs = self._run_base_with_inputs_embeds(
                        base_inputs={
                            k: v
                            for k, v in base_inputs.items()
                            if k != "dino_pixel_values"
                        },
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=False,
                    )
                    lm_loss = getattr(fused_outputs, "loss", lm_loss)
                    if lm_loss is not None:
                        result["lm_loss"] = lm_loss
                    del fused_outputs
                except Exception as exc:
                    if not self._warned_lm_fusion_fail:
                        print(
                            "WARNING: fused-visual-to-LM second pass failed; "
                            f"falling back to base LM path. error={type(exc).__name__}: {exc}"
                        )
                        self._warned_lm_fusion_fail = True

        if (
            use_det_lm_fusion
            and query_states is not None
            and ("labels" in base_inputs)
            and (base_inputs.get("labels") is not None)
        ):
            input_ids = base_inputs.get("input_ids")
            inputs_embeds, injected = self._inject_det_queries_into_inputs_embeds(
                input_ids=input_ids,
                query_states=query_states,
            )
            result["lm_det_query_injected"] = bool(injected > 0)
            result["lm_det_query_injected_count"] = int(injected)
            if inputs_embeds is not None:
                try:
                    fused_outputs = self._run_base_with_inputs_embeds(
                        base_inputs={
                            k: v
                            for k, v in base_inputs.items()
                            if k != "dino_pixel_values"
                        },
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=False,
                    )
                    lm_loss = getattr(fused_outputs, "loss", lm_loss)
                    if lm_loss is not None:
                        result["lm_loss"] = lm_loss
                    del fused_outputs
                except Exception as exc:
                    if not self._warned_lm_fusion_fail:
                        print(
                            "WARNING: DETR-to-LM fusion second pass failed; "
                            f"falling back to base LM path. error={type(exc).__name__}: {exc}"
                        )
                        self._warned_lm_fusion_fail = True

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

    @torch.no_grad()
    def generate_with_visual_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        dino_pixel_values: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate text using configured visual-token injection into LM prompt embeddings."""
        if (
            not self.branch_cfg.inject_det_queries_to_lm
            and not self.branch_cfg.inject_dino_tokens_to_lm
            and not self.branch_cfg.inject_fused_visual_tokens_to_lm
        ):
            gen_inputs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                gen_inputs["attention_mask"] = attention_mask
            if pixel_values is not None:
                gen_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_inputs["image_grid_thw"] = image_grid_thw
            return self.base_model.generate(**gen_inputs, **generate_kwargs)

        if (
            self.branch_cfg.inject_dino_tokens_to_lm
            and not self.branch_cfg.inject_det_queries_to_lm
            and not self.branch_cfg.inject_fused_visual_tokens_to_lm
        ):
            inputs_embeds, injected = self._inject_dino_tokens_into_inputs_embeds(
                input_ids=input_ids,
                dino_pixel_values=dino_pixel_values,
            )
            if inputs_embeds is not None and injected > 0:
                gen_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
                if attention_mask is not None:
                    gen_inputs["attention_mask"] = attention_mask
                if pixel_values is not None:
                    gen_inputs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    gen_inputs["image_grid_thw"] = image_grid_thw
                try:
                    return self.base_model.generate(**gen_inputs, **generate_kwargs)
                except Exception as exc:
                    if not self._warned_lm_fusion_fail:
                        print(
                            "WARNING: DINO direct-fusion generate with pixel inputs failed; "
                            f"retrying without pixel tensors. error={type(exc).__name__}: {exc}"
                        )
                        self._warned_lm_fusion_fail = True
                    gen_inputs.pop("pixel_values", None)
                    gen_inputs.pop("image_grid_thw", None)
                    return self.base_model.generate(**gen_inputs, **generate_kwargs)

        if (
            self.branch_cfg.inject_fused_visual_tokens_to_lm
            and not self.branch_cfg.inject_det_queries_to_lm
        ):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            inputs_embeds, injected = self._inject_fused_visual_tokens_into_inputs_embeds(
                input_ids=input_ids,
                hidden_states=hidden,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                dino_pixel_values=dino_pixel_values,
                outputs=outputs,
            )
            if inputs_embeds is not None and injected > 0:
                gen_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
                if attention_mask is not None:
                    gen_inputs["attention_mask"] = attention_mask
                if pixel_values is not None:
                    gen_inputs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    gen_inputs["image_grid_thw"] = image_grid_thw
                try:
                    return self.base_model.generate(**gen_inputs, **generate_kwargs)
                except Exception as exc:
                    if not self._warned_lm_fusion_fail:
                        print(
                            "WARNING: fused-visual generate with pixel inputs failed; "
                            f"retrying without pixel tensors. error={type(exc).__name__}: {exc}"
                        )
                        self._warned_lm_fusion_fail = True
                    gen_inputs.pop("pixel_values", None)
                    gen_inputs.pop("image_grid_thw", None)
                    return self.base_model.generate(**gen_inputs, **generate_kwargs)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        memory, memory_mask = self._extract_visual_memory(
            hidden_states=hidden,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dino_pixel_values=dino_pixel_values,
            outputs=outputs,
        )
        grid_coords = self._build_grid_coord_tensor(
            memory=memory,
            memory_mask=memory_mask,
            image_grid_thw=image_grid_thw,
        )
        if grid_coords is not None:
            mlp_dtype = next(self.grid_pos_mlp.parameters()).dtype
            pos = self.grid_pos_mlp(grid_coords.to(dtype=mlp_dtype))
            if pos.dtype != memory.dtype:
                pos = pos.to(memory.dtype)
            memory = memory + pos
        memory = self._fuse_qwen_memory_with_dino(
            memory=memory,
            dino_pixel_values=dino_pixel_values,
        )
        query_states = self._decode_queries(memory=memory, memory_mask=memory_mask)
        inputs_embeds, injected = self._inject_det_queries_into_inputs_embeds(
            input_ids=input_ids,
            query_states=query_states,
        )
        if inputs_embeds is None or injected <= 0:
            gen_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                gen_inputs["attention_mask"] = attention_mask
            if pixel_values is not None:
                gen_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_inputs["image_grid_thw"] = image_grid_thw
            return self.base_model.generate(**gen_inputs, **generate_kwargs)

        gen_inputs = {"inputs_embeds": inputs_embeds}
        if attention_mask is not None:
            gen_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            gen_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_inputs["image_grid_thw"] = image_grid_thw
        try:
            return self.base_model.generate(**gen_inputs, **generate_kwargs)
        except Exception as exc:
            if not self._warned_lm_fusion_fail:
                print(
                    "WARNING: DETR-query generate with pixel inputs failed; "
                    f"retrying without pixel tensors. error={type(exc).__name__}: {exc}"
                )
                self._warned_lm_fusion_fail = True
            gen_inputs.pop("pixel_values", None)
            gen_inputs.pop("image_grid_thw", None)
            return self.base_model.generate(**gen_inputs, **generate_kwargs)

    @torch.no_grad()
    def generate_with_det_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        dino_pixel_values: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        return self.generate_with_visual_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            dino_pixel_values=dino_pixel_values,
            **generate_kwargs,
        )
