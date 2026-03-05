"""Qwen3-VL DETR-style loss adapter.

This package adds Hungarian set supervision (objectness + boxes) on top of
token-level hidden states, so it can be combined with standard LM loss.
"""

try:
    from .modeling import Qwen3VLDetrAdapter, Qwen3VLAuxDetrAdapter
except Exception:
    # Allow utility modules (e.g., dataset generation) to run in environments
    # where torch/transformers model deps are unavailable.
    Qwen3VLDetrAdapter = None
    Qwen3VLAuxDetrAdapter = None

__all__ = ["Qwen3VLDetrAdapter", "Qwen3VLAuxDetrAdapter"]
