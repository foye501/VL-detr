"""Qwen3-VL DETR-style loss adapter.

This package adds Hungarian set supervision (objectness + boxes) on top of
token-level hidden states, so it can be combined with standard LM loss.
"""

from .modeling import Qwen3VLDetrAdapter

__all__ = ["Qwen3VLDetrAdapter"]
