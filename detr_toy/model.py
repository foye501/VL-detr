"""Small MLP-backed toy detector with fixed query slots.

This is not a production DETR. It is a lightweight educational scaffold that
implements DETR-like set prediction ideas:
- fixed number of slots (queries)
- one-to-one Hungarian assignment for supervision
- explicit no-object slots
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor


class ToySetDetector:
    def __init__(
        self,
        num_queries: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        learning_rate: float = 2e-3,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.num_queries = int(num_queries)
        self.hidden_dims = hidden_dims
        self.output_dim = self.num_queries * 5  # [obj, cx, cy, w, h] per query
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            batch_size=batch_size,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_step(self, x: np.ndarray, y_slots: np.ndarray) -> None:
        y_flat = y_slots.reshape(len(y_slots), -1).astype(np.float32)
        self.net.partial_fit(x.astype(np.float32), y_flat)
        self._fitted = True

    def predict_raw(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")
        out = self.net.predict(x.astype(np.float32)).reshape(-1, self.num_queries, 5).astype(np.float32)
        obj_logits = out[:, :, 0]
        boxes = np.clip(out[:, :, 1:], 0.0, 1.0)
        return obj_logits, boxes

    def save(self, path: str) -> None:
        payload = {
            "num_queries": self.num_queries,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "model": self.net,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "ToySetDetector":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = ToySetDetector(
            num_queries=payload["num_queries"],
            hidden_dims=tuple(payload["hidden_dims"]),
        )
        obj.output_dim = payload["output_dim"]
        obj.net = payload["model"]
        obj._fitted = True
        return obj
