"""SSL speech feature projectors and SSL feature cache I/O.

This module groups:

- ``SSLReferenceProjector`` / ``WavLMReferenceEncoder`` — project SSL frame
  features into the F5 text-conditioning dimension.
- Cache helpers (``iter_ssl_cache`` / ``index_ssl_cache`` /
  ``resolve_ssl_feature_path`` / ``load_ssl_feature_array``) for the JSONL +
  ``.npz`` cache written by ``scripts/extract_ssl_reference.py``.
- ``cached_ssl_condition`` — convenience used by the inference path and tests
  to wrap a cache entry into a projected condition tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SSLReferenceProjector(nn.Module):
    """Project SSL frame features into the F5 text-conditioning dimension."""

    def __init__(self, ssl_dim: int, text_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(ssl_dim, text_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(ssl_dim),
            nn.Linear(ssl_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, ssl_features: torch.Tensor, *, target_len: int | None = None, mask: torch.Tensor | None = None):
        if ssl_features.ndim != 3:
            raise ValueError("ssl_features must have shape [batch, seq_len, ssl_dim]")

        projected = self.net(ssl_features)
        if target_len is not None and projected.shape[1] != target_len:
            projected = F.interpolate(projected.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
            projected = projected.transpose(1, 2)

        if mask is not None:
            projected = projected.masked_fill(~mask.unsqueeze(-1), 0.0)
        return projected


class WavLMReferenceEncoder(nn.Module):
    """Thin wrapper around a Hugging Face SSL encoder plus an F5 projector."""

    def __init__(self, ssl_model: nn.Module, projector: SSLReferenceProjector):
        super().__init__()
        self.ssl_model = ssl_model
        self.projector = projector

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        text_dim: int,
        ssl_dim: int | None = None,
        freeze_ssl: bool = True,
    ) -> "WavLMReferenceEncoder":
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_id)
        ssl_model = AutoModel.from_pretrained(model_id)
        inferred_ssl_dim = ssl_dim or getattr(config, "hidden_size", None)
        if inferred_ssl_dim is None:
            raise ValueError("Could not infer SSL hidden size; pass ssl_dim explicitly")

        if freeze_ssl:
            for param in ssl_model.parameters():
                param.requires_grad = False

        return cls(ssl_model, SSLReferenceProjector(inferred_ssl_dim, text_dim))

    def forward(self, input_values: torch.Tensor, *, target_len: int | None = None, attention_mask=None):
        outputs = self.ssl_model(input_values=input_values, attention_mask=attention_mask)
        return self.projector(outputs.last_hidden_state, target_len=target_len)


# ---------------------------------------------------------------------------
# SSL feature cache I/O
# ---------------------------------------------------------------------------


def iter_ssl_cache(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def index_ssl_cache(path: str | Path) -> dict[str, dict]:
    return {str(row["utterance_id"]): row for row in iter_ssl_cache(path)}


def resolve_ssl_feature_path(entry: dict, *, base_dir: str | Path | None = None) -> Path:
    feature_path = Path(entry["feature_path"])
    if feature_path.is_absolute() or base_dir is None:
        return feature_path

    resolved = Path(base_dir) / feature_path
    if resolved.exists():
        return resolved
    nested = Path(base_dir) / "ssl_npz" / feature_path.name
    if nested.exists():
        return nested
    return resolved


def load_ssl_feature_array(entry: dict, *, base_dir: str | Path | None = None) -> torch.Tensor:
    import numpy as np

    path = resolve_ssl_feature_path(entry, base_dir=base_dir)
    with np.load(path) as data:
        features = data[entry.get("feature_key", "features")]
    return torch.tensor(features, dtype=torch.float32)


def cached_ssl_condition(
    entry: dict,
    *,
    text_dim: int,
    target_len: int,
    base_dir: str | Path | None = None,
    device=None,
) -> torch.Tensor:
    """Wrap one cache entry into a projected condition tensor.

    A fresh ``SSLReferenceProjector`` is constructed with random weights — this
    is intended for the inference smoke path and shape-only tests, where the
    projector trained alongside the F5 backbone would normally be injected.
    """

    features = load_ssl_feature_array(entry, base_dir=base_dir).unsqueeze(0)
    if device is not None:
        features = features.to(device)
    projector = SSLReferenceProjector(features.shape[-1], text_dim).to(features.device)
    projector.eval()
    with torch.inference_mode():
        return projector(features, target_len=target_len)
