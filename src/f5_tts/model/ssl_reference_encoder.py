"""SSL speech feature projectors for transcript-free reference conditioning."""

from __future__ import annotations

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
