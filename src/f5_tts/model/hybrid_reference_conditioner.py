"""Hybrid soft-text plus SSL reference conditioners."""

from __future__ import annotations

import torch
from torch import nn

from f5_tts.posterior.gating import combine_gate, mix_conditions


class HybridReferenceConditioner(nn.Module):
    """Mix soft text and SSL reference features with an entropy-aware gate."""

    def __init__(self, entropy_threshold: float = 1.0, blank_threshold: float = 0.5):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.blank_threshold = blank_threshold

    def forward(
        self,
        soft_text: torch.Tensor,
        ssl: torch.Tensor,
        *,
        alpha: torch.Tensor | float | None = None,
        entropy: torch.Tensor | None = None,
        blank_prob: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if soft_text.shape != ssl.shape:
            raise ValueError("soft_text and ssl conditions must have the same shape")

        if alpha is None:
            alpha = combine_gate(
                entropy=entropy,
                blank_prob=blank_prob,
                entropy_threshold=self.entropy_threshold,
                blank_threshold=self.blank_threshold,
            )
        if alpha is None:
            alpha = 0.5

        mixed = mix_conditions(soft_text, ssl, alpha)
        if mask is not None:
            mixed = mixed.masked_fill(~mask.unsqueeze(-1), 0.0)
        return mixed
