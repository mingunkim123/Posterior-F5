"""Small posterior-to-text encoder modules."""

from __future__ import annotations

import torch
from torch import nn


class TopKPosteriorEncoder(nn.Module):
    """Encode frame/bin-level top-k posterior rows into F5 text-conditioning space."""

    def __init__(
        self,
        vocab_size: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        f5_vocab_offset: int = 1,
        blank_id: int | None = None,
        filler_id: int | None = None,
    ):
        super().__init__()
        self.f5_vocab_offset = f5_vocab_offset
        self.blank_id = blank_id
        self.filler_id = filler_id

        self.token_embed = nn.Embedding(vocab_size + f5_vocab_offset + 1, hidden_dim)
        self.proj_in = nn.Linear(hidden_dim + 2, hidden_dim)
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(hidden_dim, text_dim)

    def _map_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_ids = token_ids + self.f5_vocab_offset
        excluded_ids = []
        if self.blank_id is not None:
            excluded_ids.append(self.blank_id)
        if self.filler_id is not None:
            excluded_ids.append(self.filler_id)
        if excluded_ids:
            excluded = torch.zeros_like(token_ids, dtype=torch.bool)
            for token_id in excluded_ids:
                excluded |= token_ids == token_id
            embed_ids = embed_ids.masked_fill(excluded, 0)
        return embed_ids

    def forward(
        self,
        token_ids: torch.Tensor,
        probs: torch.Tensor,
        entropy: torch.Tensor | None = None,
        blank_prob: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.shape != probs.shape:
            raise ValueError("token_ids and probs must have the same shape")
        if token_ids.ndim != 3:
            raise ValueError("token_ids and probs must have shape [batch, seq_len, top_k]")

        probs = probs.to(dtype=self.token_embed.weight.dtype).clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = self.token_embed(self._map_token_ids(token_ids.long()))
        expected = (embeds * probs.unsqueeze(-1)).sum(dim=-2)

        batch, seq_len, _ = expected.shape
        if entropy is None:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        if blank_prob is None:
            blank_prob = torch.zeros((batch, seq_len), device=expected.device, dtype=expected.dtype)

        features = torch.cat((expected, entropy.unsqueeze(-1), blank_prob.unsqueeze(-1)), dim=-1)
        hidden = self.proj_in(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        output = self.proj_out(hidden)

        if mask is not None:
            output = output.masked_fill(~mask.unsqueeze(-1), 0.0)
        return output


def distillation_loss(
    posterior_hidden: torch.Tensor,
    oracle_hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE distillation loss against oracle F5 text embeddings."""

    if posterior_hidden.shape != oracle_hidden.shape:
        raise ValueError("posterior_hidden and oracle_hidden must have the same shape")

    loss = (posterior_hidden - oracle_hidden.detach()).pow(2)
    if mask is not None:
        loss = loss.masked_select(mask.unsqueeze(-1))
    return loss.mean()
