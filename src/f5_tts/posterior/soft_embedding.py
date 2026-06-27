"""Convert posterior token distributions into F5 text embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from f5_tts.posterior.normalize import normalize_topk_rows


def expected_embedding_from_topk(
    token_ids: Sequence[Sequence[int]] | torch.Tensor,
    probs: Sequence[Sequence[float]] | torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    blank_id: int | None = None,
    filler_id: int | None = None,
    f5_vocab_offset: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Map top-k posterior rows to expected F5 text embeddings.

    ``embedding_weight`` is the F5 ``TextEmbedding.text_embed.weight`` matrix,
    where index 0 is the filler token. Non-blank ASR/F5 vocab ids are shifted by
    ``f5_vocab_offset`` by default, matching the existing F5 ``text + 1`` path.
    """

    if not torch.is_tensor(token_ids):
        token_ids, probs = normalize_topk_rows(token_ids, probs) if normalize else (token_ids, probs)

    id_tensor = torch.as_tensor(token_ids, device=embedding_weight.device, dtype=torch.long)
    prob_tensor = torch.as_tensor(probs, device=embedding_weight.device, dtype=embedding_weight.dtype)
    if id_tensor.shape != prob_tensor.shape:
        raise ValueError("token_ids and probs must have the same shape")
    if id_tensor.ndim < 2:
        raise ValueError("token_ids and probs must have shape [..., top_k]")

    if normalize and torch.is_tensor(token_ids):
        row_sum = prob_tensor.clamp_min(0).sum(dim=-1, keepdim=True)
        uniform = torch.full_like(prob_tensor, 1.0 / prob_tensor.shape[-1])
        prob_tensor = torch.where(row_sum > 0, prob_tensor.clamp_min(0) / row_sum.clamp_min(1e-12), uniform)

    embed_ids = id_tensor + f5_vocab_offset
    excluded_ids = []
    if blank_id is not None:
        excluded_ids.append(blank_id)
    if filler_id is not None:
        excluded_ids.append(filler_id)
    if excluded_ids:
        excluded_mask = torch.zeros_like(id_tensor, dtype=torch.bool)
        for token_id in excluded_ids:
            excluded_mask |= id_tensor == token_id
        embed_ids = embed_ids.masked_fill(excluded_mask, 0)

    if embed_ids.max().item() >= embedding_weight.shape[0] or embed_ids.min().item() < 0:
        raise ValueError("mapped token ids are outside the F5 embedding range")

    token_embeds = embedding_weight[embed_ids]
    return (token_embeds * prob_tensor.unsqueeze(-1)).sum(dim=-2)

