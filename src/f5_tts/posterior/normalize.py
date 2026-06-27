"""Probability normalization helpers for ASR posterior features."""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalize_probs(probs: Sequence[float], *, fallback_uniform: bool = True) -> list[float]:
    """Clamp negative values and normalize a probability row."""

    if not probs:
        return []

    clipped = [max(float(prob), 0.0) for prob in probs]
    total = sum(clipped)
    if total <= 0.0:
        if not fallback_uniform:
            return [0.0 for _ in clipped]
        return [1.0 / len(clipped)] * len(clipped)

    return [prob / total for prob in clipped]


def temperature_scale_probs(probs: Sequence[float], temperature: float) -> list[float]:
    """Apply probability-space temperature scaling and renormalize."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    normalized = normalize_probs(probs)
    if not normalized:
        return []

    inv_temperature = 1.0 / temperature
    scaled = [prob**inv_temperature for prob in normalized]
    return normalize_probs(scaled)


def top_k_truncate(
    token_ids: Sequence[int],
    probs: Sequence[float],
    *,
    k: int,
    renormalize: bool = True,
) -> tuple[list[int], list[float]]:
    """Keep the top-k token/probability pairs sorted by probability."""

    if k <= 0:
        raise ValueError("k must be positive")

    if len(token_ids) != len(probs):
        raise ValueError("token_ids and probs must have the same length")

    pairs = sorted(zip(token_ids, probs), key=lambda item: float(item[1]), reverse=True)[:k]
    kept_ids = [int(token_id) for token_id, _ in pairs]
    kept_probs = [float(prob) for _, prob in pairs]
    if renormalize:
        kept_probs = normalize_probs(kept_probs)

    return kept_ids, kept_probs


def normalize_topk_rows(
    token_id_rows: Sequence[Sequence[int]],
    prob_rows: Sequence[Sequence[float]],
    *,
    temperature: float | None = None,
    top_k: int | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Normalize optional top-k posterior rows while preserving token ids."""

    if len(token_id_rows) != len(prob_rows):
        raise ValueError("token_id_rows and prob_rows must have the same number of rows")

    normalized_ids: list[list[int]] = []
    normalized_probs: list[list[float]] = []
    for token_ids, probs in zip(token_id_rows, prob_rows):
        ids = [int(token_id) for token_id in token_ids]
        row = [float(prob) for prob in probs]
        if len(ids) != len(row):
            raise ValueError("token id and probability rows must have matching lengths")

        row = normalize_probs(row)
        if temperature is not None:
            row = temperature_scale_probs(row, temperature)
        if top_k is not None:
            ids, row = top_k_truncate(ids, row, k=top_k)

        normalized_ids.append(ids)
        normalized_probs.append(row)

    return normalized_ids, normalized_probs


def entropy(probs: Sequence[float], *, base: float | None = None) -> float:
    """Compute entropy for a probability row."""

    normalized = normalize_probs(probs, fallback_uniform=False)
    value = 0.0
    for prob in normalized:
        if prob > 0.0:
            value -= prob * math.log(prob)

    if base is not None:
        if base <= 0.0 or base == 1.0:
            raise ValueError("base must be positive and not equal to 1")
        value /= math.log(base)

    return value

