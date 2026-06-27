"""Length estimates derived from ASR uncertainty."""

from __future__ import annotations

from collections.abc import Sequence

from f5_tts.posterior.schema import Hypothesis


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    total = float(sum(max(weight, 0.0) for weight in weights))
    if total <= 0.0:
        if not weights:
            return []
        return [1.0 / len(weights)] * len(weights)

    return [max(weight, 0.0) / total for weight in weights]


def expected_text_len(
    hypotheses: Sequence[Hypothesis | str],
    *,
    encoding: str = "utf-8",
    use_bytes: bool = True,
) -> float | None:
    """Return the posterior-weighted expected text length for an n-best list.

    F5-TTS currently estimates generation duration from UTF-8 byte length, so
    ``use_bytes`` defaults to True. If no hypothesis is provided, None is
    returned so callers can fall back to the hard transcript length.
    """

    if not hypotheses:
        return None

    texts: list[str] = []
    weights: list[float] = []
    for hypothesis in hypotheses:
        if isinstance(hypothesis, Hypothesis):
            texts.append(hypothesis.text)
            weights.append(1.0 if hypothesis.posterior is None else hypothesis.posterior)
        else:
            texts.append(hypothesis)
            weights.append(1.0)

    normalized = _normalize_weights(weights)
    lengths = [len(text.encode(encoding)) if use_bytes else len(text) for text in texts]
    return sum(weight * length for weight, length in zip(normalized, lengths))


def expected_occupancy_len(
    probs: Sequence[Sequence[float]],
    token_ids: Sequence[Sequence[int]] | None = None,
    *,
    blank_id: int | None = None,
    filler_id: int | None = None,
    normalize_rows: bool = False,
) -> float | None:
    """Estimate non-blank token occupancy from frame/bin posterior rows.

    If ``token_ids`` are provided, probability mass assigned to ``blank_id`` or
    ``filler_id`` is excluded. If no ids are provided, each row is treated as
    total non-blank occupancy unless row normalization is requested.
    """

    if not probs:
        return None

    expected_len = 0.0
    excluded_ids = {token_id for token_id in (blank_id, filler_id) if token_id is not None}

    for row_index, prob_row in enumerate(probs):
        row = [max(float(prob), 0.0) for prob in prob_row]
        row_total = sum(row)
        if row_total <= 0.0:
            continue

        scale = 1.0 / row_total if normalize_rows else 1.0
        if token_ids is None:
            expected_len += row_total * scale
            continue

        id_row = token_ids[row_index]
        if len(id_row) != len(row):
            raise ValueError("token_ids and probs must have matching row lengths")

        non_blank_mass = sum(prob for token_id, prob in zip(id_row, row) if token_id not in excluded_ids)
        expected_len += non_blank_mass * scale

    return expected_len

