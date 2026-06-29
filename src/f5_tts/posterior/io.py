"""I/O helpers for posterior cache manifests and sidecar arrays."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from f5_tts.posterior.schema import PosteriorUtterance, TopKPosterior


def iter_posterior_manifest(path: str | Path) -> Iterator[PosteriorUtterance]:
    """Yield posterior utterances from a JSONL manifest."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {manifest_path}") from exc

            yield PosteriorUtterance.from_dict(data)


def load_posterior_manifest(path: str | Path) -> list[PosteriorUtterance]:
    """Load all posterior utterances from a JSONL manifest."""

    return list(iter_posterior_manifest(path))


def index_posterior_manifest(path: str | Path) -> dict[str, PosteriorUtterance]:
    """Load a JSONL manifest keyed by utterance id."""

    index: dict[str, PosteriorUtterance] = {}
    for utterance in iter_posterior_manifest(path):
        if utterance.utterance_id in index:
            raise ValueError(f"Duplicate utterance_id in posterior manifest: {utterance.utterance_id}")
        index[utterance.utterance_id] = utterance

    return index


def find_posterior_utterance(path: str | Path, utterance_id: str) -> PosteriorUtterance | None:
    """Return one utterance from a JSONL manifest, or None if absent."""

    for utterance in iter_posterior_manifest(path):
        if utterance.utterance_id == utterance_id:
            return utterance

    return None


def resolve_shard_path(frame_posteriors: TopKPosterior, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a posterior sidecar path relative to an optional base directory."""

    if not frame_posteriors.shard_path:
        raise ValueError("frame_posteriors.shard_path is required")

    shard_path = Path(frame_posteriors.shard_path)
    if shard_path.is_absolute() or base_dir is None:
        return shard_path

    resolved = Path(base_dir) / shard_path
    if resolved.exists():
        return resolved

    nested = Path(base_dir) / "posterior_npz" / shard_path.name
    if nested.exists():
        return nested

    return resolved


def load_topk_arrays(
    frame_posteriors: TopKPosterior,
    *,
    base_dir: str | Path | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Load top-k token ids and probabilities from inline data or an NPZ shard."""

    if frame_posteriors.token_ids is not None and frame_posteriors.probs is not None:
        return frame_posteriors.token_ids, frame_posteriors.probs

    shard_path = resolve_shard_path(frame_posteriors, base_dir=base_dir)
    import numpy as np

    with np.load(shard_path) as data:
        token_ids = data[frame_posteriors.ids_key].tolist()
        probs = data[frame_posteriors.probs_key].tolist()

    return token_ids, probs
