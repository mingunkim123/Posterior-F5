"""Shared data structures for ASR posterior caches.

The schema stays intentionally lightweight: large posterior matrices can live in
sidecar ``.npz`` shards, while these dataclasses carry JSON-serializable
metadata that inference, training, and evaluation code can agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class Hypothesis:
    """One ASR hypothesis from an n-best list."""

    text: str
    posterior: float | None = None
    score: float | None = None
    tokens: list[str] | None = None
    token_ids: list[int] | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "text": self.text,
            "posterior": self.posterior,
            "score": self.score,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Hypothesis":
        return cls(
            text=data["text"],
            posterior=data.get("posterior"),
            score=data.get("score"),
            tokens=data.get("tokens"),
            token_ids=data.get("token_ids"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TopKPosterior:
    """Frame/bin-level top-k posterior metadata or inline values."""

    token_ids: list[list[int]] | None = None
    probs: list[list[float]] | None = None
    shard_path: str | None = None
    ids_key: str = "token_ids"
    probs_key: str = "probs"
    num_frames: int | None = None
    top_k: int | None = None
    frame_rate: float | None = None
    sample_rate: int | None = None
    blank_id: int | None = None
    source: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "token_ids": self.token_ids,
            "probs": self.probs,
            "shard_path": self.shard_path,
            "ids_key": self.ids_key,
            "probs_key": self.probs_key,
            "num_frames": self.num_frames,
            "top_k": self.top_k,
            "frame_rate": self.frame_rate,
            "sample_rate": self.sample_rate,
            "blank_id": self.blank_id,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "TopKPosterior | None":
        if data is None:
            return None

        return cls(
            token_ids=data.get("token_ids"),
            probs=data.get("probs"),
            shard_path=data.get("shard_path"),
            ids_key=data.get("ids_key", "token_ids"),
            probs_key=data.get("probs_key", "probs"),
            num_frames=data.get("num_frames"),
            top_k=data.get("top_k"),
            frame_rate=data.get("frame_rate"),
            sample_rate=data.get("sample_rate"),
            blank_id=data.get("blank_id"),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PosteriorTokenMap:
    """Mapping metadata between ASR posterior tokens and F5-TTS tokens."""

    source: str
    tokens: list[str] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)
    blank_id: int | None = None
    filler_id: int | None = None
    unk_id: int | None = None
    f5_vocab_offset: int = 1
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "source": self.source,
            "tokens": self.tokens,
            "token_to_id": self.token_to_id,
            "blank_id": self.blank_id,
            "filler_id": self.filler_id,
            "unk_id": self.unk_id,
            "f5_vocab_offset": self.f5_vocab_offset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict | None) -> "PosteriorTokenMap | None":
        if data is None:
            return None

        return cls(
            source=data["source"],
            tokens=data.get("tokens", []),
            token_to_id=data.get("token_to_id", {}),
            blank_id=data.get("blank_id"),
            filler_id=data.get("filler_id"),
            unk_id=data.get("unk_id"),
            f5_vocab_offset=data.get("f5_vocab_offset", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PosteriorUtterance:
    """Posterior cache entry for one reference utterance."""

    utterance_id: str
    audio_path: str
    sample_rate: int | None = None
    asr_source: str | None = None
    one_best: str | None = None
    nbest: list[Hypothesis] = field(default_factory=list)
    frame_posteriors: TopKPosterior | None = None
    token_map: PosteriorTokenMap | None = None
    expected_ref_len: float | None = None
    mean_entropy: float | None = None
    duration_sec: float | None = None
    language: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "utterance_id": self.utterance_id,
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "asr_source": self.asr_source,
            "one_best": self.one_best,
            "nbest": [hyp.to_dict() for hyp in self.nbest],
            "frame_posteriors": self.frame_posteriors.to_dict() if self.frame_posteriors else None,
            "token_map": self.token_map.to_dict() if self.token_map else None,
            "expected_ref_len": self.expected_ref_len,
            "mean_entropy": self.mean_entropy,
            "duration_sec": self.duration_sec,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PosteriorUtterance":
        return cls(
            utterance_id=data["utterance_id"],
            audio_path=data["audio_path"],
            sample_rate=data.get("sample_rate"),
            asr_source=data.get("asr_source"),
            one_best=data.get("one_best"),
            nbest=[Hypothesis.from_dict(item) for item in data.get("nbest", [])],
            frame_posteriors=TopKPosterior.from_dict(data.get("frame_posteriors")),
            token_map=PosteriorTokenMap.from_dict(data.get("token_map")),
            expected_ref_len=data.get("expected_ref_len"),
            mean_entropy=data.get("mean_entropy"),
            duration_sec=data.get("duration_sec"),
            language=data.get("language"),
            metadata=data.get("metadata", {}),
        )
