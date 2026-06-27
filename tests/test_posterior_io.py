import json

import pytest

from f5_tts.posterior.io import (
    find_posterior_utterance,
    index_posterior_manifest,
    load_posterior_manifest,
    load_topk_arrays,
    resolve_shard_path,
)
from f5_tts.posterior.schema import Hypothesis, PosteriorUtterance, TopKPosterior


def _write_manifest(path, utterances):
    with path.open("w", encoding="utf-8") as file:
        for utterance in utterances:
            file.write(json.dumps(utterance.to_dict()) + "\n")


def test_load_posterior_manifest_and_find_by_utterance_id(tmp_path):
    manifest_path = tmp_path / "posterior.jsonl"
    utterances = [
        PosteriorUtterance(
            utterance_id="utt-001",
            audio_path="audio/one.wav",
            one_best="hello",
            nbest=[Hypothesis(text="hello", posterior=0.9)],
        ),
        PosteriorUtterance(
            utterance_id="utt-002",
            audio_path="audio/two.wav",
            one_best="world",
        ),
    ]
    _write_manifest(manifest_path, utterances)

    loaded = load_posterior_manifest(manifest_path)
    found = find_posterior_utterance(manifest_path, "utt-002")

    assert loaded == utterances
    assert found is not None
    assert found.one_best == "world"
    assert find_posterior_utterance(manifest_path, "missing") is None


def test_index_posterior_manifest_rejects_duplicate_ids(tmp_path):
    manifest_path = tmp_path / "posterior.jsonl"
    _write_manifest(
        manifest_path,
        [
            PosteriorUtterance(utterance_id="dup", audio_path="audio/one.wav"),
            PosteriorUtterance(utterance_id="dup", audio_path="audio/two.wav"),
        ],
    )

    with pytest.raises(ValueError):
        index_posterior_manifest(manifest_path)


def test_load_topk_arrays_prefers_inline_values():
    posterior = TopKPosterior(
        token_ids=[[1, 2]],
        probs=[[0.75, 0.25]],
        shard_path="unused.npz",
    )

    token_ids, probs = load_topk_arrays(posterior)

    assert token_ids == [[1, 2]]
    assert probs == [[0.75, 0.25]]


def test_load_topk_arrays_from_npz_shard(tmp_path):
    np = pytest.importorskip("numpy")

    shard_path = tmp_path / "utt-001.npz"
    np.savez(
        shard_path,
        token_ids=np.array([[1, 2], [3, 4]], dtype=np.int64),
        probs=np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
    )
    posterior = TopKPosterior(
        shard_path=shard_path.name,
        ids_key="token_ids",
        probs_key="probs",
    )

    token_ids, probs = load_topk_arrays(posterior, base_dir=tmp_path)

    assert token_ids == [[1, 2], [3, 4]]
    assert probs[0] == pytest.approx([0.8, 0.2])
    assert probs[1] == pytest.approx([0.6, 0.4])


def test_resolve_shard_path_requires_shard_path():
    with pytest.raises(ValueError):
        resolve_shard_path(TopKPosterior())
