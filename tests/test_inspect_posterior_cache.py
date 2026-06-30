import json
import subprocess
import sys
from pathlib import Path

import pytest

from f5_tts.posterior.schema import PosteriorUtterance, TopKPosterior
from f5_tts.scripts.inspect_posterior_cache import inspect_posterior_cache


SCRIPT = Path("src/f5_tts/scripts/inspect_posterior_cache.py")


def _write_manifest(path, utterances):
    with path.open("w", encoding="utf-8") as file:
        for utterance in utterances:
            file.write(json.dumps(utterance.to_dict()) + "\n")


def test_inspect_posterior_cache_accepts_valid_npz_shard(tmp_path):
    np = pytest.importorskip("numpy")
    shard_dir = tmp_path / "posterior_npz"
    shard_dir.mkdir()
    np.savez(
        shard_dir / "utt-001.npz",
        token_ids=np.array([[0, 1], [1, 2]], dtype=np.int64),
        probs=np.array([[0.7, 0.2], [0.6, 0.3]], dtype=np.float32),
    )
    manifest = tmp_path / "posterior.jsonl"
    _write_manifest(
        manifest,
        [
            PosteriorUtterance(
                utterance_id="utt-001",
                audio_path="ref.wav",
                expected_ref_len=12.0,
                frame_posteriors=TopKPosterior(
                    shard_path="utt-001.npz",
                    num_frames=2,
                    top_k=2,
                    blank_id=0,
                ),
            )
        ],
    )

    report = inspect_posterior_cache(manifest, require_frame_posteriors=True, expected_count=1)

    assert report["ok"] is True
    assert report["num_utterances"] == 1
    assert report["utterances"][0]["topk_mass_mean"] == pytest.approx(0.9)


def test_inspect_posterior_cache_reports_missing_shard(tmp_path):
    manifest = tmp_path / "posterior.jsonl"
    _write_manifest(
        manifest,
        [
            PosteriorUtterance(
                utterance_id="utt-001",
                audio_path="ref.wav",
                expected_ref_len=12.0,
                frame_posteriors=TopKPosterior(shard_path="missing.npz", num_frames=2, top_k=2),
            )
        ],
    )

    report = inspect_posterior_cache(manifest, require_frame_posteriors=True)

    assert report["ok"] is False
    assert any("missing shard" in error or "failed to load" in error for error in report["errors"])


def test_inspect_posterior_cache_reports_shape_mismatch(tmp_path):
    np = pytest.importorskip("numpy")
    np.savez(
        tmp_path / "utt-001.npz",
        token_ids=np.array([[0, 1], [1, 2]], dtype=np.int64),
        probs=np.array([[0.7, 0.2]], dtype=np.float32),
    )
    manifest = tmp_path / "posterior.jsonl"
    _write_manifest(
        manifest,
        [
            PosteriorUtterance(
                utterance_id="utt-001",
                audio_path="ref.wav",
                expected_ref_len=12.0,
                frame_posteriors=TopKPosterior(shard_path="utt-001.npz", num_frames=2, top_k=2),
            )
        ],
    )

    report = inspect_posterior_cache(manifest)

    assert report["ok"] is False
    assert any("token_ids rows" in error for error in report["errors"])


def test_inspect_posterior_cache_cli_exits_nonzero_on_error(tmp_path):
    manifest = tmp_path / "posterior.jsonl"
    _write_manifest(manifest, [PosteriorUtterance(utterance_id="utt-001", audio_path="ref.wav")])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--posterior_file",
            str(manifest),
            "--require_frame_posteriors",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing frame_posteriors" in result.stdout
