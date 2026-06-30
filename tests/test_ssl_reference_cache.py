import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from f5_tts.model.ssl_reference_encoder import cached_ssl_condition, index_ssl_cache, load_ssl_feature_array


SCRIPT = Path("src/f5_tts/scripts/extract_ssl_reference.py")


def test_extract_ssl_reference_synthetic_cache(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"utterance_id": "utt-001", "ref_audio": "ref.wav"}) + "\n", encoding="utf-8")
    output = tmp_path / "ssl.jsonl"
    shard_dir = tmp_path / "ssl_npz"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--shard_dir",
            str(shard_dir),
            "--synthetic_dim",
            "3",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    entry = index_ssl_cache(output)["utt-001"]
    features = load_ssl_feature_array(entry, base_dir=output.parent)

    assert entry["ssl_dim"] == 3
    assert features.shape == (4, 3)


def test_cached_ssl_condition_projects_to_target_length(tmp_path):
    np = pytest.importorskip("numpy")
    shard_dir = tmp_path / "ssl_npz"
    shard_dir.mkdir()
    np.savez(shard_dir / "utt-001.npz", features=np.ones((4, 3), dtype=np.float32))
    entry = {"utterance_id": "utt-001", "feature_path": "utt-001.npz", "feature_key": "features"}

    condition = cached_ssl_condition(entry, text_dim=5, target_len=7, base_dir=tmp_path)

    assert condition.shape == (1, 7, 5)
