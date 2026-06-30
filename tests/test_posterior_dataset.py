import json

import pytest

torch = pytest.importorskip("torch")

from f5_tts.model.posterior_dataset import PosteriorCacheDataset, collate_posterior_batch
from f5_tts.posterior.schema import PosteriorUtterance, TopKPosterior


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"utterance_id": "utt-001", "audio_path": "audio/ref.wav", "text": "hello"}


def test_posterior_cache_dataset_attaches_manifest_entry(tmp_path):
    manifest = tmp_path / "posterior.jsonl"
    utterance = PosteriorUtterance(
        utterance_id="utt-001",
        audio_path="audio/ref.wav",
        expected_ref_len=5.0,
        mean_entropy=0.25,
    )
    manifest.write_text(json.dumps(utterance.to_dict()) + "\n", encoding="utf-8")

    dataset = PosteriorCacheDataset(ToyDataset(), manifest)
    item = dataset[0]

    assert item["posterior_utterance"] == utterance
    assert item["expected_ref_len"] == 5.0
    assert item["posterior_mean_entropy"] == 0.25


def test_collate_posterior_batch_pads_tensors():
    batch = [
        {"posterior_token_ids": torch.tensor([[1, 2]]), "text": "a"},
        {"posterior_token_ids": torch.tensor([[1, 2], [3, 4]]), "text": "b"},
    ]

    collated = collate_posterior_batch(batch)

    assert collated["posterior_token_ids"].shape == (2, 2, 2)
    assert collated["text"] == ["a", "b"]


def test_posterior_cache_dataset_adds_training_keys_for_frame_posteriors(tmp_path):
    np = pytest.importorskip("numpy")
    shard_dir = tmp_path / "posterior_npz"
    shard_dir.mkdir()
    np.savez(
        shard_dir / "utt-001.npz",
        token_ids=np.array([[1, 0], [2, 0]], dtype=np.int64),
        probs=np.array([[0.7, 0.3], [0.8, 0.2]], dtype=np.float32),
    )
    manifest = tmp_path / "posterior.jsonl"
    utterance = PosteriorUtterance(
        utterance_id="utt-001",
        audio_path="audio/ref.wav",
        one_best="hi",
        frame_posteriors=TopKPosterior(shard_path="utt-001.npz", blank_id=0),
    )
    manifest.write_text(json.dumps(utterance.to_dict()) + "\n", encoding="utf-8")

    dataset = PosteriorCacheDataset(ToyDataset(), manifest, vocab_char_map={"h": 1, "e": 2, "l": 3, "o": 4})
    item = dataset[0]

    assert item["posterior_token_ids"].shape == (2, 2)
    assert item["posterior_mask"].tolist() == [True, True]
    assert item["blank_prob"].tolist() == pytest.approx([0.3, 0.2])
    assert item["seq_len"] == 2
    assert item["oracle_text_tensor"].dtype == torch.long
