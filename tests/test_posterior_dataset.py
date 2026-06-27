import json

import pytest

torch = pytest.importorskip("torch")

from f5_tts.model.posterior_dataset import PosteriorCacheDataset, collate_posterior_batch
from f5_tts.posterior.schema import PosteriorUtterance


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
