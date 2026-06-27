"""Dataset wrappers that attach posterior cache entries to F5-TTS samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from f5_tts.posterior.io import index_posterior_manifest, load_topk_arrays


class PosteriorCacheDataset(Dataset):
    """Wrap an existing dataset and attach posterior arrays by audio path or utterance id."""

    def __init__(self, base_dataset: Dataset, posterior_manifest: str | Path):
        self.base_dataset = base_dataset
        self.posterior_manifest = Path(posterior_manifest)
        self.posterior_index = index_posterior_manifest(self.posterior_manifest)
        self.posterior_base_dir = self.posterior_manifest.parent

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _lookup_key_candidates(self, item: dict[str, Any]) -> list[str]:
        candidates = []
        for key in ("utterance_id", "id", "audio_path"):
            if key in item and item[key] is not None:
                value = str(item[key])
                candidates.extend([value, Path(value).name, Path(value).stem])
        return candidates

    def _find_posterior(self, item: dict[str, Any]):
        for key in self._lookup_key_candidates(item):
            if key in self.posterior_index:
                return self.posterior_index[key]
        raise KeyError(f"No posterior entry found for sample keys: {self._lookup_key_candidates(item)}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        posterior = self._find_posterior(item)
        item["posterior_utterance"] = posterior
        item["expected_ref_len"] = posterior.expected_ref_len
        item["posterior_mean_entropy"] = posterior.mean_entropy

        if posterior.frame_posteriors is not None:
            token_ids, probs = load_topk_arrays(posterior.frame_posteriors, base_dir=self.posterior_base_dir)
            item["posterior_token_ids"] = torch.tensor(token_ids, dtype=torch.long)
            item["posterior_probs"] = torch.tensor(probs, dtype=torch.float32)

        return item


def collate_posterior_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal collate helper for posterior distillation batches."""

    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            collated[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        else:
            collated[key] = values
    return collated
