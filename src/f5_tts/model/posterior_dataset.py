"""Dataset wrappers that attach posterior cache entries to F5-TTS samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from f5_tts.posterior.io import index_posterior_manifest, load_topk_arrays


class PosteriorCacheDataset(Dataset):
    """Wrap an existing dataset and attach posterior arrays by audio path or utterance id."""

    def __init__(self, base_dataset: Dataset, posterior_manifest: str | Path, vocab_char_map: dict[str, int] | None = None):
        self.base_dataset = base_dataset
        self.posterior_manifest = Path(posterior_manifest)
        self.posterior_index = index_posterior_manifest(self.posterior_manifest)
        self.posterior_base_dir = self.posterior_manifest.parent
        self.vocab_char_map = vocab_char_map

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
        text = item.get("text") or posterior.one_best or ""
        item.setdefault("oracle_text_tensor", self._text_to_tensor(text, item))

        if posterior.frame_posteriors is not None:
            token_ids, probs = load_topk_arrays(posterior.frame_posteriors, base_dir=self.posterior_base_dir)
            item["posterior_token_ids"] = torch.tensor(token_ids, dtype=torch.long)
            item["posterior_probs"] = torch.tensor(probs, dtype=torch.float32)
            item["seq_len"] = item["posterior_token_ids"].shape[0]
            item["posterior_mask"] = torch.ones(item["seq_len"], dtype=torch.bool)
            blank_id = posterior.frame_posteriors.blank_id
            if blank_id is None and posterior.token_map is not None:
                blank_id = posterior.token_map.blank_id
            if blank_id is None:
                item["blank_prob"] = torch.zeros(item["seq_len"], dtype=torch.float32)
            else:
                item["blank_prob"] = (item["posterior_probs"] * (item["posterior_token_ids"] == blank_id)).sum(dim=-1)

        return item

    def _text_to_tensor(self, text: str, item: dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(item.get("oracle_text_tensor")):
            return item["oracle_text_tensor"].long()
        if torch.is_tensor(item.get("text_token_ids")):
            return item["text_token_ids"].long()
        if isinstance(item.get("text_token_ids"), list):
            return torch.tensor(item["text_token_ids"], dtype=torch.long)
        if self.vocab_char_map is not None:
            return torch.tensor([self.vocab_char_map.get(char, 0) for char in str(text)], dtype=torch.long)
        return torch.tensor([ord(char) for char in str(text)], dtype=torch.long)


def collate_posterior_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal collate helper for posterior distillation batches."""

    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            padding_value = False if values[0].dtype == torch.bool else 0
            collated[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding_value)
        elif isinstance(values[0], int):
            collated[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], float):
            collated[key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated[key] = values
    return collated
