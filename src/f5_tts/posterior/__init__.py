"""Posterior-aware reference conditioning utilities for F5-TTS.

Module layout:

- ``schema``           dataclasses for posterior cache JSONL entries
- ``io``               JSONL/NPZ readers for posterior caches
- ``normalize``        probability normalization, top-k, entropy
- ``length``           expected reference length from ASR uncertainty
- ``gating``           entropy/blank gates and hybrid mixing
- ``soft_embedding``   posterior top-k → expected F5 text embedding
- ``token_projection`` ASR token id → F5 vocab id mapping
"""

from f5_tts.posterior.gating import (
    blank_to_text_weight,
    combine_gate,
    entropy_to_text_weight,
    mix_conditions,
)
from f5_tts.posterior.io import (
    find_posterior_utterance,
    index_posterior_manifest,
    iter_posterior_manifest,
    load_posterior_manifest,
    load_topk_arrays,
    resolve_shard_path,
)
from f5_tts.posterior.length import expected_occupancy_len, expected_text_len
from f5_tts.posterior.normalize import (
    entropy,
    normalize_probs,
    normalize_topk_rows,
    temperature_scale_probs,
    top_k_truncate,
)
from f5_tts.posterior.schema import (
    Hypothesis,
    PosteriorTokenMap,
    PosteriorUtterance,
    TopKPosterior,
)
from f5_tts.posterior.soft_embedding import expected_embedding_from_topk
from f5_tts.posterior.token_projection import (
    project_asr_token_to_f5_id,
    project_topk_token_ids,
)


__all__ = [
    "Hypothesis",
    "PosteriorTokenMap",
    "PosteriorUtterance",
    "TopKPosterior",
    "blank_to_text_weight",
    "combine_gate",
    "entropy",
    "entropy_to_text_weight",
    "expected_embedding_from_topk",
    "expected_occupancy_len",
    "expected_text_len",
    "find_posterior_utterance",
    "index_posterior_manifest",
    "iter_posterior_manifest",
    "load_posterior_manifest",
    "load_topk_arrays",
    "mix_conditions",
    "normalize_probs",
    "normalize_topk_rows",
    "project_asr_token_to_f5_id",
    "project_topk_token_ids",
    "resolve_shard_path",
    "temperature_scale_probs",
    "top_k_truncate",
]
