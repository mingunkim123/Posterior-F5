# Posterior-F5 Reproducibility Checklist

이 체크리스트는 Posterior-F5 논문용 run을 freeze할 때 함께 보관할 항목이다.

## Code

| 항목 | 값 |
| --- | --- |
| git commit |  |
| branch | `feature/posterior-aware-f5` |
| dirty working tree | yes / no |
| Python |  |
| PyTorch |  |
| CUDA / driver |  |
| GPU |  |

## Data

| 항목 | 값 |
| --- | --- |
| manifest path |  |
| manifest sha256 |  |
| num utterances |  |
| subsets |  |
| dataset version/license |  |
| reference audio root |  |

Required manifest fields:

```text
utterance_id
ref_audio
ref_text
gen_text
text
subset
language
```

## Posterior Cache

| 항목 | 값 |
| --- | --- |
| posterior JSONL |  |
| posterior JSONL sha256 |  |
| posterior ASR |  |
| CTC model |  |
| top_k |  |
| inspect command | `PYTHONPATH=src python src/f5_tts/scripts/inspect_posterior_cache.py --posterior_file <jsonl> --require_frame_posteriors` |
| inspect result | pass / fail |

## Generation

| 항목 | 값 |
| --- | --- |
| run_id |  |
| artifact_root |  |
| modes | `hard`, `oracle`, `length_only`, `soft_ctc` |
| Stage 2 modes | `posterior_encoder`, `hybrid` |
| F5 checkpoint id |  |
| F5 checkpoint hash |  |
| posterior encoder checkpoint |  |
| SSL cache |  |
| vocoder |  |
| seed |  |
| nfe_step |  |
| cfg_strength |  |
| sway_sampling_coef |  |
| speed |  |

## Evaluation

| 항목 | 값 |
| --- | --- |
| eval ASR |  |
| eval ASR differs from posterior source | yes / no |
| metric summary |  |
| per-utterance metrics |  |
| bootstrap output |  |
| paper table CSV |  |
| paper table Markdown |  |
| qualitative examples |  |

## Freeze Criteria

1. `git status -sb` is clean except ignored artifact directories.
2. `run.json` records the intended commit, seed, modes, checkpoint, posterior source, and eval ASR.
3. Every manifest row has a generated wav and prediction row for every frozen mode.
4. `metrics/summary.json`, `metrics/per_utterance.jsonl`, paper CSV/Markdown, and bootstrap JSON exist.
5. Stage 1 and Stage 2 runs use separate `run_id`s.
