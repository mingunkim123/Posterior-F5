# Posterior Cache Format

이 문서는 `Posterior-F5`의 ASR posterior cache 형식을 정의한다. cache는 inference, training, evaluation이 같은 ASR 불확실성 정보를 읽도록 하는 공통 인터페이스다.

## 파일 구성

권장 구조:

```text
posterior_cache/
  dev.posterior.jsonl
  posterior_npz/
    utt-0001.npz
    utt-0002.npz
```

`*.posterior.jsonl`은 utterance-level metadata를 저장한다. 큰 frame-level posterior matrix는 `.npz` shard로 분리한다.

## 입력 manifest

`extract_asr_posterior.py`는 JSONL 또는 plain text audio path list를 받는다.

JSONL 예:

```json
{"utterance_id":"utt-0001","audio_path":"data/ref_0001.wav","text":"optional oracle or existing transcript","language":"en"}
```

plain text 예:

```text
data/ref_0001.wav
data/ref_0002.wav
```

plain text 입력에서는 `utterance_id`가 audio file stem으로 자동 생성된다.

## 생성 명령

Whisper 1-best만 저장:

```bash
PYTHONPATH=src python3 src/f5_tts/scripts/extract_asr_posterior.py \
  --manifest manifests/dev.jsonl \
  --output posterior_cache/dev.posterior.jsonl \
  --language en
```

이미 manifest에 있는 `text`를 사용하고 ASR은 건너뛰기:

```bash
PYTHONPATH=src python3 src/f5_tts/scripts/extract_asr_posterior.py \
  --manifest manifests/dev.jsonl \
  --output posterior_cache/dev.posterior.jsonl \
  --skip_whisper
```

CTC top-k posterior shard까지 저장:

```bash
PYTHONPATH=src python3 src/f5_tts/scripts/extract_asr_posterior.py \
  --manifest manifests/dev.jsonl \
  --output posterior_cache/dev.posterior.jsonl \
  --shard_dir posterior_cache/posterior_npz \
  --ctc_model facebook/wav2vec2-base-960h \
  --top_k 8 \
  --language en
```

생성된 cache 검사:

```bash
PYTHONPATH=src python3 src/f5_tts/scripts/inspect_posterior_cache.py \
  --posterior_file posterior_cache/dev.posterior.jsonl \
  --require_frame_posteriors \
  --expected_count 1
```

JSON report가 필요하면 `--json`을 추가한다.

## JSONL schema

각 line은 `PosteriorUtterance` 하나다.

```json
{
  "utterance_id": "utt-0001",
  "audio_path": "data/ref_0001.wav",
  "sample_rate": null,
  "asr_source": "facebook/wav2vec2-base-960h",
  "one_best": "hello world",
  "nbest": [
    {
      "text": "hello world",
      "posterior": 1.0,
      "score": null,
      "tokens": null,
      "token_ids": null,
      "metadata": {"confidence": 1.0}
    }
  ],
  "frame_posteriors": {
    "token_ids": null,
    "probs": null,
    "shard_path": "utt-0001.npz",
    "ids_key": "token_ids",
    "probs_key": "probs",
    "num_frames": 512,
    "top_k": 8,
    "frame_rate": 49.9,
    "sample_rate": 16000,
    "blank_id": 0,
    "source": "facebook/wav2vec2-base-960h",
    "metadata": {
      "duration_sec": 10.26,
      "requested_top_k": 8,
      "stored_top_k": 8,
      "probability_source": "softmax",
      "probability_space": "raw_topk_not_renormalized",
      "topk_mass_min": 0.91,
      "topk_mass_mean": 0.97,
      "topk_mass_max": 0.99,
      "entropy_normalization": "topk_renormalized",
      "entropy_base": "e"
    }
  },
  "token_map": {
    "source": "facebook/wav2vec2-base-960h",
    "tokens": ["<pad>", "a", "b"],
    "token_to_id": {"<pad>": 0, "a": 1, "b": 2},
    "blank_id": 0,
    "filler_id": 0,
    "unk_id": 3,
    "f5_vocab_offset": 1,
    "metadata": {"tokenizer_class": "Wav2Vec2CTCTokenizer"}
  },
  "expected_ref_len": 11.0,
  "mean_entropy": 0.42,
  "duration_sec": 10.26,
  "language": "en",
  "metadata": {
    "input": {},
    "entropy_normalization": "topk_renormalized",
    "topk_probability_space": "raw_topk_not_renormalized"
  }
}
```

## NPZ shard schema

각 `.npz` shard는 기본적으로 두 array를 가진다.

```text
token_ids: int64 array, shape [num_frames, top_k]
probs: float32 array, shape [num_frames, top_k]
```

JSONL의 `frame_posteriors.ids_key`와 `frame_posteriors.probs_key`가 실제 `.npz` key 이름이다. 기본값은 각각 `token_ids`, `probs`다.

`probs`는 full softmax에서 top-k만 잘라 저장한 raw mass다. 따라서 각 row의 합은 보통 1 이하이고, `inspect_posterior_cache.py`는 기본적으로 row sum이 `1.0001`을 넘으면 오류로 본다. soft embedding이나 entropy 계산에서 조건부 top-k 분포가 필요하면 row 안에서 다시 normalize한다.

## Token id 규칙

F5-TTS `TextEmbedding`은 내부에서 hard token id에 `+1`을 적용하고, embedding index `0`을 filler token으로 사용한다.

따라서 soft CTC expected embedding에서는 기본적으로 다음 규칙을 쓴다.

```text
F5 embedding id = posterior token id + f5_vocab_offset
f5_vocab_offset = 1
blank_id 또는 filler_id는 F5 embedding id 0으로 매핑
```

이 규칙은 `PosteriorTokenMap.f5_vocab_offset`, `blank_id`, `filler_id`에 저장된다.

ASR token id를 F5 vocab id로 projection할 때는 `src/f5_tts/posterior/token_projection.py`의 helper를 기준으로 한다.

```text
blank_id 또는 filler_id -> -1
out-of-range token id -> -1
special token like <pad>, <s> -> -1
CTC word delimiter "|" -> " "
exact token lookup -> lowercase lookup -> unknown fallback
```

`-1`은 F5 hard text path에서 filler/padding id로 쓰이고, `TextEmbedding` 안에서 filler embedding index 0으로 처리된다.

## 사용 모드

기존 hard F5 inference:

```bash
f5-tts_infer-cli --ref_text_mode hard
```

expected length만 사용:

```bash
f5-tts_infer-cli \
  --ref_text_mode length_only \
  --posterior_file posterior_cache/dev.posterior.jsonl
```

CTC soft reference embedding 사용:

```bash
f5-tts_infer-cli \
  --ref_text_mode soft_ctc \
  --posterior_file posterior_cache/dev.posterior.jsonl
```

`hard`가 기본값이다. `posterior_file`이 없거나 matching entry가 없으면 hard length로 fallback한다.
