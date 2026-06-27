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
    "metadata": {"duration_sec": 10.26}
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
  "duration_sec": null,
  "language": "en",
  "metadata": {"input": {}}
}
```

## NPZ shard schema

각 `.npz` shard는 기본적으로 두 array를 가진다.

```text
token_ids: int64 array, shape [num_frames, top_k]
probs: float32 array, shape [num_frames, top_k]
```

JSONL의 `frame_posteriors.ids_key`와 `frame_posteriors.probs_key`가 실제 `.npz` key 이름이다. 기본값은 각각 `token_ids`, `probs`다.

## Token id 규칙

F5-TTS `TextEmbedding`은 내부에서 hard token id에 `+1`을 적용하고, embedding index `0`을 filler token으로 사용한다.

따라서 soft CTC expected embedding에서는 기본적으로 다음 규칙을 쓴다.

```text
F5 embedding id = posterior token id + f5_vocab_offset
f5_vocab_offset = 1
blank_id 또는 filler_id는 F5 embedding id 0으로 매핑
```

이 규칙은 `PosteriorTokenMap.f5_vocab_offset`, `blank_id`, `filler_id`에 저장된다.

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
