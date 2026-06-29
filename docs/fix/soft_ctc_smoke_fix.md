# Soft CTC Smoke Fix

## Context

`soft_ctc` smoke test는 ASR CTC posterior를 F5-TTS reference-side text conditioning에 넣는 첫 end-to-end 검증이었다. 같은 reference audio와 target text로 아래 세 출력을 비교했다.

```text
tmp/smoke/hard/hard.wav
tmp/smoke/length_only/length_only.wav
tmp/smoke/soft_ctc/soft_ctc.wav
```

기대 동작은 `soft_ctc.wav`가 `hard.wav`와 같은 target sentence를 말하되, reference-side conditioning만 posterior-aware하게 달라지는 것이었다.

## Symptom

초기 `soft_ctc.wav`는 target sentence와 완전히 다른 말을 했다. `length_only.wav`도 한때 매우 이상하게 들렸지만, `soft_ctc`의 문제는 더 명확했다. 이것은 모델 성능 문제가 아니라 implementation bug였다.

## Root Causes

### 1. ASR token ids were treated as F5 token ids

처음 구현은 wav2vec2 CTC posterior의 token id를 F5 vocab id처럼 거의 그대로 사용했다.

그러나 두 vocabulary는 순서와 의미가 다르다.

```text
wav2vec2 CTC:
<pad>, <s>, </s>, <unk>, |, E, T, A, ...

F5 custom vocab:
space, !, ", #, ..., A, B, C, ...
```

따라서 CTC token id에 단순히 `+1` offset을 적용하면 엉뚱한 F5 embedding을 읽게 된다. 이 문제는 commit `687141d`에서 CTC token을 F5 token으로 명시적으로 project하도록 수정했다.

Mapping rule:

```text
CTC blank / special token -> F5 filler
CTC |                     -> F5 space
CTC alphabet              -> matching F5 vocab character
```

### 2. CTC frame positions were used as F5 text positions

두 번째 문제는 더 근본적이었다. F5의 `TextEmbedding` position은 mel frame alignment가 아니다. 실제 구조는 다음에 가깝다.

```text
[ref_text tokens][gen_text tokens][filler padding...]
```

반면 CTC posterior는 frame-level sequence다.

```text
[ctc frame 0][ctc frame 1]...[ctc frame 265]
```

초기 `soft_ctc`는 CTC frame 266개를 F5 embedding 앞쪽 266 position에 그대로 덮어썼다. smoke text의 `ref_text + gen_text`는 약 93 token이므로, 이 방식은 reference text뿐 아니라 target text token region까지 모두 덮어쓴다. 결과적으로 모델이 생성해야 할 문장 조건을 잃어버렸고, 다른 말을 하게 됐다.

이 문제는 commit `b136d13`에서 수정했다.

Current rule:

```text
CTC frame posterior
-> compress to reference-token length
-> blend only into ref_text token region
-> leave gen_text token region untouched
```

### 3. The first smoke used low NFE

초기 smoke는 빠른 확인을 위해 `--nfe_step 8`을 사용했다. 이 값은 crash/smoke 확인에는 유용하지만 음질 판단에는 부족했다. 이후 smoke artifacts는 `--nfe_step 32`로 다시 생성했다.

## Final Fix

현재 `soft_ctc` builder는 다음 순서로 동작한다.

1. Load CTC top-k posterior from JSONL/NPZ.
2. Project ASR token ids into F5 vocab ids.
3. Build normal hard F5 text embedding for `ref_text + gen_text`.
4. Convert `ref_text` through the same pinyin/token path to estimate reference token length.
5. Compress frame-level CTC posterior to that reference token length.
6. Blend compressed posterior embedding only into `hard_embed[:, :ref_token_len, :]`.
7. Keep target text token embeddings unchanged.

The blend is intentionally conservative:

```text
new_ref_embed = 0.8 * hard_ref_embed + 0.2 * soft_ctc_embed
```

This keeps `soft_ctc` useful as a safe smoke baseline while avoiding target text corruption.

## Lessons

1. F5 text conditioning positions must not be confused with mel frames.
2. ASR posterior token ids must always be projected into the F5 tokenizer space.
3. Reference-side soft conditioning must never overwrite target text tokens.
4. `length_only` and `soft_ctc` should be debugged separately.
5. Smoke tests should use `nfe_step 8` only for crash checks; perceptual checks should use `nfe_step 32` or the final evaluation setting.

## Validation

After the fixes, the smoke run generated:

```text
tmp/smoke/hard/hard.wav
tmp/smoke/length_only/length_only.wav
tmp/smoke/soft_ctc/soft_ctc.wav
```

The corrected `soft_ctc.wav` preserves the target sentence instead of producing unrelated speech.

Relevant commits:

```text
687141d Fix soft CTC token projection
b136d13 Keep soft CTC on reference tokens
```
