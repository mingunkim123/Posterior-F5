# Posterior-Aware F5-TTS 구현 계획

## 1. 목표

이 프로젝트의 목표는 기존 F5-TTS의 reference transcript 의존성을 줄이는 개인 확장 시스템을 구현하는 것이다. 핵심 아이디어는 ASR의 1-best transcript를 그대로 hard text condition으로 쓰는 대신, ASR n-best, confusion network, CTC posterior, posterior entropy를 이용해 reference side를 soft text condition으로 바꾸는 것이다.

최종 시스템의 추천 형태는 다음과 같다.

> posterior-aware latent text encoder + expected-length correction + entropy-aware SSL fallback

target text는 사용자의 명시적 제어 신호이므로 hard text로 유지한다. 바꾸는 대상은 reference audio에서 얻는 reference text conditioning 경로다.

## 2. 현재 F5-TTS 코드에서 중요한 지점

현재 코드 기준으로 reference transcript가 영향을 주는 곳은 크게 두 곳이다.

| 위치 | 역할 | 변경 필요성 |
| --- | --- | --- |
| `src/f5_tts/infer/utils_infer.py`의 `preprocess_ref_audio_text()` | reference text가 없으면 Whisper ASR 1-best를 생성 | ASR posterior/n-best 추출 경로 추가 필요 |
| `src/f5_tts/infer/utils_infer.py`의 `infer_process()` | `len(ref_text)`로 chunk 크기 추정 | expected reference length로 대체 가능 |
| `src/f5_tts/infer/utils_infer.py`의 `infer_batch_process()` | `ref_text + gen_text`를 만들고 duration을 문자 길이 비율로 계산 | length-only baseline과 soft condition 주입 지점 |
| `src/f5_tts/model/cfm.py`의 `CFM.sample()` | `text`를 받아 transformer에 전달하고 duration 하한을 `(text != -1).sum()`으로 계산 | soft text embedding 또는 structured text condition 입력 허용 필요 |
| `src/f5_tts/model/backbones/dit.py`의 `TextEmbedding` | 정수 token id를 embedding으로 변환 | posterior expected embedding 또는 posterior encoder 출력을 우회 주입해야 함 |
| `src/f5_tts/model/dataset.py`와 `trainer.py` | 학습 데이터는 `mel_spec`, `text`만 collate | posterior cache와 oracle distillation target을 학습 batch에 넣어야 함 |

현재 F5-TTS는 `TextEmbedding.forward(text, seq_len)`에서 integer token만 받는다. 따라서 soft conditioning을 구현하려면 `text_embed_override` 같은 우회 경로를 만들거나, `text`를 구조체로 확장해 transformer가 hard text와 soft embedding을 모두 받을 수 있게 해야 한다.

## 3. Git 작업 계획

실제 구현 시작 전 새 feature branch를 만든다.

```bash
cd /Users/mingun/Posterior-F5/F5-TTS
git switch -c feature/posterior-aware-f5
```

문서만 추가하는 현재 작업은 `main`에 남아 있어도 되지만, 실제 코드 구현은 별도 branch에서 진행하는 것이 좋다. 구현 중 생성되는 ASR posterior cache, checkpoint, 평가 음성, W&B 로그는 git에 넣지 않는다.

필요하면 `.gitignore`에 아래 경로를 추가한다.

```gitignore
posterior_cache/
experiments/
eval_outputs/
wandb/
ckpts/posterior_*
```

## 4. 전체 아키텍처

권장 파이프라인은 다음 순서다.

```text
reference audio
  -> ASR posterior extractor
  -> posterior normalization / calibration
  -> expected reference length estimator
  -> posterior-to-text module
  -> soft reference text condition

reference audio
  -> optional SSL encoder
  -> SSL reference condition

soft reference text condition + optional SSL condition
  -> entropy-aware gate
  -> F5 text-conditioning space

target text
  -> 기존 hard text encoder

reference mel + target text + reference condition
  -> F5 / DiT flow-matching backbone
  -> vocoder
  -> generated audio
```

초기 구현에서는 F5 checkpoint 재사용을 최우선으로 둔다. backbone 전체를 바로 재학습하지 않고, inference-time baseline과 작은 adapter부터 검증한다.

## 5. 구현 단계

### Phase 0. 기준선 고정

목표는 수정 전 F5-TTS baseline을 명확히 고정하는 것이다.

작업:

1. `F5-hard-1best`: 현재 코드 그대로, reference text가 없을 때 Whisper 1-best 사용.
2. `F5-oracle`: gold reference transcript 사용.
3. `F5-manual`: 사용자가 직접 고친 reference transcript 사용.
4. 평가용 입력 manifest 형식 정의.
5. baseline output과 metric을 `experiments/baselines/` 아래에 저장.

새 파일 후보:

```text
src/f5_tts/eval/eval_posterior_f5.py
src/f5_tts/eval/error_breakdown.py
configs/eval/posterior_f5_baseline.yaml
```

성공 조건:

F5 기존 추론 결과가 재현되고, 이후 soft 방식과 비교할 수 있는 WER/CER, Sub/Del/Ins, speaker similarity, RTF 측정 경로가 생겨야 한다.

### Phase 1. ASR posterior/n-best 추출 파이프라인

목표는 reference audio마다 hard transcript뿐 아니라 uncertainty 정보를 저장하는 것이다.

권장 입력 형태:

| source | 산출물 | 용도 |
| --- | --- | --- |
| Whisper | 1-best, optional n-best/segment confidence | 현실적인 robust ASR baseline |
| CTC ASR | frame-level posteriorgram | soft CTC expected embedding |
| n-best aligner | confusion network | WCN baseline |

새 파일 후보:

```text
src/f5_tts/posterior/__init__.py
src/f5_tts/posterior/schema.py
src/f5_tts/posterior/extract_asr.py
src/f5_tts/posterior/normalize.py
src/f5_tts/posterior/confnet.py
src/f5_tts/posterior/calibration.py
src/f5_tts/scripts/extract_asr_posterior.py
```

`schema.py`에는 다음 데이터 구조를 둔다.

```python
PosteriorUtterance:
    utterance_id: str
    audio_path: str
    sample_rate: int
    asr_source: str
    one_best: str
    nbest: list[Hypothesis]
    frame_posteriors: Optional[TopKPosterior]
    token_map: PosteriorTokenMap
    expected_ref_len: float
    mean_entropy: float
```

저장 형식은 처음에는 JSONL + NumPy `.npz` 조합을 추천한다. frame-level posterior는 커질 수 있으므로 JSON에 직접 넣지 않는다.

성공 조건:

하나의 reference audio에 대해 `1-best`, `n-best`, `top-k posterior`, `entropy`, `expected_ref_len`을 cache로 재사용할 수 있어야 한다.

### Phase 2. Expected-length correction baseline

목표는 모델 embedding을 바꾸기 전에 duration 추정만 uncertainty-aware하게 바꿔 효과를 분리하는 것이다.

현재 duration 계산은 아래 구조다.

```python
ref_text_len = len(ref_text.encode("utf-8"))
gen_text_len = len(gen_text.encode("utf-8"))
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
```

이를 posterior expected length로 바꾸는 baseline을 추가한다.

```python
expected_ref_text_len = sum(p_i * len(h_i.encode("utf-8")) for h_i, p_i in nbest)
duration = ref_audio_len + int(ref_audio_len / expected_ref_text_len * gen_text_len / local_speed)
```

CTC posterior를 쓸 때는 blank/filler를 제외한 expected occupancy를 길이로 쓴다.

수정 후보:

```text
src/f5_tts/infer/utils_infer.py
src/f5_tts/infer/infer_cli.py
src/f5_tts/api.py
src/f5_tts/socket_server.py
```

새 파일 후보:

```text
src/f5_tts/posterior/length.py
tests/test_posterior_length.py
```

CLI 옵션 후보:

```bash
--ref_text_mode hard|length_only|soft_nbest|soft_ctc|posterior_encoder|hybrid
--posterior_file posterior_cache/dev.jsonl
--expected_ref_len auto
```

성공 조건:

`length_only`가 기존 hard 1-best 대비 deletion과 word skipping을 줄이는지 확인한다. 이 baseline은 반드시 남겨야 한다. 나중에 soft embedding이 좋아져도, 개선 원인이 duration인지 representation인지 분해하는 데 필요하다.

### Phase 3. 학습 없는 soft text baseline

목표는 F5 checkpoint를 그대로 쓰면서 soft text condition만 주입하는 것이다.

#### 3.1 n-best weighted baseline

가장 쉬운 방법은 각 n-best hypothesis를 기존 F5 text encoder에 통과시키고 hidden state를 가중 평균하는 것이다.

```text
H_ref = sum_i p_i * TextEncoder(h_i)
```

단점은 hypothesis별 길이와 위치가 달라 평균 위치가 어긋날 수 있다는 점이다. 따라서 최종 모델이 아니라 sanity check로 둔다.

#### 3.2 CTC expected embedding baseline

frame-level posterior에서 각 time step의 expected embedding을 만든다.

```text
e_t = sum_c P(c | t) * Embedding(c)
```

blank posterior는 F5의 filler token에 매핑한다. F5의 `TextEmbedding`은 내부에서 `text + 1`을 하고 0을 filler로 쓰므로, posterior vocabulary와 F5 vocabulary의 offset 규칙을 명확히 해야 한다.

수정 후보:

```text
src/f5_tts/model/backbones/dit.py
src/f5_tts/model/backbones/unett.py
src/f5_tts/model/cfm.py
src/f5_tts/infer/utils_infer.py
```

새 파일 후보:

```text
src/f5_tts/posterior/soft_embedding.py
tests/test_soft_text_embedding.py
```

권장 코드 방향:

1. `TextEmbedding.forward()`는 기존 hard text 경로를 유지한다.
2. `DiT.get_input_embed()`에 `text_embed_override=None`을 추가한다.
3. override가 있으면 `self.text_embed(text, ...)` 대신 override를 사용한다.
4. `CFM.sample()`에 `text_embed_override` 또는 `text_condition` 인자를 추가한다.
5. 기존 CLI와 API는 아무 옵션을 주지 않으면 기존과 완전히 같은 결과를 내야 한다.

성공 조건:

soft mode를 꺼도 기존 hard inference가 깨지지 않아야 한다. soft CTC mode에서는 training 없이도 일부 noisy/accented prompt에서 deletion 감소나 oracle gap 회복이 보여야 한다.

### Phase 4. Learned posterior encoder

목표는 posterior matrix를 F5 text-conditioning space로 직접 사상하는 작은 encoder를 학습하는 것이다.

입력 feature:

| feature | 설명 |
| --- | --- |
| `topk_token_ids` | 각 frame/bin의 top-k token |
| `topk_probs` | top-k posterior probability |
| `entropy` | uncertainty scalar |
| `blank_prob` | CTC blank 또는 filler mass |
| `asr_hidden` | 가능하면 ASR encoder hidden state |

출력:

```text
H_post: [batch, seq_len, text_dim]
```

teacher:

```text
H_oracle = 기존 F5 TextEmbedding(oracle reference text)
```

loss:

```text
L_kd = mse(H_post, stopgrad(H_oracle))
L_aux_len = smooth_l1(expected_len_pred, oracle_ref_len)
L_entropy_reg = optional confidence regularizer
```

새 파일 후보:

```text
src/f5_tts/model/posterior_encoder.py
src/f5_tts/model/posterior_dataset.py
src/f5_tts/train/train_posterior.py
src/f5_tts/configs/F5TTS_v1_Base_Posterior.yaml
tests/test_posterior_encoder_shapes.py
```

학습 순서:

1. F5 backbone freeze.
2. posterior encoder만 distillation으로 학습.
3. posterior encoder + text-side adapter 또는 LoRA만 fine-tune.
4. 필요할 때만 DiT 일부 block을 제한적으로 unfreeze.

성공 조건:

posterior encoder가 expected embedding보다 안정적으로 좋아야 한다. 특히 ASR entropy가 중간 정도인 prompt에서 hard 1-best 대비 WER/CER와 deletion이 개선되어야 한다.

### Phase 5. Entropy-aware SSL hybrid

목표는 ASR posterior가 너무 불확실한 구간에서 transcript-free SSL feature를 fallback으로 쓰는 것이다.

하이브리드 representation:

```text
H_ref = alpha_t * H_soft_text + (1 - alpha_t) * H_ssl
```

`alpha_t`는 posterior entropy, blank mass, utterance-level confidence로 계산한다.

초기 gate:

```text
alpha_t = sigmoid(a * (entropy_threshold - entropy_t))
```

이후 learned gate로 확장한다.

새 파일 후보:

```text
src/f5_tts/model/ssl_reference_encoder.py
src/f5_tts/model/hybrid_reference_conditioner.py
src/f5_tts/posterior/gating.py
tests/test_entropy_gate.py
```

추가 dependency 후보:

```text
transformers
```

`transformers`는 이미 `pyproject.toml`에 있으므로 WavLM 계열 encoder를 붙이기 쉽다. 다만 실제 모델 weight와 feature cache는 git에 넣지 않는다.

성공 조건:

severe dysarthria나 ASR confidence가 낮은 noisy prompt에서 soft text only보다 hybrid가 좋아야 한다. RTFree-F5를 항상 이길 필요는 없지만, hybrid가 가장 안정적인 운영 모드라는 결론을 목표로 한다.

## 6. 평가 계획

비교군은 최소한 아래를 포함한다.

| 이름 | reference conditioning | duration | 목적 |
| --- | --- | --- | --- |
| `F5-hard-1best` | ASR 1-best | 1-best length | 현실 baseline |
| `F5-oracle` | gold transcript | oracle length | ceiling |
| `F5-length-only` | 1-best text | posterior expected length | duration 효과 분리 |
| `F5-soft-nbest` | n-best weighted text | expected length | 쉬운 soft baseline |
| `F5-soft-ctc` | CTC expected embedding | posterior length | 강한 학습 없는 baseline |
| `F5-post-encoder` | learned posterior encoder | learned/expected length | 핵심 제안법 |
| `RTFree-F5` | SSL reference feature | transcript-free | 강한 비교축 |
| `Hybrid-soft+SSL` | soft text + SSL gate | learned/expected length | 최종 후보 |

데이터 축:

| 축 | 추천 |
| --- | --- |
| clean | LibriTTS, VCTK |
| accented | L2-ARCTIC, Common Voice accent subset |
| dysarthric | UASpeech, TORGO |
| noisy | clean prompt에 noise/RIR augmentation |
| Korean sanity check | 공개 한국어 ASR/TTS subset 또는 Emilia Korean subset |

지표:

| 지표 | 이유 |
| --- | --- |
| WER/CER | intelligibility |
| Sub/Del/Ins | failure mode 분해 |
| deletion rate | duration/alignment 개선 확인 |
| repetition indicator | hard sentence 안정성 |
| speaker similarity cosine | 화자 보존 |
| UTMOS 또는 자동 품질 점수 | 빠른 sweep |
| MOS/CMOS/SMOS | 최종 주관 평가 |
| latency/RTF | 실전 비용 |

주의할 점:

posterior를 만든 ASR와 평가 ASR를 분리한다. 예를 들어 posterior source가 CTC 모델이면 평가 WER는 Whisper 계열로, posterior source가 Whisper이면 별도 CTC/Conformer 평가 모델로 교차 확인한다.

통계:

| 대상 | 검정 |
| --- | --- |
| WER/CER, Sub/Del/Ins | paired bootstrap resampling |
| 여러 모델 비교 | Holm 보정 또는 FDR 보정 |
| speaker cosine | paired permutation test |
| MOS/SMOS | mixed-effects model 또는 Wilcoxon signed-rank |

## 7. MVP 범위

가장 먼저 완성할 MVP는 아래까지만 한다.

1. `feature/posterior-aware-f5` branch 생성.
2. ASR posterior cache schema 작성.
3. n-best 기반 expected reference length 계산.
4. `F5-length-only` CLI 옵션 추가.
5. CTC top-k posterior를 F5 vocab으로 매핑하는 함수 작성.
6. `text_embed_override` 경로를 DiT와 CFM에 추가.
7. `F5-soft-ctc` inference-only baseline 구현.
8. hard mode regression test와 soft embedding shape test 작성.
9. clean + noisy small subset에서 WER/CER와 deletion 비교.

MVP에서 아직 하지 않는 것:

1. backbone full fine-tuning.
2. SSL hybrid.
3. 대규모 주관 평가.
4. 한국어 tokenization 전체 최적화.

MVP 성공 기준:

| 항목 | 기준 |
| --- | --- |
| hard compatibility | 기존 hard inference 옵션이 그대로 동작 |
| length-only | deletion 또는 duration mismatch가 줄어드는 subset이 있음 |
| soft-ctc | hard 1-best 대비 악화 없이 일부 noisy/accented prompt에서 개선 |
| engineering | posterior cache를 재사용하고 CLI에서 mode 선택 가능 |

## 8. 예상 파일 변경 목록

최소 구현:

```text
src/f5_tts/posterior/__init__.py
src/f5_tts/posterior/schema.py
src/f5_tts/posterior/length.py
src/f5_tts/posterior/normalize.py
src/f5_tts/posterior/soft_embedding.py
src/f5_tts/scripts/extract_asr_posterior.py
src/f5_tts/model/backbones/dit.py
src/f5_tts/model/cfm.py
src/f5_tts/infer/utils_infer.py
src/f5_tts/infer/infer_cli.py
src/f5_tts/api.py
tests/test_posterior_length.py
tests/test_soft_text_embedding.py
tests/test_hard_infer_compat.py
```

학습형 확장:

```text
src/f5_tts/model/posterior_encoder.py
src/f5_tts/model/posterior_dataset.py
src/f5_tts/train/train_posterior.py
src/f5_tts/configs/F5TTS_v1_Base_Posterior.yaml
tests/test_posterior_encoder_shapes.py
```

hybrid 확장:

```text
src/f5_tts/model/ssl_reference_encoder.py
src/f5_tts/model/hybrid_reference_conditioner.py
src/f5_tts/posterior/gating.py
tests/test_entropy_gate.py
```

평가 확장:

```text
src/f5_tts/eval/eval_posterior_f5.py
src/f5_tts/eval/error_breakdown.py
configs/eval/posterior_f5_baseline.yaml
configs/eval/posterior_f5_full.yaml
```

문서:

```text
docs/md/posterior_aware_f5_tts_implementation_plan.md
docs/md/posterior_cache_format.md
docs/md/experiment_protocol.md
```

## 9. 주요 리스크와 대응

| 리스크 | 설명 | 대응 |
| --- | --- | --- |
| posterior가 너무 퍼짐 | expected embedding이 애매한 평균 벡터가 되어 content가 흐려질 수 있음 | top-k truncation, temperature sharpening, entropy clipping |
| duration 이득과 representation 이득이 섞임 | soft text가 좋아 보이지만 사실 length correction 때문일 수 있음 | `length-only` baseline 유지 |
| ASR vocab과 F5 vocab mismatch | BPE/subword posterior를 character/pinyin vocab으로 바꿀 때 오류 발생 | 명시적 projection table과 unit test 작성 |
| hard path regression | 기존 F5 사용자가 쓰던 CLI/API가 깨질 수 있음 | soft 인자는 optional로 두고 hard regression test 작성 |
| severe dysarthria에서 soft text 한계 | ASR posterior 자체가 정보적으로 무너질 수 있음 | entropy-aware SSL fallback |
| cache 크기 증가 | frame-level posterior가 매우 큼 | top-k posterior, fp16 저장, utterance shard |

## 10. 추천 진행 순서

가장 좋은 순서는 아래다.

1. branch 생성과 baseline 재현.
2. posterior cache schema와 extractor 작성.
3. length-only baseline 구현.
4. CTC expected embedding 구현.
5. DiT/CFM에 `text_embed_override` 경로 추가.
6. soft CTC inference-only baseline 평가.
7. posterior encoder distillation.
8. adapter/LoRA fine-tuning.
9. SSL hybrid.
10. 전체 평가와 통계 검정.

이 순서의 장점은 실패해도 배울 수 있는 결과가 남는다는 점이다. length-only가 강하면 duration correction 논문 포인트가 되고, soft CTC가 강하면 training 없이 reference uncertainty를 쓰는 실용 포인트가 된다. 학습형 posterior encoder가 추가로 개선되면 핵심 제안법이 되고, severe case에서 SSL hybrid가 강하면 최신 transcript-free 흐름과도 연결된다.

## 11. 최종 성공 기준

| 구간 | 성공 기준 |
| --- | --- |
| clean | 기존 F5-hard-1best 대비 유의미한 성능 저하 없음 |
| accented/noisy | WER/CER 상대 8-15% 개선 또는 oracle gap 30% 이상 회복 |
| dysarthric | WER/CER 상대 5-10% 개선, deletion 10-20% 감소 |
| hard sentence | deletion 감소, insertion 급증 없음 |
| speaker similarity | baseline 대비 유의미한 하락 없음 |
| severe case | soft text only가 부족하면 hybrid가 가장 안정적이어야 함 |

이 프로젝트의 핵심 기여는 "F5-TTS에 n-best를 넣었다"가 아니다. 기여는 reference transcript uncertainty가 F5의 text conditioning과 duration estimation을 어떻게 망가뜨리는지 분리하고, hard transcript와 transcript-free SSL 사이의 중간 설계 공간을 구현 및 검증하는 것이다.

## 12. 한 번에 한 파일만 바꾸는 구현 스텝

아래 순서는 실제 구현할 때 작은 commit 단위로 따라가기 위한 체크리스트다. 원칙은 단순하다.

1. 한 step에서는 파일 하나만 새로 만들거나 수정한다.
2. 같은 파일을 여러 번 수정해도 되지만, 각 수정은 별도 step으로 둔다.
3. 가능한 한 각 step 뒤에 import test, unit test, smoke test 중 하나를 실행한다.
4. hard F5 inference 기본 동작은 매 단계에서 깨지지 않아야 한다.

### 준비

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 1 | 없음 | `git switch -c feature/posterior-aware-f5`로 작업 branch를 만든다. 이미 있으면 `git switch feature/posterior-aware-f5`만 한다. | `git status --short --branch` |
| Step 2 | `.gitignore` | posterior cache, 실험 출력, checkpoint 출력 경로를 ignore한다. | `git diff -- .gitignore` |

### posterior 기본 패키지

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 3 | `src/f5_tts/posterior/__init__.py` | 빈 posterior package를 만든다. 처음에는 import side effect가 없게 둔다. | `python -c "import f5_tts.posterior"` |
| Step 4 | `src/f5_tts/posterior/schema.py` | `Hypothesis`, `TopKPosterior`, `PosteriorUtterance`, `PosteriorTokenMap` dataclass를 정의한다. | `python -m compileall src/f5_tts/posterior/schema.py` |
| Step 5 | `tests/test_posterior_schema.py` | schema 생성, JSON 직렬화용 dict 변환, 필수 필드 검증 테스트를 추가한다. | `pytest tests/test_posterior_schema.py` |
| Step 6 | `src/f5_tts/posterior/length.py` | n-best 기반 `expected_text_len()`과 CTC occupancy 기반 `expected_occupancy_len()`을 구현한다. | `python -m compileall src/f5_tts/posterior/length.py` |
| Step 7 | `tests/test_posterior_length.py` | n-best 확률 정규화, 빈 후보, blank 제외 occupancy 계산 테스트를 추가한다. | `pytest tests/test_posterior_length.py` |
| Step 8 | `src/f5_tts/posterior/normalize.py` | temperature scaling, top-k truncation, probability renormalization 함수를 구현한다. | `python -m compileall src/f5_tts/posterior/normalize.py` |
| Step 9 | `tests/test_posterior_normalize.py` | top-k 후 합이 1이 되는지, temperature가 entropy를 바꾸는지 테스트한다. | `pytest tests/test_posterior_normalize.py` |
| Step 10 | `src/f5_tts/posterior/io.py` | JSONL manifest와 `.npz` posterior shard를 읽는 최소 loader를 만든다. | `python -m compileall src/f5_tts/posterior/io.py` |
| Step 11 | `tests/test_posterior_io.py` | 임시 JSONL/NPZ를 만들어 utterance id로 posterior를 찾는 테스트를 추가한다. | `pytest tests/test_posterior_io.py` |

### length-only baseline

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 12 | `src/f5_tts/infer/utils_infer.py` | `infer_process()`에 optional `expected_ref_text_len=None` 인자를 추가한다. 기본값이면 기존 동작과 같게 둔다. | `python -m compileall src/f5_tts/infer/utils_infer.py` |
| Step 13 | `src/f5_tts/infer/utils_infer.py` | `infer_process()`의 `max_chars` 계산에서 `expected_ref_text_len`이 있으면 `len(ref_text)` 대신 사용한다. | 기존 CLI help 실행 |
| Step 14 | `src/f5_tts/infer/utils_infer.py` | `infer_batch_process()`에 optional `expected_ref_text_len=None` 인자를 추가한다. | `python -m compileall src/f5_tts/infer/utils_infer.py` |
| Step 15 | `src/f5_tts/infer/utils_infer.py` | duration 계산에서 `expected_ref_text_len`이 있으면 `ref_text_len` 대신 사용한다. 0 이하 값은 fallback한다. | 짧은 inference smoke test |
| Step 16 | `src/f5_tts/infer/infer_cli.py` | `--ref_text_mode hard|length_only`와 `--posterior_file` CLI 옵션을 추가한다. 아직 동작 연결은 하지 않는다. | `python src/f5_tts/infer/infer_cli.py --help` |
| Step 17 | `src/f5_tts/infer/infer_cli.py` | `posterior_file`을 읽어서 현재 `ref_audio`의 `expected_ref_len`을 찾는 코드를 연결한다. | `python -m compileall src/f5_tts/infer/infer_cli.py` |
| Step 18 | `src/f5_tts/infer/infer_cli.py` | `infer_process()` 호출에 `expected_ref_text_len`을 넘긴다. `hard` mode에서는 항상 `None`을 넘긴다. | hard mode smoke test |
| Step 19 | `tests/test_hard_infer_compat.py` | `infer_process()`와 `infer_batch_process()`의 새 인자가 기본값에서 기존 호출을 깨지 않는지 테스트한다. | `pytest tests/test_hard_infer_compat.py` |

### soft embedding 기반 MVP

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 20 | `src/f5_tts/posterior/soft_embedding.py` | posterior token id/prob에서 expected embedding을 만드는 순수 함수를 구현한다. | `python -m compileall src/f5_tts/posterior/soft_embedding.py` |
| Step 21 | `tests/test_soft_text_embedding.py` | hard one-hot posterior가 기존 embedding lookup과 같은 결과를 내는지 테스트한다. | `pytest tests/test_soft_text_embedding.py` |
| Step 22 | `src/f5_tts/model/backbones/dit.py` | `DiT.get_input_embed()`에 optional `text_embed_override=None` 인자를 추가한다. 기본값은 기존 경로다. | `python -m compileall src/f5_tts/model/backbones/dit.py` |
| Step 23 | `src/f5_tts/model/backbones/dit.py` | override가 들어오면 `self.text_embed()` 대신 override를 쓰되, cache와 CFG uncond 경로는 기존처럼 유지한다. | hard inference smoke test |
| Step 24 | `src/f5_tts/model/backbones/dit.py` | `DiT.forward()`에 `text_embed_override=None` 인자를 추가하고 `get_input_embed()`로 전달한다. | `python -m compileall src/f5_tts/model/backbones/dit.py` |
| Step 25 | `src/f5_tts/model/cfm.py` | `CFM.sample()`에 optional `text_embed_override=None` 인자를 추가한다. | `python -m compileall src/f5_tts/model/cfm.py` |
| Step 26 | `src/f5_tts/model/cfm.py` | `self.transformer(...)` 호출에 `text_embed_override`를 전달한다. 기본값이면 기존 결과가 같아야 한다. | hard inference smoke test |
| Step 27 | `tests/test_soft_text_override_shapes.py` | 임의 tensor override가 DiT 입력 길이와 text_dim을 맞출 때 forward shape가 맞는지 테스트한다. | `pytest tests/test_soft_text_override_shapes.py` |
| Step 28 | `src/f5_tts/infer/utils_infer.py` | `infer_batch_process()`에 optional `text_embed_override_builder=None` 인자를 추가한다. 아직 사용하지 않는다. | `python -m compileall src/f5_tts/infer/utils_infer.py` |
| Step 29 | `src/f5_tts/infer/utils_infer.py` | `_infer_basic()` 안에서 builder가 있으면 gen_text별 soft text override를 만들고 `model_obj.sample()`에 넘긴다. | hard mode smoke test |
| Step 30 | `src/f5_tts/infer/infer_cli.py` | `--ref_text_mode soft_ctc` 값을 허용한다. 아직 posterior builder 연결은 최소 stub로 둔다. | CLI help 확인 |
| Step 31 | `src/f5_tts/infer/infer_cli.py` | `soft_ctc` mode에서 posterior loader와 soft embedding builder를 연결한다. | 작은 posterior fixture로 dry run |

### ASR posterior 추출

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 32 | `src/f5_tts/scripts/extract_asr_posterior.py` | manifest를 읽고 output JSONL을 쓰는 CLI skeleton을 만든다. | `python src/f5_tts/scripts/extract_asr_posterior.py --help` |
| Step 33 | `src/f5_tts/scripts/extract_asr_posterior.py` | 기존 Whisper transcription만 사용해 `one_best`와 dummy confidence를 저장하는 MVP를 구현한다. | 1개 wav로 JSONL 생성 |
| Step 34 | `src/f5_tts/scripts/extract_asr_posterior.py` | CTC model을 선택적으로 로드해 top-k frame posterior를 `.npz`로 저장한다. | 짧은 wav 1개로 NPZ 생성 |
| Step 35 | `src/f5_tts/posterior/schema.py` | CTC posterior 저장에 필요한 shard path, frame rate, top-k metadata 필드를 추가한다. | schema test 재실행 |
| Step 36 | `tests/test_posterior_schema.py` | 새 CTC metadata 필드의 기본값과 dict 변환 테스트를 추가한다. | `pytest tests/test_posterior_schema.py` |
| Step 37 | `docs/md/posterior_cache_format.md` | JSONL/NPZ cache format, token id offset, blank/filler 규칙을 문서화한다. | 문서 직접 확인 |

### API와 socket 호환

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 38 | `src/f5_tts/api.py` | public API에 optional `ref_text_mode`, `posterior_file` 인자를 추가하되 기본값은 hard로 둔다. | `python -m compileall src/f5_tts/api.py` |
| Step 39 | `src/f5_tts/api.py` | API 경로에서 length-only expected length를 `infer_process()`로 전달한다. | 기존 API 호출 smoke test |
| Step 40 | `src/f5_tts/socket_server.py` | socket server 설정에 optional posterior mode 필드를 추가하되 기본값은 hard로 둔다. | `python -m compileall src/f5_tts/socket_server.py` |
| Step 41 | `src/f5_tts/socket_server.py` | streaming inference에도 expected length를 넘길 수 있게 연결한다. | streaming hard mode smoke test |

### learned posterior encoder

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 42 | `src/f5_tts/model/posterior_encoder.py` | top-k ids/probs/entropy를 받아 `[batch, seq_len, text_dim]`을 내는 작은 encoder class를 만든다. | `python -m compileall src/f5_tts/model/posterior_encoder.py` |
| Step 43 | `tests/test_posterior_encoder_shapes.py` | posterior encoder 입력/출력 shape, padding mask 동작 테스트를 추가한다. | `pytest tests/test_posterior_encoder_shapes.py` |
| Step 44 | `src/f5_tts/model/posterior_dataset.py` | oracle text, posterior cache, mel length를 함께 반환하는 dataset wrapper를 만든다. | `python -m compileall src/f5_tts/model/posterior_dataset.py` |
| Step 45 | `tests/test_posterior_dataset.py` | 임시 cache fixture로 dataset item이 필요한 key를 반환하는지 테스트한다. | `pytest tests/test_posterior_dataset.py` |
| Step 46 | `src/f5_tts/train/train_posterior.py` | F5 backbone freeze + posterior encoder distillation 학습 script skeleton을 만든다. | `python src/f5_tts/train/train_posterior.py --help` |
| Step 47 | `src/f5_tts/configs/F5TTS_v1_Base_Posterior.yaml` | posterior encoder 학습용 config를 추가한다. | config load smoke test |
| Step 48 | `src/f5_tts/train/train_posterior.py` | oracle `TextEmbedding` hidden state를 teacher로 뽑고 MSE distillation loss를 계산한다. | 1 batch overfit smoke test |

### 평가

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 49 | `src/f5_tts/eval/error_breakdown.py` | WER/CER, Sub/Del/Ins 분해 helper를 구현한다. | `python -m compileall src/f5_tts/eval/error_breakdown.py` |
| Step 50 | `tests/test_error_breakdown.py` | substitution, deletion, insertion toy example 테스트를 추가한다. | `pytest tests/test_error_breakdown.py` |
| Step 51 | `src/f5_tts/eval/eval_posterior_f5.py` | manifest를 읽고 hard/length-only/soft-ctc mode별 생성과 평가를 수행하는 skeleton을 만든다. | `python src/f5_tts/eval/eval_posterior_f5.py --help` |
| Step 52 | `configs/eval/posterior_f5_baseline.yaml` | clean/noisy small subset baseline eval config를 추가한다. | YAML load 확인 |
| Step 53 | `configs/eval/posterior_f5_full.yaml` | accented/dysarthric/noisy full eval config template을 추가한다. | YAML load 확인 |
| Step 54 | `docs/md/experiment_protocol.md` | dataset split, ASR 분리 평가, 통계 검정, 결과 table format을 문서화한다. | 문서 직접 확인 |

### SSL hybrid 확장

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 55 | `src/f5_tts/posterior/gating.py` | entropy 기반 `alpha` gate 계산 함수를 만든다. | `python -m compileall src/f5_tts/posterior/gating.py` |
| Step 56 | `tests/test_entropy_gate.py` | entropy가 낮으면 soft text weight가 커지고, 높으면 SSL weight가 커지는지 테스트한다. | `pytest tests/test_entropy_gate.py` |
| Step 57 | `src/f5_tts/model/ssl_reference_encoder.py` | WavLM 등 SSL feature를 F5 text_dim으로 projection하는 module skeleton을 만든다. | `python -m compileall src/f5_tts/model/ssl_reference_encoder.py` |
| Step 58 | `src/f5_tts/model/hybrid_reference_conditioner.py` | soft text condition과 SSL condition을 gate로 섞는 module을 만든다. | `python -m compileall src/f5_tts/model/hybrid_reference_conditioner.py` |
| Step 59 | `tests/test_hybrid_reference_conditioner.py` | gate alpha 1.0이면 soft text, 0.0이면 SSL branch와 같아지는지 테스트한다. | `pytest tests/test_hybrid_reference_conditioner.py` |
| Step 60 | `src/f5_tts/infer/infer_cli.py` | `--ref_text_mode hybrid`를 추가하고 hybrid conditioner를 optional로 연결한다. | hard/soft/hybrid CLI dry run |

### 마무리 검증

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 61 | `docs/md/posterior_aware_f5_tts_implementation_plan.md` | 구현하면서 바뀐 실제 파일명, CLI 옵션, 성공/실패한 실험을 반영해 계획서를 갱신한다. | 문서 diff 확인 |
| Step 62 | 없음 | 전체 unit test를 실행한다. 파일 수정은 하지 않는다. | `pytest tests/test_posterior_* tests/test_soft_* tests/test_hard_infer_compat.py` |
| Step 63 | 없음 | hard F5 baseline 1개, length-only 1개, soft-ctc 1개를 같은 prompt로 생성해 비교한다. | wav와 metric 출력 확인 |
| Step 64 | 없음 | `git diff --stat`과 `git status`로 변경 범위를 확인한다. | commit 전 최종 확인 |

이 순서를 따르면 한 번에 여러 곳을 동시에 건드리지 않아도 된다. 특히 `utils_infer.py`, `infer_cli.py`, `dit.py`, `cfm.py`처럼 핵심 경로 파일은 여러 step으로 나누어 바꾸는 것이 좋다. 이렇게 하면 어느 step에서 hard inference가 깨졌는지 바로 찾을 수 있다.

## 13. 논문 실험 가능 상태까지의 후속 스텝

Step 64까지는 연구 scaffold 완성에 가깝다. 논문 실험을 실제로 돌릴 수 있으려면 아래 조건이 추가로 충족되어야 한다.

| 조건 | 의미 |
| --- | --- |
| dataset manifest가 고정됨 | 같은 split과 utterance id로 모든 mode를 반복 실행할 수 있어야 한다. |
| posterior cache가 검증됨 | ASR 1-best, expected length, CTC top-k posterior, token map이 실제 wav에서 정상이어야 한다. |
| generation runner가 있음 | `hard`, `length_only`, `soft_ctc`, `posterior_encoder`를 같은 manifest에서 자동 생성해야 한다. |
| eval ASR runner가 있음 | generated wav를 별도 ASR로 transcribe해 prediction JSONL을 만들어야 한다. |
| metric/statistics runner가 있음 | WER/CER/Sub/Del/Ins와 bootstrap 유의성 검정을 자동화해야 한다. |
| posterior encoder 학습 loop가 완성됨 | skeleton이 아니라 checkpoint save/load, validation, overfit test가 있어야 한다. |
| 결과표가 자동 생성됨 | 논문 table과 appendix table을 재현 가능하게 만들어야 한다. |

아래 Step 65 이후는 이 조건들을 채우기 위한 실제 후속 계획이다. 가능하면 계속 한 step에 한 파일만 바꾼다.

### 데이터 manifest와 smoke set

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 65 | `docs/md/paper_experiment_readiness.md` | 논문 실험을 시작할 수 있는 최소 조건과 full 조건을 명시한다. | 문서 직접 확인 |
| Step 66 | `configs/eval/datasets_template.yaml` | clean/noisy/accented/dysarthric split 경로와 subset 이름의 표준 config를 만든다. | YAML load 확인 |
| Step 67 | `src/f5_tts/eval/build_eval_manifest.py` | dataset별 원본 metadata를 posterior-F5 공통 JSONL manifest로 바꾸는 CLI skeleton을 만든다. | `--help` 확인 |
| Step 68 | `tests/test_build_eval_manifest.py` | toy metadata에서 `utterance_id`, `ref_audio`, `ref_text`, `gen_text`, `text`, `subset`이 나오는지 테스트한다. | `pytest tests/test_build_eval_manifest.py` |
| Step 69 | `docs/md/dataset_manifest_format.md` | 공통 manifest schema와 dataset별 필드 매핑 규칙을 문서화한다. | 문서 직접 확인 |
| Step 70 | 없음 | 5개 utterance짜리 local smoke manifest를 생성한다. git에는 넣지 않는다. | `wc -l manifests/smoke.jsonl` |

### posterior cache 실제 검증

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 71 | `src/f5_tts/scripts/inspect_posterior_cache.py` | posterior JSONL/NPZ의 shape, top-k 합, blank id, expected length를 검사하는 CLI를 만든다. | `--help` 확인 |
| Step 72 | `tests/test_inspect_posterior_cache.py` | toy JSONL/NPZ에서 검사 결과가 정상인지 테스트한다. | `pytest tests/test_inspect_posterior_cache.py` |
| Step 73 | `docs/md/posterior_cache_format.md` | 실제 smoke cache 생성 명령과 inspect 명령을 추가한다. | 문서 diff 확인 |
| Step 74 | 없음 | smoke manifest 5개에 대해 `extract_asr_posterior.py --skip_whisper`를 실행한다. | `inspect_posterior_cache.py` 통과 |
| Step 75 | 없음 | 실제 CTC model로 1개 wav top-k posterior를 추출한다. | NPZ shape와 entropy 확인 |
| Step 76 | `src/f5_tts/posterior/token_projection.py` | ASR token id를 F5 vocab id로 매핑하거나 검증하는 helper를 만든다. | compile 확인 |
| Step 77 | `tests/test_token_projection.py` | blank/filler/unknown/token offset 규칙을 toy vocab으로 테스트한다. | `pytest tests/test_token_projection.py` |

### generation runner

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 78 | `src/f5_tts/eval/generate_posterior_f5.py` | manifest와 mode를 받아 generated wav와 generation metadata JSONL을 만드는 CLI skeleton을 만든다. | `--help` 확인 |
| Step 79 | `src/f5_tts/eval/generate_posterior_f5.py` | `hard` mode generation을 API 또는 `infer_process()`로 연결한다. | smoke 1개 wav 생성 |
| Step 80 | `src/f5_tts/eval/generate_posterior_f5.py` | `length_only` mode에서 posterior expected length를 연결한다. | hard/length wav 둘 다 생성 |
| Step 81 | `src/f5_tts/eval/generate_posterior_f5.py` | `soft_ctc` mode에서 posterior file과 NPZ shard를 연결한다. | soft_ctc 1개 wav 생성 |
| Step 82 | `tests/test_generate_posterior_f5_args.py` | mode, output path, manifest parsing을 lightweight test로 검증한다. | `pytest tests/test_generate_posterior_f5_args.py` |
| Step 83 | `configs/eval/posterior_f5_baseline.yaml` | smoke run용 subset 크기, output path, mode list를 실제 runner 옵션에 맞춘다. | YAML load 확인 |
| Step 84 | 없음 | smoke manifest에서 `hard`, `length_only`, `soft_ctc` wav를 각각 1개 이상 생성한다. | output wav 존재 확인 |

### eval ASR와 metric runner

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 85 | `src/f5_tts/eval/transcribe_generated.py` | generated wav manifest를 별도 ASR로 transcribe해 prediction JSONL을 만드는 CLI를 만든다. | `--help` 확인 |
| Step 86 | `tests/test_transcribe_generated_args.py` | generated manifest parsing과 prediction row schema를 테스트한다. | `pytest tests/test_transcribe_generated_args.py` |
| Step 87 | `src/f5_tts/eval/eval_posterior_f5.py` | subset/mode별 aggregation과 per-utterance metrics JSONL 출력을 추가한다. | toy JSONL smoke |
| Step 88 | `tests/test_eval_posterior_f5.py` | toy references/predictions에서 WER/CER 집계가 맞는지 테스트한다. | `pytest tests/test_eval_posterior_f5.py` |
| Step 89 | `src/f5_tts/eval/bootstrap_significance.py` | paired bootstrap resampling으로 mode 간 WER/CER 차이 confidence interval을 계산한다. | `--help` 확인 |
| Step 90 | `tests/test_bootstrap_significance.py` | 고정 seed toy data로 bootstrap 결과 shape와 부호를 테스트한다. | `pytest tests/test_bootstrap_significance.py` |
| Step 91 | 없음 | smoke generated wav를 eval ASR로 transcribe하고 metric JSON을 만든다. | WER/CER JSON 생성 |

### posterior encoder 학습 완성

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 92 | `src/f5_tts/model/posterior_dataset.py` | `oracle_text_tensor`, `seq_len`, `posterior_mask`를 batch에 포함하도록 확장한다. | dataset test 재실행 |
| Step 93 | `src/f5_tts/train/train_posterior.py` | 실제 dataloader, optimizer, scheduler, checkpoint save/load loop를 추가한다. | `--dry_run` 확인 |
| Step 94 | `tests/test_train_posterior_step.py` | tiny posterior encoder 1-step overfit loss가 감소하는지 테스트한다. | `pytest tests/test_train_posterior_step.py` |
| Step 95 | `src/f5_tts/infer/infer_cli.py` | `--posterior_encoder_ckpt` 옵션과 `posterior_encoder` mode를 추가한다. | CLI help 확인 |
| Step 96 | `src/f5_tts/model/posterior_encoder.py` | checkpoint load helper와 config serialization helper를 추가한다. | compile 확인 |
| Step 97 | 없음 | 1개 batch overfit posterior encoder를 저장하고 `posterior_encoder` inference smoke를 실행한다. | wav 생성 확인 |

### 결과표와 논문 재현성

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 98 | `src/f5_tts/eval/make_result_tables.py` | metric JSON들을 모아 paper table CSV/Markdown을 생성한다. | `--help` 확인 |
| Step 99 | `tests/test_make_result_tables.py` | toy metrics에서 mode/subset table이 올바르게 나오는지 테스트한다. | `pytest tests/test_make_result_tables.py` |
| Step 100 | `configs/eval/posterior_f5_full.yaml` | full run에 필요한 mode, subset, ASR source, eval ASR, bootstrap seed를 확정한다. | YAML load 확인 |
| Step 101 | `docs/md/experiment_protocol.md` | smoke, dev, full run 명령을 실제 파일명 기준으로 갱신한다. | 문서 diff 확인 |
| Step 102 | `docs/md/reproducibility_checklist.md` | commit hash, dataset version, ASR model, F5 checkpoint, seed, hardware 기록 양식을 만든다. | 문서 직접 확인 |
| Step 103 | 없음 | dev subset에서 `hard`, `length_only`, `soft_ctc` 전체 sweep을 실행한다. | table 생성 확인 |
| Step 104 | 없음 | full clean/noisy/accented/dysarthric run을 서버 GPU에서 실행한다. | output completeness check |
| Step 105 | 없음 | bootstrap significance와 paper table을 생성한다. | CSV/Markdown table 확인 |

### 논문 실험 가능 판정

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 106 | `docs/md/paper_experiment_readiness.md` | 실제 완료된 smoke/dev/full run 결과와 남은 gap을 기록한다. | 문서 diff 확인 |
| Step 107 | 없음 | `git status`, `git log`, run output completeness를 확인한다. | dirty file 없음 |
| Step 108 | 없음 | 논문 main table에 들어갈 최소 결과를 고정한다. | `results/tables/main_results.md` 존재 |

Step 108까지 완료되면 “논문 실험을 돌릴 수 있는 상태”가 아니라, 최소한의 논문용 결과표를 실제로 생성한 상태가 된다. 이때부터는 모델 아이디어 개선보다는 ablation 추가, human evaluation, reviewer가 물을 비교군 보강으로 넘어간다.
