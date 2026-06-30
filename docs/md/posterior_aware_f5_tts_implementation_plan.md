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

Step 60의 현재 구현은 SSL feature extractor가 연결되기 전 단계의 안전한 skeleton이다. `hybrid` mode는 soft CTC posterior가 있을 때 `HybridReferenceConditioner`를 통과하지만, 실제 SSL branch 입력이 없으면 soft text condition을 그대로 유지하는 fallback으로 둔다. 실제 WavLM/SSL cache 연결은 Step 65 이후의 논문 실험 준비 단계에서 완성한다.

### 마무리 검증

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 61 | `docs/md/posterior_aware_f5_tts_implementation_plan.md` | 구현하면서 바뀐 실제 파일명, CLI 옵션, 성공/실패한 실험을 반영해 계획서를 갱신한다. | 문서 diff 확인 |
| Step 62 | 없음 | 전체 unit test를 실행한다. 파일 수정은 하지 않는다. | `pytest tests/test_posterior_* tests/test_soft_* tests/test_hard_infer_compat.py` |
| Step 63 | 없음 | hard F5 baseline 1개, length-only 1개, soft-ctc 1개를 같은 prompt로 생성해 비교한다. | wav와 metric 출력 확인 |
| Step 64 | 없음 | `git diff --stat`과 `git status`로 변경 범위를 확인한다. | commit 전 최종 확인 |

이 순서를 따르면 한 번에 여러 곳을 동시에 건드리지 않아도 된다. 특히 `utils_infer.py`, `infer_cli.py`, `dit.py`, `cfm.py`처럼 핵심 경로 파일은 여러 step으로 나누어 바꾸는 것이 좋다. 이렇게 하면 어느 step에서 hard inference가 깨졌는지 바로 찾을 수 있다.

### Step 64 진행 기록

현재 코드 기준으로 Step 1~64의 연구 scaffold는 구현되어 있다.

확인된 항목:

- posterior 기본 패키지: `schema.py`, `length.py`, `normalize.py`, `io.py`, `soft_embedding.py`
- length-only baseline: `infer_process()`, `infer_batch_process()`, `infer_cli.py`의 `--ref_text_mode`, `--posterior_file`
- soft CTC MVP: `DiT`/`CFM`의 `text_embed_override`, CLI의 `soft_ctc`/`hybrid` builder
- posterior extraction MVP: `src/f5_tts/scripts/extract_asr_posterior.py`
- public API/socket 호환: `ref_text_mode`, `posterior_file`, expected length 전달
- learned posterior encoder scaffold: `posterior_encoder.py`, `posterior_dataset.py`, `train_posterior.py`
- eval scaffold: `error_breakdown.py`, `eval_posterior_f5.py`, baseline/full config
- SSL hybrid scaffold: `gating.py`, `ssl_reference_encoder.py`, `hybrid_reference_conditioner.py`

검증 결과:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_posterior_* \
  tests/test_soft_* \
  tests/test_hard_infer_compat.py \
  tests/test_entropy_gate.py \
  tests/test_hybrid_reference_conditioner.py \
  tests/test_error_breakdown.py -q
```

결과:

```text
55 passed
```

Step 63은 로컬 GPU inference 대신 lightweight artifact smoke로 확인했다.

```bash
PYTHONPATH=src .venv/bin/python platform/workers/run_posterior_f5_pipeline.py \
  --run_id step64_posterior_scaffold_smoke \
  --artifact_root /tmp/posterior_f5_step64_runs \
  --manifest platform/samples/manifests/dev_smoke.jsonl \
  --mode hard \
  --mode length_only \
  --mode soft_ctc \
  --run_posterior_extraction \
  --skip_whisper \
  --run_inference \
  --inference_dry_run \
  --run_prediction \
  --prediction_dry_run \
  --run_metrics \
  --fail_if_exists
```

생성 확인:

```text
generated/{hard,length_only,soft_ctc}/commands.jsonl
predictions/{hard,length_only,soft_ctc}.jsonl
metrics/{hard,length_only,soft_ctc}.metrics.json
metrics/summary.csv
posterior_cache/run.posterior.jsonl
logs/events.jsonl
```

주의:

Step 64까지는 연구 scaffold 완성이다. 실제 wav 품질 비교는 GPU worker에서 `inference_dry_run=false`로 hard/length_only/soft_ctc를 다시 실행해야 한다.

## 13. STEP64 이후: 완성형 실험으로 가는 후속 스텝

Step 64까지는 모델 아이디어를 검증할 수 있는 연구 scaffold가 완성된 상태다. 현재 코드 감사 기준으로, 다음 단계는 새 runner를 따로 만드는 것이 아니라 이미 있는 `platform/workers/run_posterior_f5_pipeline.py`를 실제 GPU/ASR 실행과 논문 표 생성까지 견고하게 만드는 쪽이 맞다.

### 현재 코드 감사 요약

이미 갖춘 것:

- `platform/workers/run_posterior_f5_pipeline.py`가 scaffold, posterior extraction, mode별 inference, evaluation ASR prediction, WER/CER metric summary까지 한 run directory로 묶는다.
- CLI는 `hard`, `length_only`, `soft_ctc`, `hybrid`를 받는다. `oracle`은 platform runner가 gold `ref_text`를 넣어 `hard`로 실행하는 ceiling mode다.
- posterior package, cache schema, CTC top-k loader, length correction, soft embedding override, WER/CER breakdown, MLOps API/frontend scaffold가 있다.
- `TopKPosteriorEncoder`, posterior dataset wrapper, distillation step은 있지만 `train_posterior.py`의 full training loop는 아직 `NotImplementedError`로 멈춘다.

아직 완성형 실험 전에 막히는 부분:

- `posterior_encoder`는 config와 UI 선택지에는 보이지만 CLI mode와 platform runner에 실제 inference mapping이 없다.
- `hybrid`는 SSL feature를 읽지 않고 soft condition을 `alpha=1.0`으로 통과시키는 placeholder다.
- `eval_posterior_f5.py`는 전체 WER/CER aggregate 중심이고, subset별/per-utterance metrics, bootstrap significance, speaker similarity, UTMOS, RTF가 아직 논문 표 수준으로 묶이지 않았다.
- API와 socket 경로는 `hard|length_only` 중심이다. 당장 논문 실험은 CLI/platform runner를 기준 경로로 삼는 것이 안전하다.
- 실제 CTC posterior cache, ASR-to-F5 token projection, top-k mass/entropy calibration은 real wav에서 별도 inspect가 필요하다.

### 실험 가능 판정 게이트

| Gate | 통과 조건 |
| --- | --- |
| G1. 재현 가능한 코드 상태 | run metadata에 clean commit hash, branch, dirty 여부가 남고 실험 전 dirty file을 정리한다. |
| G2. manifest 고정 | `utterance_id`, `ref_audio`, `ref_text`, `gen_text`, `text`, `subset`이 모든 split에서 동일 규칙으로 들어간다. |
| G3. posterior cache 검증 | JSONL line 수, NPZ shape, top-k probability, blank/filler id, expected length, token projection을 inspect한다. |
| G4. 실제 wav 생성 | dry-run이 아니라 GPU에서 `hard`, `oracle`, `length_only`, `soft_ctc` wav가 같은 manifest로 생성된다. |
| G5. 평가 ASR 분리 | posterior source ASR와 다른 eval ASR로 prediction JSONL을 만든다. |
| G6. 논문 metric/table | subset별 WER/CER/Sub/Del/Ins, deletion rate, bootstrap CI, paper table CSV/Markdown이 자동 생성된다. |
| G7. 확장 방법 검증 | Stage 1 결과가 고정된 뒤 `posterior_encoder`와 진짜 SSL `hybrid`를 추가한다. |

아래 Step 65 이후는 이 게이트를 통과하기 위한 후속 계획이다. 원칙은 계속 유지한다. 가능하면 한 step은 파일 하나만 바꾸고, runner는 새로 만들기보다 기존 platform pipeline을 보강한다.

### A. 실험 입력과 run contract 고정

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 65 | `docs/md/posterior_aware_f5_tts_implementation_plan.md` | 현재 코드 감사 결과와 STEP65 이후 실행 계획을 이 문서에 반영한다. | 문서 diff 확인 |
| Step 66 | 없음 | 실험 전 working tree를 정리한다. 문서/코드/실험 산출물 중 무엇을 커밋할지 분리한다. | `git status -sb` |
| Step 67 | `platform/config/datasets.yaml` | `dev_small`, `clean_test`, `noisy_test`, `accented_test`, `dysarthric_test` registry 항목을 실제 manifest 경로로 추가한다. | backend dataset list 확인 |
| Step 68 | `platform/config/checkpoints.yaml` | upstream F5 checkpoint와 실험용 posterior checkpoint id, hash, vocoder를 고정한다. | checkpoint registry load 확인 |
| Step 69 | `configs/eval/posterior_f5_baseline.yaml` | 현재 platform runner 옵션에 맞게 Stage 1 mode(`hard`, `oracle`, `length_only`, `soft_ctc`), seed, ASR, artifact root를 정리한다. | YAML load 확인 |
| Step 70 | `configs/eval/posterior_f5_full.yaml` | `posterior_encoder`, `hybrid`를 Stage 2 mode로 분리해 아직 준비 전인 mode가 full baseline run을 막지 않게 한다. | YAML load 확인 |
| Step 71 | 없음 | dev smoke manifest와 dev small manifest를 실제 데이터 경로로 만든다. 대용량 데이터와 wav는 git에 넣지 않는다. | `wc -l manifests/dev_small.jsonl` |
| Step 72 | `docs/md/experiment_protocol.md` | Stage 1 최소 실험 명령과 Stage 2 확장 실험 명령을 현재 파일명 기준으로 갱신한다. | 문서 diff 확인 |

진행 기록:

- Step 66: 기존 dirty file은 `src/f5_tts/infer/infer_cli.py`, `tests/test_soft_text_override_shapes.py`로 확인했고, A 섹션 작업에서는 건드리지 않는다.
- Step 67: `platform/config/datasets.yaml`에 `dev_small`, `clean_test`, `noisy_test`, `accented_test`, `dysarthric_test` registry 계약을 추가했다.
- Step 68: `platform/config/checkpoints.yaml`에 Stage 1 upstream F5 checkpoint와 Stage 2 posterior encoder checkpoint slot을 고정했다.
- Step 69: `configs/eval/posterior_f5_baseline.yaml`을 `hard/oracle/length_only/soft_ctc` Stage 1 runner 계약으로 정리했다.
- Step 70: `configs/eval/posterior_f5_full.yaml`에서 Stage 1 mode와 Stage 2 mode(`posterior_encoder`, `hybrid`)를 분리했다.
- Step 71: `manifests/dev_small.jsonl`을 checked-in smoke/dev manifest로 추가했다. full split manifest와 wav/cache 산출물은 로컬 데이터로 유지한다.
- Step 72: `docs/md/experiment_protocol.md`에 manifest contract, Stage 1 smoke/full, Stage 2 확장 실행 순서를 반영했다.

### B. Posterior cache 실제 검증

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 73 | `src/f5_tts/scripts/inspect_posterior_cache.py` | posterior JSONL/NPZ의 line 수, shard 존재, shape, top-k 합, blank id, expected length 범위를 검사하는 CLI를 만든다. | `--help` 확인 |
| Step 74 | `tests/test_inspect_posterior_cache.py` | toy JSONL/NPZ로 정상 cache, 누락 shard, shape mismatch를 테스트한다. | `pytest tests/test_inspect_posterior_cache.py` |
| Step 75 | `src/f5_tts/posterior/token_projection.py` | `infer_cli.py` 안의 ASR token to F5 vocab projection 규칙을 독립 helper로 분리한다. | compile 확인 |
| Step 76 | `tests/test_token_projection.py` | blank/filler/unknown/space/token offset 규칙을 toy vocab으로 고정한다. | `pytest tests/test_token_projection.py` |
| Step 77 | `src/f5_tts/scripts/extract_asr_posterior.py` | real CTC extraction metadata에 top-k mass, duration, sample rate, entropy 계산 기준을 명시적으로 저장한다. | 1개 wav cache inspect |
| Step 78 | 없음 | smoke/dev small reference audio로 실제 CTC posterior를 생성한다. | `inspect_posterior_cache.py` 통과 |
| Step 79 | `docs/md/posterior_cache_format.md` | real CTC cache 생성/검사 명령과 token projection 규칙을 실제 구현 기준으로 갱신한다. | 문서 diff 확인 |

진행 기록:

- Step 73: `src/f5_tts/scripts/inspect_posterior_cache.py`를 추가했다. JSONL count, shard 존재, NPZ shape, top-k mass, blank id, expected length를 검사한다.
- Step 74: `tests/test_inspect_posterior_cache.py`를 추가해 정상 shard, 누락 shard, shape mismatch, CLI failure path를 고정했다.
- Step 75: `src/f5_tts/posterior/token_projection.py`를 추가해 ASR token id to F5 vocab id projection 규칙을 독립 helper로 분리했다. 기존 dirty 상태의 `infer_cli.py` refactor는 별도 정리 후 연결한다.
- Step 76: `tests/test_token_projection.py`를 추가해 blank/filler/unknown/space/special token 규칙을 고정했다.
- Step 77: `extract_asr_posterior.py`가 CTC shard metadata에 raw top-k mass와 entropy normalization 기준을 남기도록 보강했다.
- Step 78: 이번 pass에서는 `--skip_whisper` lightweight cache를 inspect했다. 실제 CTC model 다운로드가 필요한 real top-k 생성은 GPU/ASR 환경에서 다음 실행 항목으로 남긴다.
- Step 79: `docs/md/posterior_cache_format.md`에 inspect 명령, raw top-k probability space, token projection 규칙을 갱신했다.

### C. Platform runner 실험 실행화

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 80 | `platform/workers/run_posterior_f5_pipeline.py` | YAML config를 받아 현재 긴 CLI 옵션을 재현 가능한 config run으로 실행할 수 있게 한다. | config smoke run |
| Step 81 | `platform/workers/run_posterior_f5_pipeline.py` | mode별 generation metadata JSONL에 elapsed time, output wav path, file size/hash, seed, checkpoint를 남긴다. | dry-run과 real-run metadata 확인 |
| Step 82 | `tests/test_mlops_run_scaffold.py` | config run과 generation metadata contract 테스트를 추가한다. | `pytest tests/test_mlops_run_scaffold.py` |
| Step 83 | 없음 | GPU에서 dev smoke 1개 utterance를 `hard`, `oracle`, `length_only`, `soft_ctc`로 dry-run 없이 생성한다. | mode별 wav 존재 확인 |
| Step 84 | `platform/workers/run_posterior_f5_pipeline.py` | non-dry-run에서 wav 누락, 0 byte, 너무 짧은 생성물을 stage failure로 처리한다. | 실패 fixture 테스트 |
| Step 85 | `tests/test_mlops_run_scaffold.py` | missing/empty wav completeness check를 lightweight test로 고정한다. | `pytest tests/test_mlops_run_scaffold.py` |

진행 기록:

- Step 80: `run_posterior_f5_pipeline.py`에 `--config` YAML 입력을 추가해 `configs/eval/*.yaml` 계약을 직접 실행할 수 있게 했다.
- Step 81: mode별 `generated/<mode>/commands.jsonl`에 seed, checkpoint, started/finished time, elapsed time, exit code, output size/hash를 기록하도록 확장했다.
- Step 82: `tests/test_mlops_run_scaffold.py`에 config run contract와 generation metadata 테스트를 추가했다.
- Step 83: GPU real wav 생성은 환경 의존 작업이라 아직 실행하지 않았다. C 섹션에서는 runner가 real-run metadata와 completeness check를 지원하도록 준비했다.
- Step 84: non-dry-run에서 wav 누락 또는 `--min_wav_bytes` 미만 출력이면 inference stage를 실패로 처리한다.
- Step 85: tiny wav fake inference fixture로 completeness failure를 테스트했다.

### D. 평가와 논문 metric 완성

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 86 | `platform/workers/run_posterior_f5_pipeline.py` | prediction JSONL에 subset, reference text, eval ASR id, generated audio path를 보존한다. | prediction JSONL 확인 |
| Step 87 | `src/f5_tts/eval/eval_posterior_f5.py` | per-utterance metrics JSONL과 subset/mode별 aggregation을 추가한다. | toy JSONL smoke |
| Step 88 | `tests/test_eval_posterior_f5.py` | toy references/predictions에서 WER/CER/Sub/Del/Ins, deletion rate, subset aggregation이 맞는지 테스트한다. | `pytest tests/test_eval_posterior_f5.py` |
| Step 89 | `src/f5_tts/eval/bootstrap_significance.py` | paired bootstrap resampling으로 mode 간 WER/CER/deletion 차이와 confidence interval을 계산한다. | `--help` 확인 |
| Step 90 | `tests/test_bootstrap_significance.py` | 고정 seed toy data로 bootstrap 출력 shape, sign, reproducibility를 테스트한다. | `pytest tests/test_bootstrap_significance.py` |
| Step 91 | `src/f5_tts/eval/make_result_tables.py` | metrics summary와 bootstrap 결과를 paper table CSV/Markdown으로 변환한다. | `--help` 확인 |
| Step 92 | `tests/test_make_result_tables.py` | toy metrics에서 main table과 ablation table이 올바르게 생성되는지 테스트한다. | `pytest tests/test_make_result_tables.py` |
| Step 93 | `src/f5_tts/eval/eval_posterior_f5.py` | generation metadata가 있으면 RTF/latency를 metric에 병합한다. speaker similarity와 UTMOS는 별도 파일을 읽는 optional input으로 둔다. | toy metadata smoke |
| Step 94 | 없음 | dev smoke generated wav를 별도 eval ASR로 transcribe하고 metrics/table까지 생성한다. | `metrics/summary.csv`, table 존재 |

### E. Stage 1 논문 최소 결과 고정

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 95 | 없음 | dev small에서 `hard`, `oracle`, `length_only`, `soft_ctc` 전체 sweep을 실행한다. | run status completed |
| Step 96 | 없음 | dev small 결과에서 deletion, insertion, repetition, entropy high case를 샘플링해 실패 유형을 확인한다. | qualitative examples 확인 |
| Step 97 | `docs/md/posterior_f5_paper_experiment_flow.md` | 실제 dev small 명령, artifact 경로, 현재까지의 결과 해석을 기록한다. | 문서 diff 확인 |
| Step 98 | 없음 | clean/noisy/accented/dysarthric full Stage 1 sweep을 서버 GPU에서 실행한다. | completeness check 통과 |
| Step 99 | 없음 | Stage 1 bootstrap과 main table을 생성하고, `length_only`와 `soft_ctc`의 기여를 분리해 판단한다. | paper table Markdown 확인 |

Stage 1까지 끝나면 `posterior_encoder`나 `hybrid`가 없어도 최소 논문 결과는 나온다. 이때 핵심 비교는 `hard` vs `oracle`로 oracle gap을 보고, `hard` vs `length_only`로 duration 효과를 분리하고, `length_only` vs `soft_ctc`로 representation 효과를 보는 것이다.

### F. Posterior encoder 완성

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 100 | `src/f5_tts/model/posterior_dataset.py` | batch에 `oracle_text_tensor`, `seq_len`, `posterior_mask`, `blank_prob`, optional entropy를 포함한다. | dataset test 재실행 |
| Step 101 | `src/f5_tts/train/train_posterior.py` | dataloader, optimizer, scheduler, validation, checkpoint save/load를 포함한 full training loop를 구현한다. | `--dry_run`과 1 batch run |
| Step 102 | `tests/test_train_posterior_step.py` | tiny posterior encoder overfit에서 loss가 감소하는지 테스트한다. | `pytest tests/test_train_posterior_step.py` |
| Step 103 | `src/f5_tts/model/posterior_encoder.py` | checkpoint load helper와 config serialization helper를 추가한다. | compile 확인 |
| Step 104 | `src/f5_tts/infer/infer_cli.py` | `posterior_encoder` mode와 `--posterior_encoder_ckpt`를 추가한다. 현재 CLI enum에는 아직 없다. | CLI help 확인 |
| Step 105 | `platform/workers/run_posterior_f5_pipeline.py` | `posterior_encoder` mode를 runner에 연결하고 ckpt path를 command metadata에 남긴다. | dry-run command 확인 |
| Step 106 | 없음 | 1개 batch overfit checkpoint로 `posterior_encoder` inference smoke를 실행한다. | wav 생성 확인 |

### G. 진짜 SSL hybrid 완성

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 107 | `src/f5_tts/scripts/extract_ssl_reference.py` | reference audio에서 WavLM/SSL feature를 cache로 저장하는 CLI를 만든다. | `--help` 확인 |
| Step 108 | `tests/test_ssl_reference_cache.py` | toy feature shard load, target length interpolation, missing shard 처리를 테스트한다. | `pytest tests/test_ssl_reference_cache.py` |
| Step 109 | `src/f5_tts/model/ssl_reference_encoder.py` | cached SSL feature load/project helper를 추가해 inference에서 모델 weight 재계산을 피할 수 있게 한다. | compile 확인 |
| Step 110 | `src/f5_tts/infer/infer_cli.py` | `hybrid` mode에서 SSL cache와 entropy/blank gate를 읽어 soft text와 SSL condition을 실제로 mix한다. 현재 alpha 1.0 passthrough를 대체한다. | hybrid smoke wav |
| Step 111 | `tests/test_hybrid_reference_conditioner.py` | entropy/blank gate가 low entropy에서는 soft text, high entropy에서는 SSL branch로 기울어지는지 end-to-end에 가깝게 테스트한다. | `pytest tests/test_hybrid_reference_conditioner.py` |
| Step 112 | 없음 | severe/noisy subset에서 `soft_ctc` vs real `hybrid`를 비교한다. | subset table 확인 |

### H. 최종 재현성과 논문 산출물

| Step | 파일 | 작업 | 확인 |
| --- | --- | --- | --- |
| Step 113 | `docs/md/reproducibility_checklist.md` | commit hash, dataset version, ASR model, F5 checkpoint, posterior encoder ckpt, seed, GPU, runtime 기록 양식을 만든다. | 문서 직접 확인 |
| Step 114 | `docs/md/experiment_protocol.md` | full run, bootstrap, table generation, failure case export 명령을 최종 파일명으로 갱신한다. | 문서 diff 확인 |
| Step 115 | 없음 | full Stage 1 결과를 freeze하고 Stage 2 posterior_encoder/hybrid 결과를 별도 run id로 freeze한다. | run directories read-only 보존 |
| Step 116 | 없음 | main table, ablation table, appendix table, qualitative examples를 생성한다. | `results/tables/*.md` 확인 |
| Step 117 | 없음 | 최종 `git status`, `git log`, artifact completeness를 확인한다. | dirty file 없음 |
| Step 118 | 없음 | 논문 main claim에 들어갈 최소 결과를 고정한다. | final table과 run id 기록 |

Step 99까지 완료되면 최소 논문 실험은 실제로 돌릴 수 있다. Step 118까지 완료되면 “실험 가능”을 넘어 main table과 ablation table을 재현 가능한 run id로 고정한 상태가 된다. 이후는 아이디어 구현보다 비교군 보강, human evaluation, reviewer 질문 대응용 ablation을 추가하는 단계다.
