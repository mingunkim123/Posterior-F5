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
