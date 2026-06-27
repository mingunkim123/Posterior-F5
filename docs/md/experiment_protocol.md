# Posterior-F5 Experiment Protocol

## 목적

이 실험은 ASR posterior-aware reference conditioning이 기존 F5-TTS hard 1-best reference transcript 의존성을 얼마나 줄이는지 검증한다.

## Split

권장 split:

```text
clean/dev_small
clean/test
noisy/test
accented/test
dysarthric/test
```

각 row는 최소한 아래 필드를 가진 JSONL로 둔다.

```json
{"utterance_id":"utt-0001","ref_audio":"ref.wav","ref_text":"oracle prompt transcript","gen_text":"target text","text":"evaluation reference text","subset":"accented"}
```

## 비교 모드

| mode | 설명 |
| --- | --- |
| `hard` | 기존 F5-TTS hard reference transcript |
| `oracle` | gold reference transcript ceiling |
| `length_only` | posterior expected reference length만 사용 |
| `soft_ctc` | CTC posterior expected embedding 사용 |
| `posterior_encoder` | learned posterior encoder 사용 |

## ASR 분리 원칙

posterior source ASR와 evaluation ASR는 분리한다. 예를 들어 posterior가 CTC model에서 나왔으면 WER/CER 평가에는 Whisper 계열을 쓰고, posterior가 Whisper에서 나왔으면 별도 CTC/Conformer 평가 모델로 교차 확인한다.

## Metric

기본 metric:

```text
WER
CER
Substitution / Deletion / Insertion
deletion rate
speaker similarity cosine
UTMOS or equivalent automatic quality score
RTF / latency
```

주관 평가:

```text
MOS: naturalness
SMOS: speaker similarity
CMOS: pairwise preference
```

## 통계 검정

| 대상 | 검정 |
| --- | --- |
| WER/CER/Sub/Del/Ins | paired bootstrap resampling |
| 여러 mode 비교 | Holm correction 또는 FDR correction |
| speaker cosine | paired permutation test |
| MOS/SMOS | mixed-effects model 또는 Wilcoxon signed-rank |

## 결과 테이블

권장 table:

```text
subset | mode | WER | CER | Sub | Del | Ins | speaker_sim | UTMOS | RTF
```

핵심 분석 table:

```text
subset | mode | oracle_gap_recovered | deletion_relative_reduction | notes
```

## 성공 기준

| 구간 | 성공 기준 |
| --- | --- |
| clean | hard baseline 대비 유의미한 성능 저하 없음 |
| accented/noisy | WER/CER 상대 8-15% 개선 또는 oracle gap 30% 이상 회복 |
| dysarthric | WER/CER 상대 5-10% 개선, deletion 10-20% 감소 |
| hard sentence | deletion 감소, insertion 급증 없음 |
| speaker similarity | hard baseline 대비 유의미한 하락 없음 |

## 실행 순서

1. `extract_asr_posterior.py`로 posterior cache 생성.
2. `hard`, `oracle`, `length_only`, `soft_ctc` 출력 생성.
3. evaluation ASR로 generated wav transcript 생성.
4. `eval_posterior_f5.py`로 WER/CER/Sub/Del/Ins 집계.
5. subset별 paired bootstrap으로 유의성 확인.
6. 실패 case를 deletion, repetition, ASR entropy 기준으로 분해.
