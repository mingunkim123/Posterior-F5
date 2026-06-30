# Posterior-F5 Experiment Protocol

## 목적

이 실험은 ASR posterior-aware reference conditioning이 기존 F5-TTS hard 1-best reference transcript 의존성을 얼마나 줄이는지 검증한다.

## Manifest Contract

모든 실험 manifest는 JSONL 한 줄이 하나의 utterance가 되도록 고정한다.

필수 필드:

| 필드 | 의미 |
| --- | --- |
| `utterance_id` | posterior cache, wav, prediction, metric을 묶는 stable key |
| `ref_audio` | reference prompt audio path |
| `ref_text` | oracle prompt transcript. `oracle` ceiling과 posterior skip-whisper smoke에 사용 |
| `gen_text` | 생성해야 할 target text |
| `text` | evaluation ASR hypothesis와 비교할 target reference text |
| `subset` | `clean/dev_small`, `clean/test`, `noisy/test`, `accented/test`, `dysarthric/test` 같은 분석 축 |
| `language` | ASR language hint |

예시:

```json
{"utterance_id":"utt-0001","ref_audio":"ref.wav","ref_text":"oracle prompt transcript","gen_text":"target text","text":"target text","subset":"accented/test","language":"en"}
```

현재 checked-in smoke/dev 계약:

```text
platform/samples/manifests/dev_smoke.jsonl
manifests/dev_small.jsonl
```

대용량 full split manifest는 같은 schema로 로컬에 배치하고, wav와 cache 산출물은 git에 넣지 않는다.

```text
clean/dev_small
clean/test
noisy/test
accented/test
dysarthric/test
```

Registry 기준 경로:

```text
platform/config/datasets.yaml
```

## 비교 모드

| mode | 설명 |
| --- | --- |
| `hard` | 기존 F5-TTS hard reference transcript. `ref_text`를 비워 ASR 1-best baseline으로 둔다. |
| `oracle` | gold `ref_text`를 넣은 hard path ceiling |
| `length_only` | posterior expected reference length만 사용 |
| `soft_ctc` | CTC posterior expected embedding 사용 |
| `posterior_encoder` | learned posterior encoder 사용. Stage 2에서 inference wiring 이후 평가 |
| `hybrid` | soft text와 SSL fallback mix. Stage 2에서 real SSL cache wiring 이후 평가 |

Stage 1 최소 논문 실험 mode:

```text
hard
oracle
length_only
soft_ctc
```

Stage 2 확장 mode:

```text
posterior_encoder
hybrid
```

## ASR 분리 원칙

posterior source ASR와 evaluation ASR는 분리한다. 예를 들어 posterior가 CTC model에서 나왔으면 WER/CER 평가에는 Whisper 계열을 쓰고, posterior가 Whisper에서 나왔으면 별도 CTC/Conformer 평가 모델로 교차 확인한다.

Stage 1 기본 계약:

| 역할 | 기본값 |
| --- | --- |
| posterior one-best | `openai/whisper-large-v3-turbo` |
| posterior CTC top-k | `facebook/wav2vec2-base-960h` |
| eval ASR | posterior CTC source와 분리된 Whisper 계열 |
| checkpoint | `f5tts_v1_base_hf` |
| seed | `1234` |

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

### Stage 1 Smoke

dry-run artifact 계약 확인:

```bash
PYTHONPATH=src .venv/bin/python platform/workers/run_posterior_f5_pipeline.py \
  --run_id stage1_contract_smoke \
  --manifest manifests/dev_small.jsonl \
  --mode hard \
  --mode oracle \
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

GPU real wav smoke:

```bash
PYTHONPATH=src .venv/bin/python platform/workers/run_posterior_f5_pipeline.py \
  --run_id stage1_real_wav_smoke \
  --manifest manifests/dev_small.jsonl \
  --mode hard \
  --mode oracle \
  --mode length_only \
  --mode soft_ctc \
  --run_posterior_extraction \
  --ctc_model facebook/wav2vec2-base-960h \
  --run_inference \
  --run_prediction \
  --run_metrics \
  --infer_device cuda \
  --asr_device cuda:0 \
  --eval_device cuda:0 \
  --fail_if_exists
```

### Stage 1 Full

1. full split manifest를 `manifests/clean_test.jsonl`, `manifests/noisy_test.jsonl`, `manifests/accented_test.jsonl`, `manifests/dysarthric_test.jsonl` schema에 맞게 고정한다.
2. 각 split에 대해 `hard`, `oracle`, `length_only`, `soft_ctc`를 같은 seed/checkpoint/posterior source로 실행한다.
3. evaluation ASR로 generated wav transcript를 만든다.
4. `eval_posterior_f5.py`로 WER/CER/Sub/Del/Ins를 집계한다.
5. subset별 paired bootstrap으로 유의성을 확인한다.
6. 실패 case를 deletion, repetition, ASR entropy 기준으로 분해한다.

full run template:

```bash
PYTHONPATH=src .venv/bin/python platform/workers/run_posterior_f5_pipeline.py \
  --run_id stage1_full_<subset> \
  --manifest manifests/<subset>.jsonl \
  --mode hard \
  --mode oracle \
  --mode length_only \
  --mode soft_ctc \
  --checkpoint_id f5tts_v1_base_hf \
  --run_posterior_extraction \
  --ctc_model facebook/wav2vec2-base-960h \
  --run_inference \
  --run_prediction \
  --run_metrics \
  --asr_device cuda:0 \
  --infer_device cuda \
  --eval_device cuda:0 \
  --fail_if_exists
```

paper table:

```bash
PYTHONPATH=src .venv/bin/python src/f5_tts/eval/make_result_tables.py \
  --summary mlops_artifacts/runs/<run_id>/metrics/summary.json \
  --output_csv results/tables/<run_id>_main.csv \
  --output_md results/tables/<run_id>_main.md
```

bootstrap:

```bash
PYTHONPATH=src .venv/bin/python src/f5_tts/eval/bootstrap_significance.py \
  --per_utterance mlops_artifacts/runs/<run_id>/metrics/per_utterance.jsonl \
  --baseline hard \
  --candidate soft_ctc \
  --metric wer \
  --samples 10000 \
  --seed 1234 \
  --output results/tables/<run_id>_hard_vs_soft_ctc_wer.bootstrap.json
```

### Stage 2 확장

Stage 1 결과를 freeze한 뒤 진행한다.

1. `posterior_encoder` training loop와 inference mode를 연결한다.
2. real SSL reference cache를 만들고 `hybrid`가 soft text와 SSL branch를 실제로 mix하게 한다.
3. `soft_ctc` vs `posterior_encoder`, `soft_ctc` vs `hybrid` ablation을 같은 manifest와 seed로 추가 실행한다.

posterior encoder smoke:

```bash
PYTHONPATH=src .venv/bin/python src/f5_tts/train/train_posterior.py \
  --synthetic_smoke \
  --output_dir /tmp/posterior_encoder_synthetic_smoke \
  --max_steps 20
```

SSL cache extraction:

```bash
PYTHONPATH=src .venv/bin/python src/f5_tts/scripts/extract_ssl_reference.py \
  --manifest manifests/<subset>.jsonl \
  --output ssl_cache/<subset>.ssl.jsonl \
  --shard_dir ssl_cache/<subset>_npz \
  --model microsoft/wavlm-base-plus \
  --device cuda:0
```

Stage 2 run template:

```bash
PYTHONPATH=src .venv/bin/python platform/workers/run_posterior_f5_pipeline.py \
  --run_id stage2_full_<subset> \
  --manifest manifests/<subset>.jsonl \
  --mode posterior_encoder \
  --mode hybrid \
  --checkpoint_id f5tts_v1_base_hf \
  --posterior_encoder_ckpt ckpts/posterior_encoder/dev_latest/model_last.pt \
  --ssl_cache ssl_cache/<subset>.ssl.jsonl \
  --run_posterior_extraction \
  --ctc_model facebook/wav2vec2-base-960h \
  --run_inference \
  --run_prediction \
  --run_metrics \
  --asr_device cuda:0 \
  --infer_device cuda \
  --eval_device cuda:0 \
  --fail_if_exists
```

## Freeze Checklist

논문 table에 들어가는 run은 [reproducibility_checklist.md](reproducibility_checklist.md)를 채운 뒤 freeze한다.
