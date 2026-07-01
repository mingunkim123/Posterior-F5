# Posterior-F5 Performance Optimization Notes

Last updated: 2026-07-02

이 문서는 Posterior-F5 MLOps 파이프라인에서 생성, 음향 평가, ASR 예측, metric 집계 단계의 성능 병목과 현재 적용한 개선을 정리한다. 목적은 실험 시간이 길어졌을 때 "정상적으로 무거운 단계"와 "코드 또는 설정 문제로 느린 단계"를 구분하는 것이다.

## Current Baseline Observation

최근 로컬 artifact `mlops_artifacts/runs/run_20260701_194044` 기준:

| Stage | Scope | Observation |
| --- | --- | --- |
| Inference | 4 modes x 50 utterances | 모두 성공 |
| Hard generation | 50 utterances | total 210.82s, mean 4.22s |
| Oracle generation | 50 utterances | total 210.27s, mean 4.21s |
| Length-only generation | 50 utterances | total 216.77s, mean 4.34s |
| Soft-CTC generation | 50 utterances | total 216.96s, mean 4.34s |
| Audio metrics | speaker + UTMOS requested | `speaker_checkpoint` 누락으로 실패 |

이 run은 실제로 음향 평가가 계속 오래 실행 중인 상태가 아니었다. `logs/job.stderr.log`에 `Speaker similarity requires --speaker_checkpoint`가 기록되어 있었고, worker 프로세스는 이미 종료된 상태였다. 문제는 stage 예외 시 `run.json`이 `running`으로 남을 수 있는 상태 갱신 경로였다.

## What Is Expected To Be Slow

음향 평가는 켠 옵션에 따라 비용이 크게 달라진다.

| Metric | Cost profile | Notes |
| --- | --- | --- |
| Duration, RTF | Cheap | wav header 또는 metadata read 중심 |
| Speaker similarity | Heavy | generated/ref audio load, resample, speaker encoder inference |
| UTMOS | Heavy | SpeechMOS model load and inference |
| WER/CER metrics | Cheap | 이미 생성된 prediction JSONL 비교 |
| ASR prediction | Heavy | Whisper 또는 ASR model inference |

따라서 duration/RTF만 계산하는데 생성보다 오래 걸리면 병목으로 봐야 한다. 반대로 speaker similarity, UTMOS, Whisper prediction을 포함하면 평가가 생성보다 길어지는 것은 충분히 가능하다.

## Implemented Improvements

### 1. Batch inference keeps F5 runtime alive

생성은 이미 `src/f5_tts/infer/infer_batch_jsonl.py`를 통해 모드마다 F5 model과 vocoder를 한 번만 로드하고 여러 utterance를 처리한다.

효과:

- 기존 CLI 반복 실행 대비 checkpoint, vocoder, config 로드 비용을 utterance마다 지불하지 않는다.
- `commands.jsonl`에 per-utterance elapsed time, output hash, status를 남긴다.
- 현재 run 기준 평균 생성 시간은 약 4.2초대이다.

관련 파일:

- `platform/workers/run_posterior_f5_pipeline.py`
- `src/f5_tts/infer/infer_batch_jsonl.py`

### 2. Audio metrics now reuse scorer instances across modes

기존 구조는 `run_audio_metrics_stage`가 모드마다 `eval_audio_metrics.py`를 별도 subprocess로 실행했다. speaker similarity 또는 UTMOS가 켜지면 모드마다 평가 모델을 다시 로드할 수 있었다.

개선 후:

- `run_audio_metrics_stage`가 worker 프로세스 안에서 `evaluate_audio_metrics`를 직접 호출한다.
- `SpeakerSimilarityScorer`와 `UtmosScorer`를 stage 시작 시 한 번만 만들고 모든 모드에서 재사용한다.
- 기존 artifact 계약은 유지한다.
  - `metrics/<mode>.audio_metrics.json`
  - `metrics/<mode>.audio_metrics.jsonl`
  - `logs/audio_metrics_<mode>.log`
- 로그에는 `DIRECT_CALL: true`와 `MODEL_REUSE`가 기록된다.

효과:

- 4개 모드 실행 시 speaker/UTMOS model load 비용을 최대 4회에서 1회로 줄인다.
- subprocess startup 비용과 model import 반복을 줄인다.

관련 파일:

- `platform/workers/run_posterior_f5_pipeline.py`

### 3. Speaker similarity caches reference embeddings

speaker similarity는 각 row마다 generated audio와 reference audio를 speaker encoder에 넣는다. 평가셋이 같은 reference audio를 반복해서 쓰면 reference embedding을 매번 다시 계산하는 낭비가 컸다.

개선 후:

- reference audio cache key는 resolved path, file size, mtime을 사용한다.
- 같은 reference audio는 embedding을 한 번만 계산하고 재사용한다.
- sample rate별 `torchaudio.transforms.Resample` 객체도 재사용한다.
- inference path는 `torch.inference_mode()`를 사용한다.

효과:

- 같은 prompt/reference를 여러 utterance와 mode에서 재사용하는 smoke, hard, paper eval에서 speaker similarity 비용이 줄어든다.
- generated audio는 대부분 unique이므로 cache하지 않아 GPU memory 증가를 피한다.

관련 파일:

- `src/f5_tts/eval/eval_audio_metrics.py`

### 4. Stage exceptions now mark the run as failed

기존에는 stage runner 내부에서 예외가 발생하면 `run.json`이 `running` 상태로 남을 수 있었다. 이 때문에 설정 오류가 실제 장기 실행처럼 보일 수 있었다.

개선 후:

- `_execute_stage`가 runner exception을 잡아 failed stage payload를 기록한다.
- `run.json`의 top-level status도 `failed`로 갱신한다.
- `error_message`에 실제 예외 메시지를 남긴다.

효과:

- `speaker_checkpoint` 누락 같은 preflight 오류가 dashboard/API에서 명확히 실패로 보인다.
- "평가가 오래 걸리는지"와 "이미 실패했는데 상태만 running인지"를 구분할 수 있다.

관련 파일:

- `platform/workers/run_posterior_f5_pipeline.py`
- `tests/test_mlops_run_scaffold.py`

## Recommended Run Modes

빠른 개발 루프:

- `run_audio_metrics=true`
- `report_rtf=true`
- `report_speaker_similarity=false`
- `report_utmos=false`
- `run_prediction=false` 또는 `prediction_dry_run=true`

논문용 또는 최종 평가:

- `run_audio_metrics=true`
- `report_rtf=true`
- `report_speaker_similarity=true`
- `speaker_checkpoint` 필수
- `report_utmos=true`
- `run_prediction=true`
- `eval_device=cuda:0`
- 가능하면 `eval_torch_dtype=float16`

주의:

- `report_speaker_similarity=true`인데 `speaker_checkpoint`가 비어 있으면 실패해야 정상이다.
- UTMOS는 `torch.hub.load("tarepan/SpeechMOS:v1.2.0", ...)`를 사용하므로 첫 실행에서 다운로드와 import 비용이 크다.
- duration/RTF만 필요하면 heavy metrics를 끄는 것이 맞다.

## Remaining Bottlenecks

아직 남아 있는 병목과 다음 개선 후보:

| Area | Current behavior | Possible next step |
| --- | --- | --- |
| Inference modes | 모드 단위 직렬 실행 | GPU memory가 허용하면 mode parallelism 검토 |
| ASR prediction | ASR model은 한 번 로드하지만 wav는 순차 처리 | pipeline list input 또는 batched file prediction 검토 |
| UTMOS | wav별 순차 inference | batching 가능 여부 확인 |
| Speaker generated embedding | generated audio는 매번 계산 | mode 비교에서 같은 generated output 재사용이 생기면 cache 고려 |
| Artifact status | 실패 상태 갱신 개선됨 | dashboard에서 failed reason을 더 눈에 띄게 표시 |

## Quick Diagnosis Checklist

평가가 생성보다 오래 걸릴 때 확인 순서:

1. `run.json`의 top-level `status`와 해당 stage status를 본다.
2. `logs/job.stderr.log`와 `logs/audio_metrics_<mode>.log`를 확인한다.
3. `run_speaker_similarity`, `run_utmos`, `run_prediction`이 켜져 있는지 확인한다.
4. `speaker_checkpoint`가 비어 있지 않은지 확인한다.
5. `nvidia-smi`와 `torch.cuda.is_available()`로 평가 프로세스가 GPU를 볼 수 있는지 확인한다.
6. duration/RTF만 켰는데 느리면 file I/O 또는 path resolution 문제를 의심한다.

