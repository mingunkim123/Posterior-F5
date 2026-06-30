# FastAPI + Python Workers + React MLOps Platform 구축 문서

이 문서는 Posterior-F5 실험을 시작점으로 삼아, 나만의 MLOps 플랫폼을 FastAPI, Python workers, React dashboard로 구축하는 방법을 정리한다.

목표는 단순한 실험 결과 페이지가 아니다. 목표는 아래 흐름을 하나의 플랫폼 안에서 관리하는 것이다.

```text
experiment config
  -> job submission
  -> posterior extraction
  -> TTS inference
  -> evaluation
  -> metrics aggregation
  -> dashboard
  -> paper table / report
```

처음에는 Posterior-F5 전용으로 작게 만들고, 나중에는 다른 speech/ML 프로젝트도 올릴 수 있는 일반 MLOps 플랫폼으로 확장한다.

## 1. 왜 FastAPI First인가

이 프로젝트의 핵심 작업은 Python 안에 있다.

| 작업 | 현재 언어/런타임 |
| --- | --- |
| ASR posterior extraction | Python, transformers, torch |
| F5-TTS inference | Python, torch |
| vocoder decode | Python, torch |
| WER/CER evaluation | Python |
| metrics aggregation | Python |
| checkpoint handling | Python |

그래서 첫 MVP는 FastAPI가 가장 자연스럽다. NestJS는 나중에 사용자, 조직, 권한, 결제, 팀 기능 같은 product backend가 커질 때 API gateway로 붙이면 된다.

추천 방향은 다음과 같다.

```text
Phase 1: FastAPI monolith + local filesystem + local workers
Phase 2: FastAPI + Redis queue + Python worker processes + Postgres
Phase 3: React dashboard + artifact browser + run comparison
Phase 4: S3/MinIO artifact store + distributed GPU workers
Phase 5: optional NestJS gateway for product/backend concerns
```

## 2. 전체 아키텍처

```mermaid
flowchart TD
    UI[React Dashboard] --> API[FastAPI Control Plane]
    API --> DB[(Postgres or SQLite)]
    API --> Q[Redis Queue]
    API --> FS[Artifact Store]

    Q --> W1[Posterior Worker]
    Q --> W2[Inference Worker]
    Q --> W3[Evaluation Worker]
    Q --> W4[Aggregation Worker]

    W1 --> FS
    W2 --> FS
    W3 --> FS
    W4 --> FS

    W1 --> DB
    W2 --> DB
    W3 --> DB
    W4 --> DB

    FS --> UI
    DB --> API
```

역할을 나누면 이렇다.

| 컴포넌트 | 책임 |
| --- | --- |
| React dashboard | 실험 생성, job 상태 확인, metric 비교, audio inspection |
| FastAPI | API, auth, run metadata, job submission, artifact URL 제공 |
| Python workers | GPU/CPU 실험 작업 실행 |
| Redis queue | 비동기 job 전달 |
| Postgres | experiment/run/job/metric metadata 저장 |
| Artifact store | wav, posterior cache, prediction JSONL, metrics JSON, checkpoint 저장 |

초기에는 Redis/Postgres 없이도 시작할 수 있다. SQLite와 local subprocess worker로 MVP를 만들고, 구조가 잡히면 Redis/Postgres로 갈아타면 된다.

## 3. 플랫폼이 관리해야 하는 핵심 객체

```mermaid
erDiagram
    PROJECT ||--o{ EXPERIMENT : has
    EXPERIMENT ||--o{ RUN : has
    RUN ||--o{ JOB : schedules
    RUN ||--o{ ARTIFACT : produces
    RUN ||--o{ METRIC : reports
    RUN ||--o{ RUN_MODE : compares
    RUN_MODE ||--o{ UTTERANCE_RESULT : contains
    UTTERANCE ||--o{ UTTERANCE_RESULT : evaluated_as
```

가장 중요한 도메인 모델은 다음이다.

| 객체 | 의미 |
| --- | --- |
| `Project` | Posterior-F5 같은 큰 연구/제품 단위 |
| `Experiment` | 하나의 연구 질문 또는 ablation 묶음 |
| `Run` | 특정 config로 실제 돌린 실행 단위 |
| `RunMode` | `hard`, `oracle`, `length_only`, `soft_ctc` 같은 비교 mode |
| `Job` | worker가 수행하는 비동기 작업 |
| `Artifact` | 파일 산출물 |
| `Metric` | run-level metric |
| `UtteranceResult` | utterance별 generated wav, prediction, error breakdown |

## 4. 권장 저장소 구조

처음에는 현재 `F5-TTS` repo 안에 아래처럼 넣는 것을 추천한다.

```text
F5-TTS/
  platform/
    backend/
      app/
        main.py
        api/
          routes_experiments.py
          routes_runs.py
          routes_jobs.py
          routes_artifacts.py
        core/
          config.py
          paths.py
        db/
          models.py
          session.py
          migrations/
        schemas/
          experiment.py
          run.py
          job.py
          metric.py
        services/
          run_service.py
          job_service.py
          artifact_service.py
          metric_service.py
        workers/
          tasks.py
          local_runner.py
          queue.py
      pyproject.toml

    frontend/
      package.json
      src/
        App.tsx
        api/
          client.ts
        pages/
          RunsPage.tsx
          RunDetailPage.tsx
          ComparePage.tsx
          ArtifactPage.tsx
        components/
          RunStatusMatrix.tsx
          MetricBars.tsx
          PipelineGraph.tsx
          UtteranceInspector.tsx

    workers/
      extract_posterior_worker.py
      inference_worker.py
      eval_worker.py
      aggregate_worker.py

  mlops_artifacts/
    runs/
      <run_id>/
        run.json
        config.yaml
        posterior_cache/
        generated/
        predictions/
        metrics/
        dashboard_snapshot/
```

나중에 플랫폼을 독립 repo로 분리하고 싶으면 `platform/`만 떼어내면 된다.

## 5. Artifact Contract

MLOps 플랫폼에서 제일 먼저 고정해야 하는 것은 API가 아니라 artifact contract다. 어떤 worker가 돌든 결과가 같은 위치와 형식으로 남아야 dashboard와 논문 script가 흔들리지 않는다.

권장 run artifact 구조:

```text
mlops_artifacts/runs/<run_id>/
  run.json
  config.yaml
  manifest.jsonl

  posterior_cache/
    run.posterior.jsonl
    posterior_npz/
      <utterance_id>.npz

  generated/
    hard/
      <utterance_id>.wav
    oracle/
      <utterance_id>.wav
    length_only/
      <utterance_id>.wav
    soft_ctc/
      <utterance_id>.wav

  predictions/
    hard.jsonl
    oracle.jsonl
    length_only.jsonl
    soft_ctc.jsonl

  metrics/
    hard.metrics.json
    oracle.metrics.json
    length_only.metrics.json
    soft_ctc.metrics.json
    summary.json
    summary.csv
    per_utterance.jsonl

  logs/
    extract_posterior.log
    inference_hard.log
    inference_soft_ctc.log
    evaluation.log
```

`run.json` 예시:

```json
{
  "run_id": "run_20260701_001",
  "project": "Posterior-F5",
  "experiment": "baseline_dev_small",
  "status": "completed",
  "created_at": "2026-07-01T10:00:00+09:00",
  "modes": ["hard", "oracle", "length_only", "soft_ctc"],
  "manifest": "manifest.jsonl",
  "posterior_file": "posterior_cache/run.posterior.jsonl",
  "model": {
    "name": "F5TTS_v1_Base",
    "checkpoint": "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "vocoder": "vocos"
  },
  "generation": {
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "speed": 1.0,
    "seed": 1234
  },
  "evaluation": {
    "posterior_asr": "facebook/wav2vec2-base-960h",
    "eval_asr": "openai/whisper-large-v3-turbo"
  }
}
```

## 6. Backend 설계

FastAPI는 control plane이다. heavy job을 API request 안에서 직접 실행하지 않는다. API는 metadata를 저장하고 job을 queue에 넣고, worker가 실제 작업을 수행한다.

### 6.1 API Endpoint MVP

| Method | Path | 역할 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 |
| `POST` | `/projects` | project 생성 |
| `GET` | `/projects` | project 목록 |
| `POST` | `/experiments` | experiment 생성 |
| `GET` | `/experiments/{experiment_id}` | experiment 상세 |
| `POST` | `/runs` | run 생성 |
| `GET` | `/runs` | run 목록 |
| `GET` | `/runs/{run_id}` | run 상세 |
| `POST` | `/runs/{run_id}/start` | pipeline job 제출 |
| `POST` | `/runs/{run_id}/cancel` | run 취소 |
| `GET` | `/runs/{run_id}/jobs` | job 상태 |
| `GET` | `/runs/{run_id}/metrics` | summary metrics |
| `GET` | `/runs/{run_id}/utterances` | utterance별 결과 |
| `GET` | `/artifacts/{artifact_id}` | artifact 다운로드 또는 signed URL |

### 6.2 Run 생성 request

```json
{
  "project_id": "posterior_f5",
  "experiment_name": "baseline_dev_small",
  "manifest_path": "manifests/dev_small.jsonl",
  "modes": ["hard", "oracle", "length_only", "soft_ctc"],
  "model": "F5TTS_v1_Base",
  "generation": {
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "speed": 1.0,
    "seed": 1234
  },
  "posterior": {
    "ctc_model": "facebook/wav2vec2-base-960h",
    "top_k": 8,
    "language": "en"
  },
  "evaluation": {
    "eval_asr": "openai/whisper-large-v3-turbo"
  }
}
```

### 6.3 Database Table 초안

처음에는 SQLAlchemy 또는 SQLModel 중 하나를 쓰면 된다. FastAPI와 같이 쓰기에는 SQLModel이 단순하지만, 장기적으로는 SQLAlchemy + Alembic이 더 표준적이다.

```text
projects
  id
  name
  description
  created_at

experiments
  id
  project_id
  name
  description
  created_at

runs
  id
  experiment_id
  name
  status
  config_json
  artifact_root
  created_at
  started_at
  finished_at

run_modes
  id
  run_id
  mode
  status
  generated_dir
  predictions_path
  metrics_path

jobs
  id
  run_id
  kind
  status
  queue_id
  progress
  error_message
  started_at
  finished_at

artifacts
  id
  run_id
  kind
  path
  media_type
  size_bytes
  created_at

metrics
  id
  run_id
  mode
  subset
  key
  value

utterance_results
  id
  run_id
  mode
  utterance_id
  subset
  wav_path
  prediction
  reference
  wer
  cer
  substitutions
  deletions
  insertions
  posterior_entropy
```

## 7. Worker 설계

Workers는 실제 ML 작업을 수행한다. API 서버와 분리하는 이유는 세 가지다.

1. GPU 작업은 오래 걸린다.
2. 실패와 재시도가 필요하다.
3. API 서버는 빠르게 응답해야 한다.

### 7.1 Worker 종류

| worker | 입력 | 출력 |
| --- | --- | --- |
| posterior worker | manifest, ASR config | posterior JSONL, NPZ shards |
| inference worker | manifest, mode, checkpoint, posterior cache | generated wav |
| evaluation worker | generated wav dir, eval ASR config | predictions JSONL |
| aggregation worker | manifest, predictions, metrics JSON | summary CSV/JSON, per-utterance JSONL |

### 7.2 Job DAG

```mermaid
flowchart TD
    A[create run] --> B[extract_posterior]
    B --> C1[infer hard]
    B --> C2[infer oracle]
    B --> C3[infer length_only]
    B --> C4[infer soft_ctc]
    C1 --> D1[evaluate hard]
    C2 --> D2[evaluate oracle]
    C3 --> D3[evaluate length_only]
    C4 --> D4[evaluate soft_ctc]
    D1 --> E[aggregate metrics]
    D2 --> E
    D3 --> E
    D4 --> E
    E --> F[run completed]
```

`hard`와 `oracle`은 posterior cache 없이도 돌릴 수 있지만, 하나의 run contract를 위해 posterior extraction을 먼저 공통으로 수행하는 것이 관리하기 쉽다.

### 7.3 Local MVP Worker

처음부터 Celery를 붙이지 않아도 된다. MVP는 FastAPI가 job row를 만들고, `platform/workers/local_runner.py`가 pending job을 polling해서 subprocess로 실행해도 충분하다.

```text
python platform/workers/local_runner.py
```

처음 MVP에서 필요한 상태:

| status | 의미 |
| --- | --- |
| `queued` | job 생성됨 |
| `running` | worker가 실행 중 |
| `succeeded` | 산출물 생성 완료 |
| `failed` | error 발생 |
| `cancelled` | 사용자가 취소 |

Phase 2에서는 Redis + RQ 또는 Celery로 바꾼다.

| 선택지 | 장점 | 단점 |
| --- | --- | --- |
| RQ | 단순함, Redis만 있으면 됨 | 복잡한 workflow에는 약함 |
| Celery | 성숙함, retry/routing 풍부 | 설정이 무거움 |
| Dramatiq | 깔끔하고 가벼움 | 생태계는 Celery보다 작음 |
| Prefect | workflow UI와 DAG 관리 좋음 | 플랫폼 내부 control plane과 역할 중복 가능 |

이 프로젝트는 처음에는 RQ 또는 local runner가 가장 현실적이다.

## 8. Posterior-F5 Worker Command Mapping

플랫폼 job은 결국 현재 repo의 Python script를 호출한다.

### 8.1 Posterior extraction

```bash
PYTHONPATH=src python3 src/f5_tts/scripts/extract_asr_posterior.py \
  --manifest mlops_artifacts/runs/<run_id>/manifest.jsonl \
  --output mlops_artifacts/runs/<run_id>/posterior_cache/run.posterior.jsonl \
  --shard_dir mlops_artifacts/runs/<run_id>/posterior_cache/posterior_npz \
  --ctc_model facebook/wav2vec2-base-960h \
  --top_k 8 \
  --language en
```

### 8.2 Mode별 inference

`hard`:

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio <ref_audio> \
  --ref_text "" \
  --gen_text "<gen_text>" \
  --ref_text_mode hard \
  --output_dir mlops_artifacts/runs/<run_id>/generated/hard \
  --output_file <utterance_id>.wav \
  --seed 1234
```

`oracle`:

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio <ref_audio> \
  --ref_text "<oracle_ref_text>" \
  --gen_text "<gen_text>" \
  --ref_text_mode hard \
  --output_dir mlops_artifacts/runs/<run_id>/generated/oracle \
  --output_file <utterance_id>.wav \
  --seed 1234
```

`length_only`:

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio <ref_audio> \
  --ref_text "" \
  --gen_text "<gen_text>" \
  --ref_text_mode length_only \
  --posterior_file mlops_artifacts/runs/<run_id>/posterior_cache/run.posterior.jsonl \
  --output_dir mlops_artifacts/runs/<run_id>/generated/length_only \
  --output_file <utterance_id>.wav \
  --seed 1234
```

`soft_ctc`:

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio <ref_audio> \
  --ref_text "" \
  --gen_text "<gen_text>" \
  --ref_text_mode soft_ctc \
  --posterior_file mlops_artifacts/runs/<run_id>/posterior_cache/run.posterior.jsonl \
  --output_dir mlops_artifacts/runs/<run_id>/generated/soft_ctc \
  --output_file <utterance_id>.wav \
  --seed 1234
```

대량 실행에서는 CLI를 utterance마다 호출하는 방식이 느릴 수 있다. MVP는 단순함을 우선하고, 나중에 batch runner를 만들어 model/vocoder를 한 번만 로드하게 최적화한다.

### 8.3 Evaluation

평가 worker는 generated wav를 평가 ASR로 transcribe해서 mode별 prediction JSONL을 만든다.

```json
{"utterance_id":"utt-0001","hypothesis":"recognized text"}
```

그 다음 기존 metric script를 호출한다.

```bash
PYTHONPATH=src python3 src/f5_tts/eval/eval_posterior_f5.py \
  --manifest mlops_artifacts/runs/<run_id>/manifest.jsonl \
  --predictions mlops_artifacts/runs/<run_id>/predictions/soft_ctc.jsonl \
  --output mlops_artifacts/runs/<run_id>/metrics/soft_ctc.metrics.json \
  --mode soft_ctc \
  --posterior_file mlops_artifacts/runs/<run_id>/posterior_cache/run.posterior.jsonl
```

## 9. React Dashboard 설계

React dashboard는 연구자에게 필요한 세 가지를 해결해야 한다.

1. 지금 어떤 실험이 어디까지 돌았는지 본다.
2. mode별 metric 차이를 바로 비교한다.
3. 실패한 utterance를 빠르게 찾아서 듣고 읽는다.

### 9.1 화면 구성

| Page | 역할 |
| --- | --- |
| Runs | run 목록, 상태, 최근 metric |
| New Run | manifest, mode, model, posterior/eval ASR config 입력 |
| Run Detail | pipeline status, mode별 artifact, logs |
| Compare | run/mode/subset별 metric 비교 |
| Utterance Inspector | audio/text/prediction/error breakdown 비교 |
| Artifacts | posterior cache, wav, JSON, CSV 다운로드 |

### 9.2 Run Detail Layout

```text
Run Header
  status, run_id, experiment, created_at, checkpoint

Pipeline
  extract posterior -> infer modes -> eval modes -> aggregate

Metric Overview
  WER, CER, Sub, Del, Ins, speaker similarity, UTMOS

Mode Matrix
  rows: utterances
  columns: hard/oracle/length_only/soft_ctc
  cells: done/failed + WER/CER

Utterance Inspector
  reference audio
  generated audio per mode
  reference text
  generated target text
  ASR prediction per mode
  posterior entropy
  error breakdown
```

### 9.3 시각화 컴포넌트

| 컴포넌트 | 내용 |
| --- | --- |
| `PipelineGraph` | job DAG와 상태 |
| `RunStatusMatrix` | subset x mode status |
| `MetricBars` | mode별 WER/CER/Sub/Del/Ins |
| `OracleGapChart` | oracle gap recovered |
| `EntropyBucketChart` | entropy bucket별 개선량 |
| `UtteranceInspector` | audio player와 text 비교 |
| `FailureCaseTable` | deletion, insertion, entropy 기준 정렬 |

처음에는 Recharts 또는 ECharts 중 하나를 쓰면 된다. UI framework는 Tailwind + shadcn/ui 조합이 빠르고 깔끔하다.

## 10. MVP 개발 순서

가장 좋은 순서는 아래다.

### Phase 0. Artifact contract 고정

목표:

```text
mlops_artifacts/runs/<run_id>/
```

아래 파일이 반드시 생기게 한다.

```text
run.json
manifest.jsonl
posterior_cache/run.posterior.jsonl
generated/<mode>/<utterance_id>.wav
predictions/<mode>.jsonl
metrics/<mode>.metrics.json
metrics/summary.csv
```

### Phase 1. CLI batch runner

FastAPI 전에 먼저 Python runner를 만든다.

```text
platform/workers/run_posterior_f5_pipeline.py
```

역할:

1. run directory 생성
2. manifest copy
3. posterior extraction 실행
4. mode별 inference 실행
5. evaluation 실행
6. metrics aggregate
7. `run.json` status update

이 단계만 완성해도 dashboard 없이 논문 실험 자동화가 시작된다.

### Phase 2. FastAPI API

FastAPI로 run metadata를 관리한다.

```bash
uvicorn platform.backend.app.main:app --reload --port 8000
```

처음 endpoint:

```text
POST /runs
POST /runs/{run_id}/start
GET /runs
GET /runs/{run_id}
GET /runs/{run_id}/metrics
GET /runs/{run_id}/utterances
```

### Phase 3. Local worker

`/runs/{run_id}/start`가 job을 만들고, local worker가 pending job을 처리한다.

```bash
python platform/backend/app/workers/local_runner.py
```

### Phase 4. React dashboard

```bash
cd platform/frontend
npm create vite@latest . -- --template react-ts
npm install
npm run dev
```

처음 구현할 화면:

1. Runs table
2. Run detail
3. Metric overview
4. Utterance inspector

### Phase 5. Queue와 DB 교체

MVP가 돌아가면:

| MVP | 확장 |
| --- | --- |
| SQLite | Postgres |
| local polling worker | Redis + RQ/Celery |
| local filesystem | S3/MinIO |
| single machine | GPU worker pool |

## 11. 개발 환경

초기 `.env` 예시:

```bash
MLOPS_ENV=dev
MLOPS_ARTIFACT_ROOT=/Users/mingun/Posterior-F5/F5-TTS/mlops_artifacts
DATABASE_URL=sqlite:///./platform.db
REDIS_URL=redis://localhost:6379/0
F5_TTS_ROOT=/Users/mingun/Posterior-F5/F5-TTS
PYTHONPATH=/Users/mingun/Posterior-F5/F5-TTS/src
```

나중에 Postgres로 바꿀 때:

```bash
DATABASE_URL=postgresql+psycopg://posterior:posterior@localhost:5432/posterior_ops
```

## 12. Docker Compose 방향

MVP 이후에는 아래 구성을 쓴다.

```text
docker-compose.yml
  api
  frontend
  redis
  postgres
  worker-cpu
  worker-gpu
  minio
```

처음부터 Docker로 모든 것을 묶으면 GPU, torch, torchaudio, audio codec 때문에 디버깅이 무거워질 수 있다. 먼저 local native로 안정화하고, 그 다음 Docker compose로 고정하는 편이 낫다.

## 13. 장기 확장: NestJS를 붙이는 경우

NestJS는 지금 당장은 필요하지 않다. 하지만 플랫폼이 연구 도구를 넘어 제품이 되면 유용해진다.

NestJS가 들어갈 만한 지점:

```text
React Dashboard
  -> NestJS API Gateway
  -> FastAPI ML Control Plane
  -> Python Workers
```

NestJS가 맡을 수 있는 책임:

| 책임 | 이유 |
| --- | --- |
| user/org/project auth | TypeScript product backend로 관리하기 좋음 |
| billing/plan/quotas | ML 코드와 분리하는 것이 좋음 |
| websocket notification | gateway layer에서 처리 가능 |
| public API gateway | 내부 FastAPI 구조를 숨김 |
| admin dashboard | product-level operation |

FastAPI는 계속 ML control plane으로 남긴다. ML 실행과 artifact contract는 Python 쪽이 중심이어야 한다.

## 14. 보안과 재현성

MLOps 플랫폼은 “돌아간다”보다 “나중에 믿을 수 있다”가 중요하다.

반드시 기록할 것:

| 항목 | 이유 |
| --- | --- |
| git commit hash | 어떤 코드로 돌렸는지 |
| config snapshot | 하이퍼파라미터 재현 |
| checkpoint path/hash | 모델 weight 재현 |
| manifest snapshot | 평가 데이터 고정 |
| posterior ASR source | posterior 생성 모델 고정 |
| evaluation ASR source | metric 생성 모델 고정 |
| seed | stochastic generation 추적 |
| environment | torch/transformers 버전 차이 기록 |

`run.json`에 최소한 아래를 넣는다.

```json
{
  "git": {
    "commit": "6cd38dd",
    "branch": "feature/posterior-aware-f5",
    "dirty": false
  },
  "environment": {
    "python": "3.11",
    "torch": "2.x",
    "transformers": "x.y.z"
  }
}
```

## 15. 논문 작업과 연결

이 플랫폼이 논문에 직접 기여하는 지점은 다음이다.

| 플랫폼 기능 | 논문 산출물 |
| --- | --- |
| run comparison | main result table |
| utterance inspector | qualitative examples |
| entropy bucket chart | uncertainty analysis |
| failure ranking | error analysis |
| artifact snapshot | reproducibility appendix |
| config history | ablation tracking |

논문 figure로 바로 쓸 수 있는 export도 만들면 좋다.

```text
GET /runs/{run_id}/figures/main_table.csv
GET /runs/{run_id}/figures/error_breakdown.csv
GET /runs/{run_id}/figures/oracle_gap.csv
GET /runs/{run_id}/figures/entropy_buckets.csv
```

## 16. 첫 번째 구현 목표

가장 먼저 만들 것은 거대한 플랫폼이 아니다.

첫 목표는 이것이다.

> Posterior-F5 dev_small run 하나를 등록하고, hard/oracle/length_only/soft_ctc 결과를 웹에서 비교할 수 있게 한다.

구현 checklist:

1. `mlops_artifacts/runs/<run_id>/` contract 생성
2. `platform/workers/run_posterior_f5_pipeline.py` 작성
3. `platform/backend/app/main.py` FastAPI skeleton 작성
4. `POST /runs`, `POST /runs/{run_id}/start`, `GET /runs/{run_id}` 구현
5. local runner로 job 실행
6. React `RunsPage`, `RunDetailPage` 구현
7. wav audio player와 WER/CER bar chart 표시
8. `summary.csv` export

이것이 완성되면 그 다음은 자연스럽다.

```text
dev_small MVP
  -> full_eval run
  -> posterior_encoder mode
  -> hybrid mode
  -> paper figure export
  -> general MLOps platform
```

## 17. 추천 기술 스택

초기 MVP:

| 영역 | 추천 |
| --- | --- |
| API | FastAPI |
| DB | SQLite, 이후 Postgres |
| ORM | SQLAlchemy + Alembic |
| Queue | local runner, 이후 Redis + RQ |
| Frontend | React + TypeScript + Vite |
| UI | Tailwind + shadcn/ui |
| Chart | Recharts |
| Artifact | local filesystem |
| Audio | HTML audio element |
| Config | YAML + JSON snapshot |

확장 단계:

| 영역 | 추천 |
| --- | --- |
| Artifact store | MinIO or S3 |
| Workflow | Prefect or Celery canvas |
| Auth | Auth.js, Clerk, or custom gateway |
| Gateway | optional NestJS |
| Deployment | Docker Compose first, then Kubernetes only if needed |
| Observability | structured logs, OpenTelemetry later |

## 18. 핵심 원칙

이 플랫폼은 예쁜 UI보다 아래 원칙이 더 중요하다.

1. 모든 run은 재현 가능한 artifact directory를 남긴다.
2. worker는 API request 안에서 직접 오래 돌지 않는다.
3. dashboard는 DB만 믿지 않고 artifact도 열 수 있어야 한다.
4. 실험 config는 실행 시점에 snapshot으로 저장한다.
5. mode별 비교는 항상 같은 manifest와 같은 seed를 기준으로 한다.
6. posterior source ASR와 evaluation ASR는 분리해서 기록한다.
7. 처음부터 범용 MLOps를 만들지 말고 Posterior-F5를 완벽히 돌린 뒤 일반화한다.

좋은 첫 버전은 작고 선명하다. `dev_small` 하나가 끝까지 돌고, 웹에서 결과를 듣고, 보고, CSV로 뽑을 수 있으면 이미 플랫폼의 뼈대는 살아 있는 것이다.

## 19. 현재 구현 상태와 앞으로의 전체 Step

이 섹션은 실제 개발 진행표다. 위쪽 섹션들이 플랫폼의 설계 철학과 목표 구조를 설명한다면, 여기서는 앞으로 무엇을 어떤 순서로 만들지 정리한다.

현재 구현은 아래 지점까지 와 있다.

```text
Step 1-4   run artifact contract + pipeline scaffold
Step 5     metrics aggregation
Step 6     FastAPI backend
Step 7     React dashboard
Step 8     local Docker Compose platform
Step 9     Docker 실행 검증은 아직 로컬 Docker 환경에서 필요
Step 10    dashboard에서 run 생성
Step 11    job polling + logs 표시
Step 12    API와 worker 분리
```

현재 구조는 다음과 같다.

```mermaid
flowchart TD
    UI[React Dashboard] --> API[FastAPI API]
    API --> J[job.json queued]
    W[Python Worker] --> J
    W --> P[run_posterior_f5_pipeline.py]
    P --> A[mlops_artifacts/runs/run_id]
    API --> A
    A --> UI
```

아직 Redis, Postgres, MinIO는 붙이지 않았다. 이 선택은 의도적이다. 지금 단계의 우선순위는 “로컬에서 하나의 실험을 끝까지 생성, 실행, 확인, 비교하는 것”이다.

### Step 13. Docker Compose End-to-End Smoke Test

목표:

```text
docker compose up --build
  -> API health 확인
  -> dashboard 접속
  -> New Run 생성
  -> job queued 확인
  -> worker running/completed 확인
  -> logs 표시 확인
  -> mlops_artifacts/runs/<run_id>/ 생성 확인
```

해야 할 일:

1. Docker Desktop 실행 상태 확인
2. `docker compose up --build` 실행
3. `http://127.0.0.1:8000/health` 확인
4. `http://127.0.0.1:5174` 확인
5. scaffold-only run 생성
6. worker가 `job.json`을 `queued -> running -> completed`로 바꾸는지 확인
7. dashboard의 Job Logs 패널에서 `job.stdout.log`, `job.stderr.log` 표시 확인

성공 기준:

```text
mlops_artifacts/runs/<run_id>/
  job.json
  run.json
  config.yaml
  logs/job.stdout.log
  logs/job.stderr.log
```

이 단계가 끝나야 “플랫폼이 실제로 켜진다”고 말할 수 있다.

### Step 14. Sample Manifest와 Demo Preset 추가

목표는 사용자가 매번 manifest path를 직접 기억하지 않아도, dashboard에서 바로 smoke run을 만들 수 있게 하는 것이다.

추가할 것:

```text
platform/samples/
  manifests/
    dev_smoke.jsonl
  README.md
```

dashboard에는 preset을 둔다.

| preset | 목적 |
| --- | --- |
| `scaffold_only` | artifact contract만 확인 |
| `posterior_text_only` | Whisper 없이 manifest text로 posterior 생성 |
| `dry_inference_plan` | inference command 생성만 확인 |
| `metrics_dry_run` | prediction/metrics 파일 경로 확인 |

성공 기준:

```text
대시보드에서 preset 선택
  -> manifest 자동 입력
  -> mode 자동 선택
  -> 안전한 dry-run 옵션 자동 설정
```

### Step 15. Run Detail UX 개선

Step11에서 logs는 보이지만, 아직 연구자가 보기 좋은 수준은 아니다. Step15는 “실패 원인 파악 시간을 줄이는 UI”를 만드는 단계다.

개선할 것:

1. stage별 상태를 더 명확히 표시
2. stage 클릭 시 관련 log만 필터링
3. error message를 상단에 표시
4. command line 복사 버튼
5. `run.json`, `config.yaml`, `summary.csv` 다운로드 링크
6. queued/running/completed/failed 색상 체계 정리

Run Detail은 아래처럼 읽혀야 한다.

```text
Run Header
  run_id, status, experiment, created_at, checkpoint

Pipeline
  scaffold -> posterior -> inference -> prediction -> metrics

Logs
  selected stage log
  stdout/stderr
  command

Artifacts
  run.json
  config.yaml
  predictions
  metrics
```

### Step 16. 실제 Inference Worker Image 준비

현재 Docker worker는 API와 같은 lightweight image를 쓴다. 이것은 scaffold/dry-run에는 좋지만, 실제 F5-TTS inference를 돌리기에는 부족할 수 있다.

Step16에서는 worker image를 분리한다.

```text
platform/docker/
  Dockerfile.api
  Dockerfile.frontend
  Dockerfile.worker-gpu
```

역할 분리:

| image | 역할 |
| --- | --- |
| `api` | FastAPI control plane |
| `frontend` | React dashboard static serving |
| `worker-gpu` | torch, torchaudio, F5-TTS inference, ASR evaluation |

worker image는 아래를 포함해야 한다.

```text
torch
torchaudio
ffmpeg
sox
libsndfile
transformers
vocos
F5-TTS local package
```

Docker Compose에서는 나중에 이렇게 나눈다.

```yaml
services:
  api:
    ...
  frontend:
    ...
  worker:
    build:
      dockerfile: platform/docker/Dockerfile.worker-gpu
    volumes:
      - ./mlops_artifacts:/app/mlops_artifacts
      - ./data:/app/data:ro
      - ./ckpts:/app/ckpts:ro
```

성공 기준:

```text
worker-gpu 컨테이너 안에서
python platform/workers/run_job_worker.py --once
실행 시 실제 inference dry-run이 아니라 wav 생성까지 가능
```

### Step 17. Checkpoint Registry 추가

논문 실험에서는 어떤 checkpoint를 썼는지가 매우 중요하다. 단순 path 문자열만 있으면 나중에 재현성이 약해진다.

추가할 파일:

```text
platform/config/checkpoints.yaml
```

예시:

```yaml
checkpoints:
  f5tts_v1_base_hf:
    model: F5TTS_v1_Base
    path: hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
    vocoder: vocos
    notes: upstream base checkpoint

  posterior_f5_finetune_001:
    model: F5TTS_v1_Base
    path: ckpts/posterior_f5/run_001/model_last.pt
    vocoder: vocos
    git_commit: ...
    dataset: ...
```

dashboard의 New Run 폼은 checkpoint path 직접 입력 대신 registry 선택을 지원한다.

성공 기준:

```text
run.json
  model.checkpoint_id
  model.checkpoint_path
  model.checkpoint_hash
```

### Step 18. Experiment Comparison View 강화

지금 dashboard는 하나의 run을 보는 쪽에 가깝다. 논문을 쓰려면 여러 run을 비교해야 한다.

추가할 화면:

```text
Compare
  rows: run_id / experiment / checkpoint
  columns: mode별 WER, CER, Sub, Del, Ins
  filters: subset, seed, checkpoint, mode
```

필요한 API:

```text
GET /experiments
GET /experiments/{experiment_id}/runs
GET /compare?run_ids=...
```

처음에는 DB 없이 `mlops_artifacts/runs/*/metrics/summary.json`을 읽어서 비교하면 된다.

성공 기준:

```text
hard vs oracle vs length_only vs soft_ctc
여러 run의 WER/CER를 한 화면에서 정렬/비교
```

### Step 19. Paper Export 기능

플랫폼이 논문에 직접 도움이 되려면, dashboard에서 본 결과를 논문 표/그림으로 바로 빼낼 수 있어야 한다.

추가할 export:

```text
GET /runs/{run_id}/exports/main_table.csv
GET /runs/{run_id}/exports/error_breakdown.csv
GET /runs/{run_id}/exports/utterance_examples.jsonl
GET /experiments/{experiment_id}/exports/ablation_table.csv
```

생성할 파일:

```text
mlops_artifacts/runs/<run_id>/paper_exports/
  main_table.csv
  error_breakdown.csv
  qualitative_examples.jsonl
  README.md
```

성공 기준:

```text
논문 table을 만들 때 수작업 복붙을 줄이고,
run artifact에서 바로 CSV를 생성할 수 있다.
```

### Step 20. SQLite Metadata Cache 추가

지금은 파일 시스템만으로 충분하다. 하지만 run이 많아지면 매번 모든 JSON을 scan하는 것이 느려진다. 이때 SQLite를 metadata cache로 붙인다.

중요한 원칙:

```text
source of truth = mlops_artifacts/runs/<run_id>/
cache/search index = SQLite
```

즉 DB가 깨져도 artifact directory만 있으면 복구 가능해야 한다.

추가할 것:

```text
platform/backend/app/db/
  models.py
  session.py
  migrations/

platform/backend/app/services/indexer.py
```

관리할 table:

```text
runs
jobs
metrics
artifacts
utterances
```

성공 기준:

```text
python platform/backend/app/services/indexer.py
  -> mlops_artifacts를 scan
  -> SQLite index 재생성
```

### Step 21. Redis Queue로 교체

파일 기반 queue는 로컬 MVP에는 충분하다. 하지만 동시에 여러 worker를 돌리거나 retry/cancel을 잘 하려면 Redis queue가 필요하다.

후보:

| queue | 추천도 | 이유 |
| --- | --- | --- |
| RQ | 높음 | 단순하고 Python worker와 잘 맞음 |
| Celery | 중간 | 강력하지만 설정이 무거움 |
| Dramatiq | 중간 | 깔끔하지만 생태계가 작음 |

처음 교체는 RQ를 추천한다.

변경 구조:

```text
POST /runs
  -> job metadata 저장
  -> Redis queue enqueue

worker
  -> Redis queue consume
  -> run_posterior_f5_pipeline.py 실행
  -> artifact 저장
  -> metadata update
```

성공 기준:

```text
docker compose up
  api
  frontend
  redis
  worker
```

### Step 22. Cancel / Retry / Resume

실험 플랫폼에서 반드시 필요한 운영 기능이다.

추가할 API:

```text
POST /runs/{run_id}/cancel
POST /runs/{run_id}/retry
POST /runs/{run_id}/resume
```

처음 구현은 다음 정도면 충분하다.

| 기능 | MVP 동작 |
| --- | --- |
| cancel | queued job은 cancelled로 변경 |
| retry | failed job의 command를 새 job으로 복사 |
| resume | 이미 완료된 stage는 skip하고 다음 stage부터 실행 |

resume을 위해서는 stage output 존재 여부를 확인해야 한다.

```text
posterior stage complete?
  posterior_cache/run.posterior.jsonl exists

inference hard complete?
  generated/hard/*.wav exists

metrics complete?
  metrics/summary.json exists
```

성공 기준:

```text
실패한 run을 처음부터 다시 돌리지 않고,
실패 stage부터 재시작할 수 있다.
```

### Step 23. Dataset / Manifest Registry

manifest path를 매번 입력하는 방식은 오래 가지 못한다. dataset registry가 필요하다.

추가할 파일:

```text
platform/config/datasets.yaml
```

예시:

```yaml
datasets:
  dev_small:
    manifest: data/manifests/dev_small.jsonl
    language: en
    num_utterances: 32
    purpose: smoke test

  eval_clean:
    manifest: data/manifests/eval_clean.jsonl
    language: en
    purpose: paper main table
```

dashboard New Run은 dataset dropdown을 제공한다.

성공 기준:

```text
dataset 선택
  -> manifest 자동 입력
  -> language 자동 입력
  -> subset 정보 표시
```

### Step 24. Audio / Utterance Inspector 고도화

논문 품질 분석에는 utterance별 비교가 중요하다.

추가할 것:

1. reference audio player
2. mode별 generated audio 동시 비교
3. target text, prediction text, reference text 표시
4. WER diff highlight
5. posterior entropy 표시
6. failure type filter

필요한 artifact:

```text
metrics/per_utterance.jsonl
posterior_cache/run.posterior.jsonl
predictions/<mode>.jsonl
generated/<mode>/<utterance_id>.wav
```

성공 기준:

```text
특정 utterance에서 soft_ctc가 hard보다 왜 좋은지/나쁜지
웹에서 바로 듣고 읽으며 판단할 수 있다.
```

### Step 25. Observability와 Structured Logs

지금 로그는 텍스트 파일이다. 다음 단계에서는 사람이 읽기 좋은 로그와 기계가 파싱할 수 있는 로그를 분리한다.

추가할 파일:

```text
logs/job.stdout.log
logs/job.stderr.log
logs/events.jsonl
```

`events.jsonl` 예시:

```json
{"time":"...","level":"info","stage":"inference","mode":"soft_ctc","message":"started"}
{"time":"...","level":"error","stage":"metrics","message":"prediction file missing"}
```

dashboard는 `events.jsonl`을 읽어서 timeline을 그릴 수 있다.

성공 기준:

```text
텍스트 로그를 뒤지지 않아도
어느 stage/mode에서 실패했는지 바로 보인다.
```

### Step 26. S3 또는 MinIO Artifact Store

로컬 파일 시스템은 혼자 쓸 때 좋다. 여러 머신이나 서버로 확장하면 object storage가 필요하다.

확장 순서:

```text
local filesystem
  -> MinIO local
  -> S3-compatible storage
```

중요한 점:

```text
artifact URI는 추상화한다.

local:
  mlops_artifacts/runs/<run_id>/...

s3:
  s3://posterior-f5/runs/<run_id>/...
```

성공 기준:

```text
dashboard는 local path인지 S3 URI인지 몰라도 artifact를 열 수 있다.
```

### Step 27. Multi-Worker / GPU Pool

실제 실험이 커지면 worker를 여러 개 띄워야 한다.

분리 방식:

| worker type | 역할 |
| --- | --- |
| `worker-cpu` | metrics aggregation, file indexing |
| `worker-asr` | posterior extraction, evaluation ASR |
| `worker-tts` | F5-TTS inference |
| `worker-gpu` | GPU가 필요한 통합 작업 |

queue routing:

```text
posterior jobs -> asr queue
inference jobs -> tts queue
metrics jobs -> cpu queue
```

성공 기준:

```text
inference worker 2개를 띄우면
mode별 inference가 병렬로 처리된다.
```

### Step 28. Auth / User / Project 권한

혼자 쓰는 연구 도구라면 auth는 나중 문제다. 하지만 포트폴리오나 제품 방향이면 필요해진다.

추가할 개념:

```text
User
Organization
Project
Role
API Token
```

이 단계에서 NestJS gateway를 고려할 수 있다.

```text
React
  -> NestJS Gateway
  -> FastAPI ML Control Plane
```

단, ML 실행과 artifact contract는 계속 Python/FastAPI 쪽에 두는 것이 좋다.

### Step 29. Deployment

배포는 과하게 시작하지 않는다.

추천 순서:

```text
local docker compose
  -> single VM docker compose
  -> Render/Fly.io style managed services
  -> Kubernetes only if really needed
```

초기 배포 구성:

```text
VM
  docker compose
    api
    frontend
    worker
    redis
    postgres
    minio
```

성공 기준:

```text
다른 컴퓨터에서 git clone 후
docker compose up --build
로 dashboard까지 뜬다.
```

### Step 30. 포트폴리오 정리

MLOps 커리어를 목표로 한다면, 플랫폼 자체를 설명할 수 있어야 한다.

정리할 문서:

```text
docs/md/mlops_platform_fastapi_workers_react.md
docs/md/posterior_f5_paper_experiment_flow.md
platform/docker/README.md
platform/README.md
```

포트폴리오에서 강조할 포인트:

1. artifact-first experiment tracking
2. FastAPI control plane
3. file-backed queue에서 worker 분리
4. React dashboard
5. Docker Compose local platform
6. reproducible paper pipeline
7. future path to Redis/Postgres/S3/GPU workers

성공 기준:

```text
README만 읽어도
왜 이 플랫폼을 만들었고,
어떤 MLOps 문제를 해결하고,
어떻게 확장할지 이해된다.
```

## 20. 추천 진행 순서 요약

앞으로는 아래 순서를 추천한다.

```text
Step 13  Docker Compose end-to-end smoke test
Step 14  sample manifest + demo preset 추가
Step 15  Run Detail UX와 logs/stage view 개선
Step 16  실제 inference 가능한 worker image 분리
Step 17  checkpoint registry
Step 18  experiment comparison view
Step 19  paper export
Step 20  SQLite metadata cache
Step 21  Redis queue
Step 22  cancel/retry/resume
Step 23  dataset/manifest registry
Step 24  utterance inspector 고도화
Step 25  structured logs/events timeline
Step 26  MinIO/S3 artifact store
Step 27  multi-worker/GPU pool
Step 28  auth/user/project 권한
Step 29  deployment
Step 30  portfolio documentation
```

가장 가까운 다음 행동은 Step13이다. 새 기능을 더 붙이기 전에 Docker Compose로 API, frontend, worker가 실제로 함께 움직이는지 확인해야 한다. Step13이 끝나면 Step14와 Step15를 통해 “매번 쓰고 싶은 도구”로 다듬고, 그 다음 Step16부터 실제 무거운 inference worker를 강화하는 것이 좋다.
