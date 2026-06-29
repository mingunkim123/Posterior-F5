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
