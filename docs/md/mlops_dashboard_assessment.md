# Posterior-F5 MLOps / 실험 대시보드 평가

- 작성일: 2026-07-01
- 대상: `platform/` (FastAPI backend + React dashboard + workers), `mlops_artifacts*/`, `src/f5_tts/eval/`
- 관점: 업계 MLOps best practice + 논문급 실험 재현성 기준으로 모델관리 / 실험결과 관리 / 모델평가 / 전과정 편의성 평가

---

## 총평

"노트북 + 스프레드시트" 수준을 한참 넘어선, **1인 연구용 MLOps로서는 상위권**이다. 특히 provenance(재현 근거) 기록, artifact-as-source-of-truth 아키텍처, ASR 분리 원칙, 가이드형 워크플로우는 업계 베스트 프랙티스와 같거나 오히려 앞선다.

다만 **"논문급 + 업계 표준"에 도달하려면 네 곳에 구멍이 있다.**

1. 평가 지표의 폭 (TTS 품질·화자유사도 미연동)
2. 통계의 UI 노출 + 멀티시드
3. 환경 provenance 자동화
4. 배포 스택

핵심은 대부분 새 아키텍처가 아니라 **이미 존재하는 조각(`ecapa_tdnn.py`, `eval_utmos.py`, `bootstrap_significance.py`, `check_ml_runtime.py`)을 파이프라인과 대시보드에 배선하는 작업**이라는 점이다.

---

## 영역별 점수

| 영역 | 등급 | 한줄 평 |
| --- | --- | --- |
| 실험 추적·결과 관리 | **A−** | `run.json` provenance + artifact 불변성 + 자동 export가 매우 견고 |
| 전과정 편의성 (UX) | **A−** | 프리셋 실행·가이드형 next-action·resume/retry·원클릭 export |
| 모델 관리 (registry) | **B** | stage/best-WER 추적은 좋으나 candidate 해시·lineage가 비어있음 |
| 재현성 | **B** | seed·git·manifest 스냅샷은 강함 / 환경·의존성 캡처는 수동 |
| **모델 평가** | **B− (TTS 기준 C+)** | WER/CER는 엄밀하나 화자유사도·UTMOS·RTF·통계가 미연동 |
| 인프라·배포 | **C+** | 관심사 분리는 깔끔 / compose가 스텁, 단일 사용자·로컬 FS |

---

## 잘 되어 있는 것 (베스트 프랙티스 수준)

### 1. Artifact = 진실의 원천, DB는 재생성 캐시
`run.json`에 git commit/branch/dirty, checkpoint hash, seed, 전체 generation/eval 파라미터, **단계별 command·inputs·outputs·exit_code·log 경로**까지 기록된다.
- 근거: `mlops_artifacts_user/runs/*/run.json`, `platform/workers/run_posterior_f5_pipeline.py`
- SQLite는 `platform/backend/app/db/models.py` 주석대로 "artifact에서 재빌드하는 검색용 캐시"로 명확히 위치. 인덱서는 `platform/backend/app/services/indexer.py`.
- 많은 팀이 반대로(DB를 진실로) 해서 망치는 부분인데, 방향이 정확하다.

### 2. 평가의 과학적 엄밀성
- `docs/md/experiment_protocol.md`의 **"posterior source ASR ≠ evaluation ASR" 분리 원칙**은 평가 누수(leakage)를 막는 고급 설계.
- normalizer 프로파일(paper/lowercase/none), `failure_type` 자동 분류(`deletion_heavy` 등), subset별 집계까지 논문 수준.
- 근거: `src/f5_tts/eval/eval_posterior_f5.py`, `src/f5_tts/eval/error_breakdown.py`

### 3. 완결된 라이프사이클 오케스트레이션
- queue → run → cancel → retry → **resume** 완비.
- `resume_command()`는 이미 끝난 stage(posterior/inference/prediction/metrics)를 감지해 플래그를 제거하는 **멱등 재개**. SIGTERM 취소, `attempts` 이력 누적까지.
- 근거: `platform/backend/app/run_store.py:845` (`resume_command`), `:887` (`cancel_run`), `:914` (`requeue_existing_job`)

### 4. 결과 관리·논문 export 자동화
- `main_table.csv`, `error_breakdown.csv`, `qualitative_examples.jsonl`, 실험단위 `ablation_table.csv`를 API 한 방으로 산출.
- 근거: `platform/backend/app/main.py:260-285`, `run_store.py:619` (`ensure_run_paper_exports`), `:679` (`ensure_experiment_ablation_export`)

### 5. 가이드형 UX
- 대시보드 8개 뷰: overview / launch / runs / models / evaluation / results / compare / logs.
- overview의 `NextActionCard` + `LifecycleTracker`(실험 진행 흐름)가 다음 할 일을 짚어줌. word-diff 단위 utterance 검토, 품질 Gate, 실패분포, Leaderboard 제공.
- 근거: `platform/frontend/src/App.tsx`

### 6. 테스트 21종
- bootstrap, error_breakdown, eval, API store, run scaffold, posterior schema/io/length/normalize, encoder shapes, training step 등.
- 근거: `tests/test_*.py` (21개). 연구 리포로선 이례적으로 탄탄.

---

## 채워야 할 구멍 (우선순위순)

### 🔴 1. 평가 지표가 intelligibility(WER/CER)에 편중 — TTS 논문의 절반이 빠짐
파이프라인이 실제 산출하는 지표는 WER/CER + S/D/I + coverage + 평균 생성시간뿐이다.
- 근거: `tmp/smoke/metrics.json`, `src/f5_tts/eval/eval_posterior_f5.py`
- 그런데 프로토콜 자신이 요구하는 **화자 유사도(SIM/ECAPA)·UTMOS·RTF·MOS/SMOS**는 `src/f5_tts/eval/ecapa_tdnn.py`, `src/f5_tts/eval/eval_utmos.py`로 존재만 하고 **pipeline `summary.json`과 대시보드에 연결되어 있지 않다.**
- TTS 평가는 "얼마나 알아듣나(WER)"만큼 "얼마나 자연스럽고 화자를 닮았나"가 중요한데, 지금 대시보드로는 후자를 볼 수 없다.

**Fix:** `run_posterior_f5_pipeline.py`에 spk-sim / UTMOS / RTF stage를 추가하고 `summary.json.modes[]`에 필드를 넣어 `MetricBars` / compare에 노출.

### 🔴 2. 통계적 유의성이 UI에 없음 + 멀티시드 부재
- `src/f5_tts/eval/bootstrap_significance.py`(paired bootstrap CI)는 있고 테스트도 되지만 **수동 CLI**다.
- compare / evaluation 뷰는 점추정값만 보여주고 **CI·에러바·p-value·Holm/FDR 보정이 없다.**
- run 하나 = seed 하나(1234) 구조라 **seed 3개 이상 mean±std 집계**가 어렵고 Leaderboard가 단일 시드 점수로 순위를 매긴다. 논문 주장에는 치명적.

**Fix:** 실험 단위 리포트에 bootstrap CI를 계산·저장하고 compare 테이블에 신뢰구간/유의성 뱃지 렌더링. run 스키마에 seed 그룹 집계 추가.

### 🟠 3. 환경 provenance가 수동
- `run.json`은 git·checkpoint hash·seed는 잡지만 **python / torch / CUDA / GPU / driver 버전과 의존성 lock을 자동 저장하지 않는다.**
- `platform/workers/check_ml_runtime.py`가 torch/cuda 버전을 찍을 수 있는데 run에 안 붙는다. `docs/md/reproducibility_checklist.md`가 이 칸들을 수동 기입으로 남겨둔 게 그 증거.
- manifest는 run 폴더로 **복사(freeze)**되지만(`snapshot_manifest`) 내용 해시는 안 남는다.

**Fix:** `scaffold_run()`에서 `check_ml_runtime`류 스냅샷 + `uv.lock`/`pip freeze` 해시를 `run.json["environment"]`에 자동 기록. manifest sha256도 함께.

### 🟠 4. 모델 레지스트리 lineage 미완성
- `platform/config/checkpoints.yaml`의 로컬 candidate 슬롯(`posterior_f5_local_latest`, `posterior_encoder_dev_latest`)은 `checkpoint_hash: ""`, `git_commit: ""`로 비어 있다.
- HF 체크포인트는 실제 콘텐츠 해시가 아니라 경로 문자열(`hf:...`)을 해시 자리에 쓴다. 즉 후보 모델이 아직 content-addressed가 아니고 "어떤 데이터/커밋이 이 ckpt를 만들었나"가 placeholder.

**Fix:** 학습 산출 시 실제 sha256 · 학습 run_id · 데이터 버전을 레지스트리에 write-back.

### 🟡 5. 배포 스택이 스텁
- `platform/docker/`에 `Dockerfile.api/frontend/worker-gpu`는 있는데 `docker-compose.gpu.yml`은 **GPU 예약 fragment 한 조각**뿐(redis/api/frontend/worker 서비스·volume·env 없음).
- 큐도 기본이 `file` 백엔드고 Redis/RQ는 opt-in인데 compose로 묶여있지 않다.
- 근거: `platform/backend/app/services/queue.py`, `platform/docker/docker-compose.gpu.yml`

**Fix:** compose에 redis + api + worker + frontend 풀 서비스 정의.

### 🟡 6. 스코프: 단일 사용자·로컬 전용
- 인증·run 소유권 없음, CORS는 localhost 고정(`main.py:64-75`), 아티팩트는 로컬 FS(S3/GCS 없음), `mlops_artifacts` vs `mlops_artifacts_user` 이원화.
- **1인 연구용으론 완전히 적절**하지만 "팀 MLOps"는 아니라는 점만 명확히 인지하면 된다.
- 품질 Gate 임계값(WER 0.05 등)도 코드 하드코딩(`run_store.py:527`)이라 실험별 설정으로 빼면 좋다.

---

## 추천 액션 (효과/노력 순)

1. **평가 stage 확장** — spk-sim(ECAPA)·UTMOS·RTF를 pipeline → `summary.json` → 대시보드까지 연결. *(가장 큰 신뢰도 상승)*
2. **통계 UI 노출** — 실험 리포트에 bootstrap CI 저장 + compare에 신뢰구간/유의성 표시, 멀티시드 mean±std.
3. **환경 자동 캡처** — `run.json`에 torch/cuda/gpu + lock 해시 + manifest sha256.
4. **레지스트리 write-back** — candidate ckpt 실제 해시·학습 lineage 기록.
5. **compose 완성** — redis + api + worker + frontend 풀 스택.

---

## 한 줄 결론

뼈대와 원칙(provenance · artifact 불변성 · ASR 분리 · 가이드 UX · export 자동화)은 **이미 베스트 프랙티스급**이다. 남은 건 "TTS 평가로서의 지표 완성도"와 "통계·환경·배포의 자동화"이며, 대부분 새 설계가 아니라 **이미 있는 조각을 배선하는 작업**이다.

---

## 부록: 조사한 파일

| 분류 | 경로 |
| --- | --- |
| Backend API | `platform/backend/app/main.py`, `run_store.py`, `db/models.py`, `services/indexer.py`, `services/queue.py`, `services/rq_tasks.py` |
| Workers | `platform/workers/run_posterior_f5_pipeline.py`, `run_job_worker.py`, `check_ml_runtime.py` |
| Frontend | `platform/frontend/src/App.tsx`, `api.ts` |
| Config | `platform/config/checkpoints.yaml`, `datasets.yaml` |
| Eval | `src/f5_tts/eval/eval_posterior_f5.py`, `bootstrap_significance.py`, `error_breakdown.py`, `ecapa_tdnn.py`, `eval_utmos.py`, `make_result_tables.py` |
| Docs | `docs/md/experiment_protocol.md`, `reproducibility_checklist.md` |
| Artifacts | `mlops_artifacts_user/runs/*/{run.json,job.json,metrics/}`, `tmp/smoke/metrics.json` |
| Tests | `tests/test_*.py` (21) |
| Deploy | `platform/docker/docker-compose.gpu.yml`, `Dockerfile.*` |
