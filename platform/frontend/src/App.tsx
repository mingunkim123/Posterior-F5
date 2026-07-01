import { Fragment, type FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  AudioLines,
  BadgeCheck,
  BarChart3,
  CheckCircle2,
  Circle,
  CircleDashed,
  CircleDot,
  Clipboard,
  Clock,
  Compass,
  Database,
  Download,
  FileAudio,
  FlaskConical,
  FolderInput,
  Gauge,
  GitBranch,
  ListFilter,
  Loader2,
  PackageCheck,
  Play,
  RefreshCcw,
  Rocket,
  Server,
  Sparkles,
  Target,
  Terminal,
  TrendingDown,
  TriangleAlert,
} from "lucide-react";
import {
  Checkpoint,
  CompareRow,
  Dataset,
  EvaluationReport,
  MetricRow,
  ModelCard,
  Run,
  RunCreatePayload,
  RunJob,
  RunEvents,
  RunLogs,
  Stage,
  Utterance,
  artifactUrl,
  createRun,
  fetchEvaluationReport,
  experimentExportUrl,
  fetchCheckpoints,
  fetchCompare,
  fetchDatasets,
  fetchMetrics,
  fetchModels,
  fetchRun,
  fetchRunEvents,
  fetchRunJob,
  fetchRunLogs,
  fetchRuns,
  fetchUtterances,
  postRunAction,
  runExportUrl,
} from "./api";
import {
  demoCompareRows,
  demoMetrics,
  demoRuns,
  demoUtterances,
  fallbackCheckpoints,
  fallbackDatasets,
  fallbackEvaluationReport,
  fallbackModelRegistry,
  fallbackSpeakerCheckpoints,
} from "./demo";

const modeColors: Record<string, string> = {
  hard: "#3d6b99",
  oracle: "#2f8f70",
  length_only: "#b07d2f",
  soft_ctc: "#8d5fbf",
  hybrid: "#b8526b",
  posterior_encoder: "#587447",
};

const availableModes = ["hard", "oracle", "length_only", "soft_ctc", "posterior_encoder", "hybrid"];
const sampleManifest = "platform/samples/manifests/dev_smoke.jsonl";
const posteriorRequiredModes = new Set(["length_only", "soft_ctc", "posterior_encoder", "hybrid"]);

const defaultRunPayload: RunCreatePayload = {
  project: "Posterior-F5",
  experiment: "baseline_dev_small",
  dataset_id: "dev_smoke",
  manifest: sampleManifest,
  modes: ["hard", "oracle", "length_only", "soft_ctc"],
  model: "F5TTS_v1_Base",
  checkpoint_id: "f5tts_v1_base_hf",
  checkpoint: "",
  checkpoint_hash: "",
  vocoder: "vocos",
  seed: 1234,
  language: "en",
  hard_ref_text_source: "manifest",
  run_posterior_extraction: false,
  skip_whisper: true,
  run_inference: false,
  inference_dry_run: true,
  run_prediction: false,
  prediction_dry_run: true,
  run_audio_metrics: false,
  audio_metrics_dry_run: true,
  run_speaker_similarity: false,
  speaker_checkpoint: "",
  speaker_device: null,
  speaker_feat_type: "wavlm_large",
  run_utmos: false,
  utmos_device: null,
  run_metrics: false,
  metrics_dry_run: false,
  metrics_normalizer: "paper",
  bootstrap_samples: 1000,
  bootstrap_seed: 1234,
  significance_baseline: "hard",
  fail_if_exists: false,
};

function mergeDatasetsWithFallback(loaded: Dataset[]): Dataset[] {
  const byId = new Map<string, Dataset>();
  for (const dataset of fallbackDatasets) {
    byId.set(dataset.id, dataset);
  }
  for (const dataset of loaded) {
    byId.set(dataset.id, dataset);
  }
  return Array.from(byId.values());
}

type RunPreset = {
  key: string;
  label: string;
  summary: string;
  payload: Partial<RunCreatePayload>;
};

function needsPosterior(modes: string[]): boolean {
  return modes.some((mode) => posteriorRequiredModes.has(mode));
}

function normalizeRunPayload(payload: RunCreatePayload): RunCreatePayload {
  const next = { ...payload };
  const speakerCheckpoint = next.speaker_checkpoint?.trim() ?? "";

  if (needsPosterior(next.modes) && (next.run_inference || next.run_prediction || next.run_metrics)) {
    next.run_posterior_extraction = true;
  }

  if (next.run_metrics) {
    next.run_prediction = true;
  }
  if (next.run_prediction || next.run_audio_metrics) {
    next.run_inference = true;
  }

  if (!next.run_inference) {
    next.inference_dry_run = true;
    next.run_prediction = false;
    next.prediction_dry_run = true;
    next.run_audio_metrics = false;
    next.audio_metrics_dry_run = true;
    next.run_metrics = false;
    next.metrics_dry_run = false;
  }

  if (next.inference_dry_run) {
    next.prediction_dry_run = true;
    next.audio_metrics_dry_run = true;
    next.metrics_dry_run = next.run_metrics;
  }

  if (!next.run_prediction) {
    next.prediction_dry_run = true;
    next.run_metrics = false;
    next.metrics_dry_run = false;
  }

  if (next.run_metrics && next.prediction_dry_run) {
    next.metrics_dry_run = true;
  }

  if (!next.run_audio_metrics) {
    next.audio_metrics_dry_run = true;
    next.run_speaker_similarity = false;
    next.run_utmos = false;
  }

  if (next.audio_metrics_dry_run) {
    next.run_speaker_similarity = false;
    next.run_utmos = false;
  }

  if (next.run_speaker_similarity && !speakerCheckpoint) {
    next.run_speaker_similarity = false;
  }

  return next;
}

function presetPayload(current: RunCreatePayload, preset: RunPreset): RunCreatePayload {
  return normalizeRunPayload({
    ...defaultRunPayload,
    dataset_id: current.dataset_id,
    manifest: current.manifest,
    language: current.language,
    model: current.model,
    checkpoint_id: current.checkpoint_id,
    checkpoint: current.checkpoint,
    checkpoint_hash: current.checkpoint_hash,
    vocoder: current.vocoder,
    seed: current.seed,
    bootstrap_samples: current.bootstrap_samples,
    bootstrap_seed: current.bootstrap_seed,
    significance_baseline: current.significance_baseline,
    ...preset.payload,
    run_id: undefined,
    fail_if_exists: false,
  });
}

const runPresets: RunPreset[] = [
  {
    key: "scaffold_only",
    label: "골격만 생성",
    summary: "데이터 로드만 — 실행 디렉터리와 설정을 만듭니다.",
    payload: {
      experiment: "scaffold_only_smoke",
      dataset_id: "dev_smoke",
      manifest: sampleManifest,
      modes: ["hard"],
      run_posterior_extraction: false,
      skip_whisper: true,
      run_inference: false,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_audio_metrics: false,
      audio_metrics_dry_run: true,
      run_speaker_similarity: false,
      speaker_checkpoint: "",
      run_utmos: false,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "posterior_text_only",
    label: "Posterior 준비",
    summary: "데이터 로드 + posterior 신호 추출까지 진행합니다.",
    payload: {
      experiment: "posterior_text_smoke",
      dataset_id: "dev_smoke",
      manifest: sampleManifest,
      modes: ["hard"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: false,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_audio_metrics: false,
      audio_metrics_dry_run: true,
      run_speaker_similarity: false,
      speaker_checkpoint: "",
      run_utmos: false,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "dry_inference_plan",
    label: "음성 생성 계획",
    summary: "모드별 음성 생성 명령을 계획(dry-run)으로 준비합니다.",
    payload: {
      experiment: "dry_inference_plan",
      dataset_id: "dev_smoke",
      manifest: sampleManifest,
      modes: ["hard", "oracle", "length_only", "soft_ctc"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: true,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_audio_metrics: false,
      audio_metrics_dry_run: true,
      run_speaker_similarity: false,
      speaker_checkpoint: "",
      run_utmos: false,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "metrics_dry_run",
    label: "평가까지 한 번에",
    summary: "실제 음성 생성 후 RTF, ASR, WER/CER까지 계산합니다.",
    payload: {
      experiment: "rtf_asr_eval_smoke",
      dataset_id: "dev_smoke",
      manifest: sampleManifest,
      modes: ["hard", "oracle", "length_only", "soft_ctc"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: true,
      inference_dry_run: false,
      run_audio_metrics: true,
      audio_metrics_dry_run: false,
      run_speaker_similarity: false,
      speaker_checkpoint: "",
      run_utmos: false,
      run_prediction: true,
      prediction_dry_run: false,
      run_metrics: true,
      metrics_dry_run: false,
    },
  },
];

function pct(value?: number | null): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function compactNumber(value?: number | null, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function pValue(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 0.001) return "<0.001";
  return value.toFixed(3);
}

function metricWithStd(value?: number | null, std?: number | null, formatter: (value?: number | null) => string = compactNumber): string {
  const base = formatter(value);
  if (std === null || std === undefined || Number.isNaN(std)) return base;
  return `${base} ± ${formatter(std)}`;
}

function gateMetricValue(value: number | null | undefined, unit?: string): string {
  if (unit === "score") return compactNumber(value);
  if (unit === "ratio") return compactNumber(value, 3);
  return pct(value);
}

function metricCount(value?: number | null): string {
  if (value === null || value === undefined) return "0";
  return Intl.NumberFormat().format(value);
}

function shortTime(value?: string): string {
  if (!value) return "";
  return value.replace("T", " ").slice(0, 16);
}

function statusGlyph(status: string | undefined, size = 16): ReactNode {
  if (status === "succeeded" || status === "completed") return <CheckCircle2 size={size} />;
  if (status === "failed") return <TriangleAlert size={size} />;
  if (status === "running" || status === "submitted") return <Loader2 className="spin" size={size} />;
  if (status === "queued") return <Clock size={size} />;
  if (status === "planned") return <CircleDashed size={size} />;
  return <CircleDot size={size} />;
}

function statusLabel(status?: string, dryRun?: boolean): string {
  if (dryRun && status === "planned") return "계획만";
  if (status === "succeeded" || status === "completed") return "완료";
  if (status === "failed") return "실패";
  if (status === "running") return "실행 중";
  if (status === "cancelling") return "취소 중";
  if (status === "cancelled") return "취소됨";
  if (status === "queued") return "대기 중";
  if (status === "submitted") return "제출됨";
  if (status === "planned") return "예정";
  return status ?? "대기";
}

function stageLabel(name: string): string {
  const labels: Record<string, string> = {
    scaffold: "실험 준비",
    posterior_extraction: "Posterior 생성",
    inference: "음성 생성",
    audio_metrics: "음향 평가",
    prediction: "ASR 변환",
    metrics: "점수 계산",
    job: "Worker 작업",
    queued: "대기열",
  };
  return labels[name] ?? name.replace(/_/g, " ");
}

function stageMeta(name: string): string {
  const labels: Record<string, string> = {
    scaffold: "파일과 설정",
    posterior_extraction: "참조 신호",
    inference: "mode별 wav",
    audio_metrics: "SIM · UTMOS · RTF",
    prediction: "ASR 결과",
    metrics: "WER · CER · CI",
    job: "worker 상태",
    queued: "실행 대기",
  };
  return labels[name] ?? "stage";
}

// ---------------------------------------------------------------------------
// Lifecycle phase model: group the technical stages into the 3 macro phases
// the operator actually thinks in — 데이터 로드 → 모델 음성 생성 → 모델 평가.
// ---------------------------------------------------------------------------

type PhaseKey = "data" | "model" | "eval";
type PhaseStatus = "succeeded" | "partial" | "running" | "failed" | "planned" | "pending";

type PhaseDef = {
  key: PhaseKey;
  label: string;
  sub: string;
  hint: string;
  stageNames: string[];
  icon: (size: number) => ReactNode;
};

const PHASES: PhaseDef[] = [
  {
    key: "data",
    label: "데이터 로드",
    sub: "manifest · posterior",
    hint: "참조 음성·텍스트를 불러오고 posterior 신호를 준비합니다.",
    stageNames: ["queued", "job", "scaffold", "posterior_extraction"],
    icon: (size) => <FolderInput size={size} />,
  },
  {
    key: "model",
    label: "모델 음성 생성",
    sub: "F5-TTS inference",
    hint: "선택한 mode별로 F5-TTS 모델이 음성을 생성합니다.",
    stageNames: ["inference"],
    icon: (size) => <AudioLines size={size} />,
  },
  {
    key: "eval",
    label: "모델 평가",
    sub: "ASR · WER / CER",
    hint: "생성 음성을 ASR로 전사하고 정확도를 점수화합니다.",
    stageNames: ["audio_metrics", "prediction", "metrics"],
    icon: (size) => <Gauge size={size} />,
  },
];

type PhaseState = {
  def: PhaseDef;
  status: PhaseStatus;
  stages: Array<{ stage: Stage; key: string }>;
  doneCount: number;
  totalCount: number;
};

function isDoneStatus(status: string): boolean {
  return status === "succeeded" || status === "completed";
}

function derivePhases(run?: Run): PhaseState[] {
  const stages = run?.stages ?? [];
  const indexed = stages.map((stage, index) => ({ stage, key: stageKey(stage, index) }));
  return PHASES.map((def) => {
    const present = indexed.filter((item) => def.stageNames.includes(item.stage.name));
    let status: PhaseStatus;
    if (present.length === 0) {
      status = "pending";
    } else if (present.some((item) => item.stage.status === "failed")) {
      status = "failed";
    } else if (present.some((item) => ["running", "submitted", "queued", "cancelling"].includes(item.stage.status))) {
      status = "running";
    } else if (present.every((item) => isDoneStatus(item.stage.status))) {
      status = "succeeded";
    } else if (present.every((item) => item.stage.status === "planned")) {
      status = "planned";
    } else {
      status = "partial";
    }
    const doneCount = present.filter((item) => isDoneStatus(item.stage.status) || item.stage.status === "planned").length;
    return { def, status, stages: present, doneCount, totalCount: present.length };
  });
}

function phaseStatusLabel(status: PhaseStatus): string {
  const labels: Record<PhaseStatus, string> = {
    succeeded: "완료",
    partial: "부분 완료",
    running: "진행 중",
    failed: "실패",
    planned: "계획만",
    pending: "대기",
  };
  return labels[status];
}

function phaseGlyph(status: PhaseStatus, size = 16): ReactNode {
  if (status === "succeeded") return <CheckCircle2 size={size} />;
  if (status === "partial") return <Activity size={size} />;
  if (status === "running") return <Loader2 className="spin" size={size} />;
  if (status === "failed") return <TriangleAlert size={size} />;
  if (status === "planned") return <CircleDashed size={size} />;
  return <Circle size={size} />;
}

type RunMetricSummary = { bestWer: number | null; bestMode?: string };

function buildRunMetrics(rows: CompareRow[]): Record<string, RunMetricSummary> {
  const map: Record<string, RunMetricSummary> = {};
  for (const row of rows) {
    const current = map[row.run_id] ?? { bestWer: null };
    if (typeof row.wer === "number" && (current.bestWer === null || row.wer < current.bestWer)) {
      current.bestWer = row.wer;
      current.bestMode = row.mode ?? undefined;
    }
    map[row.run_id] = current;
  }
  return map;
}

function bestMetric(metrics: MetricRow[]): RunMetricSummary {
  let bestWer: number | null = null;
  let bestMode: string | undefined;
  for (const metric of metrics) {
    if (typeof metric.wer === "number" && (bestWer === null || metric.wer < bestWer)) {
      bestWer = metric.wer;
      bestMode = metric.mode;
    }
  }
  return { bestWer, bestMode };
}

type ViewKey = "overview" | "launch" | "runs" | "models" | "evaluation" | "results" | "compare" | "logs" | "utterances";

const viewLabels: Record<ViewKey, string> = {
  overview: "개요",
  launch: "실험 시작",
  runs: "실험 기록",
  models: "모델 관리",
  evaluation: "평가 센터",
  results: "결과 분석",
  compare: "비교 분석",
  logs: "실행 로그",
  utterances: "샘플 검토",
};

const viewDescriptions: Record<ViewKey, string> = {
  overview: "지금 무엇을 해야 하는지와 실험 진행 흐름을 한눈에",
  launch: "데이터셋·mode를 골라 새 실험을 실행",
  runs: "실험별 진행 상태와 남은 결과",
  models: "checkpoint registry와 검증 결과를 함께 관리",
  evaluation: "품질 gate, leaderboard, 실패 분해",
  results: "선택한 실험의 WER / CER 분석",
  compare: "실험·mode 간 정량 비교",
  logs: "단계별 명령과 실시간 로그",
  utterances: "생성 음성과 전사 결과 청취",
};

function viewIcon(view: ViewKey, size = 17): ReactNode {
  if (view === "overview") return <Compass size={size} />;
  if (view === "launch") return <Rocket size={size} />;
  if (view === "runs") return <ListFilter size={size} />;
  if (view === "models") return <PackageCheck size={size} />;
  if (view === "evaluation") return <Target size={size} />;
  if (view === "results") return <BarChart3 size={size} />;
  if (view === "compare") return <Database size={size} />;
  if (view === "logs") return <Terminal size={size} />;
  return <FileAudio size={size} />;
}

const navGroups: Array<{ title: string; views: ViewKey[] }> = [
  { title: "워크플로", views: ["overview", "launch"] },
  { title: "실험", views: ["runs", "models", "evaluation", "results", "compare"] },
  { title: "진단", views: ["logs", "utterances"] },
];

const viewKeys: ViewKey[] = ["overview", "launch", "runs", "models", "evaluation", "results", "compare", "logs", "utterances"];

function readViewFromHash(): ViewKey | undefined {
  const raw = window.location.hash.replace(/^#\/?/, "");
  return (viewKeys as string[]).includes(raw) ? (raw as ViewKey) : undefined;
}

function stageKey(stage: Stage, index: number): string {
  return `${stage.name}-${index}`;
}

function checkpointLabel(run?: Run): string {
  const checkpoint = run?.model?.checkpoint_id || run?.model?.checkpoint_path || run?.model?.checkpoint;
  if (!checkpoint) return "checkpoint n/a";
  return checkpoint.split("/").pop() || checkpoint;
}

function runError(run?: Run, job?: RunJob, selectedStage?: Stage): string {
  return selectedStage?.error_message || job?.error_message || run?.error_message || "";
}

function stageLogPaths(stage?: Stage): string[] {
  if (!stage) return [];
  const paths = new Set<string>();
  if (stage.log) paths.add(stage.log);
  for (const mode of stage.modes ?? []) {
    if (mode.log) paths.add(mode.log);
  }
  if (stage.name === "scaffold" || stage.name === "job" || stage.name === "queued") {
    paths.add("logs/job.stdout.log");
    paths.add("logs/job.stderr.log");
  }
  return [...paths];
}

function filteredLogFiles(logs?: RunLogs, selectedStage?: Stage) {
  const files = logs?.files ?? [];
  const paths = stageLogPaths(selectedStage);
  if (!selectedStage || paths.length === 0) return files;
  return files.filter((file) => paths.includes(file.path));
}

function defaultStageSelection(run?: Run): string {
  const stages = run?.stages ?? [];
  const targetIndex = stages.findIndex((stage) => ["failed", "running"].includes(stage.status));
  if (targetIndex >= 0) return stageKey(stages[targetIndex], targetIndex);
  if (stages.length === 0) return "";
  return stageKey(stages[stages.length - 1], stages.length - 1);
}

function SideNav({
  activeView,
  apiState,
  onViewChange,
  run,
  runsCount,
}: {
  activeView: ViewKey;
  apiState: "api" | "demo" | "loading";
  onViewChange: (view: ViewKey) => void;
  run?: Run;
  runsCount: number;
}) {
  return (
    <aside className="sideNav">
      <div className="sideBrand">
        <div className="brandMark"><GitBranch size={20} /></div>
        <div>
          <h1>Posterior-F5</h1>
          <span>Ops Console</span>
        </div>
      </div>
      <nav className="navMenu">
        {navGroups.map((group) => (
          <div className="navGroup" key={group.title}>
            <small className="navGroupTitle">{group.title}</small>
            {group.views.map((view) => (
              <button
                className={activeView === view ? "navItem active" : "navItem"}
                key={view}
                onClick={() => onViewChange(view)}
              >
                {viewIcon(view)}
                <span>{viewLabels[view]}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>
      <div className="sideMeta">
        <small>선택된 실험</small>
        <strong>{run?.run_id ?? "none"}</strong>
        <em className={`pill status-${run?.status ?? "planned"}`}>{statusGlyph(run?.status, 12)}{statusLabel(run?.status)}</em>
      </div>
      <div className="sideFooter">
        <span className={apiState === "api" ? "apiBadge live" : "apiBadge"}>
          <Server size={15} />
          {apiState === "loading" ? "연결 중" : apiState === "api" ? "연결됨" : "데모"}
        </span>
        <span className="runCount">{runsCount} runs</span>
      </div>
    </aside>
  );
}

type NextActionModel = {
  tone: "info" | "success" | "warning" | "danger" | "neutral";
  eyebrow: string;
  title: string;
  body: string;
  icon: ReactNode;
  cta?: { label: string; icon: ReactNode; onClick: () => void };
  secondary?: { label: string; onClick: () => void };
};

function computeNextAction(params: {
  run?: Run;
  job?: RunJob;
  metrics: MetricRow[];
  runsCount: number;
  apiState: "api" | "demo" | "loading";
  best: RunMetricSummary;
  goTo: (view: ViewKey) => void;
  onRetry: () => void;
}): NextActionModel {
  const { run, job, metrics, runsCount, apiState, best, goTo, onRetry } = params;

  if (apiState === "demo") {
    return {
      tone: "warning",
      eyebrow: "데모 모드",
      title: "API 서버에 연결되어 있지 않습니다",
      body: "지금 보이는 값은 예시 데이터입니다. 백엔드(uvicorn)를 실행하고 새로고침하면 실제 실험을 제어할 수 있습니다.",
      icon: <Server size={26} />,
      cta: { label: "실험 시작 화면", icon: <Play size={16} />, onClick: () => goTo("launch") },
    };
  }

  if (!run || runsCount === 0) {
    return {
      tone: "info",
      eyebrow: "시작하기",
      title: "첫 실험을 시작하세요",
      body: "데이터셋과 mode를 고르고 ‘실험 시작’을 누르면 데이터 로드 → 음성 생성 → 평가 파이프라인이 순서대로 실행됩니다.",
      icon: <Rocket size={26} />,
      cta: { label: "실험 시작", icon: <Play size={16} />, onClick: () => goTo("launch") },
    };
  }

  const status = job?.status ?? run.status;

  if (["failed", "cancelled", "cancelling"].includes(status)) {
    return {
      tone: "danger",
      eyebrow: "조치 필요",
      title: "실험이 중단되었습니다",
      body: run.error_message || job?.error_message || "실패한 단계의 로그를 확인한 뒤 재시도하세요.",
      icon: <TriangleAlert size={26} />,
      cta: { label: "재시도", icon: <RefreshCcw size={16} />, onClick: onRetry },
      secondary: { label: "로그 보기", onClick: () => goTo("logs") },
    };
  }

  if (["queued", "submitted", "running"].includes(status)) {
    return {
      tone: "info",
      eyebrow: "진행 중",
      title: "실험이 실행되고 있습니다",
      body: "단계가 끝나면 결과가 자동으로 갱신됩니다. 실시간 로그로 진행 상황을 확인하세요.",
      icon: <Loader2 className="spin" size={26} />,
      cta: { label: "실시간 로그", icon: <Terminal size={16} />, onClick: () => goTo("logs") },
    };
  }

  if (metrics.length > 0) {
    const werText =
      best.bestWer != null
        ? `최고 성능은 ${best.bestMode} mode (WER ${pct(best.bestWer)})입니다.`
        : "평가 점수가 준비되었습니다.";
    return {
      tone: "success",
      eyebrow: "결과 준비 완료",
      title: "실험이 완료되어 결과가 있습니다",
      body: `${werText} 결과 분석과 비교 화면에서 mode별 품질을 확인하세요.`,
      icon: <Sparkles size={26} />,
      cta: { label: "결과 분석", icon: <BarChart3 size={16} />, onClick: () => goTo("results") },
      secondary: { label: "실험 비교", onClick: () => goTo("compare") },
    };
  }

  return {
    tone: "neutral",
    eyebrow: "다음 단계",
    title: "파이프라인 골격이 준비되었습니다",
    body: "아직 평가 점수가 없습니다. 음성 생성과 평가 단계를 켜고 새 실험을 실행하면 WER / CER 결과를 얻을 수 있습니다.",
    icon: <Compass size={26} />,
    cta: { label: "실험 구성", icon: <Play size={16} />, onClick: () => goTo("launch") },
    secondary: { label: "진행 흐름 보기", onClick: () => goTo("runs") },
  };
}

function NextActionCard({ action }: { action: NextActionModel }) {
  return (
    <section className={`panel nextAction tone-${action.tone}`}>
      <div className="nextActionMain">
        <div className="nextActionGlyph">{action.icon}</div>
        <div className="nextActionText">
          <small>{action.eyebrow}</small>
          <h2>{action.title}</h2>
          <p>{action.body}</p>
        </div>
      </div>
      {action.cta || action.secondary ? (
        <div className="nextActionButtons">
          {action.cta ? (
            <button className="nextActionPrimary" onClick={action.cta.onClick} type="button">
              {action.cta.icon}
              <span>{action.cta.label}</span>
              <ArrowRight size={16} />
            </button>
          ) : null}
          {action.secondary ? (
            <button className="nextActionSecondary" onClick={action.secondary.onClick} type="button">
              {action.secondary.label}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function LifecycleTracker({
  run,
  onSelectStage,
  selectedStageKey,
}: {
  run?: Run;
  onSelectStage?: (stage: Stage, key: string) => void;
  selectedStageKey?: string;
}) {
  const phases = useMemo(() => derivePhases(run), [run]);
  return (
    <section className="panel lifecyclePanel">
      <div className="panelHeader">
        <div>
          <h2>실험 진행 흐름</h2>
          <p>데이터 로드 → 모델 음성 생성 → 모델 평가</p>
        </div>
        <Activity size={20} />
      </div>
      <div className="lifecycleRail">
        {phases.map((phase, index) => (
          <Fragment key={phase.def.key}>
            <div className={`phaseCard phase-${phase.status}`}>
              <div className="phaseTop">
                <span className="phaseIcon">{phase.def.icon(20)}</span>
                <div className="phaseTitle">
                  <strong>{phase.def.label}</strong>
                  <span>{phase.def.sub}</span>
                </div>
                <em className={`phasePill phase-${phase.status}`}>
                  {phaseGlyph(phase.status, 13)}
                  {phaseStatusLabel(phase.status)}
                </em>
              </div>
              <p className="phaseHint">{phase.def.hint}</p>
              <div className="phaseStages">
                {phase.stages.length === 0 ? (
                  <span className="phaseEmpty">아직 실행되지 않음</span>
                ) : (
                  phase.stages.map(({ stage, key }) => (
                    <button
                      className={selectedStageKey === key ? `stageChip status-${stage.status} active` : `stageChip status-${stage.status}`}
                      key={key}
                      onClick={() => onSelectStage?.(stage, key)}
                      type="button"
                    >
                      {statusGlyph(stage.status, 13)}
                      <span>{stageLabel(stage.name)}</span>
                      <em>{statusLabel(stage.status, stage.dry_run)}</em>
                    </button>
                  ))
                )}
              </div>
            </div>
            {index < phases.length - 1 ? (
              <div className="phaseConnector">
                <ArrowRight size={18} />
              </div>
            ) : null}
          </Fragment>
        ))}
      </div>
    </section>
  );
}

function StatStrip({
  run,
  job,
  metrics,
  utterances,
  phases,
  best,
}: {
  run?: Run;
  job?: RunJob;
  metrics: MetricRow[];
  utterances: Utterance[];
  phases: PhaseState[];
  best: RunMetricSummary;
}) {
  const phasesDone = phases.filter((phase) => ["succeeded", "planned", "partial"].includes(phase.status)).length;
  const effectiveStatus = job?.status ?? run?.status;
  const stats: Array<{ icon: ReactNode; label: string; value: string }> = [
    { icon: <FlaskConical size={16} />, label: "실험", value: run?.experiment ?? "—" },
    { icon: statusGlyph(effectiveStatus, 16), label: "상태", value: statusLabel(effectiveStatus) },
    { icon: <Compass size={16} />, label: "진행", value: `${phasesDone} / ${phases.length} 단계` },
    { icon: <AudioLines size={16} />, label: "모드", value: String(run?.modes?.length ?? metrics.length ?? 0) },
    { icon: <Gauge size={16} />, label: "최고 WER", value: best.bestWer != null ? pct(best.bestWer) : "n/a" },
    { icon: <FileAudio size={16} />, label: "샘플", value: String(utterances.length) },
  ];
  return (
    <section className="statStrip">
      {stats.map((stat) => (
        <div className="statCard" key={stat.label}>
          <span className="statIcon">{stat.icon}</span>
          <div>
            <small>{stat.label}</small>
            <strong>{stat.value}</strong>
          </div>
        </div>
      ))}
    </section>
  );
}

function RunHeader({
  job,
  metrics,
  onAction,
  run,
  selectedStage,
}: {
  job?: RunJob;
  metrics: MetricRow[];
  onAction?: (action: "cancel" | "retry" | "resume") => void;
  run?: Run;
  selectedStage?: Stage;
}) {
  const error = runError(run, job, selectedStage);
  const canCancel = ["queued", "submitted", "running"].includes(job?.status ?? run?.status ?? "");
  const canRetry = ["failed", "cancelled"].includes(job?.status ?? run?.status ?? "");
  const artifacts = [
    { label: "run.json", href: run?.run_id ? artifactUrl(run.run_id, "run.json") : null, available: true },
    { label: "config.yaml", href: run?.run_id ? artifactUrl(run.run_id, "config.yaml") : null, available: true },
    { label: "summary.csv", href: run?.run_id ? artifactUrl(run.run_id, "metrics/summary.csv") : null, available: metrics.length > 0 },
    { label: "main_table.csv", href: run?.run_id ? runExportUrl(run.run_id, "main_table.csv") : null, available: true },
    { label: "error_breakdown.csv", href: run?.run_id ? runExportUrl(run.run_id, "error_breakdown.csv") : null, available: true },
    { label: "utterance_examples.jsonl", href: run?.run_id ? runExportUrl(run.run_id, "utterance_examples.jsonl") : null, available: true },
  ];
  return (
    <section className="panel runHeaderPanel">
      <div className="runHeaderMain">
        <div>
          <small>현재 실험 · 산출물</small>
          <h2>{run?.run_id ?? "선택된 실험 없음"}</h2>
          <p>{run?.experiment ?? "experiment"} · {checkpointLabel(run)}</p>
        </div>
        <em className={`pill status-${run?.status ?? "planned"}`}>{statusGlyph(run?.status, 13)}{statusLabel(run?.status)}</em>
      </div>
      {error ? (
        <div className="errorBanner">
          <TriangleAlert size={17} />
          <span>{error}</span>
        </div>
      ) : null}
      <div className="runMetaGrid">
        <div>
          <small>Created</small>
          <strong>{shortTime(run?.created_at) || "n/a"}</strong>
        </div>
        <div>
          <small>Updated</small>
          <strong>{shortTime(run?.updated_at ?? job?.finished_at) || "n/a"}</strong>
        </div>
        <div>
          <small>Job</small>
          <strong>{statusLabel(job?.status)}</strong>
        </div>
        <div>
          <small>Exit</small>
          <strong>{job?.exit_code ?? "n/a"}</strong>
        </div>
      </div>
      <div className="artifactLinks">
        <button className="artifactAction" disabled={!canCancel} onClick={() => onAction?.("cancel")} type="button">
          중단
        </button>
        <button className="artifactAction" disabled={!canRetry} onClick={() => onAction?.("retry")} type="button">
          재시도
        </button>
        <button className="artifactAction" disabled={!canRetry} onClick={() => onAction?.("resume")} type="button">
          이어서
        </button>
        {artifacts.map((artifact) => {
          return artifact.href && artifact.available ? (
            <a href={artifact.href} key={artifact.label} rel="noreferrer" target="_blank">
              <Download size={15} />
              <span>{artifact.label}</span>
            </a>
          ) : (
            <span className="artifactDisabled" key={artifact.label}>{artifact.label}</span>
          );
        })}
      </div>
    </section>
  );
}

function PipelineGraph({
  onSelectStage,
  run,
  selectedStageKey,
}: {
  onSelectStage?: (stage: Stage, key: string) => void;
  run?: Run;
  selectedStageKey?: string;
}) {
  const stages = run?.stages ?? [];
  return (
    <section className="panel pipeline">
      <div className="panelHeader">
        <div>
          <h2>단계별 상태</h2>
          <p>{run?.run_id ?? "선택된 실험 없음"}</p>
        </div>
        <Activity size={20} />
      </div>
      <div className="stageRail">
        {stages.map((stage, index) => {
          const key = stageKey(stage, index);
          return (
            <button
              className={selectedStageKey === key ? "stageNode active" : "stageNode"}
              key={key}
              onClick={() => onSelectStage?.(stage, key)}
              type="button"
            >
              <div className={`stageDot status-${stage.status}`}>{statusGlyph(stage.status)}</div>
              <div>
                <strong>{stageLabel(stage.name)}</strong>
                <span>{stageMeta(stage.name)}</span>
                <em>{statusLabel(stage.status, stage.dry_run)}</em>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function MetricBars({ metrics }: { metrics: MetricRow[] }) {
  const maxWer = Math.max(0.01, ...metrics.map((item) => item.wer ?? 0));
  const maxCer = Math.max(0.01, ...metrics.map((item) => item.cer ?? 0));
  const best = bestMetric(metrics);
  return (
    <section className="panel metricPanel">
      <div className="panelHeader">
        <div>
          <h2>품질 지표</h2>
          <p>mode별 intelligibility · speaker · naturalness · speed</p>
        </div>
        <BarChart3 size={20} />
      </div>
      <div className="metricRows">
        {metrics.length === 0 ? (
          <span className="emptyText">아직 평가 결과가 없습니다.</span>
        ) : (
          metrics.map((item) => (
            <div className={item.mode === best.bestMode ? "metricRow best" : "metricRow"} key={item.mode}>
              <div className="metricName">
                <span className="modeSwatch" style={{ background: modeColors[item.mode] ?? "#607080" }} />
                <strong>{item.mode}</strong>
                {item.mode === best.bestMode ? <span className="bestBadge">최저</span> : null}
                <small>{item.num_utterances ?? 0} utt</small>
              </div>
              <div className="bars">
                <div className="barTrack" title={`WER ${pct(item.wer)}`}>
                  <span style={{ width: `${Math.max(4, ((item.wer ?? 0) / maxWer) * 100)}%`, background: modeColors[item.mode] ?? "#607080" }} />
                </div>
                <div className="barTrack muted" title={`CER ${pct(item.cer)}`}>
                  <span style={{ width: `${Math.max(4, ((item.cer ?? 0) / maxCer) * 100)}%`, background: modeColors[item.mode] ?? "#607080" }} />
                </div>
              </div>
              <div className="metricNumbers">
                <span>{pct(item.wer)}</span>
                <span>{pct(item.cer)}</span>
              </div>
              <div className="acousticPills">
                <span title="Speaker similarity">SIM {compactNumber(item.speaker_similarity_mean ?? item.spk_sim_mean)}</span>
                <span title="UTMOS naturalness">UTMOS {compactNumber(item.utmos_mean)}</span>
                <span title="Real-time factor">RTF {compactNumber(item.rtf_mean, 3)}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function RunList({
  runs,
  selectedRunId,
  onSelect,
  runMetrics,
  title,
}: {
  runs: Run[];
  selectedRunId?: string;
  onSelect: (run: Run) => void;
  runMetrics: Record<string, RunMetricSummary>;
  title?: string;
}) {
  return (
    <section className="panel runList">
      <div className="panelHeader">
        <div>
          <h2>{title ?? "실험 기록"}</h2>
          <p>{runs.length} runs · 진행 단계와 최고 WER</p>
        </div>
        <ListFilter size={20} />
      </div>
      <div className="runRows">
        {runs.length === 0 ? (
          <span className="emptyText">아직 실험이 없습니다.</span>
        ) : (
          runs.map((run) => {
            const phases = derivePhases(run);
            const summary = runMetrics[run.run_id];
            return (
              <button
                className={run.run_id === selectedRunId ? "runRow active" : "runRow"}
                key={run.run_id}
                onClick={() => onSelect(run)}
              >
                <span className="runRowMain">
                  <strong>{run.run_id}</strong>
                  <small>{run.experiment ?? "experiment"}{run.updated_at ? ` · ${shortTime(run.updated_at)}` : ""}</small>
                </span>
                <span className="runRowProgress" aria-hidden>
                  {phases.map((phase) => (
                    <i
                      className={`phaseDot phase-${phase.status}`}
                      key={phase.def.key}
                      title={`${phase.def.label}: ${phaseStatusLabel(phase.status)}`}
                    />
                  ))}
                </span>
                <span className="runRowMeta">
                  {summary?.bestWer != null ? <em className="werTag">WER {pct(summary.bestWer)}</em> : null}
                  <em className={`pill status-${run.status}`}>{statusLabel(run.status)}</em>
                </span>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}

function RunLauncher({
  apiState,
  checkpoints,
  datasets,
  onCreate,
  speakerCheckpoints,
}: {
  apiState: "api" | "demo" | "loading";
  checkpoints: Checkpoint[];
  datasets: Dataset[];
  onCreate: (payload: RunCreatePayload) => Promise<void>;
  speakerCheckpoints: Checkpoint[];
}) {
  const [payload, setPayload] = useState<RunCreatePayload>(defaultRunPayload);
  const [activePreset, setActivePreset] = useState(runPresets[0].key);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function setValue<K extends keyof RunCreatePayload>(key: K, value: RunCreatePayload[K]) {
    setActivePreset("");
    setPayload((current) => normalizeRunPayload({ ...current, [key]: value }));
  }

  function applyPreset(preset: RunPreset) {
    setActivePreset(preset.key);
    setPayload((current) => presetPayload(current, preset));
  }

  function selectCheckpoint(checkpointId: string) {
    setActivePreset("");
    const checkpoint = checkpoints.find((item) => item.id === checkpointId);
    setPayload((current) => ({
      ...current,
      checkpoint_id: checkpointId,
      model: checkpoint?.model ?? current.model,
      checkpoint: checkpoint?.path ?? current.checkpoint,
      checkpoint_hash: checkpoint?.checkpoint_hash ?? current.checkpoint_hash,
      vocoder: checkpoint?.vocoder ?? current.vocoder,
    }));
  }

  function selectSpeakerCheckpoint(checkpointId: string) {
    setActivePreset("");
    const checkpoint = speakerCheckpoints.find((item) => item.id === checkpointId);
    setPayload((current) => normalizeRunPayload({
      ...current,
      speaker_checkpoint: checkpoint?.path ?? "",
      speaker_feat_type: checkpoint?.feat_type ?? current.speaker_feat_type,
    }));
  }

  function selectDataset(datasetId: string) {
    setActivePreset("");
    const dataset = datasets.find((item) => item.id === datasetId);
    setPayload((current) => normalizeRunPayload({
      ...current,
      dataset_id: datasetId,
      manifest: dataset?.manifest ?? current.manifest,
      language: dataset?.language ?? current.language,
    }));
  }

  useEffect(() => {
    if (datasets.length === 0 || datasets.some((item) => item.id === payload.dataset_id)) {
      return;
    }
    const dataset = datasets.find((item) => item.id === "dev_small_20") ?? datasets[0];
    setPayload((current) => normalizeRunPayload({
      ...current,
      dataset_id: dataset.id,
      manifest: dataset.manifest,
      language: dataset.language ?? current.language,
    }));
  }, [datasets, payload.dataset_id]);

  function toggleMode(mode: string) {
    setActivePreset("");
    setPayload((current) => {
      const modes = current.modes.includes(mode) ? current.modes.filter((item) => item !== mode) : [...current.modes, mode];
      return normalizeRunPayload({ ...current, modes });
    });
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    const normalizedPayload = normalizeRunPayload({
      ...payload,
      run_speaker_similarity: selectedSpeakerMissing ? false : payload.run_speaker_similarity,
      run_id: payload.run_id?.trim() || undefined,
      manifest: payload.manifest.trim(),
      experiment: payload.experiment.trim(),
      speaker_checkpoint: payload.speaker_checkpoint?.trim() ?? "",
    });
    setPayload(normalizedPayload);
    try {
      await onCreate(normalizedPayload);
    } finally {
      setIsSubmitting(false);
    }
  }

  const disabled = isSubmitting || apiState !== "api";
  const activePresetSummary = runPresets.find((preset) => preset.key === activePreset)?.summary;
  const selectedDataset = datasets.find((item) => item.id === payload.dataset_id);
  const audioMetricsReady = payload.run_audio_metrics && !payload.audio_metrics_dry_run;
  const selectedSpeakerCheckpoint = speakerCheckpoints.find((item) => item.path === payload.speaker_checkpoint);
  const selectedSpeakerMissing = selectedSpeakerCheckpoint?.exists === false;
  const speakerSimDisabled = !audioMetricsReady || !(payload.speaker_checkpoint?.trim() ?? "") || selectedSpeakerMissing;
  const utmosDisabled = !audioMetricsReady;
  useEffect(() => {
    if (selectedSpeakerMissing && payload.run_speaker_similarity) {
      setPayload((current) => ({ ...current, run_speaker_similarity: false }));
    }
  }, [payload.run_speaker_similarity, selectedSpeakerMissing]);
  return (
    <section className="panel launcher">
      <div className="panelHeader">
        <div>
          <h2>실험 시작</h2>
          <p>{apiState === "demo" ? "API 연결 필요" : "데이터 로드 → 음성 생성 → 평가"}</p>
        </div>
        <Rocket size={20} />
      </div>
      <form className="runForm" onSubmit={(event) => void submit(event)}>
        <div className="presetIntro">
          <strong>1. 시작 프리셋</strong>
          <span>{activePresetSummary ?? "세부 설정을 직접 조정했습니다."}</span>
        </div>
        <div className="presetPicker" aria-label="Run presets">
          {runPresets.map((preset) => (
            <button
              className={activePreset === preset.key ? "presetChoice active" : "presetChoice"}
              key={preset.key}
              type="button"
              onClick={() => applyPreset(preset)}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <div className="formSectionTitle">2. 데이터와 모델</div>
        <label>
          <span>실험 이름</span>
          <input value={payload.experiment} onChange={(event) => setValue("experiment", event.target.value)} required />
        </label>
        <label>
          <span>Dataset · {datasets.length}개</span>
          <select value={payload.dataset_id ?? ""} onChange={(event) => selectDataset(event.target.value)}>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.id} · {dataset.purpose || dataset.manifest}
              </option>
            ))}
          </select>
        </label>
        <div className="datasetPicker" aria-label="Datasets">
          {datasets.map((dataset) => (
            <button
              className={dataset.id === payload.dataset_id ? "datasetChoice active" : "datasetChoice"}
              key={dataset.id}
              onClick={() => selectDataset(dataset.id)}
              type="button"
            >
              <strong>{dataset.id}</strong>
              <small>
                {dataset.num_utterances ?? "n/a"} utt · {dataset.subsets?.join(", ") || "all subsets"}
              </small>
            </button>
          ))}
        </div>
        <div className="datasetMeta">
          <span>{selectedDataset?.num_utterances ?? "n/a"} utt</span>
          <span>{selectedDataset?.subsets?.join(", ") || "all subsets"}</span>
          <span>{payload.language}</span>
        </div>
        <label>
          <span>Manifest 경로</span>
          <input value={payload.manifest} onChange={(event) => setValue("manifest", event.target.value)} required />
        </label>
        <label>
          <span>Checkpoint</span>
          <select value={payload.checkpoint_id ?? ""} onChange={(event) => selectCheckpoint(event.target.value)}>
            {checkpoints.map((checkpoint) => (
              <option key={checkpoint.id} value={checkpoint.id}>
                {checkpoint.id} · {checkpoint.model}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Run ID</span>
          <input placeholder="auto" value={payload.run_id ?? ""} onChange={(event) => setValue("run_id", event.target.value)} />
        </label>
        <div className="formSectionTitle">3. 비교할 mode</div>
        <div className="modePicker" aria-label="Modes">
          {availableModes.map((mode) => (
            <button className={payload.modes.includes(mode) ? "modeChoice active" : "modeChoice"} key={mode} type="button" onClick={() => toggleMode(mode)}>
              <span className="modeSwatch" style={{ background: modeColors[mode] ?? "#607080" }} />
              {mode}
            </button>
          ))}
        </div>
        <div className="formSectionTitle">4. 실행할 단계</div>
        <div className="toggleGrid">
          <label className="checkRow">
            <input checked={payload.run_posterior_extraction} type="checkbox" onChange={(event) => setValue("run_posterior_extraction", event.target.checked)} />
            <span>Posterior 생성</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_inference} type="checkbox" onChange={(event) => setValue("run_inference", event.target.checked)} />
            <span>음성 생성</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_audio_metrics} type="checkbox" onChange={(event) => setValue("run_audio_metrics", event.target.checked)} />
            <span>음향 평가</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_prediction} type="checkbox" onChange={(event) => setValue("run_prediction", event.target.checked)} />
            <span>ASR 변환</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_metrics} type="checkbox" onChange={(event) => setValue("run_metrics", event.target.checked)} />
            <span>점수 계산</span>
          </label>
        </div>
        <div className="formSectionTitle">5. 안전 옵션</div>
        <label>
          <span>Metric normalizer</span>
          <select value={payload.metrics_normalizer} onChange={(event) => setValue("metrics_normalizer", event.target.value)}>
            <option value="paper">paper</option>
            <option value="lowercase">lowercase</option>
            <option value="none">none</option>
          </select>
        </label>
        <label>
          <span>Hard ref text</span>
          <select value={payload.hard_ref_text_source} onChange={(event) => setValue("hard_ref_text_source", event.target.value)}>
            <option value="manifest">manifest</option>
            <option value="empty">empty ASR baseline</option>
          </select>
        </label>
        <div className="toggleGrid dryToggles">
          <label className="checkRow">
            <input checked={payload.skip_whisper} type="checkbox" onChange={(event) => setValue("skip_whisper", event.target.checked)} />
            <span>Manifest text 사용</span>
          </label>
          <label className="checkRow">
            <input checked={payload.inference_dry_run} type="checkbox" onChange={(event) => setValue("inference_dry_run", event.target.checked)} />
            <span>음성 계획만</span>
          </label>
          <label className="checkRow">
            <input checked={payload.prediction_dry_run} type="checkbox" onChange={(event) => setValue("prediction_dry_run", event.target.checked)} />
            <span>ASR 계획만</span>
          </label>
          <label className="checkRow">
            <input checked={payload.audio_metrics_dry_run} type="checkbox" onChange={(event) => setValue("audio_metrics_dry_run", event.target.checked)} />
            <span>음향 계획만</span>
          </label>
          <label className="checkRow">
            <input
              checked={payload.run_speaker_similarity && !speakerSimDisabled}
              disabled={speakerSimDisabled}
              type="checkbox"
              onChange={(event) => setValue("run_speaker_similarity", event.target.checked)}
            />
            <span>Speaker SIM</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_utmos} disabled={utmosDisabled} type="checkbox" onChange={(event) => setValue("run_utmos", event.target.checked)} />
            <span>UTMOS</span>
          </label>
          <label className="checkRow">
            <input checked={payload.fail_if_exists} type="checkbox" onChange={(event) => setValue("fail_if_exists", event.target.checked)} />
            <span>새 ID 강제</span>
          </label>
        </div>
        <label>
          <span>Speaker checkpoint</span>
          <select value={selectedSpeakerCheckpoint?.id ?? ""} onChange={(event) => selectSpeakerCheckpoint(event.target.value)}>
            <option value="">직접 입력 또는 선택 안 함</option>
            {speakerCheckpoints.map((checkpoint) => (
              <option disabled={checkpoint.exists === false} key={checkpoint.id} value={checkpoint.id}>
                {checkpoint.id} · {checkpoint.feat_type ?? checkpoint.model}{checkpoint.exists === false ? " · 파일 없음" : ""}
              </option>
            ))}
          </select>
          <input placeholder="optional ECAPA checkpoint" value={payload.speaker_checkpoint ?? ""} onChange={(event) => setValue("speaker_checkpoint", event.target.value)} />
        </label>
        <div className="formFooter">
          <label>
            <span>Seed</span>
            <input
              min="0"
              type="number"
              value={payload.seed}
              onChange={(event) => setValue("seed", Number(event.target.value))}
            />
          </label>
          <button className="primaryButton" disabled={disabled || payload.modes.length === 0} type="submit">
            <Play size={16} />
            <span>{isSubmitting ? "시작 중" : "실험 시작"}</span>
          </button>
        </div>
      </form>
    </section>
  );
}

function UtteranceInspector({ utterances, runId }: { utterances: Utterance[]; runId?: string }) {
  const [selected, setSelected] = useState(0);
  const [failureFilter, setFailureFilter] = useState("all");
  const failureTypes = useMemo(
    () => [...new Set(utterances.flatMap((item) => [item.failure_type, ...Object.values(item.modes).map((mode) => mode.failure_type)]).filter(Boolean).map(String))].sort(),
    [utterances],
  );
  const filteredUtterances = useMemo(
    () =>
      failureFilter === "all"
        ? utterances
        : utterances.filter((item) => item.failure_type === failureFilter || Object.values(item.modes).some((mode) => mode.failure_type === failureFilter)),
    [failureFilter, utterances],
  );
  const utterance = filteredUtterances[Math.min(selected, Math.max(0, filteredUtterances.length - 1))];
  const modes = Object.keys(utterance?.modes ?? {});
  const refSrc = runId ? artifactUrl(runId, utterance?.ref_audio) : null;
  return (
    <section className="panel inspector">
      <div className="panelHeader">
        <div>
          <h2>샘플 검토</h2>
          <p>{filteredUtterances.length} / {utterances.length} samples</p>
        </div>
        <select className="compactSelect" value={failureFilter} onChange={(event) => {
          setFailureFilter(event.target.value);
          setSelected(0);
        }}>
          <option value="all">all failures</option>
          {failureTypes.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </div>
      <div className="inspectorGrid">
        <div className="utteranceList">
          {filteredUtterances.map((item, index) => (
            <button className={index === selected ? "utteranceButton active" : "utteranceButton"} key={item.utterance_id} onClick={() => setSelected(index)}>
              <strong>{item.utterance_id}</strong>
              <span>{item.subset ?? "subset"} · H {item.posterior_entropy ?? "n/a"}</span>
            </button>
          ))}
        </div>
        <div className="utteranceDetail">
          <div className="textGrid">
            <div className="textBlock">
              <small>Reference</small>
              <p>{utterance?.reference_text ?? utterance?.text ?? "No text"}</p>
            </div>
            <div className="textBlock">
              <small>Target</small>
              <p>{utterance?.gen_text ?? "No target text"}</p>
            </div>
            <div className="textBlock">
              <small>Posterior</small>
              <p>{utterance?.one_best ?? "No posterior text"}</p>
            </div>
          </div>
          <div className="referenceAudio">
            <strong>reference audio</strong>
            {refSrc ? <audio controls src={refSrc} /> : <span className="audioPlaceholder">external or missing ref</span>}
          </div>
          <div className="audioGrid">
            {modes.map((mode) => {
              const src = runId ? artifactUrl(runId, utterance?.modes[mode]?.generated_audio) : null;
              const modeData = utterance?.modes[mode];
              return (
                <div className="audioItem" key={mode}>
                  <div>
                    <span className="modeSwatch" style={{ background: modeColors[mode] ?? "#607080" }} />
                    <strong>{mode}</strong>
                    <em>{modeData?.status ?? "n/a"} · WER {pct(modeData?.wer)}</em>
                  </div>
                  {src ? <audio controls src={src} /> : <span className="audioPlaceholder">no wav</span>}
                  <small>prediction</small>
                  <p>{modeData?.prediction_text || "empty"}</p>
                  <div className="diffLine">
                    {(modeData?.diff ?? []).map((token, index) => (
                      <span className={`diff-${token.op}`} key={`${mode}-${index}`}>{token.text}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function FailureTable({ metrics }: { metrics: MetricRow[] }) {
  const ranked = useMemo(
    () =>
      [...metrics].sort(
        (a, b) =>
          (b.wer_deletions ?? 0) + (b.wer_insertions ?? 0) + (b.wer_substitutions ?? 0) -
          ((a.wer_deletions ?? 0) + (a.wer_insertions ?? 0) + (a.wer_substitutions ?? 0)),
      ),
    [metrics],
  );
  return (
    <section className="panel compactPanel">
      <div className="panelHeader">
        <div>
          <h2>오류 분석</h2>
          <p>word-level counts</p>
        </div>
        <Database size={20} />
      </div>
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Sub</th>
            <th>Del</th>
            <th>Ins</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((row) => (
            <tr key={row.mode}>
              <td>{row.mode}</td>
              <td>{row.wer_substitutions ?? 0}</td>
              <td>{row.wer_deletions ?? 0}</td>
              <td>{row.wer_insertions ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function MetricContractPanel({ metrics }: { metrics: MetricRow[] }) {
  const normalizers = [...new Set(metrics.map((item) => item.normalizer).filter(Boolean).map(String))];
  const coverage = metrics
    .map((item) => item.prediction_coverage)
    .filter((value): value is number => typeof value === "number");
  const deletionRates = metrics
    .map((item) => item.wer_deletion_rate)
    .filter((value): value is number => typeof value === "number");
  const latency = metrics
    .map((item) => item.generation_elapsed_sec_mean)
    .filter((value): value is number => typeof value === "number");
  const rtf = metrics
    .map((item) => item.rtf_mean)
    .filter((value): value is number => typeof value === "number");
  const speakerSimilarity = metrics
    .map((item) => item.speaker_similarity_mean ?? item.spk_sim_mean)
    .filter((value): value is number => typeof value === "number");
  const utmos = metrics
    .map((item) => item.utmos_mean)
    .filter((value): value is number => typeof value === "number");
  const minCoverage = coverage.length > 0 ? Math.min(...coverage) : null;
  const maxDeletion = deletionRates.length > 0 ? Math.max(...deletionRates) : null;
  const meanLatency = latency.length > 0 ? latency.reduce((sum, value) => sum + value, 0) / latency.length : null;
  const meanRtf = rtf.length > 0 ? rtf.reduce((sum, value) => sum + value, 0) / rtf.length : null;
  const meanSpeaker = speakerSimilarity.length > 0 ? speakerSimilarity.reduce((sum, value) => sum + value, 0) / speakerSimilarity.length : null;
  const meanUtmos = utmos.length > 0 ? utmos.reduce((sum, value) => sum + value, 0) / utmos.length : null;
  const rows = [
    { label: "Normalizer", value: normalizers.join(", ") || "n/a", icon: <BadgeCheck size={16} /> },
    { label: "Coverage", value: pct(minCoverage), icon: <Target size={16} /> },
    { label: "Max deletion", value: pct(maxDeletion), icon: <TrendingDown size={16} /> },
    { label: "Mean gen sec", value: compactNumber(meanLatency), icon: <Clock size={16} /> },
    { label: "Mean RTF", value: compactNumber(meanRtf, 3), icon: <Gauge size={16} /> },
    { label: "Mean SIM", value: compactNumber(meanSpeaker), icon: <AudioLines size={16} /> },
    { label: "Mean UTMOS", value: compactNumber(meanUtmos), icon: <Sparkles size={16} /> },
  ];
  return (
    <section className="panel contractPanel">
      <div className="panelHeader">
        <div>
          <h2>평가 계약</h2>
          <p>normalization · coverage · acoustic metrics</p>
        </div>
        <BadgeCheck size={20} />
      </div>
      <div className="contractGrid">
        {rows.map((row) => (
          <div className="contractItem" key={row.label}>
            <span>{row.icon}</span>
            <small>{row.label}</small>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function gateLabel(name: string): string {
  const labels: Record<string, string> = {
    best_wer: "Best WER",
    best_cer: "Best CER",
    prediction_coverage: "Coverage",
    deletion_rate: "Deletion",
    speaker_similarity: "Speaker SIM",
    utmos: "UTMOS",
    rtf: "RTF",
  };
  return labels[name] ?? name.replace(/_/g, " ");
}

function QualityGatePanel({ report }: { report: EvaluationReport }) {
  return (
    <section className="panel gatePanel">
      <div className="panelHeader">
        <div>
          <h2>품질 Gate</h2>
          <p>{report.evaluated_runs} evaluated runs · {report.evaluated_rows} rows</p>
        </div>
        <Target size={20} />
      </div>
      <div className="gateGrid">
        {report.quality_gates.length === 0 ? (
          <span className="emptyText">No gates</span>
        ) : (
          report.quality_gates.map((gate) => (
            <div className={`gateItem gate-${gate.status}`} key={gate.name}>
              <div>
                <strong>{gateLabel(gate.name)}</strong>
                <em>{gate.status}</em>
              </div>
              <span>{gateMetricValue(gate.value, gate.unit)}</span>
              <small>{gate.direction === "lower" ? "≤" : "≥"} {gateMetricValue(gate.target, gate.unit)}</small>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function FailureMixPanel({ report }: { report: EvaluationReport }) {
  const entries = [
    { key: "substitutions", label: "Sub", color: "#8d5fbf" },
    { key: "deletions", label: "Del", color: "#b8526b" },
    { key: "insertions", label: "Ins", color: "#b07d2f" },
  ];
  return (
    <section className="panel failureMixPanel">
      <div className="panelHeader">
        <div>
          <h2>실패 분포</h2>
          <p>word-level aggregate</p>
        </div>
        <Database size={20} />
      </div>
      <div className="failureBars">
        {entries.map((entry) => (
          <div className="failureBarRow" key={entry.key}>
            <div>
              <strong>{entry.label}</strong>
              <span>{metricCount(report.failure_mix?.[entry.key])}</span>
            </div>
            <div className="barTrack">
              <span style={{ width: `${Math.max(2, (report.failure_rates?.[entry.key] ?? 0) * 100)}%`, background: entry.color }} />
            </div>
            <em>{pct(report.failure_rates?.[entry.key])}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function LeaderboardTable({
  rows,
  runs,
  onSelectRun,
}: {
  rows: CompareRow[];
  runs: Run[];
  onSelectRun: (run: Run) => void;
}) {
  function selectRun(runId: string) {
    const run = runs.find((item) => item.run_id === runId);
    if (run) onSelectRun(run);
  }
  return (
    <section className="panel leaderboardPanel">
      <div className="panelHeader">
        <div>
          <h2>Leaderboard</h2>
          <p>{rows.length} ranked mode rows</p>
        </div>
        <BarChart3 size={20} />
      </div>
      <div className="tableScroller">
        <table className="compareTable">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Run</th>
              <th>Mode</th>
              <th>WER</th>
              <th>CER</th>
              <th>Coverage</th>
              <th>Del</th>
              <th>SIM</th>
              <th>UTMOS</th>
              <th>RTF</th>
              <th>Seeds</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr><td colSpan={11}>No ranked rows</td></tr>
            ) : (
              rows.map((row, index) => (
                <tr key={`${row.run_id}-${row.mode}-${row.subset ?? "all"}-${index}`}>
                  <td>{index + 1}</td>
                  <td>
                    <button className="tableLink" type="button" onClick={() => selectRun(row.run_id)}>
                      {row.run_id}
                    </button>
                  </td>
                  <td>
                    <span className="modeCell">
                      <span className="modeSwatch" style={{ background: modeColors[row.mode ?? ""] ?? "#607080" }} />
                      {row.mode ?? "mode"}
                    </span>
                  </td>
                  <td>{pct(row.wer)}</td>
                  <td>{pct(row.cer)}</td>
                  <td>{pct(row.prediction_coverage)}</td>
                  <td>{pct(row.wer_deletion_rate)}</td>
                  <td>{compactNumber(row.speaker_similarity_mean ?? row.spk_sim_mean)}</td>
                  <td>{compactNumber(row.utmos_mean)}</td>
                  <td>{compactNumber(row.rtf_mean, 3)}</td>
                  <td>{row.seed_count ?? 1}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelRegistryPage({
  models,
  runs,
  onSelectRun,
}: {
  models: ModelCard[];
  runs: Run[];
  onSelectRun: (run: Run) => void;
}) {
  function selectRun(runId?: string | null) {
    const run = runs.find((item) => item.run_id === runId);
    if (run) onSelectRun(run);
  }
  return (
    <section className="panel modelRegistryPanel">
      <div className="panelHeader">
        <div>
          <h2>Checkpoint Registry</h2>
          <p>{models.length} registered or observed models</p>
        </div>
        <PackageCheck size={20} />
      </div>
      <div className="modelGrid">
        {models.length === 0 ? (
          <span className="emptyText">No models</span>
        ) : (
          models.map((model) => (
            <article className="modelCard" key={model.id}>
              <div className="modelCardTop">
                <span className={`modelStage stage-${model.stage ?? "registered"}`}>{model.stage ?? "registered"}</span>
                <em>{model.vocoder || "vocoder n/a"}</em>
              </div>
              <h3>{model.id}</h3>
              <p>{model.model}</p>
              <div className="modelPath">{model.path}</div>
              <div className="modelMetricGrid">
                <div><small>Best WER</small><strong>{pct(model.best_wer)}</strong></div>
                <div><small>Best CER</small><strong>{pct(model.best_cer)}</strong></div>
                <div><small>Runs</small><strong>{model.num_runs ?? 0}</strong></div>
                <div><small>Eval rows</small><strong>{model.num_evaluations ?? 0}</strong></div>
              </div>
              <div className="modelLineage">
                <span>{model.dataset || "dataset n/a"}</span>
                <span>{model.git_commit ? model.git_commit.slice(0, 8) : "git n/a"}</span>
                <span>{model.checkpoint_hash ? "hash set" : "hash n/a"}</span>
              </div>
              <div className="modelActions">
                <button disabled={!model.best_run_id} onClick={() => selectRun(model.best_run_id)} type="button">
                  <BarChart3 size={15} />
                  <span>best run</span>
                </button>
                <button disabled={!model.latest_run_id} onClick={() => selectRun(model.latest_run_id)} type="button">
                  <Clock size={15} />
                  <span>latest</span>
                </button>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

function EvaluationCenter({
  report,
  runs,
  onSelectRun,
}: {
  report: EvaluationReport;
  runs: Run[];
  onSelectRun: (run: Run) => void;
}) {
  const summary = [
    { label: "Best WER", value: pct(report.best?.wer), icon: <TrendingDown size={16} /> },
    { label: "Best mode", value: report.best?.mode ?? "n/a", icon: <AudioLines size={16} /> },
    { label: "SIM", value: compactNumber(report.best?.speaker_similarity_mean ?? report.best?.spk_sim_mean), icon: <AudioLines size={16} /> },
    { label: "UTMOS", value: compactNumber(report.best?.utmos_mean), icon: <Sparkles size={16} /> },
    { label: "Runs", value: String(report.total_runs ?? 0), icon: <FlaskConical size={16} /> },
    { label: "Evaluated", value: String(report.evaluated_runs ?? 0), icon: <BadgeCheck size={16} /> },
  ];
  return (
    <div className="dashboardGrid evaluationGrid">
      <section className="statStrip">
        {summary.map((item) => (
          <div className="statCard" key={item.label}>
            <span className="statIcon">{item.icon}</span>
            <div>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </div>
          </div>
        ))}
      </section>
      <div className="evaluationSplit">
        <QualityGatePanel report={report} />
        <FailureMixPanel report={report} />
      </div>
      <LeaderboardTable rows={report.leaderboard ?? []} runs={runs} onSelectRun={onSelectRun} />
    </div>
  );
}

function uniqueValues(rows: CompareRow[], key: keyof CompareRow): string[] {
  return [...new Set(rows.map((row) => row[key]).filter((value) => value !== undefined && value !== null && value !== "").map(String))].sort();
}

function ComparePage({
  rows,
  runs,
  onSelectRun,
}: {
  rows: CompareRow[];
  runs: Run[];
  onSelectRun: (run: Run) => void;
}) {
  const [experiment, setExperiment] = useState("all");
  const [checkpoint, setCheckpoint] = useState("all");
  const [mode, setMode] = useState("all");
  const [subset, setSubset] = useState("all");
  const [seed, setSeed] = useState("all");
  const [rowType, setRowType] = useState("all");
  const [query, setQuery] = useState("");

  const experiments = uniqueValues(rows, "experiment");
  const checkpoints = uniqueValues(rows, "checkpoint");
  const modes = uniqueValues(rows, "mode");
  const subsets = uniqueValues(rows, "subset");
  const seeds = uniqueValues(rows, "seed");
  const rowTypes = uniqueValues(rows, "row_type");

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return rows
      .filter((row) => experiment === "all" || row.experiment === experiment)
      .filter((row) => checkpoint === "all" || row.checkpoint === checkpoint)
      .filter((row) => mode === "all" || row.mode === mode)
      .filter((row) => subset === "all" || row.subset === subset)
      .filter((row) => seed === "all" || String(row.seed ?? "") === seed)
      .filter((row) => rowType === "all" || row.row_type === rowType)
      .filter((row) => {
        if (!normalizedQuery) return true;
        return [row.run_id, row.experiment, row.checkpoint, row.mode].some((value) => String(value ?? "").toLowerCase().includes(normalizedQuery));
      })
      .sort((a, b) => (a.wer ?? Number.POSITIVE_INFINITY) - (b.wer ?? Number.POSITIVE_INFINITY));
  }, [checkpoint, experiment, mode, query, rows, rowType, seed, subset]);

  function selectRun(runId: string) {
    const run = runs.find((item) => item.run_id === runId);
    if (run) {
      onSelectRun(run);
    }
  }

  return (
    <section className="panel comparePanel">
      <div className="panelHeader">
        <div>
          <h2>실험 비교</h2>
          <p>{filtered.length} mode rows</p>
        </div>
        {experiment !== "all" ? (
          <a className="exportButton" href={experimentExportUrl(experiment, "ablation_table.csv")} rel="noreferrer" target="_blank">
            <Download size={15} />
            <span>ablation_table.csv</span>
          </a>
        ) : (
          <Database size={20} />
        )}
      </div>
      <div className="compareFilters">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="run / experiment / checkpoint" />
        <select value={experiment} onChange={(event) => setExperiment(event.target.value)}>
          <option value="all">all experiments</option>
          {experiments.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)}>
          <option value="all">all checkpoints</option>
          {checkpoints.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <select value={mode} onChange={(event) => setMode(event.target.value)}>
          <option value="all">all modes</option>
          {modes.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <select value={subset} onChange={(event) => setSubset(event.target.value)}>
          <option value="all">all subsets</option>
          {subsets.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <select value={seed} onChange={(event) => setSeed(event.target.value)}>
          <option value="all">all seeds</option>
          {seeds.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <select value={rowType} onChange={(event) => setRowType(event.target.value)}>
          <option value="all">all row types</option>
          {rowTypes.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </div>
      <div className="tableScroller">
        <table className="compareTable">
          <thead>
            <tr>
              <th>Run</th>
              <th>Experiment</th>
              <th>Checkpoint</th>
              <th>Type</th>
              <th>Seed</th>
              <th>Subset</th>
              <th>Mode</th>
              <th>WER</th>
              <th>WER CI</th>
              <th>p Holm</th>
              <th>Sig</th>
              <th>CER</th>
              <th>Coverage</th>
              <th>SIM</th>
              <th>UTMOS</th>
              <th>RTF</th>
              <th>Del%</th>
              <th>Norm</th>
              <th>Sub</th>
              <th>Del</th>
              <th>Ins</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={21}>No metric rows</td>
              </tr>
            ) : (
              filtered.map((row) => (
                <tr key={`${row.run_id}-${row.mode}-${row.subset ?? "all"}`}>
                  <td>
                    <button className="tableLink" type="button" onClick={() => selectRun(row.run_id)}>
                      {row.run_id}
                    </button>
                  </td>
                  <td>{row.experiment ?? "n/a"}</td>
                  <td>{row.checkpoint || "n/a"}</td>
                  <td>{row.row_type === "seed_aggregate" ? "seed group" : "seed"}</td>
                  <td>{row.seed ?? "n/a"}</td>
                  <td>{row.subset ?? "all"}</td>
                  <td>
                    <span className="modeCell">
                      <span className="modeSwatch" style={{ background: modeColors[row.mode ?? ""] ?? "#607080" }} />
                      {row.mode ?? "mode"}
                    </span>
                  </td>
                  <td>{metricWithStd(row.wer, row.wer_std, pct)}</td>
                  <td>{row.wer_ci_low === undefined || row.wer_ci_low === null ? "n/a" : `${pct(row.wer_ci_low)}..${pct(row.wer_ci_high)}`}</td>
                  <td>{pValue(row.wer_holm_p_value ?? row.wer_p_value)}</td>
                  <td>
                    {row.wer_significant === undefined || row.wer_significant === null ? (
                      <span className="sigBadge neutral">n/a</span>
                    ) : (
                      <span className={row.wer_significant ? "sigBadge significant" : "sigBadge neutral"}>
                        {row.wer_significant ? "sig" : "ns"}
                      </span>
                    )}
                  </td>
                  <td>{metricWithStd(row.cer, row.cer_std, pct)}</td>
                  <td>{pct(row.prediction_coverage)}</td>
                  <td>{metricWithStd(row.speaker_similarity_mean ?? row.spk_sim_mean, row.speaker_similarity_mean_std ?? row.spk_sim_mean_std, compactNumber)}</td>
                  <td>{metricWithStd(row.utmos_mean, row.utmos_mean_std, compactNumber)}</td>
                  <td>{metricWithStd(row.rtf_mean, row.rtf_mean_std, (value) => compactNumber(value, 3))}</td>
                  <td>{pct(row.wer_deletion_rate)}</td>
                  <td>{row.normalizer ?? "n/a"}</td>
                  <td>{row.substitutions ?? 0}</td>
                  <td>{row.deletions ?? 0}</td>
                  <td>{row.insertions ?? 0}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function JobLogs({ job, logs, selectedStage }: { job?: RunJob; logs?: RunLogs; selectedStage?: Stage }) {
  const [copied, setCopied] = useState(false);
  const files = filteredLogFiles(logs, selectedStage);
  const command = selectedStage?.command ?? job?.command;
  const commandText = command?.join(" ") ?? "";

  async function copyCommand() {
    if (!commandText) return;
    await navigator.clipboard.writeText(commandText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <section className="panel logsPanel">
      <div className="panelHeader">
        <div>
          <h2>실행 로그</h2>
          <p>{selectedStage ? `${stageLabel(selectedStage.name)} · ${statusLabel(selectedStage.status, selectedStage.dry_run)}` : job ? `${statusLabel(job.status)}${job.pid ? ` pid ${job.pid}` : ""}` : "선택된 작업 없음"}</p>
        </div>
        <Terminal size={20} />
      </div>
      {commandText ? (
        <div className="commandBox">
          <div>
            <strong>Command</strong>
            <button className="copyButton" onClick={() => void copyCommand()} type="button">
              <Clipboard size={15} />
              <span>{copied ? "복사됨" : "복사"}</span>
            </button>
          </div>
          <pre>{commandText}</pre>
        </div>
      ) : null}
      <div className="jobMeta">
        <span>{shortTime(job?.started_at) || "not started"}</span>
        <span>{shortTime(job?.finished_at) || "waiting"}</span>
        <span>{job?.exit_code ?? "exit n/a"}</span>
      </div>
      <div className="logFiles">
        {files.length === 0 ? (
          <pre>아직 로그가 없습니다.</pre>
        ) : (
          files.map((file) => (
            <div className="logFile" key={file.path}>
              <div>
                <strong>{file.path}</strong>
                <span>{file.size_bytes} bytes</span>
              </div>
              <pre>{file.content || "empty"}</pre>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function EventTimeline({ events }: { events?: RunEvents }) {
  const rows = events?.events ?? [];
  return (
    <section className="panel eventPanel">
      <div className="panelHeader">
        <div>
          <h2>이벤트 타임라인</h2>
          <p>{rows.length} structured events</p>
        </div>
        <Activity size={20} />
      </div>
      <div className="eventRows">
        {rows.length === 0 ? (
          <span className="emptyText">events.jsonl 없음</span>
        ) : (
          rows.map((event, index) => (
            <div className={`eventRow level-${event.level}`} key={`${event.time}-${index}`}>
              <span>{shortTime(event.time)}</span>
              <strong>{stageLabel(event.stage)}{event.mode ? ` · ${event.mode}` : ""}</strong>
              <em>{event.level}</em>
              <p>{event.message}</p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function App() {
  const [runs, setRuns] = useState<Run[]>(demoRuns);
  const [selectedRun, setSelectedRun] = useState<Run | undefined>(demoRuns[0]);
  const [metrics, setMetrics] = useState<MetricRow[]>(demoMetrics.modes);
  const [utterances, setUtterances] = useState<Utterance[]>(demoUtterances);
  const [compareRows, setCompareRows] = useState<CompareRow[]>(demoCompareRows);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>(fallbackCheckpoints);
  const [speakerCheckpoints, setSpeakerCheckpoints] = useState<Checkpoint[]>(fallbackSpeakerCheckpoints);
  const [datasets, setDatasets] = useState<Dataset[]>(fallbackDatasets);
  const [models, setModels] = useState<ModelCard[]>(fallbackModelRegistry.models);
  const [evaluationReport, setEvaluationReport] = useState<EvaluationReport>(fallbackEvaluationReport);
  const [job, setJob] = useState<RunJob | undefined>();
  const [logs, setLogs] = useState<RunLogs | undefined>();
  const [events, setEvents] = useState<RunEvents | undefined>();
  const [apiState, setApiState] = useState<"api" | "demo" | "loading">("loading");
  const [activeView, setActiveView] = useState<ViewKey>(() => readViewFromHash() ?? "overview");
  const [selectedStageKey, setSelectedStageKey] = useState("");
  const [notice, setNotice] = useState("");

  const selectedStage = useMemo(() => {
    const stages = selectedRun?.stages ?? [];
    return stages.find((stage, index) => stageKey(stage, index) === selectedStageKey);
  }, [selectedRun?.stages, selectedStageKey]);

  const phases = useMemo(() => derivePhases(selectedRun), [selectedRun]);
  const best = useMemo(() => bestMetric(metrics), [metrics]);
  const runMetrics = useMemo(() => buildRunMetrics(compareRows), [compareRows]);

  async function loadRunDetails(run: Run) {
    const [loadedMetrics, loadedUtterances, loadedJob, loadedLogs, loadedEvents] = await Promise.all([
      fetchMetrics(run.run_id),
      fetchUtterances(run.run_id),
      fetchRunJob(run.run_id),
      fetchRunLogs(run.run_id),
      fetchRunEvents(run.run_id),
    ]);
    setMetrics(loadedMetrics.modes ?? []);
    setUtterances(loadedUtterances);
    setJob(loadedJob);
    setLogs(loadedLogs);
    setEvents(loadedEvents);
  }

  async function refresh() {
    setApiState("loading");
    try {
      const [loadedRuns, loadedCheckpoints, loadedDatasets, loadedCompare, loadedModels, loadedEvaluationReport] = await Promise.all([
        fetchRuns(),
        fetchCheckpoints(),
        fetchDatasets(),
        fetchCompare(),
        fetchModels(),
        fetchEvaluationReport(),
      ]);
      setCheckpoints(loadedCheckpoints.checkpoints.length > 0 ? loadedCheckpoints.checkpoints : fallbackCheckpoints);
      setSpeakerCheckpoints((loadedCheckpoints.speaker_checkpoints ?? []).length > 0 ? loadedCheckpoints.speaker_checkpoints ?? [] : fallbackSpeakerCheckpoints);
      setDatasets(mergeDatasetsWithFallback(loadedDatasets.datasets ?? []));
      setCompareRows(loadedCompare.rows ?? []);
      setModels(loadedModels.models ?? []);
      setEvaluationReport(loadedEvaluationReport);
      if (loadedRuns.length === 0) {
        setRuns([]);
        setSelectedRun(undefined);
        setMetrics([]);
        setUtterances([]);
        setCompareRows([]);
        setModels(loadedModels.models ?? []);
        setEvaluationReport(loadedEvaluationReport);
        setJob(undefined);
        setEvents(undefined);
        setLogs(undefined);
        setApiState("api");
        setSelectedStageKey("");
        return;
      }
      const current = loadedRuns[0];
      setRuns(loadedRuns);
      setSelectedRun(current);
      await loadRunDetails(current);
      setApiState("api");
    } catch {
      setRuns(demoRuns);
      setSelectedRun(demoRuns[0]);
      setMetrics(demoMetrics.modes);
      setUtterances(demoUtterances);
      setCompareRows(demoCompareRows);
      setCheckpoints(fallbackCheckpoints);
      setSpeakerCheckpoints(fallbackSpeakerCheckpoints);
      setDatasets(fallbackDatasets);
      setModels(fallbackModelRegistry.models);
      setEvaluationReport(fallbackEvaluationReport);
      setJob(undefined);
      setLogs(undefined);
      setEvents(undefined);
      setSelectedStageKey(defaultStageSelection(demoRuns[0]));
      setApiState("demo");
    }
  }

  async function selectRun(run: Run) {
    setSelectedRun(run);
    setSelectedStageKey(defaultStageSelection(run));
    try {
      await loadRunDetails(run);
      setApiState("api");
    } catch {
      setMetrics(demoMetrics.modes);
      setUtterances(demoUtterances);
      setJob(undefined);
      setLogs(undefined);
      setEvents(undefined);
      setSelectedStageKey(defaultStageSelection(demoRuns[0]));
      setApiState("demo");
    }
  }

  async function handleCreateRun(payload: RunCreatePayload) {
    setNotice("");
    try {
      const response = await createRun(payload);
      if (response.run) {
        setRuns((current) => [response.run as Run, ...current.filter((run) => run.run_id !== response.run?.run_id)]);
        await selectRun(response.run);
        setNotice(`Started ${response.run.run_id}`);
      } else {
        await refresh();
        setNotice(`Started ${response.run_id ?? response.status}`);
      }
      setActiveView("logs");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to start run";
      setNotice(message);
    }
  }

  useEffect(() => {
    if (apiState !== "api" || !selectedRun) {
      return;
    }
    const status = job?.status ?? selectedRun.status;
    if (!["running", "submitted", "queued"].includes(status)) {
      return;
    }

    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const [freshRun, freshJob, freshLogs, freshEvents] = await Promise.all([
            fetchRun(selectedRun.run_id),
            fetchRunJob(selectedRun.run_id),
            fetchRunLogs(selectedRun.run_id),
            fetchRunEvents(selectedRun.run_id),
          ]);
          setSelectedRun(freshRun);
          setRuns((current) => current.map((run) => (run.run_id === freshRun.run_id ? freshRun : run)));
          setJob(freshJob);
          setLogs(freshLogs);
          setEvents(freshEvents);
          if (!selectedStageKey) {
            setSelectedStageKey(defaultStageSelection(freshRun));
          }
          if (!["running", "submitted", "queued"].includes(freshJob.status)) {
            const [loadedMetrics, loadedUtterances, loadedCompare] = await Promise.all([
              fetchMetrics(freshRun.run_id),
              fetchUtterances(freshRun.run_id),
              fetchCompare(),
            ]);
            setMetrics(loadedMetrics.modes ?? []);
            setUtterances(loadedUtterances);
            setCompareRows(loadedCompare.rows ?? []);
            try {
              const [loadedModels, loadedReport] = await Promise.all([fetchModels(), fetchEvaluationReport()]);
              setModels(loadedModels.models ?? []);
              setEvaluationReport(loadedReport);
            } catch {
              setNotice("Report refresh failed");
            }
          }
        } catch {
          setNotice("Polling failed");
        }
      })();
    }, 2000);

    return () => window.clearInterval(interval);
  }, [apiState, selectedRun?.run_id, selectedRun?.status, job?.status]);

  async function handleRunAction(action: "cancel" | "retry" | "resume") {
    if (!selectedRun) return;
    setNotice("");
    try {
      const updatedJob = await postRunAction(selectedRun.run_id, action);
      setJob(updatedJob);
      await refresh();
      setNotice(`${action} ${selectedRun.run_id}`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `${action} failed`);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  // Keep the URL hash in sync so views are bookmarkable and the browser
  // back/forward buttons move between them.
  useEffect(() => {
    const desired = `#/${activeView}`;
    if (window.location.hash !== desired) {
      window.history.replaceState(null, "", desired);
    }
  }, [activeView]);

  useEffect(() => {
    const onHashChange = () => {
      const view = readViewFromHash();
      if (view) {
        setActiveView(view);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    const stages = selectedRun?.stages ?? [];
    if (stages.length === 0) {
      setSelectedStageKey("");
      return;
    }
    const hasSelectedStage = stages.some((stage, index) => stageKey(stage, index) === selectedStageKey);
    if (!hasSelectedStage) {
      setSelectedStageKey(defaultStageSelection(selectedRun));
    }
  }, [selectedRun?.run_id, selectedRun?.stages, selectedStageKey]);

  const nextAction = useMemo(
    () =>
      computeNextAction({
        run: selectedRun,
        job,
        metrics,
        runsCount: runs.length,
        apiState,
        best,
        goTo: setActiveView,
        onRetry: () => void handleRunAction("retry"),
      }),
    [selectedRun, job, metrics, runs.length, apiState, best],
  );

  function selectStageToLogs(stage: Stage, key: string) {
    setSelectedStageKey(key);
    setActiveView("logs");
  }

  return (
    <div className="appShell">
      <SideNav activeView={activeView} apiState={apiState} onViewChange={setActiveView} run={selectedRun} runsCount={runs.length} />
      <main className="dashboardMain">
        <header className="topbar">
          <div className="pageTitle">
            <span>{viewIcon(activeView, 19)}</span>
            <div>
              <h1>{viewLabels[activeView]}</h1>
              <p>{viewDescriptions[activeView]}</p>
            </div>
          </div>
          <div className="toolbar">
            <span className={apiState === "api" ? "apiBadge live" : "apiBadge"}>
              <Server size={15} />
              {apiState === "loading" ? "연결 중" : apiState === "api" ? "연결됨" : "데모"}
            </span>
            <button className="iconButton" onClick={() => void refresh()} title="새로고침">
              <RefreshCcw size={18} />
            </button>
          </div>
        </header>

        {activeView === "overview" ? (
          <div className="dashboardGrid overviewGrid">
            <NextActionCard action={nextAction} />
            <LifecycleTracker onSelectStage={selectStageToLogs} run={selectedRun} selectedStageKey={selectedStageKey} />
            <StatStrip best={best} job={job} metrics={metrics} phases={phases} run={selectedRun} utterances={utterances} />
            <div className="overviewSplit">
              <MetricBars metrics={metrics} />
              <RunHeader job={job} metrics={metrics} onAction={(action) => void handleRunAction(action)} run={selectedRun} selectedStage={selectedStage} />
            </div>
            <RunList onSelect={(run) => void selectRun(run)} runMetrics={runMetrics} runs={runs} selectedRunId={selectedRun?.run_id} title="최근 실험" />
          </div>
        ) : null}

        {activeView === "launch" ? (
          <div className="dashboardGrid launchGrid">
            <RunLauncher apiState={apiState} checkpoints={checkpoints} datasets={datasets} onCreate={handleCreateRun} speakerCheckpoints={speakerCheckpoints} />
            <div className="stack">
              <NextActionCard action={nextAction} />
              <RunList onSelect={(run) => void selectRun(run)} runMetrics={runMetrics} runs={runs} selectedRunId={selectedRun?.run_id} />
            </div>
          </div>
        ) : null}

        {activeView === "runs" ? (
          <div className="dashboardGrid runsGrid">
            <RunList onSelect={(run) => void selectRun(run)} runMetrics={runMetrics} runs={runs} selectedRunId={selectedRun?.run_id} />
            <div className="stack">
              <StatStrip best={best} job={job} metrics={metrics} phases={phases} run={selectedRun} utterances={utterances} />
              <LifecycleTracker onSelectStage={selectStageToLogs} run={selectedRun} selectedStageKey={selectedStageKey} />
              <RunHeader job={job} metrics={metrics} onAction={(action) => void handleRunAction(action)} run={selectedRun} selectedStage={selectedStage} />
            </div>
          </div>
        ) : null}

        {activeView === "models" ? (
          <div className="dashboardGrid resultsPageGrid">
            <ModelRegistryPage
              models={models}
              runs={runs}
              onSelectRun={(run) => {
                void selectRun(run);
                setActiveView("results");
              }}
            />
          </div>
        ) : null}

        {activeView === "evaluation" ? (
          <EvaluationCenter
            report={evaluationReport}
            runs={runs}
            onSelectRun={(run) => {
              void selectRun(run);
              setActiveView("results");
            }}
          />
        ) : null}

        {activeView === "results" ? (
          <div className="dashboardGrid resultsPageGrid">
            <StatStrip best={best} job={job} metrics={metrics} phases={phases} run={selectedRun} utterances={utterances} />
            <div className="resultsGrid">
              <MetricBars metrics={metrics} />
              <FailureTable metrics={metrics} />
              <MetricContractPanel metrics={metrics} />
            </div>
          </div>
        ) : null}

        {activeView === "compare" ? (
          <div className="dashboardGrid resultsPageGrid">
            <ComparePage
              rows={compareRows}
              runs={runs}
              onSelectRun={(run) => {
                void selectRun(run);
                setActiveView("results");
              }}
            />
          </div>
        ) : null}

        {activeView === "logs" ? (
          <div className="dashboardGrid logsGrid">
            <div className="stack">
              <RunHeader job={job} metrics={metrics} onAction={(action) => void handleRunAction(action)} run={selectedRun} selectedStage={selectedStage} />
              <PipelineGraph
                onSelectStage={(_stage, key) => setSelectedStageKey(key)}
                run={selectedRun}
                selectedStageKey={selectedStageKey}
              />
              <EventTimeline events={events} />
            </div>
            <JobLogs job={job} logs={logs} selectedStage={selectedStage} />
          </div>
        ) : null}

        {activeView === "utterances" ? (
          <div className="dashboardGrid utteranceGrid">
            <UtteranceInspector utterances={utterances} runId={selectedRun?.run_id} />
          </div>
        ) : null}

        {notice ? <div className="toast">{notice}</div> : null}
      </main>
    </div>
  );
}

export default App;
