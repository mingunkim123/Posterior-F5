import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDot,
  Clipboard,
  Database,
  Download,
  FileAudio,
  GitBranch,
  ListFilter,
  Play,
  RefreshCcw,
  Server,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import {
  Checkpoint,
  CompareRow,
  Dataset,
  MetricRow,
  Run,
  RunCreatePayload,
  RunJob,
  RunEvents,
  RunLogs,
  Stage,
  Utterance,
  artifactUrl,
  createRun,
  experimentExportUrl,
  fetchCheckpoints,
  fetchCompare,
  fetchDatasets,
  fetchMetrics,
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
  run_posterior_extraction: false,
  skip_whisper: true,
  run_inference: false,
  inference_dry_run: true,
  run_prediction: false,
  prediction_dry_run: true,
  run_metrics: false,
  metrics_dry_run: false,
  fail_if_exists: false,
};

type RunPreset = {
  key: string;
  label: string;
  payload: Partial<RunCreatePayload>;
};

const runPresets: RunPreset[] = [
  {
    key: "scaffold_only",
    label: "scaffold_only",
    payload: {
      experiment: "scaffold_only_smoke",
      manifest: sampleManifest,
      modes: ["hard"],
      run_posterior_extraction: false,
      skip_whisper: true,
      run_inference: false,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "posterior_text_only",
    label: "posterior_text_only",
    payload: {
      experiment: "posterior_text_smoke",
      manifest: sampleManifest,
      modes: ["hard"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: false,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "dry_inference_plan",
    label: "dry_inference_plan",
    payload: {
      experiment: "dry_inference_plan",
      manifest: sampleManifest,
      modes: ["hard", "oracle", "length_only", "soft_ctc"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: true,
      inference_dry_run: true,
      run_prediction: false,
      prediction_dry_run: true,
      run_metrics: false,
      metrics_dry_run: false,
    },
  },
  {
    key: "metrics_dry_run",
    label: "metrics_dry_run",
    payload: {
      experiment: "metrics_dry_run",
      manifest: sampleManifest,
      modes: ["hard", "oracle", "length_only", "soft_ctc"],
      run_posterior_extraction: true,
      skip_whisper: true,
      run_inference: true,
      inference_dry_run: true,
      run_prediction: true,
      prediction_dry_run: true,
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

function statusIcon(status: string) {
  if (status === "succeeded" || status === "completed") {
    return <CheckCircle2 size={16} />;
  }
  if (status === "failed") {
    return <TriangleAlert size={16} />;
  }
  return <CircleDot size={16} />;
}

function statusLabel(status?: string, dryRun?: boolean): string {
  if (dryRun && status === "planned") return "계획만";
  if (status === "succeeded" || status === "completed") return "완료";
  if (status === "failed") return "실패";
  if (status === "running") return "실행 중";
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
    prediction: "ASR 결과",
    metrics: "WER and CER",
    job: "worker 상태",
    queued: "실행 대기",
  };
  return labels[name] ?? "stage";
}

type ViewKey = "overview" | "launch" | "runs" | "results" | "compare" | "logs" | "utterances";

const viewLabels: Record<ViewKey, string> = {
  overview: "개요",
  launch: "실험 시작",
  runs: "실험 기록",
  results: "결과 분석",
  compare: "비교 분석",
  logs: "실행 로그",
  utterances: "샘플 검토",
};

function viewIcon(view: ViewKey) {
  if (view === "overview") return <Activity size={17} />;
  if (view === "launch") return <Play size={17} />;
  if (view === "runs") return <ListFilter size={17} />;
  if (view === "results") return <BarChart3 size={17} />;
  if (view === "compare") return <Database size={17} />;
  if (view === "logs") return <Terminal size={17} />;
  return <FileAudio size={17} />;
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
  const views: ViewKey[] = ["overview", "launch", "runs", "results", "compare", "logs", "utterances"];
  return (
    <aside className="sideNav">
      <div className="sideBrand">
        <div className="brandMark"><GitBranch size={20} /></div>
        <div>
          <h1>Posterior-F5</h1>
          <span>Ops</span>
        </div>
      </div>
      <nav className="navMenu">
        {views.map((view) => (
          <button className={activeView === view ? "navItem active" : "navItem"} key={view} onClick={() => onViewChange(view)}>
            {viewIcon(view)}
            <span>{viewLabels[view]}</span>
          </button>
        ))}
      </nav>
      <div className="sideMeta">
        <small>선택된 실험</small>
        <strong>{run?.run_id ?? "none"}</strong>
        <em className={`pill status-${run?.status ?? "planned"}`}>{statusLabel(run?.status)}</em>
      </div>
      <div className="sideFooter">
        <span className={apiState === "api" ? "apiBadge live" : "apiBadge"}>
          <Server size={15} />
          {apiState === "loading" ? "loading" : apiState}
        </span>
        <span className="runCount">{runsCount} runs</span>
      </div>
    </aside>
  );
}

function RunSummary({ job, metrics, run, utterances }: { job?: RunJob; metrics: MetricRow[]; run?: Run; utterances: Utterance[] }) {
  return (
    <section className="summaryBand">
      <div>
        <small>현재 실험</small>
        <strong>{run?.run_id ?? "none"}</strong>
      </div>
      <div>
        <small>실험 상태</small>
        <strong>{statusLabel(run?.status)}</strong>
      </div>
      <div>
        <small>작업 상태</small>
        <strong>{statusLabel(job?.status)}</strong>
      </div>
      <div>
        <small>모드</small>
        <strong>{run?.modes?.length ?? metrics.length}</strong>
      </div>
      <div>
        <small>샘플</small>
        <strong>{utterances.length}</strong>
      </div>
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
          <small>Run</small>
          <h2>{run?.run_id ?? "No run selected"}</h2>
          <p>{run?.experiment ?? "experiment"} · {checkpointLabel(run)}</p>
        </div>
        <em className={`pill status-${run?.status ?? "planned"}`}>{statusLabel(run?.status)}</em>
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
          <strong>{run?.created_at ?? "n/a"}</strong>
        </div>
        <div>
          <small>Updated</small>
          <strong>{run?.updated_at ?? job?.finished_at ?? "n/a"}</strong>
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
          Cancel
        </button>
        <button className="artifactAction" disabled={!canRetry} onClick={() => onAction?.("retry")} type="button">
          Retry
        </button>
        <button className="artifactAction" disabled={!canRetry} onClick={() => onAction?.("resume")} type="button">
          Resume
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
          <h2>실험 진행 상태</h2>
          <p>{run?.run_id ?? "No run selected"}</p>
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
            <div className={`stageDot status-${stage.status}`}>{statusIcon(stage.status)}</div>
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
  return (
    <section className="panel metricPanel">
      <div className="panelHeader">
        <div>
          <h2>품질 지표</h2>
          <p>mode별 WER / CER</p>
        </div>
        <BarChart3 size={20} />
      </div>
      <div className="metricRows">
        {metrics.map((item) => (
          <div className="metricRow" key={item.mode}>
            <div className="metricName">
              <span className="modeSwatch" style={{ background: modeColors[item.mode] ?? "#607080" }} />
              <strong>{item.mode}</strong>
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
          </div>
        ))}
      </div>
    </section>
  );
}

function RunList({
  runs,
  selectedRunId,
  onSelect,
}: {
  runs: Run[];
  selectedRunId?: string;
  onSelect: (run: Run) => void;
}) {
  return (
    <section className="panel runList">
      <div className="panelHeader">
        <div>
          <h2>실험 기록</h2>
          <p>{runs.length} runs</p>
        </div>
        <ListFilter size={20} />
      </div>
      <div className="runRows">
        {runs.map((run) => (
          <button className={run.run_id === selectedRunId ? "runRow active" : "runRow"} key={run.run_id} onClick={() => onSelect(run)}>
            <span>
              <strong>{run.run_id}</strong>
              <small>{run.experiment ?? "experiment"}</small>
            </span>
            <em className={`pill status-${run.status}`}>{statusLabel(run.status)}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

function RunLauncher({
  apiState,
  checkpoints,
  datasets,
  onCreate,
}: {
  apiState: "api" | "demo" | "loading";
  checkpoints: Checkpoint[];
  datasets: Dataset[];
  onCreate: (payload: RunCreatePayload) => Promise<void>;
}) {
  const [payload, setPayload] = useState<RunCreatePayload>(defaultRunPayload);
  const [activePreset, setActivePreset] = useState(runPresets[0].key);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function setValue<K extends keyof RunCreatePayload>(key: K, value: RunCreatePayload[K]) {
    setActivePreset("");
    setPayload((current) => ({ ...current, [key]: value }));
  }

  function applyPreset(preset: RunPreset) {
    setActivePreset(preset.key);
    setPayload((current) => ({
      ...current,
      ...preset.payload,
      run_id: undefined,
      fail_if_exists: false,
    }));
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

  function selectDataset(datasetId: string) {
    setActivePreset("");
    const dataset = datasets.find((item) => item.id === datasetId);
    setPayload((current) => ({
      ...current,
      dataset_id: datasetId,
      manifest: dataset?.manifest ?? current.manifest,
      language: dataset?.language ?? current.language,
    }));
  }

  function toggleMode(mode: string) {
    setActivePreset("");
    setPayload((current) => {
      const modes = current.modes.includes(mode) ? current.modes.filter((item) => item !== mode) : [...current.modes, mode];
      return { ...current, modes };
    });
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    try {
      await onCreate({
        ...payload,
        run_id: payload.run_id?.trim() || undefined,
        manifest: payload.manifest.trim(),
        experiment: payload.experiment.trim(),
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  const disabled = isSubmitting || apiState !== "api";
  return (
    <section className="panel launcher">
      <div className="panelHeader">
        <div>
          <h2>실험 시작</h2>
          <p>{apiState === "demo" ? "API unavailable" : "local artifact run"}</p>
        </div>
        <Play size={20} />
      </div>
      <form className="runForm" onSubmit={(event) => void submit(event)}>
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
        <label>
          <span>실험 이름</span>
          <input value={payload.experiment} onChange={(event) => setValue("experiment", event.target.value)} required />
        </label>
        <label>
          <span>Dataset</span>
          <select value={payload.dataset_id ?? ""} onChange={(event) => selectDataset(event.target.value)}>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.id} · {dataset.purpose || dataset.manifest}
              </option>
            ))}
          </select>
        </label>
        <div className="datasetMeta">
          <span>{datasets.find((item) => item.id === payload.dataset_id)?.num_utterances ?? "n/a"} utt</span>
          <span>{datasets.find((item) => item.id === payload.dataset_id)?.subsets?.join(", ") || "all subsets"}</span>
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
        <div className="modePicker" aria-label="Modes">
          {availableModes.map((mode) => (
            <button className={payload.modes.includes(mode) ? "modeChoice active" : "modeChoice"} key={mode} type="button" onClick={() => toggleMode(mode)}>
              <span className="modeSwatch" style={{ background: modeColors[mode] ?? "#607080" }} />
              {mode}
            </button>
          ))}
        </div>
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
            <input checked={payload.run_prediction} type="checkbox" onChange={(event) => setValue("run_prediction", event.target.checked)} />
            <span>ASR 변환</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_metrics} type="checkbox" onChange={(event) => setValue("run_metrics", event.target.checked)} />
            <span>점수 계산</span>
          </label>
        </div>
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
            <input checked={payload.fail_if_exists} type="checkbox" onChange={(event) => setValue("fail_if_exists", event.target.checked)} />
            <span>새 ID 강제</span>
          </label>
        </div>
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
            <span>{isSubmitting ? "Starting" : "실험 시작"}</span>
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
                    <em>{modeData?.status ?? "n/a"}</em>
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
  const [query, setQuery] = useState("");

  const experiments = uniqueValues(rows, "experiment");
  const checkpoints = uniqueValues(rows, "checkpoint");
  const modes = uniqueValues(rows, "mode");
  const subsets = uniqueValues(rows, "subset");
  const seeds = uniqueValues(rows, "seed");

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return rows
      .filter((row) => experiment === "all" || row.experiment === experiment)
      .filter((row) => checkpoint === "all" || row.checkpoint === checkpoint)
      .filter((row) => mode === "all" || row.mode === mode)
      .filter((row) => subset === "all" || row.subset === subset)
      .filter((row) => seed === "all" || String(row.seed ?? "") === seed)
      .filter((row) => {
        if (!normalizedQuery) return true;
        return [row.run_id, row.experiment, row.checkpoint, row.mode].some((value) => String(value ?? "").toLowerCase().includes(normalizedQuery));
      })
      .sort((a, b) => (a.wer ?? Number.POSITIVE_INFINITY) - (b.wer ?? Number.POSITIVE_INFINITY));
  }, [checkpoint, experiment, mode, query, rows, seed, subset]);

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
      </div>
      <div className="tableScroller">
        <table className="compareTable">
          <thead>
            <tr>
              <th>Run</th>
              <th>Experiment</th>
              <th>Checkpoint</th>
              <th>Seed</th>
              <th>Subset</th>
              <th>Mode</th>
              <th>WER</th>
              <th>CER</th>
              <th>Sub</th>
              <th>Del</th>
              <th>Ins</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={11}>No metric rows</td>
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
                  <td>{row.seed ?? "n/a"}</td>
                  <td>{row.subset ?? "all"}</td>
                  <td>
                    <span className="modeCell">
                      <span className="modeSwatch" style={{ background: modeColors[row.mode ?? ""] ?? "#607080" }} />
                      {row.mode ?? "mode"}
                    </span>
                  </td>
                  <td>{pct(row.wer)}</td>
                  <td>{pct(row.cer)}</td>
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
          <p>{selectedStage ? `${stageLabel(selectedStage.name)} · ${statusLabel(selectedStage.status, selectedStage.dry_run)}` : job ? `${statusLabel(job.status)}${job.pid ? ` pid ${job.pid}` : ""}` : "No job selected"}</p>
        </div>
        <Terminal size={20} />
      </div>
      {commandText ? (
        <div className="commandBox">
          <div>
            <strong>Command</strong>
            <button className="copyButton" onClick={() => void copyCommand()} type="button">
              <Clipboard size={15} />
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
          </div>
          <pre>{commandText}</pre>
        </div>
      ) : null}
      <div className="jobMeta">
        <span>{job?.started_at ?? "not started"}</span>
        <span>{job?.finished_at ?? "waiting"}</span>
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
              <span>{event.time}</span>
              <strong>{event.stage}{event.mode ? ` · ${event.mode}` : ""}</strong>
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
  const [datasets, setDatasets] = useState<Dataset[]>(fallbackDatasets);
  const [job, setJob] = useState<RunJob | undefined>();
  const [logs, setLogs] = useState<RunLogs | undefined>();
  const [events, setEvents] = useState<RunEvents | undefined>();
  const [apiState, setApiState] = useState<"api" | "demo" | "loading">("loading");
  const [activeView, setActiveView] = useState<ViewKey>("overview");
  const [selectedStageKey, setSelectedStageKey] = useState("");
  const [notice, setNotice] = useState("");

  const selectedStage = useMemo(() => {
    const stages = selectedRun?.stages ?? [];
    return stages.find((stage, index) => stageKey(stage, index) === selectedStageKey);
  }, [selectedRun?.stages, selectedStageKey]);

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
      const [loadedRuns, loadedCheckpoints, loadedDatasets, loadedCompare] = await Promise.all([
        fetchRuns(),
        fetchCheckpoints(),
        fetchDatasets(),
        fetchCompare(),
      ]);
      setCheckpoints(loadedCheckpoints.checkpoints.length > 0 ? loadedCheckpoints.checkpoints : fallbackCheckpoints);
      setDatasets(loadedDatasets.datasets.length > 0 ? loadedDatasets.datasets : fallbackDatasets);
      setCompareRows(loadedCompare.rows ?? []);
      if (loadedRuns.length === 0) {
        setRuns([]);
        setSelectedRun(undefined);
        setMetrics([]);
        setUtterances([]);
        setCompareRows([]);
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
      setDatasets(fallbackDatasets);
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

  const pageMeta = selectedRun?.experiment ?? selectedRun?.artifact_root ?? "mlops_artifacts/runs";

  return (
    <div className="appShell">
      <SideNav activeView={activeView} apiState={apiState} onViewChange={setActiveView} run={selectedRun} runsCount={runs.length} />
      <main className="dashboardMain">
        <header className="topbar">
          <div className="pageTitle">
            <span>{viewIcon(activeView)}</span>
            <div>
              <h1>{viewLabels[activeView]}</h1>
              <p>{pageMeta}</p>
            </div>
          </div>
          <div className="toolbar">
            <span className={apiState === "api" ? "apiBadge live" : "apiBadge"}>
              <Server size={15} />
              {apiState === "loading" ? "loading" : apiState}
            </span>
            <button className="iconButton" onClick={() => void refresh()} title="Refresh runs">
              <RefreshCcw size={18} />
            </button>
          </div>
        </header>

        {activeView === "overview" ? (
          <div className="dashboardGrid overviewGrid">
            <RunHeader job={job} metrics={metrics} onAction={(action) => void handleRunAction(action)} run={selectedRun} selectedStage={selectedStage} />
            <RunSummary job={job} metrics={metrics} run={selectedRun} utterances={utterances} />
            <PipelineGraph
              onSelectStage={(stage, key) => {
                setSelectedStageKey(key);
                setActiveView("logs");
              }}
              run={selectedRun}
              selectedStageKey={selectedStageKey}
            />
            <MetricBars metrics={metrics} />
            <JobLogs job={job} logs={logs} selectedStage={selectedStage} />
          </div>
        ) : null}

        {activeView === "launch" ? (
          <div className="dashboardGrid launchGrid">
            <RunLauncher apiState={apiState} checkpoints={checkpoints} datasets={datasets} onCreate={handleCreateRun} />
            <RunList runs={runs} selectedRunId={selectedRun?.run_id} onSelect={(run) => void selectRun(run)} />
          </div>
        ) : null}

        {activeView === "runs" ? (
          <div className="dashboardGrid runsGrid">
            <RunList runs={runs} selectedRunId={selectedRun?.run_id} onSelect={(run) => void selectRun(run)} />
            <div className="stack">
              <RunHeader job={job} metrics={metrics} onAction={(action) => void handleRunAction(action)} run={selectedRun} selectedStage={selectedStage} />
              <RunSummary job={job} metrics={metrics} run={selectedRun} utterances={utterances} />
              <PipelineGraph
                onSelectStage={(_stage, key) => setSelectedStageKey(key)}
                run={selectedRun}
                selectedStageKey={selectedStageKey}
              />
            </div>
          </div>
        ) : null}

        {activeView === "results" ? (
          <div className="dashboardGrid resultsPageGrid">
            <RunSummary job={job} metrics={metrics} run={selectedRun} utterances={utterances} />
            <div className="resultsGrid">
              <MetricBars metrics={metrics} />
              <FailureTable metrics={metrics} />
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
