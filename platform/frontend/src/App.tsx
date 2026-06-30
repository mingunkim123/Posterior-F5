import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDot,
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
  MetricRow,
  Run,
  RunCreatePayload,
  RunJob,
  RunLogs,
  Utterance,
  artifactUrl,
  createRun,
  demoMetrics,
  demoRuns,
  demoUtterances,
  fetchMetrics,
  fetchRun,
  fetchRunJob,
  fetchRunLogs,
  fetchRuns,
  fetchUtterances,
} from "./api";

const modeColors: Record<string, string> = {
  hard: "#3d6b99",
  oracle: "#2f8f70",
  length_only: "#b07d2f",
  soft_ctc: "#8d5fbf",
  hybrid: "#b8526b",
  posterior_encoder: "#587447",
};

const availableModes = ["hard", "oracle", "length_only", "soft_ctc", "posterior_encoder", "hybrid"];

const defaultRunPayload: RunCreatePayload = {
  project: "Posterior-F5",
  experiment: "baseline_dev_small",
  manifest: "data/dev_manifest.jsonl",
  modes: ["hard", "oracle", "length_only", "soft_ctc"],
  model: "F5TTS_v1_Base",
  checkpoint: "",
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

function stageLabel(name: string): string {
  return name.replace(/_/g, " ");
}

function PipelineGraph({ run }: { run?: Run }) {
  const stages = run?.stages ?? [];
  return (
    <section className="panel pipeline">
      <div className="panelHeader">
        <div>
          <h2>Pipeline</h2>
          <p>{run?.run_id ?? "No run selected"}</p>
        </div>
        <Activity size={20} />
      </div>
      <div className="stageRail">
        {stages.map((stage, index) => (
          <div className="stageNode" key={`${stage.name}-${index}`}>
            <div className={`stageDot status-${stage.status}`}>{statusIcon(stage.status)}</div>
            <div>
              <strong>{stageLabel(stage.name)}</strong>
              <span>{stage.dry_run ? `${stage.status} dry` : stage.status}</span>
            </div>
          </div>
        ))}
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
          <h2>Metrics</h2>
          <p>WER and CER by mode</p>
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
          <h2>Runs</h2>
          <p>{runs.length} tracked</p>
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
            <em className={`pill status-${run.status}`}>{run.status}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

function RunLauncher({
  apiState,
  onCreate,
}: {
  apiState: "api" | "demo" | "loading";
  onCreate: (payload: RunCreatePayload) => Promise<void>;
}) {
  const [payload, setPayload] = useState<RunCreatePayload>(defaultRunPayload);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function setValue<K extends keyof RunCreatePayload>(key: K, value: RunCreatePayload[K]) {
    setPayload((current) => ({ ...current, [key]: value }));
  }

  function toggleMode(mode: string) {
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
          <h2>New Run</h2>
          <p>{apiState === "demo" ? "API unavailable" : "Create local artifact run"}</p>
        </div>
        <Play size={20} />
      </div>
      <form className="runForm" onSubmit={(event) => void submit(event)}>
        <label>
          <span>Experiment</span>
          <input value={payload.experiment} onChange={(event) => setValue("experiment", event.target.value)} required />
        </label>
        <label>
          <span>Manifest</span>
          <input value={payload.manifest} onChange={(event) => setValue("manifest", event.target.value)} required />
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
            <span>Posterior</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_inference} type="checkbox" onChange={(event) => setValue("run_inference", event.target.checked)} />
            <span>Inference</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_prediction} type="checkbox" onChange={(event) => setValue("run_prediction", event.target.checked)} />
            <span>Prediction</span>
          </label>
          <label className="checkRow">
            <input checked={payload.run_metrics} type="checkbox" onChange={(event) => setValue("run_metrics", event.target.checked)} />
            <span>Metrics</span>
          </label>
        </div>
        <div className="toggleGrid dryToggles">
          <label className="checkRow">
            <input checked={payload.skip_whisper} type="checkbox" onChange={(event) => setValue("skip_whisper", event.target.checked)} />
            <span>Skip Whisper</span>
          </label>
          <label className="checkRow">
            <input checked={payload.inference_dry_run} type="checkbox" onChange={(event) => setValue("inference_dry_run", event.target.checked)} />
            <span>Inference dry</span>
          </label>
          <label className="checkRow">
            <input checked={payload.prediction_dry_run} type="checkbox" onChange={(event) => setValue("prediction_dry_run", event.target.checked)} />
            <span>Prediction dry</span>
          </label>
          <label className="checkRow">
            <input checked={payload.fail_if_exists} type="checkbox" onChange={(event) => setValue("fail_if_exists", event.target.checked)} />
            <span>Unique run</span>
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
            <span>{isSubmitting ? "Starting" : "Start"}</span>
          </button>
        </div>
      </form>
    </section>
  );
}

function UtteranceInspector({ utterances, runId }: { utterances: Utterance[]; runId?: string }) {
  const [selected, setSelected] = useState(0);
  const utterance = utterances[selected];
  const modes = Object.keys(utterance?.modes ?? {});
  return (
    <section className="panel inspector">
      <div className="panelHeader">
        <div>
          <h2>Utterances</h2>
          <p>{utterances.length} samples</p>
        </div>
        <FileAudio size={20} />
      </div>
      <div className="inspectorGrid">
        <div className="utteranceList">
          {utterances.map((item, index) => (
            <button className={index === selected ? "utteranceButton active" : "utteranceButton"} key={item.utterance_id} onClick={() => setSelected(index)}>
              <strong>{item.utterance_id}</strong>
              <span>{item.subset ?? "subset"}</span>
            </button>
          ))}
        </div>
        <div className="utteranceDetail">
          <div className="textBlock">
            <small>Target</small>
            <p>{utterance?.text ?? utterance?.gen_text ?? "No text"}</p>
          </div>
          <div className="audioGrid">
            {modes.map((mode) => {
              const src = runId ? artifactUrl(runId, utterance?.modes[mode]?.generated_audio) : null;
              return (
                <div className="audioItem" key={mode}>
                  <div>
                    <span className="modeSwatch" style={{ background: modeColors[mode] ?? "#607080" }} />
                    <strong>{mode}</strong>
                  </div>
                  {src ? <audio controls src={src} /> : <span className="audioPlaceholder">no wav</span>}
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
          <h2>Breakdown</h2>
          <p>Word error counts</p>
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

function JobLogs({ job, logs }: { job?: RunJob; logs?: RunLogs }) {
  const files = logs?.files ?? [];
  return (
    <section className="panel logsPanel">
      <div className="panelHeader">
        <div>
          <h2>Job Logs</h2>
          <p>{job ? `${job.status}${job.pid ? ` pid ${job.pid}` : ""}` : "No job selected"}</p>
        </div>
        <Terminal size={20} />
      </div>
      <div className="jobMeta">
        <span>{job?.started_at ?? "not started"}</span>
        <span>{job?.finished_at ?? "waiting"}</span>
        <span>{job?.exit_code ?? "exit n/a"}</span>
      </div>
      <div className="logFiles">
        {files.length === 0 ? (
          <pre>No logs yet.</pre>
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

function App() {
  const [runs, setRuns] = useState<Run[]>(demoRuns);
  const [selectedRun, setSelectedRun] = useState<Run | undefined>(demoRuns[0]);
  const [metrics, setMetrics] = useState<MetricRow[]>(demoMetrics.modes);
  const [utterances, setUtterances] = useState<Utterance[]>(demoUtterances);
  const [job, setJob] = useState<RunJob | undefined>();
  const [logs, setLogs] = useState<RunLogs | undefined>();
  const [apiState, setApiState] = useState<"api" | "demo" | "loading">("loading");
  const [notice, setNotice] = useState("");

  async function loadRunDetails(run: Run) {
    const [loadedMetrics, loadedUtterances, loadedJob, loadedLogs] = await Promise.all([
      fetchMetrics(run.run_id),
      fetchUtterances(run.run_id),
      fetchRunJob(run.run_id),
      fetchRunLogs(run.run_id),
    ]);
    setMetrics(loadedMetrics.modes ?? []);
    setUtterances(loadedUtterances);
    setJob(loadedJob);
    setLogs(loadedLogs);
  }

  async function refresh() {
    setApiState("loading");
    try {
      const loadedRuns = await fetchRuns();
      if (loadedRuns.length === 0) {
        setRuns([]);
        setSelectedRun(undefined);
        setMetrics([]);
        setUtterances([]);
        setJob(undefined);
        setLogs(undefined);
        setApiState("api");
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
      setJob(undefined);
      setLogs(undefined);
      setApiState("demo");
    }
  }

  async function selectRun(run: Run) {
    setSelectedRun(run);
    try {
      await loadRunDetails(run);
      setApiState("api");
    } catch {
      setMetrics(demoMetrics.modes);
      setUtterances(demoUtterances);
      setJob(undefined);
      setLogs(undefined);
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
          const [freshRun, freshJob, freshLogs] = await Promise.all([
            fetchRun(selectedRun.run_id),
            fetchRunJob(selectedRun.run_id),
            fetchRunLogs(selectedRun.run_id),
          ]);
          setSelectedRun(freshRun);
          setRuns((current) => current.map((run) => (run.run_id === freshRun.run_id ? freshRun : run)));
          setJob(freshJob);
          setLogs(freshLogs);
          if (!["running", "submitted", "queued"].includes(freshJob.status)) {
            const [loadedMetrics, loadedUtterances] = await Promise.all([fetchMetrics(freshRun.run_id), fetchUtterances(freshRun.run_id)]);
            setMetrics(loadedMetrics.modes ?? []);
            setUtterances(loadedUtterances);
          }
        } catch {
          setNotice("Polling failed");
        }
      })();
    }, 2000);

    return () => window.clearInterval(interval);
  }, [apiState, selectedRun?.run_id, selectedRun?.status, job?.status]);

  useEffect(() => {
    void refresh();
  }, []);

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <div className="brandMark"><GitBranch size={20} /></div>
          <div>
            <h1>Posterior-F5 Ops</h1>
            <p>{selectedRun?.artifact_root ?? "mlops_artifacts/runs"}</p>
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
          <button className="iconButton" title="Open artifacts">
            <Download size={18} />
          </button>
        </div>
      </header>

      <section className="summaryBand">
        <div>
          <small>Selected run</small>
          <strong>{selectedRun?.run_id}</strong>
        </div>
        <div>
          <small>Status</small>
          <strong>{selectedRun?.status}</strong>
        </div>
        <div>
          <small>Modes</small>
          <strong>{selectedRun?.modes?.length ?? metrics.length}</strong>
        </div>
        <div>
          <small>Utterances</small>
          <strong>{utterances.length}</strong>
        </div>
      </section>

      <div className="layout">
        <RunList runs={runs} selectedRunId={selectedRun?.run_id} onSelect={(run) => void selectRun(run)} />
        <PipelineGraph run={selectedRun} />
        <MetricBars metrics={metrics} />
        <RunLauncher apiState={apiState} onCreate={handleCreateRun} />
        <FailureTable metrics={metrics} />
        <JobLogs job={job} logs={logs} />
        <UtteranceInspector utterances={utterances} runId={selectedRun?.run_id} />
      </div>

      {notice ? <div className="toast">{notice}</div> : null}
    </main>
  );
}

export default App;
