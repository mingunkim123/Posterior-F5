import { useEffect, useMemo, useState } from "react";
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
  TriangleAlert,
} from "lucide-react";
import {
  MetricRow,
  Run,
  Utterance,
  artifactUrl,
  demoMetrics,
  demoRuns,
  demoUtterances,
  fetchMetrics,
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

function App() {
  const [runs, setRuns] = useState<Run[]>(demoRuns);
  const [selectedRun, setSelectedRun] = useState<Run | undefined>(demoRuns[0]);
  const [metrics, setMetrics] = useState<MetricRow[]>(demoMetrics.modes);
  const [utterances, setUtterances] = useState<Utterance[]>(demoUtterances);
  const [apiState, setApiState] = useState<"api" | "demo" | "loading">("loading");

  async function refresh() {
    setApiState("loading");
    try {
      const loadedRuns = await fetchRuns();
      if (loadedRuns.length === 0) {
        throw new Error("No runs");
      }
      const current = loadedRuns[0];
      const [loadedMetrics, loadedUtterances] = await Promise.all([fetchMetrics(current.run_id), fetchUtterances(current.run_id)]);
      setRuns(loadedRuns);
      setSelectedRun(current);
      setMetrics(loadedMetrics.modes ?? []);
      setUtterances(loadedUtterances);
      setApiState("api");
    } catch {
      setRuns(demoRuns);
      setSelectedRun(demoRuns[0]);
      setMetrics(demoMetrics.modes);
      setUtterances(demoUtterances);
      setApiState("demo");
    }
  }

  async function selectRun(run: Run) {
    setSelectedRun(run);
    try {
      const [loadedMetrics, loadedUtterances] = await Promise.all([fetchMetrics(run.run_id), fetchUtterances(run.run_id)]);
      setMetrics(loadedMetrics.modes ?? []);
      setUtterances(loadedUtterances);
      setApiState("api");
    } catch {
      setMetrics(demoMetrics.modes);
      setUtterances(demoUtterances);
      setApiState("demo");
    }
  }

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
        <FailureTable metrics={metrics} />
        <UtteranceInspector utterances={utterances} runId={selectedRun?.run_id} />
      </div>

      <button className="floatingAction" title="Start planned run">
        <Play size={18} />
        <span>Run</span>
      </button>
    </main>
  );
}

export default App;
