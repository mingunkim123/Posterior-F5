export type Stage = {
  name: string;
  status: string;
  dry_run?: boolean;
  log?: string;
  command?: string[];
  outputs?: string[];
  error_message?: string;
  modes?: Array<{
    mode: string;
    status?: string;
    log?: string;
    metrics?: string;
    predictions?: string;
    commands?: string;
    output_dir?: string;
  }>;
};

export type Run = {
  run_id: string;
  project?: string;
  experiment?: string;
  dataset_id?: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  modes?: string[];
  stages?: Stage[];
  artifact_root?: string;
  error_message?: string;
  model?: {
    name?: string;
    checkpoint_id?: string | null;
    checkpoint?: string;
    checkpoint_path?: string;
    checkpoint_hash?: string;
    vocoder?: string;
  };
  generation?: Record<string, unknown>;
  posterior?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
};

export type MetricRow = {
  mode: string;
  status?: string;
  num_utterances?: number;
  wer?: number | null;
  cer?: number | null;
  wer_deletions?: number;
  wer_insertions?: number;
  wer_substitutions?: number;
};

export type MetricsSummary = {
  run_id?: string;
  modes: MetricRow[];
};

export type UtteranceMode = {
  generated_audio?: string | null;
  prediction_file?: string | null;
  prediction_text?: string | null;
  status?: string;
  failure_type?: string;
  diff?: Array<{ op: "equal" | "insert" | "delete"; text: string }>;
};

export type Utterance = {
  utterance_id: string;
  subset?: string;
  text?: string;
  reference_text?: string;
  gen_text?: string;
  ref_audio?: string;
  posterior_entropy?: number | null;
  one_best?: string;
  failure_type?: string;
  modes: Record<string, UtteranceMode>;
};

export type RunCreatePayload = {
  run_id?: string;
  project: string;
  experiment: string;
  dataset_id?: string;
  manifest: string;
  modes: string[];
  model: string;
  checkpoint_id?: string;
  checkpoint?: string;
  checkpoint_hash?: string;
  vocoder: string;
  seed: number;
  language: string;
  run_posterior_extraction: boolean;
  skip_whisper: boolean;
  run_inference: boolean;
  inference_dry_run: boolean;
  run_prediction: boolean;
  prediction_dry_run: boolean;
  run_metrics: boolean;
  metrics_dry_run: boolean;
  fail_if_exists: boolean;
};

export type RunCreateResponse = {
  status: string;
  run_id?: string;
  exit_code: number | null;
  command: string[];
  stdout: string;
  stderr: string;
  job?: RunJob;
  run?: Run;
};

export type RunJob = {
  run_id: string;
  status: string;
  pid?: number | null;
  command?: string[];
  started_at?: string;
  finished_at?: string;
  exit_code?: number | null;
  stdout_log?: string;
  stderr_log?: string;
  error_message?: string;
  run?: Run;
};

export type RunLogFile = {
  path: string;
  size_bytes: number;
  content: string;
};

export type RunLogs = {
  run_id: string;
  files: RunLogFile[];
};

export type Checkpoint = {
  id: string;
  model: string;
  path: string;
  vocoder: string;
  checkpoint_hash?: string;
  notes?: string;
  dataset?: string | null;
  git_commit?: string | null;
};

export type CheckpointRegistry = {
  checkpoints: Checkpoint[];
};

export type Dataset = {
  id: string;
  manifest: string;
  language: string;
  num_utterances?: number | null;
  purpose?: string;
  subsets?: string[];
  notes?: string;
};

export type DatasetRegistry = {
  datasets: Dataset[];
};

export type Experiment = {
  experiment_id: string;
  name: string;
  project?: string;
  num_runs: number;
  runs_with_metrics: number;
  latest_updated_at?: string;
  run_ids: string[];
};

export type CompareRow = {
  run_id: string;
  project?: string;
  experiment?: string;
  checkpoint?: string;
  seed?: number | string | null;
  subset?: string;
  mode?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  num_utterances?: number;
  wer?: number | null;
  cer?: number | null;
  substitutions?: number;
  deletions?: number;
  insertions?: number;
  wer_reference_length?: number;
  cer_reference_length?: number;
};

export type CompareResponse = {
  run_ids: string[];
  rows: CompareRow[];
};

export type RunEvent = {
  time: string;
  level: string;
  stage: string;
  mode?: string;
  message: string;
};

export type RunEvents = {
  run_id: string;
  events: RunEvent[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchRuns(): Promise<Run[]> {
  return getJson<Run[]>("/runs");
}

export async function fetchCheckpoints(): Promise<CheckpointRegistry> {
  return getJson<CheckpointRegistry>("/checkpoints");
}

export async function fetchDatasets(): Promise<DatasetRegistry> {
  return getJson<DatasetRegistry>("/datasets");
}

export async function fetchExperiments(): Promise<Experiment[]> {
  return getJson<Experiment[]>("/experiments");
}

export async function fetchCompare(runIds?: string[]): Promise<CompareResponse> {
  const query = runIds && runIds.length > 0 ? `?run_ids=${encodeURIComponent(runIds.join(","))}` : "";
  return getJson<CompareResponse>(`/compare${query}`);
}

export async function fetchRun(runId: string): Promise<Run> {
  return getJson<Run>(`/runs/${encodeURIComponent(runId)}`);
}

export async function createRun(payload: RunCreatePayload): Promise<RunCreateResponse> {
  return postJson<RunCreateResponse>("/runs", payload);
}

export async function fetchRunJob(runId: string): Promise<RunJob> {
  return getJson<RunJob>(`/runs/${encodeURIComponent(runId)}/job`);
}

export async function fetchRunLogs(runId: string): Promise<RunLogs> {
  return getJson<RunLogs>(`/runs/${encodeURIComponent(runId)}/logs`);
}

export async function fetchRunEvents(runId: string): Promise<RunEvents> {
  return getJson<RunEvents>(`/runs/${encodeURIComponent(runId)}/events`);
}

export async function postRunAction(runId: string, action: "cancel" | "retry" | "resume"): Promise<RunJob> {
  return postJson<RunJob>(`/runs/${encodeURIComponent(runId)}/${action}`, {});
}

export async function fetchMetrics(runId: string): Promise<MetricsSummary> {
  return getJson<MetricsSummary>(`/runs/${encodeURIComponent(runId)}/metrics`);
}

export async function fetchUtterances(runId: string): Promise<Utterance[]> {
  return getJson<Utterance[]>(`/runs/${encodeURIComponent(runId)}/utterances`);
}

export function artifactUrl(runId: string, artifactPath?: string | null): string | null {
  if (!artifactPath) {
    return null;
  }
  return `${API_BASE}/artifacts/${encodeURIComponent(runId)}/${artifactPath}`;
}

export function runExportUrl(runId: string, exportName: "main_table.csv" | "error_breakdown.csv" | "utterance_examples.jsonl"): string {
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/exports/${exportName}`;
}

export function experimentExportUrl(experimentId: string, exportName: "ablation_table.csv"): string {
  return `${API_BASE}/experiments/${encodeURIComponent(experimentId)}/exports/${exportName}`;
}
