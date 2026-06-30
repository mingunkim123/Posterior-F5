export type Stage = {
  name: string;
  status: string;
  dry_run?: boolean;
};

export type Run = {
  run_id: string;
  project?: string;
  experiment?: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  modes?: string[];
  stages?: Stage[];
  artifact_root?: string;
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
};

export type Utterance = {
  utterance_id: string;
  subset?: string;
  text?: string;
  gen_text?: string;
  ref_audio?: string;
  modes: Record<string, UtteranceMode>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchRuns(): Promise<Run[]> {
  return getJson<Run[]>("/runs");
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

export const demoRuns: Run[] = [
  {
    run_id: "run_demo_dev_small",
    project: "Posterior-F5",
    experiment: "baseline_dev_small",
    status: "completed",
    created_at: "2026-06-30T10:00:00+09:00",
    updated_at: "2026-06-30T10:08:00+09:00",
    modes: ["hard", "oracle", "length_only", "soft_ctc"],
    stages: [
      { name: "scaffold", status: "succeeded" },
      { name: "posterior_extraction", status: "succeeded" },
      { name: "inference", status: "planned", dry_run: true },
      { name: "prediction", status: "planned", dry_run: true },
      { name: "metrics", status: "succeeded" },
    ],
    artifact_root: "mlops_artifacts/runs/run_demo_dev_small",
  },
];

export const demoMetrics: MetricsSummary = {
  run_id: "run_demo_dev_small",
  modes: [
    { mode: "hard", status: "succeeded", num_utterances: 24, wer: 0.192, cer: 0.094, wer_deletions: 18, wer_insertions: 5, wer_substitutions: 11 },
    { mode: "oracle", status: "succeeded", num_utterances: 24, wer: 0.131, cer: 0.058, wer_deletions: 9, wer_insertions: 4, wer_substitutions: 9 },
    { mode: "length_only", status: "succeeded", num_utterances: 24, wer: 0.166, cer: 0.076, wer_deletions: 12, wer_insertions: 6, wer_substitutions: 10 },
    { mode: "soft_ctc", status: "succeeded", num_utterances: 24, wer: 0.148, cer: 0.069, wer_deletions: 10, wer_insertions: 5, wer_substitutions: 10 },
  ],
};

export const demoUtterances: Utterance[] = [
  {
    utterance_id: "utt-001",
    subset: "clean",
    text: "Target text.",
    gen_text: "Target text.",
    ref_audio: "data/ref_001.wav",
    modes: {
      hard: { generated_audio: null, prediction_file: "predictions/hard.jsonl" },
      oracle: { generated_audio: null, prediction_file: "predictions/oracle.jsonl" },
      length_only: { generated_audio: null, prediction_file: "predictions/length_only.jsonl" },
      soft_ctc: { generated_audio: null, prediction_file: "predictions/soft_ctc.jsonl" },
    },
  },
  {
    utterance_id: "utt-002",
    subset: "noisy",
    text: "Another target sentence.",
    gen_text: "Another target sentence.",
    ref_audio: "data/ref_002.wav",
    modes: {
      hard: { generated_audio: null, prediction_file: "predictions/hard.jsonl" },
      oracle: { generated_audio: null, prediction_file: "predictions/oracle.jsonl" },
      length_only: { generated_audio: null, prediction_file: "predictions/length_only.jsonl" },
      soft_ctc: { generated_audio: null, prediction_file: "predictions/soft_ctc.jsonl" },
    },
  },
];
