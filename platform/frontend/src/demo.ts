import type {
  Checkpoint,
  CompareRow,
  Dataset,
  MetricsSummary,
  Run,
  Utterance,
} from "./api";

const sampleManifest = "platform/samples/manifests/dev_smoke.jsonl";

export const fallbackCheckpoints: Checkpoint[] = [
  {
    id: "f5tts_v1_base_hf",
    model: "F5TTS_v1_Base",
    path: "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    vocoder: "vocos",
    checkpoint_hash: "hf:SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    notes: "upstream base checkpoint",
  },
];

export const fallbackDatasets: Dataset[] = [
  {
    id: "dev_smoke",
    manifest: sampleManifest,
    language: "en",
    num_utterances: 1,
    purpose: "smoke test",
    subsets: ["smoke"],
  },
];

export const demoRuns: Run[] = [
  {
    run_id: "run_demo_dev_small",
    project: "Posterior-F5",
    experiment: "baseline_dev_small",
    dataset_id: "dev_smoke",
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

export const demoCompareRows: CompareRow[] = demoMetrics.modes.map((metric) => ({
  run_id: "run_demo_dev_small",
  project: "Posterior-F5",
  experiment: "baseline_dev_small",
  checkpoint: "f5tts_v1_base_hf",
  mode: metric.mode,
  status: metric.status,
  num_utterances: metric.num_utterances,
  wer: metric.wer,
  cer: metric.cer,
  substitutions: metric.wer_substitutions,
  deletions: metric.wer_deletions,
  insertions: metric.wer_insertions,
}));

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
