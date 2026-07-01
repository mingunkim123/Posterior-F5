import type {
  Checkpoint,
  CompareRow,
  Dataset,
  EvaluationReport,
  MetricsSummary,
  ModelRegistry,
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
  {
    id: "dev_small_20",
    manifest: "manifests/dev_small_20.jsonl",
    language: "en",
    num_utterances: 20,
    purpose: "Local development stability run",
    subsets: ["clean/dev_small"],
  },
  {
    id: "hard_50",
    manifest: "manifests/hard_50.jsonl",
    language: "en",
    num_utterances: 50,
    purpose: "Hard sentence stress test",
    subsets: ["hard/test"],
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
    { mode: "hard", status: "succeeded", num_utterances: 24, wer: 0.192, cer: 0.094, normalizer: "paper", prediction_coverage: 1, wer_deletion_rate: 0.102, wer_deletions: 18, wer_insertions: 5, wer_substitutions: 11, generation_elapsed_sec_mean: 1.82, rtf_mean: 0.46, speaker_similarity_mean: 0.71, spk_sim_mean: 0.71, utmos_mean: 3.42 },
    { mode: "oracle", status: "succeeded", num_utterances: 24, wer: 0.131, cer: 0.058, normalizer: "paper", prediction_coverage: 1, wer_deletion_rate: 0.061, wer_deletions: 9, wer_insertions: 4, wer_substitutions: 9, generation_elapsed_sec_mean: 1.77, rtf_mean: 0.44, speaker_similarity_mean: 0.78, spk_sim_mean: 0.78, utmos_mean: 3.72 },
    { mode: "length_only", status: "succeeded", num_utterances: 24, wer: 0.166, cer: 0.076, normalizer: "paper", prediction_coverage: 1, wer_deletion_rate: 0.082, wer_deletions: 12, wer_insertions: 6, wer_substitutions: 10, generation_elapsed_sec_mean: 1.69, rtf_mean: 0.43, speaker_similarity_mean: 0.73, spk_sim_mean: 0.73, utmos_mean: 3.53 },
    { mode: "soft_ctc", status: "succeeded", num_utterances: 24, wer: 0.148, cer: 0.069, normalizer: "paper", prediction_coverage: 1, wer_deletion_rate: 0.071, wer_deletions: 10, wer_insertions: 5, wer_substitutions: 10, generation_elapsed_sec_mean: 1.73, rtf_mean: 0.45, speaker_similarity_mean: 0.76, spk_sim_mean: 0.76, utmos_mean: 3.66, significance: { wer: { baseline: "hard", candidate: "soft_ctc", metric: "wer", mean_diff: -0.044, ci_low: -0.071, ci_high: -0.012, p_value: 0.018, holm_p_value: 0.036, fdr_p_value: 0.024, significant: true } } },
  ],
};

export const demoCompareRows: CompareRow[] = demoMetrics.modes.map((metric) => ({
  run_id: "run_demo_dev_small",
  project: "Posterior-F5",
  experiment: "baseline_dev_small",
  checkpoint: "f5tts_v1_base_hf",
  row_type: "seed",
  seed: 1234,
  seed_count: 1,
  mode: metric.mode,
  status: metric.status,
  num_utterances: metric.num_utterances,
  wer: metric.wer,
  cer: metric.cer,
  normalizer: metric.normalizer,
  prediction_coverage: metric.prediction_coverage,
  wer_deletion_rate: metric.wer_deletion_rate,
  generation_elapsed_sec_mean: metric.generation_elapsed_sec_mean,
  rtf_mean: metric.rtf_mean,
  speaker_similarity_mean: metric.speaker_similarity_mean,
  spk_sim_mean: metric.spk_sim_mean,
  utmos_mean: metric.utmos_mean,
  substitutions: metric.wer_substitutions,
  deletions: metric.wer_deletions,
  insertions: metric.wer_insertions,
  wer_diff_mean: metric.significance?.wer?.mean_diff,
  wer_ci_low: metric.significance?.wer?.ci_low,
  wer_ci_high: metric.significance?.wer?.ci_high,
  wer_p_value: metric.significance?.wer?.p_value,
  wer_holm_p_value: metric.significance?.wer?.holm_p_value,
  wer_fdr_p_value: metric.significance?.wer?.fdr_p_value,
  wer_significant: metric.significance?.wer?.significant,
  significance_baseline: metric.significance?.wer?.baseline,
}));

export const fallbackModelRegistry: ModelRegistry = {
  generated_at: "2026-06-30T10:08:00+09:00",
  models: [
    {
      ...fallbackCheckpoints[0],
      stage: "baseline",
      num_runs: 1,
      num_evaluations: 4,
      best_run_id: "run_demo_dev_small",
      best_mode: "oracle",
      best_wer: 0.131,
      best_cer: 0.058,
      latest_run_id: "run_demo_dev_small",
      latest_status: "completed",
      latest_updated_at: "2026-06-30T10:08:00+09:00",
    },
  ],
};

export const fallbackEvaluationReport: EvaluationReport = {
  generated_at: "2026-06-30T10:08:00+09:00",
  total_runs: 1,
  evaluated_runs: 1,
  evaluated_rows: demoCompareRows.length,
  best: demoCompareRows[1],
  quality_gates: [
    { name: "best_wer", value: 0.131, target: 0.05, direction: "lower", unit: "rate", status: "fail" },
    { name: "best_cer", value: 0.058, target: 0.03, direction: "lower", unit: "rate", status: "warn" },
    { name: "prediction_coverage", value: 1, target: 0.99, direction: "higher", unit: "rate", status: "pass" },
    { name: "deletion_rate", value: 0.102, target: 0.15, direction: "lower", unit: "rate", status: "pass" },
    { name: "speaker_similarity", value: 0.78, target: 0.75, direction: "higher", unit: "score", status: "pass" },
    { name: "utmos", value: 3.72, target: 3.5, direction: "higher", unit: "score", status: "pass" },
    { name: "rtf", value: 0.43, target: 1, direction: "lower", unit: "ratio", status: "pass" },
  ],
  failure_mix: { substitutions: 40, deletions: 49, insertions: 20 },
  failure_rates: { substitutions: 0.367, deletions: 0.45, insertions: 0.183 },
  leaderboard: [...demoCompareRows].sort((a, b) => (a.wer ?? 1) - (b.wer ?? 1)),
  experiments: [
    {
      experiment_id: "baseline_dev_small",
      name: "baseline_dev_small",
      project: "Posterior-F5",
      num_runs: 1,
      runs_with_metrics: 1,
      latest_updated_at: "2026-06-30T10:08:00+09:00",
      run_ids: ["run_demo_dev_small"],
    },
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
