# Platform Samples

Small manifests in this directory are used by the dashboard presets for local
smoke tests. They are intentionally tiny so API, worker, artifact, and dashboard
flows can be checked without preparing an external dataset.

## Manifests

- `manifests/dev_smoke.jsonl`: one English reference item for scaffold,
  posterior-text, dry-inference, and metrics smoke runs.

## Dashboard Presets

- `scaffold_only`: creates the run artifact contract and logs.
- `posterior_text_only`: builds posterior artifacts from manifest text without
  loading Whisper.
- `dry_inference_plan`: writes mode-specific inference command manifests without
  generating wav files.
- `metrics_dry_run`: writes placeholder predictions and computes summary metrics
  for dashboard chart checks.
