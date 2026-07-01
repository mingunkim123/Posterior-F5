"""FastAPI control plane for local Posterior-F5 MLOps runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from run_store import (
    cancel_run,
    compare_run_rows,
    ensure_experiment_ablation_export,
    ensure_run_paper_exports,
    list_experiment_runs,
    list_experiments,
    list_runs,
    load_checkpoint_registry,
    load_dataset_registry,
    load_evaluation_report,
    load_model_registry,
    load_run,
    load_run_events,
    load_run_job,
    load_run_logs,
    load_run_metrics,
    load_run_utterances,
    resume_run,
    retry_run,
    start_run,
)


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    project: str = "Posterior-F5"
    experiment: str = "baseline_dev_small"
    dataset_id: str = "dev_smoke"
    manifest: str = "platform/samples/manifests/dev_smoke.jsonl"
    modes: list[str] = Field(default_factory=lambda: ["hard", "oracle", "length_only", "soft_ctc"])
    model: str = "F5TTS_v1_Base"
    checkpoint_id: str = "f5tts_v1_base_hf"
    checkpoint: str = ""
    checkpoint_hash: str = ""
    vocoder: str = "vocos"
    seed: int = 1234
    language: str = "en"
    hard_ref_text_source: str = "manifest"
    run_posterior_extraction: bool = True
    skip_whisper: bool = True
    run_inference: bool = False
    inference_dry_run: bool = True
    run_prediction: bool = False
    prediction_dry_run: bool = True
    run_audio_metrics: bool = False
    audio_metrics_dry_run: bool = True
    run_speaker_similarity: bool = False
    speaker_checkpoint: str = ""
    speaker_device: str | None = None
    speaker_feat_type: str = "wavlm_large"
    run_utmos: bool = False
    utmos_device: str | None = None
    run_metrics: bool = False
    metrics_dry_run: bool = False
    metrics_normalizer: str = "paper"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 1234
    significance_baseline: str = "hard"
    fail_if_exists: bool = False


app = FastAPI(title="Posterior-F5 MLOps API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_run_or_job(run_id: str) -> None:
    """Raise 404 if neither a run.json nor a queued/running job exists for ``run_id``."""

    try:
        load_run(run_id)
        return
    except FileNotFoundError as exc:
        if load_run_job(run_id)["status"] != "unknown":
            return
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _wrap_404(action):
    try:
        return action()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _export_response(run_id: str, name: str, *, media_type: str, export_filename: str) -> FileResponse:
    export_root = _wrap_404(lambda: ensure_run_paper_exports(run_id))
    return FileResponse(export_root / name, media_type=media_type, filename=export_filename)


# ---------------------------------------------------------------------------
# Read-only registries
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/checkpoints")
def get_checkpoints() -> dict[str, Any]:
    return load_checkpoint_registry()


@app.get("/datasets")
def get_datasets() -> dict[str, Any]:
    return load_dataset_registry()


@app.get("/models")
def get_models() -> dict[str, Any]:
    return load_model_registry()


@app.get("/evaluation-report")
def get_evaluation_report() -> dict[str, Any]:
    return load_evaluation_report()


# ---------------------------------------------------------------------------
# Run listing / detail
# ---------------------------------------------------------------------------


@app.get("/runs")
def get_runs() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/experiments")
def get_experiments() -> list[dict[str, Any]]:
    return list_experiments()


@app.get("/experiments/{experiment_id}/runs")
def get_experiment_runs(experiment_id: str) -> list[dict[str, Any]]:
    return list_experiment_runs(experiment_id)


@app.get("/compare")
def get_compare(run_ids: str = "") -> dict[str, Any]:
    selected = [item.strip() for item in run_ids.split(",") if item.strip()]
    return compare_run_rows(selected)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        job = load_run_job(run_id)
        if job["status"] == "unknown":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": run_id,
            "status": job["status"],
            "project": job.get("project"),
            "experiment": job.get("experiment"),
            "created_at": job.get("started_at"),
            "updated_at": job.get("finished_at") or job.get("started_at"),
            "modes": job.get("modes", []),
            "stages": [{"name": "job", "status": job["status"]}],
        }


@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict[str, Any]:
    _require_run_or_job(run_id)
    return load_run_metrics(run_id)


@app.get("/runs/{run_id}/utterances")
def get_run_utterances(run_id: str) -> list[dict[str, Any]]:
    _require_run_or_job(run_id)
    return load_run_utterances(run_id)


@app.get("/runs/{run_id}/job")
def get_run_job(run_id: str) -> dict[str, Any]:
    job = load_run_job(run_id)
    if job["status"] != "unknown":
        return job
    run = _wrap_404(lambda: load_run(run_id))
    return {
        "run_id": run_id,
        "status": run.get("status", "unknown"),
        "run": run,
        "stdout_log": "logs/job.stdout.log",
        "stderr_log": "logs/job.stderr.log",
    }


@app.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str) -> dict[str, Any]:
    _require_run_or_job(run_id)
    return load_run_logs(run_id)


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str) -> dict[str, Any]:
    _require_run_or_job(run_id)
    return load_run_events(run_id)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@app.post("/runs")
def create_run(request: RunCreateRequest) -> dict[str, Any]:
    response = start_run(request.model_dump())
    if response["status"] == "failed":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.post("/runs/{run_id}/start")
def start_existing_run(run_id: str, request: RunCreateRequest) -> dict[str, Any]:
    payload = request.model_dump()
    payload["run_id"] = run_id
    response = start_run(payload)
    if response["status"] == "failed":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.post("/runs/{run_id}/cancel")
def cancel_existing_run(run_id: str) -> dict[str, Any]:
    return _wrap_404(lambda: cancel_run(run_id))


@app.post("/runs/{run_id}/retry")
def retry_existing_run(run_id: str) -> dict[str, Any]:
    return _wrap_404(lambda: retry_run(run_id))


@app.post("/runs/{run_id}/resume")
def resume_existing_run(run_id: str) -> dict[str, Any]:
    return _wrap_404(lambda: resume_run(run_id))


# ---------------------------------------------------------------------------
# Exports / artifacts
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/exports/main_table.csv")
def get_run_main_table(run_id: str) -> FileResponse:
    return _export_response(run_id, "main_table.csv", media_type="text/csv", export_filename=f"{run_id}_main_table.csv")


@app.get("/runs/{run_id}/exports/error_breakdown.csv")
def get_run_error_breakdown(run_id: str) -> FileResponse:
    return _export_response(
        run_id, "error_breakdown.csv", media_type="text/csv", export_filename=f"{run_id}_error_breakdown.csv"
    )


@app.get("/runs/{run_id}/exports/utterance_examples.jsonl")
def get_run_utterance_examples(run_id: str) -> FileResponse:
    return _export_response(
        run_id,
        "qualitative_examples.jsonl",
        media_type="application/x-ndjson",
        export_filename=f"{run_id}_utterance_examples.jsonl",
    )


@app.get("/experiments/{experiment_id}/exports/ablation_table.csv")
def get_experiment_ablation_table(experiment_id: str) -> FileResponse:
    path = ensure_experiment_ablation_export(experiment_id)
    return FileResponse(path, media_type="text/csv", filename=f"{experiment_id}_ablation_table.csv")


@app.get("/artifacts/{run_id}/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
    run = _wrap_404(lambda: load_run(run_id))
    root = Path(run["artifact_root"]).resolve()
    path = (root / artifact_path).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Artifact path escapes run root")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    return FileResponse(path)
