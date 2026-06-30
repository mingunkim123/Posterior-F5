"""FastAPI control plane for local Posterior-F5 MLOps runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from run_store import list_runs, load_run, load_run_metrics, load_run_utterances, start_run


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    project: str = "Posterior-F5"
    experiment: str = "baseline_dev_small"
    manifest: str
    modes: list[str] = Field(default_factory=lambda: ["hard", "oracle", "length_only", "soft_ctc"])
    model: str = "F5TTS_v1_Base"
    checkpoint: str = ""
    vocoder: str = "vocos"
    seed: int = 1234
    language: str = "en"
    run_posterior_extraction: bool = True
    skip_whisper: bool = True
    run_inference: bool = False
    inference_dry_run: bool = True
    run_prediction: bool = False
    prediction_dry_run: bool = True
    run_metrics: bool = False
    metrics_dry_run: bool = False
    fail_if_exists: bool = False


app = FastAPI(title="Posterior-F5 MLOps API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5174", "http://localhost:5174"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def get_runs() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs")
def create_run(request: RunCreateRequest) -> dict[str, Any]:
    response = start_run(model_to_dict(request))
    if response["status"] == "failed":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.post("/runs/{run_id}/start")
def start_existing_run(run_id: str, request: RunCreateRequest) -> dict[str, Any]:
    payload = model_to_dict(request)
    payload["run_id"] = run_id
    response = start_run(payload)
    if response["status"] == "failed":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict[str, Any]:
    try:
        load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return load_run_metrics(run_id)


@app.get("/runs/{run_id}/utterances")
def get_run_utterances(run_id: str) -> list[dict[str, Any]]:
    try:
        load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return load_run_utterances(run_id)


@app.get("/artifacts/{run_id}/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
    try:
        run = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    root = Path(run["artifact_root"]).resolve()
    path = (root / artifact_path).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Artifact path escapes run root")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    return FileResponse(path)
