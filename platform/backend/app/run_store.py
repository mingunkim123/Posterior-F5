"""Filesystem-backed run store for the Posterior-F5 MLOps API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_artifact_root(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    return root / "mlops_artifacts" / "runs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(run_id: str, *, artifact_root: Path | None = None) -> Path:
    root = artifact_root or default_artifact_root()
    return root / run_id


def load_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    path = run_path(run_id, artifact_root=artifact_root) / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return read_json(path)


def list_runs(*, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = artifact_root or default_artifact_root()
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_json in sorted(root.glob("*/run.json"), reverse=True):
        try:
            runs.append(read_json(run_json))
        except json.JSONDecodeError:
            runs.append(
                {
                    "run_id": run_json.parent.name,
                    "status": "invalid",
                    "artifact_root": str(run_json.parent),
                }
            )
    return runs


def load_run_metrics(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    metrics_path = run_path(run_id, artifact_root=artifact_root) / "metrics" / "summary.json"
    if not metrics_path.exists():
        return {"run_id": run_id, "modes": []}
    payload = read_json(metrics_path)
    payload.setdefault("run_id", run_id)
    return payload


def load_run_utterances(run_id: str, *, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = run_path(run_id, artifact_root=artifact_root)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    modes = [item.name for item in (root / "generated").iterdir()] if (root / "generated").exists() else []
    utterances: list[dict[str, Any]] = []
    for row in rows:
        utterance_id = row.get("utterance_id")
        if not utterance_id:
            continue
        utterance = {
            "utterance_id": utterance_id,
            "subset": row.get("subset"),
            "text": row.get("text"),
            "gen_text": row.get("gen_text"),
            "ref_audio": row.get("ref_audio") or row.get("audio_path"),
            "modes": {},
        }
        for mode in modes:
            wav = root / "generated" / mode / f"{utterance_id}.wav"
            prediction = root / "predictions" / f"{mode}.jsonl"
            utterance["modes"][mode] = {
                "generated_audio": str(wav.relative_to(root)) if wav.exists() else None,
                "prediction_file": str(prediction.relative_to(root)) if prediction.exists() else None,
            }
        utterances.append(utterance)
    return utterances


def build_pipeline_command(payload: dict[str, Any], *, artifact_root: Path | None = None) -> list[str]:
    repo_root = repository_root()
    script = repo_root / "platform" / "workers" / "run_posterior_f5_pipeline.py"
    command = [sys.executable, str(script)]

    scalar_options = {
        "run_id": "--run_id",
        "project": "--project",
        "experiment": "--experiment",
        "manifest": "--manifest",
        "model": "--model",
        "checkpoint": "--checkpoint",
        "vocoder": "--vocoder",
        "seed": "--seed",
        "language": "--language",
    }
    for key, flag in scalar_options.items():
        value = payload.get(key)
        if value is not None and value != "":
            command.extend([flag, str(value)])

    if artifact_root is not None:
        command.extend(["--artifact_root", str(artifact_root)])

    for mode in payload.get("modes", []) or []:
        command.extend(["--mode", str(mode)])

    boolean_flags = {
        "run_posterior_extraction": "--run_posterior_extraction",
        "skip_whisper": "--skip_whisper",
        "run_inference": "--run_inference",
        "inference_dry_run": "--inference_dry_run",
        "run_prediction": "--run_prediction",
        "prediction_dry_run": "--prediction_dry_run",
        "run_metrics": "--run_metrics",
        "metrics_dry_run": "--metrics_dry_run",
        "fail_if_exists": "--fail_if_exists",
    }
    for key, flag in boolean_flags.items():
        if payload.get(key):
            command.append(flag)

    return command


def start_run(payload: dict[str, Any], *, artifact_root: Path | None = None) -> dict[str, Any]:
    command = build_pipeline_command(payload, artifact_root=artifact_root)
    completed = subprocess.run(command, cwd=repository_root(), check=False, capture_output=True, text=True)
    response = {
        "status": "submitted" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0 and completed.stdout.strip():
        run_root = Path(completed.stdout.strip().splitlines()[-1])
        run_json = run_root / "run.json"
        if run_json.exists():
            response["run"] = read_json(run_json)
    return response
