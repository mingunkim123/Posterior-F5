"""Filesystem-backed run store for the Posterior-F5 MLOps API."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_artifact_root(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    return root / "mlops_artifacts" / "runs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_run_id() -> str:
    return f"run_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"


def run_path(run_id: str, *, artifact_root: Path | None = None) -> Path:
    root = artifact_root or default_artifact_root()
    return root / run_id


def job_path(run_id: str, *, artifact_root: Path | None = None) -> Path:
    return run_path(run_id, artifact_root=artifact_root) / "job.json"


def load_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    path = run_path(run_id, artifact_root=artifact_root) / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return read_json(path)


def list_runs(*, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = artifact_root or default_artifact_root()
    if not root.exists():
        return []

    runs_by_id: dict[str, dict[str, Any]] = {}
    for run_json in sorted(root.glob("*/run.json"), reverse=True):
        try:
            run = read_json(run_json)
            runs_by_id[run_json.parent.name] = run
        except json.JSONDecodeError:
            runs_by_id[run_json.parent.name] = {
                "run_id": run_json.parent.name,
                "status": "invalid",
                "artifact_root": str(run_json.parent),
            }

    for job_json in sorted(root.glob("*/job.json"), reverse=True):
        run_id = job_json.parent.name
        if run_id in runs_by_id:
            continue
        try:
            job = read_json(job_json)
        except json.JSONDecodeError:
            job = {"run_id": run_id, "status": "invalid"}
        runs_by_id[run_id] = {
            "run_id": run_id,
            "project": job.get("project"),
            "experiment": job.get("experiment"),
            "status": job.get("status", "running"),
            "created_at": job.get("started_at"),
            "updated_at": job.get("finished_at") or job.get("started_at"),
            "modes": job.get("modes", []),
            "stages": [{"name": "queued", "status": job.get("status", "running")}],
            "artifact_root": str(job_json.parent),
        }
    return sorted(runs_by_id.values(), key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)


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


def load_run_job(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    path = job_path(run_id, artifact_root=artifact_root)
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "unknown",
            "stdout_log": "logs/job.stdout.log",
            "stderr_log": "logs/job.stderr.log",
        }
    job = read_json(path)
    job.setdefault("run_id", run_id)
    return job


def read_text_tail(path: Path, max_bytes: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as file:
        if size > max_bytes:
            file.seek(size - max_bytes)
        return file.read().decode("utf-8", errors="replace")


def load_run_logs(run_id: str, *, artifact_root: Path | None = None, max_bytes: int = 12000) -> dict[str, Any]:
    root = run_path(run_id, artifact_root=artifact_root)
    log_root = root / "logs"
    files: list[dict[str, Any]] = []
    if log_root.exists():
        for path in sorted(log_root.glob("*.log")):
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "content": read_text_tail(path, max_bytes),
                }
            )
    return {"run_id": run_id, "files": files}


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


def finalize_job(process: subprocess.Popen[str], run_id: str, run_root: Path, job_file: Path) -> None:
    exit_code = process.wait()
    job = read_json(job_file) if job_file.exists() else {"run_id": run_id}
    job["exit_code"] = exit_code
    job["finished_at"] = timestamp()
    job["status"] = "succeeded" if exit_code == 0 else "failed"
    run_json = run_root / "run.json"
    if run_json.exists():
        try:
            job["run"] = read_json(run_json)
            job["status"] = job["run"].get("status", job["status"])
        except json.JSONDecodeError:
            job["status"] = "failed"
            job["error_message"] = "run.json is not valid JSON"
    write_json(job_file, job)


def start_run(payload: dict[str, Any], *, artifact_root: Path | None = None) -> dict[str, Any]:
    payload = dict(payload)
    run_id = payload.get("run_id") or generate_run_id()
    payload["run_id"] = run_id
    root = artifact_root or default_artifact_root()
    run_root = run_path(run_id, artifact_root=root)
    log_root = run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    command = build_pipeline_command(payload, artifact_root=artifact_root)
    stdout_path = log_root / "job.stdout.log"
    stderr_path = log_root / "job.stderr.log"
    job_file = job_path(run_id, artifact_root=root)
    job = {
        "run_id": run_id,
        "project": payload.get("project"),
        "experiment": payload.get("experiment"),
        "modes": payload.get("modes", []),
        "status": "running",
        "pid": None,
        "command": command,
        "started_at": timestamp(),
        "stdout_log": str(stdout_path.relative_to(run_root)),
        "stderr_log": str(stderr_path.relative_to(run_root)),
    }
    write_json(job_file, job)

    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=repository_root(),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
    except OSError as exc:
        stdout_file.close()
        stderr_file.close()
        job["status"] = "failed"
        job["exit_code"] = 1
        job["finished_at"] = timestamp()
        job["error_message"] = str(exc)
        write_json(job_file, job)
        return {
            "status": "failed",
            "run_id": run_id,
            "exit_code": 1,
            "command": command,
            "stdout": "",
            "stderr": str(exc),
            "job": job,
        }

    job["pid"] = process.pid
    write_json(job_file, job)
    stdout_file.close()
    stderr_file.close()
    thread = threading.Thread(target=finalize_job, args=(process, run_id, run_root, job_file), daemon=True)
    thread.start()

    response = {
        "status": "submitted",
        "run_id": run_id,
        "exit_code": None,
        "command": command,
        "stdout": "",
        "stderr": "",
        "job": job,
    }
    return response
