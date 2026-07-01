"""Run queued Posterior-F5 jobs outside the FastAPI process."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_APP = REPO_ROOT / "platform" / "backend" / "app"
if str(BACKEND_APP) not in sys.path:
    sys.path.insert(0, str(BACKEND_APP))

from run_store import (  # noqa: E402
    append_event,
    default_artifact_root,
    job_path,
    list_queued_jobs,
    load_run_job,
    read_json,
    read_text_tail,
    run_path,
    timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process queued Posterior-F5 MLOps jobs.")
    parser.add_argument("--artifact_root", default=None, help="Root directory for run artifacts.")
    parser.add_argument("--poll_interval", type=float, default=2.0, help="Seconds between queue polls.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    return parser.parse_args()


def artifact_root_from_arg(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    return default_artifact_root(REPO_ROOT)


def mark_failed(job_file: Path, job: dict[str, Any], message: str, *, exit_code: int = 1) -> dict[str, Any]:
    job["status"] = "failed"
    job["exit_code"] = exit_code
    job["finished_at"] = timestamp()
    job["error_message"] = message
    write_json(job_file, job)
    append_event(str(job.get("run_id")), level="error", stage="job", message=message, artifact_root=job_file.parent.parent)
    return job


def mark_running_stages_failed(run: dict[str, Any]) -> None:
    for stage in run.get("stages", []):
        if stage.get("status") == "running":
            stage["status"] = "failed"


def failure_message(stderr_path: Path, exit_code: int) -> str:
    stderr = read_text_tail(stderr_path, 4000).strip()
    return stderr or f"Job exited with code {exit_code}"


def normalize_worker_command(command: list[str]) -> list[str]:
    if not command:
        return command

    executable = command[0]
    executable_path = Path(executable)
    if executable_path.name in {"python", "python3"} and executable_path.is_absolute() and not executable_path.exists():
        return [sys.executable, *command[1:]]
    return command


def run_job(run_id: str, *, artifact_root: Path) -> dict[str, Any]:
    run_root = run_path(run_id, artifact_root=artifact_root)
    job_file = job_path(run_id, artifact_root=artifact_root)
    job = load_run_job(run_id, artifact_root=artifact_root)
    if job.get("status") != "queued":
        return job

    stdout_path = run_root / job.get("stdout_log", "logs/job.stdout.log")
    stderr_path = run_root / job.get("stderr_log", "logs/job.stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    command = normalize_worker_command(job.get("command") or [])
    if not command:
        return mark_failed(job_file, job, "Job command is empty")

    job["command"] = command
    job["status"] = "running"
    job["started_at"] = timestamp()
    write_json(job_file, job)
    append_event(run_id, level="info", stage="job", message="started", artifact_root=artifact_root)

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        except OSError as exc:
            return mark_failed(job_file, job, str(exc))

        job["pid"] = process.pid
        write_json(job_file, job)
        exit_code = process.wait()

    job = read_json(job_file) if job_file.exists() else job
    job["exit_code"] = exit_code
    job["finished_at"] = timestamp()
    job["status"] = "succeeded" if exit_code == 0 else "failed"
    if exit_code != 0:
        job["error_message"] = failure_message(stderr_path, exit_code)
        append_event(run_id, level="error", stage="job", message=job["error_message"], artifact_root=artifact_root)
    else:
        append_event(run_id, level="info", stage="job", message="completed", artifact_root=artifact_root)

    run_json = run_root / "run.json"
    if run_json.exists():
        try:
            run = read_json(run_json)
            if exit_code != 0:
                if run.get("status") in {"queued", "submitted", "running"}:
                    run["status"] = "failed"
                    run["updated_at"] = timestamp()
                    run["error_message"] = job["error_message"]
                    mark_running_stages_failed(run)
                    write_json(run_json, run)
                job["run"] = run
                job["status"] = "failed"
            else:
                job["run"] = run
                job["status"] = run.get("status", job["status"])
        except ValueError:
            job["status"] = "failed"
            job["error_message"] = "run.json is not valid JSON"

    write_json(job_file, job)
    return job


def work_loop(*, artifact_root: Path, poll_interval: float, once: bool) -> int:
    processed = 0
    while True:
        queued = list_queued_jobs(artifact_root=artifact_root)
        if queued:
            job = queued[0]
            run_job(str(job["run_id"]), artifact_root=artifact_root)
            processed += 1
            if once:
                return 0
            continue

        if once:
            return 0 if processed else 2
        time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    artifact_root = artifact_root_from_arg(args.artifact_root)
    raise SystemExit(work_loop(artifact_root=artifact_root, poll_interval=args.poll_interval, once=args.once))


if __name__ == "__main__":
    main()
