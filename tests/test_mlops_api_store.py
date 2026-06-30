import json
import subprocess
import sys
import time
from pathlib import Path


RUNNER = Path("platform/workers/run_posterior_f5_pipeline.py")


def _load_run_store():
    import importlib.util

    path = Path("platform/backend/app/run_store.py")
    spec = importlib.util.spec_from_file_location("posterior_f5_run_store_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_store_lists_runs_and_metrics(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello"}\n', encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run_id",
            "run_api_store",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_prediction",
            "--prediction_dry_run",
            "--run_metrics",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_store = _load_run_store()
    runs = run_store.list_runs(artifact_root=tmp_path / "runs")
    run = run_store.load_run("run_api_store", artifact_root=tmp_path / "runs")
    metrics = run_store.load_run_metrics("run_api_store", artifact_root=tmp_path / "runs")

    assert [item["run_id"] for item in runs] == ["run_api_store"]
    assert run["status"] == "completed"
    assert metrics["run_id"] == "run_api_store"
    assert len(metrics["modes"]) == 4


def test_run_store_builds_pipeline_command(tmp_path):
    run_store = _load_run_store()
    command = run_store.build_pipeline_command(
        {
            "run_id": "run_api_command",
            "manifest": "manifests/dev_small.jsonl",
            "modes": ["hard", "soft_ctc"],
            "run_posterior_extraction": True,
            "skip_whisper": True,
            "run_inference": True,
            "inference_dry_run": True,
        },
        artifact_root=tmp_path / "runs",
    )

    assert "platform/workers/run_posterior_f5_pipeline.py" in command[1]
    assert "--run_id" in command
    assert "--mode" in command
    assert command.count("--mode") == 2
    assert "--run_posterior_extraction" in command
    assert "--inference_dry_run" in command


def test_run_store_starts_job_and_collects_logs(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello"}\n', encoding="utf-8")

    run_store = _load_run_store()
    response = run_store.start_run(
        {
            "run_id": "run_async_job",
            "project": "Posterior-F5",
            "experiment": "async_job",
            "manifest": str(manifest),
            "modes": ["hard"],
            "run_posterior_extraction": False,
            "run_inference": False,
            "run_prediction": False,
            "run_metrics": False,
        },
        artifact_root=tmp_path / "runs",
    )

    assert response["status"] == "submitted"
    assert response["run_id"] == "run_async_job"

    job = {}
    for _ in range(50):
        job = run_store.load_run_job("run_async_job", artifact_root=tmp_path / "runs")
        if job["status"] != "running":
            break
        time.sleep(0.1)

    run = run_store.load_run("run_async_job", artifact_root=tmp_path / "runs")
    logs = run_store.load_run_logs("run_async_job", artifact_root=tmp_path / "runs")

    assert job["status"] == "completed"
    assert run["status"] == "completed"
    assert {item["path"] for item in logs["files"]} == {"logs/job.stderr.log", "logs/job.stdout.log"}
