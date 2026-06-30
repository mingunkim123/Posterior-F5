import json
import subprocess
import sys
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


def _load_indexer():
    import importlib.util

    path = Path("platform/backend/app/services/indexer.py")
    spec = importlib.util.spec_from_file_location("posterior_f5_indexer_test", path)
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


def test_sqlite_indexer_rebuilds_metadata_cache(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello","subset":"clean"}\n', encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run_id",
            "run_indexed",
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

    indexer = _load_indexer()
    result = indexer.rebuild_index(artifact_root=tmp_path / "runs", db_path=tmp_path / "metadata.sqlite3")

    assert result["runs"] == 1
    assert result["metrics"] == 4
    assert result["utterances"] == 1
    assert (tmp_path / "metadata.sqlite3").exists()


def test_run_store_compare_rows_and_paper_exports(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello","subset":"clean"}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run_id",
            "run_paper_export",
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
    compare = run_store.compare_run_rows(artifact_root=tmp_path / "runs")
    assert len(compare["rows"]) == 4
    assert {row["mode"] for row in compare["rows"]} == {"hard", "oracle", "length_only", "soft_ctc"}
    assert {row["subset"] for row in compare["rows"]} == {"all"}

    export_root = run_store.ensure_run_paper_exports("run_paper_export", artifact_root=tmp_path / "runs")
    assert (export_root / "main_table.csv").exists()
    assert (export_root / "error_breakdown.csv").exists()
    assert (export_root / "qualitative_examples.jsonl").exists()
    assert "soft_ctc" in (export_root / "main_table.csv").read_text(encoding="utf-8")

    ablation_path = run_store.ensure_experiment_ablation_export("baseline_dev_small", artifact_root=tmp_path / "runs")
    assert ablation_path.exists()
    assert "run_paper_export" in ablation_path.read_text(encoding="utf-8")


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
    assert run_store.load_run_job("run_async_job", artifact_root=tmp_path / "runs")["status"] == "queued"

    subprocess.run(
        [
            sys.executable,
            "platform/workers/run_job_worker.py",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--once",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    job = run_store.load_run_job("run_async_job", artifact_root=tmp_path / "runs")
    run = run_store.load_run("run_async_job", artifact_root=tmp_path / "runs")
    logs = run_store.load_run_logs("run_async_job", artifact_root=tmp_path / "runs")

    assert job["status"] == "completed"
    assert run["status"] == "completed"
    events = run_store.load_run_events("run_async_job", artifact_root=tmp_path / "runs")
    assert {item["path"] for item in logs["files"]} == {"logs/job.stderr.log", "logs/job.stdout.log"}
    assert any(event["stage"] == "job" and event["message"] == "completed" for event in events["events"])


def test_run_store_cancel_retry_resume_and_dataset_registry(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello"}\n', encoding="utf-8")

    run_store = _load_run_store()
    response = run_store.start_run(
        {
            "run_id": "run_cancel_retry",
            "project": "Posterior-F5",
            "experiment": "ops",
            "manifest": str(manifest),
            "modes": ["hard"],
        },
        artifact_root=tmp_path / "runs",
    )

    assert response["status"] == "submitted"
    assert run_store.cancel_run("run_cancel_retry", artifact_root=tmp_path / "runs")["status"] == "cancelled"

    retry = run_store.retry_run("run_cancel_retry", artifact_root=tmp_path / "runs")
    assert retry["status"] == "queued"

    posterior = tmp_path / "runs" / "run_cancel_retry" / "posterior_cache" / "run.posterior.jsonl"
    posterior.parent.mkdir(parents=True, exist_ok=True)
    posterior.write_text('{"utterance_id":"utt-001","mean_entropy":0.4}\n', encoding="utf-8")
    resume = run_store.resume_run("run_cancel_retry", artifact_root=tmp_path / "runs")
    assert resume["status"] == "queued"
    assert "--run_posterior_extraction" not in resume["command"]

    datasets = run_store.load_dataset_registry()
    assert any(dataset["id"] == "dev_smoke" for dataset in datasets["datasets"])


def test_run_store_fail_if_exists_does_not_self_conflict(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello"}\n', encoding="utf-8")

    run_store = _load_run_store()
    response = run_store.start_run(
        {
            "run_id": "run_fail_if_exists",
            "project": "Posterior-F5",
            "experiment": "fail_if_exists",
            "manifest": str(manifest),
            "modes": ["hard"],
            "fail_if_exists": True,
        },
        artifact_root=tmp_path / "runs",
    )

    assert response["status"] == "submitted"
    assert "--fail_if_exists" not in response["command"]

    duplicate = run_store.start_run(
        {
            "run_id": "run_fail_if_exists",
            "project": "Posterior-F5",
            "experiment": "fail_if_exists",
            "manifest": str(manifest),
            "modes": ["hard"],
            "fail_if_exists": True,
        },
        artifact_root=tmp_path / "runs",
    )

    assert duplicate["status"] == "failed"
    assert "already exists" in duplicate["stderr"]


def test_checkpoint_registry_resolves_command_options():
    run_store = _load_run_store()
    registry = run_store.load_checkpoint_registry()

    assert registry["checkpoints"]
    assert registry["checkpoints"][0]["id"] == "f5tts_v1_base_hf"

    command = run_store.build_pipeline_command(
        run_store.resolve_checkpoint_payload(
            {
                "run_id": "run_checkpoint_registry",
                "manifest": "platform/samples/manifests/dev_smoke.jsonl",
                "modes": ["hard"],
                "checkpoint_id": "f5tts_v1_base_hf",
            }
        )
    )

    assert "--checkpoint_id" in command
    assert "f5tts_v1_base_hf" in command
    assert "--checkpoint" in command
    assert "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors" in command
    assert "--checkpoint_hash" in command


def test_run_scaffold_records_checkpoint_registry_metadata(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "platform/workers/run_posterior_f5_pipeline.py",
            "--run_id",
            "run_checkpoint_metadata",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--checkpoint_id",
            "f5tts_v1_base_hf",
            "--checkpoint",
            "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
            "--checkpoint_hash",
            "hf:SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert run["model"]["checkpoint_id"] == "f5tts_v1_base_hf"
    assert run["model"]["checkpoint_path"] == "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    assert run["model"]["checkpoint_hash"] == "hf:SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"


def test_worker_marks_stale_running_run_failed_after_process_error(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","text":"hello"}\n', encoding="utf-8")

    run_store = _load_run_store()
    run_store.start_run(
        {
            "run_id": "run_async_failure",
            "project": "Posterior-F5",
            "experiment": "async_failure",
            "manifest": str(manifest),
            "modes": ["hard"],
            "run_posterior_extraction": False,
            "run_inference": False,
            "run_prediction": True,
            "prediction_dry_run": True,
            "run_metrics": False,
        },
        artifact_root=tmp_path / "runs",
    )

    subprocess.run(
        [
            sys.executable,
            "platform/workers/run_job_worker.py",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--once",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    job = run_store.load_run_job("run_async_failure", artifact_root=tmp_path / "runs")
    run = run_store.load_run("run_async_failure", artifact_root=tmp_path / "runs")

    assert job["exit_code"] == 1
    assert job["status"] == "failed"
    assert run["status"] == "failed"
    assert run["stages"][-1]["name"] == "prediction"
    assert run["stages"][-1]["status"] == "failed"
    assert "missing ref_audio or audio_path" in job["error_message"]


def test_utterance_inspector_payload_includes_predictions_diff_and_entropy(tmp_path):
    run_root = tmp_path / "runs" / "run_utterance_payload"
    (run_root / "predictions").mkdir(parents=True)
    (run_root / "posterior_cache").mkdir(parents=True)
    (run_root / "generated" / "hard").mkdir(parents=True)
    (run_root / "manifest.jsonl").write_text(
        '{"utterance_id":"utt-001","ref_audio":"ref.wav","text":"hello world","subset":"clean"}\n',
        encoding="utf-8",
    )
    (run_root / "run.json").write_text(
        json.dumps({"run_id": "run_utterance_payload", "status": "completed", "artifact_root": str(run_root)}),
        encoding="utf-8",
    )
    (run_root / "predictions" / "hard.jsonl").write_text(
        '{"utterance_id":"utt-001","hypothesis":"hello there","status":"succeeded"}\n',
        encoding="utf-8",
    )
    (run_root / "posterior_cache" / "run.posterior.jsonl").write_text(
        '{"utterance_id":"utt-001","one_best":"hello world","mean_entropy":0.25}\n',
        encoding="utf-8",
    )

    run_store = _load_run_store()
    utterances = run_store.load_run_utterances("run_utterance_payload", artifact_root=tmp_path / "runs")

    assert utterances[0]["posterior_entropy"] == 0.25
    assert utterances[0]["modes"]["hard"]["prediction_text"] == "hello there"
    assert {token["op"] for token in utterances[0]["modes"]["hard"]["diff"]} >= {"equal", "delete", "insert"}
