import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("platform/workers/run_posterior_f5_pipeline.py")


def test_run_scaffold_creates_artifact_contract(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utterance_id":"utt-001","text":"hello"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_test_001",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--experiment",
            "unit_test",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    assert run_root.exists()
    assert (run_root / "run.json").exists()
    assert (run_root / "config.yaml").exists()
    assert (run_root / "manifest.jsonl").read_text(encoding="utf-8") == manifest.read_text(encoding="utf-8")

    for directory in [
        "posterior_cache/posterior_npz",
        "generated/hard",
        "generated/oracle",
        "generated/length_only",
        "generated/soft_ctc",
        "predictions",
        "metrics",
        "logs",
        "dashboard_snapshot",
    ]:
        assert (run_root / directory).is_dir()

    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_test_001"
    assert payload["experiment"] == "unit_test"
    assert payload["status"] == "completed"
    assert payload["manifest"] == "manifest.jsonl"
    assert payload["modes"] == ["hard", "oracle", "length_only", "soft_ctc"]
    assert payload["posterior_file"] == "posterior_cache/run.posterior.jsonl"
    assert payload["generation"]["seed"] == 1234
    assert "git" in payload


def test_run_scaffold_fails_for_missing_manifest(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_missing_manifest",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(tmp_path / "missing.jsonl"),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Manifest does not exist" in result.stderr


def test_run_scaffold_can_extract_posterior_from_manifest_text(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-used-when-skip-whisper")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": str(ref_audio),
                "ref_text": "Some call me nature.",
                "gen_text": "Target text.",
                "text": "Target text.",
                "subset": "clean",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_posterior_text",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_posterior_extraction",
            "--skip_whisper",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    posterior_input = run_root / "posterior_cache" / "posterior_input_manifest.jsonl"
    posterior_output = run_root / "posterior_cache" / "run.posterior.jsonl"
    log_path = run_root / "logs" / "extract_posterior.log"

    assert posterior_input.exists()
    assert posterior_output.exists()
    assert log_path.exists()

    input_row = json.loads(posterior_input.read_text(encoding="utf-8").strip())
    output_row = json.loads(posterior_output.read_text(encoding="utf-8").strip())
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert input_row["audio_path"] == str(ref_audio)
    assert input_row["text"] == "Some call me nature."
    assert output_row["utterance_id"] == "utt-001"
    assert output_row["one_best"] == "Some call me nature."
    assert output_row["expected_ref_len"] == len("Some call me nature.".encode("utf-8"))
    assert payload["status"] == "completed"
    assert payload["posterior"]["asr"] == "manifest_text"
    assert payload["stages"][1]["name"] == "posterior_extraction"
    assert payload["stages"][1]["status"] == "succeeded"


def test_run_scaffold_can_plan_mode_inference_commands(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-used-in-dry-run")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": str(ref_audio),
                "ref_text": "Reference transcript.",
                "gen_text": "Target text.",
                "text": "Target text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_inference_plan",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_posterior_extraction",
            "--skip_whisper",
            "--run_inference",
            "--inference_dry_run",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["stages"][2]["name"] == "inference"
    assert payload["stages"][2]["status"] == "planned"
    assert payload["generation"]["inference_dry_run"] is True

    hard_command = json.loads((run_root / "generated" / "hard" / "commands.jsonl").read_text(encoding="utf-8"))
    oracle_command = json.loads((run_root / "generated" / "oracle" / "commands.jsonl").read_text(encoding="utf-8"))
    soft_command = json.loads((run_root / "generated" / "soft_ctc" / "commands.jsonl").read_text(encoding="utf-8"))

    assert hard_command["utterance_id"] == "utt-001"
    assert hard_command["output"] == "generated/hard/utt-001.wav"
    assert "--ref_text_mode" in hard_command["command"]
    assert "hard" in hard_command["command"]
    assert oracle_command["output"] == "generated/oracle/utt-001.wav"
    assert "Reference transcript." in oracle_command["command"]
    assert soft_command["output"] == "generated/soft_ctc/utt-001.wav"
    assert "--posterior_file" in soft_command["command"]


def test_run_scaffold_can_plan_prediction_jsonl(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-used-in-dry-run")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": str(ref_audio),
                "ref_text": "Reference transcript.",
                "gen_text": "Target text.",
                "text": "Target text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_prediction_plan",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_posterior_extraction",
            "--skip_whisper",
            "--run_inference",
            "--inference_dry_run",
            "--run_prediction",
            "--prediction_dry_run",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    hard_prediction = json.loads((run_root / "predictions" / "hard.jsonl").read_text(encoding="utf-8"))
    soft_prediction = json.loads((run_root / "predictions" / "soft_ctc.jsonl").read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["stages"][3]["name"] == "prediction"
    assert payload["stages"][3]["status"] == "planned"
    assert payload["evaluation"]["prediction_dry_run"] is True
    assert hard_prediction["utterance_id"] == "utt-001"
    assert hard_prediction["mode"] == "hard"
    assert hard_prediction["generated_audio"] == "generated/hard/utt-001.wav"
    assert hard_prediction["hypothesis"] == ""
    assert hard_prediction["status"] == "planned"
    assert soft_prediction["generated_audio"] == "generated/soft_ctc/utt-001.wav"


def test_run_scaffold_can_compute_metrics_summary(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-used-in-dry-run")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": str(ref_audio),
                "ref_text": "Reference transcript.",
                "gen_text": "Target text.",
                "text": "Target text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_id",
            "run_metrics",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_posterior_extraction",
            "--skip_whisper",
            "--run_inference",
            "--inference_dry_run",
            "--run_prediction",
            "--prediction_dry_run",
            "--run_metrics",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    hard_metrics = json.loads((run_root / "metrics" / "hard.metrics.json").read_text(encoding="utf-8"))
    oracle_metrics = json.loads((run_root / "metrics" / "oracle.metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "metrics" / "summary.json").read_text(encoding="utf-8"))
    summary_csv = (run_root / "metrics" / "summary.csv").read_text(encoding="utf-8")

    assert payload["status"] == "completed"
    assert payload["stages"][4]["name"] == "metrics"
    assert payload["stages"][4]["status"] == "succeeded"
    assert hard_metrics["status"] == "succeeded"
    assert hard_metrics["mode"] == "hard"
    assert hard_metrics["num_utterances"] == 1
    assert hard_metrics["wer"] == 1.0
    assert oracle_metrics["mode"] == "oracle"
    assert len(summary["modes"]) == 4
    assert "mode,num_utterances,wer,cer" in summary_csv
