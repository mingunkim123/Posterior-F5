import json
import importlib.util
from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("platform/workers/run_posterior_f5_pipeline.py")


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("run_posterior_f5_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_run_scaffold_can_plan_posterior_encoder_command(tmp_path):
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
            "run_posterior_encoder_plan",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--mode",
            "posterior_encoder",
            "--posterior_encoder_ckpt",
            "ckpts/posterior_encoder/dev_latest/model_last.pt",
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
    command_row = json.loads((run_root / "generated" / "posterior_encoder" / "commands.jsonl").read_text(encoding="utf-8"))

    assert "posterior_encoder" in command_row["command"]
    assert "--posterior_encoder_ckpt" in command_row["command"]


def test_run_scaffold_can_plan_hybrid_ssl_cache_command(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-used-in-dry-run")
    manifest = tmp_path / "manifest.jsonl"
    ssl_cache = tmp_path / "ssl.jsonl"
    ssl_cache.write_text('{"utterance_id":"utt-001","audio_path":"ref.wav","feature_path":"utt-001.npz"}\n')
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
            "run_hybrid_plan",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--mode",
            "hybrid",
            "--ssl_cache",
            str(ssl_cache),
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
    command_row = json.loads((run_root / "generated" / "hybrid" / "commands.jsonl").read_text(encoding="utf-8"))

    assert "hybrid" in command_row["command"]
    assert "--ssl_cache" in command_row["command"]


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
    assert (run_root / "metrics" / "significance.json").exists()


def test_run_scaffold_can_plan_audio_metrics_stage(tmp_path):
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
            "run_audio_metrics_plan",
            "--artifact_root",
            str(tmp_path / "runs"),
            "--manifest",
            str(manifest),
            "--run_inference",
            "--inference_dry_run",
            "--run_audio_metrics",
            "--audio_metrics_dry_run",
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
    audio_metrics = json.loads((run_root / "metrics" / "hard.audio_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "metrics" / "summary.json").read_text(encoding="utf-8"))

    assert payload["stages"][2]["name"] == "audio_metrics"
    assert payload["stages"][2]["status"] == "planned"
    assert audio_metrics["status"] == "planned"
    assert "rtf_mean" in summary["modes"][0]
    assert "speaker_similarity_mean" in summary["modes"][0]
    assert "utmos_mean" in summary["modes"][0]


def test_run_scaffold_can_apply_yaml_config(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": "src/f5_tts/infer/examples/basic/basic_ref_en.wav",
                "ref_text": "Reference transcript.",
                "gen_text": "Target text.",
                "text": "Target text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
name: config_test
manifest: "{manifest}"
artifact_root: "{tmp_path / 'runs'}"
modes:
  - hard
generation:
  model: F5TTS_v1_Base
  checkpoint_id: f5tts_v1_base_hf
  checkpoint: hf://example/checkpoint.safetensors
  checkpoint_hash: hf:example
  vocoder: vocos
  seed: 77
  run_inference: true
  inference_dry_run: true
metrics:
  run_prediction: true
  prediction_dry_run: true
  run_metrics: true
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--run_id",
            "run_from_config",
            "--fail_if_exists",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    run_root = Path(result.stdout.strip())
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    command_row = json.loads((run_root / "generated" / "hard" / "commands.jsonl").read_text(encoding="utf-8"))

    assert payload["experiment"] == "config_test"
    assert payload["modes"] == ["hard"]
    assert payload["generation"]["seed"] == 77
    assert command_row["status"] == "planned"
    assert command_row["checkpoint_id"] == "f5tts_v1_base_hf"


def test_run_scaffold_records_generation_metadata_and_rejects_tiny_wavs(tmp_path):
    module = _load_worker_module()
    repo_root = tmp_path / "repo"
    infer_dir = repo_root / "src" / "f5_tts" / "infer"
    infer_dir.mkdir(parents=True)
    (infer_dir / "infer_cli.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--output_file", required=True)
args, _ = parser.parse_known_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir, args.output_file).write_bytes(b"tiny")
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "ref_audio": str(tmp_path / "ref.wav"),
                "ref_text": "Reference transcript.",
                "gen_text": "Target text.",
                "text": "Target text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    args = SimpleNamespace(
        run_id="run_tiny_wav",
        project="Posterior-F5",
        experiment="unit_test",
        manifest=str(manifest),
        artifact_root=str(tmp_path / "runs"),
        modes=["hard"],
        model="F5TTS_v1_Base",
        checkpoint_id="",
        checkpoint="",
        checkpoint_hash="",
        vocoder="vocos",
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
        seed=1234,
        posterior_asr="facebook/wav2vec2-base-960h",
        run_posterior_extraction=False,
        whisper_model="openai/whisper-large-v3-turbo",
        skip_whisper=False,
        ctc_model="",
        asr_device=None,
        torch_dtype="float32",
        run_inference=True,
        inference_dry_run=False,
        hard_ref_text_source="empty",
        infer_device=None,
        vocab_file="",
        posterior_encoder_ckpt="",
        ssl_cache="",
        min_wav_bytes=44,
        eval_asr="openai/whisper-large-v3-turbo",
        run_prediction=False,
        prediction_dry_run=False,
        eval_device=None,
        eval_torch_dtype="float32",
        run_metrics=False,
        metrics_dry_run=False,
        language="en",
        ctc_top_k=8,
        fail_if_exists=True,
    )

    try:
        module.scaffold_run(args, repo_root=repo_root)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected tiny wav inference to fail")

    run_root = Path(args.artifact_root) / args.run_id
    command_row = json.loads((run_root / "generated" / "hard" / "commands.jsonl").read_text(encoding="utf-8"))
    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert command_row["status"] == "failed"
    assert command_row["output_exists"] is True
    assert command_row["output_size_bytes"] == 4
    assert "too small" in command_row["error_message"]
