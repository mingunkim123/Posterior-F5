"""Create a reproducible Posterior-F5 MLOps run directory.

Step 1 of the platform is intentionally small: establish the artifact
contract before running expensive ML jobs. Later stages can append posterior
extraction, inference, evaluation, and aggregation to the same run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from f5_tts.eval.bootstrap_significance import apply_multiple_comparison_corrections, paired_bootstrap_result


DEFAULT_MODES = ["hard", "oracle", "length_only", "soft_ctc"]
DEFAULT_ARTIFACT_ROOT = "mlops_artifacts/runs"
DEFAULT_RUN_OPTIONS = {
    "run_audio_metrics": False,
    "audio_metrics_dry_run": False,
    "run_speaker_similarity": False,
    "speaker_checkpoint": "",
    "speaker_device": None,
    "speaker_feat_type": "wavlm_large",
    "run_utmos": False,
    "utmos_device": None,
    "metrics_normalizer": "paper",
    "bootstrap_samples": 1000,
    "bootstrap_seed": 1234,
    "significance_baseline": "hard",
}


def ensure_run_option_defaults(args: argparse.Namespace) -> None:
    for name, value in DEFAULT_RUN_OPTIONS.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Posterior-F5 MLOps run scaffold.")
    parser.add_argument("--config", help="Optional YAML experiment config. CLI flags override only by rerunning with desired values.")
    parser.add_argument("--run_id", help="Run id. Defaults to run_<timestamp>.")
    parser.add_argument("--project", default="Posterior-F5", help="Project name recorded in run.json.")
    parser.add_argument("--experiment", default="baseline_dev_small", help="Experiment name recorded in run.json.")
    parser.add_argument("--manifest", help="Input manifest JSONL to snapshot into the run directory.")
    parser.add_argument("--artifact_root", default=DEFAULT_ARTIFACT_ROOT, help="Root directory for run artifacts.")
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=DEFAULT_MODES + ["posterior_encoder", "hybrid"],
        help="Mode to include. May be passed multiple times. Defaults to baseline modes.",
    )
    parser.add_argument("--model", default="F5TTS_v1_Base", help="F5-TTS model name.")
    parser.add_argument("--checkpoint_id", default="", help="Checkpoint registry id.")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path or URI.")
    parser.add_argument("--checkpoint_hash", default="", help="Checkpoint hash or immutable provenance fingerprint.")
    parser.add_argument("--vocoder", default="vocos", help="Vocoder name.")
    parser.add_argument("--nfe_step", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--posterior_asr", default="facebook/wav2vec2-base-960h")
    parser.add_argument(
        "--run_posterior_extraction",
        action="store_true",
        help="Run the posterior extraction stage after creating the scaffold.",
    )
    parser.add_argument("--whisper_model", default="openai/whisper-large-v3-turbo", help="Whisper model for 1-best ASR.")
    parser.add_argument("--skip_whisper", action="store_true", help="Use manifest text and skip Whisper transcription.")
    parser.add_argument("--ctc_model", default="", help="Optional CTC model for frame-level top-k posterior shards.")
    parser.add_argument("--asr_device", default=None, help="ASR device passed to the extractor, e.g. cuda:0 or cpu.")
    parser.add_argument("--torch_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="Run mode-by-mode F5-TTS inference after scaffold/posterior stages.",
    )
    parser.add_argument(
        "--inference_dry_run",
        action="store_true",
        help="Write inference command manifests without executing TTS.",
    )
    parser.add_argument(
        "--hard_ref_text_source",
        choices=["empty", "manifest"],
        default="manifest",
        help="Use empty ref_text for hard ASR baseline, or manifest ref_text for a cheap smoke run.",
    )
    parser.add_argument("--infer_device", default=None, help="Device passed to infer_cli.py, e.g. cuda, mps, or cpu.")
    parser.add_argument("--vocab_file", default="", help="Optional vocab file passed to infer_cli.py.")
    parser.add_argument("--posterior_encoder_ckpt", default="", help="Posterior encoder checkpoint for posterior_encoder mode.")
    parser.add_argument("--ssl_cache", default="", help="SSL feature cache JSONL for hybrid mode.")
    parser.add_argument("--min_wav_bytes", type=int, default=44, help="Minimum accepted wav file size after non-dry-run inference.")
    parser.add_argument(
        "--run_audio_metrics",
        action="store_true",
        help="Compute generated-audio duration/RTF and optional speaker-sim/UTMOS metrics.",
    )
    parser.add_argument(
        "--audio_metrics_dry_run",
        action="store_true",
        help="Write planned audio metric artifacts without loading audio metric models.",
    )
    parser.add_argument("--run_speaker_similarity", action="store_true", help="Compute ECAPA speaker similarity against ref_audio.")
    parser.add_argument("--speaker_checkpoint", default="", help="ECAPA/WavLM speaker similarity checkpoint.")
    parser.add_argument("--speaker_device", default=None, help="Speaker similarity device, e.g. cuda:0 or cpu.")
    parser.add_argument("--speaker_feat_type", default="wavlm_large", help="ECAPA upstream feature type.")
    parser.add_argument("--run_utmos", action="store_true", help="Compute UTMOS with SpeechMOS.")
    parser.add_argument("--utmos_device", default=None, help="UTMOS device, e.g. cuda:0 or cpu.")
    parser.add_argument("--eval_asr", default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--run_prediction",
        action="store_true",
        help="Transcribe generated wavs with the evaluation ASR into predictions/<mode>.jsonl.",
    )
    parser.add_argument(
        "--prediction_dry_run",
        action="store_true",
        help="Write prediction JSONL placeholders without loading the evaluation ASR.",
    )
    parser.add_argument("--eval_device", default=None, help="Evaluation ASR device, e.g. cuda:0, mps, or cpu.")
    parser.add_argument("--eval_torch_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--run_metrics",
        action="store_true",
        help="Evaluate predictions/<mode>.jsonl into metrics/<mode>.metrics.json and summary files.",
    )
    parser.add_argument(
        "--metrics_dry_run",
        action="store_true",
        help="Write planned metric artifacts without executing eval_posterior_f5.py.",
    )
    parser.add_argument(
        "--metrics_normalizer",
        default="paper",
        choices=["paper", "lowercase", "none"],
        help="Text normalization profile for WER/CER.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Paired bootstrap samples per comparison.")
    parser.add_argument("--bootstrap_seed", type=int, default=1234, help="Paired bootstrap RNG seed.")
    parser.add_argument("--significance_baseline", default="hard", help="Baseline mode for paired bootstrap comparisons.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--ctc_top_k", type=int, default=8)
    parser.add_argument(
        "--fail_if_exists",
        action="store_true",
        help="Fail if the run directory already exists instead of updating metadata.",
    )
    args = parser.parse_args()
    if args.config:
        apply_config(args, load_config(args.config))
    return args


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Apply the small experiment YAML contract to argparse options."""

    scalar_map = {
        "name": "experiment",
        "manifest": "manifest",
        "artifact_root": "artifact_root",
    }
    for source, target in scalar_map.items():
        if config.get(source) not in (None, ""):
            setattr(args, target, config[source])

    modes = config.get("modes") or config.get("stage1_modes")
    if modes:
        args.modes = list(modes)

    posterior = config.get("posterior") or {}
    posterior_map = {
        "run_extraction": "run_posterior_extraction",
        "whisper_model": "whisper_model",
        "skip_whisper": "skip_whisper",
        "ctc_model": "ctc_model",
        "ctc_top_k": "ctc_top_k",
        "language": "language",
    }
    for source, target in posterior_map.items():
        if source in posterior:
            setattr(args, target, posterior[source])

    generation = config.get("generation") or {}
    generation_map = {
        "model": "model",
        "checkpoint_id": "checkpoint_id",
        "checkpoint": "checkpoint",
        "checkpoint_hash": "checkpoint_hash",
        "vocoder": "vocoder",
        "nfe_step": "nfe_step",
        "cfg_strength": "cfg_strength",
        "sway_sampling_coef": "sway_sampling_coef",
        "speed": "speed",
        "seed": "seed",
        "run_inference": "run_inference",
        "inference_dry_run": "inference_dry_run",
        "hard_ref_text_source": "hard_ref_text_source",
        "posterior_encoder_ckpt": "posterior_encoder_ckpt",
        "ssl_cache": "ssl_cache",
    }
    for source, target in generation_map.items():
        if source in generation:
            setattr(args, target, generation[source])

    audio_metrics = config.get("audio_metrics") or config.get("acoustic_metrics") or {}
    audio_metrics_map = {
        "run_audio_metrics": "run_audio_metrics",
        "audio_metrics_dry_run": "audio_metrics_dry_run",
        "run_speaker_similarity": "run_speaker_similarity",
        "speaker_checkpoint": "speaker_checkpoint",
        "speaker_device": "speaker_device",
        "speaker_feat_type": "speaker_feat_type",
        "run_utmos": "run_utmos",
        "utmos_device": "utmos_device",
    }
    for source, target in audio_metrics_map.items():
        if source in audio_metrics:
            setattr(args, target, audio_metrics[source])

    metrics = config.get("metrics") or {}
    metrics_map = {
        "run_prediction": "run_prediction",
        "prediction_dry_run": "prediction_dry_run",
        "eval_asr": "eval_asr",
        "run_metrics": "run_metrics",
        "metrics_dry_run": "metrics_dry_run",
        "normalizer": "metrics_normalizer",
        "metrics_normalizer": "metrics_normalizer",
        "report_rtf": "run_audio_metrics",
        "report_speaker_similarity": "run_speaker_similarity",
        "report_utmos": "run_utmos",
        "speaker_checkpoint": "speaker_checkpoint",
        "speaker_device": "speaker_device",
        "speaker_feat_type": "speaker_feat_type",
        "utmos_device": "utmos_device",
        "bootstrap_samples": "bootstrap_samples",
        "bootstrap_seed": "bootstrap_seed",
        "significance_baseline": "significance_baseline",
    }
    for source, target in metrics_map.items():
        if source in metrics:
            setattr(args, target, metrics[source])


def utc_or_local_now() -> datetime:
    return datetime.now().astimezone()


def generate_run_id(now: datetime | None = None) -> str:
    now = now or utc_or_local_now()
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(command: list[str], *, cwd: Path) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except OSError:
        return 1, ""
    return completed.returncode, completed.stdout.strip()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    branch_code, branch = run_command(["git", "branch", "--show-current"], cwd=repo_root)
    commit_code, commit = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status_code, status = run_command(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "branch": branch if branch_code == 0 else None,
        "commit": commit if commit_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
    }


def ensure_run_dirs(run_root: Path, modes: list[str]) -> None:
    dirs = [
        run_root / "posterior_cache" / "posterior_npz",
        run_root / "predictions",
        run_root / "metrics",
        run_root / "logs",
        run_root / "dashboard_snapshot",
    ]
    dirs.extend(run_root / "generated" / mode for mode in modes)
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def snapshot_manifest(manifest: str | None, run_root: Path) -> str | None:
    if not manifest:
        return None

    source = Path(manifest).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Manifest does not exist: {source}")

    destination = run_root / "manifest.jsonl"
    shutil.copy2(source, destination)
    return destination.name


def resolve_manifest_path(path: str, *, manifest_base_dir: Path | None, repo_root: Path) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return str(candidate)

    if manifest_base_dir is not None:
        manifest_candidate = manifest_base_dir / candidate
        if manifest_candidate.exists():
            return str(manifest_candidate.resolve())

    return str(candidate)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                rows.append(json.loads(line))
            else:
                rows.append({"audio_path": line})
    return rows


def prepare_posterior_input_manifest(
    *,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
    repo_root: Path,
) -> Path:
    if manifest_name is None:
        raise ValueError("A manifest is required for posterior extraction")

    run_manifest = run_root / manifest_name
    manifest_base_dir = source_manifest.expanduser().resolve().parent if source_manifest is not None else None
    output = run_root / "posterior_cache" / "posterior_input_manifest.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for index, row in enumerate(iter_jsonl(run_manifest)):
            audio_path = row.get("audio_path") or row.get("ref_audio")
            if not audio_path:
                raise ValueError(f"Manifest row {index + 1} is missing audio_path or ref_audio")

            normalized = {
                "utterance_id": row.get("utterance_id") or Path(str(audio_path)).stem or f"utt-{index:06d}",
                "audio_path": resolve_manifest_path(str(audio_path), manifest_base_dir=manifest_base_dir, repo_root=repo_root),
            }
            text = row.get("ref_text")
            if text is None:
                text = row.get("text")
            if text is not None:
                normalized["text"] = text
            if row.get("language"):
                normalized["language"] = row["language"]

            file.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    return output


def prepared_manifest_rows(
    *,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if manifest_name is None:
        raise ValueError("A manifest is required for inference")

    run_manifest = run_root / manifest_name
    manifest_base_dir = source_manifest.expanduser().resolve().parent if source_manifest is not None else None
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(run_manifest)):
        ref_audio = row.get("ref_audio") or row.get("audio_path")
        if not ref_audio:
            raise ValueError(f"Manifest row {index + 1} is missing ref_audio or audio_path")

        utterance_id = row.get("utterance_id") or Path(str(ref_audio)).stem or f"utt-{index:06d}"
        gen_text = row.get("gen_text")
        if gen_text is None:
            gen_text = row.get("target_text")
        if gen_text is None:
            gen_text = row.get("text")
        if gen_text is None:
            raise ValueError(f"Manifest row {index + 1} is missing gen_text, target_text, or text")

        rows.append(
            {
                **row,
                "utterance_id": utterance_id,
                "ref_audio": resolve_manifest_path(str(ref_audio), manifest_base_dir=manifest_base_dir, repo_root=repo_root),
                "ref_text": row.get("ref_text") or "",
                "gen_text": gen_text,
            }
        )
    return rows


def write_simple_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write the run config snapshot as YAML for human inspection."""

    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False), encoding="utf-8")


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    """Build a subprocess environment that exposes ``src/`` on ``PYTHONPATH``."""

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return env


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def output_file_metadata(path: Path, *, min_bytes: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": None,
        "sha256": "",
        "ok": False,
    }
    if not path.exists():
        metadata["error_message"] = f"Missing generated wav: {path}"
        return metadata

    size = path.stat().st_size
    metadata["size_bytes"] = size
    metadata["sha256"] = file_sha256(path)
    metadata["ok"] = size >= min_bytes
    if not metadata["ok"]:
        metadata["error_message"] = f"Generated wav is too small: {path} ({size} bytes < {min_bytes})"
    return metadata


def build_run_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_root: Path,
    manifest_name: str | None,
    status: str,
    created_at: str,
    updated_at: str,
    git: dict[str, Any],
    stages: list[dict[str, Any]] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    ensure_run_option_defaults(args)
    modes = args.modes or DEFAULT_MODES
    checkpoint = args.checkpoint or "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    scaffold_outputs = [
        "run.json",
        "config.yaml",
        "posterior_cache/",
        "generated/",
        "predictions/",
        "metrics/",
        "logs/",
    ]
    if manifest_name:
        scaffold_outputs.insert(2, "manifest.jsonl")
    if stages is None:
        stages = [
            {
                "name": "scaffold",
                "status": "succeeded" if status in {"completed", "running"} else status,
                "outputs": scaffold_outputs,
            }
        ]

    if args.ctc_model:
        posterior_source = args.ctc_model
    elif args.skip_whisper:
        posterior_source = "manifest_text"
    else:
        posterior_source = args.whisper_model or args.posterior_asr

    payload = {
        "run_id": run_id,
        "project": args.project,
        "experiment": args.experiment,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "modes": modes,
        "manifest": manifest_name,
        "posterior_file": "posterior_cache/run.posterior.jsonl",
        "artifact_root": str(run_root),
        "model": {
            "name": args.model,
            "checkpoint_id": args.checkpoint_id or None,
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint,
            "checkpoint_hash": args.checkpoint_hash or "",
            "vocoder": args.vocoder,
        },
        "generation": {
            "nfe_step": args.nfe_step,
            "cfg_strength": args.cfg_strength,
            "sway_sampling_coef": args.sway_sampling_coef,
            "speed": args.speed,
            "seed": args.seed,
            "run_inference": args.run_inference,
            "inference_dry_run": args.inference_dry_run,
            "hard_ref_text_source": args.hard_ref_text_source,
            "min_wav_bytes": args.min_wav_bytes,
            "posterior_encoder_ckpt": args.posterior_encoder_ckpt,
            "ssl_cache": args.ssl_cache,
        },
        "posterior": {
            "asr": posterior_source,
            "run_extraction": args.run_posterior_extraction,
            "whisper_model": args.whisper_model,
            "skip_whisper": args.skip_whisper,
            "ctc_model": args.ctc_model or None,
            "language": args.language,
            "ctc_top_k": args.ctc_top_k,
        },
        "evaluation": {
            "eval_asr": args.eval_asr,
            "run_prediction": args.run_prediction,
            "prediction_dry_run": args.prediction_dry_run,
            "run_metrics": args.run_metrics,
            "metrics_dry_run": args.metrics_dry_run,
            "normalizer": args.metrics_normalizer,
            "run_audio_metrics": args.run_audio_metrics,
            "audio_metrics_dry_run": args.audio_metrics_dry_run,
            "run_speaker_similarity": args.run_speaker_similarity,
            "speaker_checkpoint": args.speaker_checkpoint,
            "speaker_device": args.speaker_device,
            "speaker_feat_type": args.speaker_feat_type,
            "run_utmos": args.run_utmos,
            "utmos_device": args.utmos_device,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "significance_baseline": args.significance_baseline,
        },
        "git": git,
        "stages": stages,
    }
    if error_message:
        payload["error_message"] = error_message
    return payload


def write_run_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_root: Path,
    manifest_name: str | None,
    status: str,
    created_at: str,
    git: dict[str, Any],
    stages: list[dict[str, Any]] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload = build_run_payload(
        args=args,
        run_id=run_id,
        run_root=run_root,
        manifest_name=manifest_name,
        status=status,
        created_at=created_at,
        updated_at=utc_or_local_now().isoformat(timespec="seconds"),
        git=git,
        stages=stages,
        error_message=error_message,
    )
    write_json(run_root / "run.json", payload)
    write_simple_yaml(run_root / "config.yaml", payload)
    return payload


def run_posterior_extraction_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
) -> dict[str, Any]:
    posterior_input = prepare_posterior_input_manifest(
        run_root=run_root,
        manifest_name=manifest_name,
        source_manifest=source_manifest,
        repo_root=repo_root,
    )
    output = run_root / "posterior_cache" / "run.posterior.jsonl"
    shard_dir = run_root / "posterior_cache" / "posterior_npz"
    log_path = run_root / "logs" / "extract_posterior.log"

    command = [
        sys.executable,
        "src/f5_tts/scripts/extract_asr_posterior.py",
        "--manifest",
        str(posterior_input),
        "--output",
        str(output),
        "--shard_dir",
        str(shard_dir),
        "--language",
        args.language,
        "--whisper_model",
        args.whisper_model,
        "--top_k",
        str(args.ctc_top_k),
        "--torch_dtype",
        args.torch_dtype,
    ]
    if args.skip_whisper:
        command.append("--skip_whisper")
    if args.ctc_model:
        command.extend(["--ctc_model", args.ctc_model])
    if args.asr_device:
        command.extend(["--device", args.asr_device])

    completed = subprocess.run(command, cwd=repo_root, check=False, capture_output=True, text=True, env=_subprocess_env(repo_root))
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                "STDOUT:",
                completed.stdout,
                "STDERR:",
                completed.stderr,
                f"EXIT_CODE: {completed.returncode}",
            ]
        ),
        encoding="utf-8",
    )

    stage = {
        "name": "posterior_extraction",
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "command": command,
        "inputs": [str(posterior_input.relative_to(run_root))],
        "outputs": [
            "posterior_cache/run.posterior.jsonl",
            "posterior_cache/posterior_npz/",
            "logs/extract_posterior.log",
        ],
        "log": "logs/extract_posterior.log",
        "exit_code": completed.returncode,
    }
    if completed.returncode != 0:
        stage["error_message"] = completed.stderr.strip() or completed.stdout.strip()
    return stage


def mode_to_cli_ref_text_mode(mode: str) -> str:
    if mode == "oracle":
        return "hard"
    if mode in {"hard", "length_only", "soft_ctc", "posterior_encoder", "hybrid"}:
        return mode
    raise ValueError(f"Inference mode is not wired yet: {mode}")


def ref_text_for_mode(row: dict[str, Any], mode: str, *, hard_ref_text_source: str) -> str:
    if mode == "oracle":
        return row.get("ref_text") or ""
    if mode == "hard" and hard_ref_text_source == "manifest":
        return row.get("ref_text") or ""
    return ""


def build_inference_command(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    mode: str,
    row: dict[str, Any],
) -> list[str]:
    output_dir = run_root / "generated" / mode
    output_file = f"{row['utterance_id']}.wav"
    command = [
        sys.executable,
        "src/f5_tts/infer/infer_cli.py",
        "--model",
        args.model,
        "--ref_audio",
        row["ref_audio"],
        "--ref_text",
        ref_text_for_mode(row, mode, hard_ref_text_source=args.hard_ref_text_source),
        "--gen_text",
        str(row["gen_text"]),
        "--ref_text_mode",
        mode_to_cli_ref_text_mode(mode),
        "--output_dir",
        str(output_dir),
        "--output_file",
        output_file,
        "--nfe_step",
        str(args.nfe_step),
        "--cfg_strength",
        str(args.cfg_strength),
        "--sway_sampling_coef",
        str(args.sway_sampling_coef),
        "--speed",
        str(args.speed),
        "--seed",
        str(args.seed),
        "--vocoder_name",
        args.vocoder,
    ]
    if args.checkpoint:
        command.extend(["--ckpt_file", args.checkpoint])
    if args.vocab_file:
        command.extend(["--vocab_file", args.vocab_file])
    if args.infer_device:
        command.extend(["--device", args.infer_device])
    if mode in {"length_only", "soft_ctc", "posterior_encoder", "hybrid"}:
        posterior_file = run_root / "posterior_cache" / "run.posterior.jsonl"
        command.extend(["--posterior_file", str(posterior_file)])
    if mode == "posterior_encoder" and args.posterior_encoder_ckpt:
        command.extend(["--posterior_encoder_ckpt", args.posterior_encoder_ckpt])
    if mode == "hybrid" and args.ssl_cache:
        command.extend(["--ssl_cache", args.ssl_cache])
    return command


def build_batch_inference_command(
    *,
    args: argparse.Namespace,
    run_root: Path,
    mode: str,
    input_jsonl: Path,
    output_jsonl: Path,
) -> list[str]:
    command = [
        sys.executable,
        "src/f5_tts/infer/infer_batch_jsonl.py",
        "--input_jsonl",
        str(input_jsonl),
        "--output_jsonl",
        str(output_jsonl),
        "--run_root",
        str(run_root),
        "--mode",
        mode,
        "--model",
        args.model,
        "--ref_text_mode",
        mode_to_cli_ref_text_mode(mode),
        "--nfe_step",
        str(args.nfe_step),
        "--cfg_strength",
        str(args.cfg_strength),
        "--sway_sampling_coef",
        str(args.sway_sampling_coef),
        "--speed",
        str(args.speed),
        "--seed",
        str(args.seed),
        "--vocoder_name",
        args.vocoder,
        "--min_wav_bytes",
        str(args.min_wav_bytes),
    ]
    if args.checkpoint:
        command.extend(["--ckpt_file", args.checkpoint])
    if args.vocab_file:
        command.extend(["--vocab_file", args.vocab_file])
    if args.infer_device:
        command.extend(["--device", args.infer_device])
    if mode in {"length_only", "soft_ctc", "posterior_encoder", "hybrid"}:
        posterior_file = run_root / "posterior_cache" / "run.posterior.jsonl"
        command.extend(["--posterior_file", str(posterior_file)])
    if mode == "posterior_encoder" and args.posterior_encoder_ckpt:
        command.extend(["--posterior_encoder_ckpt", args.posterior_encoder_ckpt])
    if mode == "hybrid" and args.ssl_cache:
        command.extend(["--ssl_cache", args.ssl_cache])
    return command


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_event(run_root: Path, *, level: str, stage: str, message: str, mode: str | None = None) -> None:
    event = {"time": utc_or_local_now().isoformat(timespec="seconds"), "level": level, "stage": stage, "message": message}
    if mode:
        event["mode"] = mode
    events_path = run_root / "logs" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_inference_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
) -> dict[str, Any]:
    modes = args.modes or DEFAULT_MODES
    rows = prepared_manifest_rows(
        run_root=run_root,
        manifest_name=manifest_name,
        source_manifest=source_manifest,
        repo_root=repo_root,
    )
    posterior_file = run_root / "posterior_cache" / "run.posterior.jsonl"
    needs_posterior = any(mode in {"length_only", "soft_ctc", "posterior_encoder", "hybrid"} for mode in modes)
    if needs_posterior and not posterior_file.exists() and not args.inference_dry_run:
        raise FileNotFoundError(
            "Inference modes length_only/soft_ctc/hybrid require posterior_cache/run.posterior.jsonl. "
            "Run with --run_posterior_extraction first."
        )

    env = _subprocess_env(repo_root)
    batch_runner = repo_root / "src" / "f5_tts" / "infer" / "infer_batch_jsonl.py"

    mode_summaries: list[dict[str, Any]] = []
    overall_status = "succeeded"
    for mode in modes:
        mode_dir = run_root / "generated" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        commands_path = mode_dir / "commands.jsonl"
        batch_input_path = mode_dir / "batch_input.jsonl"
        log_path = run_root / "logs" / f"inference_{mode}.log"
        command_rows: list[dict[str, Any]] = []
        log_parts: list[str] = []
        mode_status = "succeeded"

        try:
            for row in rows:
                command = build_inference_command(args=args, repo_root=repo_root, run_root=run_root, mode=mode, row=row)
                output_path = mode_dir / f"{row['utterance_id']}.wav"
                command_row = {
                    "utterance_id": row["utterance_id"],
                    "mode": mode,
                    "command": command,
                    "output": str(output_path.relative_to(run_root)),
                    "dry_run": args.inference_dry_run,
                    "status": "planned" if args.inference_dry_run else "running",
                    "ref_audio": row["ref_audio"],
                    "ref_text": ref_text_for_mode(row, mode, hard_ref_text_source=args.hard_ref_text_source),
                    "gen_text": str(row["gen_text"]),
                    "seed": args.seed,
                    "checkpoint_id": args.checkpoint_id or "",
                    "checkpoint": args.checkpoint or "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
                    "checkpoint_hash": args.checkpoint_hash or "",
                    "started_at": None,
                    "finished_at": None,
                    "elapsed_sec": None,
                    "exit_code": None,
                    "output_exists": False,
                    "output_size_bytes": None,
                    "output_sha256": "",
                }

                command_rows.append(command_row)

            if args.inference_dry_run:
                pass
            elif batch_runner.exists():
                write_jsonl(batch_input_path, command_rows)
                write_jsonl(commands_path, command_rows)
                batch_command = build_batch_inference_command(
                    args=args,
                    run_root=run_root,
                    mode=mode,
                    input_jsonl=batch_input_path,
                    output_jsonl=commands_path,
                )
                completed = subprocess.run(
                    batch_command,
                    cwd=repo_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                log_parts.extend(
                    [
                        "$ " + " ".join(batch_command),
                        "",
                        "STDOUT:",
                        completed.stdout,
                        "STDERR:",
                        completed.stderr,
                        f"EXIT_CODE: {completed.returncode}",
                        "",
                    ]
                )
                if commands_path.exists():
                    command_rows = iter_jsonl(commands_path)
                if completed.returncode != 0:
                    mode_status = "failed"
                    overall_status = "failed"
                    message = completed.stderr.strip() or completed.stdout.strip() or "Batch inference failed"
                    if not command_rows:
                        command_rows = [
                            {
                                "utterance_id": rows[0]["utterance_id"] if rows else "",
                                "mode": mode,
                                "status": "failed",
                                "error_message": message,
                            }
                        ]
                    elif not any(row.get("status") == "failed" for row in command_rows):
                        command_rows[-1]["status"] = "failed"
                        command_rows[-1]["error_message"] = message
                for command_row in command_rows:
                    if command_row.get("status") != "succeeded":
                        continue
                    output_path = run_root / str(command_row.get("output") or "")
                    output_metadata = output_file_metadata(output_path, min_bytes=args.min_wav_bytes)
                    command_row["output_exists"] = output_metadata["exists"]
                    command_row["output_size_bytes"] = output_metadata["size_bytes"]
                    command_row["output_sha256"] = output_metadata["sha256"]
                    if not output_metadata["ok"]:
                        command_row["status"] = "failed"
                        command_row["error_message"] = output_metadata["error_message"]
                        mode_status = "failed"
                        overall_status = "failed"
                        log_parts.append(output_metadata["error_message"])
                        break
            else:
                legacy_rows: list[dict[str, Any]] = []
                for command_row in command_rows:
                    output_path = run_root / str(command_row["output"])
                    started_at = utc_or_local_now()
                    command_row["started_at"] = started_at.isoformat(timespec="seconds")
                    completed = subprocess.run(
                        command_row["command"],
                        cwd=repo_root,
                        check=False,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    finished_at = utc_or_local_now()
                    command_row["finished_at"] = finished_at.isoformat(timespec="seconds")
                    command_row["elapsed_sec"] = max((finished_at - started_at).total_seconds(), 0.0)
                    command_row["exit_code"] = completed.returncode
                    log_parts.extend(
                        [
                            "$ " + " ".join(command_row["command"]),
                            "",
                            "STDOUT:",
                            completed.stdout,
                            "STDERR:",
                            completed.stderr,
                            f"EXIT_CODE: {completed.returncode}",
                            "",
                        ]
                    )
                    if completed.returncode != 0:
                        command_row["status"] = "failed"
                        command_row["error_message"] = completed.stderr.strip() or completed.stdout.strip()
                        legacy_rows.append(command_row)
                        mode_status = "failed"
                        overall_status = "failed"
                        break

                    output_metadata = output_file_metadata(output_path, min_bytes=args.min_wav_bytes)
                    command_row["output_exists"] = output_metadata["exists"]
                    command_row["output_size_bytes"] = output_metadata["size_bytes"]
                    command_row["output_sha256"] = output_metadata["sha256"]
                    if output_metadata["ok"]:
                        command_row["status"] = "succeeded"
                        legacy_rows.append(command_row)
                    else:
                        command_row["status"] = "failed"
                        command_row["error_message"] = output_metadata["error_message"]
                        legacy_rows.append(command_row)
                        log_parts.append(output_metadata["error_message"])
                        mode_status = "failed"
                        overall_status = "failed"
                        break
                command_rows = legacy_rows
        except Exception as exc:
            mode_status = "failed"
            overall_status = "failed"
            log_parts.append(f"ERROR: {exc}")

        write_jsonl(commands_path, command_rows)
        log_path.write_text("\n".join(log_parts), encoding="utf-8")
        mode_summaries.append(
            {
                "mode": mode,
                "status": "planned" if args.inference_dry_run and mode_status == "succeeded" else mode_status,
                "num_utterances": len(command_rows),
                "commands": str(commands_path.relative_to(run_root)),
                "log": str(log_path.relative_to(run_root)),
                "output_dir": str(mode_dir.relative_to(run_root)),
            }
        )
        if overall_status == "failed":
            break

    if args.inference_dry_run and overall_status == "succeeded":
        stage_status = "planned"
    else:
        stage_status = overall_status
    return {
        "name": "inference",
        "status": stage_status,
        "dry_run": args.inference_dry_run,
        "modes": mode_summaries,
        "outputs": ["generated/<mode>/<utterance_id>.wav", "generated/<mode>/commands.jsonl", "logs/inference_<mode>.log"],
    }


def load_eval_asr_pipeline(args: argparse.Namespace):
    import torch
    from transformers import pipeline

    dtype = torch.float16 if args.eval_torch_dtype == "float16" else torch.float32
    return pipeline("automatic-speech-recognition", model=args.eval_asr, torch_dtype=dtype, device=args.eval_device)


def transcribe_generated_audio(asr_pipe, audio_path: Path, *, language: str | None) -> str:
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = asr_pipe(
        str(audio_path),
        chunk_length_s=30,
        batch_size=16,
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    return result["text"].strip()


def run_prediction_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
) -> dict[str, Any]:
    modes = args.modes or DEFAULT_MODES
    rows = prepared_manifest_rows(
        run_root=run_root,
        manifest_name=manifest_name,
        source_manifest=source_manifest,
        repo_root=repo_root,
    )
    asr_pipe = None if args.prediction_dry_run else load_eval_asr_pipeline(args)
    mode_summaries: list[dict[str, Any]] = []
    overall_status = "succeeded"

    for mode in modes:
        prediction_path = run_root / "predictions" / f"{mode}.jsonl"
        log_path = run_root / "logs" / f"prediction_{mode}.log"
        prediction_rows: list[dict[str, Any]] = []
        log_parts: list[str] = []
        mode_status = "succeeded"

        for row in rows:
            wav_path = run_root / "generated" / mode / f"{row['utterance_id']}.wav"
            prediction_row = {
                "utterance_id": row["utterance_id"],
                "mode": mode,
                "subset": row.get("subset"),
                "reference": row.get("text") or row.get("reference") or row.get("gen_text") or "",
                "target_text": row.get("gen_text"),
                "ref_text": row.get("ref_text"),
                "generated_audio": str(wav_path.relative_to(run_root)),
                "hypothesis": "",
                "eval_asr": args.eval_asr,
                "dry_run": args.prediction_dry_run,
            }

            if args.prediction_dry_run:
                prediction_row["status"] = "planned"
                prediction_rows.append(prediction_row)
                continue

            if not wav_path.exists():
                mode_status = "failed"
                overall_status = "failed"
                message = f"Missing generated wav: {wav_path}"
                log_parts.append(message)
                prediction_row["status"] = "missing_audio"
                prediction_row["error_message"] = message
                prediction_rows.append(prediction_row)
                break

            try:
                prediction_row["hypothesis"] = transcribe_generated_audio(
                    asr_pipe,
                    wav_path,
                    language=args.language,
                )
                prediction_row["status"] = "succeeded"
                prediction_rows.append(prediction_row)
            except Exception as exc:
                mode_status = "failed"
                overall_status = "failed"
                prediction_row["status"] = "failed"
                prediction_row["error_message"] = str(exc)
                prediction_rows.append(prediction_row)
                log_parts.append(f"ERROR {row['utterance_id']}: {exc}")
                break

        write_jsonl(prediction_path, prediction_rows)
        log_path.write_text("\n".join(log_parts), encoding="utf-8")
        mode_summaries.append(
            {
                "mode": mode,
                "status": "planned" if args.prediction_dry_run and mode_status == "succeeded" else mode_status,
                "num_utterances": len(prediction_rows),
                "predictions": str(prediction_path.relative_to(run_root)),
                "log": str(log_path.relative_to(run_root)),
            }
        )
        if overall_status == "failed":
            break

    if args.prediction_dry_run and overall_status == "succeeded":
        stage_status = "planned"
    else:
        stage_status = overall_status
    return {
        "name": "prediction",
        "status": stage_status,
        "dry_run": args.prediction_dry_run,
        "eval_asr": args.eval_asr,
        "modes": mode_summaries,
        "outputs": ["predictions/<mode>.jsonl", "logs/prediction_<mode>.log"],
    }


def run_audio_metrics_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    manifest_name: str | None,
    source_manifest: Path | None,
) -> dict[str, Any]:
    if manifest_name is None:
        raise ValueError("A manifest is required for audio metrics")
    if args.run_speaker_similarity and not args.speaker_checkpoint and not args.audio_metrics_dry_run:
        raise ValueError("Speaker similarity requires --speaker_checkpoint")

    modes = args.modes or DEFAULT_MODES
    manifest_path = run_root / manifest_name
    manifest_base_dir = source_manifest.expanduser().resolve().parent if source_manifest is not None else None
    mode_summaries: list[dict[str, Any]] = []
    overall_status = "succeeded"
    speaker_scorer = None
    utmos_scorer = None
    evaluate_audio_metrics = None
    manifest_rows: list[dict[str, Any]] | None = None

    if not args.audio_metrics_dry_run:
        from f5_tts.eval.eval_audio_metrics import (
            SpeakerSimilarityScorer,
            UtmosScorer,
            evaluate_audio_metrics as evaluate_audio_metrics_fn,
        )

        evaluate_audio_metrics = evaluate_audio_metrics_fn
        manifest_rows = iter_jsonl(manifest_path)

        if args.run_speaker_similarity:
            speaker_scorer = SpeakerSimilarityScorer(
                checkpoint=args.speaker_checkpoint,
                device=args.speaker_device,
                feat_type=args.speaker_feat_type,
            )
        if args.run_utmos:
            utmos_scorer = UtmosScorer(device=args.utmos_device)

    for mode in modes:
        generation_metadata_path = run_root / "generated" / mode / "commands.jsonl"
        output_path = run_root / "metrics" / f"{mode}.audio_metrics.json"
        per_utterance_path = run_root / "metrics" / f"{mode}.audio_metrics.jsonl"
        log_path = run_root / "logs" / f"audio_metrics_{mode}.log"
        command = [
            sys.executable,
            "src/f5_tts/eval/eval_audio_metrics.py",
            "--manifest",
            str(manifest_path),
            "--generation_metadata",
            str(generation_metadata_path),
            "--run_root",
            str(run_root),
            "--output",
            str(output_path),
            "--per_utterance_output",
            str(per_utterance_path),
            "--mode",
            mode,
            "--repo_root",
            str(repo_root),
        ]
        if manifest_base_dir is not None:
            command.extend(["--manifest_base_dir", str(manifest_base_dir)])
        if args.run_speaker_similarity:
            command.append("--run_speaker_similarity")
            command.extend(["--speaker_checkpoint", args.speaker_checkpoint])
            command.extend(["--speaker_feat_type", args.speaker_feat_type])
            if args.speaker_device:
                command.extend(["--speaker_device", args.speaker_device])
        if args.run_utmos:
            command.append("--run_utmos")
            if args.utmos_device:
                command.extend(["--utmos_device", args.utmos_device])

        if args.audio_metrics_dry_run:
            metrics = {
                "mode": mode,
                "status": "planned",
                "num_utterances": 0,
                "num_audio": 0,
                "audio_metric_coverage": None,
                "generated_audio_duration_sec_mean": None,
                "rtf_mean": None,
                "speaker_similarity_mean": None,
                "spk_sim_mean": None,
                "utmos_mean": None,
            }
            write_json(output_path, metrics)
            per_utterance_path.write_text("", encoding="utf-8")
            log_path.write_text("$ " + " ".join(command) + "\nDRY_RUN: true\n", encoding="utf-8")
            mode_status = "planned"
        elif not generation_metadata_path.exists():
            metrics = {
                "mode": mode,
                "status": "failed",
                "error_message": f"Missing generation metadata: {generation_metadata_path}",
                "num_utterances": 0,
                "num_audio": 0,
                "audio_metric_coverage": None,
                "generated_audio_duration_sec_mean": None,
                "rtf_mean": None,
                "speaker_similarity_mean": None,
                "spk_sim_mean": None,
                "utmos_mean": None,
            }
            write_json(output_path, metrics)
            per_utterance_path.write_text("", encoding="utf-8")
            log_path.write_text(metrics["error_message"], encoding="utf-8")
            mode_status = "failed"
            overall_status = "failed"
        else:
            try:
                if evaluate_audio_metrics is None or manifest_rows is None:
                    raise RuntimeError("Audio metric evaluator was not initialized")
                metrics, rows = evaluate_audio_metrics(
                    manifest_rows=manifest_rows,
                    generation_rows=iter_jsonl(generation_metadata_path),
                    run_root=run_root,
                    mode=mode,
                    repo_root=repo_root,
                    manifest_base_dir=manifest_base_dir,
                    speaker_scorer=speaker_scorer,
                    utmos_scorer=utmos_scorer,
                )
                metrics["status"] = "succeeded"
                write_json(output_path, metrics)
                write_jsonl(per_utterance_path, rows)
                log_path.write_text(
                    "\n".join(
                        [
                            "$ " + " ".join(command),
                            "DIRECT_CALL: true",
                            "MODEL_REUSE: one scorer instance is reused across all modes in this stage",
                            "",
                            "SUMMARY:",
                            json.dumps(metrics, indent=2, ensure_ascii=False),
                            "EXIT_CODE: 0",
                        ]
                    ),
                    encoding="utf-8",
                )
                mode_status = "succeeded"
            except Exception as exc:
                metrics = {
                    "mode": mode,
                    "status": "failed",
                    "error_message": str(exc),
                    "num_utterances": 0,
                    "num_audio": 0,
                    "audio_metric_coverage": None,
                    "generated_audio_duration_sec_mean": None,
                    "rtf_mean": None,
                    "speaker_similarity_mean": None,
                    "spk_sim_mean": None,
                    "utmos_mean": None,
                }
                write_json(output_path, metrics)
                per_utterance_path.write_text("", encoding="utf-8")
                log_path.write_text(
                    "\n".join(
                        [
                            "$ " + " ".join(command),
                            "DIRECT_CALL: true",
                            "",
                            "ERROR:",
                            str(exc),
                            "EXIT_CODE: 1",
                        ]
                    ),
                    encoding="utf-8",
                )
                mode_status = "failed"
                overall_status = "failed"

        mode_summaries.append(
            {
                "mode": mode,
                "status": mode_status,
                "metrics": str(output_path.relative_to(run_root)),
                "per_utterance": str(per_utterance_path.relative_to(run_root)),
                "log": str(log_path.relative_to(run_root)),
            }
        )
        if overall_status == "failed":
            break

    if args.audio_metrics_dry_run and overall_status == "succeeded":
        stage_status = "planned"
    else:
        stage_status = overall_status
    return {
        "name": "audio_metrics",
        "status": stage_status,
        "dry_run": args.audio_metrics_dry_run,
        "modes": mode_summaries,
        "outputs": [
            "metrics/<mode>.audio_metrics.json",
            "metrics/<mode>.audio_metrics.jsonl",
            "logs/audio_metrics_<mode>.log",
        ],
    }


def numeric_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": metrics.get("mode"),
        "num_utterances": metrics.get("num_utterances", 0),
        "wer": metrics.get("wer"),
        "cer": metrics.get("cer"),
        "prediction_coverage": metrics.get("prediction_coverage"),
        "normalizer": metrics.get("normalizer"),
        "wer_substitutions": metrics.get("wer_substitutions", 0),
        "wer_deletions": metrics.get("wer_deletions", 0),
        "wer_insertions": metrics.get("wer_insertions", 0),
        "wer_reference_length": metrics.get("wer_reference_length", 0),
        "wer_deletion_rate": metrics.get("wer_deletion_rate"),
        "wer_insertion_rate": metrics.get("wer_insertion_rate"),
        "wer_substitution_rate": metrics.get("wer_substitution_rate"),
        "cer_substitutions": metrics.get("cer_substitutions", 0),
        "cer_deletions": metrics.get("cer_deletions", 0),
        "cer_insertions": metrics.get("cer_insertions", 0),
        "cer_reference_length": metrics.get("cer_reference_length", 0),
        "generation_elapsed_sec_mean": metrics.get("generation_elapsed_sec_mean"),
        "generated_audio_duration_sec_mean": metrics.get("generated_audio_duration_sec_mean"),
        "rtf_mean": metrics.get("rtf_mean"),
        "speaker_similarity_mean": metrics.get("speaker_similarity_mean"),
        "spk_sim_mean": metrics.get("spk_sim_mean"),
        "utmos_mean": metrics.get("utmos_mean"),
    }


def write_summary_files(run_root: Path, metrics_by_mode: list[dict[str, Any]]) -> None:
    summary_json = run_root / "metrics" / "summary.json"
    summary_csv = run_root / "metrics" / "summary.csv"
    summary_payload = {
        "modes": metrics_by_mode,
        "created_at": utc_or_local_now().isoformat(timespec="seconds"),
    }
    write_json(summary_json, summary_payload)

    fieldnames = [
        "mode",
        "num_utterances",
        "wer",
        "cer",
        "prediction_coverage",
        "normalizer",
        "wer_substitutions",
        "wer_deletions",
        "wer_insertions",
        "wer_reference_length",
        "wer_deletion_rate",
        "wer_insertion_rate",
        "wer_substitution_rate",
        "cer_substitutions",
        "cer_deletions",
        "cer_insertions",
        "cer_reference_length",
        "generation_elapsed_sec_mean",
        "generated_audio_duration_sec_mean",
        "rtf_mean",
        "speaker_similarity_mean",
        "spk_sim_mean",
        "utmos_mean",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_by_mode:
            writer.writerow(numeric_metric_row(metrics))


def write_significance_files(
    *,
    run_root: Path,
    rows: list[dict[str, Any]],
    modes: list[str],
    baseline: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    significance_path = run_root / "metrics" / "significance.json"
    if baseline not in modes or not rows:
        payload = {
            "created_at": utc_or_local_now().isoformat(timespec="seconds"),
            "baseline": baseline,
            "samples": samples,
            "seed": seed,
            "comparisons": [],
        }
        write_json(significance_path, payload)
        return []

    comparisons: list[dict[str, Any]] = []
    metrics = ["wer", "cer", "wer_deletion_rate"]
    for candidate in modes:
        if candidate == baseline:
            continue
        for metric in metrics:
            comparisons.append(
                paired_bootstrap_result(
                    rows,
                    baseline=baseline,
                    candidate=candidate,
                    metric=metric,
                    samples=samples,
                    seed=seed,
                )
            )

    comparisons = apply_multiple_comparison_corrections(comparisons)
    payload = {
        "created_at": utc_or_local_now().isoformat(timespec="seconds"),
        "baseline": baseline,
        "samples": samples,
        "seed": seed,
        "correction_scope": "within_run_by_metric",
        "comparisons": comparisons,
    }
    write_json(significance_path, payload)
    return comparisons


def attach_significance_to_metrics(metrics_by_mode: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    by_candidate: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        candidate = str(comparison.get("candidate") or "")
        metric = str(comparison.get("metric") or "")
        if not candidate or not metric:
            continue
        by_candidate.setdefault(candidate, {})[metric] = comparison

    for metrics in metrics_by_mode:
        mode = str(metrics.get("mode") or "")
        if mode in by_candidate:
            metrics["significance"] = by_candidate[mode]


def run_metrics_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    manifest_name: str | None,
) -> dict[str, Any]:
    if manifest_name is None:
        raise ValueError("A manifest is required for metrics")

    modes = args.modes or DEFAULT_MODES
    manifest_path = run_root / manifest_name
    posterior_file = run_root / "posterior_cache" / "run.posterior.jsonl"
    env = _subprocess_env(repo_root)

    metrics_by_mode: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    overall_status = "succeeded"
    for mode in modes:
        prediction_path = run_root / "predictions" / f"{mode}.jsonl"
        output_path = run_root / "metrics" / f"{mode}.metrics.json"
        per_utterance_path = run_root / "metrics" / f"{mode}.per_utterance.jsonl"
        generation_metadata_path = run_root / "generated" / mode / "commands.jsonl"
        audio_metrics_path = run_root / "metrics" / f"{mode}.audio_metrics.jsonl"
        log_path = run_root / "logs" / f"metrics_{mode}.log"
        command = [
            sys.executable,
            "src/f5_tts/eval/eval_posterior_f5.py",
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(output_path),
            "--per_utterance_output",
            str(per_utterance_path),
            "--mode",
            mode,
            "--normalizer",
            args.metrics_normalizer,
            "--posterior_file",
            str(posterior_file) if posterior_file.exists() else "",
        ]
        if generation_metadata_path.exists():
            command.extend(["--generation_metadata", str(generation_metadata_path)])
        if audio_metrics_path.exists():
            command.extend(["--audio_metrics", str(audio_metrics_path)])

        if args.metrics_dry_run:
            metrics = {
                "mode": mode,
                "status": "planned",
                "num_utterances": 0,
                "wer": None,
                "cer": None,
                "posterior_file": str(posterior_file) if posterior_file.exists() else "",
                "normalizer": args.metrics_normalizer,
                "generated_audio_duration_sec_mean": None,
                "rtf_mean": None,
                "speaker_similarity_mean": None,
                "spk_sim_mean": None,
                "utmos_mean": None,
            }
            write_json(output_path, metrics)
            per_utterance_path.write_text("", encoding="utf-8")
            log_path.write_text("$ " + " ".join(command) + "\nDRY_RUN: true\n", encoding="utf-8")
            mode_status = "planned"
        else:
            if not prediction_path.exists():
                metrics = {
                    "mode": mode,
                    "status": "failed",
                    "error_message": f"Missing predictions: {prediction_path}",
                    "num_utterances": 0,
                    "wer": None,
                    "cer": None,
                    "normalizer": args.metrics_normalizer,
                    "generated_audio_duration_sec_mean": None,
                    "rtf_mean": None,
                    "speaker_similarity_mean": None,
                    "spk_sim_mean": None,
                    "utmos_mean": None,
                }
                write_json(output_path, metrics)
                log_path.write_text(metrics["error_message"], encoding="utf-8")
                mode_status = "failed"
                overall_status = "failed"
            else:
                completed = subprocess.run(command, cwd=repo_root, check=False, capture_output=True, text=True, env=env)
                log_path.write_text(
                    "\n".join(
                        [
                            "$ " + " ".join(command),
                            "",
                            "STDOUT:",
                            completed.stdout,
                            "STDERR:",
                            completed.stderr,
                            f"EXIT_CODE: {completed.returncode}",
                        ]
                    ),
                    encoding="utf-8",
                )
                if completed.returncode == 0:
                    metrics = json.loads(output_path.read_text(encoding="utf-8"))
                    metrics["status"] = "succeeded"
                    write_json(output_path, metrics)
                    mode_status = "succeeded"
                else:
                    metrics = {
                        "mode": mode,
                        "status": "failed",
                        "error_message": completed.stderr.strip() or completed.stdout.strip(),
                        "num_utterances": 0,
                        "wer": None,
                        "cer": None,
                        "normalizer": args.metrics_normalizer,
                        "generated_audio_duration_sec_mean": None,
                        "rtf_mean": None,
                        "speaker_similarity_mean": None,
                        "spk_sim_mean": None,
                        "utmos_mean": None,
                    }
                    write_json(output_path, metrics)
                    mode_status = "failed"
                    overall_status = "failed"

        metrics_by_mode.append(metrics)
        mode_summaries.append(
            {
                "mode": mode,
                "status": mode_status,
                "metrics": str(output_path.relative_to(run_root)),
                "per_utterance": str(per_utterance_path.relative_to(run_root)),
                "log": str(log_path.relative_to(run_root)),
            }
        )
        if overall_status == "failed":
            break

    combined_per_utterance = []
    for mode in modes:
        per_mode_path = run_root / "metrics" / f"{mode}.per_utterance.jsonl"
        if per_mode_path.exists():
            combined_per_utterance.extend(iter_jsonl(per_mode_path))
    write_jsonl(run_root / "metrics" / "per_utterance.jsonl", combined_per_utterance)
    comparisons = write_significance_files(
        run_root=run_root,
        rows=[] if args.metrics_dry_run else combined_per_utterance,
        modes=modes,
        baseline=args.significance_baseline,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    attach_significance_to_metrics(metrics_by_mode, comparisons)
    write_summary_files(run_root, metrics_by_mode)
    if args.metrics_dry_run and overall_status == "succeeded":
        stage_status = "planned"
    else:
        stage_status = overall_status
    return {
        "name": "metrics",
        "status": stage_status,
        "dry_run": args.metrics_dry_run,
        "modes": mode_summaries,
        "outputs": [
            "metrics/<mode>.metrics.json",
            "metrics/<mode>.per_utterance.jsonl",
            "metrics/per_utterance.jsonl",
            "metrics/summary.json",
            "metrics/summary.csv",
            "metrics/significance.json",
            "logs/metrics_<mode>.log",
        ],
    }


def _execute_stage(
    *,
    name: str,
    runner,
    runner_kwargs: dict[str, Any],
    running_payload: dict[str, Any],
    stages: list[dict[str, Any]],
    write_payload,
    run_root: Path,
    failure_message: str,
    fail_status: str = "failed",
) -> list[dict[str, Any]]:
    """Append-event + write-running-payload + run + handle failure for one stage."""

    append_event(run_root, level="info", stage=name, message="started")
    write_payload(status="running", stages=[*stages, running_payload])
    try:
        stage_payload = runner(**runner_kwargs)
    except Exception as exc:
        message = str(exc) or failure_message
        stage_payload = {**running_payload, "status": "failed", "error_message": message}
        stages = [*stages, stage_payload]
        append_event(run_root, level="error", stage=name, message=message)
        write_payload(status="failed", stages=stages, error_message=message)
        raise RuntimeError(message) from exc

    stages = [*stages, stage_payload]

    if stage_payload["status"] == fail_status or (name == "posterior_extraction" and stage_payload["status"] != "succeeded"):
        message = stage_payload.get("error_message") or failure_message
        append_event(run_root, level="error", stage=name, message=stage_payload.get("error_message") or "failed")
        write_payload(status="failed", stages=stages, error_message=stage_payload.get("error_message") or failure_message)
        if name == "posterior_extraction":
            raise RuntimeError(f"Posterior extraction failed; see {run_root / stage_payload['log']}")
        raise RuntimeError(failure_message)

    append_event(run_root, level="info", stage=name, message=stage_payload["status"])
    return stages


def scaffold_run(args: argparse.Namespace, *, repo_root: Path | None = None) -> Path:
    ensure_run_option_defaults(args)
    repo_root = repo_root or repository_root()
    modes = args.modes or DEFAULT_MODES
    run_id = args.run_id or generate_run_id()
    run_root = (repo_root / args.artifact_root / run_id).resolve()
    source_manifest = Path(args.manifest).expanduser() if args.manifest else None

    if run_root.exists() and args.fail_if_exists:
        raise FileExistsError(f"Run directory already exists: {run_root}")

    run_root.mkdir(parents=True, exist_ok=True)
    ensure_run_dirs(run_root, modes)

    now = utc_or_local_now().isoformat(timespec="seconds")
    manifest_name = snapshot_manifest(args.manifest, run_root)
    git = git_snapshot(repo_root)

    def write_payload(*, status: str, stages: list[dict[str, Any]], error_message: str | None = None) -> None:
        write_run_payload(
            args=args,
            run_id=run_id,
            run_root=run_root,
            manifest_name=manifest_name,
            status=status,
            created_at=now,
            git=git,
            stages=stages,
            error_message=error_message,
        )

    scaffold_stage = {
        "name": "scaffold",
        "status": "succeeded",
        "outputs": [
            "run.json",
            "config.yaml",
            *([] if manifest_name is None else ["manifest.jsonl"]),
            "posterior_cache/",
            "generated/",
            "predictions/",
            "metrics/",
            "logs/",
        ],
    }
    stages = [scaffold_stage]
    append_event(run_root, level="info", stage="scaffold", message="succeeded")
    write_payload(status="running", stages=stages)

    base_kwargs = {"args": args, "repo_root": repo_root, "run_root": run_root, "manifest_name": manifest_name}
    stage_specs = [
        (
            args.run_posterior_extraction,
            "posterior_extraction",
            run_posterior_extraction_stage,
            {**base_kwargs, "source_manifest": source_manifest},
            {"outputs": ["posterior_cache/run.posterior.jsonl", "posterior_cache/posterior_npz/"]},
            "Posterior extraction failed",
        ),
        (
            args.run_inference,
            "inference",
            run_inference_stage,
            {**base_kwargs, "source_manifest": source_manifest},
            {"outputs": ["generated/<mode>/<utterance_id>.wav", "generated/<mode>/commands.jsonl"], "dry_run": args.inference_dry_run},
            "Inference failed; inspect logs/inference_<mode>.log",
        ),
        (
            args.run_audio_metrics,
            "audio_metrics",
            run_audio_metrics_stage,
            {**base_kwargs, "source_manifest": source_manifest},
            {
                "outputs": ["metrics/<mode>.audio_metrics.json", "metrics/<mode>.audio_metrics.jsonl"],
                "dry_run": args.audio_metrics_dry_run,
            },
            "Audio metrics failed; inspect logs/audio_metrics_<mode>.log",
        ),
        (
            args.run_prediction,
            "prediction",
            run_prediction_stage,
            {**base_kwargs, "source_manifest": source_manifest},
            {"outputs": ["predictions/<mode>.jsonl"], "dry_run": args.prediction_dry_run},
            "Prediction failed; inspect logs/prediction_<mode>.log",
        ),
        (
            args.run_metrics,
            "metrics",
            run_metrics_stage,
            base_kwargs,
            {"outputs": ["metrics/<mode>.metrics.json", "metrics/summary.json", "metrics/summary.csv"], "dry_run": args.metrics_dry_run},
            "Metrics failed; inspect logs/metrics_<mode>.log",
        ),
    ]

    for gate, name, runner, runner_kwargs, running_extras, failure_message in stage_specs:
        if not gate:
            continue
        running_payload = {"name": name, "status": "running", **running_extras}
        stages = _execute_stage(
            name=name,
            runner=runner,
            runner_kwargs=runner_kwargs,
            running_payload=running_payload,
            stages=stages,
            write_payload=write_payload,
            run_root=run_root,
            failure_message=failure_message,
        )

    write_payload(status="completed", stages=stages)
    append_event(run_root, level="info", stage="run", message="completed")
    return run_root


def main() -> None:
    args = parse_args()
    try:
        run_root = scaffold_run(args)
    except Exception as exc:
        print(f"Failed to create run scaffold: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(run_root)


if __name__ == "__main__":
    main()
