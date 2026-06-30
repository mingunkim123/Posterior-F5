"""Filesystem-backed run store for the Posterior-F5 MLOps API."""

from __future__ import annotations

import json
import csv
import hashlib
import re
import sys
import os
import signal
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_artifact_root(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    return root / "mlops_artifacts" / "runs"


def checkpoint_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    return root / "platform" / "config" / "checkpoints.yaml"


def dataset_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    return root / "platform" / "config" / "datasets.yaml"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checkpoint_file_hash(path: str, *, repo_root: Path | None = None) -> str:
    if not path or path.startswith(("hf://", "http://", "https://", "s3://")):
        return ""

    root = repo_root or repository_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists() or not candidate.is_file():
        return ""

    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_checkpoint_registry(*, repo_root: Path | None = None) -> dict[str, Any]:
    path = checkpoint_registry_path(repo_root)
    if not path.exists():
        return {"checkpoints": []}

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    checkpoints = []
    for checkpoint_id, entry in (payload.get("checkpoints") or {}).items():
        if not isinstance(entry, dict):
            continue
        checkpoint_path = str(entry.get("path") or "")
        checkpoint_hash = str(entry.get("checkpoint_hash") or entry.get("sha256") or "")
        if not checkpoint_hash:
            checkpoint_hash = checkpoint_file_hash(checkpoint_path, repo_root=repo_root)
        checkpoints.append(
            {
                "id": str(checkpoint_id),
                "model": str(entry.get("model") or "F5TTS_v1_Base"),
                "path": checkpoint_path,
                "vocoder": str(entry.get("vocoder") or "vocos"),
                "checkpoint_hash": checkpoint_hash,
                "notes": str(entry.get("notes") or ""),
                "dataset": entry.get("dataset"),
                "git_commit": entry.get("git_commit"),
            }
        )
    return {"checkpoints": checkpoints}


def checkpoint_by_id(checkpoint_id: str, *, repo_root: Path | None = None) -> dict[str, Any] | None:
    for checkpoint in load_checkpoint_registry(repo_root=repo_root)["checkpoints"]:
        if checkpoint["id"] == checkpoint_id:
            return checkpoint
    return None


def resolve_checkpoint_payload(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    payload = dict(payload)
    checkpoint_id = payload.get("checkpoint_id")
    if checkpoint_id:
        checkpoint = checkpoint_by_id(str(checkpoint_id), repo_root=repo_root)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint_id: {checkpoint_id}")
        payload["checkpoint_id"] = checkpoint["id"]
        payload["model"] = checkpoint["model"]
        payload["checkpoint"] = checkpoint["path"]
        payload["vocoder"] = checkpoint["vocoder"]
        payload["checkpoint_hash"] = checkpoint["checkpoint_hash"]
        return payload

    checkpoint_path = str(payload.get("checkpoint") or "")
    payload["checkpoint_hash"] = str(payload.get("checkpoint_hash") or checkpoint_file_hash(checkpoint_path, repo_root=repo_root))
    return payload


def load_dataset_registry(*, repo_root: Path | None = None) -> dict[str, Any]:
    path = dataset_registry_path(repo_root)
    if not path.exists():
        return {"datasets": []}

    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = []
    for dataset_id, entry in (payload.get("datasets") or {}).items():
        if not isinstance(entry, dict):
            continue
        datasets.append(
            {
                "id": str(dataset_id),
                "manifest": str(entry.get("manifest") or ""),
                "language": str(entry.get("language") or "en"),
                "num_utterances": entry.get("num_utterances"),
                "purpose": str(entry.get("purpose") or ""),
                "subsets": entry.get("subsets") or [],
                "notes": str(entry.get("notes") or ""),
            }
        )
    return {"datasets": datasets}


def dataset_by_id(dataset_id: str, *, repo_root: Path | None = None) -> dict[str, Any] | None:
    for dataset in load_dataset_registry(repo_root=repo_root)["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    return None


def resolve_dataset_payload(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    payload = dict(payload)
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        return payload
    dataset = dataset_by_id(str(dataset_id), repo_root=repo_root)
    if dataset is None:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    payload["dataset_id"] = dataset["id"]
    payload["manifest"] = dataset["manifest"]
    payload["language"] = dataset["language"]
    return payload


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
            "created_at": job.get("queued_at") or job.get("started_at"),
            "updated_at": job.get("finished_at") or job.get("started_at") or job.get("queued_at"),
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_event(
    run_id: str,
    *,
    level: str,
    stage: str,
    message: str,
    mode: str | None = None,
    artifact_root: Path | None = None,
) -> None:
    root = run_path(run_id, artifact_root=artifact_root)
    events_path = root / "logs" / "events.jsonl"
    event = {"time": timestamp(), "level": level, "stage": stage, "message": message}
    if mode:
        event["mode"] = mode
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_run_events(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    events = load_jsonl(run_path(run_id, artifact_root=artifact_root) / "logs" / "events.jsonl")
    return {"run_id": run_id, "events": events}


def word_diff(reference: str | None, hypothesis: str | None) -> list[dict[str, str]]:
    reference_words = str(reference or "").split()
    hypothesis_words = str(hypothesis or "").split()
    diff: list[dict[str, str]] = []
    matcher = SequenceMatcher(a=reference_words, b=hypothesis_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            diff.extend({"op": "equal", "text": word} for word in hypothesis_words[j1:j2])
        elif tag == "insert":
            diff.extend({"op": "insert", "text": word} for word in hypothesis_words[j1:j2])
        elif tag == "delete":
            diff.extend({"op": "delete", "text": word} for word in reference_words[i1:i2])
        else:
            diff.extend({"op": "delete", "text": word} for word in reference_words[i1:i2])
            diff.extend({"op": "insert", "text": word} for word in hypothesis_words[j1:j2])
    return diff


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def experiment_name(run: dict[str, Any]) -> str:
    return str(run.get("experiment") or "default")


def run_checkpoint_id(run: dict[str, Any]) -> str:
    model = run.get("model") or {}
    return str(model.get("checkpoint_id") or model.get("checkpoint_path") or model.get("checkpoint") or "")


def run_seed(run: dict[str, Any]) -> Any:
    generation = run.get("generation") or {}
    return generation.get("seed")


def list_experiments(*, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run in list_runs(artifact_root=artifact_root):
        name = experiment_name(run)
        experiment = grouped.setdefault(
            name,
            {
                "experiment_id": name,
                "name": name,
                "project": run.get("project"),
                "num_runs": 0,
                "runs_with_metrics": 0,
                "latest_updated_at": "",
                "run_ids": [],
            },
        )
        experiment["num_runs"] += 1
        experiment["run_ids"].append(run.get("run_id"))
        updated_at = run.get("updated_at") or run.get("created_at") or ""
        if updated_at > experiment["latest_updated_at"]:
            experiment["latest_updated_at"] = updated_at
        run_id = run.get("run_id")
        if run_id and load_run_metrics(str(run_id), artifact_root=artifact_root).get("modes"):
            experiment["runs_with_metrics"] += 1

    return sorted(grouped.values(), key=lambda item: item.get("latest_updated_at") or "", reverse=True)


def list_experiment_runs(experiment_id: str, *, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    return [run for run in list_runs(artifact_root=artifact_root) if experiment_name(run) == experiment_id]


def compare_run_rows(run_ids: list[str] | None = None, *, artifact_root: Path | None = None) -> dict[str, Any]:
    selected_ids = {run_id for run_id in (run_ids or []) if run_id}
    runs = list_runs(artifact_root=artifact_root)
    rows: list[dict[str, Any]] = []

    for run in runs:
        run_id = str(run.get("run_id") or "")
        if not run_id or (selected_ids and run_id not in selected_ids):
            continue
        metrics = load_run_metrics(run_id, artifact_root=artifact_root)
        for metric in metrics.get("modes", []):
            rows.append(
                {
                    "run_id": run_id,
                    "project": run.get("project"),
                    "experiment": experiment_name(run),
                    "checkpoint": run_checkpoint_id(run),
                    "seed": run_seed(run),
                    "subset": metric.get("subset") or "all",
                    "mode": metric.get("mode"),
                    "status": metric.get("status") or run.get("status"),
                    "created_at": run.get("created_at"),
                    "updated_at": run.get("updated_at"),
                    "num_utterances": metric.get("num_utterances", 0),
                    "wer": metric.get("wer"),
                    "cer": metric.get("cer"),
                    "substitutions": metric.get("wer_substitutions", 0),
                    "deletions": metric.get("wer_deletions", 0),
                    "insertions": metric.get("wer_insertions", 0),
                    "wer_reference_length": metric.get("wer_reference_length", 0),
                    "cer_reference_length": metric.get("cer_reference_length", 0),
                }
            )

    rows.sort(key=lambda item: (item.get("experiment") or "", item.get("run_id") or "", item.get("mode") or ""))
    return {"run_ids": sorted(selected_ids) if selected_ids else [], "rows": rows}


def run_export_rows(run_id: str, *, artifact_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = load_run(run_id, artifact_root=artifact_root)
    metrics = load_run_metrics(run_id, artifact_root=artifact_root)
    rows: list[dict[str, Any]] = []
    for metric in metrics.get("modes", []):
        rows.append(
            {
                "run_id": run_id,
                "project": run.get("project"),
                "experiment": experiment_name(run),
                "checkpoint": run_checkpoint_id(run),
                "seed": run_seed(run),
                "subset": metric.get("subset") or "all",
                "mode": metric.get("mode"),
                "status": metric.get("status") or run.get("status"),
                "num_utterances": metric.get("num_utterances", 0),
                "wer": metric.get("wer"),
                "cer": metric.get("cer"),
                "wer_substitutions": metric.get("wer_substitutions", 0),
                "wer_deletions": metric.get("wer_deletions", 0),
                "wer_insertions": metric.get("wer_insertions", 0),
                "wer_reference_length": metric.get("wer_reference_length", 0),
                "cer_substitutions": metric.get("cer_substitutions", 0),
                "cer_deletions": metric.get("cer_deletions", 0),
                "cer_insertions": metric.get("cer_insertions", 0),
                "cer_reference_length": metric.get("cer_reference_length", 0),
            }
        )
    return run, rows


def qualitative_example_rows(run_id: str, *, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = run_path(run_id, artifact_root=artifact_root)
    manifest_rows = load_jsonl(root / "manifest.jsonl")
    prediction_by_mode = {
        path.stem: {str(row.get("utterance_id")): row for row in load_jsonl(path)}
        for path in sorted((root / "predictions").glob("*.jsonl"))
    }
    modes = sorted({path.name for path in (root / "generated").iterdir() if path.is_dir()}) if (root / "generated").exists() else sorted(prediction_by_mode)

    rows: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifest_rows):
        utterance_id = str(manifest.get("utterance_id") or manifest.get("id") or f"utt-{index:06d}")
        mode_payload: dict[str, Any] = {}
        for mode in modes:
            wav_path = root / "generated" / mode / f"{utterance_id}.wav"
            prediction = prediction_by_mode.get(mode, {}).get(utterance_id, {})
            mode_payload[mode] = {
                "generated_audio": str(wav_path.relative_to(root)) if wav_path.exists() else None,
                "hypothesis": prediction.get("hypothesis", ""),
                "status": prediction.get("status"),
            }
        rows.append(
            {
                "run_id": run_id,
                "utterance_id": utterance_id,
                "subset": manifest.get("subset"),
                "reference_text": manifest.get("text") or manifest.get("ref_text"),
                "target_text": manifest.get("gen_text") or manifest.get("target_text") or manifest.get("text"),
                "ref_audio": manifest.get("ref_audio") or manifest.get("audio_path"),
                "modes": mode_payload,
            }
        )
    return rows


def ensure_run_paper_exports(run_id: str, *, artifact_root: Path | None = None) -> Path:
    root = run_path(run_id, artifact_root=artifact_root)
    run, metric_rows = run_export_rows(run_id, artifact_root=artifact_root)
    export_root = root / "paper_exports"
    main_fields = ["run_id", "experiment", "checkpoint", "seed", "subset", "mode", "status", "num_utterances", "wer", "cer"]
    error_fields = [
        "run_id",
        "experiment",
        "checkpoint",
        "seed",
        "subset",
        "mode",
        "status",
        "wer_substitutions",
        "wer_deletions",
        "wer_insertions",
        "wer_reference_length",
        "cer_substitutions",
        "cer_deletions",
        "cer_insertions",
        "cer_reference_length",
    ]
    write_csv(export_root / "main_table.csv", metric_rows, main_fields)
    write_csv(export_root / "error_breakdown.csv", metric_rows, error_fields)
    write_jsonl(export_root / "qualitative_examples.jsonl", qualitative_example_rows(run_id, artifact_root=artifact_root))
    (export_root / "README.md").write_text(
        "\n".join(
            [
                f"# Paper exports for {run_id}",
                "",
                f"- experiment: {experiment_name(run)}",
                f"- checkpoint: {run_checkpoint_id(run) or 'n/a'}",
                "- main_table.csv: mode-level WER/CER values.",
                "- error_breakdown.csv: word/character edit counts by mode.",
                "- qualitative_examples.jsonl: manifest examples with per-mode generated audio and hypotheses.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return export_root


def ensure_experiment_ablation_export(experiment_id: str, *, artifact_root: Path | None = None) -> Path:
    root = artifact_root or default_artifact_root()
    rows = compare_run_rows([str(run["run_id"]) for run in list_experiment_runs(experiment_id, artifact_root=artifact_root)], artifact_root=artifact_root)["rows"]
    export_root = root.parent / "experiments" / safe_segment(experiment_id) / "paper_exports"
    path = export_root / "ablation_table.csv"
    fieldnames = [
        "run_id",
        "experiment",
        "checkpoint",
        "seed",
        "subset",
        "mode",
        "status",
        "num_utterances",
        "wer",
        "cer",
        "substitutions",
        "deletions",
        "insertions",
        "wer_reference_length",
        "cer_reference_length",
    ]
    write_csv(path, rows, fieldnames)
    return path


def load_run_utterances(run_id: str, *, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = run_path(run_id, artifact_root=artifact_root)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        return []

    rows = load_jsonl(manifest_path)

    modes = [item.name for item in (root / "generated").iterdir()] if (root / "generated").exists() else []
    predictions_by_mode = {
        path.stem: {str(row.get("utterance_id")): row for row in load_jsonl(path)}
        for path in sorted((root / "predictions").glob("*.jsonl"))
    }
    posterior_by_id = {
        str(row.get("utterance_id")): row
        for row in load_jsonl(root / "posterior_cache" / "run.posterior.jsonl")
        if row.get("utterance_id")
    }
    metrics_by_id = {
        str(row.get("utterance_id")): row
        for row in load_jsonl(root / "metrics" / "per_utterance.jsonl")
        if row.get("utterance_id")
    }
    utterances: list[dict[str, Any]] = []
    for row in rows:
        utterance_id = row.get("utterance_id")
        if not utterance_id:
            continue
        reference_text = row.get("text") or row.get("ref_text")
        posterior = posterior_by_id.get(str(utterance_id), {})
        per_utterance_metrics = metrics_by_id.get(str(utterance_id), {})
        utterance = {
            "utterance_id": utterance_id,
            "subset": row.get("subset"),
            "text": reference_text,
            "reference_text": reference_text,
            "gen_text": row.get("gen_text") or row.get("target_text") or row.get("text"),
            "ref_audio": row.get("ref_audio") or row.get("audio_path"),
            "posterior_entropy": posterior.get("mean_entropy"),
            "one_best": posterior.get("one_best"),
            "failure_type": per_utterance_metrics.get("failure_type"),
            "modes": {},
        }
        for mode in modes:
            wav = root / "generated" / mode / f"{utterance_id}.wav"
            prediction_file = root / "predictions" / f"{mode}.jsonl"
            prediction_row = predictions_by_mode.get(mode, {}).get(str(utterance_id), {})
            hypothesis = prediction_row.get("hypothesis")
            utterance["modes"][mode] = {
                "generated_audio": str(wav.relative_to(root)) if wav.exists() else None,
                "prediction_file": str(prediction_file.relative_to(root)) if prediction_file.exists() else None,
                "prediction_text": hypothesis,
                "status": prediction_row.get("status"),
                "failure_type": prediction_row.get("failure_type") or prediction_row.get("status"),
                "diff": word_diff(reference_text, hypothesis),
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


def list_queued_jobs(*, artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = artifact_root or default_artifact_root()
    if not root.exists():
        return []

    jobs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/job.json")):
        try:
            job = read_json(path)
        except json.JSONDecodeError:
            continue
        if job.get("status") == "queued":
            job.setdefault("run_id", path.parent.name)
            jobs.append(job)
    return sorted(jobs, key=lambda item: item.get("queued_at") or "")


def remove_flag(command: list[str], flag: str, *, takes_value: bool = False) -> list[str]:
    updated: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == flag:
            index += 2 if takes_value else 1
            continue
        updated.append(command[index])
        index += 1
    return updated


def remove_repeated_flag(command: list[str], flag: str) -> list[str]:
    updated: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == flag:
            index += 2
            continue
        updated.append(command[index])
        index += 1
    return updated


def resume_command(command: list[str], run_id: str, *, artifact_root: Path | None = None) -> list[str]:
    root = run_path(run_id, artifact_root=artifact_root)
    updated = list(command)
    if (root / "posterior_cache" / "run.posterior.jsonl").exists():
        updated = remove_flag(updated, "--run_posterior_extraction")
    modes = [updated[index + 1] for index, item in enumerate(updated[:-1]) if item == "--mode"]
    if modes:
        inference_done = all((root / "generated" / mode / "commands.jsonl").exists() for mode in modes)
        prediction_done = all((root / "predictions" / f"{mode}.jsonl").exists() for mode in modes)
    else:
        inference_done = (root / "generated").exists() and any((root / "generated").glob("*/commands.jsonl"))
        prediction_done = (root / "predictions").exists() and any((root / "predictions").glob("*.jsonl"))
    if inference_done:
        updated = remove_flag(updated, "--run_inference")
    if prediction_done:
        updated = remove_flag(updated, "--run_prediction")
    if (root / "metrics" / "summary.json").exists():
        updated = remove_flag(updated, "--run_metrics")
    return updated


def queue_job(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    try:
        from services.queue import enqueue_run

        return enqueue_run(run_id, artifact_root=artifact_root)
    except Exception as exc:
        return {"backend": "file", "queued": False, "error": str(exc)}


def set_run_status(run_id: str, status: str, *, artifact_root: Path | None = None, error_message: str | None = None) -> None:
    path = run_path(run_id, artifact_root=artifact_root) / "run.json"
    if not path.exists():
        return
    run = read_json(path)
    run["status"] = status
    run["updated_at"] = timestamp()
    if error_message:
        run["error_message"] = error_message
    write_json(path, run)


def cancel_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    job_file = job_path(run_id, artifact_root=artifact_root)
    job = load_run_job(run_id, artifact_root=artifact_root)
    if job["status"] == "unknown":
        raise FileNotFoundError(f"Run not found: {run_id}")
    if job.get("status") == "queued":
        job["status"] = "cancelled"
        job["finished_at"] = timestamp()
        write_json(job_file, job)
        set_run_status(run_id, "cancelled", artifact_root=artifact_root)
        append_event(run_id, level="info", stage="job", message="cancelled queued job", artifact_root=artifact_root)
        return job
    if job.get("status") == "running":
        job["status"] = "cancelling"
        job["cancel_requested_at"] = timestamp()
        pid = job.get("pid")
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                job["cancel_error"] = str(exc)
        write_json(job_file, job)
        append_event(run_id, level="warning", stage="job", message="cancel requested", artifact_root=artifact_root)
        return job
    return job


def requeue_existing_job(run_id: str, command: list[str], *, artifact_root: Path | None = None, action: str) -> dict[str, Any]:
    job_file = job_path(run_id, artifact_root=artifact_root)
    current = load_run_job(run_id, artifact_root=artifact_root)
    if current["status"] == "unknown":
        raise FileNotFoundError(f"Run not found: {run_id}")
    attempts = current.get("attempts") or []
    attempts.append({key: current.get(key) for key in ["status", "started_at", "finished_at", "exit_code", "error_message"]})
    current.update(
        {
            "status": "queued",
            "pid": None,
            "command": command,
            "queued_at": timestamp(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "error_message": None,
            "attempts": attempts,
        }
    )
    write_json(job_file, current)
    set_run_status(run_id, "queued", artifact_root=artifact_root)
    append_event(run_id, level="info", stage="job", message=action, artifact_root=artifact_root)
    current["queue"] = queue_job(run_id, artifact_root=artifact_root)
    return current


def retry_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    job = load_run_job(run_id, artifact_root=artifact_root)
    if job.get("status") not in {"failed", "cancelled", "cancelling"}:
        return job
    return requeue_existing_job(run_id, job.get("command") or [], artifact_root=artifact_root, action="retry queued")


def resume_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    job = load_run_job(run_id, artifact_root=artifact_root)
    if job.get("status") == "unknown":
        raise FileNotFoundError(f"Run not found: {run_id}")
    command = resume_command(job.get("command") or [], run_id, artifact_root=artifact_root)
    return requeue_existing_job(run_id, command, artifact_root=artifact_root, action="resume queued")


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
        "checkpoint_id": "--checkpoint_id",
        "checkpoint": "--checkpoint",
        "checkpoint_hash": "--checkpoint_hash",
        "vocoder": "--vocoder",
        "seed": "--seed",
        "language": "--language",
        "posterior_encoder_ckpt": "--posterior_encoder_ckpt",
        "ssl_cache": "--ssl_cache",
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
    try:
        payload = resolve_dataset_payload(payload)
        payload = resolve_checkpoint_payload(payload)
    except ValueError as exc:
        return {
            "status": "failed",
            "run_id": str(payload.get("run_id") or ""),
            "exit_code": 1,
            "command": [],
            "stdout": "",
            "stderr": str(exc),
        }
    run_id = payload.get("run_id") or generate_run_id()
    payload["run_id"] = run_id
    root = artifact_root or default_artifact_root()
    run_root = run_path(run_id, artifact_root=root)
    if payload.get("fail_if_exists") and run_root.exists():
        return {
            "status": "failed",
            "run_id": run_id,
            "exit_code": 1,
            "command": [],
            "stdout": "",
            "stderr": f"Run directory already exists: {run_root}",
        }

    log_root = run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    command_payload = dict(payload)
    command_payload["fail_if_exists"] = False
    command = build_pipeline_command(command_payload, artifact_root=artifact_root)
    stdout_path = log_root / "job.stdout.log"
    stderr_path = log_root / "job.stderr.log"
    job_file = job_path(run_id, artifact_root=root)
    job = {
        "run_id": run_id,
        "project": payload.get("project"),
        "experiment": payload.get("experiment"),
        "dataset_id": payload.get("dataset_id"),
        "modes": payload.get("modes", []),
        "status": "queued",
        "pid": None,
        "command": command,
        "queued_at": timestamp(),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "stdout_log": str(stdout_path.relative_to(run_root)),
        "stderr_log": str(stderr_path.relative_to(run_root)),
    }
    write_json(job_file, job)
    append_event(run_id, level="info", stage="job", message="queued", artifact_root=root)
    queue_info = queue_job(run_id, artifact_root=root)

    response = {
        "status": "submitted",
        "run_id": run_id,
        "exit_code": None,
        "command": command,
        "stdout": "",
        "stderr": "",
        "job": job,
        "queue": queue_info,
    }
    return response
