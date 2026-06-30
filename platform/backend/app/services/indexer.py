"""Rebuild the SQLite metadata cache from run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_APP = Path(__file__).resolve().parents[1]
if str(BACKEND_APP) not in sys.path:
    sys.path.insert(0, str(BACKEND_APP))

from db.models import SCHEMA  # noqa: E402
from db.session import connect, default_db_path  # noqa: E402
from run_store import (  # noqa: E402
    default_artifact_root,
    list_runs,
    load_run_job,
    load_run_metrics,
    load_run_utterances,
    run_checkpoint_id,
    run_seed,
)


def artifact_kind(path: Path) -> str:
    parts = set(path.parts)
    if "metrics" in parts:
        return "metrics"
    if "predictions" in parts:
        return "prediction"
    if "generated" in parts:
        return "audio"
    if "logs" in parts:
        return "log"
    if "posterior_cache" in parts:
        return "posterior"
    return "metadata"


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rebuild_index(*, artifact_root: Path | None = None, db_path: Path | None = None) -> dict[str, int | str]:
    root = artifact_root or default_artifact_root()
    path = db_path or default_db_path()
    connection = connect(path)
    with connection:
        connection.executescript(SCHEMA)
        run_count = metric_count = artifact_count = utterance_count = 0
        for run in list_runs(artifact_root=root):
            run_id = str(run.get("run_id") or "")
            if not run_id:
                continue
            connection.execute(
                """
                INSERT INTO runs
                  (run_id, project, experiment, status, created_at, updated_at, artifact_root, checkpoint, seed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run.get("project"),
                    run.get("experiment"),
                    run.get("status"),
                    run.get("created_at"),
                    run.get("updated_at"),
                    run.get("artifact_root"),
                    run_checkpoint_id(run),
                    safe_int(run_seed(run)),
                ),
            )
            run_count += 1

            job = load_run_job(run_id, artifact_root=root)
            if job.get("status") != "unknown":
                connection.execute(
                    """
                    INSERT INTO jobs
                      (run_id, status, queued_at, started_at, finished_at, exit_code, pid, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        job.get("status"),
                        job.get("queued_at"),
                        job.get("started_at"),
                        job.get("finished_at"),
                        safe_int(job.get("exit_code")),
                        safe_int(job.get("pid")),
                        job.get("error_message"),
                    ),
                )

            for metric in load_run_metrics(run_id, artifact_root=root).get("modes", []):
                connection.execute(
                    """
                    INSERT INTO metrics
                      (run_id, mode, status, num_utterances, wer, cer, substitutions, deletions, insertions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        metric.get("mode"),
                        metric.get("status"),
                        safe_int(metric.get("num_utterances")),
                        metric.get("wer"),
                        metric.get("cer"),
                        safe_int(metric.get("wer_substitutions")),
                        safe_int(metric.get("wer_deletions")),
                        safe_int(metric.get("wer_insertions")),
                    ),
                )
                metric_count += 1

            for utterance in load_run_utterances(run_id, artifact_root=root):
                connection.execute(
                    """
                    INSERT INTO utterances
                      (run_id, utterance_id, subset, reference_text, target_text, ref_audio)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        utterance.get("utterance_id"),
                        utterance.get("subset"),
                        utterance.get("text"),
                        utterance.get("gen_text") or utterance.get("text"),
                        utterance.get("ref_audio"),
                    ),
                )
                utterance_count += 1

            run_root = root / run_id
            for artifact in sorted(path for path in run_root.rglob("*") if path.is_file()):
                relative = artifact.relative_to(run_root)
                connection.execute(
                    """
                    INSERT INTO artifacts (run_id, path, kind, size_bytes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, str(relative), artifact_kind(relative), artifact.stat().st_size),
                )
                artifact_count += 1

    connection.close()
    return {
        "db_path": str(path),
        "runs": run_count,
        "metrics": metric_count,
        "utterances": utterance_count,
        "artifacts": artifact_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the Posterior-F5 SQLite metadata cache.")
    parser.add_argument("--artifact_root", default=None, help="Run artifact root. Defaults to mlops_artifacts/runs.")
    parser.add_argument("--db", default=None, help="SQLite database path. Defaults to mlops_artifacts/metadata.sqlite3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rebuild_index(
        artifact_root=Path(args.artifact_root).expanduser() if args.artifact_root else None,
        db_path=Path(args.db).expanduser() if args.db else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
