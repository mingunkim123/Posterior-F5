"""Queue adapter for local file queue and optional Redis/RQ."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def queue_backend() -> str:
    return os.environ.get("MLOPS_QUEUE_BACKEND", "file").strip().lower() or "file"


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


def enqueue_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    backend = queue_backend()
    if backend not in {"redis", "rq"}:
        return {"backend": "file", "queued": False}

    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        return {"backend": "file", "queued": False, "error": f"RQ unavailable: {exc}"}

    from services.rq_tasks import run_queued_job

    queue = Queue("posterior-f5", connection=Redis.from_url(redis_url()))
    job = queue.enqueue(run_queued_job, run_id, str(artifact_root) if artifact_root else None, job_id=f"posterior-f5-{run_id}")
    return {"backend": "redis", "queued": True, "queue_id": job.id}
