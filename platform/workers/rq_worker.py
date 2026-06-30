"""Run Redis/RQ queued Posterior-F5 jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from redis import Redis
from rq import Worker


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    backend_app = repo_root / "platform" / "backend" / "app"
    if str(backend_app) not in sys.path:
        sys.path.insert(0, str(backend_app))
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    worker = Worker(["posterior-f5"], connection=Redis.from_url(redis_url))
    worker.work()


if __name__ == "__main__":
    main()
