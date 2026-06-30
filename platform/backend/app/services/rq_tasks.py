"""RQ task entrypoints importable from the backend app path."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_APP = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_APP.parents[2]
WORKERS = REPO_ROOT / "platform" / "workers"
for path in (BACKEND_APP, WORKERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_job_worker import run_job  # noqa: E402
from run_store import default_artifact_root  # noqa: E402


def run_queued_job(run_id: str, artifact_root: str | None = None) -> dict[str, Any]:
    root = Path(artifact_root).expanduser() if artifact_root else default_artifact_root(REPO_ROOT)
    return run_job(run_id, artifact_root=root)
