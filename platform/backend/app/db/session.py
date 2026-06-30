"""SQLite connection helpers for the metadata cache."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_db_path(repo_root: Path | None = None) -> Path:
    root = repo_root or repository_root()
    configured = os.environ.get("MLOPS_SQLITE_PATH")
    if configured:
        return Path(configured)
    artifact_root = root / "mlops_artifacts"
    if artifact_root.exists() and os.access(artifact_root, os.W_OK):
        return artifact_root / "metadata.sqlite3"
    return root / ".mlops_metadata" / "metadata.sqlite3"


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
