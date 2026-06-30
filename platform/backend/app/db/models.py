"""SQLite schema for the local metadata cache.

The artifact directory remains the source of truth. These tables are rebuilt
from mlops_artifacts/runs when search/index speed starts to matter.
"""

SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS artifacts;
DROP TABLE IF EXISTS utterances;
DROP TABLE IF EXISTS metrics;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS runs;

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  project TEXT,
  experiment TEXT,
  status TEXT,
  created_at TEXT,
  updated_at TEXT,
  artifact_root TEXT,
  checkpoint TEXT,
  seed INTEGER
);

CREATE TABLE jobs (
  run_id TEXT PRIMARY KEY,
  status TEXT,
  queued_at TEXT,
  started_at TEXT,
  finished_at TEXT,
  exit_code INTEGER,
  pid INTEGER,
  error_message TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  mode TEXT,
  status TEXT,
  num_utterances INTEGER,
  wer REAL,
  cer REAL,
  substitutions INTEGER,
  deletions INTEGER,
  insertions INTEGER,
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  path TEXT NOT NULL,
  kind TEXT,
  size_bytes INTEGER,
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE utterances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  utterance_id TEXT,
  subset TEXT,
  reference_text TEXT,
  target_text TEXT,
  ref_audio TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_runs_experiment ON runs(experiment);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_metrics_run_mode ON metrics(run_id, mode);
CREATE INDEX idx_utterances_run_id ON utterances(run_id, utterance_id);
"""
