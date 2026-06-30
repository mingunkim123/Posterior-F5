"""Create paper-friendly tables from Posterior-F5 metric summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TABLE_FIELDS = ["subset", "mode", "num_utterances", "wer", "cer", "wer_deletions", "wer_insertions", "wer_substitutions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Posterior-F5 metric summaries into CSV/Markdown tables.")
    parser.add_argument("--summary", required=True, help="metrics/summary.json file.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rows_from_summary(summary: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_metrics in summary.get("modes", []):
        mode = mode_metrics.get("mode")
        if mode_metrics.get("subsets"):
            for subset in mode_metrics["subsets"]:
                rows.append({"subset": subset.get("subset"), "mode": mode, **subset})
        else:
            rows.append({"subset": "all", **mode_metrics})
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TABLE_FIELDS})


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    lines = ["| " + " | ".join(TABLE_FIELDS) + " |", "| " + " | ".join(["---"] * len(TABLE_FIELDS)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in TABLE_FIELDS) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = rows_from_summary(summary)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)


if __name__ == "__main__":
    main()
