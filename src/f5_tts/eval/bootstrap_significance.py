"""Paired bootstrap significance helpers for Posterior-F5 metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap confidence intervals between two modes.")
    parser.add_argument("--per_utterance", required=True, help="Combined per-utterance metrics JSONL.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", default="wer", choices=["wer", "cer", "wer_deletion_rate", "cer_deletion_rate"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_differences(rows: list[dict], *, baseline: str, candidate: str, metric: str) -> list[float]:
    by_utterance: dict[str, dict[str, float]] = {}
    for row in rows:
        utterance_id = str(row.get("utterance_id"))
        mode = row.get("mode")
        if mode not in {baseline, candidate} or row.get(metric) is None:
            continue
        by_utterance.setdefault(utterance_id, {})[str(mode)] = float(row[metric])

    diffs = []
    for values in by_utterance.values():
        if baseline in values and candidate in values:
            diffs.append(values[candidate] - values[baseline])
    return diffs


def bootstrap_ci(differences: list[float], *, samples: int, seed: int) -> dict:
    if not differences:
        return {
            "num_pairs": 0,
            "mean_diff": None,
            "ci_low": None,
            "ci_high": None,
        }

    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(samples):
        draw = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return {
        "num_pairs": n,
        "mean_diff": sum(differences) / n,
        "ci_low": means[low_index],
        "ci_high": means[high_index],
    }


def main() -> None:
    args = parse_args()
    diffs = paired_differences(
        read_jsonl(args.per_utterance),
        baseline=args.baseline,
        candidate=args.candidate,
        metric=args.metric,
    )
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "metric": args.metric,
        "samples": args.samples,
        "seed": args.seed,
        **bootstrap_ci(diffs, samples=args.samples, seed=args.seed),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
