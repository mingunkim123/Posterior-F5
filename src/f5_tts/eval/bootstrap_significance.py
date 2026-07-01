"""Paired bootstrap significance helpers for Posterior-F5 metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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
            "p_value": None,
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
        "p_value": two_sided_zero_p_value(means),
    }


def two_sided_zero_p_value(samples: list[float]) -> float:
    """Approximate a two-sided paired-bootstrap p-value around zero difference."""

    if not samples:
        return 1.0
    non_positive = sum(1 for value in samples if value <= 0) / len(samples)
    non_negative = sum(1 for value in samples if value >= 0) / len(samples)
    return min(1.0, 2 * min(non_positive, non_negative))


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(index, p_value) for index, p_value in enumerate(p_values) if p_value is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    previous = 0.0
    m = len(indexed)
    for rank, (index, p_value) in enumerate(sorted(indexed, key=lambda item: item[1]), start=1):
        value = min(1.0, (m - rank + 1) * p_value)
        value = max(previous, value)
        previous = value
        adjusted[index] = value
    return adjusted


def fdr_bh_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(index, p_value) for index, p_value in enumerate(p_values) if p_value is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    m = len(indexed)
    running_min = 1.0
    for reverse_rank, (index, p_value) in enumerate(sorted(indexed, key=lambda item: item[1], reverse=True), start=1):
        rank = m - reverse_rank + 1
        value = min(running_min, min(1.0, p_value * m / rank))
        running_min = value
        adjusted[index] = value
    return adjusted


def apply_multiple_comparison_corrections(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach Holm and FDR adjusted p-values per metric family."""

    corrected = [dict(item) for item in results]
    metrics = sorted({str(item.get("metric")) for item in corrected if item.get("metric")})
    for metric in metrics:
        indices = [index for index, item in enumerate(corrected) if item.get("metric") == metric]
        p_values = [corrected[index].get("p_value") for index in indices]
        holm = holm_adjust(p_values)
        fdr = fdr_bh_adjust(p_values)
        for local_index, index in enumerate(indices):
            corrected[index]["holm_p_value"] = holm[local_index]
            corrected[index]["fdr_p_value"] = fdr[local_index]
            corrected[index]["significant"] = (
                corrected[index].get("ci_low") is not None
                and corrected[index].get("ci_high") is not None
                and (
                    corrected[index]["ci_high"] < 0
                    or corrected[index]["ci_low"] > 0
                )
            )
    return corrected


def paired_bootstrap_result(
    rows: list[dict],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    diffs = paired_differences(rows, baseline=baseline, candidate=candidate, metric=metric)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "samples": samples,
        "seed": seed,
        "direction": "lower_is_better",
        **bootstrap_ci(diffs, samples=samples, seed=seed),
    }


def main() -> None:
    args = parse_args()
    result = paired_bootstrap_result(
        read_jsonl(args.per_utterance),
        baseline=args.baseline,
        candidate=args.candidate,
        metric=args.metric,
        samples=args.samples,
        seed=args.seed,
    )
    result = apply_multiple_comparison_corrections([result])[0]
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
