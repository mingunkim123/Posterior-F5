"""Evaluate posterior-aware F5-TTS outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2]))

from f5_tts.eval.error_breakdown import cer_breakdown, wer_breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hard, length-only, or soft-CTC F5-TTS outputs.")
    parser.add_argument("--manifest", required=True, help="Evaluation JSONL manifest with reference text.")
    parser.add_argument("--predictions", required=True, help="JSONL predictions with utterance_id and hypothesis text.")
    parser.add_argument("--output", required=True, help="Output JSON metrics file.")
    parser.add_argument("--per_utterance_output", help="Optional JSONL output with per-utterance metrics.")
    parser.add_argument("--generation_metadata", help="Optional commands/generation metadata JSONL for this mode.")
    parser.add_argument(
        "--mode",
        default="hard",
        choices=["hard", "oracle", "length_only", "soft_ctc", "posterior_encoder", "hybrid"],
    )
    parser.add_argument("--posterior_file", default="", help="Optional posterior cache manifest used for this run.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_rate(errors: int, reference_length: int) -> float:
    return errors / max(reference_length, 1)


def _index_by_utterance(rows: list[dict]) -> dict[str, dict]:
    return {str(row["utterance_id"]): row for row in rows if row.get("utterance_id") is not None}


def _empty_totals() -> dict:
    return {
        "wer_substitutions": 0,
        "wer_deletions": 0,
        "wer_insertions": 0,
        "wer_reference_length": 0,
        "cer_substitutions": 0,
        "cer_deletions": 0,
        "cer_insertions": 0,
        "cer_reference_length": 0,
        "num_utterances": 0,
        "generation_elapsed_sec": 0.0,
        "num_generation_elapsed": 0,
    }


def _finalize_totals(totals: dict) -> dict:
    wer_errors = totals["wer_substitutions"] + totals["wer_deletions"] + totals["wer_insertions"]
    cer_errors = totals["cer_substitutions"] + totals["cer_deletions"] + totals["cer_insertions"]
    totals["wer"] = _safe_rate(wer_errors, totals["wer_reference_length"])
    totals["cer"] = _safe_rate(cer_errors, totals["cer_reference_length"])
    totals["wer_deletion_rate"] = _safe_rate(totals["wer_deletions"], totals["wer_reference_length"])
    totals["cer_deletion_rate"] = _safe_rate(totals["cer_deletions"], totals["cer_reference_length"])
    if totals["num_generation_elapsed"]:
        totals["generation_elapsed_sec_mean"] = totals["generation_elapsed_sec"] / totals["num_generation_elapsed"]
    else:
        totals["generation_elapsed_sec_mean"] = None
    return totals


def _add_row_to_totals(totals: dict, row: dict) -> None:
    for key in (
        "wer_substitutions",
        "wer_deletions",
        "wer_insertions",
        "wer_reference_length",
        "cer_substitutions",
        "cer_deletions",
        "cer_insertions",
        "cer_reference_length",
    ):
        totals[key] += row[key]
    totals["num_utterances"] += 1
    if row.get("generation_elapsed_sec") is not None:
        totals["generation_elapsed_sec"] += float(row["generation_elapsed_sec"])
        totals["num_generation_elapsed"] += 1


def evaluate_rows(
    manifest_rows: list[dict],
    prediction_rows: list[dict],
    generation_rows: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    manifests = _index_by_utterance(manifest_rows)
    predictions = _index_by_utterance(prediction_rows)
    generations = _index_by_utterance(generation_rows or [])

    totals = _empty_totals()
    subset_totals: dict[str, dict] = {}
    per_utterance: list[dict] = []

    for utterance_id, manifest_row in manifests.items():
        if utterance_id not in predictions:
            continue
        prediction_row = predictions[utterance_id]
        generation_row = generations.get(utterance_id, {})
        reference = manifest_row.get("text") or manifest_row.get("reference") or ""
        hypothesis = prediction_row.get("hypothesis") or prediction_row.get("text") or ""
        subset = manifest_row.get("subset") or prediction_row.get("subset") or "all"
        wer = wer_breakdown(reference, hypothesis)
        cer = cer_breakdown(reference, hypothesis)
        row = {
            "utterance_id": utterance_id,
            "subset": subset,
            "mode": prediction_row.get("mode"),
            "reference": reference,
            "hypothesis": hypothesis,
            "generated_audio": prediction_row.get("generated_audio"),
            "eval_asr": prediction_row.get("eval_asr"),
            "wer_substitutions": wer.substitutions,
            "wer_deletions": wer.deletions,
            "wer_insertions": wer.insertions,
            "wer_reference_length": wer.reference_length,
            "wer": wer.rate,
            "wer_deletion_rate": _safe_rate(wer.deletions, wer.reference_length),
            "cer_substitutions": cer.substitutions,
            "cer_deletions": cer.deletions,
            "cer_insertions": cer.insertions,
            "cer_reference_length": cer.reference_length,
            "cer": cer.rate,
            "cer_deletion_rate": _safe_rate(cer.deletions, cer.reference_length),
            "generation_elapsed_sec": generation_row.get("elapsed_sec"),
            "output_size_bytes": generation_row.get("output_size_bytes"),
            "output_sha256": generation_row.get("output_sha256"),
        }
        per_utterance.append(row)
        _add_row_to_totals(totals, row)
        subset_totals.setdefault(subset, _empty_totals())
        _add_row_to_totals(subset_totals[subset], row)

    metrics = _finalize_totals(totals)
    metrics["subsets"] = [
        {"subset": subset, **_finalize_totals(values)} for subset, values in sorted(subset_totals.items())
    ]
    return metrics, per_utterance


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    generation_rows = read_jsonl(args.generation_metadata) if args.generation_metadata else []
    metrics, per_utterance = evaluate_rows(read_jsonl(args.manifest), read_jsonl(args.predictions), generation_rows)
    metrics["mode"] = args.mode
    metrics["posterior_file"] = args.posterior_file

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_utterance_output:
        write_jsonl(args.per_utterance_output, per_utterance)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
