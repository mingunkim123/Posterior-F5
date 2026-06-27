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
    parser.add_argument("--mode", default="hard", choices=["hard", "length_only", "soft_ctc", "posterior_encoder"])
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


def evaluate_rows(manifest_rows: list[dict], prediction_rows: list[dict]) -> dict:
    references = {row["utterance_id"]: row.get("text") or row.get("reference") or "" for row in manifest_rows}
    predictions = {row["utterance_id"]: row.get("hypothesis") or row.get("text") or "" for row in prediction_rows}

    totals = {
        "wer_substitutions": 0,
        "wer_deletions": 0,
        "wer_insertions": 0,
        "wer_reference_length": 0,
        "cer_substitutions": 0,
        "cer_deletions": 0,
        "cer_insertions": 0,
        "cer_reference_length": 0,
        "num_utterances": 0,
    }

    for utterance_id, reference in references.items():
        if utterance_id not in predictions:
            continue
        hypothesis = predictions[utterance_id]
        wer = wer_breakdown(reference, hypothesis)
        cer = cer_breakdown(reference, hypothesis)
        totals["wer_substitutions"] += wer.substitutions
        totals["wer_deletions"] += wer.deletions
        totals["wer_insertions"] += wer.insertions
        totals["wer_reference_length"] += wer.reference_length
        totals["cer_substitutions"] += cer.substitutions
        totals["cer_deletions"] += cer.deletions
        totals["cer_insertions"] += cer.insertions
        totals["cer_reference_length"] += cer.reference_length
        totals["num_utterances"] += 1

    wer_errors = totals["wer_substitutions"] + totals["wer_deletions"] + totals["wer_insertions"]
    cer_errors = totals["cer_substitutions"] + totals["cer_deletions"] + totals["cer_insertions"]
    totals["wer"] = wer_errors / max(totals["wer_reference_length"], 1)
    totals["cer"] = cer_errors / max(totals["cer_reference_length"], 1)
    return totals


def main() -> None:
    args = parse_args()
    metrics = evaluate_rows(read_jsonl(args.manifest), read_jsonl(args.predictions))
    metrics["mode"] = args.mode
    metrics["posterior_file"] = args.posterior_file

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
