"""Evaluate posterior-aware F5-TTS outputs.

Per-mode WER/CER and per-utterance breakdown for one prediction JSONL against
its manifest reference text. See ``make_result_tables`` for paper-style output
and ``bootstrap_significance`` for paired-bootstrap comparisons.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from f5_tts.eval.error_breakdown import cer_breakdown, normalize_text, wer_breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hard, length-only, or soft-CTC F5-TTS outputs.")
    parser.add_argument("--manifest", required=True, help="Evaluation JSONL manifest with reference text.")
    parser.add_argument("--predictions", required=True, help="JSONL predictions with utterance_id and hypothesis text.")
    parser.add_argument("--output", required=True, help="Output JSON metrics file.")
    parser.add_argument("--per_utterance_output", help="Optional JSONL output with per-utterance metrics.")
    parser.add_argument("--generation_metadata", help="Optional commands/generation metadata JSONL for this mode.")
    parser.add_argument("--audio_metrics", help="Optional generated-audio metric JSONL for this mode.")
    parser.add_argument(
        "--normalizer",
        default="paper",
        choices=["paper", "lowercase", "none"],
        help="Text normalization profile used before WER/CER.",
    )
    parser.add_argument(
        "--mode",
        default="hard",
        choices=["hard", "oracle", "length_only", "soft_ctc", "posterior_encoder", "hybrid"],
    )
    parser.add_argument("--posterior_file", default="", help="Optional posterior cache manifest used for this run.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _safe_rate(errors: int, reference_length: int) -> float:
    return errors / max(reference_length, 1)


def _index_by_utterance(rows: list[dict]) -> dict[str, dict]:
    return {str(row["utterance_id"]): row for row in rows if row.get("utterance_id") is not None}


def _safe_mean(total: float, count: int) -> float | None:
    return total / count if count else None


def _audio_duration_sec(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / float(info.samplerate) if info.samplerate else None
    except Exception:
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                return wav.getnframes() / float(rate) if rate else None
        except Exception:
            return None


def _run_root_from_generation_metadata(path: str | Path | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.parents[2]
    except IndexError:
        return None


def _resolve_generated_audio(row: dict, *, run_root: Path | None) -> Path | None:
    value = row.get("generated_audio") or row.get("output")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if run_root is not None:
        return run_root / path
    return path


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
        "num_manifest_utterances": 0,
        "num_missing_predictions": 0,
        "generation_elapsed_sec": 0.0,
        "num_generation_elapsed": 0,
        "generated_audio_duration_sec": 0.0,
        "num_generated_audio_duration": 0,
        "rtf": 0.0,
        "num_rtf": 0,
        "speaker_similarity": 0.0,
        "num_speaker_similarity": 0,
        "utmos": 0.0,
        "num_utmos": 0,
    }


def _finalize_totals(totals: dict) -> dict:
    wer_errors = totals["wer_substitutions"] + totals["wer_deletions"] + totals["wer_insertions"]
    cer_errors = totals["cer_substitutions"] + totals["cer_deletions"] + totals["cer_insertions"]
    totals["wer"] = _safe_rate(wer_errors, totals["wer_reference_length"])
    totals["cer"] = _safe_rate(cer_errors, totals["cer_reference_length"])
    totals["wer_deletion_rate"] = _safe_rate(totals["wer_deletions"], totals["wer_reference_length"])
    totals["wer_insertion_rate"] = _safe_rate(totals["wer_insertions"], totals["wer_reference_length"])
    totals["wer_substitution_rate"] = _safe_rate(totals["wer_substitutions"], totals["wer_reference_length"])
    totals["cer_deletion_rate"] = _safe_rate(totals["cer_deletions"], totals["cer_reference_length"])
    totals["cer_insertion_rate"] = _safe_rate(totals["cer_insertions"], totals["cer_reference_length"])
    totals["cer_substitution_rate"] = _safe_rate(totals["cer_substitutions"], totals["cer_reference_length"])
    totals["prediction_coverage"] = _safe_rate(
        totals["num_utterances"] - totals["num_missing_predictions"],
        totals["num_manifest_utterances"] or totals["num_utterances"],
    )
    totals["generation_elapsed_sec_mean"] = _safe_mean(totals["generation_elapsed_sec"], totals["num_generation_elapsed"])
    totals["generated_audio_duration_sec_mean"] = _safe_mean(
        totals["generated_audio_duration_sec"],
        totals["num_generated_audio_duration"],
    )
    totals["rtf_mean"] = _safe_mean(totals["rtf"], totals["num_rtf"])
    totals["speaker_similarity_mean"] = _safe_mean(totals["speaker_similarity"], totals["num_speaker_similarity"])
    totals["spk_sim_mean"] = totals["speaker_similarity_mean"]
    totals["utmos_mean"] = _safe_mean(totals["utmos"], totals["num_utmos"])
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
    totals["num_manifest_utterances"] += 1
    if row.get("missing_prediction"):
        totals["num_missing_predictions"] += 1
    if row.get("generation_elapsed_sec") is not None:
        totals["generation_elapsed_sec"] += float(row["generation_elapsed_sec"])
        totals["num_generation_elapsed"] += 1
    if row.get("generated_audio_duration_sec") is not None:
        totals["generated_audio_duration_sec"] += float(row["generated_audio_duration_sec"])
        totals["num_generated_audio_duration"] += 1
    if row.get("rtf") is not None:
        totals["rtf"] += float(row["rtf"])
        totals["num_rtf"] += 1
    if row.get("speaker_similarity") is not None:
        totals["speaker_similarity"] += float(row["speaker_similarity"])
        totals["num_speaker_similarity"] += 1
    if row.get("utmos") is not None:
        totals["utmos"] += float(row["utmos"])
        totals["num_utmos"] += 1


def _failure_type(row: dict) -> str:
    if row.get("missing_prediction"):
        return "missing_prediction"
    if not row.get("hypothesis_normalized"):
        return "empty_hypothesis"
    if row["wer"] == 0 and row["cer"] == 0:
        return "exact"
    edits = {
        "substitution": row["wer_substitutions"],
        "deletion": row["wer_deletions"],
        "insertion": row["wer_insertions"],
    }
    dominant, count = max(edits.items(), key=lambda item: item[1])
    if count == 0:
        return "character_only"
    return f"{dominant}_heavy"


def evaluate_rows(
    manifest_rows: list[dict],
    prediction_rows: list[dict],
    generation_rows: list[dict] | None = None,
    audio_metric_rows: list[dict] | None = None,
    *,
    normalizer: str = "paper",
    mode: str | None = None,
    run_root: Path | None = None,
) -> tuple[dict, list[dict]]:
    manifests = _index_by_utterance(manifest_rows)
    predictions = _index_by_utterance(prediction_rows)
    generations = _index_by_utterance(generation_rows or [])
    audio_metrics = _index_by_utterance(audio_metric_rows or [])

    totals = _empty_totals()
    subset_totals: dict[str, dict] = {}
    per_utterance: list[dict] = []

    for utterance_id, manifest_row in manifests.items():
        prediction_row = predictions.get(utterance_id, {})
        generation_row = generations.get(utterance_id, {})
        audio_metric_row = audio_metrics.get(utterance_id, {})
        reference = manifest_row.get("text") or manifest_row.get("reference") or ""
        hypothesis = prediction_row.get("hypothesis") or prediction_row.get("text") or ""
        missing_prediction = utterance_id not in predictions
        subset = manifest_row.get("subset") or prediction_row.get("subset") or "all"
        wer = wer_breakdown(reference, hypothesis, profile=normalizer)
        cer = cer_breakdown(reference, hypothesis, profile=normalizer)
        generated_audio = prediction_row.get("generated_audio") or generation_row.get("output")
        generated_audio_path = _resolve_generated_audio(
            {"generated_audio": generated_audio},
            run_root=run_root,
        )
        generated_audio_duration_sec = audio_metric_row.get("generated_audio_duration_sec")
        if generated_audio_duration_sec is None and generated_audio_path is not None:
            generated_audio_duration_sec = _audio_duration_sec(generated_audio_path)
        generation_elapsed_sec = generation_row.get("elapsed_sec")
        rtf = audio_metric_row.get("rtf")
        if rtf is None and isinstance(generation_elapsed_sec, (int, float)) and generated_audio_duration_sec:
            rtf = float(generation_elapsed_sec) / float(generated_audio_duration_sec)
        speaker_similarity = audio_metric_row.get("speaker_similarity")
        spk_sim = audio_metric_row.get("spk_sim")
        if speaker_similarity is None:
            speaker_similarity = spk_sim
        if spk_sim is None:
            spk_sim = speaker_similarity
        row = {
            "utterance_id": utterance_id,
            "subset": subset,
            "mode": prediction_row.get("mode") or mode,
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_normalized": normalize_text(reference, profile=normalizer),
            "hypothesis_normalized": normalize_text(hypothesis, profile=normalizer),
            "normalizer": normalizer,
            "missing_prediction": missing_prediction,
            "generated_audio": generated_audio,
            "eval_asr": prediction_row.get("eval_asr"),
            "wer_substitutions": wer.substitutions,
            "wer_deletions": wer.deletions,
            "wer_insertions": wer.insertions,
            "wer_reference_length": wer.reference_length,
            "wer": wer.rate,
            "wer_deletion_rate": _safe_rate(wer.deletions, wer.reference_length),
            "wer_insertion_rate": _safe_rate(wer.insertions, wer.reference_length),
            "wer_substitution_rate": _safe_rate(wer.substitutions, wer.reference_length),
            "cer_substitutions": cer.substitutions,
            "cer_deletions": cer.deletions,
            "cer_insertions": cer.insertions,
            "cer_reference_length": cer.reference_length,
            "cer": cer.rate,
            "cer_deletion_rate": _safe_rate(cer.deletions, cer.reference_length),
            "cer_insertion_rate": _safe_rate(cer.insertions, cer.reference_length),
            "cer_substitution_rate": _safe_rate(cer.substitutions, cer.reference_length),
            "generation_elapsed_sec": generation_elapsed_sec,
            "generated_audio_duration_sec": generated_audio_duration_sec,
            "rtf": rtf,
            "speaker_similarity": speaker_similarity,
            "spk_sim": spk_sim,
            "utmos": audio_metric_row.get("utmos"),
            "output_size_bytes": generation_row.get("output_size_bytes"),
            "output_sha256": generation_row.get("output_sha256"),
        }
        row["failure_type"] = _failure_type(row)
        per_utterance.append(row)
        _add_row_to_totals(totals, row)
        subset_totals.setdefault(subset, _empty_totals())
        _add_row_to_totals(subset_totals[subset], row)

    metrics = _finalize_totals(totals)
    metrics["normalizer"] = normalizer
    metrics["subsets"] = [
        {"subset": subset, "normalizer": normalizer, **_finalize_totals(values)}
        for subset, values in sorted(subset_totals.items())
    ]
    return metrics, per_utterance


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    generation_rows = read_jsonl(args.generation_metadata) if args.generation_metadata else []
    audio_metric_rows = read_jsonl(args.audio_metrics) if args.audio_metrics else []
    metrics, per_utterance = evaluate_rows(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        generation_rows,
        audio_metric_rows,
        normalizer=args.normalizer,
        mode=args.mode,
        run_root=_run_root_from_generation_metadata(args.generation_metadata),
    )
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
