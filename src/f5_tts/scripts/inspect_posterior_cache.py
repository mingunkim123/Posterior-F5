"""Inspect Posterior-F5 JSONL/NPZ cache integrity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from f5_tts.posterior.io import iter_posterior_manifest, load_topk_arrays, resolve_shard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Posterior-F5 posterior cache JSONL and NPZ shards.")
    parser.add_argument("--posterior_file", required=True, help="Posterior cache JSONL file.")
    parser.add_argument("--base_dir", help="Base directory for relative NPZ shards. Defaults to posterior file parent.")
    parser.add_argument("--expected_count", type=int, help="Expected number of JSONL utterances.")
    parser.add_argument("--require_frame_posteriors", action="store_true", help="Fail if an entry has no CTC top-k shard.")
    parser.add_argument("--min_expected_ref_len", type=float, default=0.0)
    parser.add_argument("--min_topk_mass", type=float, default=0.0, help="Fail if any top-k row mass is below this value.")
    parser.add_argument("--max_topk_mass", type=float, default=1.0001, help="Fail if any top-k row mass is above this value.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _row_sums(rows: list[list[float]]) -> list[float]:
    return [sum(max(float(prob), 0.0) for prob in row) for row in rows]


def _shape(rows: list[list[Any]]) -> tuple[int, int | None]:
    if not rows:
        return 0, None
    return len(rows), len(rows[0])


def inspect_posterior_cache(
    posterior_file: str | Path,
    *,
    base_dir: str | Path | None = None,
    expected_count: int | None = None,
    require_frame_posteriors: bool = False,
    min_expected_ref_len: float = 0.0,
    min_topk_mass: float = 0.0,
    max_topk_mass: float = 1.0001,
) -> dict[str, Any]:
    manifest_path = Path(posterior_file)
    shard_base_dir = Path(base_dir) if base_dir is not None else manifest_path.parent

    errors: list[str] = []
    warnings: list[str] = []
    utterance_summaries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for utterance in iter_posterior_manifest(manifest_path):
        summary: dict[str, Any] = {
            "utterance_id": utterance.utterance_id,
            "audio_path": utterance.audio_path,
            "expected_ref_len": utterance.expected_ref_len,
            "has_frame_posteriors": utterance.frame_posteriors is not None,
        }

        if utterance.utterance_id in seen_ids:
            errors.append(f"{utterance.utterance_id}: duplicate utterance_id")
        seen_ids.add(utterance.utterance_id)

        if not utterance.audio_path:
            errors.append(f"{utterance.utterance_id}: missing audio_path")

        if utterance.expected_ref_len is None:
            warnings.append(f"{utterance.utterance_id}: missing expected_ref_len")
        elif not _finite_number(utterance.expected_ref_len) or utterance.expected_ref_len < min_expected_ref_len:
            errors.append(f"{utterance.utterance_id}: invalid expected_ref_len={utterance.expected_ref_len}")

        frame_posteriors = utterance.frame_posteriors
        if frame_posteriors is None:
            if require_frame_posteriors:
                errors.append(f"{utterance.utterance_id}: missing frame_posteriors")
            utterance_summaries.append(summary)
            continue

        try:
            shard_path = resolve_shard_path(frame_posteriors, base_dir=shard_base_dir)
            summary["shard_path"] = str(shard_path)
            if frame_posteriors.token_ids is None or frame_posteriors.probs is None:
                if not shard_path.exists():
                    errors.append(f"{utterance.utterance_id}: missing shard {shard_path}")
            token_ids, probs = load_topk_arrays(frame_posteriors, base_dir=shard_base_dir)
        except Exception as exc:
            errors.append(f"{utterance.utterance_id}: failed to load top-k arrays: {exc}")
            utterance_summaries.append(summary)
            continue

        id_rows, id_width = _shape(token_ids)
        prob_rows, prob_width = _shape(probs)
        summary["num_frames"] = id_rows
        summary["top_k"] = id_width

        if id_rows == 0:
            errors.append(f"{utterance.utterance_id}: empty token_ids")
        if id_rows != prob_rows:
            errors.append(f"{utterance.utterance_id}: token_ids rows {id_rows} != probs rows {prob_rows}")
        if id_width != prob_width:
            errors.append(f"{utterance.utterance_id}: token_ids width {id_width} != probs width {prob_width}")

        if any(len(row) != id_width for row in token_ids):
            errors.append(f"{utterance.utterance_id}: ragged token_ids rows")
        if any(len(row) != prob_width for row in probs):
            errors.append(f"{utterance.utterance_id}: ragged probs rows")

        if frame_posteriors.num_frames is not None and frame_posteriors.num_frames != id_rows:
            errors.append(
                f"{utterance.utterance_id}: metadata num_frames={frame_posteriors.num_frames} != loaded {id_rows}"
            )
        if frame_posteriors.top_k is not None and frame_posteriors.top_k != id_width:
            errors.append(f"{utterance.utterance_id}: metadata top_k={frame_posteriors.top_k} != loaded {id_width}")

        flat_probs = [float(prob) for row in probs for prob in row]
        if any(not math.isfinite(prob) for prob in flat_probs):
            errors.append(f"{utterance.utterance_id}: non-finite posterior probability")
        if any(prob < 0.0 for prob in flat_probs):
            errors.append(f"{utterance.utterance_id}: negative posterior probability")

        sums = _row_sums(probs)
        if sums:
            min_mass = min(sums)
            max_mass = max(sums)
            mean_mass = sum(sums) / len(sums)
            summary["topk_mass_min"] = min_mass
            summary["topk_mass_mean"] = mean_mass
            summary["topk_mass_max"] = max_mass
            if min_mass < min_topk_mass:
                errors.append(f"{utterance.utterance_id}: min top-k mass {min_mass:.6f} < {min_topk_mass:.6f}")
            if max_mass > max_topk_mass:
                errors.append(f"{utterance.utterance_id}: max top-k mass {max_mass:.6f} > {max_topk_mass:.6f}")

        token_map = utterance.token_map
        blank_id = frame_posteriors.blank_id if frame_posteriors.blank_id is not None else getattr(token_map, "blank_id", None)
        summary["blank_id"] = blank_id
        if blank_id is None:
            warnings.append(f"{utterance.utterance_id}: missing blank_id")
        elif token_map is not None and token_map.tokens and not (0 <= int(blank_id) < len(token_map.tokens)):
            errors.append(f"{utterance.utterance_id}: blank_id {blank_id} outside token_map")

        utterance_summaries.append(summary)

    if expected_count is not None and len(utterance_summaries) != expected_count:
        errors.append(f"expected_count={expected_count} but found {len(utterance_summaries)}")

    return {
        "ok": not errors,
        "num_utterances": len(utterance_summaries),
        "errors": errors,
        "warnings": warnings,
        "utterances": utterance_summaries,
    }


def print_human_report(report: dict[str, Any]) -> None:
    status = "OK" if report["ok"] else "FAILED"
    print(f"Posterior cache inspection: {status}")
    print(f"utterances: {report['num_utterances']}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")


def main() -> None:
    args = parse_args()
    report = inspect_posterior_cache(
        args.posterior_file,
        base_dir=args.base_dir,
        expected_count=args.expected_count,
        require_frame_posteriors=args.require_frame_posteriors,
        min_expected_ref_len=args.min_expected_ref_len,
        min_topk_mass=args.min_topk_mass,
        max_topk_mass=args.max_topk_mass,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human_report(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
