import json

import pytest

from f5_tts.eval.eval_posterior_f5 import evaluate_rows


def test_evaluate_rows_outputs_per_utterance_and_subset_metrics():
    manifest_rows = [
        {"utterance_id": "utt-1", "text": "hello world", "subset": "clean"},
        {"utterance_id": "utt-2", "text": "small test", "subset": "noisy"},
    ]
    prediction_rows = [
        {"utterance_id": "utt-1", "mode": "hard", "hypothesis": "hello world", "generated_audio": "a.wav"},
        {"utterance_id": "utt-2", "mode": "hard", "hypothesis": "small", "generated_audio": "b.wav"},
    ]
    generation_rows = [
        {"utterance_id": "utt-1", "elapsed_sec": 1.5, "output_size_bytes": 100, "output_sha256": "sha256:a"},
        {"utterance_id": "utt-2", "elapsed_sec": 2.5, "output_size_bytes": 120, "output_sha256": "sha256:b"},
    ]

    metrics, per_utterance = evaluate_rows(manifest_rows, prediction_rows, generation_rows)

    assert metrics["num_utterances"] == 2
    assert metrics["wer_deletions"] == 1
    assert metrics["wer"] == pytest.approx(0.25)
    assert metrics["generation_elapsed_sec_mean"] == pytest.approx(2.0)
    assert {row["subset"] for row in metrics["subsets"]} == {"clean", "noisy"}
    assert len(per_utterance) == 2
    assert per_utterance[1]["wer_deletion_rate"] == pytest.approx(0.5)
    assert per_utterance[0]["output_sha256"] == "sha256:a"
    assert metrics["normalizer"] == "paper"
    assert metrics["prediction_coverage"] == pytest.approx(1.0)


def test_eval_cli_writes_per_utterance_jsonl(tmp_path):
    from f5_tts.eval.eval_posterior_f5 import main
    import sys

    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    per_utterance = tmp_path / "per_utterance.jsonl"
    manifest.write_text(json.dumps({"utterance_id": "utt-1", "text": "hello world", "subset": "clean"}) + "\n")
    predictions.write_text(json.dumps({"utterance_id": "utt-1", "mode": "hard", "hypothesis": "hello"}) + "\n")

    old_argv = sys.argv
    sys.argv = [
        "eval_posterior_f5.py",
        "--manifest",
        str(manifest),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
        "--per_utterance_output",
        str(per_utterance),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    metrics = json.loads(output.read_text(encoding="utf-8"))
    row = json.loads(per_utterance.read_text(encoding="utf-8"))
    assert metrics["wer_deletions"] == 1
    assert row["utterance_id"] == "utt-1"


def test_evaluate_rows_counts_missing_predictions_as_deletions():
    metrics, per_utterance = evaluate_rows(
        [
            {"utterance_id": "utt-1", "text": "hello world"},
            {"utterance_id": "utt-2", "text": "missing words"},
        ],
        [{"utterance_id": "utt-1", "mode": "hard", "hypothesis": "hello world"}],
        mode="hard",
    )

    assert metrics["num_manifest_utterances"] == 2
    assert metrics["num_missing_predictions"] == 1
    assert metrics["prediction_coverage"] == pytest.approx(0.5)
    assert metrics["wer_deletions"] == 2
    assert per_utterance[1]["failure_type"] == "missing_prediction"
