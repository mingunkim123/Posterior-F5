import pytest

from f5_tts.eval.bootstrap_significance import bootstrap_ci, paired_differences


def test_paired_differences_match_utterance_ids():
    rows = [
        {"utterance_id": "a", "mode": "hard", "wer": 0.5},
        {"utterance_id": "a", "mode": "soft_ctc", "wer": 0.25},
        {"utterance_id": "b", "mode": "hard", "wer": 0.25},
        {"utterance_id": "b", "mode": "soft_ctc", "wer": 0.0},
        {"utterance_id": "c", "mode": "hard", "wer": 1.0},
    ]

    diffs = paired_differences(rows, baseline="hard", candidate="soft_ctc", metric="wer")

    assert diffs == [-0.25, -0.25]


def test_bootstrap_ci_is_reproducible_for_fixed_seed():
    result = bootstrap_ci([-0.25, -0.25, 0.0], samples=50, seed=7)

    assert result["num_pairs"] == 3
    assert result["mean_diff"] == pytest.approx(-1 / 6)
    assert result == bootstrap_ci([-0.25, -0.25, 0.0], samples=50, seed=7)
