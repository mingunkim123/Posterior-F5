import pytest

from f5_tts.posterior.normalize import (
    entropy,
    normalize_probs,
    normalize_topk_rows,
    temperature_scale_probs,
    top_k_truncate,
)


def test_normalize_probs_clamps_and_renormalizes():
    assert normalize_probs([2.0, -1.0, 2.0]) == pytest.approx([0.5, 0.0, 0.5])


def test_normalize_probs_uses_uniform_fallback_for_zero_mass():
    assert normalize_probs([0.0, 0.0]) == pytest.approx([0.5, 0.5])


def test_normalize_probs_can_disable_uniform_fallback():
    assert normalize_probs([0.0, 0.0], fallback_uniform=False) == [0.0, 0.0]


def test_temperature_scaling_sharpens_and_softens_distribution():
    probs = [0.8, 0.2]

    sharpened = temperature_scale_probs(probs, temperature=0.5)
    softened = temperature_scale_probs(probs, temperature=2.0)

    assert sharpened[0] > probs[0]
    assert softened[0] < probs[0]
    assert sum(sharpened) == pytest.approx(1.0)
    assert sum(softened) == pytest.approx(1.0)


def test_temperature_scaling_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        temperature_scale_probs([0.5, 0.5], temperature=0.0)


def test_top_k_truncate_keeps_highest_probs_and_renormalizes():
    token_ids, probs = top_k_truncate([10, 20, 30], [0.1, 0.6, 0.3], k=2)

    assert token_ids == [20, 30]
    assert probs == pytest.approx([2 / 3, 1 / 3])


def test_top_k_truncate_validates_inputs():
    with pytest.raises(ValueError):
        top_k_truncate([1], [1.0], k=0)

    with pytest.raises(ValueError):
        top_k_truncate([1, 2], [1.0], k=1)


def test_normalize_topk_rows_applies_temperature_and_top_k():
    token_rows, prob_rows = normalize_topk_rows(
        [[1, 2, 3]],
        [[0.1, 0.7, 0.2]],
        temperature=0.5,
        top_k=2,
    )

    assert token_rows == [[2, 3]]
    assert sum(prob_rows[0]) == pytest.approx(1.0)
    assert prob_rows[0][0] > prob_rows[0][1]


def test_normalize_topk_rows_validates_row_counts():
    with pytest.raises(ValueError):
        normalize_topk_rows([[1]], [[1.0], [0.0]])

    with pytest.raises(ValueError):
        normalize_topk_rows([[1, 2]], [[1.0]])


def test_entropy_matches_binary_uniform_distribution():
    assert entropy([0.5, 0.5], base=2) == pytest.approx(1.0)


def test_entropy_rejects_invalid_base():
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=1.0)

