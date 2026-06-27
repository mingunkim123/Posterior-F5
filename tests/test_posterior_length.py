import pytest

from f5_tts.posterior.length import expected_occupancy_len, expected_text_len
from f5_tts.posterior.schema import Hypothesis


def test_expected_text_len_uses_normalized_hypothesis_posteriors():
    length = expected_text_len(
        [
            Hypothesis(text="abc", posterior=2.0),
            Hypothesis(text="abcdef", posterior=1.0),
        ]
    )

    assert length == pytest.approx(4.0)


def test_expected_text_len_falls_back_to_uniform_weights():
    length = expected_text_len(
        [
            Hypothesis(text="aa", posterior=0.0),
            Hypothesis(text="aaaaaa", posterior=0.0),
        ]
    )

    assert length == pytest.approx(4.0)


def test_expected_text_len_can_use_character_length():
    length = expected_text_len(
        [
            Hypothesis(text="가", posterior=0.5),
            Hypothesis(text="가나", posterior=0.5),
        ],
        use_bytes=False,
    )

    assert length == pytest.approx(1.5)


def test_expected_text_len_returns_none_for_empty_input():
    assert expected_text_len([]) is None


def test_expected_occupancy_len_excludes_blank_and_filler_mass():
    length = expected_occupancy_len(
        probs=[
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ],
        token_ids=[
            [1, 0, 9],
            [0, 2, 9],
        ],
        blank_id=0,
        filler_id=9,
    )

    assert length == pytest.approx(1.1)


def test_expected_occupancy_len_can_normalize_rows():
    length = expected_occupancy_len(
        probs=[
            [3.0, 1.0],
            [2.0, 2.0],
        ],
        token_ids=[
            [1, 0],
            [0, 2],
        ],
        blank_id=0,
        normalize_rows=True,
    )

    assert length == pytest.approx(1.25)


def test_expected_occupancy_len_without_token_ids_uses_row_mass():
    length = expected_occupancy_len(
        probs=[
            [0.25, 0.25],
            [0.5, 0.25],
        ]
    )

    assert length == pytest.approx(1.25)


def test_expected_occupancy_len_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        expected_occupancy_len(
            probs=[[0.5, 0.5]],
            token_ids=[[1]],
        )


def test_expected_occupancy_len_returns_none_for_empty_input():
    assert expected_occupancy_len([]) is None

