from f5_tts.eval.error_breakdown import cer_breakdown, edit_breakdown, normalize_text, wer_breakdown


def test_edit_breakdown_counts_substitution_deletion_and_insertion():
    result = edit_breakdown(["a", "b", "c"], ["a", "x", "c", "d"])

    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 1
    assert result.reference_length == 3
    assert result.errors == 2


def test_wer_breakdown_counts_deletion():
    result = wer_breakdown("hello small world", "hello world")

    assert result.substitutions == 0
    assert result.deletions == 1
    assert result.insertions == 0
    assert result.rate == 1 / 3


def test_cer_breakdown_counts_insertion():
    result = cer_breakdown("abc", "abbc")

    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.insertions == 1
    assert result.rate == 1 / 3


def test_paper_normalizer_makes_case_and_punctuation_comparable():
    assert normalize_text("Hello, WORLD!", profile="paper") == "hello world"
    assert wer_breakdown("Hello, WORLD!", "hello world").rate == 0
