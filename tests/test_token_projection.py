from f5_tts.posterior.schema import PosteriorTokenMap
from f5_tts.posterior.token_projection import project_asr_token_to_f5_id, project_topk_token_ids


def _token_map():
    return PosteriorTokenMap(
        source="toy-ctc",
        tokens=["<pad>", "A", "|", "<s>", "z"],
        token_to_id={"<pad>": 0, "A": 1, "|": 2, "<s>": 3, "z": 4},
        blank_id=0,
        filler_id=0,
        unk_id=3,
    )


def test_project_blank_and_filler_to_f5_padding_id():
    assert project_asr_token_to_f5_id(0, token_map=_token_map(), f5_vocab={"A": 7}) == -1


def test_project_space_marker_to_space_token():
    assert project_asr_token_to_f5_id(2, token_map=_token_map(), f5_vocab={" ": 4}) == 4


def test_project_unknown_token_uses_lowercase_then_fallback():
    token_map = _token_map()

    assert project_asr_token_to_f5_id(1, token_map=token_map, f5_vocab={"a": 9}) == 9
    assert project_asr_token_to_f5_id(4, token_map=token_map, f5_vocab={}, unknown_value=123) == 123


def test_project_special_and_out_of_range_to_padding_id():
    token_map = _token_map()

    assert project_asr_token_to_f5_id(3, token_map=token_map, f5_vocab={}) == -1
    assert project_asr_token_to_f5_id(99, token_map=token_map, f5_vocab={}) == -1


def test_project_topk_token_ids_preserves_matrix_shape():
    projected = project_topk_token_ids([[0, 1, 2]], token_map=_token_map(), f5_vocab={"a": 1, " ": 2})

    assert projected == [[-1, 1, 2]]


def test_projection_without_maps_returns_original_id():
    assert project_asr_token_to_f5_id(5) == 5
