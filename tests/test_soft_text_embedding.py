import pytest

torch = pytest.importorskip("torch")

from f5_tts.posterior.soft_embedding import expected_embedding_from_topk


def test_one_hot_posterior_matches_f5_embedding_lookup():
    embedding_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    token_ids = [[0], [2]]
    probs = [[1.0], [1.0]]

    expected = expected_embedding_from_topk(token_ids, probs, embedding_weight)

    assert torch.allclose(expected, embedding_weight[torch.tensor([1, 3])])


def test_blank_id_maps_to_filler_embedding():
    embedding_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    expected = expected_embedding_from_topk([[0]], [[1.0]], embedding_weight, blank_id=0)

    assert torch.allclose(expected, embedding_weight[0].unsqueeze(0))


def test_expected_embedding_weights_multiple_tokens():
    embedding_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    expected = expected_embedding_from_topk([[1, 2]], [[0.25, 0.75]], embedding_weight)
    target = embedding_weight[2] * 0.25 + embedding_weight[3] * 0.75

    assert torch.allclose(expected, target.unsqueeze(0))


def test_expected_embedding_rejects_mismatched_shapes():
    embedding_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    with pytest.raises(ValueError):
        expected_embedding_from_topk([[1, 2]], [[1.0]], embedding_weight)

