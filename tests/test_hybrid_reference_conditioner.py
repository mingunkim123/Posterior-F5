import pytest


torch = pytest.importorskip("torch")

from f5_tts.model.hybrid_reference_conditioner import HybridReferenceConditioner


def test_alpha_one_returns_soft_text_condition():
    conditioner = HybridReferenceConditioner()
    soft_text = torch.ones(1, 2, 3)
    ssl = torch.zeros(1, 2, 3)

    mixed = conditioner(soft_text, ssl, alpha=1.0)

    assert torch.allclose(mixed, soft_text)


def test_alpha_zero_returns_ssl_condition():
    conditioner = HybridReferenceConditioner()
    soft_text = torch.ones(1, 2, 3)
    ssl = torch.zeros(1, 2, 3)

    mixed = conditioner(soft_text, ssl, alpha=0.0)

    assert torch.allclose(mixed, ssl)


def test_shape_mismatch_raises_error():
    conditioner = HybridReferenceConditioner()

    with pytest.raises(ValueError):
        conditioner(torch.ones(1, 2, 3), torch.ones(1, 3, 3), alpha=0.5)


def test_entropy_gate_moves_high_entropy_toward_ssl():
    conditioner = HybridReferenceConditioner(entropy_threshold=1.0)
    soft_text = torch.ones(1, 2, 3)
    ssl = torch.zeros(1, 2, 3)

    low_entropy = conditioner(soft_text, ssl, entropy=torch.full((1, 2), 0.1))
    high_entropy = conditioner(soft_text, ssl, entropy=torch.full((1, 2), 3.0))

    assert low_entropy.mean() > high_entropy.mean()
    assert low_entropy.mean() > 0.5
    assert high_entropy.mean() < 0.5
