import pytest

torch = pytest.importorskip("torch")

from f5_tts.model.posterior_encoder import (
    TopKPosteriorEncoder,
    distillation_loss,
    load_posterior_encoder_checkpoint,
    save_posterior_encoder_checkpoint,
)


def test_topk_posterior_encoder_output_shape_and_masking():
    encoder = TopKPosteriorEncoder(vocab_size=16, text_dim=8, hidden_dim=12, num_layers=1, blank_id=0)
    token_ids = torch.tensor([[[1, 0], [2, 0], [3, 0]]])
    probs = torch.tensor([[[0.8, 0.2], [0.5, 0.5], [0.9, 0.1]]])
    mask = torch.tensor([[True, True, False]])

    output = encoder(token_ids, probs, mask=mask)

    assert output.shape == (1, 3, 8)
    assert torch.allclose(output[:, 2], torch.zeros_like(output[:, 2]))


def test_topk_posterior_encoder_rejects_bad_shapes():
    encoder = TopKPosteriorEncoder(vocab_size=16, text_dim=8)

    with pytest.raises(ValueError):
        encoder(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2))

    with pytest.raises(ValueError):
        encoder(torch.ones(1, 2, 3, dtype=torch.long), torch.ones(1, 2, 2))


def test_distillation_loss_masks_padding():
    posterior_hidden = torch.tensor([[[1.0], [10.0]]])
    oracle_hidden = torch.tensor([[[3.0], [0.0]]])
    mask = torch.tensor([[True, False]])

    loss = distillation_loss(posterior_hidden, oracle_hidden, mask=mask)

    assert loss == pytest.approx(4.0)


def test_posterior_encoder_checkpoint_round_trip(tmp_path):
    encoder = TopKPosteriorEncoder(vocab_size=16, text_dim=8, hidden_dim=12, num_layers=1, blank_id=0)
    checkpoint = tmp_path / "posterior_encoder.pt"

    save_posterior_encoder_checkpoint(checkpoint, encoder, step=3, extra={"split": "dev"})
    restored, payload = load_posterior_encoder_checkpoint(checkpoint)

    assert payload["step"] == 3
    assert payload["extra"]["split"] == "dev"
    assert restored.proj_out.out_features == 8
    assert restored.blank_id == 0
