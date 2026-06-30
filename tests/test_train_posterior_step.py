import pytest

torch = pytest.importorskip("torch")

from f5_tts.model.posterior_encoder import load_posterior_encoder_checkpoint
from f5_tts.train.train_posterior import run_synthetic_smoke


def test_synthetic_posterior_training_writes_checkpoint(tmp_path):
    result = run_synthetic_smoke(tmp_path, max_steps=4)

    assert result["steps"] == 4
    assert result["checkpoint"].endswith("model_last.pt")
    encoder, payload = load_posterior_encoder_checkpoint(result["checkpoint"])
    assert payload["step"] == 4
    assert encoder.proj_out.out_features == 4


def test_synthetic_posterior_training_loss_is_finite(tmp_path):
    result = run_synthetic_smoke(tmp_path, max_steps=3)

    assert result["initial_loss"] >= 0.0
    assert result["final_loss"] >= 0.0
