"""Train a posterior encoder against frozen F5 text embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior encoder distillation trainer.")
    parser.add_argument("--config", default="src/f5_tts/configs/F5TTS_v1_Base_Posterior.yaml")
    parser.add_argument("--posterior_manifest", required=False, help="JSONL posterior cache manifest.")
    parser.add_argument("--checkpoint", required=False, help="Optional posterior encoder checkpoint path.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit before training.")
    return parser.parse_args()


def load_config(path: str | Path):
    from omegaconf import OmegaConf

    return OmegaConf.load(path)


def oracle_text_embedding(f5_model, text_tensor, seq_len, *, drop_text=False):
    """Return the frozen F5 oracle text embedding used as distillation target."""

    with __import__("torch").inference_mode():
        return f5_model.transformer.text_embed(text_tensor, seq_len=seq_len, drop_text=drop_text).detach()


def posterior_distillation_step(posterior_encoder, f5_model, batch, optimizer=None):
    """Run one posterior encoder distillation step.

    Expected batch keys:
    - `posterior_token_ids`: [batch, seq_len, top_k]
    - `posterior_probs`: [batch, seq_len, top_k]
    - `oracle_text_tensor`: F5 token tensor
    - `seq_len`: target conditioning length
    - optional `posterior_mask`
    """

    from f5_tts.model.posterior_encoder import distillation_loss

    posterior_hidden = posterior_encoder(batch["posterior_token_ids"], batch["posterior_probs"], mask=batch.get("posterior_mask"))
    oracle_hidden = oracle_text_embedding(f5_model, batch["oracle_text_tensor"], batch["seq_len"])
    loss = distillation_loss(posterior_hidden, oracle_hidden, mask=batch.get("posterior_mask"))

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded posterior training config: {args.config}")
    if args.posterior_manifest:
        print(f"Posterior manifest: {args.posterior_manifest}")
    if args.dry_run:
        print(config)
        return

    raise NotImplementedError(
        "Full posterior encoder training loop is intentionally staged. "
        "Use posterior_distillation_step() as the first overfit harness."
    )


if __name__ == "__main__":
    main()
