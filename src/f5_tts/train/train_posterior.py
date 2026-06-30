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
    parser.add_argument("--synthetic_smoke", action="store_true", help="Run a tiny synthetic distillation loop.")
    parser.add_argument("--output_dir", default="", help="Override checkpoint output directory.")
    parser.add_argument("--max_steps", type=int, default=None)
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

    posterior_hidden = posterior_encoder(
        batch["posterior_token_ids"],
        batch["posterior_probs"],
        entropy=batch.get("entropy"),
        blank_prob=batch.get("blank_prob"),
        mask=batch.get("posterior_mask"),
    )
    oracle_hidden = oracle_text_embedding(f5_model, batch["oracle_text_tensor"], batch["seq_len"])
    loss = distillation_loss(posterior_hidden, oracle_hidden, mask=batch.get("posterior_mask"))

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def move_batch_to_device(batch, device):
    import torch

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_posterior_encoder(
    *,
    posterior_encoder,
    f5_model,
    dataloader,
    optimizer,
    max_steps: int,
    output_dir: str | Path,
    save_every: int = 1000,
    log_every: int = 50,
    device=None,
) -> dict:
    """Run a checkpointable posterior encoder distillation loop."""

    import torch

    from f5_tts.model.posterior_encoder import save_posterior_encoder_checkpoint

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = device or next(posterior_encoder.parameters()).device
    posterior_encoder.train()
    losses: list[float] = []
    step = 0

    while step < max_steps:
        for batch in dataloader:
            step += 1
            batch = move_batch_to_device(batch, device)
            loss = posterior_distillation_step(posterior_encoder, f5_model, batch, optimizer=optimizer)
            losses.append(float(loss.detach().cpu()))
            if log_every and step % log_every == 0:
                print(f"step={step} loss={losses[-1]:.6f}")
            if save_every and step % save_every == 0:
                save_posterior_encoder_checkpoint(output_path / f"model_{step}.pt", posterior_encoder, optimizer=optimizer, step=step)
            if step >= max_steps:
                break

    save_posterior_encoder_checkpoint(output_path / "model_last.pt", posterior_encoder, optimizer=optimizer, step=step)
    return {
        "steps": step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "checkpoint": str(output_path / "model_last.pt"),
    }


def run_synthetic_smoke(output_dir: str | Path, *, max_steps: int = 20) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from f5_tts.model.posterior_encoder import TopKPosteriorEncoder

    class _TextEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embed = nn.Embedding(8, 4)

        def forward(self, text_tensor, seq_len, drop_text=False):
            del drop_text
            hidden = self.text_embed(text_tensor[:, :seq_len])
            if hidden.shape[1] < seq_len:
                hidden = torch.nn.functional.pad(hidden, (0, 0, 0, seq_len - hidden.shape[1]))
            return hidden

    class _F5(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.text_embed = _TextEmbedding()

    token_ids = torch.tensor([[[1, 0], [2, 0], [3, 0]]], dtype=torch.long).repeat(4, 1, 1)
    probs = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]], dtype=torch.float32).repeat(4, 1, 1)
    oracle = torch.tensor([[1, 2, 3]], dtype=torch.long).repeat(4, 1)
    mask = torch.ones((4, 3), dtype=torch.bool)
    blank_prob = probs[:, :, 1]
    dataset = TensorDataset(token_ids, probs, oracle, mask, blank_prob)

    def _collate(items):
        ids, ps, oracle_text, masks, blanks = zip(*items)
        return {
            "posterior_token_ids": torch.stack(ids),
            "posterior_probs": torch.stack(ps),
            "oracle_text_tensor": torch.stack(oracle_text),
            "seq_len": 3,
            "posterior_mask": torch.stack(masks),
            "blank_prob": torch.stack(blanks),
        }

    encoder = TopKPosteriorEncoder(vocab_size=8, text_dim=4, hidden_dim=8, num_layers=1, blank_id=0)
    f5_model = _F5()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=_collate)
    return train_posterior_encoder(
        posterior_encoder=encoder,
        f5_model=f5_model,
        dataloader=dataloader,
        optimizer=optimizer,
        max_steps=max_steps,
        output_dir=output_dir,
        save_every=max_steps,
        log_every=0,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded posterior training config: {args.config}")
    if args.posterior_manifest:
        print(f"Posterior manifest: {args.posterior_manifest}")
    if args.dry_run:
        print(config)
        return

    if args.synthetic_smoke:
        output_dir = args.output_dir or config.distillation.output_dir
        max_steps = args.max_steps or 20
        print(run_synthetic_smoke(output_dir, max_steps=max_steps))
        return

    raise ValueError(
        "Real posterior training requires constructing the F5 model and dataloader for the target dataset. "
        "The checkpointable distillation loop is available as train_posterior_encoder(); "
        "use --synthetic_smoke to verify the training contract without a full F5 checkpoint."
    )


if __name__ == "__main__":
    main()
