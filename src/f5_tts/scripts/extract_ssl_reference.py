"""Extract cached SSL reference features for hybrid Posterior-F5 inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an SSL reference feature cache JSONL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic_dim", type=int, default=0, help="Write deterministic synthetic features for tests.")
    return parser.parse_args()


def iter_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line) if line.startswith("{") else {"audio_path": line}
            audio_path = row.get("audio_path") or row.get("ref_audio")
            row["audio_path"] = audio_path
            row.setdefault("utterance_id", Path(str(audio_path)).stem or f"utt-{index:06d}")
            rows.append(row)
    return rows


def synthetic_features(row: dict[str, Any], dim: int):
    import numpy as np

    seed = sum(ord(char) for char in str(row["utterance_id"])) % 997
    values = np.arange(4 * dim, dtype=np.float32).reshape(4, dim)
    return (values + seed) / max(dim, 1)


def load_ssl_bundle(model_id: str, device: str | None):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return processor, model.to(resolved_device).eval(), resolved_device


def extract_ssl_features(audio_path: str, processor, model, device):
    import numpy as np
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_sample_rate = getattr(processor.feature_extractor, "sampling_rate", sample_rate)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        sample_rate = target_sample_rate

    inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.inference_mode():
        features = model(input_values=input_values).last_hidden_state[0]
    return features.cpu().numpy().astype(np.float32), sample_rate


def main() -> None:
    args = parse_args()
    rows = iter_manifest(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    bundle = None if args.synthetic_dim else load_ssl_bundle(args.model, args.device)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            shard_path = shard_dir / f"{row['utterance_id']}.npz"
            if args.synthetic_dim:
                features = synthetic_features(row, args.synthetic_dim)
                sample_rate = None
                source = "synthetic"
            else:
                processor, model, device = bundle
                features, sample_rate = extract_ssl_features(row["audio_path"], processor, model, device)
                source = args.model

            import numpy as np

            np.savez(shard_path, features=features)
            entry = {
                "utterance_id": row["utterance_id"],
                "audio_path": row["audio_path"],
                "feature_path": shard_path.name,
                "feature_key": "features",
                "num_frames": int(features.shape[0]),
                "ssl_dim": int(features.shape[1]),
                "sample_rate": sample_rate,
                "source": source,
            }
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
