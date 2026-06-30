"""Check that the ML worker image has the runtime packages needed by jobs."""

from __future__ import annotations

import importlib
import json


PACKAGES = [
    "torch",
    "torchaudio",
    "transformers",
    "vocos",
    "soundfile",
    "librosa",
    "cached_path",
    "hydra",
    "omegaconf",
    "x_transformers",
    "torchdiffeq",
    "accelerate",
    "wandb",
]


def main() -> None:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        versions[package] = str(getattr(module, "__version__", "installed"))

    import torch

    payload = {
        "packages": versions,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
