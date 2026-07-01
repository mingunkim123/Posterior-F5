# Local Docker Platform

This compose setup runs the local Posterior-F5 MLOps control plane without
putting experiment artifacts inside containers.

## Services

- `api`: FastAPI server inside the compose network at `http://api:8000`
- `worker`: ML-capable queue consumer built from `Dockerfile.worker-gpu`
- `frontend`: React dashboard on `http://127.0.0.1:5174`

The browser should use the frontend URL only. The frontend is built with
`VITE_API_BASE=/api`, and nginx proxies `/api/*` to the internal API service.
This keeps the dashboard on one stable origin and avoids mixing local `.venv`
servers with compose containers.

The default worker image includes torch, torchaudio, transformers, vocos,
soundfile, and the local F5-TTS package so it can run posterior extraction,
actual inference, ASR prediction, and metrics jobs. It can run on CPU, and will
use CUDA when Docker exposes NVIDIA devices to the container.

## Persistent Local Paths

The compose file mounts these host directories into the API container:

- `./mlops_artifacts` for run outputs
- `./data` for local manifests/audio inputs
- `./manifests` for checked-in experiment manifests
- `./platform/config` for dataset and checkpoint registries
- `./ckpts` for checkpoints
- `./posterior_cache` for reusable posterior artifacts

The worker also uses named Docker volumes for model caches:

- `hf_cache` mounted at `/root/.cache/huggingface`
- `torch_cache` mounted at `/root/.cache/torch`

The important output path is:

```text
mlops_artifacts/runs/<run_id>/
```

## Run

From the repository root:

```bash
docker compose down --remove-orphans
docker compose up --build
```

Then open:

```text
http://127.0.0.1:5174
```

Useful checks:

```bash
curl http://127.0.0.1:5174/health
curl http://127.0.0.1:5174/api/datasets
docker compose ps
docker compose logs -f api worker frontend
```

For a fast scaffold-only control-plane smoke test without the ML dependency
image, run the lightweight worker profile instead:

```bash
docker compose --profile lite up --build api frontend worker-lite
```

To check the ML worker runtime after building:

```bash
docker compose run --rm worker python platform/workers/check_ml_runtime.py
```

On an NVIDIA GPU host, use the GPU override so Compose requests device access:

```bash
docker compose -f docker-compose.yml -f platform/docker/docker-compose.gpu.yml up --build
```

## Notes

The API image is intentionally lightweight. It installs the local package
without the full ML dependency stack, so dashboard browsing stays quick. The
default worker is now separated into a heavier ML runtime image and consumes
file-backed jobs from `mlops_artifacts/runs/*/job.json`.
