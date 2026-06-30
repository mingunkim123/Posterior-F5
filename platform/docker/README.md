# Local Docker Platform

This compose setup runs the local Posterior-F5 MLOps control plane without
putting experiment artifacts inside containers.

## Services

- `api`: FastAPI server on `http://127.0.0.1:8000`
- `worker`: queue consumer that runs `platform/workers/run_posterior_f5_pipeline.py`
- `frontend`: React dashboard on `http://127.0.0.1:5174`

## Persistent Local Paths

The compose file mounts these host directories into the API container:

- `./mlops_artifacts` for run outputs
- `./data` for local manifests/audio inputs
- `./ckpts` for checkpoints
- `./posterior_cache` for reusable posterior artifacts

The important output path is:

```text
mlops_artifacts/runs/<run_id>/
```

## Run

From the repository root:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:5174
```

## Notes

The API image is intentionally lightweight. It installs the local package
without the full ML dependency stack, so dashboard browsing and dry-run pipeline
commands stay quick. The local worker currently uses the same image as the API
and consumes file-backed jobs from `mlops_artifacts/runs/*/job.json`. Full GPU
training or inference should later move to a heavier worker image with the
required CUDA/model dependencies.
