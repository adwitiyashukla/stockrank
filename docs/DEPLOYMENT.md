# Deployment

## Hugging Face Space

Live at <https://huggingface.co/spaces/adwitiyashukla/stockrank>.

The Space is a separate git repo with its own remote at `https://huggingface.co/spaces/adwitiyashukla/stockrank`. It holds a copy of the dashboard plus `demo_artifacts/`, and reads pre-computed artifacts rather than refitting anything.

The Streamlit SDK for Spaces is deprecated, so the Space uses `sdk: docker` with Streamlit pinned in its own Dockerfile. Port must match `app_port` in the README frontmatter, 7860 here.

`predictions.parquet` is 15.8 MB, above the 10 MB threshold, so the Space tracks `*.parquet` with Git LFS.

```bash
cd stockrank-space
git push origin main
```

Pushing triggers a rebuild. Username is the HF account, password is a write token from <https://huggingface.co/settings/tokens>.

## Streamlit Community Cloud

Also deployable there from `dashboard/app.py`, using the repo's own `requirements.txt`. Community Cloud allocates about 1 GB, and `scripts/prepare_demo.py --tail-days 900` keeps `predictions.parquet` inside that.

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

API on <http://localhost:8000/docs>, dashboard on <http://localhost:8501>.

Run the pipeline in the container:

```bash
docker compose -f docker/docker-compose.yml --profile research run --rm research
```

Multi-stage build with CPU-only torch, about 1 GB.

## API

```bash
pip install -e ".[api]"
uvicorn stockrank.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, runs and models loaded |
| `GET /runs` | completed runs on disk |
| `GET /runs/{run}/metrics` | data, model, performance and significance stats |
| `GET /runs/{run}/features` | feature contract the model expects, in order |
| `GET /runs/{run}/screen` | top longs and shorts on the last date in the sample |
| `POST /score` | score a batch of feature vectors and rank them |

```bash
curl -s "localhost:8000/runs/baseline/screen?model=lightgbm&n=10" | python -m json.tool
```

A single score is not meaningful on its own. The model orders a cross-section.

## Scheduled refresh

```cron
0 2 * * 1-5  cd /opt/stockrank && \
  .venv/bin/python scripts/fetch_data.py --config configs/default.yaml && \
  .venv/bin/python -m stockrank.cli run --config configs/default.yaml
```

Ingestion is incremental and resumable.
