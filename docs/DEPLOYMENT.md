# Deployment

## Streamlit Community Cloud

Deployed at <https://stockrank.streamlit.app>.

The dashboard renders artifacts from disk, so `demo_artifacts/` is committed for the hosted app to read.

1. Push the repo to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Create app, deploy from GitHub, repository `adwitiyashukla/stockrank`, branch `main`, main file `dashboard/app.py`.

Streamlit Cloud installs from `requirements.txt`, not `pyproject.toml`. It leaves out torch, shap and yfinance because the hosted app only reads pre-computed artifacts.

Community Cloud allocates about 1 GB. `scripts/prepare_demo.py --tail-days 900` keeps `predictions.parquet` inside that.

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
