# Deployment

## Streamlit Community Cloud (free, gives a public URL)

**Deployed at <https://equity-alpha-engine.streamlit.app>.**

The research console renders artifacts from disk, so the repository ships a small
`demo_artifacts/` bundle that the hosted app reads.

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app**, then **Deploy a public app from GitHub**, and set:
   - Repository: `adwitiyashukla/equity-alpha-engine`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
4. Deploy. The first build takes three to five minutes while dependencies install.

The app lands at roughly `https://equity-alpha-engine.streamlit.app`. Put that
link at the top of the README so a reviewer can see the work without cloning
anything.

**Why a separate `requirements.txt` exists.** Streamlit Cloud installs from it
rather than from `pyproject.toml`, and it deliberately omits `torch`, `shap` and
`yfinance`. The hosted app only reads pre-computed artifacts, so pulling a
600 MB deep learning wheel would slow every build for no benefit.

**Memory.** Community Cloud allocates about 1 GB. Trimming
`predictions.parquet` with `scripts/prepare_demo.py --tail-days 900` keeps the
app comfortably inside that.

---

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>

To run the research pipeline inside the container:

```bash
docker compose -f docker/docker-compose.yml --profile research run --rm research
```

The image is multi-stage and installs the CPU-only build of PyTorch, which keeps
it near 1 GB rather than 6 GB.

---

## Running the API alone

```bash
pip install -e ".[api]"
uvicorn alpha_engine.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, plus which runs and models are loaded |
| `GET /runs` | Completed runs found on disk |
| `GET /runs/{run}/metrics` | Headline data, model, performance and significance statistics |
| `GET /runs/{run}/features` | The exact feature contract the model expects, in order |
| `GET /runs/{run}/screen` | Top longs and shorts on the most recent date in the sample |
| `POST /score` | Score a batch of feature vectors and rank them cross-sectionally |

Example:

```bash
curl -s "localhost:8000/runs/baseline/screen?model=ensemble&n=10" | python -m json.tool
```

The scoring endpoint deliberately warns when a single observation is submitted
on its own. The model is trained to order a cross-section, so an isolated score
carries very little meaning.

---

## Scheduled refresh

The pipeline is a single command, so a nightly refresh is a one-line cron entry:

```cron
0 2 * * 1-5  cd /opt/equity-alpha-engine && \
  .venv/bin/python scripts/fetch_data.py --config configs/default.yaml && \
  .venv/bin/python -m alpha_engine.cli run --config configs/default.yaml
```

Ingestion is incremental and resumable, so a failed night costs one retry rather
than a full re-download.
