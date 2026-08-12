# Scripts

| Script | Purpose |
|---|---|
| `fetch_data.py` | Download and cache the market panel. Resumable, so a failed run only fetches what is still missing. |
| `factor_diagnostics.py` | Univariate IC of every factor at several horizons and target definitions. This is where the 21-day horizon and the two rejected targets come from. |
| `prepare_demo.py` | Package a completed run into `demo_artifacts/` so the dashboard can be deployed from the repo. |

Order:

```bash
python scripts/fetch_data.py --config configs/default.yaml
python scripts/factor_diagnostics.py --config configs/default.yaml
python -m stockrank.cli run --config configs/default.yaml
python scripts/prepare_demo.py --run baseline
```

Changing a portfolio or cost assumption does not need a refit:

```bash
python -m stockrank.cli rebacktest --config configs/default.yaml
```
