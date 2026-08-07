# Scripts

| Script | Purpose |
|---|---|
| `fetch_data.py` | Download and cache the market panel. Resumable: rerunning after a failure only fetches what is still missing. |
| `factor_diagnostics.py` | Screen every feature against forward returns at several horizons and target definitions. This is what the horizon choice in `configs/default.yaml` is based on, and it is where the two rejected target definitions are reproducible. |
| `prepare_demo.py` | Package a completed run into `demo_artifacts/` so the Streamlit console can be deployed from the repository. |
| `update_readme_results.py` | Regenerate the results and runtime blocks in `README.md` from a completed run, so the front page never drifts from the artifacts. |
| `init_repo.ps1` | Build the git history in readable, topic-scoped commits. |

Typical order:

```bash
python scripts/fetch_data.py --config configs/default.yaml
python scripts/factor_diagnostics.py --config configs/default.yaml
python -m stockrank.cli run --config configs/default.yaml
python scripts/prepare_demo.py --run baseline
python scripts/update_readme_results.py --run baseline
```

Changing a portfolio or cost assumption does not require refitting anything:

```bash
python -m stockrank.cli rebacktest --config configs/default.yaml
```
