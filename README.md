# StockRank

**Cross-sectional equity return forecasting, leakage-safe validation, and constrained portfolio construction on 16 years of real US market data.**

[![CI](https://github.com/adwitiyashukla/stockrank/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/stockrank/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Live demo](https://img.shields.io/badge/live%20demo-research%20console-4C8DFF.svg)](https://stockrank.streamlit.app)

### [Open the live research console](https://stockrank.streamlit.app)

No install required. Signal quality, strategy performance, the significance tests and the latest ranked long/short screen, all rendered from the artifacts this repository produces.

---

## The problem this actually solves

Most stock-prediction projects forecast tomorrow's **price**, report an R-squared above 0.99, and have discovered nothing. The best forecast of tomorrow's price is today's price; a model can reproduce it perfectly while containing zero information.

This project forecasts the **cross-section**. Given 250 US large caps today, which will outperform the others over the next month? That is the question a systematic equity desk actually asks, and it is hard: the honest ceiling is an information coefficient somewhere between 0.02 and 0.05, and everything after that is a fight against transaction costs and your own overfitting.

|  | Price-level prediction | This project |
|---|---|---|
| Target | `close[t+1]` | forward return relative to the cross-section |
| Metric | RMSE, R-squared | information coefficient, quantile spread, deflated Sharpe |
| Typical headline | R-squared 0.998 | IC 0.02 to 0.05 |
| Economic value | none | a tradable dollar- and beta-neutral long/short book |

---

## What is in here

**Real data, honestly assembled.** 16 years of split- and dividend-adjusted daily bars for a **point-in-time reconstructed S&P 500**, so the universe on any past date is what it actually was rather than what survived. Plus real Fama-French factors from the Ken French Data Library for attribution.

**A validation scheme that cannot leak.** Purged, embargoed walk-forward splits. A test rebuilds every feature on a panel whose future has been corrupted and asserts nothing in the past moved. A second test plants exactly zero alpha in a simulator and fails the build if any model finds signal in it.

**Models that are actually compared.** Ridge and ElasticNet baselines, LightGBM with a Huber objective, a GRU over 40-day factor sequences, a small transformer, and a rank-average ensemble. The linear baseline is there because if four hundred boosted trees cannot beat it, the extra machinery is decoration.

**Portfolio construction with the constraint everyone forgets.** Equal dollars long and short is *not* market neutral. An early version of this engine lost 28% a year with a positive information coefficient purely because the book carried a large negative beta. The fix, an orthogonal projection onto the zero-beta subspace, is in `portfolio/construction.py` and the story is in [METHODOLOGY.md](docs/METHODOLOGY.md).

**Statistics that try to kill the result.** Deflated Sharpe ratio, probability of backtest overfitting via CSCV, stationary bootstrap intervals, a randomisation test, and a Fama-French attribution with Newey-West errors. Plus a break-even cost curve, because a signal that needs sub-3bp execution is not a research result.

**An econometrics study alongside the ML.** Return levels are close to unpredictable; return *variance* is not. GARCH(1,1) with Student-t errors, HAR-RV and RiskMetrics EWMA are compared out of sample under QLIKE.

**Things you can click.** A FastAPI scoring service and a Streamlit research console.

---

## Results

<!-- RESULTS_BLOCK -->

Universe of **250 liquid US large caps**, **2010-01-04 to 2026-06-29** (4,146 trading days, 952,019 observations). Six purged walk-forward folds. Every figure below is out of sample and net of commission, slippage and short financing.

| Model | Mean IC | t (Newey-West) | Ann. return | Sharpe | Max DD | Beta |
|---|---|---|---|---|---|---|
| `ridge` | +0.0192 | +1.31 | +8.38% | 0.83 | -14.87% | 0.091 |
| `elasticnet` | +0.0191 | +1.29 | +7.50% | 0.76 | -14.69% | 0.106 |
| `lightgbm` | +0.0128 | +1.00 | +10.83% | 1.06 | -10.52% | 0.128 |
| `ensemble` | +0.0101 | +0.76 | -0.22% | -0.02 | -22.05% | 0.077 |
| `gru` | +0.0073 | +0.58 | +1.83% | 0.19 | -22.13% | 0.087 |
| `factor_composite` | -0.0127 | -0.52 | -1.52% | -0.15 | -23.21% | -0.011 |

`factor_composite` is the zero-parameter benchmark built from published anomalies. It fits nothing, so it cannot overfit, and every learned model is judged against it.

**Best model by Sharpe: `lightgbm`**

- Deflated Sharpe Ratio **0.623** (observed annualised Sharpe 1.06 against a selection-adjusted threshold of 0.94 for 60 trials)
- Stationary bootstrap 95% CI for the Sharpe: [0.29, 1.83], P(Sharpe <= 0) = 0.003
- Probability of backtest overfitting: **0.029** (CSCV over 6 candidate strategies)
- Fama-French six-factor alpha: +6.23% annualised, t = 1.67 under Newey-West errors, R2 = 0.180
- Net beta is **exactly zero at every rebalance** by construction, against the rolling beta estimates used to build the book. The full-sample regression beta against the S&P 500 is 0.128, and that gap is the residual left by estimation error in rolling betas rather than a deliberate market bet
- Point-in-time universe coverage: 79.6% (647 of 813 historical index members priced)

**Reading this honestly.** The raw evidence is positive: Sharpe 1.06, Newey-West t of 2.67, bootstrap P(Sharpe <= 0) of 0.003, and a probability of backtest overfitting of 0.029, which is low enough to say the selection process is not simply picking noise. Two things stop this being a claim of significance. The **deflated Sharpe of 0.623** falls short of the conventional 0.95: measured against the 60 configurations tried, the observed Sharpe sits only just above what a worthless strategy would be expected to reach. And the six-factor alpha carries a t-statistic of 1.67, so a meaningful part of the return is exposure that can be bought cheaply elsewhere.

That conclusion is the deliverable. Price and volume features alone carry very little cross-sectional information in US large caps, and a research pipeline is only worth having if it is capable of saying so instead of tuning until the number looks good.

![Out-of-sample equity curves](reports/figures/baseline/equity_curves.png)

![Information coefficient by model](reports/figures/baseline/ic_by_model.png)

![Information coefficient by fold](reports/figures/baseline/ic_by_fold.png)

Full write-up with every figure: **[reports/RESULTS_baseline.md](reports/RESULTS_baseline.md)**.
Click through it interactively: **[stockrank.streamlit.app](https://stockrank.streamlit.app)**.

---

## Quickstart

```bash
git clone https://github.com/adwitiyashukla/stockrank.git
cd stockrank

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 1. Download real market data (about 2 minutes, cached and resumable)
python scripts/fetch_data.py --config configs/default.yaml

# 2. Run the full research pipeline (about 15 minutes on a CPU-only laptop)
python -m stockrank.cli run --config configs/default.yaml

# 3. Explore
streamlit run dashboard/app.py                        # research console
uvicorn stockrank.api.main:app --port 8000         # API, docs at /docs
```

No API keys are required. Everything comes from public sources.

**In a hurry?** `make run-fast` uses a reduced configuration that finishes in a couple of minutes.

**No internet?** `make control` runs the two synthetic control experiments, which need no network at all.

### Hardware

Built and validated on **8 GB RAM, CPU only, no GPU**. The default configuration runs comfortably inside that budget: the sequence models read from a single 146 MB feature tensor rather than materialising every training window, which is what keeps a GRU trainable on a laptop. `configs/real_market.yaml` widens the universe to 450 names and wants 16 GB.

<!-- RUNTIME_BLOCK -->

Measured wall clock for the run reported above (8 GB RAM, CPU only): **9.5 minutes** end to end (data 0s, features 20s, training 456s, backtest 8s, evaluation 46s, explain 12s, volatility 30s), with market data already cached.

---

## Architecture

```
                    Yahoo Finance          Wikipedia            Ken French
                    daily OHLCV       index change log       FF5 + momentum
                          |                   |                     |
                          v                   v                     v
                 +------------------------------------------------------+
                 |  data/    point-in-time universe, resumable cache,    |
                 |           quality filters, liquidity screen           |
                 +------------------------------------------------------+
                                          |
                                          v
                 +------------------------------------------------------+
                 |  features/  36 factors, backward looking only         |
                 |             cross-sectional winsorise + z-score       |
                 |             labels: the only forward shift in the repo|
                 +------------------------------------------------------+
                                          |
                                          v
                 +------------------------------------------------------+
                 |  validation/  purged walk-forward + embargo           |
                 +------------------------------------------------------+
                                          |
                          +---------------+---------------+
                          v                               v
              +----------------------+       +-------------------------+
              |  models/             |       |  models/volatility.py   |
              |  ridge, elasticnet,  |       |  GARCH, HAR-RV, EWMA    |
              |  lightgbm, gru,      |       |  compared under QLIKE   |
              |  transformer, ensemble|      +-------------------------+
              +----------------------+
                          |
                          v
                 +------------------------------------------------------+
                 |  portfolio/  rank long/short, mean-variance (Ledoit-  |
                 |              Wolf), risk parity; dollar AND beta      |
                 |              neutral; volatility targeting            |
                 +------------------------------------------------------+
                                          |
                                          v
                 +------------------------------------------------------+
                 |  backtest/   commission, slippage, borrow, turnover cap|
                 +------------------------------------------------------+
                                          |
                                          v
                 +------------------------------------------------------+
                 |  evaluation/  IC, deflated Sharpe, PBO, bootstrap,    |
                 |               randomisation, FF attribution (HAC)     |
                 +------------------------------------------------------+
                          |                    |                  |
                          v                    v                  v
                   reports/RESULTS.md    Streamlit console    FastAPI service
```

---

## Repository layout

```
src/stockrank/
  config.py              typed config; an experiment is one YAML file plus a seed
  experiment.py          end-to-end orchestration
  cli.py                 typer CLI: run, fetch, report, models, list-runs
  data/
    universe.py          point-in-time S&P 500 reconstruction
    providers.py         resumable Yahoo Finance ingestion with per-ticker cache
    factors.py           Fama-French five factors plus momentum
    simulator.py         synthetic market with planted alpha (test fixture only)
    loader.py            cleaning, quality filters, liquidity screen
  features/
    technical.py         36 factors, no forward shift anywhere
    labels.py            the only place a negative shift is allowed
    cross_section.py     per-date winsorise, z-score, rank, sector neutralise
    pipeline.py          wide matrices to modelling frame
  validation/splitters.py   purged walk-forward and purged k-fold
  models/
    linear.py gbm.py sequence.py ensemble.py volatility.py persistence.py
    trainer.py           the walk-forward loop
  portfolio/construction.py  sizing, constraints, beta projection, vol targeting
  backtest/engine.py     costs, borrow, turnover cap, exposure tracking
  evaluation/            metrics, performance, significance, attribution
  explain/               SHAP and feature-importance stability
  reporting/             matplotlib figures and the generated results document
  api/                   FastAPI service
dashboard/               Streamlit research console
tests/                   leakage guards, null control, accounting invariants
configs/                 default, fast, leakage_control, signal_recovery, real_market
scripts/                 fetch_data.py, factor_diagnostics.py
docs/METHODOLOGY.md      design decisions, including the ones that failed
```

---

## Three decisions worth defending

**1. The universe is reconstructed point in time.** Running today's index members back 16 years tells the backtest in advance which firms survive. The change log expands 503 current names into **813 distinct historical members** including Alcoa, Aetna, Monsanto and First Republic. Roughly 20% of them cannot be priced because Yahoo drops delisted securities, and that residual bias is measured and reported on every run rather than hidden.

**2. Two attractive targets were tested and rejected.** A beta-adjusted residual target produced the strongest information coefficients in the entire study, and they were spurious: estimation error in rolling beta induces a mechanical negative correlation with the residual, which a model exploits while earning nothing. A volatility-scaled target fails for a simpler reason, since dividing by trailing volatility anti-correlates the target with every volatility feature by construction. Both experiments are reproducible via `scripts/factor_diagnostics.py`.

**3. Dollar neutral is not market neutral.** See above, and `tests/test_portfolio.py::test_beta_neutrality_removes_net_beta`.

Full reasoning in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## Testing

```bash
pytest -q                      # full suite
pytest -q -m "not slow"        # skip the two end-to-end control experiments
ruff check src tests
```

The suite runs entirely on the synthetic simulator, so it needs no network and produces identical numbers on every machine. The tests worth reading:

| Test | What it proves |
|---|---|
| `test_leakage.py::test_features_are_causal` | Corrupting the last 60 days changes nothing stamped earlier. A single `shift(-1)` fails this. |
| `test_leakage.py::test_purged_splits_never_overlap` | Train and test share no dates, and the gap covers the full label horizon. |
| `test_null_control.py::test_null_alpha_gives_zero_ic` | With zero planted alpha, no model may report a reliable IC. |
| `test_null_control.py::test_planted_alpha_is_recovered` | The mirror image, so a pipeline that always returns zero cannot pass trivially. |
| `test_portfolio.py::test_beta_neutrality_removes_net_beta` | The projection actually zeroes net beta. |
| `test_backtest.py::test_reversed_signal_loses_money` | The accounting has the right sign. |

---

## Deploy

The research console is deployed at **[stockrank.streamlit.app](https://stockrank.streamlit.app)**, served straight from this repository's `demo_artifacts/`.

To run the whole stack locally:

```bash
docker compose -f docker/docker-compose.yml up --build     # API on :8000, dashboard on :8501
```

The Streamlit console deploys to Streamlit Community Cloud directly from this repository; point it at `dashboard/app.py` and it reads the committed `demo_artifacts/`. Step by step instructions, plus the API endpoint reference, are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Limitations

Stated plainly, because a results document that lists none is not credible.

1. **Residual survivorship bias.** About 20% of historically correct index members cannot be priced. The surviving sample is mildly favourable.
2. **No market impact model.** Costs are linear in turnover; real impact is concave in participation rate.
3. **Close-to-close execution.** Fills are assumed at the close with a one-day lag, which ignores auction risk.
4. **Price and volume only.** No fundamentals, revisions, short interest or options-implied data. These are the obvious next inputs and would likely matter more than any modelling change.
5. **One market.** US large caps. Nothing here should be assumed to transfer without re-testing.
6. **Multiple testing is bounded, not eliminated.** The deflated Sharpe ratio uses an assumed trial count, and human judgement about which configurations to try is itself a form of fitting.

**This is research code. It is not investment advice and it places no orders.**

---

## References

Lopez de Prado (2018) *Advances in Financial Machine Learning* &middot; Bailey and Lopez de Prado (2014) *The Deflated Sharpe Ratio* &middot; Bailey, Borwein, Lopez de Prado and Zhu (2017) *The Probability of Backtest Overfitting* &middot; Corsi (2009) *HAR-RV* &middot; Amihud (2002) *Illiquidity and Stock Returns* &middot; Fama and French (2015) *A Five-Factor Asset Pricing Model* &middot; Frazzini and Pedersen (2014) *Betting Against Beta* &middot; Ledoit and Wolf (2004) *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*

## License

MIT. See [LICENSE](LICENSE).
