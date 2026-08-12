# StockRank

Cross-sectional stock ranking on the S&P 500. Forecasts which stocks outperform the rest over the next month, builds a dollar- and beta-neutral long/short book, and tests whether the result survives trading costs and multiple testing.

[![CI](https://github.com/adwitiyashukla/stockrank/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/stockrank/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/demo-Hugging%20Face%20Space-blue.svg)](https://huggingface.co/spaces/adwitiyashukla/stockrank)

Live demo: [huggingface.co/spaces/adwitiyashukla/stockrank](https://huggingface.co/spaces/adwitiyashukla/stockrank)

## Problem

Predicting tomorrow's price is easy and worthless, because today's price is already a near-perfect forecast. This ranks the cross-section instead: given 250 stocks today, which beat the others over the next 21 days. The metric is the information coefficient, the daily cross-sectional rank correlation between forecast and realised return, not R-squared.

## Data

- Daily split- and dividend-adjusted OHLCV from Yahoo Finance, 2010-01-04 to 2026-06-29
- Point-in-time S&P 500 membership rebuilt by walking the index change log backwards: 813 distinct historical members instead of the 503 that survive today
- 647 of 813 priced (79.6%). Yahoo drops most delisted and acquired names, so some survivorship bias remains
- Fama-French 5 factors plus momentum from the Ken French Data Library
- Universe capped at the 250 most liquid names by median dollar volume, 952,019 rows over 4,146 trading days

## Method

- 36 factors from price and volume: momentum, volatility, trend, RSI, MACD, liquidity, Amihud illiquidity, rolling CAPM beta and residual volatility
- Winsorised at 1% and z-scored within each date
- Target: 21-day forward return, cross-sectionally demeaned, entered one day after the signal
- Purged walk-forward CV, 6 folds, 22-day purge and 21-day embargo
- Models: ridge, elastic net, LightGBM with Huber loss, GRU over 40-day sequences, rank-average ensemble, and a zero-parameter factor composite as benchmark
- Portfolio: 25 long, 25 short, dollar and beta neutral in a single least-squares residualisation, 10% annual volatility target
- Costs: 5bp commission, 2bp slippage, 50bp annual borrow on the short leg
- GARCH(1,1), HAR-RV and EWMA compared out of sample under QLIKE

Horizon chosen from `scripts/factor_diagnostics.py`, which screens every factor at 1, 5, 10, 21 and 63 days. Mean absolute IC rises from 0.005 at 1 day to 0.018 at 63 days. 21 days gives three times more independent observations than 63 for a similar IC.

Two targets were tried and dropped. A beta-adjusted residual target gave the strongest ICs in the study (`beta_63` at -0.084, t = -4.1) but estimation error in rolling beta is negatively correlated with the residual by construction, so a model exploits it and earns nothing. A volatility-scaled target is anti-correlated with every volatility factor through its own denominator. Both are reproducible from the diagnostics script.

## Results

All out of sample, net of costs.

| Model | Mean IC | t (NW) | Ann. return | Sharpe | Max DD | Beta |
|---|---|---|---|---|---|---|
| `ridge` | +0.0192 | +1.31 | +8.38% | 0.83 | -14.87% | 0.091 |
| `elasticnet` | +0.0191 | +1.29 | +7.50% | 0.76 | -14.69% | 0.106 |
| `lightgbm` | +0.0128 | +1.00 | +10.83% | 1.06 | -10.52% | 0.128 |
| `ensemble` | +0.0101 | +0.76 | -0.22% | -0.02 | -22.05% | 0.077 |
| `gru` | +0.0073 | +0.58 | +1.83% | 0.19 | -22.13% | 0.087 |
| `factor_composite` | -0.0127 | -0.52 | -1.52% | -0.15 | -23.21% | -0.011 |

`factor_composite` fits nothing. It is a fixed combination of published anomalies (momentum, one-month reversal, low volatility, betting against beta, illiquidity), so it cannot overfit and every trained model is measured against it.

Best model is `lightgbm`:

- Deflated Sharpe 0.623, against a selection-adjusted threshold of 0.94 for 60 trials
- Bootstrap 95% CI for the Sharpe [0.29, 1.83], P(Sharpe <= 0) = 0.003
- Probability of backtest overfitting 0.029 by CSCV
- Fama-French six-factor alpha +6.23% a year, t = 1.67 under Newey-West errors, R2 = 0.180
- Net beta is zero at every rebalance by construction. Full-sample regression beta against the index is 0.128, which is estimation error in the rolling betas, not a market bet
- Break-even cost roughly 55bp per unit turnover

Sharpe 1.06 with a Newey-West t of 2.67 is positive, but the deflated Sharpe of 0.623 is below the usual 0.95 bar, so it does not clear multiple testing across the 60 configurations tried. The six-factor alpha t of 1.67 is not significant either. Price and volume alone carry little cross-sectional signal in US large caps.

![Equity curves](reports/figures/baseline/equity_curves.png)

![IC by model](reports/figures/baseline/ic_by_model.png)

![IC by fold](reports/figures/baseline/ic_by_fold.png)

Full numbers and every figure: [reports/RESULTS_baseline.md](reports/RESULTS_baseline.md)

## Install and run

```bash
git clone https://github.com/adwitiyashukla/stockrank.git
cd stockrank
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[all]"

python scripts/fetch_data.py --config configs/default.yaml
python -m stockrank.cli run --config configs/default.yaml

streamlit run dashboard/app.py
uvicorn stockrank.api.main:app --port 8000
```

No API keys needed. `make run-fast` uses a reduced config that finishes in about two minutes. `make control` runs the two synthetic control experiments offline.

Built on 8 GB RAM, CPU only. Full run is 9.2 minutes with data cached. The sequence models read from one 146 MB feature tensor instead of materialising every training window, which is what keeps the GRU trainable on a laptop.

## Layout

```
src/stockrank/
  config.py            typed config, one YAML per experiment
  experiment.py        orchestration
  cli.py               run, rebacktest, fetch, report, models, list-runs
  data/                point-in-time universe, resumable ingestion, simulator, cleaning
  features/            36 factors, cross-sectional normalisation, labels
  validation/          purged walk-forward and purged k-fold
  models/              linear, gbm, sequence, ensemble, volatility, factor composite
  portfolio/           sizing, neutrality constraints, volatility targeting
  backtest/            daily mark-to-market with costs and turnover cap
  evaluation/          IC, performance, deflated Sharpe, PBO, attribution
  explain/             SHAP and importance stability
  reporting/           figures and generated results document
  api/                 FastAPI service
dashboard/             Streamlit console
tests/                 leakage guards, null control, accounting invariants
configs/               default, fast, leakage_control, signal_recovery, real_market
scripts/               fetch_data, factor_diagnostics, prepare_demo
```

## Tests

```bash
pytest
ruff check src tests dashboard scripts
```

57 tests, no network needed. They run on the synthetic simulator so results are identical on any machine.

| Test | Checks |
|---|---|
| `test_leakage.py::test_features_are_causal` | corrupting the last 60 days changes nothing stamped earlier |
| `test_leakage.py::test_purged_splits_never_overlap` | train and test share no dates and the gap covers the label horizon |
| `test_null_control.py::test_null_alpha_gives_zero_information_coefficient` | with zero planted alpha no model may report a reliable IC |
| `test_null_control.py::test_planted_alpha_is_recovered` | with alpha planted the pipeline finds it |
| `test_portfolio.py::test_beta_neutrality_removes_net_beta` | the projection zeroes net beta |
| `test_backtest.py::test_returns_are_not_artificially_smooth` | daily marking, so the Sharpe is not inflated by the holding period |

## Deploy

```bash
docker compose -f docker/docker-compose.yml up --build
```

API on 8000, dashboard on 8501. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Limitations

- About 20% of historical index members cannot be priced, so some survivorship bias remains
- Costs are linear in turnover, no market impact model
- Fills assumed at the close with a one-day lag
- Price and volume only, no fundamentals, revisions or short interest
- US large caps only
- The deflated Sharpe uses an assumed trial count, so multiple testing is bounded but not eliminated

Research code. Not investment advice.

## References

Lopez de Prado (2018) Advances in Financial Machine Learning. Bailey and Lopez de Prado (2014) The Deflated Sharpe Ratio. Bailey, Borwein, Lopez de Prado and Zhu (2017) The Probability of Backtest Overfitting. Corsi (2009) HAR-RV. Amihud (2002) Illiquidity and Stock Returns. Fama and French (2015) A Five-Factor Asset Pricing Model. Frazzini and Pedersen (2014) Betting Against Beta. Ledoit and Wolf (2004) A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.

## License

MIT
