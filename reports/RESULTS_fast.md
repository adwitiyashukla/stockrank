# Results: `fast`

> Every number below is **out of sample**. Models are fitted on a trailing window and scored on a later window they never saw, with the label horizon purged and an embargo applied at the boundary. Returns are net of commission, slippage and short financing.

## 1. Dataset

- **Source**: yfinance, universe `sp500_pit`
- **Period**: 2016-01-04 to 2026-06-29 (2,636 trading days)
- **Cross-section**: 80 names, 8 GICS sectors, 202,092 rows
- **Point-in-time coverage**: 612 of 714 historical index members priced (85.7%). The remainder are delistings and acquisitions that Yahoo Finance no longer serves; this residual survivorship bias is discussed in section 8.
- **Features**: 36 cross-sectionally normalised factors, target = forward_excess_return over 5 days with a 1-day execution lag

## 2. Forecast quality

The information coefficient is the daily cross-sectional Spearman correlation between forecast and realised forward return. For daily equity signals, a mean IC of 0.02 to 0.04 is the range that supports a real strategy; anything above 0.10 in a backtest of this type is nearly always a leak.

| model | mean IC | ICIR | t (Newey-West) | Q5-Q1 | OOS R2 | fit (s) |
| --- | --- | --- | --- | --- | --- | --- |
| ensemble | 0.0080 | 0.6466 | 0.6707 | -0.0041 | -21.8577 | 0.0000 |
| lightgbm | 0.0042 | 0.3863 | 0.4074 | -0.0037 | -0.0006 | 2.0000 |
| elasticnet | 0.0029 | 0.2613 | 0.2612 | -0.0013 | -0.0014 | 12.1000 |
| gru | 0.0028 | 0.2289 | 0.2221 | -0.0034 | -0.0051 | 101.8000 |
| ridge | -0.0030 | -0.2652 | -0.2605 | -0.0019 | -0.0026 | 0.4000 |

![Information coefficient by model](figures\fast\ic_by_model.png)

Stability across folds matters more than the average. A signal that is strong in two folds and negative in three is not a signal.

![IC by fold](figures\fast\ic_by_fold.png)

![Quantile ladder](figures\fast\quantile_ladder.png)

## 3. Strategy performance

Construction: rank_long_short, 12 long and 12 short, dollar neutral, gross leverage 2.0x, targeting 10% annualised volatility, rebalanced every 5 days. Costs: 5 bp commission plus 2 bp slippage per unit of turnover, plus 50 bp annual borrow on the short leg.

| strategy | ann. return | ann. vol | Sharpe | Sortino | max DD | Calmar | t (NW) | beta to mkt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | -0.055 | 0.084 | -0.661 | -0.997 | -0.297 | -0.186 | -0.622 | 0.025 |
| elasticnet | -0.018 | 0.081 | -0.226 | -0.343 | -0.221 | -0.083 | -0.216 | 0.007 |
| lightgbm | -0.280 | 0.089 | -3.152 | -4.575 | -0.597 | -0.469 | -2.977 | -0.086 |
| gru | -0.252 | 0.084 | -2.998 | -4.337 | -0.541 | -0.465 | -2.695 | -0.057 |
| ensemble | -0.227 | 0.086 | -2.642 | -3.364 | -0.519 | -0.437 | -2.424 | -0.076 |
| benchmark_buy_hold | 0.204 | 0.152 | 1.339 | 1.771 | -0.188 | 1.088 | 2.560 |  |

![Equity curves](figures\fast\equity_curves.png)

![Drawdown](figures\fast\drawdown.png)

![Rolling Sharpe](figures\fast\rolling_sharpe.png)

![Monthly returns](figures\fast\monthly_returns.png)

## 4. Is the result real?

Best strategy by Sharpe: **elasticnet**

- **Deflated Sharpe Ratio**: 0.005. Observed annualised Sharpe -0.23, versus a selection-adjusted threshold of 1.28 implied by 40 effective trials. Values above 0.95 are conventionally treated as evidence the result is not a product of the search.
- **Stationary bootstrap 95% CI for the Sharpe**: [-2.38, 2.40], P(Sharpe <= 0) = 0.604.
- **Probability of backtest overfitting** (CSCV over 5 candidate strategies, 70 splits): 0.171. Below 0.5 means the in-sample winner tends to stay above median out of sample.
- **Randomisation test**: shuffling forecasts within each date 100 times gives p = 0.327.

### Factor attribution

Regression of daily strategy excess returns on the Fama-French five factors plus momentum, with Newey-West standard errors. The intercept is what is left after cheap factor exposure is stripped out.

| strategy | alpha (ann.) | t(alpha) HAC | R2 | b_mkt | b_smb | b_hml | b_mom |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | -0.103 | -1.110 | 0.006 | 0.011 | 0.036 | -0.036 | 0.006 |
| elasticnet | -0.062 | -0.702 | 0.004 | -0.005 | 0.043 | -0.036 | 0.007 |
| lightgbm | -0.309 | -3.364 | 0.028 | -0.077 | 0.003 | 0.000 | -0.014 |
| gru | -0.288 | -3.126 | 0.026 | -0.037 | -0.005 | 0.019 | 0.000 |
| ensemble | -0.259 | -2.889 | 0.031 | -0.059 | 0.002 | 0.006 | -0.006 |

![Factor exposures](figures\fast\factor_exposures.png)

## 5. Implementability

Sharpe ratio as a function of the assumed one-way cost. The break-even point is the honest test of whether a signal is tradable outside a simulation.

| cost (bp) | ann. return | Sharpe |
| --- | --- | --- |
| 0.000 | -0.018 | -0.226 |
| 2.000 | -0.031 | -0.382 |
| 5.000 | -0.050 | -0.616 |
| 10.000 | -0.081 | -1.004 |
| 15.000 | -0.113 | -1.386 |
| 20.000 | -0.145 | -1.762 |
| 30.000 | -0.208 | -2.480 |
| 50.000 | -0.334 | -3.747 |

![Cost sensitivity](figures\fast\cost_sensitivity.png)

## 6. What the model uses

![Feature importance](figures\fast\feature_importance.png)

## 7. Volatility forecasting study

Return levels are close to unpredictable; return variance is not. GARCH(1,1) with Student-t errors, HAR-RV and RiskMetrics EWMA are compared out of sample using QLIKE, which is robust to the fact that true variance is unobservable.

| metric | mean across names |
| --- | --- |
| qlike_ewma | 2.788058 |
| mse_ewma | 0.000012 |
| qlike_har | 1.706542 |
| mse_har | 0.000010 |
| qlike_garch | 1.690934 |
| mse_garch | 0.000009 |

![Volatility models](figures\fast\volatility_models.png)

## 8. Limitations

Stated plainly, because a results document that lists none is not credible.

1. **Residual survivorship bias.** Index membership is reconstructed point in time, but 14% of historical members cannot be priced because Yahoo Finance drops delisted securities. The surviving sample is therefore mildly favourable, and the true out-of-sample edge is likely a little lower than reported.
2. **No market impact model.** Costs are linear in turnover. At institutional size, impact is concave in participation rate and would bite harder than a flat basis-point charge.
3. **Close-to-close execution.** Fills are assumed at the closing price with a one-day lag. Real execution against the close carries auction risk not modelled here.
4. **Price and volume data only.** No fundamentals, no analyst revisions, no short interest, no options-implied information. Those are the natural next inputs.
5. **One market, one regime set.** US large caps only. The result should not be assumed to transfer to small caps or non-US markets without re-testing.
6. **Multiple testing is bounded, not eliminated.** The deflated Sharpe ratio uses an assumed trial count. Every configuration ever run adds to the true count, and human judgement about which configurations to try is itself a form of fitting.

## 9. Reproducing this run

```bash
pip install -e ".[all]"
python scripts/fetch_data.py --config configs/fast.yaml   # or your config
python -m alpha_engine.cli run --config configs/fast.yaml
```

Wall clock on the reference machine (8 GB RAM, CPU only): data 62s, features 9s, training 137s, backtest 6s, total 301s. Seed `42`.
