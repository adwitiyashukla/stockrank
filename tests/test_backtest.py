"""Backtest accounting.

The toy market has a fixed per-name drift. A model that knows the drift should
make money; the reverse of that model should lose it; and costs must strictly
reduce the result. If any of those fail, the accounting has a sign or a scaling
error somewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.config import BacktestConfig, PortfolioConfig

HORIZON = 5
N_TICKERS = 40
N_DAYS = 500


@pytest.fixture(scope="module")
def toy_market():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]

    drift = rng.normal(0.0, 0.0006, N_TICKERS)          # the thing to be discovered
    noise = rng.normal(0.0, 0.012, (N_DAYS, N_TICKERS))
    rets = pd.DataFrame(drift + noise, index=dates, columns=tickers)

    close = 100 * (1 + rets).cumprod()
    fwd = close.shift(-(1 + HORIZON)) / close.shift(-1) - 1

    # A noisy but informative signal about the drift.
    signal = pd.DataFrame(
        drift + rng.normal(0.0, 0.0004, (N_DAYS, N_TICKERS)), index=dates, columns=tickers
    )

    long = signal.stack().rename("pred_test").reset_index()
    long.columns = ["date", "ticker", "pred_test"]
    fl = fwd.stack(future_stack=True).rename("fwd_return").reset_index()
    fl.columns = ["date", "ticker", "fwd_return"]
    preds = long.merge(fl, on=["date", "ticker"], how="left")
    preds["target"] = preds["fwd_return"]
    preds["beta_raw"] = np.tile(rng.normal(1.0, 0.15, N_TICKERS), N_DAYS)[: len(preds)]
    return preds, rets


def _run(preds, rets, **kw):
    pcfg = PortfolioConfig(
        n_long=8, n_short=8, rebalance_days=HORIZON, vol_target_annual=0.10,
        **{k: v for k, v in kw.items() if k in PortfolioConfig.model_fields},
    )
    bcfg = BacktestConfig(
        **{k: v for k, v in kw.items() if k in BacktestConfig.model_fields}
    )
    return run_backtest(preds, "pred_test", pcfg, bcfg, daily_returns=rets, label_horizon=HORIZON)


def test_informative_signal_makes_money(toy_market):
    preds, rets = toy_market
    res = _run(preds, rets, cost_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0,
               beta_neutral=False)
    assert res.returns.mean() > 0
    assert res.equity_curve.iloc[-1] > 1.0


def test_reversed_signal_loses_money(toy_market):
    preds, rets = toy_market
    flipped = preds.copy()
    flipped["pred_test"] = -flipped["pred_test"]
    res = run_backtest(
        flipped, "pred_test",
        PortfolioConfig(n_long=8, n_short=8, rebalance_days=HORIZON, beta_neutral=False),
        BacktestConfig(cost_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0),
        daily_returns=rets, label_horizon=HORIZON,
    )
    assert res.returns.mean() < 0


def test_costs_strictly_reduce_returns(toy_market):
    preds, rets = toy_market
    free = _run(preds, rets, cost_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0,
                beta_neutral=False)
    costly = _run(preds, rets, cost_bps=50.0, slippage_bps=25.0, borrow_bps_annual=300.0,
                  beta_neutral=False)
    assert costly.returns.mean() < free.returns.mean()
    assert costly.costs.sum() > free.costs.sum() >= 0


def test_returns_are_not_artificially_smooth(toy_market):
    """Guards against the bug this engine was rewritten to fix.

    Booking one lump return per holding period and spreading it evenly across the
    days produces a series where consecutive days are identical, which inflates
    the annualised Sharpe by roughly sqrt(horizon). Daily marking must produce a
    series with many distinct values and no such smoothing.
    """
    preds, rets = toy_market
    res = _run(preds, rets, beta_neutral=False)
    r = res.returns.dropna()
    assert r.nunique() > 0.9 * len(r), "returns look smoothed across the holding period"
    # A repeated-value series has an autocorrelation near 1 at lag 1.
    assert abs(r.autocorr(lag=1)) < 0.5


def test_beta_neutrality_holds_in_the_backtest(toy_market):
    preds, rets = toy_market
    res = _run(preds, rets, beta_neutral=True)
    assert res.exposure["net_beta"].abs().max() < 1e-6


def test_gross_exposure_respects_leverage(toy_market):
    preds, rets = toy_market
    res = _run(preds, rets, gross_leverage=2.0, max_vol_scalar=1.0, beta_neutral=False)
    assert res.exposure["gross_exposure"].max() <= 2.0 + 1e-6


def test_turnover_cap_is_respected(toy_market):
    preds, rets = toy_market
    res = _run(preds, rets, turnover_cap_daily=0.02, beta_neutral=False)
    assert res.turnover.max() <= 0.02 * HORIZON + 1e-6


def test_equity_curve_matches_the_return_series(toy_market):
    preds, rets = toy_market
    res = _run(preds, rets, beta_neutral=False)
    expected = float((1 + res.returns.fillna(0)).prod())
    assert abs(res.equity_curve.iloc[-1] - expected) < 1e-9
