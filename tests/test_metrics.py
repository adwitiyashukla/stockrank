"""Statistical machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.evaluation.metrics import daily_ic, ic_summary, matrix_ic, quantile_spread
from alpha_engine.evaluation.performance import performance_stats
from alpha_engine.evaluation.significance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    stationary_bootstrap_sharpe,
)
from alpha_engine.utils.stats import max_drawdown, newey_west_tstat, sharpe_ratio


def test_perfect_forecast_has_ic_of_one():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=50), 20)
    rng = np.random.default_rng(0)
    target = rng.normal(size=len(dates))
    ic = daily_ic(target, target, pd.Series(dates))
    assert np.allclose(ic.to_numpy(), 1.0)


def test_random_forecast_has_ic_near_zero():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=400), 30)
    rng = np.random.default_rng(1)
    ic = daily_ic(rng.normal(size=len(dates)), rng.normal(size=len(dates)), pd.Series(dates))
    assert abs(ic.mean()) < 0.02


def test_matrix_ic_matches_groupby_version():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=120)
    cols = [f"T{i}" for i in range(25)]
    f = pd.DataFrame(rng.normal(size=(120, 25)), index=idx, columns=cols)
    t = f * 0.3 + pd.DataFrame(rng.normal(size=(120, 25)), index=idx, columns=cols)
    fast = matrix_ic(f, t)
    long = f.stack().rename("f").reset_index()
    long.columns = ["date", "ticker", "f"]
    tl = t.stack().rename("t").reset_index()
    tl.columns = ["date", "ticker", "t"]
    m = long.merge(tl, on=["date", "ticker"])
    slow = daily_ic(m["f"], m["t"], m["date"])
    assert abs(fast.mean() - slow.mean()) < 1e-6


def test_sharpe_and_drawdown_are_consistent():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0006, 0.01, 2000),
                  index=pd.bdate_range("2015-01-01", periods=2000))
    stats = performance_stats(r)
    assert abs(stats["sharpe"] - sharpe_ratio(r)) < 1e-9
    assert stats["max_drawdown"] <= 0
    assert abs(stats["max_drawdown"] - max_drawdown((1 + r).cumprod())) < 1e-9


def test_newey_west_widens_with_autocorrelation():
    rng = np.random.default_rng(4)
    iid = pd.Series(rng.normal(0.05, 1.0, 1500))
    ar = iid.copy()
    for i in range(1, len(ar)):
        ar.iloc[i] = 0.7 * ar.iloc[i - 1] + iid.iloc[i]
    # Positive serial correlation inflates the naive t; the HAC version corrects it.
    naive = ar.mean() / ar.std(ddof=1) * np.sqrt(len(ar))
    assert abs(newey_west_tstat(ar)) < abs(naive)


def test_expected_max_sharpe_grows_with_trials():
    v = 0.001
    assert expected_max_sharpe(2, v) < expected_max_sharpe(20, v) < expected_max_sharpe(200, v)


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0007, 0.009, 2000))
    few = deflated_sharpe_ratio(r, n_trials=1)["deflated_sharpe"]
    many = deflated_sharpe_ratio(r, n_trials=500)["deflated_sharpe"]
    assert many < few


def test_pbo_is_high_for_pure_noise():
    rng = np.random.default_rng(6)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1200, 12)))
    pbo = probability_of_backtest_overfitting(noise, n_splits=6)["pbo"]
    assert 0.25 < pbo < 0.85


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    out = stationary_bootstrap_sharpe(r, n_boot=200, seed=0)
    point = sharpe_ratio(r)
    assert out["sharpe_ci_low"] < point < out["sharpe_ci_high"]


def test_quantile_spread_is_monotone_for_a_clean_signal():
    rng = np.random.default_rng(8)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=200), 50)
    pred = rng.normal(size=len(dates))
    fwd = 0.02 * pred + rng.normal(0, 0.01, len(dates))
    qs = quantile_spread(pred, fwd, pd.Series(dates), 5)
    assert qs["mean"].is_monotonic_increasing
    assert qs["mean"].iloc[-1] > qs["mean"].iloc[0]


def test_ic_summary_handles_empty_input():
    out = ic_summary(pd.Series(dtype=float))
    assert np.isnan(out["mean_ic"])


def test_factor_composite_needs_no_training_data():
    """The benchmark must be identical whether it 'sees' 10 rows or 100000."""
    from alpha_engine.config import Config
    from alpha_engine.models.registry import build_model

    rng = np.random.default_rng(9)
    feats = ["mom_12_1", "ret_21", "vol_63", "beta_63", "amihud_illiq"]
    frame = pd.DataFrame(rng.normal(size=(500, len(feats))), columns=feats)
    frame["target"] = rng.normal(size=500)

    cfg = Config()
    a = build_model("factor_composite", feats, cfg).fit(frame.head(10))
    b = build_model("factor_composite", feats, cfg).fit(frame)
    assert np.allclose(a.predict(frame), b.predict(frame))


def test_factor_composite_signs_follow_the_literature():
    from alpha_engine.config import Config
    from alpha_engine.models.registry import build_model

    feats = ["mom_12_1", "ret_21", "vol_63", "beta_63", "amihud_illiq"]
    m = build_model("factor_composite", feats, Config())
    w = m.weights_
    assert w["mom_12_1"] > 0        # momentum
    assert w["ret_21"] < 0          # one-month reversal
    assert w["vol_63"] < 0          # low-volatility anomaly
    assert w["beta_63"] < 0         # betting against beta
    assert w["amihud_illiq"] > 0    # illiquidity premium
