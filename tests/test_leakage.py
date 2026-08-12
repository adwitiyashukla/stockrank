from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockrank.config import FeatureConfig, LabelConfig
from stockrank.features.labels import build_labels, forward_return
from stockrank.features.technical import build_feature_matrices
from stockrank.validation.splitters import PurgedWalkForward


def test_features_are_causal(market_data, small_config):
    md = market_data
    close = md.close_matrix().astype("float64")
    high = md.panel.pivot(index="date", columns="ticker", values="high").astype("float64")
    low = md.panel.pivot(index="date", columns="ticker", values="low").astype("float64")
    vol = md.panel.pivot(index="date", columns="ticker", values="volume").astype("float64")
    dv = md.panel.pivot(index="date", columns="ticker", values="dollar_volume").astype("float64")
    mkt = md.market.set_index("date")["mkt_return"].reindex(close.index).ffill().fillna(0.0)

    cfg = FeatureConfig()
    base = build_feature_matrices(close, high, low, vol, dv, mkt, cfg)

    cut = len(close) - 60
    c2, h2, l2, v2, d2 = (x.copy() for x in (close, high, low, vol, dv))
    for frame in (c2, h2, l2):
        frame.iloc[cut:] *= 3.0
    v2.iloc[cut:] *= 5.0
    d2.iloc[cut:] *= 15.0
    m2 = mkt.copy()
    m2.iloc[cut:] += 0.05

    corrupted = build_feature_matrices(c2, h2, l2, v2, d2, m2, cfg)

    for name in base:
        a = base[name].iloc[: cut - 1].to_numpy()
        b = corrupted[name].iloc[: cut - 1].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both], atol=1e-9), f"Feature '{name}' leaks future information"


def test_forward_return_matches_manual_calculation(price_matrix):
    h, lag = 5, 1
    fwd = forward_return(price_matrix, h, lag=lag)
    col = price_matrix.columns[0]
    for t in (300, 500, 700):
        expected = price_matrix[col].iloc[t + lag + h] / price_matrix[col].iloc[t + lag] - 1
        got = fwd[col].iloc[t]
        if np.isfinite(expected) and np.isfinite(got):
            assert abs(got - expected) < 1e-9


def test_label_tail_is_nan(price_matrix):
    h, lag = 5, 1
    fwd = forward_return(price_matrix, h, lag=lag)
    assert fwd.iloc[-(h + lag) :].isna().all().all()


def test_execution_lag_is_applied(price_matrix):
    a = forward_return(price_matrix, 5, lag=0)
    b = forward_return(price_matrix, 5, lag=1)
    common = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    assert not np.allclose(a.to_numpy()[common], b.to_numpy()[common])


def test_cross_sectional_target_sums_to_zero(price_matrix):
    cfg = LabelConfig(type="forward_excess_return", horizon=5, neutralise_market=False)
    target, _ = build_labels(price_matrix, cfg)
    row_means = target.mean(axis=1).dropna()
    assert np.allclose(row_means.to_numpy(), 0.0, atol=1e-9)


@pytest.mark.parametrize("horizon,embargo", [(1, 0), (5, 10), (21, 21)])
def test_purged_splits_never_overlap(feature_set, horizon, embargo):
    cv = PurgedWalkForward(
        n_splits=2, train_window_days=400, test_window_days=120,
        label_horizon=horizon, embargo_days=embargo,
    )
    dates = feature_set.frame["date"]
    for fold in cv.get_folds(dates):
        train_dates = set(dates.iloc[fold.train_idx].unique())
        test_dates = set(dates.iloc[fold.test_idx].unique())
        assert not (train_dates & test_dates), "train and test share dates"
        assert max(train_dates) < min(test_dates), "training extends past the test start"

        uniq = pd.DatetimeIndex(sorted(dates.unique()))
        gap = uniq.get_loc(min(test_dates)) - uniq.get_loc(max(train_dates))
        assert gap >= horizon + 1, f"purge gap of {gap} days is shorter than the {horizon}d label"


def test_folds_move_forward_in_time(feature_set):
    cv = PurgedWalkForward(n_splits=3, train_window_days=400, test_window_days=120, label_horizon=5)
    folds = cv.get_folds(feature_set.frame["date"])
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start > a.test_start
        assert b.train_end >= a.train_end


def test_no_row_appears_in_both_train_and_test(feature_set):
    cv = PurgedWalkForward(n_splits=2, train_window_days=400, test_window_days=120, label_horizon=5)
    for fold in cv.get_folds(feature_set.frame["date"]):
        assert len(np.intersect1d(fold.train_idx, fold.test_idx)) == 0
