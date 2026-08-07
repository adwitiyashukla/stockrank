"""Configuration, simulator and data loading."""

from __future__ import annotations

import numpy as np
import pytest

from stockrank.config import Config, SimulatorConfig, load_config
from stockrank.data.loader import load_market_data
from stockrank.data.simulator import simulate_market


def test_every_shipped_config_parses():
    from pathlib import Path

    for p in sorted(Path("configs").glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.label.horizon >= 1
        assert cfg.validation.n_splits >= 1


def test_rebalance_matches_label_horizon_in_shipped_configs():
    """A mismatch would book returns the portfolio never earned."""
    from pathlib import Path

    for p in sorted(Path("configs").glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.portfolio.rebalance_days == cfg.label.horizon, p.name


def test_garch_parameters_must_be_stationary():
    with pytest.raises(ValueError):
        SimulatorConfig(garch_alpha=0.5, garch_beta=0.6)


def test_simulator_is_deterministic():
    cfg = Config()
    cfg.data.n_assets = 20
    cfg.data.start, cfg.data.end = "2018-01-01", "2020-12-31"
    a, _ = simulate_market(cfg.data, cfg.simulator, seed=42)
    b, _ = simulate_market(cfg.data, cfg.simulator, seed=42)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_simulator_produces_realistic_moments():
    cfg = Config()
    cfg.data.n_assets = 60
    cfg.data.start, cfg.data.end = "2010-01-01", "2020-12-31"
    panel, market = simulate_market(cfg.data, cfg.simulator, seed=1)
    ann_mkt_vol = market["mkt_return"].std() * np.sqrt(252)
    assert 0.10 < ann_mkt_vol < 0.25, "market volatility outside a plausible range"
    assert (panel["high"] >= panel["close"] - 1e-6).all()
    assert (panel["low"] <= panel["close"] + 1e-6).all()
    assert (panel["volume"] > 0).all()


def test_null_alpha_flag_zeroes_the_planted_signal():
    cfg = Config()
    cfg.data.n_assets = 30
    cfg.data.start, cfg.data.end = "2015-01-01", "2019-12-31"
    cfg.simulator.null_alpha = True
    panel, _ = simulate_market(cfg.data, cfg.simulator, seed=3)
    assert np.allclose(panel["alpha_true"].to_numpy(), 0.0)


def test_quality_filters_drop_short_histories(small_config):
    md = load_market_data(small_config, cache=False)
    counts = md.panel.groupby("ticker", observed=True)["date"].size()
    assert counts.min() >= small_config.data.min_history_days


def test_feature_set_has_no_infinities(feature_set):
    X = feature_set.X.to_numpy()
    assert np.isfinite(X).all(), "features contain inf or nan after the pipeline"


def test_feature_set_is_sorted_by_date(feature_set):
    d = feature_set.frame["date"].to_numpy()
    assert (d[1:] >= d[:-1]).all()
