"""Shared fixtures. Every test runs on the synthetic simulator so the suite needs
no network access and produces identical numbers on every machine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockrank.config import Config
from stockrank.data.loader import load_market_data
from stockrank.features.pipeline import build_feature_set


@pytest.fixture(scope="session")
def small_config() -> Config:
    cfg = Config()
    cfg.run.name = "pytest"
    cfg.run.seed = 7
    cfg.data.source = "synthetic"
    cfg.data.start = "2015-01-01"
    cfg.data.end = "2021-12-31"
    cfg.data.n_assets = 40
    cfg.data.max_assets = 40
    cfg.data.benchmark = "MKT"
    cfg.data.use_fama_french = False
    cfg.data.cache_dir = ".pytest_cache/stockrank_data_data"
    cfg.data.min_history_days = 200
    cfg.data.min_names_per_date = 10
    cfg.label.horizon = 5
    cfg.validation.n_splits = 2
    cfg.validation.train_window_days = 400
    cfg.validation.test_window_days = 120
    cfg.models.enabled = ["ridge", "lightgbm"]
    cfg.models.lightgbm = {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 50}
    cfg.portfolio.n_long = 5
    cfg.portfolio.n_short = 5
    cfg.portfolio.rebalance_days = 5
    cfg.volatility.max_assets_fitted = 0
    return cfg


@pytest.fixture(scope="session")
def market_data(small_config):
    return load_market_data(small_config, cache=False)


@pytest.fixture(scope="session")
def feature_set(market_data, small_config):
    return build_feature_set(market_data, small_config)


@pytest.fixture(scope="session")
def price_matrix(market_data) -> pd.DataFrame:
    return market_data.close_matrix().astype("float64")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
