"""Portfolio construction invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.config import PortfolioConfig
from alpha_engine.portfolio.construction import (
    apply_beta_neutrality,
    build_weights_for_date,
    ledoit_wolf_cov,
    rank_long_short_weights,
    volatility_scalar,
)


@pytest.fixture
def scores() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(size=60), index=[f"T{i:02d}" for i in range(60)])


def test_rank_long_short_is_dollar_neutral(scores):
    w = rank_long_short_weights(scores, 10, 10, gross_leverage=2.0, max_weight=1.0)
    assert abs(w.sum()) < 1e-9
    assert abs(w.abs().sum() - 2.0) < 1e-9
    assert (w > 0).sum() == 10
    assert (w < 0).sum() == 10


def test_longs_are_the_highest_scores(scores):
    w = rank_long_short_weights(scores, 5, 5, 2.0, 1.0)
    assert set(w[w > 0].index) == set(scores.nlargest(5).index)
    assert set(w[w < 0].index) == set(scores.nsmallest(5).index)


def test_max_weight_is_binding(scores):
    w = rank_long_short_weights(scores, 3, 3, gross_leverage=2.0, max_weight=0.10)
    assert w.abs().max() <= 0.10 + 1e-12


def test_beta_neutrality_removes_net_beta(scores):
    rng = np.random.default_rng(2)
    beta = pd.Series(rng.normal(1.0, 0.35, len(scores)), index=scores.index)
    w = rank_long_short_weights(scores, 12, 12, 2.0, 1.0)
    assert abs((w * beta).sum()) > 1e-6, "test setup should start with non-zero net beta"
    w_n = apply_beta_neutrality(w, beta)
    assert abs((w_n * beta).sum()) < 1e-8


def test_beta_neutrality_preserves_ordering_direction(scores):
    """The projection removes market exposure; it must not invert the book."""
    rng = np.random.default_rng(3)
    beta = pd.Series(rng.normal(1.0, 0.2, len(scores)), index=scores.index)
    w = rank_long_short_weights(scores, 15, 15, 2.0, 1.0)
    w_n = apply_beta_neutrality(w, beta)
    assert np.corrcoef(w.to_numpy(), w_n.to_numpy())[0, 1] > 0.8


def test_full_construction_respects_gross_leverage(scores):
    cfg = PortfolioConfig(n_long=10, n_short=10, gross_leverage=1.5, max_weight=0.2)
    betas = pd.Series(np.random.default_rng(4).normal(1.0, 0.3, len(scores)), index=scores.index)
    w = build_weights_for_date(scores, cfg, betas=betas)
    assert abs(w.abs().sum() - 1.5) < 1e-6
    assert abs((w * betas).sum()) < 1e-6


def test_volatility_scalar_targets_the_right_level():
    rng = np.random.default_rng(5)
    # Realised annual vol of 20% against a 10% target should halve the exposure.
    r = pd.Series(rng.normal(0, 0.20 / np.sqrt(252), 300))
    s = volatility_scalar(r, target_annual=0.10, lookback=252, cap=3.0)
    assert 0.4 < s < 0.65


def test_volatility_scalar_is_capped():
    r = pd.Series(np.full(300, 1e-6))
    assert volatility_scalar(r, 0.10, 252, cap=2.0) <= 2.0


def test_ledoit_wolf_is_positive_definite():
    rng = np.random.default_rng(6)
    R = pd.DataFrame(rng.normal(size=(120, 60)))  # fewer rows than columns squared
    cov = ledoit_wolf_cov(R)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)
