"""Forward-looking targets. Every ``shift`` with a negative argument lives here.

Timing convention
-----------------
Features carry the timestamp of the close at which they were observable. A signal
formed on the close of day ``t`` cannot be executed at that same close in any
realistic setting, so the label deliberately starts one day later::

    fwd_return[t] = close[t + lag + h] / close[t + lag] - 1        (lag = 1 by default)

That single day of slack is the cheapest insurance against timing leakage there
is, and it costs a little performance rather than manufacturing it.

The default target then removes the cross-sectional mean, because a dollar-neutral
long/short book earns the *spread* between names, not the market. Ranking on raw
forward returns would mostly rank on beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.config import LabelConfig
from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)


def forward_return(close: pd.DataFrame, horizon: int, lag: int = 1) -> pd.DataFrame:
    """Simple forward return over ``horizon`` days, starting ``lag`` days ahead."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    entry = close.shift(-lag)
    exit_ = close.shift(-(lag + horizon))
    return exit_ / entry - 1.0


def cross_sectional_demean(fwd: pd.DataFrame, min_names: int = 10) -> pd.DataFrame:
    """Subtract the equal-weighted cross-sectional mean, date by date."""
    counts = fwd.notna().sum(axis=1)
    mean = fwd.mean(axis=1)
    out = fwd.sub(mean, axis=0)
    return out.where(counts.reindex(out.index) >= min_names, np.nan)


def beta_adjusted_forward_return(
    fwd: pd.DataFrame, fwd_mkt: pd.Series, beta: pd.DataFrame
) -> pd.DataFrame:
    """Residual forward return after removing each name's beta times the market."""
    return fwd.sub(beta.mul(fwd_mkt, axis=0))


def triple_barrier_labels(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    upper_sigma: float = 2.0,
    lower_sigma: float = 2.0,
    max_holding_days: int = 10,
    lag: int = 1,
) -> pd.DataFrame:
    """Lopez de Prado triple-barrier labels: +1 profit target, -1 stop, 0 timeout.

    Barriers are set in units of each name's own recent volatility, so a quiet
    utility and a volatile semiconductor face equally difficult targets. This
    matters because a fixed percentage barrier silently labels high-volatility
    names as "wins" far more often.
    """
    entry = close.shift(-lag)
    sigma = vol.reindex_like(close)
    labels = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    up_mult = np.exp(upper_sigma * sigma * np.sqrt(max_holding_days))
    dn_mult = np.exp(-lower_sigma * sigma * np.sqrt(max_holding_days))
    upper = entry * up_mult
    lower = entry * dn_mult

    e = entry.to_numpy()
    up = upper.to_numpy()
    dn = lower.to_numpy()
    px = close.to_numpy()
    out = np.full(px.shape, np.nan)
    n_rows = px.shape[0]

    for t in range(n_rows):
        end = min(t + lag + max_holding_days, n_rows)
        if t + lag >= n_rows:
            break
        window = px[t + lag : end]
        if window.shape[0] == 0:
            continue
        hit_up = (window >= up[t]).argmax(axis=0)
        any_up = (window >= up[t]).any(axis=0)
        hit_dn = (window <= dn[t]).argmax(axis=0)
        any_dn = (window <= dn[t]).any(axis=0)
        first_up = np.where(any_up, hit_up, np.iinfo(np.int32).max)
        first_dn = np.where(any_dn, hit_dn, np.iinfo(np.int32).max)
        lab = np.where(first_up < first_dn, 1.0, np.where(first_dn < first_up, -1.0, 0.0))
        out[t] = np.where(np.isfinite(e[t]), lab, np.nan)

    labels.loc[:, :] = out
    return labels


def build_labels(
    close: pd.DataFrame,
    cfg: LabelConfig,
    vol: pd.DataFrame | None = None,
    raw_vol: pd.DataFrame | None = None,
    beta: pd.DataFrame | None = None,
    mkt_return: pd.Series | None = None,
    execution_lag: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(target, raw_forward_return)`` as wide matrices."""
    fwd = forward_return(close, cfg.horizon, lag=execution_lag)

    scaled = fwd
    if cfg.scale_by_volatility and raw_vol is not None:
        # Risk-adjusted target: a 3% move in a quiet utility is a much stronger
        # signal than the same move in a volatile semiconductor, and an unscaled
        # target lets the noisiest names dominate the loss.
        scaled = fwd / raw_vol.replace(0.0, np.nan).clip(lower=1e-4)

    if cfg.type == "forward_return":
        target = scaled
    elif cfg.type == "forward_excess_return":
        if cfg.neutralise_market and beta is not None and mkt_return is not None:
            fwd_mkt = (
                (1 + mkt_return).rolling(cfg.horizon).apply(np.prod, raw=True).shift(
                    -(execution_lag + cfg.horizon)
                )
                - 1
            )
            target = beta_adjusted_forward_return(scaled, fwd_mkt.reindex(close.index), beta)
            target = cross_sectional_demean(target)
        else:
            target = cross_sectional_demean(scaled)
    elif cfg.type == "triple_barrier":
        if vol is None:
            raise ValueError("triple_barrier labels require a volatility matrix")
        tb = cfg.triple_barrier
        target = triple_barrier_labels(
            close,
            vol,
            upper_sigma=tb.upper_sigma,
            lower_sigma=tb.lower_sigma,
            max_holding_days=tb.max_holding_days,
            lag=execution_lag,
        )
    else:
        raise ValueError(f"Unknown label type: {cfg.type}")

    coverage = float(target.notna().to_numpy().mean())
    logger.info(
        "Labels built: type=%s horizon=%dd lag=%dd vol_scaled=%s beta_adj=%s coverage=%.1f%%",
        cfg.type,
        cfg.horizon,
        execution_lag,
        cfg.scale_by_volatility,
        cfg.neutralise_market,
        100 * coverage,
    )
    return target, fwd
