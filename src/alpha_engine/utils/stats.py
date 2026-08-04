"""Numerically defensive statistical helpers used across the codebase."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    if q <= 0:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def cross_sectional_rank(s: pd.Series) -> pd.Series:
    """Rank into [-0.5, 0.5], robust to outliers and constant slices."""
    n = s.notna().sum()
    if n <= 1:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True) - 0.5


def nan_safe_corr(a: np.ndarray, b: np.ndarray, method: str = "spearman") -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    x, y = a[mask], b[mask]
    if method == "spearman":
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def annualised_return(daily: pd.Series) -> float:
    daily = daily.dropna()
    if daily.empty:
        return np.nan
    return float(daily.mean() * TRADING_DAYS)


def annualised_vol(daily: pd.Series) -> float:
    daily = daily.dropna()
    if daily.empty:
        return np.nan
    return float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily.dropna() - rf_daily
    if excess.empty or excess.std(ddof=1) == 0:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean of x under HAC (Newey-West) standard errors."""
    v = x.dropna().to_numpy(dtype=float)
    n = v.size
    if n < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    demeaned = v - v.mean()
    gamma0 = float(demeaned @ demeaned / n)
    var = gamma0
    for lag in range(1, max(lags, 0) + 1):
        w = 1.0 - lag / (lags + 1)
        cov = float(demeaned[lag:] @ demeaned[:-lag] / n)
        var += 2.0 * w * cov
    if var <= 0:
        return np.nan
    return float(v.mean() / np.sqrt(var / n))
