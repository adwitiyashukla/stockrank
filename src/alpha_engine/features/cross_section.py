"""Cross-sectional normalisation.

A raw feature such as ``vol_63`` is not comparable across time: realised
volatility in March 2020 was three times its 2017 level for almost every name at
once. Feeding raw levels to a model means most of what it learns is the level of
the market, not the relative attractiveness of one stock against another.

Normalising each feature *within each date* removes that common component and
turns every input into a statement about relative position in the cross-section,
which is exactly what a dollar-neutral long/short book trades on. It also makes
the features stationary almost for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_rows(df: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Clip each row at its own ``q`` and ``1-q`` quantiles."""
    if q <= 0:
        return df
    lo = df.quantile(q, axis=1)
    hi = df.quantile(1 - q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Uniform [-0.5, 0.5] cross-sectional rank: fully outlier proof."""
    return df.rank(axis=1, pct=True) - 0.5


def normalise(df: pd.DataFrame, method: str = "zscore", winsor_q: float = 0.01) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "rank":
        return rank_rows(df)
    if method == "zscore":
        return zscore_rows(winsorize_rows(df, winsor_q)).clip(-5.0, 5.0)
    raise ValueError(f"Unknown normalisation method: {method}")


def sector_neutralise(long_df: pd.DataFrame, cols: list[str], sector_col: str = "sector") -> pd.DataFrame:
    """Demean each feature within (date, sector) so sector bets are stripped out."""
    out = long_df.copy()
    grp = out.groupby(["date", sector_col], observed=True)
    for c in cols:
        out[c] = out[c] - grp[c].transform("mean")
    return out


def min_names_mask(df: pd.DataFrame, min_names: int) -> pd.Series:
    """True for dates where the cross-section is wide enough to rank."""
    return df.notna().sum(axis=1) >= min_names
