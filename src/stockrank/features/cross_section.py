from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_rows(df: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
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
    out = long_df.copy()
    grp = out.groupby(["date", sector_col], observed=True)
    for c in cols:
        out[c] = out[c] - grp[c].transform("mean")
    return out


def min_names_mask(df: pd.DataFrame, min_names: int) -> pd.Series:
    return df.notna().sum(axis=1) >= min_names
