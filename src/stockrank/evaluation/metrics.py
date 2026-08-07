"""Forecast quality metrics for cross-sectional prediction.

The headline number is the **information coefficient**: the cross-sectional rank
correlation between the forecast and the realised forward return, computed date
by date and then averaged. This is the right metric here and R-squared is not,
for a specific reason. A model that predicts every stock will return roughly the
market has a decent R-squared and is worth nothing to a market-neutral book. A
model with an R-squared indistinguishable from zero but a stable IC of 0.02 is a
viable strategy. What is being traded is the ordering.

``ICIR`` (mean IC divided by its standard deviation, annualised) is the more
useful summary because it captures consistency, and consistency is what survives
transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stockrank.utils.stats import newey_west_tstat

TRADING_DAYS = 252


def daily_ic(
    pred: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    dates: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional IC for every date."""
    df = pd.DataFrame(
        {"date": pd.to_datetime(pd.Series(dates).to_numpy()),
         "p": np.asarray(pred, dtype=float),
         "t": np.asarray(target, dtype=float)}
    ).dropna()

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 5 or g["p"].std() == 0 or g["t"].std() == 0:
            return np.nan
        if method == "spearman":
            return float(stats.spearmanr(g["p"], g["t"]).statistic)
        return float(np.corrcoef(g["p"], g["t"])[0, 1])

    return df.groupby("date", observed=True)[["p", "t"]].apply(_corr).dropna()


def ic_summary(ic: pd.Series, label_horizon: int = 5) -> dict[str, float]:
    """Summarise a series of daily ICs.

    ``n_effective`` corrects for the fact that overlapping ``h``-day labels make
    consecutive ICs dependent, so the naive t-statistic is inflated by roughly
    sqrt(h). The Newey-West t-statistic in the same dict handles this properly and
    is the one to quote.
    """
    ic = pd.Series(ic).dropna()
    if ic.empty:
        return {k: np.nan for k in ("mean_ic", "std_ic", "icir", "t_stat", "t_stat_nw", "hit_rate", "n_obs")}
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    n_eff = max(len(ic) / max(label_horizon, 1), 1.0)
    return {
        "mean_ic": mean,
        "std_ic": sd,
        "icir": float(mean / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "t_stat": float(mean / sd * np.sqrt(n_eff)) if sd > 0 else np.nan,
        "t_stat_nw": newey_west_tstat(ic, lags=2 * label_horizon),
        "hit_rate": float((ic > 0).mean()),
        "n_obs": int(len(ic)),
    }


def quantile_spread(
    pred: np.ndarray, fwd_return: np.ndarray, dates: pd.Series, n_quantiles: int = 5
) -> pd.DataFrame:
    """Mean forward return per prediction quantile: the classic factor decile chart.

    Monotonicity across quantiles is a much stronger claim than a positive IC. A
    signal can have a positive IC driven entirely by one extreme bucket, which is
    fragile; a monotone ladder is what a real factor looks like.
    """
    df = pd.DataFrame(
        {"date": pd.to_datetime(pd.Series(dates).to_numpy()),
         "p": np.asarray(pred, dtype=float),
         "r": np.asarray(fwd_return, dtype=float)}
    ).dropna()

    def _bucket(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < n_quantiles * 2:
            return pd.DataFrame()
        q = pd.qcut(g["p"].rank(method="first"), n_quantiles, labels=False)
        return pd.DataFrame({"q": q, "r": g["r"].to_numpy()})

    parts = [
        _bucket(g).assign(date=d)
        for d, g in df.groupby("date", observed=True)[["p", "r"]]
        if len(g) >= n_quantiles * 2
    ]
    if not parts:
        return pd.DataFrame()
    stacked = pd.concat(parts, ignore_index=True)
    out = stacked.groupby("q", observed=True)["r"].agg(["mean", "std", "count"]).reset_index()
    out["quantile"] = out["q"].astype(int) + 1
    return out[["quantile", "mean", "std", "count"]]


def prediction_metrics(
    pred: np.ndarray, target: np.ndarray, fwd_return: np.ndarray, dates: pd.Series, horizon: int = 5
) -> dict[str, float]:
    ic_s = daily_ic(pred, target, dates, "spearman")
    ic_p = daily_ic(pred, target, dates, "pearson")
    out = {f"rank_{k}": v for k, v in ic_summary(ic_s, horizon).items()}
    out["pearson_ic"] = float(ic_p.mean()) if len(ic_p) else np.nan

    ok = np.isfinite(pred) & np.isfinite(target)
    if ok.sum() > 10:
        resid = target[ok] - pred[ok]
        sst = float(np.sum((target[ok] - target[ok].mean()) ** 2))
        out["oos_r2"] = float(1 - np.sum(resid**2) / sst) if sst > 0 else np.nan
        out["rmse"] = float(np.sqrt(np.mean(resid**2)))
    qs = quantile_spread(pred, fwd_return, dates)
    if not qs.empty:
        out["q5_minus_q1"] = float(qs["mean"].iloc[-1] - qs["mean"].iloc[0])
        out["monotonicity"] = float(
            stats.spearmanr(qs["quantile"], qs["mean"]).statistic
        )
    return out


def matrix_ic(feature: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Row-wise (per-date) Spearman IC between two aligned wide matrices.

    Vectorised: rank once, then compute the row correlation in numpy. Roughly two
    orders of magnitude faster than a groupby-apply over the long frame, which
    matters when screening dozens of features across several horizons.
    """
    f, t = feature.align(target, join="inner")
    both = np.isfinite(f.to_numpy()) & np.isfinite(t.to_numpy())
    fm = f.where(both)
    tm = t.where(both)
    fr = fm.rank(axis=1).to_numpy()
    tr = tm.rank(axis=1).to_numpy()

    n = np.isfinite(fr).sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fmean = np.nanmean(fr, axis=1, keepdims=True)
        tmean = np.nanmean(tr, axis=1, keepdims=True)
        fd = np.nan_to_num(fr - fmean)
        td = np.nan_to_num(tr - tmean)
        num = (fd * td).sum(axis=1)
        den = np.sqrt((fd**2).sum(axis=1) * (td**2).sum(axis=1))
        ic = np.where((den > 0) & (n >= 5), num / den, np.nan)
    return pd.Series(ic, index=f.index).dropna()
