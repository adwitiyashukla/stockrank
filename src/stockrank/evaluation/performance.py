from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from stockrank.utils.stats import newey_west_tstat

TRADING_DAYS = 252


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def performance_stats(
    returns: pd.Series, benchmark: pd.Series | None = None, rf_daily: float = 0.0
) -> dict[str, Any]:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {}

    n = len(r)
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() - rf_daily) / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan

    downside = r[r < rf_daily]
    sortino = (
        float((r.mean() - rf_daily) / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(downside) > 5 and downside.std() > 0
        else np.nan
    )

    dd = drawdown_series(r)
    max_dd = float(dd.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    under = (dd < -1e-9).astype(int)
    longest, cur = 0, 0
    for v in under.to_numpy():
        cur = cur + 1 if v else 0
        longest = max(longest, cur)

    out: dict[str, Any] = {
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "longest_drawdown_days": int(longest),
        "hit_rate_daily": float((r > 0).mean()),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "var_95_daily": float(np.percentile(r, 5)),
        "cvar_95_daily": float(r[r <= np.percentile(r, 5)].mean()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "n_days": int(n),
        "t_stat_nw": newey_west_tstat(r),
    }

    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1) if isinstance(r.index, pd.DatetimeIndex) else None
    if monthly is not None and len(monthly) > 2:
        out["hit_rate_monthly"] = float((monthly > 0).mean())
        out["best_month"] = float(monthly.max())
        out["worst_month"] = float(monthly.min())

    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index).fillna(0.0)
        if b.std() > 0:
            beta = float(np.cov(r, b)[0, 1] / np.var(b))
            out["beta_to_benchmark"] = beta
            out["alpha_annual"] = float((r.mean() - beta * b.mean()) * TRADING_DAYS)
            out["correlation_to_benchmark"] = float(np.corrcoef(r, b)[0, 1])
            active = r - b
            out["information_ratio"] = (
                float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
                if active.std() > 0 else np.nan
            )
    return out


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    r = pd.Series(returns).dropna()
    if not isinstance(r.index, pd.DatetimeIndex) or r.empty:
        return pd.DataFrame()
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    piv = tbl.pivot_table(index="year", columns="month", values="ret", observed=True)
    piv.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in piv.columns]
    piv["Year"] = r.resample("YE").apply(lambda x: (1 + x).prod() - 1).to_numpy()
    return piv


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    r = pd.Series(returns).dropna()
    return r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
