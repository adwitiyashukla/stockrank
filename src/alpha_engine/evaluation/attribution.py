"""Factor attribution: is this alpha, or repackaged beta?

The question an allocator asks about any long/short equity strategy is whether it
survives a regression on the standard factors. If a strategy's return is fully
explained by loadings on market, size, value, profitability, investment and
momentum, then it can be replicated with cheap index products and is not worth a
fee. What matters is the intercept, and whether it stands up to HAC standard
errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FF_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def factor_regression(
    strategy_returns: pd.Series, factors: pd.DataFrame, hac_lags: int = 10
) -> dict[str, Any]:
    """OLS of strategy excess returns on available factors with Newey-West errors."""
    import statsmodels.api as sm

    f = factors.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date")
    cols = [c for c in FF_COLS if c in f.columns]
    if not cols:
        return {"error": "no factor columns found", "available": list(f.columns)}

    r = pd.Series(strategy_returns).dropna()
    rf = f["rf"] if "rf" in f.columns else pd.Series(0.0, index=f.index)
    joined = pd.concat([r.rename("y"), f[cols], rf.rename("rf")], axis=1).dropna()
    if len(joined) < 100:
        return {"error": f"only {len(joined)} overlapping observations"}

    y = (joined["y"] - joined["rf"]).to_numpy()
    X = sm.add_constant(joined[cols].to_numpy())
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    names = ["alpha"] + cols
    out: dict[str, Any] = {
        "n_obs": int(len(joined)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "alpha_daily": float(model.params[0]),
        "alpha_annual": float(model.params[0] * TRADING_DAYS),
        "alpha_tstat_hac": float(model.tvalues[0]),
        "alpha_pvalue_hac": float(model.pvalues[0]),
        "hac_lags": hac_lags,
    }
    for i, nm in enumerate(names):
        out[f"beta_{nm}"] = float(model.params[i])
        out[f"tstat_{nm}"] = float(model.tvalues[i])
    return out


def exposure_over_time(
    strategy_returns: pd.Series, factors: pd.DataFrame, window: int = 252
) -> pd.DataFrame:
    """Rolling factor betas, which reveal style drift a full-sample regression hides."""
    import statsmodels.api as sm

    f = factors.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date")
    cols = [c for c in FF_COLS if c in f.columns]
    joined = pd.concat([pd.Series(strategy_returns).rename("y"), f[cols]], axis=1).dropna()
    if len(joined) < window + 20:
        return pd.DataFrame()

    rows = []
    for end in range(window, len(joined), 21):
        w = joined.iloc[end - window : end]
        model = sm.OLS(w["y"].to_numpy(), sm.add_constant(w[cols].to_numpy())).fit()
        rows.append({"date": w.index[-1], **dict(zip(cols, model.params[1:], strict=False))})
    return pd.DataFrame(rows).set_index("date")


def turnover_capacity_analysis(
    returns: pd.Series, turnover: pd.Series, cost_bps_grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Net Sharpe as a function of assumed round-trip cost.

    The break-even cost is the single most informative number about whether a
    strategy is implementable. A signal that needs sub-3bp execution is a
    high-frequency shop's problem, not a research result.
    """
    if cost_bps_grid is None:
        cost_bps_grid = np.array([0, 2, 5, 10, 15, 20, 30, 50])

    r = pd.Series(returns).dropna()
    t = pd.Series(turnover).reindex(r.index).fillna(0.0)
    rows = []
    for bps in cost_bps_grid:
        adj = r - t * bps / 10_000.0
        sd = adj.std(ddof=1)
        rows.append(
            {
                "cost_bps": float(bps),
                "ann_return": float(adj.mean() * TRADING_DAYS),
                "sharpe": float(adj.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
