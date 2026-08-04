"""Volatility forecasting: GARCH, HAR and EWMA, compared properly.

This is the econometrics half of the project. Return *levels* are close to
unpredictable; return *variance* is strongly predictable, and the strategy needs
a variance forecast anyway to size positions. So this module fits three standard
models and compares them the way the volatility literature does.

Models
------
* **GARCH(1,1)** with Student-t errors: the workhorse, captures clustering and
  fat tails, estimated by maximum likelihood.
* **HAR-RV** (Corsi 2009): regresses realised variance on its own daily, weekly
  and monthly averages. Trivially cheap and famously hard to beat.
* **EWMA** (RiskMetrics, lambda = 0.94): one parameter, no estimation, the
  benchmark any model must clear to justify itself.

Evaluation uses QLIKE alongside MSE. Squared error on a variance forecast is
dominated by a handful of crisis days and rewards over-prediction; QLIKE is the
loss the literature prefers because it is robust to the fact that true variance
is never observed, only proxied.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def ewma_variance(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA. The forecast for t+1 uses information through t."""
    r2 = returns.fillna(0.0) ** 2
    var = r2.ewm(alpha=1 - lam, adjust=False).mean()
    return var.shift(1)


def har_rv_forecast(returns: pd.Series, min_obs: int = 252) -> pd.Series:
    """Rolling one-step-ahead HAR-RV forecasts of daily variance."""
    import statsmodels.api as sm

    rv = (returns.fillna(0.0) ** 2).rename("rv")
    d = rv.shift(1)
    w = rv.rolling(5).mean().shift(1)
    m = rv.rolling(22).mean().shift(1)
    df = pd.concat([rv, d.rename("d"), w.rename("w"), m.rename("m")], axis=1).dropna()
    if len(df) < min_obs + 20:
        return pd.Series(np.nan, index=returns.index)

    out = pd.Series(np.nan, index=returns.index)
    X = sm.add_constant(df[["d", "w", "m"]].to_numpy())
    y = df["rv"].to_numpy()
    # Refit monthly on an expanding window: cheap, and avoids look-ahead.
    for start in range(min_obs, len(df), 21):
        end = min(start + 21, len(df))
        model = sm.OLS(y[:start], X[:start]).fit()
        out.loc[df.index[start:end]] = np.maximum(model.predict(X[start:end]), 1e-12)
    return out


def garch_forecast(
    returns: pd.Series, refit_every: int = 63, min_obs: int = 500, dist: str = "t"
) -> pd.Series:
    """Rolling one-step-ahead GARCH(1,1) variance forecasts.

    Refitting every day would be honest but pointlessly slow; parameters of a
    GARCH(1,1) move very little over a quarter. Between refits the recursion is
    rolled forward with the fixed parameters, which is exactly what a risk desk
    does in practice.
    """
    from arch import arch_model

    r = returns.dropna() * 100.0  # arch is better conditioned on percent returns
    if len(r) < min_obs + refit_every:
        return pd.Series(np.nan, index=returns.index)

    out = pd.Series(np.nan, index=r.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(min_obs, len(r), refit_every):
            end = min(start + refit_every, len(r))
            try:
                res = arch_model(r.iloc[:start], p=1, q=1, dist=dist, rescale=False).fit(
                    disp="off", show_warning=False
                )
                omega = res.params.get("omega", np.nan)
                alpha = res.params.get("alpha[1]", np.nan)
                beta = res.params.get("beta[1]", np.nan)
                if not np.isfinite([omega, alpha, beta]).all():
                    continue
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                eps = r.iloc[start - 1] - res.params.get("mu", 0.0)
                for i in range(start, end):
                    sigma2 = omega + alpha * eps**2 + beta * sigma2
                    out.iloc[i] = sigma2
                    eps = r.iloc[i] - res.params.get("mu", 0.0)
            except Exception as exc:  # noqa: BLE001 - convergence failures are routine
                logger.debug("GARCH fit failed at %d: %s", start, exc)
                continue
    return (out / 10_000.0).reindex(returns.index)  # back to return-squared units


def qlike(realised: np.ndarray, forecast: np.ndarray) -> float:
    """QLIKE loss: robust to a noisy variance proxy, penalises under-prediction."""
    ok = np.isfinite(realised) & np.isfinite(forecast) & (forecast > 0)
    if ok.sum() < 10:
        return np.nan
    r, f = realised[ok], forecast[ok]
    return float(np.mean(r / f - np.log(np.maximum(r, 1e-16) / f) - 1.0))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy with HAC variance.

    Returns ``(statistic, two-sided p-value)``. Negative statistic favours model A.
    """
    from scipy import stats

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 20:
        return np.nan, np.nan
    dbar = d.mean()
    dm = d - dbar
    gamma0 = float(dm @ dm / n)
    var = gamma0
    for lag in range(1, h):
        cov = float(dm[lag:] @ dm[:-lag] / n)
        var += 2.0 * (1.0 - lag / h) * cov
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def compare_volatility_models(
    returns_matrix: pd.DataFrame, max_assets: int = 40, refit_every: int = 63
) -> pd.DataFrame:
    """Fit all three models per name and report average out-of-sample losses."""
    cols = (
        returns_matrix.std().sort_values(ascending=False).head(max_assets).index
        if returns_matrix.shape[1] > max_assets
        else returns_matrix.columns
    )
    rows = []
    for i, col in enumerate(cols, 1):
        r = returns_matrix[col].dropna()
        if len(r) < 800:
            continue
        realised = (r**2).to_numpy()
        fc = {
            "ewma": ewma_variance(r).to_numpy(),
            "har": har_rv_forecast(r).to_numpy(),
            "garch": garch_forecast(r, refit_every=refit_every).to_numpy(),
        }
        row = {"ticker": col}
        for k, v in fc.items():
            ok = np.isfinite(v) & np.isfinite(realised)
            row[f"qlike_{k}"] = qlike(realised, v)
            row[f"mse_{k}"] = float(np.mean((realised[ok] - v[ok]) ** 2)) if ok.sum() > 10 else np.nan
        rows.append(row)
        if i % 10 == 0:
            logger.info("  volatility models fitted for %d/%d names", i, len(cols))
    out = pd.DataFrame(rows)
    logger.info("Volatility comparison complete on %d names", len(out))
    return out
