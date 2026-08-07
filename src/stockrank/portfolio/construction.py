"""Turning a forecast into a book of positions.

A prediction is not a portfolio. Three things have to happen in between, and each
one destroys some of the paper alpha:

1. **Sizing.** Rank-based long/short is the default because it is what the IC
   actually justifies: the model is trusted to order names, not to say by how
   much. Mean-variance is offered as well, with Ledoit-Wolf shrinkage, because a
   sample covariance matrix over 250 names and 252 days is close to singular and
   an unshrunk optimiser will happily put its entire risk budget into the
   noisiest eigenvector.

2. **Constraints.** Position caps, dollar neutrality and optional sector
   neutrality. Without a position cap the optimiser concentrates; without dollar
   neutrality the book is a market bet wearing a long/short costume.

3. **Volatility targeting.** Scaling gross exposure by the ratio of target to
   recent realised volatility. This uses only trailing information, and the
   scalar is capped so the strategy cannot lever into a calm period and then get
   caught by a regime change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockrank.config import PortfolioConfig
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def rank_long_short_weights(
    scores: pd.Series, n_long: int, n_short: int, gross_leverage: float, max_weight: float
) -> pd.Series:
    """Equal-weight the top ``n_long`` and bottom ``n_short`` names, dollar neutral."""
    s = scores.dropna()
    w = pd.Series(0.0, index=scores.index)
    if len(s) < max(n_long + n_short, 10):
        n_long = n_short = max(1, len(s) // 4)
    if len(s) < 4:
        return w

    order = s.sort_values(ascending=False)
    longs, shorts = order.index[:n_long], order.index[-n_short:]
    side = gross_leverage / 2.0
    if len(longs):
        w.loc[longs] = side / len(longs)
    if len(shorts):
        w.loc[shorts] = -side / len(shorts)
    return w.clip(-max_weight, max_weight)


def score_weighted_weights(
    scores: pd.Series, gross_leverage: float, max_weight: float
) -> pd.Series:
    """Continuous exposure proportional to the cross-sectionally demeaned score."""
    s = scores.dropna()
    if len(s) < 4:
        return pd.Series(0.0, index=scores.index)
    z = s - s.mean()
    denom = z.abs().sum()
    if denom == 0:
        return pd.Series(0.0, index=scores.index)
    w = (z / denom) * gross_leverage
    w = w.clip(-max_weight, max_weight)
    return w.reindex(scores.index).fillna(0.0)


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Shrunk covariance. Falls back to the sample matrix if shrinkage fails."""
    from sklearn.covariance import LedoitWolf

    X = returns.dropna(axis=1, how="all").fillna(0.0).to_numpy()
    if X.shape[0] < 30 or X.shape[1] < 2:
        return np.cov(X, rowvar=False) if X.size else np.eye(1)
    try:
        return LedoitWolf().fit(X).covariance_
    except Exception:  # noqa: BLE001
        return np.cov(X, rowvar=False)


def mean_variance_weights(
    scores: pd.Series,
    returns_window: pd.DataFrame,
    risk_aversion: float,
    gross_leverage: float,
    max_weight: float,
    dollar_neutral: bool = True,
) -> pd.Series:
    """Analytic mean-variance solution with shrinkage, caps and neutrality."""
    names = [c for c in returns_window.columns if c in scores.index and np.isfinite(scores.get(c, np.nan))]
    if len(names) < 5:
        return pd.Series(0.0, index=scores.index)

    R = returns_window[names]
    Sigma = ledoit_wolf_cov(R)
    Sigma += np.eye(len(names)) * 1e-8  # ridge for invertibility

    mu = scores.loc[names].to_numpy(dtype=float)
    mu = (mu - mu.mean()) / (mu.std() or 1.0)
    daily_vol = float(np.sqrt(np.mean(np.diag(Sigma))))
    mu = mu * daily_vol  # put the score on a return scale

    try:
        raw = np.linalg.solve(risk_aversion * Sigma, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(risk_aversion * Sigma) @ mu

    w = pd.Series(raw, index=names)
    if dollar_neutral:
        w = w - w.mean()
    gross = w.abs().sum()
    if gross > 0:
        w = w / gross * gross_leverage
    w = w.clip(-max_weight, max_weight)
    return w.reindex(scores.index).fillna(0.0)


def risk_parity_weights(
    scores: pd.Series, vol_window: pd.Series, gross_leverage: float, max_weight: float,
    n_long: int, n_short: int
) -> pd.Series:
    """Long/short selection by score, then size inversely to each name's volatility."""
    base = rank_long_short_weights(scores, n_long, n_short, gross_leverage, 1.0)
    inv_vol = 1.0 / vol_window.reindex(base.index).replace(0.0, np.nan)
    w = base * inv_vol
    w = w.fillna(0.0)
    for sign in (1, -1):
        side = w[np.sign(w) == sign]
        if len(side) and side.abs().sum() > 0:
            w.loc[side.index] = side / side.abs().sum() * (gross_leverage / 2.0) * sign * np.sign(sign)
    return w.clip(-max_weight, max_weight)


def apply_beta_neutrality(w: pd.Series, beta: pd.Series) -> pd.Series:
    """Project the market-beta component out of the weight vector.

    This is the single most important constraint in the whole construction stack
    and it is easy to get wrong. Equal dollars long and short does **not** give a
    market-neutral book: if the model prefers low-volatility names on the long
    side and high-beta names on the short side, which price-based signals
    routinely do, the book carries a large negative beta. In a rising market that
    loses money regardless of how good the stock selection is.

    The fix is a projection. Writing ``b`` for the vector of trailing betas,

        w_neutral = w - (w . b / b . b) * b

    is the orthogonal projection of ``w`` onto the subspace with zero net beta,
    so it removes the market exposure while changing the intended cross-sectional
    tilt as little as possible.
    """
    b = beta.reindex(w.index)
    held = (w != 0) & np.isfinite(b)
    if held.sum() < 4:
        return w
    # The projection is taken over the held names only. Applying the hedge to the
    # whole universe would open positions the model never asked for and, worse,
    # would leave residual beta in exactly the names it was meant to remove it from.
    b_h = b[held]
    w_h = w[held]
    bb = float((b_h**2).sum())
    if bb <= 1e-12:
        return w
    hedge = float((w_h * b_h).sum()) / bb
    out = w.copy()
    out.loc[held] = w_h - hedge * b_h
    return out


def apply_sector_neutrality(w: pd.Series, sectors: pd.Series) -> pd.Series:
    """Zero out the net exposure inside each sector."""
    df = pd.DataFrame({"w": w, "s": sectors.reindex(w.index)})
    df["w"] = df["w"] - df.groupby("s", observed=True)["w"].transform("mean")
    return df["w"]


def neutralise_exposures(w: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """Residualise the weight vector against a set of exposures, all at once.

    Applying constraints one after another does not work: projecting out beta and
    then subtracting the mean to restore dollar neutrality puts beta straight back
    in, because the constant vector is not orthogonal to the beta vector. The
    correct object is a projection onto the orthogonal complement of the span of
    *all* the constraint vectors together, which is exactly the residual of a
    least-squares regression of the weights on the constraint matrix.

    Passing a column of ones gives dollar neutrality, a column of betas gives
    market neutrality, and sector dummies give sector neutrality, in any
    combination and with no interaction bugs.
    """
    held = w != 0
    if held.sum() < 4 or exposures.empty:
        return w
    X = exposures.loc[held].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.std(axis=0) > 1e-12
    keep |= np.abs(X.mean(axis=0)) > 1e-12  # keep the constant column
    X = X[:, keep]
    if X.shape[1] == 0:
        return w
    y = w[held].to_numpy(dtype=float)
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return w
    out = w.copy()
    out.loc[held] = y - X @ coef
    return out


def volatility_scalar(
    strategy_returns: pd.Series, target_annual: float, lookback: int, cap: float
) -> float:
    """Ratio of target to trailing realised volatility, capped both ways."""
    r = strategy_returns.dropna().tail(lookback)
    if len(r) < max(20, lookback // 3):
        return 1.0
    realised = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if realised <= 1e-8:
        return 1.0
    return float(np.clip(target_annual / realised, 1.0 / cap, cap))


def build_weights_for_date(
    scores: pd.Series,
    cfg: PortfolioConfig,
    returns_window: pd.DataFrame | None = None,
    vol_window: pd.Series | None = None,
    sectors: pd.Series | None = None,
    betas: pd.Series | None = None,
) -> pd.Series:
    """Dispatch to the configured construction method and apply shared constraints."""
    if cfg.method == "rank_long_short":
        w = rank_long_short_weights(
            scores, cfg.n_long, cfg.n_short, cfg.gross_leverage, cfg.max_weight
        )
    elif cfg.method == "mean_variance":
        if returns_window is None or returns_window.empty:
            w = rank_long_short_weights(
                scores, cfg.n_long, cfg.n_short, cfg.gross_leverage, cfg.max_weight
            )
        else:
            w = mean_variance_weights(
                scores, returns_window, cfg.mean_variance.risk_aversion,
                cfg.gross_leverage, cfg.max_weight, cfg.dollar_neutral,
            )
    elif cfg.method == "risk_parity":
        if vol_window is None:
            vol_window = pd.Series(1.0, index=scores.index)
        w = risk_parity_weights(
            scores, vol_window, cfg.gross_leverage, cfg.max_weight, cfg.n_long, cfg.n_short
        )
    else:
        raise ValueError(f"Unknown portfolio method: {cfg.method}")

    # Impose every neutrality constraint simultaneously, never sequentially.
    cons: dict[str, pd.Series] = {}
    if cfg.dollar_neutral:
        cons["const"] = pd.Series(1.0, index=w.index)
    if cfg.beta_neutral and betas is not None:
        b = betas.reindex(w.index)
        cons["beta"] = b.fillna(b.mean() if np.isfinite(b).any() else 1.0)
    if cfg.sector_neutral and sectors is not None:
        sec = sectors.reindex(w.index).astype(str)
        for level in sorted(sec.dropna().unique())[:-1]:  # drop one to avoid collinearity
            cons[f"sec_{level}"] = (sec == level).astype(float)
    if cons:
        w = neutralise_exposures(w, pd.DataFrame(cons))

    # Re-scale to the intended gross exposure. The projection shrinks the book, so
    # without this step the strategy quietly runs under its risk target.
    gross = w.abs().sum()
    if gross > 1e-9:
        w = w / gross * cfg.gross_leverage
    return w.fillna(0.0)
