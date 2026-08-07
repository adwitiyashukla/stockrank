"""Daily backtest with explicit, conservative cost assumptions.

What this engine does and does not claim
----------------------------------------
It is a *portfolio* backtester, not an execution simulator. Positions are marked
against daily closes and trades are assumed to fill at the close with a linear
cost. It does not model the order book, queue position, or market impact beyond a
fixed slippage term. That is the honest scope, and it is the standard scope for
research at this horizon.

What it does model, because ignoring these is how backtests lie:

* **A one-day execution lag**, already baked into the label, so a signal formed on
  the close of day t is traded into on day t+1.
* **Turnover-proportional costs**, charged on the change in weights, not on gross
  exposure. Commission plus half-spread plus slippage, in basis points per unit
  of one-way turnover.
* **Short financing.** Borrowing stock is not free; a flat annual rate is charged
  on the short leg every day it is held.
* **A turnover cap**, which blends the target book towards the current book when
  the model wants to trade more than the cap allows. Without it, a high-frequency
  signal shows enormous gross returns and negative net returns.
* **Volatility targeting on trailing data only**, recomputed on each rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockrank.config import BacktestConfig, PortfolioConfig
from stockrank.portfolio.construction import build_weights_for_date, volatility_scalar
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Daily series and summary statistics for one strategy run."""

    returns: pd.Series  # net of costs
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    exposure: pd.DataFrame  # gross, net, n_long, n_short, vol_scalar
    weights: pd.DataFrame  # date x ticker
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns.fillna(0.0)).cumprod()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "net_return": self.returns,
                "gross_return": self.gross_returns,
                "cost": self.costs,
                "turnover": self.turnover,
                "equity": self.equity_curve,
            }
        ).join(self.exposure)


def _pivot(predictions: pd.DataFrame, col: str) -> pd.DataFrame:
    return predictions.pivot_table(index="date", columns="ticker", values=col, observed=True).sort_index()


def run_backtest(
    predictions: pd.DataFrame,
    score_col: str,
    pcfg: PortfolioConfig,
    bcfg: BacktestConfig,
    sectors: pd.Series | None = None,
    daily_returns: pd.DataFrame | None = None,
    label_horizon: int | None = None,
) -> BacktestResult:
    """Simulate the strategy implied by ``score_col``, marking to market daily.

    Timing, spelled out because it is where backtests usually cheat:

    * A score stamped on date ``d`` is formed from the close of ``d``.
    * The book is traded into at the close of ``d + 1``. Costs are charged there.
    * Profit and loss accrues daily from the close of ``d + 1`` to the close of
      ``d + 1 + horizon``, using the actual daily returns of each holding.
    * The next rebalance happens on ``d + horizon``.

    Marking daily rather than booking one lump return per period matters a great
    deal for the statistics. Spreading a single period return evenly across its
    days produces an artificially smooth series whose annualised Sharpe is
    inflated by roughly the square root of the holding period. Here every day
    carries its own realised return, so the volatility, the drawdowns and the
    Sharpe ratio are all computed on genuinely independent observations.

    Positions are held at fixed weights rather than allowed to drift, which is
    what a leverage-targeted book does in practice.
    """
    scores = _pivot(predictions, score_col)
    fwd = _pivot(predictions, "fwd_return")
    betas = _pivot(predictions, "beta_raw") if "beta_raw" in predictions.columns else None
    dates = scores.index
    tickers = scores.columns
    del fwd

    horizon = max(int(label_horizon or pcfg.rebalance_days), 1)
    if label_horizon and int(pcfg.rebalance_days) != int(label_horizon):
        logger.warning(
            "rebalance_days=%d does not match the label horizon of %d; using %d",
            pcfg.rebalance_days, label_horizon, horizon,
        )

    if daily_returns is None:
        raise ValueError(
            "run_backtest needs a daily returns matrix to mark the book to market"
        )
    rets = daily_returns.reindex(index=dates, columns=tickers).fillna(0.0)

    cost_rate = (bcfg.cost_bps + bcfg.slippage_bps) / 10_000.0
    borrow_daily = bcfg.borrow_bps_annual / 10_000.0 / TRADING_DAYS
    execution_lag = 1

    n = len(dates)
    daily_net = pd.Series(0.0, index=dates)
    daily_gross = pd.Series(0.0, index=dates)
    daily_cost = pd.Series(0.0, index=dates)
    turnover_s = pd.Series(0.0, index=dates)

    prev_w = pd.Series(0.0, index=tickers)
    w_hist: dict[pd.Timestamp, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    realised_so_far: list[float] = []
    first_active: int | None = None

    for i in range(0, n, horizon):
        d = dates[i]
        s = scores.loc[d].dropna()
        if s.empty:
            continue

        rw = vw = None
        hist = rets.iloc[max(0, i - pcfg.mean_variance.cov_lookback_days) : i]
        if len(hist) > 30:
            rw = hist[[c for c in hist.columns if c in s.index]]
            vw = rw.std()

        beta_row = betas.loc[d].reindex(tickers) if betas is not None and d in betas.index else None
        target = build_weights_for_date(
            s.reindex(tickers), pcfg, returns_window=rw, vol_window=vw,
            sectors=sectors, betas=beta_row,
        )

        scalar = volatility_scalar(
            pd.Series(realised_so_far), pcfg.vol_target_annual,
            pcfg.vol_lookback_days, pcfg.max_vol_scalar,
        )
        target = target * scalar

        raw_turnover = float((target - prev_w).abs().sum())
        if bcfg.turnover_cap_daily > 0:
            cap = bcfg.turnover_cap_daily * horizon
            if raw_turnover > cap and raw_turnover > 0:
                target = prev_w + (target - prev_w) * (cap / raw_turnover)

        turnover = float((target - prev_w).abs().sum())
        entry = i + execution_lag
        exit_ = min(entry + horizon, n)
        if entry >= n:
            break
        if first_active is None:
            first_active = entry

        # Costs land on the day the book is actually traded into.
        daily_cost.iloc[entry] += turnover * cost_rate
        turnover_s.iloc[entry] = turnover

        short_notional = float(target[target < 0].abs().sum())
        window = rets.iloc[entry:exit_]
        pnl = window.to_numpy() @ target.reindex(window.columns).fillna(0.0).to_numpy()

        daily_gross.iloc[entry:exit_] += pnl
        daily_cost.iloc[entry:exit_] += short_notional * borrow_daily
        realised_so_far.extend(pnl.tolist())

        w_hist[d] = target.copy()
        rows.append(
            {
                "date": dates[entry],
                "turnover": turnover,
                "gross_exposure": float(target.abs().sum()),
                "net_exposure": float(target.sum()),
                "n_long": int((target > 1e-9).sum()),
                "n_short": int((target < -1e-9).sum()),
                "net_beta": float((target * beta_row.fillna(1.0)).sum()) if beta_row is not None else np.nan,
                "vol_scalar": scalar,
            }
        )
        prev_w = target

    if not rows:
        raise RuntimeError(f"Backtest produced no periods for score column {score_col}")

    daily_net = daily_gross - daily_cost
    per = pd.DataFrame(rows).set_index("date").sort_index()

    active = dates[first_active:] if first_active is not None else dates
    exposure = per[
        ["gross_exposure", "net_exposure", "n_long", "n_short", "net_beta", "vol_scalar"]
    ].reindex(active).ffill()
    weights = pd.DataFrame(w_hist).T
    weights.index = [dates[min(dates.get_loc(d) + execution_lag, n - 1)] for d in weights.index]
    weights = weights.reindex(active).ffill().fillna(0.0)

    result = BacktestResult(
        returns=daily_net.reindex(active),
        gross_returns=daily_gross.reindex(active),
        costs=daily_cost.reindex(active),
        turnover=turnover_s.reindex(active),
        exposure=exposure,
        weights=weights,
    )
    r = result.returns
    logger.info(
        "Backtest %-16s | net ann %+.2f%% | vol %.2f%% | Sharpe %+.2f | turnover/rebal %.2f | net beta %+.3f",
        score_col,
        100 * r.mean() * TRADING_DAYS,
        100 * r.std(ddof=1) * np.sqrt(TRADING_DAYS),
        (r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan,
        per["turnover"].mean(),
        float(per["net_beta"].mean()) if "net_beta" in per else np.nan,
    )
    return result


def buy_and_hold_benchmark(market_returns: pd.Series, index: pd.Index) -> pd.Series:
    return market_returns.reindex(index).fillna(0.0)
