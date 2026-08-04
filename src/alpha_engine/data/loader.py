"""Single entry point that turns a config into a clean, analysis-ready panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_engine.config import Config
from alpha_engine.utils.io import ensure_dir
from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)

MAX_ABS_DAILY_RETURN = 1.0  # 100% in one session almost always means a bad adjustment


@dataclass
class MarketData:
    """Everything downstream stages need, in one immutable-ish container."""

    panel: pd.DataFrame  # date, ticker, sector, open, high, low, close, volume, dollar_volume
    market: pd.DataFrame  # date, mkt_return, rf_rate
    factors: pd.DataFrame | None = None  # Fama-French daily factors
    pit_membership: pd.DataFrame | None = None  # date, ticker
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.panel["ticker"].unique())

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.panel["date"].unique()))

    def close_matrix(self) -> pd.DataFrame:
        return self.panel.pivot(index="date", columns="ticker", values="close").sort_index()

    def returns_matrix(self) -> pd.DataFrame:
        return self.close_matrix().pct_change()

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": int(len(self.panel)),
            "n_tickers": int(self.panel["ticker"].nunique()),
            "start": str(self.panel["date"].min().date()),
            "end": str(self.panel["date"].max().date()),
            "n_trading_days": int(self.panel["date"].nunique()),
            "n_sectors": int(self.panel["sector"].nunique()) if "sector" in self.panel else 0,
            **self.quality,
        }


# --------------------------------------------------------------------- cleaning
def _apply_quality_filters(panel: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    dc = cfg.data
    q: dict[str, Any] = {}
    n0 = len(panel)

    panel = panel[(panel["close"] > 0) & panel["close"].notna()].copy()
    panel = panel.sort_values(["ticker", "date"])
    panel["ret"] = panel.groupby("ticker", sort=False)["close"].pct_change()

    # Suspicious returns usually indicate an unadjusted corporate action.
    bad = panel["ret"].abs() > MAX_ABS_DAILY_RETURN
    q["n_suspicious_returns"] = int(bad.sum())
    offenders = panel.loc[bad, "ticker"].value_counts()
    drop_tickers = set(offenders[offenders >= 3].index)
    if drop_tickers:
        logger.warning("Dropping %d tickers with repeated implausible returns", len(drop_tickers))
    panel = panel[~panel["ticker"].isin(drop_tickers)]
    q["n_tickers_dropped_bad_returns"] = len(drop_tickers)

    # Price floor removes penny stocks where the bid-ask spread dominates any signal.
    med_price = panel.groupby("ticker")["close"].median()
    keep = set(med_price[med_price >= dc.min_price].index)
    q["n_tickers_dropped_price_floor"] = int(panel["ticker"].nunique() - len(keep))
    panel = panel[panel["ticker"].isin(keep)]

    # History floor: a model that needs 252 days of features cannot use a 60-day listing.
    counts = panel.groupby("ticker")["date"].size()
    keep = set(counts[counts >= dc.min_history_days].index)
    q["n_tickers_dropped_short_history"] = int(panel["ticker"].nunique() - len(keep))
    panel = panel[panel["ticker"].isin(keep)]

    panel["dollar_volume"] = panel["close"] * panel["volume"]

    # Liquidity cap keeps the panel inside a laptop's memory budget and keeps the
    # investable universe honest: these are names you could actually trade.
    if dc.max_assets and panel["ticker"].nunique() > dc.max_assets:
        adv = panel.groupby("ticker")["dollar_volume"].median().sort_values(ascending=False)
        keep = set(adv.head(dc.max_assets).index)
        q["n_tickers_dropped_liquidity_cap"] = int(panel["ticker"].nunique() - len(keep))
        panel = panel[panel["ticker"].isin(keep)]

    panel = panel.drop(columns=["ret"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    q["rows_before_cleaning"] = n0
    q["rows_after_cleaning"] = len(panel)
    return panel, q


def _downcast(panel: pd.DataFrame) -> pd.DataFrame:
    for c in ("open", "high", "low", "close", "dollar_volume", "alpha_true"):
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    if "volume" in panel.columns:
        panel["volume"] = panel["volume"].astype("float32")
    for c in ("ticker", "sector"):
        if c in panel.columns:
            panel[c] = panel[c].astype("category")
    return panel


# ------------------------------------------------------------------- assembling
def _load_real(cfg: Config) -> MarketData:
    from alpha_engine.data.providers import download_benchmark, download_prices
    from alpha_engine.data.universe import fetch_sp500_tables, resolve_universe

    dc = cfg.data
    tickers, pit = resolve_universe(
        dc.universe, dc.start, dc.end, cache_dir=dc.cache_dir, universe_file=dc.universe_file
    )
    logger.info("Universe '%s' resolved to %d unique tickers", dc.universe, len(tickers))

    panel, report = download_prices(tickers, dc.start, dc.end, cache_dir=dc.cache_dir)

    # GICS sectors from the current constituent table; unmatched names get "Unknown".
    try:
        current, _ = fetch_sp500_tables(dc.cache_dir)
        sector_map = dict(zip(current["ticker"], current["sector"], strict=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sector metadata unavailable (%s)", exc)
        sector_map = {}
    panel["sector"] = panel["ticker"].map(sector_map).fillna("Unknown")

    panel, quality = _apply_quality_filters(panel, cfg)
    quality["download"] = report.to_dict()

    bench = download_benchmark(dc.benchmark, dc.start, dc.end, cache_dir=dc.cache_dir)
    market = bench[["date", "mkt_return"]].copy()

    factors = None
    if dc.use_fama_french:
        try:
            from alpha_engine.data.factors import load_fama_french

            factors = load_fama_french(dc.start, dc.end, cache_dir=dc.cache_dir)
            market = market.merge(factors[["date", "rf"]], on="date", how="left")
            market = market.rename(columns={"rf": "rf_rate"})
            market["rf_rate"] = market["rf_rate"].ffill().fillna(0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fama-French factors unavailable (%s); using a flat risk-free rate", exc)
    if "rf_rate" not in market.columns:
        market["rf_rate"] = 0.02 / 252

    if pit is not None:
        pit = pit[pit["ticker"].isin(set(panel["ticker"].unique()))].reset_index(drop=True)

    return MarketData(
        panel=_downcast(panel), market=market, factors=factors, pit_membership=pit, quality=quality
    )


def _load_synthetic(cfg: Config) -> MarketData:
    from alpha_engine.data.simulator import simulate_market

    panel, market = simulate_market(cfg.data, cfg.simulator, seed=cfg.run.seed)
    panel, quality = _apply_quality_filters(panel, cfg)
    quality["source"] = "synthetic_control"
    return MarketData(panel=_downcast(panel), market=market, quality=quality)


def load_market_data(cfg: Config, cache: bool = True) -> MarketData:
    """Build (or reload) the market panel described by ``cfg``."""
    cache_path = Path(cfg.data.cache_dir) / f"panel_{cfg.run.name}.parquet"
    mkt_path = Path(cfg.data.cache_dir) / f"market_{cfg.run.name}.parquet"

    if cache and cache_path.exists() and mkt_path.exists():
        logger.info("Loading cached panel from %s", cache_path)
        panel = pd.read_parquet(cache_path)
        market = pd.read_parquet(mkt_path)

        # Factors are cheap to reload and the evaluation stage needs them for
        # attribution, so the cache path must not silently drop them.
        factors = None
        if cfg.data.use_fama_french:
            try:
                from alpha_engine.data.factors import load_fama_french

                factors = load_fama_french(cfg.data.start, cfg.data.end, cache_dir=cfg.data.cache_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fama-French factors unavailable on the cache path (%s)", exc)

        summary_path = Path(cfg.data.cache_dir) / f"data_summary_{cfg.run.name}.json"
        quality = {"source": "cache"}
        if summary_path.exists():
            try:
                import json

                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                quality["download"] = cached.get("download", {})
            except Exception:  # noqa: BLE001
                pass
        return MarketData(
            panel=_downcast(panel), market=market, factors=factors, quality=quality
        )

    data = _load_synthetic(cfg) if cfg.data.source == "synthetic" else _load_real(cfg)

    if cache:
        ensure_dir(cache_path.parent)
        data.panel.to_parquet(cache_path, index=False)
        data.market.to_parquet(mkt_path, index=False)

    logger.info("Market data ready: %s", data.summary())
    return data


def align_calendar(panel: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    """Drop dates with too thin a cross-section to rank meaningfully."""
    counts = panel.groupby("date")["ticker"].size()
    good = set(counts[counts >= min_names].index)
    removed = len(counts) - len(good)
    if removed:
        logger.info("Dropped %d dates with fewer than %d listed names", removed, min_names)
    return panel[panel["date"].isin(good)].reset_index(drop=True)


def infer_trading_calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(panel["date"].unique()))
