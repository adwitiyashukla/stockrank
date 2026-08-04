"""Real market data ingestion with an incremental on-disk cache.

Design notes
------------
* Every ticker is cached to its own parquet file. A download that dies halfway
  through 500 names resumes from where it stopped instead of starting over, which
  matters a great deal on a laptop with a flaky connection.
* Prices are split and dividend adjusted at source (``auto_adjust=True``), so the
  ``close`` column is a total-return price series and simple percentage changes
  are economically meaningful.
* Failures are collected rather than raised. A universe of 500 names will always
  contain a handful that Yahoo cannot serve, and the run reports the coverage
  ratio instead of dying.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from alpha_engine.utils.io import ensure_dir
from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)

_COLUMNS = ["open", "high", "low", "close", "volume"]


class DownloadReport:
    """Bookkeeping for a bulk download so the pipeline can report data quality."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self.succeeded: list[str] = []
        self.failed: list[str] = []
        self.from_cache: list[str] = []

    @property
    def coverage(self) -> float:
        return len(self.succeeded) / max(len(self.requested), 1)

    def to_dict(self) -> dict:
        return {
            "n_requested": len(self.requested),
            "n_succeeded": len(self.succeeded),
            "n_failed": len(self.failed),
            "n_from_cache": len(self.from_cache),
            "coverage": round(self.coverage, 4),
            "failed_tickers": sorted(self.failed)[:80],
        }


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "prices" / f"{ticker}.parquet"


def _normalise_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        df = df.xs(ticker, axis=1, level=1) if ticker in set(lvl1) else df.droplevel(1, axis=1)
        del lvl0
    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        return None
    out = df[_COLUMNS].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    out = out[(out["close"] > 0) & (out["volume"] >= 0)]
    if out.empty:
        return None
    out.insert(0, "ticker", ticker)
    return out.reset_index().rename(columns={"index": "date", "Date": "date"})


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
    batch_size: int = 40,
    max_retries: int = 3,
    force_refresh: bool = False,
    pause: float = 0.4,
) -> tuple[pd.DataFrame, DownloadReport]:
    """Download daily adjusted OHLCV for ``tickers`` from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "yfinance is required for real market data. Install with: pip install -e '.[market]'"
        ) from exc

    cache = ensure_dir(Path(cache_dir) / "prices").parent
    # yfinance keeps a shared sqlite timezone cache. With threaded batch downloads
    # that database locks and healthy tickers get reported as delisted, so point it
    # at a run-local directory.
    try:
        yf.set_tz_cache_location(str(ensure_dir(cache / "yf_tz")))
    except Exception:  # noqa: BLE001 - older yfinance versions lack this helper
        pass
    report = DownloadReport()
    report.requested = list(tickers)

    frames: list[pd.DataFrame] = []
    to_fetch: list[str] = []
    for t in tickers:
        p = _cache_path(cache, t)
        if p.exists() and not force_refresh:
            frames.append(pd.read_parquet(p))
            report.succeeded.append(t)
            report.from_cache.append(t)
        else:
            to_fetch.append(t)

    if to_fetch:
        logger.info(
            "Downloading %d tickers from Yahoo Finance (%d already cached)",
            len(to_fetch),
            len(report.from_cache),
        )

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]
        raw = None
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    tickers=" ".join(batch),
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=True,
                    group_by="column",
                )
                break
            except Exception as exc:  # noqa: BLE001 - network is inherently flaky
                wait = 2.0 * (attempt + 1)
                logger.warning("Batch %d attempt %d failed (%s); retrying in %.0fs", i, attempt + 1, exc, wait)
                time.sleep(wait)

        for t in batch:
            df = None
            if raw is not None and not raw.empty:
                try:
                    sub = (
                        raw.xs(t, axis=1, level=1)
                        if isinstance(raw.columns, pd.MultiIndex)
                        else raw
                    )
                    df = _normalise_yf(sub, t)
                except (KeyError, IndexError):
                    df = None
            if df is None:
                report.failed.append(t)
                continue
            df.to_parquet(_cache_path(cache, t), index=False)
            frames.append(df)
            report.succeeded.append(t)

        logger.info(
            "Progress: %d/%d tickers (%d failures so far)",
            min(i + batch_size, len(to_fetch)),
            len(to_fetch),
            len(report.failed),
        )
        time.sleep(pause)

    # Second pass: retry failures one at a time without threading. Most "possibly
    # delisted" errors in a threaded batch are transient rate limits, not real
    # delistings, and this pass typically recovers the majority of them.
    if report.failed:
        retry = list(report.failed)
        logger.info("Retrying %d failed tickers individually", len(retry))
        report.failed = []
        for t in retry:
            df = None
            try:
                raw = yf.download(
                    tickers=t,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                )
                df = _normalise_yf(raw, t)
            except Exception:  # noqa: BLE001
                df = None
            if df is None:
                report.failed.append(t)
                continue
            df.to_parquet(_cache_path(cache, t), index=False)
            frames.append(df)
            report.succeeded.append(t)
            time.sleep(0.15)
        logger.info(
            "Individual retry recovered %d tickers; %d remain unavailable (these are almost "
            "all genuine delistings that Yahoo no longer serves)",
            len(retry) - len(report.failed),
            len(report.failed),
        )

    if not frames:
        raise RuntimeError(
            "No price data could be downloaded. Check the network connection and ticker list."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "Price panel: %d rows, %d tickers, %s to %s, coverage %.1f%%",
        len(panel),
        panel["ticker"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
        100 * report.coverage,
    )
    return panel, report


def download_benchmark(
    symbol: str, start: str, end: str, cache_dir: str | Path = "data/cache"
) -> pd.DataFrame:
    """Daily total-return series for a single benchmark symbol such as SPY."""
    panel, _ = download_prices([symbol], start, end, cache_dir=cache_dir)
    out = panel[["date", "close"]].rename(columns={"close": "benchmark_close"})
    out["mkt_return"] = out["benchmark_close"].pct_change()
    return out.dropna().reset_index(drop=True)
