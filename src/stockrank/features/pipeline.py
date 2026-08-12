from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockrank.config import Config
from stockrank.data.loader import MarketData
from stockrank.features.cross_section import normalise, sector_neutralise
from stockrank.features.labels import build_labels
from stockrank.features.technical import build_feature_matrices
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

ID_COLS = ["date", "ticker"]
AUX_COLS = ["sector", "target", "fwd_return", "close", "dollar_volume"]


@dataclass
class FeatureSet:

    frame: pd.DataFrame
    feature_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_names]

    @property
    def y(self) -> pd.Series:
        return self.frame["target"]

    @property
    def dates(self) -> pd.Series:
        return self.frame["date"]

    def slice_dates(self, start, end) -> pd.DataFrame:
        m = (self.frame["date"] >= start) & (self.frame["date"] <= end)
        return self.frame.loc[m]

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": int(len(self.frame)),
            "n_features": len(self.feature_names),
            "n_tickers": int(self.frame["ticker"].nunique()),
            "start": str(self.frame["date"].min().date()),
            "end": str(self.frame["date"].max().date()),
            **self.meta,
        }


def _wide(md: MarketData, value: str) -> pd.DataFrame:
    return md.panel.pivot(index="date", columns="ticker", values=value).sort_index()


def build_feature_set(md: MarketData, cfg: Config, execution_lag: int = 1) -> FeatureSet:
    close = _wide(md, "close").astype("float64")
    high = _wide(md, "high").astype("float64")
    low = _wide(md, "low").astype("float64")
    volume = _wide(md, "volume").astype("float64")
    dollar_volume = _wide(md, "dollar_volume").astype("float64")

    market = md.market.set_index("date").sort_index()
    mkt_ret = market["mkt_return"].reindex(close.index).ffill().fillna(0.0)

    raw = build_feature_matrices(close, high, low, volume, dollar_volume, mkt_ret, cfg.features)

    target, fwd = build_labels(
        close,
        cfg.label,
        vol=raw.get("vol_21"),
        raw_vol=raw.get("vol_63"),
        beta=raw.get("beta_63"),
        mkt_return=mkt_ret,
        execution_lag=execution_lag,
    )

    feature_names = sorted(raw.keys())
    norm = {
        name: normalise(raw[name], cfg.features.standardise, cfg.features.winsorize_q)
        for name in feature_names
    }

    stacked_valid = np.zeros(close.shape, dtype=np.int16)
    for name in feature_names:
        stacked_valid += np.isfinite(norm[name].to_numpy(dtype="float32")).astype(np.int16)
    min_features = int(0.8 * len(feature_names))
    mask = np.isfinite(target.to_numpy(dtype="float64")) & (stacked_valid >= min_features)

    n_dates, n_tickers = close.shape
    dates_arr = np.repeat(close.index.to_numpy(), n_tickers)
    tick_arr = np.tile(np.asarray(close.columns, dtype=object), n_dates)
    flat = mask.ravel()

    data: dict[str, np.ndarray] = {
        "date": dates_arr[flat],
        "ticker": tick_arr[flat],
        "target": target.to_numpy(dtype="float32").ravel()[flat],
        "fwd_return": fwd.to_numpy(dtype="float32").ravel()[flat],
        "close": close.to_numpy(dtype="float32").ravel()[flat],
        "dollar_volume": dollar_volume.to_numpy(dtype="float32").ravel()[flat],
        "beta_raw": raw["beta_63"].to_numpy(dtype="float32").ravel()[flat],
        "vol_raw": raw["vol_63"].to_numpy(dtype="float32").ravel()[flat],
    }
    for name in feature_names:
        data[name] = norm[name].to_numpy(dtype="float32").ravel()[flat]

    frame = pd.DataFrame(data)
    frame["date"] = pd.to_datetime(frame["date"])

    if "sector" in md.panel.columns:
        meta_rows = md.panel[["ticker", "sector"]].astype(str).drop_duplicates("ticker")
        sector_map = dict(zip(meta_rows["ticker"], meta_rows["sector"], strict=False))
        frame["sector"] = frame["ticker"].astype(str).map(sector_map).fillna("Unknown")
    else:
        frame["sector"] = "Unknown"
    frame["sector"] = frame["sector"].astype("category")

    frame[feature_names] = frame[feature_names].fillna(0.0)

    if cfg.portfolio.sector_neutral:
        frame = sector_neutralise(frame, feature_names)

    counts = frame.groupby("date", observed=True)["ticker"].transform("size")
    before = len(frame)
    frame = frame[counts >= cfg.data.min_names_per_date].reset_index(drop=True)

    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)

    meta = {
        "execution_lag_days": execution_lag,
        "label_type": cfg.label.type,
        "label_horizon": cfg.label.horizon,
        "normalisation": cfg.features.standardise,
        "rows_dropped_thin_dates": int(before - len(frame)),
        "mean_cross_section": round(float(frame.groupby("date", observed=True).size().mean()), 1),
    }
    fs = FeatureSet(frame=frame, feature_names=feature_names, meta=meta)
    logger.info("Feature set ready: %s", fs.summary())
    return fs
