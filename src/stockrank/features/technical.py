from __future__ import annotations

import numpy as np
import pandas as pd

from stockrank.config import FeatureConfig
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

EPS = 1e-12


def _safe_div(a: pd.DataFrame, b: pd.DataFrame | pd.Series) -> pd.DataFrame:
    return a.div(b.replace(0.0, np.nan))


def momentum_features(logret: pd.DataFrame, windows: list[int]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    cum = logret.cumsum()
    for w in windows:
        out[f"ret_{w}"] = cum - cum.shift(w)
    if 252 in windows and 21 in windows:
        out["mom_12_1"] = (cum.shift(21) - cum.shift(252))
    if 126 in windows and 21 in windows:
        out["mom_6_1"] = cum.shift(21) - cum.shift(126)
    return out


def volatility_features(
    logret: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, windows: list[int]
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for w in windows:
        out[f"vol_{w}"] = logret.rolling(w, min_periods=max(3, w // 2)).std()
    if len(windows) >= 2:
        short, long = min(windows), max(windows)
        out["vol_ratio"] = _safe_div(out[f"vol_{short}"], out[f"vol_{long}"])

    w = max(windows)
    hl = np.log(_safe_div(high, low)) ** 2
    out["parkinson_vol"] = np.sqrt(hl.rolling(w, min_periods=w // 2).mean() / (4 * np.log(2)))
    out["downside_vol"] = logret.where(logret < 0).rolling(w, min_periods=w // 2).std()
    out["ret_skew"] = logret.rolling(w, min_periods=w // 2).skew()
    out["ret_kurt"] = logret.rolling(w, min_periods=w // 2).kurt()
    return out


def trend_features(
    close: pd.DataFrame, ma_windows: list[int]
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    mas: dict[int, pd.DataFrame] = {}
    for w in ma_windows:
        ma = close.rolling(w, min_periods=max(3, w // 2)).mean()
        mas[w] = ma
        out[f"px_to_ma_{w}"] = _safe_div(close, ma) - 1.0
    if len(ma_windows) >= 2:
        a, b = sorted(ma_windows)[0], sorted(ma_windows)[-1]
        out["ma_cross"] = _safe_div(mas[a], mas[b]) - 1.0

    hi52 = close.rolling(252, min_periods=60).max()
    lo52 = close.rolling(252, min_periods=60).min()
    out["dist_52w_high"] = _safe_div(close, hi52) - 1.0
    out["dist_52w_low"] = _safe_div(close, lo52) - 1.0

    m21 = close.rolling(21, min_periods=10).mean()
    s21 = close.rolling(21, min_periods=10).std()
    out["bollinger_z"] = _safe_div(close - m21, s21)
    return out


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def macd_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    return {
        "macd_norm": _safe_div(macd, close),
        "macd_hist": _safe_div(macd - signal, close),
    }


def liquidity_features(
    logret: pd.DataFrame, volume: pd.DataFrame, dollar_volume: pd.DataFrame, windows: list[int]
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    log_dv = np.log1p(dollar_volume)
    short, long = min(windows), max(windows)
    out["log_dollar_volume"] = log_dv
    out["dv_trend"] = log_dv.rolling(short, min_periods=3).mean() - log_dv.rolling(
        long, min_periods=long // 2
    ).mean()
    vol_mean = volume.rolling(long, min_periods=long // 2).mean()
    out["volume_ratio"] = _safe_div(volume, vol_mean)
    illiq = _safe_div(logret.abs(), dollar_volume) * 1e9
    out["amihud_illiq"] = np.log1p(illiq.rolling(long, min_periods=long // 2).mean())
    return out


def market_relative_features(
    logret: pd.DataFrame, mkt_logret: pd.Series, window: int = 63
) -> dict[str, pd.DataFrame]:
    mp = max(window // 2, 20)
    var_m = mkt_logret.rolling(window, min_periods=mp).var()
    cov = logret.rolling(window, min_periods=mp).cov(mkt_logret)
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)

    mean_i = logret.rolling(window, min_periods=mp).mean()
    mean_m = mkt_logret.rolling(window, min_periods=mp).mean()
    alpha = mean_i.sub(beta.mul(mean_m, axis=0))

    resid = logret.sub(beta.mul(mkt_logret, axis=0)).sub(alpha)
    idio_vol = resid.rolling(window, min_periods=mp).std()

    std_i = logret.rolling(window, min_periods=mp).std()
    std_m = mkt_logret.rolling(window, min_periods=mp).std()
    corr = cov.div((std_i.mul(std_m, axis=0)).replace(0.0, np.nan))

    return {
        "beta_63": beta,
        "alpha_63": alpha,
        "idio_vol_63": idio_vol,
        "corr_mkt_63": corr,
    }


def build_feature_matrices(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    mkt_return: pd.Series,
    cfg: FeatureConfig,
) -> dict[str, pd.DataFrame]:
    logret = np.log(close).diff()
    mkt_logret = np.log1p(mkt_return.reindex(close.index)).ffill()

    feats: dict[str, pd.DataFrame] = {}
    feats.update(momentum_features(logret, cfg.return_windows))
    feats.update(volatility_features(logret, high, low, cfg.vol_windows))
    feats.update(trend_features(close, cfg.ma_windows))
    feats["rsi_14"] = rsi(close, 14) / 100.0 - 0.5
    feats.update(macd_features(close))
    feats.update(liquidity_features(logret, volume, dollar_volume, cfg.volume_windows))
    feats.update(market_relative_features(logret, mkt_logret, window=63))

    logger.info("Built %d raw feature matrices", len(feats))
    return feats
