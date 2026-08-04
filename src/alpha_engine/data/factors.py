"""Fama-French factor returns from the Ken French Data Library.

These are the reference series academic finance uses, which lets the backtest
answer the question a real allocator asks first: is this strategy doing anything
that a cheap exposure to market, size, value, profitability, investment or
momentum would not already have done?
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from alpha_engine.utils.io import ensure_dir
from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _read_french_zip(url: str) -> pd.DataFrame:
    import requests

    resp = requests.get(url, timeout=60, headers={"User-Agent": "alpha-engine/0.1"})
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin-1")

    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:8].isdigit())
    rows = []
    for ln in lines[start:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2 or not parts[0][:8].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    header = [h.strip() for h in lines[start - 1].split(",")]
    header[0] = "date"
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # French reports percent
    return df.dropna().reset_index(drop=True)


def load_fama_french(
    start: str, end: str, cache_dir: str | Path = "data/cache", include_momentum: bool = True
) -> pd.DataFrame:
    """Daily FF5 factors (plus momentum), cached to parquet."""
    cache = ensure_dir(cache_dir)
    path = cache / "fama_french_daily.parquet"
    if path.exists():
        ff = pd.read_parquet(path)
    else:
        logger.info("Fetching Fama-French daily factors from the Ken French Data Library")
        ff = _read_french_zip(FF5_URL)
        ff.columns = [c.lower().replace("-", "_") for c in ff.columns]
        if include_momentum:
            try:
                mom = _read_french_zip(MOM_URL)
                mom.columns = ["date", "mom"]
                ff = ff.merge(mom, on="date", how="left")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Momentum factor unavailable (%s); continuing without it", exc)
        ff.to_parquet(path, index=False)

    ff["date"] = pd.to_datetime(ff["date"])
    out = ff[(ff["date"] >= start) & (ff["date"] <= end)].reset_index(drop=True)
    logger.info("Fama-French factors: %d rows, columns %s", len(out), list(out.columns[1:]))
    return out
