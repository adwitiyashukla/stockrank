from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from stockrank.utils.io import ensure_dir
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_TICKER_FIXES = {"BRK.B": "BRK-B", "BF.B": "BF-B", "BF.A": "BF-A", "GEV": "GEV", "LEN.B": "LEN-B"}


def _clean_ticker(t: str) -> str:
    t = str(t).strip().upper()
    t = re.sub(r"\s+", "", t)
    return _TICKER_FIXES.get(t, t.replace(".", "-"))


def fetch_sp500_tables(cache_dir: str | Path = "data/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = ensure_dir(cache_dir)
    cur_p, chg_p = cache / "sp500_current.parquet", cache / "sp500_changes.parquet"
    if cur_p.exists() and chg_p.exists():
        return pd.read_parquet(cur_p), pd.read_parquet(chg_p)

    logger.info("Fetching S&P 500 membership tables from Wikipedia")
    import requests

    resp = requests.get(
        WIKI_SP500,
        timeout=60,
        headers={"User-Agent": "stockrank/0.1 (research project; contact via GitHub)"},
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    current = tables[0].copy()
    current.columns = [str(c).strip() for c in current.columns]
    sym_col = next(c for c in current.columns if "Symbol" in c)
    sec_col = next((c for c in current.columns if "GICS Sector" in c), None)
    name_col = next((c for c in current.columns if "Security" in c), None)
    current = current.rename(
        columns={sym_col: "ticker", sec_col: "sector", name_col: "name"}
    )[["ticker", "name", "sector"]]
    current["ticker"] = current["ticker"].map(_clean_ticker)

    changes = tables[1].copy()
    changes.columns = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in changes.columns]
    date_col = next(c for c in changes.columns if "Date" in c)
    add_col = next(c for c in changes.columns if "Added" in c and "Ticker" in c)
    rem_col = next(c for c in changes.columns if "Removed" in c and "Ticker" in c)
    changes = changes.rename(columns={date_col: "date", add_col: "added", rem_col: "removed"})[
        ["date", "added", "removed"]
    ]
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce", format="mixed")
    changes = changes.dropna(subset=["date"])
    for col in ("added", "removed"):
        changes[col] = changes[col].apply(lambda x: _clean_ticker(x) if pd.notna(x) else "")
        changes[col] = changes[col].where(
            changes[col].str.fullmatch(r"[A-Z][A-Z0-9-]{0,5}"), ""
        )
    changes["added"] = changes["added"].astype(str)
    changes["removed"] = changes["removed"].astype(str)

    current.to_parquet(cur_p, index=False)
    changes.to_parquet(chg_p, index=False)
    logger.info("Cached %d current members and %d change events", len(current), len(changes))
    return current, changes


def build_pit_membership(
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
    freq: str = "MS",
) -> pd.DataFrame:
    current, changes = fetch_sp500_tables(cache_dir)
    members = set(current["ticker"])
    snapshots: dict[pd.Timestamp, set[str]] = {}

    grid = pd.date_range(start=start, end=end, freq=freq)
    changes = changes.sort_values("date", ascending=False)
    today = pd.Timestamp.today().normalize()

    cursor = today
    idx = 0
    for snap in reversed(grid):
        while idx < len(changes) and changes.iloc[idx]["date"] > snap:
            row = changes.iloc[idx]
            added, removed = str(row["added"] or ""), str(row["removed"] or "")
            if added and added != "nan":
                members.discard(added)
            if removed and removed != "nan":
                members.add(removed)
            idx += 1
        snapshots[snap] = set(members)
        cursor = snap

    del cursor
    frame = pd.DataFrame(
        [(d, t) for d, ts in snapshots.items() for t in sorted(ts)], columns=["date", "ticker"]
    ).sort_values(["date", "ticker"])
    logger.info(
        "Point-in-time universe: %d snapshots, %d to %d names per snapshot",
        frame["date"].nunique(),
        frame.groupby("date").size().min(),
        frame.groupby("date").size().max(),
    )
    return frame.reset_index(drop=True)


def resolve_universe(
    universe: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
    universe_file: str | Path | None = None,
) -> tuple[list[str], pd.DataFrame | None]:
    if universe == "file":
        if universe_file is None:
            raise ValueError("universe='file' requires universe_file")
        tickers = [
            _clean_ticker(line)
            for line in Path(universe_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return sorted(set(tickers)), None

    if universe == "sp500":
        current, _ = fetch_sp500_tables(cache_dir)
        return sorted(set(current["ticker"])), None

    if universe == "sp500_pit":
        pit = build_pit_membership(start, end, cache_dir)
        return sorted(set(pit["ticker"])), pit

    raise ValueError(f"Unknown universe: {universe}")
