from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)

    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)

    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
