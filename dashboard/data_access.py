from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

SEARCH_ROOTS = ["artifacts", "demo_artifacts"]


def find_runs() -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        r = Path(root)
        if not r.exists():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and (d / "model_metrics.csv").exists():
                label = d.name if root == "artifacts" else f"{d.name} (demo)"
                runs.setdefault(label, d)
    return runs


def _json(p: Path) -> Any:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _csv(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_run(path_str: str) -> dict[str, Any]:
    p = Path(path_str)
    out: dict[str, Any] = {
        "name": p.name,
        "path": str(p),
        "data_summary": _json(p / "data_summary.json"),
        "feature_summary": _json(p / "feature_summary.json"),
        "performance": _json(p / "performance.json"),
        "significance": _json(p / "significance.json"),
        "attribution": _json(p / "attribution.json"),
        "timings": _json(p / "timings.json"),
        "folds": _json(p / "folds.json"),
        "model_metrics": _csv(p / "model_metrics.csv"),
        "fold_metrics": _csv(p / "fold_metrics.csv"),
        "importance": _csv(p / "feature_importance.csv", index_col=0),
        "ladders": _csv(p / "quantile_ladders.csv", index_col=0),
        "volatility": _csv(p / "volatility_comparison.csv"),
        "monthly": _csv(p / "monthly_returns.csv", index_col=0),
    }

    bts: dict[str, pd.DataFrame] = {}
    for f in sorted(p.glob("backtest_*.parquet")):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        bts[f.stem.replace("backtest_", "")] = df
    out["backtests"] = bts

    cfg_p = p / "config.yaml"
    if cfg_p.exists():
        import yaml

        out["config"] = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    else:
        out["config"] = {}
    return out


@st.cache_data(show_spinner=False)
def load_predictions(path_str: str, tail_days: int = 400) -> pd.DataFrame:
    p = Path(path_str) / "predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    if tail_days > 0:
        cutoff = df["date"].drop_duplicates().nlargest(tail_days).min()
        df = df[df["date"] >= cutoff]
    return df


@st.cache_data(show_spinner=False)
def load_all_predictions(path_str: str) -> pd.DataFrame:
    p = Path(path_str) / "predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df
