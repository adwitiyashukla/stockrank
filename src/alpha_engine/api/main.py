"""FastAPI scoring service.

Exposes the trained forecaster and the latest cross-sectional screen. Kept
deliberately thin: it loads artifacts produced by the research pipeline and does
no modelling of its own, so what is served is exactly what was evaluated.

    uvicorn alpha_engine.api.main:app --reload --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from alpha_engine import __version__
from alpha_engine.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ScoredItem,
    ScoreRequest,
    ScoreResponse,
    ScreenResponse,
)
from alpha_engine.utils.logging import get_logger, setup_logging

logger = get_logger("api")
setup_logging()

ARTIFACT_ROOTS = [Path("artifacts"), Path("demo_artifacts")]

DISCLAIMER = (
    "Research output from a historical backtest. Not investment advice, not a "
    "recommendation, and the reference date is the end of the research sample "
    "rather than today."
)

app = FastAPI(
    title="Equity Alpha Engine API",
    description=(
        "Cross-sectional equity return forecasting. Serves the model evaluated in the "
        "walk-forward study, plus the latest ranked screen and the run's headline statistics."
    ),
    version=__version__,
    contact={"name": "Adwitiya Shukla"},
    license_info={"name": "MIT"},
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _run_dir(run: str) -> Path:
    for root in ARTIFACT_ROOTS:
        p = root / run
        if p.exists():
            return p
    raise HTTPException(404, f"Run '{run}' not found. Available: {available_runs()}")


def available_runs() -> list[str]:
    out: list[str] = []
    for root in ARTIFACT_ROOTS:
        if root.exists():
            out += [d.name for d in root.iterdir() if d.is_dir() and (d / "model_metrics.csv").exists()]
    return sorted(set(out))


@lru_cache(maxsize=8)
def _load_model(run: str) -> tuple[Any, list[str], str]:
    from alpha_engine.models.persistence import load_model

    d = _run_dir(run)
    candidates = sorted(d.glob("model_*.joblib"))
    if not candidates:
        raise HTTPException(
            404,
            f"No serialised model in run '{run}'. Run the pipeline to produce one: "
            "python -m alpha_engine.cli run --config configs/default.yaml",
        )
    model, feats = load_model(candidates[0])
    return model, feats, candidates[0].stem.replace("model_", "")


@lru_cache(maxsize=8)
def _load_predictions(run: str) -> pd.DataFrame:
    p = _run_dir(run) / "predictions.parquet"
    if not p.exists():
        raise HTTPException(404, f"No predictions artifact in run '{run}'")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "Equity Alpha Engine",
        "version": __version__,
        "docs": "/docs",
        "runs": available_runs(),
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    runs = available_runs()
    loaded = []
    for r in runs:
        try:
            loaded.append(f"{r}:{_load_model(r)[2]}")
        except HTTPException:
            continue
    return HealthResponse(version=__version__, models_loaded=loaded, runs_available=runs)


@app.get("/runs/{run}/metrics", response_model=MetricsResponse, tags=["research"])
def metrics(run: str) -> MetricsResponse:
    d = _run_dir(run)
    mm = pd.read_csv(d / "model_metrics.csv") if (d / "model_metrics.csv").exists() else pd.DataFrame()
    return MetricsResponse(
        run=run,
        data=_read_json(d / "data_summary.json"),
        model_metrics=mm.replace({np.nan: None}).to_dict("records"),
        performance=_read_json(d / "performance.json"),
        significance=_read_json(d / "significance.json"),
    )


@app.get("/runs/{run}/features", tags=["research"])
def features(run: str) -> dict:
    """Feature contract the model expects, in order."""
    _, feats, name = _load_model(run)
    return {"run": run, "model": name, "n_features": len(feats), "feature_names": feats}


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a batch of feature vectors.

    Features must already be cross-sectionally normalised the same way the training
    pipeline normalises them. Missing features are filled with zero, which for a
    z-scored input means "at the cross-sectional average", i.e. no view.
    """
    model, feats, name = _load_model(req.run)
    if not req.observations:
        raise HTTPException(400, "No observations supplied")

    warnings: list[str] = []
    rows = []
    for obs in req.observations:
        missing = [f for f in feats if f not in obs.features]
        if missing and len(warnings) < 3:
            warnings.append(f"{obs.ticker}: {len(missing)} of {len(feats)} features missing, filled with 0")
        rows.append([float(obs.features.get(f, 0.0)) for f in feats])

    frame = pd.DataFrame(rows, columns=feats)
    frame["ticker"] = [o.ticker for o in req.observations]
    frame["date"] = pd.Timestamp.today().normalize()

    preds = np.asarray(model.predict(frame), dtype=float)
    results = [ScoredItem(ticker=t, score=float(s)) for t, s in zip(frame["ticker"], preds, strict=False)]

    if req.rank_within_request and len(results) > 1:
        order = np.argsort(-preds)
        for rank, i in enumerate(order, start=1):
            results[i].rank = rank
            results[i].percentile = float(1.0 - (rank - 0.5) / len(results))
    else:
        warnings.append(
            "A single score is not meaningful on its own: the model is trained to order a "
            "cross-section, not to predict an absolute return."
        )

    return ScoreResponse(run=req.run, model=name, n_scored=len(results), results=results, warnings=warnings)


@app.get("/runs/{run}/screen", response_model=ScreenResponse, tags=["inference"])
def screen(
    run: str,
    model: str = Query("ensemble", description="Which prediction column to rank on"),
    n: int = Query(20, ge=1, le=100),
) -> ScreenResponse:
    """Top longs and shorts on the most recent date in the run's sample."""
    df = _load_predictions(run)
    col = f"pred_{model}"
    if col not in df.columns:
        available = [c[5:] for c in df.columns if c.startswith("pred_")]
        raise HTTPException(404, f"Model '{model}' not in run. Available: {available}")

    latest = df["date"].max()
    day = df[df["date"] == latest].copy()
    keep = [c for c in ["ticker", "sector", col, "close", "beta_raw", "vol_raw"] if c in day.columns]

    def _fmt(sub: pd.DataFrame) -> list[dict]:
        out = sub[keep].rename(columns={col: "score", "beta_raw": "beta", "vol_raw": "volatility"})
        return out.replace({np.nan: None}).to_dict("records")

    return ScreenResponse(
        run=run,
        model=model,
        as_of=str(latest.date()),
        n_universe=int(len(day)),
        longs=_fmt(day.nlargest(n, col)),
        shorts=_fmt(day.nsmallest(n, col)),
        disclaimer=DISCLAIMER,
    )


@app.get("/runs", tags=["meta"])
def runs() -> dict:
    return {"runs": available_runs()}
