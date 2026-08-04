"""Saving and loading a production forecaster.

The walk-forward loop fits one model per fold, which is right for evaluation and
useless for deployment. For serving, a single model is refitted on the most
recent training window and persisted with the exact feature list and the
normalisation contract it expects, so the API cannot silently score a differently
ordered or differently scaled matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from alpha_engine.utils.io import ensure_dir
from alpha_engine.utils.logging import get_logger

logger = get_logger(__name__)


def save_model(model, feature_names: list[str], path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump({"model": model, "feature_names": list(feature_names)}, p)
    meta = {
        "model_name": getattr(model, "name", model.__class__.__name__),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        **(metadata or {}),
    }
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved production model to %s", p)
    return p


def load_model(path: str | Path) -> tuple[Any, list[str]]:
    payload = joblib.load(Path(path))
    return payload["model"], payload["feature_names"]


def fit_production_model(fs, cfg, model_name: str = "lightgbm", train_days: int | None = None):
    """Refit a single model on the most recent window for serving."""
    from alpha_engine.models.registry import build_model

    frame = fs.frame
    days = train_days or cfg.validation.train_window_days
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    cutoff = dates[max(0, len(dates) - days)]
    train = frame[frame["date"] >= cutoff]

    model = build_model(model_name, fs.feature_names, cfg)
    model.fit(train)
    logger.info(
        "Production model '%s' fitted on %d rows from %s to %s",
        model_name, len(train), cutoff.date(), dates[-1].date(),
    )
    return model, {
        "trained_from": str(cutoff.date()),
        "trained_to": str(dates[-1].date()),
        "n_train_rows": int(len(train)),
        "label_horizon": cfg.label.horizon,
        "normalisation": cfg.features.standardise,
    }
