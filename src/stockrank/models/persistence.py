"""Saving and loading a production forecaster.

The walk-forward loop fits one model per fold, which is right for evaluation and
useless for deployment. For serving, a single model is refitted on the most
recent training window and persisted alongside the exact feature list it expects,
so the API cannot silently score a differently ordered matrix.

Why this does not simply pickle the wrapper
-------------------------------------------
It used to. Pickling a custom class stores the fully qualified module path, so
the artifact silently breaks the moment the package is renamed, a module is
moved, or the library version shifts. That is a real failure mode: renaming this
package from ``alpha_engine`` to ``stockrank`` invalidated every previously saved
model, and the only symptom was a 500 from the scoring endpoint.

Gradient boosted models are therefore stored in **LightGBM's own text format**,
which is a stable, self-describing, human-readable serialisation with no
dependency on this codebase at all. The wrapper is reconstructed at load time
from the feature list and parameters recorded in the sidecar JSON. Other model
types still fall back to joblib, and the loader tells you plainly when an
artifact was written by an incompatible layout instead of raising a pickle error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stockrank.utils.io import ensure_dir
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

FORMAT_LIGHTGBM_TEXT = "lightgbm_text_v1"
FORMAT_JOBLIB = "joblib_v1"


def save_model(
    model, feature_names: list[str], path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Persist ``model`` plus its feature contract. Returns the artifact path."""
    p = Path(path)
    ensure_dir(p.parent)
    meta: dict[str, Any] = {
        "model_name": getattr(model, "name", model.__class__.__name__),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        **(metadata or {}),
    }

    booster = getattr(getattr(model, "model_", None), "booster_", None)
    if booster is not None:
        # Native text format: portable across package renames and library versions.
        txt = p.with_suffix(".txt")
        booster.save_model(str(txt))
        meta["format"] = FORMAT_LIGHTGBM_TEXT
        meta["artifact"] = txt.name
        meta["params"] = {
            k: v for k, v in getattr(model, "params", {}).items()
            if isinstance(v, (int, float, str, bool))
        }
        if p.exists():
            p.unlink()  # remove any stale pickle from an earlier layout
    else:
        import joblib

        joblib.dump({"model": model, "feature_names": list(feature_names)}, p)
        meta["format"] = FORMAT_JOBLIB
        meta["artifact"] = p.name

    p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved production model (%s) to %s", meta["format"], meta["artifact"])
    return p


class _BoosterForecaster:
    """Minimal scorer wrapping a raw LightGBM booster loaded from text."""

    name = "lightgbm"

    def __init__(self, booster, feature_names: list[str]) -> None:
        self.booster = booster
        self.feature_names = list(feature_names)

    def predict(self, frame: pd.DataFrame):
        X = frame[self.feature_names].to_numpy(dtype="float32")
        return self.booster.predict(X)

    def feature_importance(self) -> pd.Series:
        imp = pd.Series(
            self.booster.feature_importance(importance_type="gain"), index=self.feature_names
        )
        total = imp.sum()
        return (imp / total if total > 0 else imp).sort_values(ascending=False)


def load_model(path: str | Path) -> tuple[Any, list[str]]:
    """Load a persisted model. Accepts both the text and the legacy joblib layout."""
    p = Path(path)
    meta_path = p.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    feature_names = meta.get("feature_names", [])

    txt = p.with_suffix(".txt")
    if meta.get("format") == FORMAT_LIGHTGBM_TEXT or txt.exists():
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(txt))
        return _BoosterForecaster(booster, feature_names), feature_names

    if not p.exists():
        raise FileNotFoundError(f"No model artifact at {p} or {txt}")

    import joblib

    try:
        payload = joblib.load(p)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            f"{p.name} was serialised by an incompatible package layout ({exc}). "
            "Regenerate it with: python -m stockrank.cli rebacktest --config configs/default.yaml"
        ) from exc
    return payload["model"], payload["feature_names"]


def fit_production_model(fs, cfg, model_name: str = "lightgbm", train_days: int | None = None):
    """Refit a single model on the most recent window for serving."""
    from stockrank.models.registry import build_model

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
