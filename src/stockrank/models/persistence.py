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
        txt = p.with_suffix(".txt")
        booster.save_model(str(txt))
        meta["format"] = FORMAT_LIGHTGBM_TEXT
        meta["artifact"] = txt.name
        meta["params"] = {
            k: v for k, v in getattr(model, "params", {}).items()
            if isinstance(v, (int, float, str, bool))
        }
        if p.exists():
            p.unlink()
    else:
        import joblib

        joblib.dump({"model": model, "feature_names": list(feature_names)}, p)
        meta["format"] = FORMAT_JOBLIB
        meta["artifact"] = p.name

    p.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved production model (%s) to %s", meta["format"], meta["artifact"])
    return p


class _BoosterForecaster:

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
