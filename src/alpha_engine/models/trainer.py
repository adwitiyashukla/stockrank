"""Walk-forward training loop: the spine of the whole project.

For each purged fold, every enabled model is fitted on the training window and
scored on the untouched test window. Concatenating the test-window predictions
gives one continuous out-of-sample track record per model, which is what the
backtest and every statistic downstream consume. No model ever sees a single row
from its own evaluation period.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alpha_engine.config import Config
from alpha_engine.evaluation.metrics import prediction_metrics
from alpha_engine.features.pipeline import FeatureSet
from alpha_engine.models.ensemble import rank_average
from alpha_engine.models.registry import SEQUENCE, build_model
from alpha_engine.models.sequence import PanelTensor, SequenceForecaster
from alpha_engine.utils.logging import get_logger
from alpha_engine.validation.splitters import Fold, PurgedWalkForward

logger = get_logger(__name__)

ID_COLS = [
    "date", "ticker", "sector", "target", "fwd_return",
    "close", "dollar_volume", "beta_raw", "vol_raw",
]


@dataclass
class TrainingResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    importances: dict[str, pd.Series] = field(default_factory=dict)
    fold_specs: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def model_names(self) -> list[str]:
        return [c[5:] for c in self.predictions.columns if c.startswith("pred_")]


def _fold_frames(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.iloc[fold.train_idx], frame.iloc[fold.test_idx]


def walk_forward_train(fs: FeatureSet, cfg: Config) -> TrainingResult:
    frame = fs.frame.reset_index(drop=True)
    feats = fs.feature_names

    cv = PurgedWalkForward(
        n_splits=cfg.validation.n_splits,
        train_window_days=cfg.validation.train_window_days,
        test_window_days=cfg.validation.test_window_days,
        label_horizon=cfg.label.horizon,
        embargo_days=cfg.validation.embargo_days,
        expanding=cfg.validation.expanding,
    )
    folds = cv.get_folds(frame["date"])

    enabled = [m for m in cfg.models.enabled if m != "ensemble"]
    wants_sequence = any(m in SEQUENCE for m in enabled)

    shared_tensor = None
    if wants_sequence:
        t0 = time.time()
        shared_tensor = PanelTensor(frame, feats)
        logger.info(
            "Panel tensor: shape=%s memory=%.0f MB (built in %.1fs)",
            shared_tensor.cube.shape, shared_tensor.nbytes / 1e6, time.time() - t0,
        )

    out_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    importances: dict[str, list[pd.Series]] = {}
    timings: dict[str, float] = {}

    for fold in folds:
        train, test = _fold_frames(frame, fold)
        logger.info(
            "Fold %d | train %s..%s (%d rows) | test %s..%s (%d rows)",
            fold.index, fold.train_start.date(), fold.train_end.date(), len(train),
            fold.test_start.date(), fold.test_end.date(), len(test),
        )

        block = test[[c for c in ID_COLS if c in test.columns]].copy()
        block["fold"] = fold.index

        for name in enabled:
            t0 = time.time()
            try:
                model = build_model(name, feats, cfg)
                if isinstance(model, SequenceForecaster) and shared_tensor is not None:
                    model.attach_tensor(shared_tensor)
                model.fit(train)
                preds = model.predict(test)
            except Exception as exc:  # noqa: BLE001 - one broken model must not kill the run
                logger.exception("Model %s failed on fold %d: %s", name, fold.index, exc)
                continue
            dt = time.time() - t0
            timings[name] = timings.get(name, 0.0) + dt

            block[f"pred_{name}"] = preds
            m = prediction_metrics(
                preds, test["target"].to_numpy(), test["fwd_return"].to_numpy(),
                test["date"], cfg.label.horizon,
            )
            fold_rows.append({"fold": fold.index, "model": name, "fit_seconds": round(dt, 1), **m})
            logger.info(
                "  %-12s IC=%+.4f  ICIR=%+.2f  t(NW)=%+.2f  q5-q1=%+.4f  [%.1fs]",
                name, m.get("rank_mean_ic", np.nan), m.get("rank_icir", np.nan),
                m.get("rank_t_stat_nw", np.nan), m.get("q5_minus_q1", np.nan), dt,
            )

            imp = model.feature_importance()
            if imp is not None:
                importances.setdefault(name, []).append(imp)

        out_parts.append(block)

    predictions = pd.concat(out_parts, ignore_index=True).sort_values(["date", "ticker"])
    predictions = predictions.reset_index(drop=True)

    # Ensemble is formed on out-of-sample predictions only.
    if "ensemble" in cfg.models.enabled:
        members = cfg.models.ensemble.get("members") or enabled
        cols = {m: predictions[f"pred_{m}"].to_numpy() for m in members if f"pred_{m}" in predictions}
        if len(cols) >= 2:
            predictions["pred_ensemble"] = rank_average(cols, predictions["date"])
            logger.info("Ensemble built by rank averaging over %s", sorted(cols))
        else:
            logger.warning("Not enough members for an ensemble (have %s)", sorted(cols))

    rows = []
    for name in [c[5:] for c in predictions.columns if c.startswith("pred_")]:
        m = prediction_metrics(
            predictions[f"pred_{name}"].to_numpy(), predictions["target"].to_numpy(),
            predictions["fwd_return"].to_numpy(), predictions["date"], cfg.label.horizon,
        )
        rows.append({"model": name, "fit_seconds": round(timings.get(name, 0.0), 1), **m})
    overall = pd.DataFrame(rows).sort_values("rank_mean_ic", ascending=False).reset_index(drop=True)

    mean_imp = {
        k: pd.concat(v, axis=1).mean(axis=1).sort_values(ascending=False)
        for k, v in importances.items()
    }

    logger.info("\n%s", overall[["model", "rank_mean_ic", "rank_icir", "rank_t_stat_nw"]].to_string(index=False))

    return TrainingResult(
        predictions=predictions,
        fold_metrics=pd.DataFrame(fold_rows),
        overall_metrics=overall,
        importances=mean_imp,
        fold_specs=[f.describe() for f in folds],
        timings=timings,
    )
