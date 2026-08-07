"""Gradient boosted trees.

Two deliberate choices worth defending:

* **Huber objective.** Cross-sectional forward returns have fat tails; a squared
  error loss lets a handful of earnings-gap observations dominate the fit.
* **Large ``min_child_samples``.** With a signal-to-noise ratio around 2%, small
  leaves are noise-fitting machines. Forcing hundreds of observations per leaf is
  the single most effective regulariser in this setting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stockrank.models.base import BaseForecaster
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMForecaster(BaseForecaster):
    name = "lightgbm"
    supports_importance = True

    DEFAULTS = {
        "objective": "huber",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "n_jobs": 4,
        "verbose": -1,
    }

    def fit(self, train: pd.DataFrame, y_col: str = "target") -> LightGBMForecaster:
        import lightgbm as lgb

        params = {**self.DEFAULTS, **self.params}
        params.setdefault("random_state", 0)

        X = train[self.feature_names].to_numpy(dtype=np.float32)
        y = train[y_col].to_numpy(dtype=np.float32)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]

        # Hold out the most recent slice of the training window for early
        # stopping. It is still strictly in-sample relative to the test fold.
        n = len(y)
        cut = int(n * 0.85)
        self.model_ = lgb.LGBMRegressor(**params)
        if cut > 1000 and n - cut > 200:
            cbs = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)]
            # LightGBM renamed the validation arguments across 4.x. Try the current
            # spelling first and fall back, so the repository works on any recent
            # version without pinning users to one.
            attempts = (
                {"eval_X": X[cut:], "eval_y": y[cut:]},
                {"eval_X": [X[cut:]], "eval_y": [y[cut:]]},
                {"eval_set": [(X[cut:], y[cut:])]},
            )
            for kwargs in attempts:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model_.fit(
                            X[:cut], y[:cut], eval_metric="l2", callbacks=cbs, **kwargs
                        )
                    break
                except TypeError:
                    continue
            else:  # pragma: no cover - every signature rejected
                self.model_.fit(X, y)
            self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        else:
            self.model_.fit(X, y)
            self.best_iteration_ = None
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = frame[self.feature_names].to_numpy(dtype=np.float32)
        return self.model_.predict(X)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        imp = pd.Series(
            self.model_.booster_.feature_importance(importance_type="gain"),
            index=self.feature_names,
        )
        total = imp.sum()
        return (imp / total if total > 0 else imp).sort_values(ascending=False)
