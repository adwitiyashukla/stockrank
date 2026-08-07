"""Regularised linear baselines.

Every serious result needs a boring benchmark. If a gradient boosted tree with
four hundred estimators cannot beat ridge regression on the same features, the
extra machinery is decoration. Ridge is also the natural model here on statistical
grounds: cross-sectionally normalised factor exposures are highly collinear and
the true signal-to-noise ratio is tiny, which is precisely the regime where
shrinkage beats flexibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from stockrank.models.base import BaseForecaster


class RidgeForecaster(BaseForecaster):
    name = "ridge"
    supports_importance = True

    def fit(self, train: pd.DataFrame, y_col: str = "target") -> RidgeForecaster:
        X = train[self.feature_names].to_numpy(dtype=np.float64)
        y = train[y_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model_ = Ridge(alpha=float(self.params.get("alpha", 10.0)), fit_intercept=True)
        self.model_.fit(X[ok], y[ok])
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = np.nan_to_num(frame[self.feature_names].to_numpy(dtype=np.float64))
        return self.model_.predict(X)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(self.model_.coef_, index=self.feature_names).sort_values(
            key=np.abs, ascending=False
        )


class ElasticNetForecaster(BaseForecaster):
    name = "elasticnet"
    supports_importance = True

    def fit(self, train: pd.DataFrame, y_col: str = "target") -> ElasticNetForecaster:
        X = train[self.feature_names].to_numpy(dtype=np.float64)
        y = train[y_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        # Targets are ~1e-2 in magnitude; scaling up keeps the L1 penalty in a
        # sensible range without having to hand-tune alpha per horizon.
        self.y_scale_ = float(np.std(y[ok])) or 1.0
        self.model_ = ElasticNet(
            alpha=float(self.params.get("alpha", 0.001)),
            l1_ratio=float(self.params.get("l1_ratio", 0.5)),
            max_iter=5000,
            selection="random",
            random_state=0,
        )
        self.model_.fit(X[ok], y[ok] / self.y_scale_)
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = np.nan_to_num(frame[self.feature_names].to_numpy(dtype=np.float64))
        return self.model_.predict(X) * self.y_scale_

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(self.model_.coef_, index=self.feature_names).sort_values(
            key=np.abs, ascending=False
        )
