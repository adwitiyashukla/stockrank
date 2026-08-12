from __future__ import annotations

import numpy as np
import pandas as pd

from stockrank.models.base import BaseForecaster
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

LITERATURE_WEIGHTS: dict[str, float] = {
    "mom_12_1": 1.0,
    "ret_21": -1.0,
    "vol_63": -1.0,
    "beta_63": -1.0,
    "amihud_illiq": 1.0,
}


class FactorCompositeForecaster(BaseForecaster):

    name = "factor_composite"
    supports_importance = True

    def __init__(self, feature_names: list[str], **params) -> None:
        super().__init__(feature_names, **params)
        weights = params.get("weights") or LITERATURE_WEIGHTS
        self.weights_ = {k: float(v) for k, v in weights.items() if k in set(feature_names)}
        missing = sorted(set(weights) - set(self.weights_))
        if missing:
            logger.warning("Factor composite: features not available and skipped: %s", missing)
        if not self.weights_:
            raise ValueError("None of the composite's factors are present in the feature set")

    def fit(self, train: pd.DataFrame, y_col: str = "target") -> FactorCompositeForecaster:
        self.fitted_ = True
        self.n_train_rows_ = int(len(train))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        cols = list(self.weights_)
        w = np.array([self.weights_[c] for c in cols], dtype=float)
        X = np.nan_to_num(frame[cols].to_numpy(dtype=float))
        return (X @ w) / np.abs(w).sum()

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.weights_).sort_values(key=np.abs, ascending=False)
