"""A zero-parameter benchmark built from documented anomalies.

Why this belongs in the model zoo
---------------------------------
Machine learning results in finance are usually reported against nothing, or
against a random baseline, which makes them impossible to judge. The right
comparison is a simple composite of anomalies that were documented in the
academic literature decades ago, because that is what a model has to beat to
justify its complexity, its compute and its operational risk.

This forecaster **fits nothing**. It has no parameters estimated from data, so it
cannot overfit by construction, and its out-of-sample performance is an unbiased
estimate of its true performance. If the gradient boosted trees and the neural
network cannot beat it, that is the finding, and it should be reported rather
than buried.

The composite is an equal-weighted sum of already cross-sectionally standardised
factors:

* ``+ mom_12_1``      medium-term momentum, skipping the last month
  (Jegadeesh and Titman 1993)
* ``- ret_21``        one-month reversal (Jegadeesh 1990)
* ``- vol_63``        the low-volatility anomaly (Frazzini and Pedersen 2014)
* ``- beta_63``       betting against beta, same reference
* ``+ amihud_illiq``  the illiquidity premium (Amihud 2002)

Signs follow the published direction of each effect. Nothing here was tuned on
the sample.
"""

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
    """Equal-weight composite of classic anomalies. No fitting, no parameters."""

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
        """No-op. Recorded explicitly so the walk-forward loop treats it like any other model."""
        self.fitted_ = True
        self.n_train_rows_ = int(len(train))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        cols = list(self.weights_)
        w = np.array([self.weights_[c] for c in cols], dtype=float)
        X = np.nan_to_num(frame[cols].to_numpy(dtype=float))
        # Features arrive cross-sectionally z-scored, so a plain weighted sum is
        # already a comparable composite score.
        return (X @ w) / np.abs(w).sum()

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.weights_).sort_values(key=np.abs, ascending=False)
