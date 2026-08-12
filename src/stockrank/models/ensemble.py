from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from stockrank.utils.logging import get_logger

logger = get_logger(__name__)


def cross_sectional_rank(scores: pd.Series, dates: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"d": dates.to_numpy(), "s": scores.to_numpy()})
    return tmp.groupby("d", observed=True)["s"].rank(pct=True).to_numpy() - 0.5


def rank_average(
    predictions: dict[str, np.ndarray], dates: pd.Series, members: list[str] | None = None
) -> np.ndarray:
    members = members or list(predictions.keys())
    available = [m for m in members if m in predictions]
    if not available:
        raise ValueError("No ensemble members available")
    if len(available) < len(members):
        logger.warning("Ensemble members missing: %s", sorted(set(members) - set(available)))
    stack = np.column_stack(
        [cross_sectional_rank(pd.Series(predictions[m]), dates) for m in available]
    )
    return stack.mean(axis=1)


class RidgeStack:

    def __init__(self, members: list[str]) -> None:
        self.members = members
        self.model_: RidgeCV | None = None

    def fit(self, oof: pd.DataFrame, y: np.ndarray) -> RidgeStack:
        X = oof[self.members].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model_ = RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.model_.fit(X[ok], y[ok])
        logger.info(
            "Ridge stack weights: %s",
            dict(zip(self.members, np.round(self.model_.coef_, 4), strict=False)),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("RidgeStack is not fitted")
        return self.model_.predict(np.nan_to_num(frame[self.members].to_numpy(dtype=np.float64)))
