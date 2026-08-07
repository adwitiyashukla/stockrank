"""Common interface every forecaster in the zoo implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Minimal contract: fit on a training frame, score any frame.

    Models receive the *frame* rather than a bare matrix because sequence models
    need the date and ticker columns to reconstruct histories, while tabular
    models only touch the feature columns.
    """

    name: str = "base"
    supports_importance: bool = False

    def __init__(self, feature_names: list[str], **params: Any) -> None:
        self.feature_names = list(feature_names)
        self.params = params
        self.fitted_ = False

    @abstractmethod
    def fit(self, train: pd.DataFrame, y_col: str = "target") -> BaseForecaster: ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> pd.Series | None:
        return None

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} has not been fitted")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r}, n_features={len(self.feature_names)})"
