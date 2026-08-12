from __future__ import annotations

from stockrank.config import Config
from stockrank.models.base import BaseForecaster
from stockrank.models.factor_composite import FactorCompositeForecaster
from stockrank.models.gbm import LightGBMForecaster
from stockrank.models.linear import ElasticNetForecaster, RidgeForecaster
from stockrank.models.sequence import SequenceForecaster

TABULAR = {"ridge", "elasticnet", "lightgbm", "factor_composite"}
SEQUENCE = {"gru", "lstm", "transformer"}
META = {"ensemble"}


def build_model(name: str, feature_names: list[str], cfg: Config) -> BaseForecaster:
    if name == "factor_composite":
        return FactorCompositeForecaster(feature_names, **cfg.models.factor_composite)
    if name == "ridge":
        return RidgeForecaster(feature_names, **cfg.models.ridge)
    if name == "elasticnet":
        return ElasticNetForecaster(feature_names, **cfg.models.elasticnet)
    if name == "lightgbm":
        return LightGBMForecaster(feature_names, **cfg.models.lightgbm)
    if name in SEQUENCE:
        params = dict(cfg.models.gru)
        params.setdefault("seed", cfg.run.seed)
        return SequenceForecaster(feature_names, kind=name, **params)
    raise ValueError(f"Unknown model: {name}")


def available_models() -> list[str]:
    return sorted(TABULAR | SEQUENCE | META)
