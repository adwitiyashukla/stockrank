"""Name to constructor mapping so configs stay declarative."""

from __future__ import annotations

from alpha_engine.config import Config
from alpha_engine.models.base import BaseForecaster
from alpha_engine.models.factor_composite import FactorCompositeForecaster
from alpha_engine.models.gbm import LightGBMForecaster
from alpha_engine.models.linear import ElasticNetForecaster, RidgeForecaster
from alpha_engine.models.sequence import SequenceForecaster

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
