from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RunConfig(BaseModel):
    name: str = "baseline"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


class DataConfig(BaseModel):

    source: Literal["yfinance", "synthetic", "stooq"] = "yfinance"
    universe: Literal["sp500_pit", "sp500", "file"] = "sp500_pit"
    universe_file: str | None = None
    start: str = "2010-01-01"
    end: str = "2025-06-30"
    max_assets: int = 250
    n_assets: int = 150
    cache_dir: str = "data/cache"
    min_price: float = 5.0
    min_history_days: int = 500
    min_names_per_date: int = 30
    benchmark: str = "SPY"
    use_fama_french: bool = True


class SimulatorConfig(BaseModel):
    n_sectors: int = 8
    ann_market_vol: float = 0.16
    ann_idio_vol: float = 0.28
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    student_t_df: int = 5
    momentum_strength: float = 0.020
    reversal_strength: float = 0.015
    lowvol_strength: float = 0.010
    null_alpha: bool = False

    @model_validator(mode="after")
    def _check_garch_stationary(self) -> SimulatorConfig:
        if self.garch_alpha + self.garch_beta >= 1.0:
            raise ValueError("garch_alpha + garch_beta must be < 1 for a stationary process")
        return self


class FeatureConfig(BaseModel):
    return_windows: list[int] = Field(default_factory=lambda: [1, 5, 10, 21, 63, 126, 252])
    vol_windows: list[int] = Field(default_factory=lambda: [10, 21, 63])
    ma_windows: list[int] = Field(default_factory=lambda: [10, 21, 50, 200])
    volume_windows: list[int] = Field(default_factory=lambda: [5, 21, 63])
    winsorize_q: float = 0.01
    cross_sectional_rank: bool = True
    standardise: Literal["zscore", "rank", "none"] = "zscore"


class TripleBarrierConfig(BaseModel):
    upper_sigma: float = 2.0
    lower_sigma: float = 2.0
    max_holding_days: int = 10


class LabelConfig(BaseModel):
    type: Literal["forward_excess_return", "forward_return", "triple_barrier"] = (
        "forward_excess_return"
    )
    horizon: int = 5
    neutralise_market: bool = False
    scale_by_volatility: bool = False
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)


class ValidationConfig(BaseModel):
    scheme: Literal["purged_walk_forward", "purged_kfold"] = "purged_walk_forward"
    n_splits: int = 6
    embargo_days: int = 10
    train_window_days: int = 1260
    test_window_days: int = 252
    expanding: bool = False


class ModelsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["ridge", "lightgbm"])
    factor_composite: dict[str, Any] = Field(default_factory=dict)
    ridge: dict[str, Any] = Field(default_factory=dict)
    elasticnet: dict[str, Any] = Field(default_factory=dict)
    lightgbm: dict[str, Any] = Field(default_factory=dict)
    gru: dict[str, Any] = Field(default_factory=dict)
    ensemble: dict[str, Any] = Field(default_factory=dict)


class VolatilityConfig(BaseModel):
    model: Literal["garch", "har", "ewma"] = "garch"
    garch_p: int = 1
    garch_q: int = 1
    refit_every_days: int = 63
    max_assets_fitted: int = 60


class MeanVarianceConfig(BaseModel):
    risk_aversion: float = 5.0
    shrinkage: Literal["ledoit_wolf", "none"] = "ledoit_wolf"
    cov_lookback_days: int = 252


class PortfolioConfig(BaseModel):
    method: Literal["rank_long_short", "mean_variance", "risk_parity"] = "rank_long_short"
    n_long: int = 25
    n_short: int = 25
    gross_leverage: float = 2.0
    max_weight: float = 0.06
    dollar_neutral: bool = True
    beta_neutral: bool = True
    sector_neutral: bool = False
    vol_target_annual: float = 0.10
    vol_lookback_days: int = 63
    max_vol_scalar: float = 2.0
    rebalance_days: int = 5
    mean_variance: MeanVarianceConfig = Field(default_factory=MeanVarianceConfig)


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 50.0
    turnover_cap_daily: float = 0.25


class EvaluationConfig(BaseModel):
    n_trials_for_deflated_sharpe: int = 40
    bootstrap_samples: int = 1000
    pbo_splits: int = 8


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)

    def apply_smoke_overrides(self) -> Config:
        cfg = self.model_copy(deep=True)
        cfg.run.name = f"{cfg.run.name}_smoke"
        cfg.data.n_assets = min(cfg.data.n_assets, 30)
        cfg.data.max_assets = min(cfg.data.max_assets, 30)
        cfg.data.start = "2019-01-01"
        cfg.validation.n_splits = 2
        cfg.validation.train_window_days = 400
        cfg.validation.test_window_days = 126
        cfg.models.lightgbm["n_estimators"] = 60
        cfg.models.gru["epochs"] = 1
        cfg.models.gru["max_train_sequences"] = 5000
        cfg.volatility.max_assets_fitted = 5
        cfg.evaluation.bootstrap_samples = 100
        cfg.portfolio.n_long = 6
        cfg.portfolio.n_short = 6
        return cfg


def load_config(path: str | Path) -> Config:
    return Config.from_yaml(path)
