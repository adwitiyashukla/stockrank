"""Request and response models for the scoring service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: list[str]
    runs_available: list[str]


class FeatureVector(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    features: dict[str, float] = Field(
        ..., description="Cross-sectionally normalised factor exposures, keyed by feature name"
    )


class ScoreRequest(BaseModel):
    run: str = Field("baseline", description="Artifact directory name")
    observations: list[FeatureVector]
    rank_within_request: bool = Field(
        True,
        description=(
            "Convert raw scores to cross-sectional ranks across the submitted batch. "
            "The model is trained to order a cross-section, so a single isolated score "
            "carries little meaning on its own."
        ),
    )


class ScoredItem(BaseModel):
    ticker: str
    score: float
    rank: int | None = None
    percentile: float | None = None


class ScoreResponse(BaseModel):
    run: str
    model: str
    n_scored: int
    results: list[ScoredItem]
    warnings: list[str] = []


class ScreenResponse(BaseModel):
    run: str
    model: str
    as_of: str
    n_universe: int
    longs: list[dict[str, Any]]
    shorts: list[dict[str, Any]]
    disclaimer: str


class MetricsResponse(BaseModel):
    run: str
    data: dict[str, Any]
    model_metrics: list[dict[str, Any]]
    performance: dict[str, Any]
    significance: dict[str, Any]
