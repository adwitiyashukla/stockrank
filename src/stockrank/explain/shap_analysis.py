"""Model explanation and feature stability.

Two questions matter here and they are different.

*What is the model using?* SHAP decomposes each individual prediction into
additive per-feature contributions, so instead of "volatility is important" you
get "for this name on this date, low realised volatility added 12 basis points to
the forecast". For a quantitative strategy this is what makes a position
defensible to a risk committee.

*Is it using the same things over time?* A model whose top features are
completely reshuffled between folds has found period-specific noise, not
structure. Feature-importance rank correlation across folds is a cheap and
surprisingly informative stability check, and it is reported alongside the
attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stockrank.utils.logging import get_logger

logger = get_logger(__name__)


def shap_summary(
    model, frame: pd.DataFrame, feature_names: list[str], max_samples: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, plus the signed mean direction."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install shap: pip install -e '.[explain]'") from exc

    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(frame), size=max_samples, replace=False)
        if len(frame) > max_samples else np.arange(len(frame))
    )
    X = frame.iloc[idx][feature_names]

    booster = getattr(model, "model_", model)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "shap_std": values.std(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    total = out["mean_abs_shap"].sum()
    out["share"] = out["mean_abs_shap"] / total if total > 0 else np.nan
    logger.info("SHAP computed on %d samples, top feature: %s", len(X), out.iloc[0]["feature"])
    return out.reset_index(drop=True)


def shap_values_frame(
    model, frame: pd.DataFrame, feature_names: list[str], max_samples: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-observation SHAP contributions, keyed by date and ticker."""
    import shap

    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(frame), size=max_samples, replace=False)
        if len(frame) > max_samples else np.arange(len(frame))
    )
    sub = frame.iloc[idx]
    booster = getattr(model, "model_", model)
    values = shap.TreeExplainer(booster).shap_values(sub[feature_names])
    if isinstance(values, list):
        values = values[0]
    out = pd.DataFrame(values, columns=feature_names)
    out.insert(0, "ticker", sub["ticker"].to_numpy())
    out.insert(0, "date", sub["date"].to_numpy())
    return out


def importance_stability(fold_importances: list[pd.Series]) -> dict[str, float]:
    """Average pairwise Spearman correlation of feature-importance rankings."""
    if len(fold_importances) < 2:
        return {"mean_rank_correlation": np.nan, "n_folds": len(fold_importances)}

    aligned = pd.concat(fold_importances, axis=1).fillna(0.0)
    corrs = []
    for i in range(aligned.shape[1]):
        for j in range(i + 1, aligned.shape[1]):
            c = stats.spearmanr(aligned.iloc[:, i], aligned.iloc[:, j]).statistic
            if np.isfinite(c):
                corrs.append(float(c))

    top5 = [set(s.abs().nlargest(5).index) for s in fold_importances]
    overlap = (
        float(np.mean([len(top5[i] & top5[j]) / 5 for i in range(len(top5)) for j in range(i + 1, len(top5))]))
        if len(top5) > 1 else np.nan
    )
    return {
        "mean_rank_correlation": float(np.mean(corrs)) if corrs else np.nan,
        "top5_overlap": overlap,
        "n_folds": len(fold_importances),
    }
