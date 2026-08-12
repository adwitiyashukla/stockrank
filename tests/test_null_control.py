from __future__ import annotations

import numpy as np
import pytest

from stockrank.data.loader import load_market_data
from stockrank.evaluation.metrics import daily_ic, ic_summary
from stockrank.features.pipeline import build_feature_set
from stockrank.models.trainer import walk_forward_train


@pytest.mark.slow
def test_null_alpha_gives_zero_information_coefficient(small_config):
    cfg = small_config.model_copy(deep=True)
    cfg.run.name = "pytest_null"
    cfg.simulator.null_alpha = True
    cfg.models.enabled = ["ridge", "lightgbm"]

    md = load_market_data(cfg, cache=False)
    fs = build_feature_set(md, cfg)
    result = walk_forward_train(fs, cfg)

    for model in result.model_names:
        preds = result.predictions
        ic = daily_ic(preds[f"pred_{model}"], preds["target"], preds["date"])
        s = ic_summary(ic, cfg.label.horizon)
        assert abs(s["t_stat_nw"]) < 4.0, (
            f"{model} found signal in null data: IC={s['mean_ic']:+.4f} t={s['t_stat_nw']:+.2f}. "
            "This means information is leaking from the future."
        )
        assert abs(s["mean_ic"]) < 0.05


@pytest.mark.slow
def test_planted_alpha_is_recovered(small_config):
    cfg = small_config.model_copy(deep=True)
    cfg.run.name = "pytest_signal"
    cfg.simulator.null_alpha = False
    cfg.simulator.momentum_strength = 0.05
    cfg.models.enabled = ["ridge"]

    md = load_market_data(cfg, cache=False)
    fs = build_feature_set(md, cfg)
    result = walk_forward_train(fs, cfg)

    preds = result.predictions
    ic = daily_ic(preds["pred_ridge"], preds["target"], preds["date"])
    assert ic.mean() > 0.01, f"planted alpha was not recovered (IC={ic.mean():+.4f})"
    assert np.isfinite(ic).all()
