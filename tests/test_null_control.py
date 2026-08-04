"""The end-to-end leakage control.

The simulator can plant zero alpha. In that world the true predictable component
of the cross-section is exactly zero, so any model that reports a reliably
non-zero out-of-sample information coefficient has found a bug, not a signal.
This is the one test that exercises data loading, feature construction, labelling,
splitting and training together, which is precisely where leaks tend to hide.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_engine.data.loader import load_market_data
from alpha_engine.evaluation.metrics import daily_ic, ic_summary
from alpha_engine.features.pipeline import build_feature_set
from alpha_engine.models.trainer import walk_forward_train


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
        # With no planted alpha the IC should be statistically indistinguishable
        # from zero. A |t| above 4 under Newey-White errors indicates leakage.
        assert abs(s["t_stat_nw"]) < 4.0, (
            f"{model} found signal in null data: IC={s['mean_ic']:+.4f} t={s['t_stat_nw']:+.2f}. "
            "This means information is leaking from the future."
        )
        assert abs(s["mean_ic"]) < 0.05


@pytest.mark.slow
def test_planted_alpha_is_recovered(small_config):
    """The mirror image: with alpha planted, the pipeline must actually find it.

    A pipeline that reports zero everywhere would pass the null test trivially,
    so this test guards against the opposite failure.
    """
    cfg = small_config.model_copy(deep=True)
    cfg.run.name = "pytest_signal"
    cfg.simulator.null_alpha = False
    cfg.simulator.momentum_strength = 0.05  # deliberately strong so the test is stable
    cfg.models.enabled = ["ridge"]

    md = load_market_data(cfg, cache=False)
    fs = build_feature_set(md, cfg)
    result = walk_forward_train(fs, cfg)

    preds = result.predictions
    ic = daily_ic(preds["pred_ridge"], preds["target"], preds["date"])
    assert ic.mean() > 0.01, f"planted alpha was not recovered (IC={ic.mean():+.4f})"
    assert np.isfinite(ic).all()
