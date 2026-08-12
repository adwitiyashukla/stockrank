from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockrank.backtest.engine import BacktestResult, run_backtest
from stockrank.config import Config
from stockrank.data.loader import load_market_data
from stockrank.evaluation.attribution import factor_regression, turnover_capacity_analysis
from stockrank.evaluation.metrics import quantile_spread
from stockrank.evaluation.performance import monthly_return_table, performance_stats
from stockrank.evaluation.significance import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    randomisation_test,
    stationary_bootstrap_sharpe,
)
from stockrank.features.pipeline import build_feature_set
from stockrank.models.trainer import walk_forward_train
from stockrank.utils.io import ensure_dir, read_json, write_json
from stockrank.utils.logging import get_logger
from stockrank.utils.seeds import set_global_seed

logger = get_logger(__name__)


class ExperimentResult:

    def __init__(self, cfg: Config, artifact_dir: Path) -> None:
        self.cfg = cfg
        self.dir = artifact_dir
        self.data_summary: dict[str, Any] = {}
        self.feature_summary: dict[str, Any] = {}
        self.training = None
        self.backtests: dict[str, BacktestResult] = {}
        self.performance: dict[str, dict] = {}
        self.significance: dict[str, Any] = {}
        self.attribution: dict[str, Any] = {}
        self.volatility: pd.DataFrame | None = None
        self.explanation: dict[str, Any] = {}
        self.timings: dict[str, float] = {}


def _benchmark_series(market: pd.DataFrame, index: pd.Index) -> pd.Series:
    m = market.copy()
    m["date"] = pd.to_datetime(m["date"])
    return m.set_index("date")["mkt_return"].reindex(index).fillna(0.0)


def rerun_from_predictions(cfg: Config) -> ExperimentResult:
    art = Path(cfg.run.artifacts_dir) / cfg.run.name
    pred_path = art / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No predictions at {pred_path}. Run the full pipeline once first."
        )

    set_global_seed(cfg.run.seed)
    res = ExperimentResult(cfg, art)
    res.data_summary = read_json(art / "data_summary.json") if (art / "data_summary.json").exists() else {}
    res.feature_summary = read_json(art / "feature_summary.json") if (art / "feature_summary.json").exists() else {}

    preds = pd.read_parquet(pred_path)
    preds["date"] = pd.to_datetime(preds["date"])
    from stockrank.models.trainer import TrainingResult

    imp = {}
    if (art / "feature_importance.csv").exists():
        idf = pd.read_csv(art / "feature_importance.csv", index_col=0)
        imp = {c: idf[c].dropna() for c in idf.columns}
    res.training = TrainingResult(
        predictions=preds,
        fold_metrics=pd.read_csv(art / "fold_metrics.csv"),
        overall_metrics=pd.read_csv(art / "model_metrics.csv"),
        importances=imp,
    )
    logger.info("Reusing %d cached predictions from %s", len(preds), pred_path)

    if (art / "timings.json").exists():
        previous = read_json(art / "timings.json")
        for k in ("data", "features", "training"):
            if k in previous:
                res.timings[k] = previous[k]

    md = load_market_data(cfg)

    fresh = md.summary()
    for k, v in fresh.items():
        if k not in res.data_summary or k == "download" or not res.data_summary.get(k):
            res.data_summary[k] = v
    write_json(res.data_summary, art / "data_summary.json")

    try:
        fs = build_feature_set(md, cfg)
    except Exception as exc:
        logger.warning("Could not rebuild features (%s); production model left unchanged", exc)
        fs = None

    _evaluate_and_persist(res, cfg, md, art, fs=fs)
    res.timings["total"] = sum(v for k, v in res.timings.items() if k != "total")
    write_json(res.timings, art / "timings.json")
    logger.info("Re-evaluation complete in %.1fs", res.timings.get("evaluation", 0.0))
    return res


def run_experiment(cfg: Config, smoke: bool = False) -> ExperimentResult:
    if smoke:
        cfg = cfg.apply_smoke_overrides()
    set_global_seed(cfg.run.seed)

    art = ensure_dir(Path(cfg.run.artifacts_dir) / cfg.run.name)
    res = ExperimentResult(cfg, art)
    cfg.to_yaml(art / "config.yaml")
    t_start = time.time()

    t0 = time.time()
    md = load_market_data(cfg)
    res.data_summary = md.summary()
    res.timings["data"] = time.time() - t0
    write_json(res.data_summary, art / "data_summary.json")

    t0 = time.time()
    fs = build_feature_set(md, cfg)
    res.feature_summary = fs.summary()
    res.timings["features"] = time.time() - t0
    write_json(res.feature_summary, art / "feature_summary.json")

    t0 = time.time()
    training = walk_forward_train(fs, cfg)
    res.training = training
    res.timings["training"] = time.time() - t0

    training.predictions.to_parquet(art / "predictions.parquet", index=False)
    training.fold_metrics.to_csv(art / "fold_metrics.csv", index=False)
    training.overall_metrics.to_csv(art / "model_metrics.csv", index=False)
    write_json(training.fold_specs, art / "folds.json")
    if training.importances:
        pd.DataFrame(training.importances).to_csv(art / "feature_importance.csv")

    _evaluate_and_persist(res, cfg, md, art, fs=fs)

    res.timings["total"] = time.time() - t_start
    write_json(res.timings, art / "timings.json")
    logger.info(
        "Run '%s' complete in %.1fs -> %s", cfg.run.name, res.timings["total"], art
    )
    return res


def _evaluate_and_persist(
    res: ExperimentResult, cfg: Config, md, art: Path, fs=None
) -> None:
    training = res.training
    preds = training.predictions
    t0 = time.time()
    sectors = (
        preds.drop_duplicates("ticker").set_index("ticker")["sector"]
        if "sector" in preds.columns else None
    )
    daily_returns = md.close_matrix().pct_change()

    for model in training.model_names:
        try:
            bt = run_backtest(
                preds, f"pred_{model}", cfg.portfolio, cfg.backtest,
                sectors=sectors, daily_returns=daily_returns,
                label_horizon=cfg.label.horizon,
            )
        except Exception as exc:
            logger.exception("Backtest failed for %s: %s", model, exc)
            continue
        res.backtests[model] = bt
        bt.to_frame().to_parquet(art / f"backtest_{model}.parquet")

    bench = _benchmark_series(md.market, next(iter(res.backtests.values())).returns.index) if res.backtests else pd.Series(dtype=float)
    res.timings["backtest"] = time.time() - t0

    t0 = time.time()
    for model, bt in res.backtests.items():
        res.performance[model] = performance_stats(bt.returns, benchmark=bench)
        res.performance[model]["mean_turnover_per_rebalance"] = float(bt.turnover.replace(0, np.nan).mean())
        res.performance[model]["gross_ann_return"] = float(bt.gross_returns.mean() * 252)
        res.performance[model]["cost_drag_annual"] = float(bt.costs.mean() * 252)
    res.performance["benchmark_buy_hold"] = performance_stats(bench)
    write_json(res.performance, art / "performance.json")

    if res.backtests:
        best = max(
            res.performance.items(),
            key=lambda kv: (kv[1].get("sharpe", -np.inf) if kv[0] != "benchmark_buy_hold" else -np.inf),
        )[0]
        res.significance["best_model"] = best
        bt = res.backtests[best]

        n_trials = cfg.evaluation.n_trials_for_deflated_sharpe
        res.significance["deflated_sharpe"] = deflated_sharpe_ratio(bt.returns, n_trials=n_trials)
        res.significance["bootstrap"] = stationary_bootstrap_sharpe(
            bt.returns, n_boot=cfg.evaluation.bootstrap_samples, seed=cfg.run.seed
        )
        family = pd.DataFrame({m: b.returns for m, b in res.backtests.items()})
        res.significance["pbo"] = probability_of_backtest_overfitting(
            family, n_splits=cfg.evaluation.pbo_splits
        )
        try:
            res.significance["randomisation"] = randomisation_test(
                preds, f"pred_{best}", n_permutations=100, seed=cfg.run.seed
            )
        except Exception as exc:
            logger.warning("Randomisation test failed: %s", exc)

        res.significance["capacity"] = turnover_capacity_analysis(
            bt.returns, bt.turnover
        ).to_dict("records")
        write_json(res.significance, art / "significance.json")

        if md.factors is not None and not md.factors.empty:
            for model, b in res.backtests.items():
                res.attribution[model] = factor_regression(b.returns, md.factors)
            write_json(res.attribution, art / "attribution.json")

        ladders = {}
        for model in training.model_names:
            qs = quantile_spread(
                preds[f"pred_{model}"].to_numpy(), preds["fwd_return"].to_numpy(), preds["date"]
            )
            if not qs.empty:
                ladders[model] = qs.set_index("quantile")["mean"]
        if ladders:
            pd.DataFrame(ladders).to_csv(art / "quantile_ladders.csv")

        mt = monthly_return_table(res.backtests[best].returns)
        if not mt.empty:
            mt.to_csv(art / "monthly_returns.csv")
    res.timings["evaluation"] = time.time() - t0

    t0 = time.time()
    try:
        if fs is None:
            raise RuntimeError("no feature set in scope; skipping model persistence")
        from stockrank.explain.shap_analysis import importance_stability, shap_summary
        from stockrank.models.persistence import fit_production_model, save_model

        tabular = [m for m in cfg.models.enabled if m in {"lightgbm", "ridge", "elasticnet"}]
        prod_name = "lightgbm" if "lightgbm" in tabular else (tabular[0] if tabular else None)
        if prod_name:
            prod, meta = fit_production_model(fs, cfg, prod_name)
            meta["run"] = cfg.run.name
            save_model(prod, fs.feature_names, art / f"model_{prod_name}.joblib", meta)

            if prod_name == "lightgbm":
                recent = fs.frame.tail(20_000)
                sh = shap_summary(prod, recent, fs.feature_names)
                sh.to_csv(art / "shap_summary.csv", index=False)
                res.explanation = {"shap_top": sh.head(15).to_dict("records")}

        by_model: dict[str, list] = {}
        for row in training.fold_metrics.to_dict("records"):
            by_model.setdefault(row["model"], [])
        if "lightgbm" in training.importances:
            stab = importance_stability(
                [training.importances[k] for k in training.importances if k == "lightgbm"]
                or [training.importances["lightgbm"]]
            )
            res.significance["importance_stability"] = stab
    except Exception as exc:
        logger.warning("Explainability / persistence step skipped: %s", exc)
    res.timings["explain"] = time.time() - t0

    if cfg.volatility.max_assets_fitted > 0:
        t0 = time.time()
        try:
            from stockrank.models.volatility import compare_volatility_models

            res.volatility = compare_volatility_models(
                daily_returns,
                max_assets=cfg.volatility.max_assets_fitted,
                refit_every=cfg.volatility.refit_every_days,
            )
            res.volatility.to_csv(art / "volatility_comparison.csv", index=False)
        except Exception as exc:
            logger.warning("Volatility study skipped: %s", exc)
        res.timings["volatility"] = time.time() - t0
