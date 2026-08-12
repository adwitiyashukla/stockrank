from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockrank.config import Config
from stockrank.evaluation.metrics import daily_ic
from stockrank.evaluation.performance import monthly_return_table
from stockrank.reporting import figures as F
from stockrank.utils.io import ensure_dir, read_json
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

PCT = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"
NUM = lambda x, d=2: "n/a" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "_no data_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.to_numpy()]
    return "\n".join([header, sep, *rows]) + "\n"


def build_report(res, figures_dir: str | Path | None = None) -> Path:
    F.apply_style()
    cfg: Config = res.cfg
    art = Path(res.dir)
    figdir = ensure_dir(Path(figures_dir) if figures_dir else Path(cfg.run.reports_dir) / "figures" / cfg.run.name)

    curves = {m: b.equity_curve for m, b in res.backtests.items()}
    bench = None
    if res.backtests:
        idx = next(iter(res.backtests.values())).returns.index
        try:
            from stockrank.data.loader import load_market_data

            md = load_market_data(cfg)
            b = md.market.copy()
            b["date"] = pd.to_datetime(b["date"])
            bench_ret = b.set_index("date")["mkt_return"].reindex(idx).fillna(0.0)
            bench = (1 + bench_ret).cumprod()
        except Exception as exc:
            logger.warning("Benchmark curve unavailable: %s", exc)

    figs: dict[str, Path] = {}
    if curves:
        figs["equity"] = F.equity_curves(
            curves, bench, figdir / "equity_curves.png",
            f"Out-of-sample equity curves, net of costs ({cfg.data.start[:4]} to {cfg.data.end[:4]})",
        )
    if res.training is not None and not res.training.overall_metrics.empty:
        figs["ic_bar"] = F.ic_bar(res.training.overall_metrics, figdir / "ic_by_model.png")
    if res.training is not None and not res.training.fold_metrics.empty:
        figs["folds"] = F.fold_stability(res.training.fold_metrics, figdir / "ic_by_fold.png")

    best = res.significance.get("best_model")
    if best and best in res.backtests:
        bt = res.backtests[best]
        figs["drawdown"] = F.drawdown_plot(bt.returns, figdir / "drawdown.png", best)
        figs["rolling_sharpe"] = F.rolling_sharpe_plot(bt.returns, figdir / "rolling_sharpe.png", best)
        preds = res.training.predictions
        ic = daily_ic(preds[f"pred_{best}"], preds["target"], preds["date"])
        figs["ic_ts"] = F.ic_timeseries(ic, figdir / "ic_timeseries.png", best)
        cap = pd.DataFrame(res.significance.get("capacity", []))
        if not cap.empty:
            figs["cost"] = F.cost_sensitivity(cap, figdir / "cost_sensitivity.png")
        mt = monthly_return_table(bt.returns)
        if not mt.empty:
            figs["monthly"] = F.monthly_heatmap(mt, figdir / "monthly_returns.png", best)
        if best in res.attribution:
            figs["factors"] = F.factor_exposures(res.attribution[best], figdir / "factor_exposures.png")

    ladders_p = art / "quantile_ladders.csv"
    if ladders_p.exists():
        lad = pd.read_csv(ladders_p, index_col=0)
        figs["ladder"] = F.quantile_ladder(lad, figdir / "quantile_ladder.png", cfg.label.horizon)

    if res.training is not None and res.training.importances:
        key = "lightgbm" if "lightgbm" in res.training.importances else next(iter(res.training.importances))
        figs["importance"] = F.feature_importance(
            res.training.importances[key], figdir / "feature_importance.png", label=key
        )

    if res.volatility is not None and not res.volatility.empty:
        figs["vol"] = F.volatility_comparison(res.volatility, figdir / "volatility_models.png")

    logger.info("Wrote %d figures to %s", len(figs), figdir)
    path = _write_markdown(res, figdir, figs)
    return path


def _write_markdown(res, figdir: Path, figs: dict[str, Path]) -> Path:
    cfg: Config = res.cfg
    art = Path(res.dir)
    def rel(p) -> str:
        return (Path("figures") / cfg.run.name / Path(p).name).as_posix()

    lines: list[str] = []
    A = lines.append

    A(f"# Results: `{cfg.run.name}`\n")
    A(
        "All figures are out of sample, net of commission, slippage and short financing.\n"
    )

    ds = res.data_summary
    A("## 1. Dataset\n")
    A(f"- **Source**: {cfg.data.source}, universe `{cfg.data.universe}`")
    A(f"- **Period**: {ds.get('start')} to {ds.get('end')} ({ds.get('n_trading_days'):,} trading days)")
    A(f"- **Cross-section**: {ds.get('n_tickers')} names, {ds.get('n_sectors')} GICS sectors, {ds.get('n_rows'):,} rows")
    dl = ds.get("download") or {}
    if dl:
        A(
            f"- **Point-in-time coverage**: {dl.get('n_succeeded')} of {dl.get('n_requested')} "
            f"historical index members priced ({100 * dl.get('coverage', 0):.1f}%)"
        )
    fsum = res.feature_summary
    A(
        f"- **Features**: {fsum.get('n_features')} cross-sectionally normalised factors, "
        f"target = {cfg.label.type} over {cfg.label.horizon} days with a "
        f"{fsum.get('execution_lag_days')}-day execution lag\n"
    )

    A("## 2. Forecast quality\n")
    A(
        "IC is the daily cross-sectional Spearman correlation between forecast and realised "
        "forward return.\n"
    )
    om = res.training.overall_metrics if res.training is not None else pd.DataFrame()
    if not om.empty:
        show = om[
            [c for c in ["model", "rank_mean_ic", "rank_icir", "rank_t_stat_nw", "hit_rate", "q5_minus_q1", "oos_r2", "fit_seconds"]
             if c in om.columns]
        ].rename(
            columns={
                "rank_mean_ic": "mean IC", "rank_icir": "ICIR", "rank_t_stat_nw": "t (Newey-West)",
                "q5_minus_q1": "Q5-Q1", "oos_r2": "OOS R2", "fit_seconds": "fit (s)",
            }
        )
        A(_md_table(show))
    if "ic_bar" in figs:
        A(f"![Information coefficient by model]({rel(figs['ic_bar'])})\n")
    if "folds" in figs:
        A(f"![IC by fold]({rel(figs['folds'])})\n")
    if "ladder" in figs:
        A(f"![Quantile ladder]({rel(figs['ladder'])})\n")

    A("## 3. Strategy performance\n")
    A(
        f"Construction: {cfg.portfolio.method}, {cfg.portfolio.n_long} long and "
        f"{cfg.portfolio.n_short} short, dollar neutral, gross leverage "
        f"{cfg.portfolio.gross_leverage:.1f}x, targeting {100 * cfg.portfolio.vol_target_annual:.0f}% "
        f"annualised volatility, rebalanced every {cfg.portfolio.rebalance_days} days. Costs: "
        f"{cfg.backtest.cost_bps:.0f} bp commission plus {cfg.backtest.slippage_bps:.0f} bp slippage "
        f"per unit of turnover, plus {cfg.backtest.borrow_bps_annual:.0f} bp annual borrow on the short leg.\n"
    )
    if res.performance:
        rows = []
        for m, p in res.performance.items():
            rows.append(
                {
                    "strategy": m,
                    "ann. return": p.get("ann_return"),
                    "ann. vol": p.get("ann_volatility"),
                    "Sharpe": p.get("sharpe"),
                    "Sortino": p.get("sortino"),
                    "max DD": p.get("max_drawdown"),
                    "Calmar": p.get("calmar"),
                    "t (NW)": p.get("t_stat_nw"),
                    "beta to mkt": p.get("beta_to_benchmark"),
                }
            )
        A(_md_table(pd.DataFrame(rows), "{:.3f}"))
    if "equity" in figs:
        A(f"![Equity curves]({rel(figs['equity'])})\n")
    if "drawdown" in figs:
        A(f"![Drawdown]({rel(figs['drawdown'])})\n")
    if "rolling_sharpe" in figs:
        A(f"![Rolling Sharpe]({rel(figs['rolling_sharpe'])})\n")
    if "monthly" in figs:
        A(f"![Monthly returns]({rel(figs['monthly'])})\n")

    A("## 4. Is the result real?\n")
    sig = res.significance or {}
    if sig:
        dsr = sig.get("deflated_sharpe", {})
        bs = sig.get("bootstrap", {})
        pbo = sig.get("pbo", {})
        rnd = sig.get("randomisation", {})
        A(f"Best strategy by Sharpe: **{sig.get('best_model')}**\n")
        A(
            f"- **Deflated Sharpe Ratio**: {NUM(dsr.get('deflated_sharpe'), 3)}. "
            f"Observed annualised Sharpe {NUM(dsr.get('sharpe_annual'))}, versus a selection-adjusted "
            f"threshold of {NUM(dsr.get('sr_threshold_annual'))} implied by "
            f"{dsr.get('n_trials')} trials."
        )
        A(
            f"- **Stationary bootstrap 95% CI for the Sharpe**: "
            f"[{NUM(bs.get('sharpe_ci_low'))}, {NUM(bs.get('sharpe_ci_high'))}], "
            f"P(Sharpe <= 0) = {NUM(bs.get('p_value_sharpe_le_0'), 3)}."
        )
        A(
            f"- **Probability of backtest overfitting** (CSCV over {pbo.get('n_strategies')} candidate "
            f"strategies, {pbo.get('n_combinations')} splits): {NUM(pbo.get('pbo'), 3)}."
        )
        if rnd:
            A(
                f"- **Randomisation test**: shuffling forecasts within each date "
                f"{rnd.get('n_permutations')} times gives p = {NUM(rnd.get('permutation_p_value'), 3)}.\n"
            )
    if res.attribution:
        A("### Factor attribution\n")
        A(
            "Daily strategy excess returns regressed on the Fama-French five factors plus "
            "momentum, Newey-West standard errors.\n"
        )
        rows = []
        for m, a in res.attribution.items():
            if "error" in a:
                continue
            rows.append(
                {
                    "strategy": m,
                    "alpha (ann.)": a.get("alpha_annual"),
                    "t(alpha) HAC": a.get("alpha_tstat_hac"),
                    "R2": a.get("r_squared"),
                    "b_mkt": a.get("beta_mkt_rf"),
                    "b_smb": a.get("beta_smb"),
                    "b_hml": a.get("beta_hml"),
                    "b_mom": a.get("beta_mom"),
                }
            )
        if rows:
            A(_md_table(pd.DataFrame(rows), "{:.3f}"))
    if "factors" in figs:
        A(f"![Factor exposures]({rel(figs['factors'])})\n")

    A("## 5. Implementability\n")
    cap = pd.DataFrame(sig.get("capacity", []))
    if not cap.empty:
        A(
            "Sharpe as a function of the assumed one-way cost.\n"
        )
        A(_md_table(cap.rename(columns={"cost_bps": "cost (bp)", "ann_return": "ann. return", "sharpe": "Sharpe"}), "{:.3f}"))
    if "cost" in figs:
        A(f"![Cost sensitivity]({rel(figs['cost'])})\n")

    if "importance" in figs:
        A("## 6. What the model uses\n")
        A(f"![Feature importance]({rel(figs['importance'])})\n")

    if res.volatility is not None and not res.volatility.empty:
        A("## 7. Volatility forecasting study\n")
        A(
            "GARCH(1,1) with Student-t errors, HAR-RV and EWMA compared out of sample under "
            "QLIKE.\n"
        )
        cols = [c for c in res.volatility.columns if c.startswith(("qlike_", "mse_"))]
        summary = res.volatility[cols].mean().to_frame("mean loss").reset_index()
        summary.columns = ["metric", "mean across names"]
        A(_md_table(summary, "{:.6f}"))
        if "vol" in figs:
            A(f"![Volatility models]({rel(figs['vol'])})\n")

    A("## 8. Limitations\n")
    A(
        f"1. About {100 * (1 - (dl.get('coverage', 1) if dl else 1)):.0f}% of historical index "
        "members cannot be priced, so some survivorship bias remains.\n"
        "2. Costs are linear in turnover, no market impact model.\n"
        "3. Fills assumed at the close with a one-day lag.\n"
        "4. Price and volume only, no fundamentals, revisions or short interest.\n"
        "5. US large caps only.\n"
        "6. The deflated Sharpe uses an assumed trial count, so multiple testing is bounded but "
        "not eliminated.\n"
    )

    A("## 9. Reproducing this run\n")
    A("```bash")
    A("pip install -e \".[all]\"")
    A(f"python scripts/fetch_data.py --config configs/{cfg.run.name}.yaml   # or your config")
    A(f"python -m stockrank.cli run --config configs/{cfg.run.name}.yaml")
    A("```\n")
    t = res.timings or {}
    A(
        f"Wall clock on the reference machine (8 GB RAM, CPU only): "
        f"data {NUM(t.get('data'), 0)}s, features {NUM(t.get('features'), 0)}s, "
        f"training {NUM(t.get('training'), 0)}s, backtest {NUM(t.get('backtest'), 0)}s, "
        f"total {NUM(t.get('total'), 0)}s. Seed `{cfg.run.seed}`.\n"
    )

    body = "\n".join(lines)
    top = Path(cfg.run.reports_dir) / f"RESULTS_{cfg.run.name}.md"
    ensure_dir(top.parent)
    top.write_text(body, encoding="utf-8")

    out = art / "RESULTS.md"
    out.write_text(body.replace("](figures/", "](../../reports/figures/"), encoding="utf-8")
    logger.info("Report written to %s and %s", out, top)
    return top


def build_report_from_disk(cfg: Config) -> Path:
    from stockrank.backtest.engine import BacktestResult
    from stockrank.experiment import ExperimentResult
    from stockrank.models.trainer import TrainingResult

    art = Path(cfg.run.artifacts_dir) / cfg.run.name
    if not art.exists():
        raise FileNotFoundError(f"No artifacts at {art}. Run the pipeline first.")

    res = ExperimentResult(cfg, art)
    res.data_summary = read_json(art / "data_summary.json") if (art / "data_summary.json").exists() else {}
    res.feature_summary = read_json(art / "feature_summary.json") if (art / "feature_summary.json").exists() else {}
    res.performance = read_json(art / "performance.json") if (art / "performance.json").exists() else {}
    res.significance = read_json(art / "significance.json") if (art / "significance.json").exists() else {}
    res.attribution = read_json(art / "attribution.json") if (art / "attribution.json").exists() else {}
    res.timings = read_json(art / "timings.json") if (art / "timings.json").exists() else {}

    preds = pd.read_parquet(art / "predictions.parquet")
    imp: dict[str, Any] = {}
    if (art / "feature_importance.csv").exists():
        imp_df = pd.read_csv(art / "feature_importance.csv", index_col=0)
        imp = {c: imp_df[c].dropna() for c in imp_df.columns}
    res.training = TrainingResult(
        predictions=preds,
        fold_metrics=pd.read_csv(art / "fold_metrics.csv"),
        overall_metrics=pd.read_csv(art / "model_metrics.csv"),
        importances=imp,
    )

    for p in sorted(art.glob("backtest_*.parquet")):
        name = p.stem.replace("backtest_", "")
        df = pd.read_parquet(p)
        res.backtests[name] = BacktestResult(
            returns=df["net_return"],
            gross_returns=df["gross_return"],
            costs=df["cost"],
            turnover=df["turnover"],
            exposure=df[[c for c in df.columns
                         if c.endswith("exposure") or c.startswith("n_")
                         or c in ("vol_scalar", "net_beta")]],
            weights=pd.DataFrame(),
        )
    if (art / "volatility_comparison.csv").exists():
        res.volatility = pd.read_csv(art / "volatility_comparison.csv")
    return build_report(res)
