"""Equity Alpha Engine: interactive research console.

Reads artifacts produced by ``python -m alpha_engine.cli run`` and presents them
the way a research desk would review a signal: forecast quality first, strategy
performance second, and the statistical case for believing any of it third.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import charts  # noqa: E402
from data_access import find_runs, load_all_predictions, load_run  # noqa: E402
from theme import CSS, kpi_card  # noqa: E402

st.set_page_config(
    page_title="Equity Alpha Engine",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

PCT = lambda v, d=2: "n/a" if v is None or not np.isfinite(v) else f"{100 * v:+.{d}f}%"  # noqa: E731
NUM = lambda v, d=2: "n/a" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"  # noqa: E731


def tone(v: float, good_high: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return ""
    if good_high:
        return "pos" if v > 0 else "neg"
    return "neg" if v > 0 else "pos"


# --------------------------------------------------------------------- sidebar
runs = find_runs()
if not runs:
    st.markdown(
        '<div class="hero"><h1>Equity Alpha Engine</h1>'
        "<p>No completed runs found. Generate one first:</p></div>",
        unsafe_allow_html=True,
    )
    st.code(
        "pip install -e \".[all]\"\n"
        "python scripts/fetch_data.py --config configs/default.yaml\n"
        "python -m alpha_engine.cli run --config configs/default.yaml",
        language="bash",
    )
    st.stop()

with st.sidebar:
    st.markdown("### Equity Alpha Engine")
    st.caption("Cross-sectional forecasting and portfolio construction")
    st.divider()

    run_label = st.selectbox("Run", list(runs.keys()), index=0)
    run = load_run(str(runs[run_label]))

    all_models = sorted(run["backtests"].keys())
    default_sel = all_models[:6]
    selected = st.multiselect("Models", all_models, default=default_sel)
    if not selected:
        selected = all_models[:1]

    st.divider()
    cfg = run.get("config", {})
    ds = run.get("data_summary", {})
    st.markdown("**Configuration**")
    st.caption(
        f"Universe: `{cfg.get('data', {}).get('universe', 'n/a')}`  \n"
        f"Period: {ds.get('start', '?')} to {ds.get('end', '?')}  \n"
        f"Names: {ds.get('n_tickers', '?')}  \n"
        f"Horizon: {cfg.get('label', {}).get('horizon', '?')}d  \n"
        f"Folds: {cfg.get('validation', {}).get('n_splits', '?')}"
    )
    st.divider()
    st.caption(
        "All statistics are out of sample under purged walk-forward validation, "
        "net of commission, slippage and short financing."
    )

perf = run["performance"]
sig = run["significance"]
best = sig.get("best_model") or (selected[0] if selected else None)
bts = run["backtests"]
horizon = cfg.get("label", {}).get("horizon", 5)

# ------------------------------------------------------------------------ hero
n_names = ds.get("n_tickers", "?")
n_feats = run.get("feature_summary", {}).get("n_features", "?")
period = f"{ds.get('start', '?')} to {ds.get('end', '?')}"
dl = ds.get("download") or {}
cov = dl.get("coverage")
src = cfg.get("data", {}).get("source", "?")

st.markdown(
    f"""
<div class="hero">
  <h1>Equity Alpha Engine</h1>
  <p>Cross-sectional return forecasting on {n_names} US large-cap equities, {period}.
  {n_feats} price and volume factors, leakage-safe walk-forward validation, and a
  dollar- and beta-neutral long/short book with explicit trading costs.</p>
  <span class="pill ok">{src} data</span>
  <span class="pill">point-in-time universe</span>
  <span class="pill">purged + embargoed CV</span>
  <span class="pill">costs modelled</span>
  {'<span class="pill">' + f'{100 * cov:.0f}% member coverage' + '</span>' if cov else ''}
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    ["Overview", "Signal quality", "Strategy", "Risk and significance", "Live screen", "Method"]
)

# =========================================================== TAB 1: OVERVIEW
with tabs[0]:
    bp = perf.get(best, {}) if best else {}
    mm = run["model_metrics"]
    best_ic = (
        float(mm.loc[mm["model"] == best, "rank_mean_ic"].iloc[0])
        if best and not mm.empty and (mm["model"] == best).any()
        else np.nan
    )
    dsr = sig.get("deflated_sharpe", {})
    pbo = sig.get("pbo", {})

    c = st.columns(5)
    c[0].markdown(
        kpi_card("Sharpe (net)", NUM(bp.get("sharpe")), f"best model: {best}", tone(bp.get("sharpe"))),
        unsafe_allow_html=True,
    )
    c[1].markdown(
        kpi_card("Annual return", PCT(bp.get("ann_return")),
                 f"vol {PCT(bp.get('ann_volatility'), 1).lstrip('+')}", tone(bp.get("ann_return"))),
        unsafe_allow_html=True,
    )
    c[2].markdown(
        kpi_card("Max drawdown", PCT(bp.get("max_drawdown")),
                 f"Calmar {NUM(bp.get('calmar'))}", "neg"),
        unsafe_allow_html=True,
    )
    c[3].markdown(
        kpi_card("Information coeff.", f"{best_ic:+.4f}" if np.isfinite(best_ic) else "n/a",
                 "daily cross-sectional rank IC", tone(best_ic)),
        unsafe_allow_html=True,
    )
    c[4].markdown(
        kpi_card("Deflated Sharpe", NUM(dsr.get("deflated_sharpe"), 3),
                 f"{dsr.get('n_trials', '?')} trials adjusted",
                 "pos" if (dsr.get("deflated_sharpe") or 0) > 0.95 else "neg"),
        unsafe_allow_html=True,
    )

    dsr_v = dsr.get("deflated_sharpe") or 0.0
    sharpe_v = bp.get("sharpe") or 0.0
    pbo_v = pbo.get("pbo")
    bs_p = (sig.get("bootstrap") or {}).get("p_value_sharpe_le_0")
    a_best = (run["attribution"] or {}).get(best, {})
    alpha_t = a_best.get("alpha_tstat_hac")

    if dsr_v > 0.95 and sharpe_v > 0:
        st.markdown(
            f'<div class="callout good"><b>Verdict.</b> The best model clears the '
            f"selection-adjusted threshold with a deflated Sharpe of {NUM(dsr_v, 3)} and a "
            f"probability of backtest overfitting of {NUM(pbo_v, 3)}. The edge is small, as it "
            f"should be, but it survives costs and the multiple-testing adjustment.</div>",
            unsafe_allow_html=True,
        )
    elif sharpe_v > 0.5:
        st.markdown(
            f'<div class="callout warn"><b>Verdict: promising, not proven.</b> '
            f"The raw evidence is positive. Sharpe {NUM(sharpe_v)} with a Newey-West "
            f"t-statistic of {NUM(bp.get('t_stat_nw'))}, a bootstrap P(Sharpe &le; 0) of "
            f"{NUM(bs_p, 3)}, and a probability of backtest overfitting of {NUM(pbo_v, 3)}, "
            f"which is low enough to say the selection process is not simply fitting noise.<br><br>"
            f"Two things stop this being a claim of significance. The <b>deflated Sharpe of "
            f"{NUM(dsr_v, 3)}</b> is below the conventional 0.95: once the observed Sharpe is "
            f"measured against the {dsr.get('n_trials', '?')} configurations that were tried, it "
            f"sits only just above the threshold a worthless strategy would be expected to reach. "
            f"And the six-factor alpha carries a t-statistic of {NUM(alpha_t)}, so a large part of "
            f"the return is exposure that could be bought cheaply elsewhere.<br><br>"
            f"Reporting that is the point. Price and volume features alone carry very little "
            f"cross-sectional information in US large caps, and a research pipeline is only "
            f"useful if it is capable of saying so.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="callout warn"><b>Verdict.</b> The observed Sharpe of {NUM(sharpe_v)} '
            f"does not support a tradable claim, and the deflated Sharpe of {NUM(dsr_v, 3)} "
            f"confirms it. Reporting this rather than tuning until it looks good is the point: "
            f"the value of the pipeline is that it can return a negative answer.</div>",
            unsafe_allow_html=True,
        )

    curves = {m: (1 + bts[m]["net_return"].fillna(0)).cumprod() for m in selected if m in bts}
    if curves:
        st.plotly_chart(charts.equity_curves(curves), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        if best in bts:
            st.plotly_chart(charts.drawdown(bts[best]["net_return"], best), width="stretch")
    with right:
        if not mm.empty:
            st.plotly_chart(charts.ic_bars(mm), width="stretch")

# ==================================================== TAB 2: SIGNAL QUALITY
with tabs[1]:
    st.markdown('<div class="section-title">How much does the model actually know?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The information coefficient is the daily cross-sectional '
        "Spearman correlation between forecast and realised forward return. For daily equity "
        "signals a mean IC of 0.02 to 0.04 supports a real strategy. Anything above 0.10 in a "
        "backtest of this type is almost always a data leak rather than a discovery.</div>",
        unsafe_allow_html=True,
    )

    mm = run["model_metrics"]
    if not mm.empty:
        show = mm[[c for c in ["model", "rank_mean_ic", "rank_icir", "rank_t_stat_nw",
                               "rank_hit_rate", "q5_minus_q1", "oos_r2", "fit_seconds"] if c in mm.columns]]
        st.dataframe(
            show.rename(columns={
                "rank_mean_ic": "mean IC", "rank_icir": "ICIR", "rank_t_stat_nw": "t (Newey-West)",
                "rank_hit_rate": "hit rate", "q5_minus_q1": "Q5-Q1", "oos_r2": "OOS R2",
                "fit_seconds": "fit (s)"}).style.format(precision=4),
            width="stretch", hide_index=True,
        )

    fm = run["fold_metrics"]
    if not fm.empty:
        st.markdown('<div class="section-title">Stability across folds</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">The average matters less than the consistency. A signal '
            "that is strong in two folds and negative in three is not a signal, it is a period "
            "effect.</div>", unsafe_allow_html=True,
        )
        st.plotly_chart(charts.ic_by_fold(fm[fm["model"].isin(selected)]), width="stretch")

    lad = run["ladders"]
    if not lad.empty:
        cols = [c for c in lad.columns if c in selected] or list(lad.columns)
        st.plotly_chart(charts.quantile_ladder(lad[cols], horizon), width="stretch")
        st.markdown(
            '<div class="section-note">A monotone ladder from Q1 to Q5 is a far stronger claim '
            "than a positive average IC, because it shows the ordering holds across the whole "
            "cross-section rather than being driven by one extreme bucket.</div>",
            unsafe_allow_html=True,
        )

    imp = run["importance"]
    if not imp.empty:
        st.markdown('<div class="section-title">What the model uses</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        model_choice = c1.selectbox("Model", list(imp.columns), key="imp_model")
        top_n = c1.slider("Features shown", 5, min(30, len(imp)), 18)
        c2.plotly_chart(
            charts.feature_importance(imp[model_choice].dropna(), top_n, model_choice),
            width="stretch",
        )

# ========================================================== TAB 3: STRATEGY
with tabs[2]:
    pcfg = cfg.get("portfolio", {})
    bcfg = cfg.get("backtest", {})
    st.markdown('<div class="section-title">Portfolio construction</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">{pcfg.get("method", "?")}, {pcfg.get("n_long", "?")} long and '
        f'{pcfg.get("n_short", "?")} short, dollar neutral'
        f'{" and beta neutral" if pcfg.get("beta_neutral") else ""}, gross leverage '
        f'{pcfg.get("gross_leverage", "?")}x, targeting {100 * float(pcfg.get("vol_target_annual", 0)):.0f}% '
        f'annualised volatility, rebalanced every {pcfg.get("rebalance_days", "?")} days. Costs: '
        f'{bcfg.get("cost_bps", "?")} bp commission plus {bcfg.get("slippage_bps", "?")} bp slippage '
        f'per unit turnover, plus {bcfg.get("borrow_bps_annual", "?")} bp annual borrow on the short leg.</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for m, p in perf.items():
        rows.append({
            "strategy": m,
            "ann. return": p.get("ann_return"), "ann. vol": p.get("ann_volatility"),
            "Sharpe": p.get("sharpe"), "Sortino": p.get("sortino"),
            "max DD": p.get("max_drawdown"), "Calmar": p.get("calmar"),
            "t (NW)": p.get("t_stat_nw"), "beta": p.get("beta_to_benchmark"),
            "gross ann.": p.get("gross_ann_return"), "cost drag": p.get("cost_drag_annual"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), width="stretch", hide_index=True)

    if best in bts:
        b = bts[best]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.return_distribution(b["net_return"], best), width="stretch")
        with c2:
            expo = [c for c in ["gross_exposure", "net_exposure", "net_beta", "vol_scalar"] if c in b.columns]
            if expo:
                st.plotly_chart(charts.exposure_chart(b[expo]), width="stretch")

        mo = run["monthly"]
        if not mo.empty:
            st.plotly_chart(charts.monthly_heatmap(mo, best), width="stretch")

# ============================================= TAB 4: RISK AND SIGNIFICANCE
with tabs[3]:
    st.markdown('<div class="section-title">Is the result real?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A Sharpe ratio computed on the winning configuration of a '
        "search is not an estimate of anything. Try forty combinations on fifteen years of data "
        "and the best will show a Sharpe near 0.8 even if every one is worthless. These tests "
        "quantify that effect instead of hoping it away.</div>",
        unsafe_allow_html=True,
    )

    dsr, bs, pbo = sig.get("deflated_sharpe", {}), sig.get("bootstrap", {}), sig.get("pbo", {})
    rnd = sig.get("randomisation", {})
    c = st.columns(4)
    c[0].markdown(kpi_card("Deflated Sharpe", NUM(dsr.get("deflated_sharpe"), 3),
                           "P(true Sharpe > threshold)",
                           "pos" if (dsr.get("deflated_sharpe") or 0) > 0.95 else "neg"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Sharpe threshold", NUM(dsr.get("sr_threshold_annual")),
                           f"expected max over {dsr.get('n_trials', '?')} trials"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Bootstrap 95% CI",
                           f"{NUM(bs.get('sharpe_ci_low'))} to {NUM(bs.get('sharpe_ci_high'))}",
                           f"P(Sharpe <= 0) = {NUM(bs.get('p_value_sharpe_le_0'), 3)}"), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Overfit probability", NUM(pbo.get("pbo"), 3),
                           f"CSCV over {pbo.get('n_strategies', '?')} strategies",
                           "pos" if (pbo.get("pbo") or 1) < 0.5 else "neg"), unsafe_allow_html=True)

    if rnd:
        st.markdown(
            f'<div class="callout"><b>Randomisation test.</b> Shuffling the forecasts within each '
            f'date {rnd.get("n_permutations")} times and recomputing the IC gives '
            f'p = {NUM(rnd.get("permutation_p_value"), 3)} against an actual IC of '
            f'{NUM(rnd.get("actual_ic"), 4)}. This is distribution free and makes no assumption '
            f"about the shape of the return distribution.</div>",
            unsafe_allow_html=True,
        )

    attr = run["attribution"]
    if attr:
        st.markdown('<div class="section-title">Factor attribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Daily strategy excess returns regressed on the Fama-French '
            "five factors plus momentum, with Newey-West standard errors. The intercept is what "
            "remains after exposure that could be bought cheaply through index products is "
            "stripped out. That intercept, not the raw return, is the thing worth paying for.</div>",
            unsafe_allow_html=True,
        )
        rows = [{"strategy": m, "alpha (ann.)": a.get("alpha_annual"),
                 "t(alpha) HAC": a.get("alpha_tstat_hac"), "R2": a.get("r_squared"),
                 "b_mkt": a.get("beta_mkt_rf"), "b_smb": a.get("beta_smb"),
                 "b_hml": a.get("beta_hml"), "b_mom": a.get("beta_mom")}
                for m, a in attr.items() if "error" not in a]
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format(precision=3),
                         width="stretch", hide_index=True)
        if best in attr and "error" not in attr[best]:
            st.plotly_chart(charts.factor_betas(attr[best]), width="stretch")

    cap = pd.DataFrame(sig.get("capacity", []))
    if not cap.empty:
        st.markdown('<div class="section-title">Implementability</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">The break-even trading cost is the single most informative '
            "number about whether a signal is tradable. A strategy that needs sub-3bp execution is "
            "a high-frequency shop's problem, not a research result.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.cost_curve(cap), width="stretch")

    vol = run["volatility"]
    if not vol.empty:
        st.markdown('<div class="section-title">Volatility forecasting study</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Return levels are close to unpredictable; return variance is '
            "not. GARCH(1,1) with Student-t errors, HAR-RV and RiskMetrics EWMA compared out of "
            "sample under QLIKE, which is robust to the fact that true variance is never observed.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.volatility_models(vol), width="stretch")

# ======================================================= TAB 5: LIVE SCREEN
with tabs[4]:
    st.markdown('<div class="section-title">Latest cross-sectional screen</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The book the model would hold on the most recent date in the '
        "sample. This is the output an investment committee would actually look at.</div>",
        unsafe_allow_html=True,
    )
    preds = load_all_predictions(run["path"])
    if preds.empty:
        st.info("No prediction artifact found for this run.")
    else:
        model_pick = st.selectbox("Signal", [m for m in selected if f"pred_{m}" in preds.columns] or
                                  [c[5:] for c in preds.columns if c.startswith("pred_")], key="screen_model")
        col = f"pred_{model_pick}"
        latest = preds["date"].max()
        day = preds[preds["date"] == latest].copy()
        day["rank"] = day[col].rank(ascending=False).astype(int)
        n_side = int(cfg.get("portfolio", {}).get("n_long", 20))

        st.caption(f"As of {latest.date()}  |  {len(day)} names ranked  |  signal: {model_pick}")
        cols_show = [c for c in ["rank", "ticker", "sector", col, "close", "beta_raw", "vol_raw"] if c in day.columns]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Long candidates**")
            st.dataframe(
                day.nlargest(n_side, col)[cols_show].rename(columns={col: "score", "beta_raw": "beta", "vol_raw": "vol"}
                                                            ).style.format(precision=4),
                width="stretch", hide_index=True, height=460,
            )
        with c2:
            st.markdown("**Short candidates**")
            st.dataframe(
                day.nsmallest(n_side, col)[cols_show].rename(columns={col: "score", "beta_raw": "beta", "vol_raw": "vol"}
                                                             ).style.format(precision=4),
                width="stretch", hide_index=True, height=460,
            )
        st.download_button(
            "Download full screen as CSV",
            day.sort_values(col, ascending=False).to_csv(index=False).encode(),
            file_name=f"screen_{model_pick}_{latest.date()}.csv",
            mime="text/csv",
        )
        st.markdown(
            '<div class="callout warn">This is a research artifact from a historical backtest, '
            "not investment advice, and the most recent date is the end of the sample rather than "
            "today.</div>", unsafe_allow_html=True,
        )

# ========================================================== TAB 6: METHOD
with tabs[5]:
    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(kpi_card("Rows", f"{ds.get('n_rows', 0):,}", "date x ticker observations"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Names", str(ds.get("n_tickers", "?")), f"{ds.get('n_sectors', '?')} GICS sectors"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Trading days", f"{ds.get('n_trading_days', 0):,}", period), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Member coverage", f"{100 * cov:.1f}%" if cov else "n/a",
                           f"{dl.get('n_succeeded', '?')} of {dl.get('n_requested', '?')} historical members"),
                  unsafe_allow_html=True)

    st.markdown(
        '<div class="callout"><b>Survivorship bias.</b> Index membership is reconstructed point in '
        "time from the historical record of additions and removals, so the universe on any past "
        "date is what it actually was, not what it is today. The residual limitation is that "
        "delisted and acquired names often have no price history available, which is what the "
        "coverage figure measures. The surviving sample is therefore mildly favourable and the "
        "true edge is likely a little lower than reported.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Validation scheme</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A label with a five-day horizon formed on Monday is still being '
        "realised on Friday. Ordinary cross-validation puts Monday in training and Wednesday in "
        "test, and the two share four days of the same price path. Purging removes training rows "
        "whose label window overlaps the test window; the embargo drops a further buffer "
        "afterwards because serial correlation in the features carries information across the "
        "boundary too.</div>",
        unsafe_allow_html=True,
    )
    folds = run.get("folds", [])
    if folds:
        st.dataframe(pd.DataFrame(folds), width="stretch", hide_index=True)

    t = run.get("timings", {})
    if t:
        st.markdown('<div class="section-title">Runtime</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{k: round(v, 1) for k, v in t.items()}]),
            width="stretch", hide_index=True,
        )

    with st.expander("Full configuration"):
        st.json(cfg)

st.divider()
st.caption(
    "Equity Alpha Engine  |  research code, not investment advice  |  "
    "every figure is out of sample and net of modelled trading costs"
)
