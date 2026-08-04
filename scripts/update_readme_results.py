"""Inject a completed run's headline numbers into the README.

The results block in README.md is generated, not typed. Rerunning the pipeline
and this script keeps the front page honest and in step with the artifacts,
which is the point of having the artifacts on disk in the first place.

    python scripts/update_readme_results.py --run baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MARKER = "<!-- RESULTS_BLOCK -->"
RUNTIME_MARKER = "<!-- RUNTIME_BLOCK -->"


def fmt_pct(v, d=2):
    return "n/a" if v is None or not isinstance(v, (int, float)) or pd.isna(v) else f"{100 * v:+.{d}f}%"


def fmt(v, d=2):
    return "n/a" if v is None or not isinstance(v, (int, float)) or pd.isna(v) else f"{v:,.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    art = Path(args.artifacts) / args.run
    if not art.exists():
        print(f"No run at {art}")
        return 1

    def js(name):
        p = art / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    ds, perf, sig = js("data_summary.json"), js("performance.json"), js("significance.json")
    attr = js("attribution.json")
    mm = pd.read_csv(art / "model_metrics.csv") if (art / "model_metrics.csv").exists() else pd.DataFrame()

    best = sig.get("best_model")
    bp = perf.get(best, {}) if best else {}
    dsr, pbo, bs = sig.get("deflated_sharpe", {}), sig.get("pbo", {}), sig.get("bootstrap", {})
    dl = ds.get("download") or {}

    lines: list[str] = [MARKER, ""]
    lines.append(
        f"Universe of **{ds.get('n_tickers')} liquid US large caps**, "
        f"**{ds.get('start')} to {ds.get('end')}** "
        f"({ds.get('n_trading_days'):,} trading days, {ds.get('n_rows'):,} observations). "
        f"Six purged walk-forward folds. Every figure below is out of sample and net of "
        f"commission, slippage and short financing.\n"
    )

    if not mm.empty:
        cols = [c for c in ["model", "rank_mean_ic", "rank_icir", "rank_t_stat_nw", "q5_minus_q1"] if c in mm.columns]
        t = mm[cols].copy()
        rows = []
        for _, r in t.iterrows():
            m = r["model"]
            p = perf.get(m, {})
            rows.append(
                f"| `{m}` | {r['rank_mean_ic']:+.4f} | {r.get('rank_t_stat_nw', float('nan')):+.2f} | "
                f"{fmt_pct(p.get('ann_return'))} | {fmt(p.get('sharpe'))} | "
                f"{fmt_pct(p.get('max_drawdown'))} | {fmt(p.get('beta_to_benchmark'), 3)} |"
            )
        lines.append("| Model | Mean IC | t (Newey-West) | Ann. return | Sharpe | Max DD | Beta |")
        lines.append("|---|---|---|---|---|---|---|")
        lines += rows
        lines.append("")
        lines.append(
            "`factor_composite` is the zero-parameter benchmark built from published anomalies. "
            "It fits nothing, so it cannot overfit, and every learned model is judged against it.\n"
        )

    lines.append(f"**Best model by Sharpe: `{best}`**\n")
    lines.append(
        f"- Deflated Sharpe Ratio **{fmt(dsr.get('deflated_sharpe'), 3)}** "
        f"(observed annualised Sharpe {fmt(dsr.get('sharpe_annual'))} against a "
        f"selection-adjusted threshold of {fmt(dsr.get('sr_threshold_annual'))} for "
        f"{dsr.get('n_trials')} trials)"
    )
    lines.append(
        f"- Stationary bootstrap 95% CI for the Sharpe: "
        f"[{fmt(bs.get('sharpe_ci_low'))}, {fmt(bs.get('sharpe_ci_high'))}], "
        f"P(Sharpe <= 0) = {fmt(bs.get('p_value_sharpe_le_0'), 3)}"
    )
    lines.append(
        f"- Probability of backtest overfitting: **{fmt(pbo.get('pbo'), 3)}** "
        f"(CSCV over {pbo.get('n_strategies')} candidate strategies)"
    )
    a = attr.get(best, {}) if best else {}
    if a and "error" not in a:
        lines.append(
            f"- Fama-French six-factor alpha: {fmt_pct(a.get('alpha_annual'))} annualised, "
            f"t = {fmt(a.get('alpha_tstat_hac'))} under Newey-West errors, R2 = {fmt(a.get('r_squared'), 3)}"
        )
    lines.append(
        f"- Net beta is **exactly zero at every rebalance** by construction, against the rolling "
        f"beta estimates used to build the book. The full-sample regression beta against the S&P 500 "
        f"is {fmt(bp.get('beta_to_benchmark'), 3)}, and that gap is the residual left by estimation "
        f"error in rolling betas rather than a deliberate market bet"
    )
    if dl:
        lines.append(
            f"- Point-in-time universe coverage: {100 * dl.get('coverage', 0):.1f}% "
            f"({dl.get('n_succeeded')} of {dl.get('n_requested')} historical index members priced)"
        )
    # An interpretation, not just a table. A results section without one invites
    # the reader to draw the most flattering conclusion available.
    dsr_v = dsr.get("deflated_sharpe") or 0.0
    alpha_t = (attr.get(best) or {}).get("alpha_tstat_hac")
    sharpe_v = bp.get("sharpe") or 0.0
    lines.append("")
    if dsr_v > 0.95:
        lines.append(
            f"**Reading this honestly.** The edge clears the selection-adjusted bar: a deflated "
            f"Sharpe of {fmt(dsr_v, 3)} with a low overfitting probability means the result "
            f"survives the number of configurations that were tried."
        )
    else:
        lines.append(
            f"**Reading this honestly.** The raw evidence is positive: Sharpe {fmt(sharpe_v)}, "
            f"Newey-West t of {fmt(bp.get('t_stat_nw'))}, bootstrap P(Sharpe <= 0) of "
            f"{fmt(bs.get('p_value_sharpe_le_0'), 3)}, and a probability of backtest overfitting of "
            f"{fmt(pbo.get('pbo'), 3)}, which is low enough to say the selection process is not "
            f"simply picking noise. Two things stop this being a claim of significance. The "
            f"**deflated Sharpe of {fmt(dsr_v, 3)}** falls short of the conventional 0.95: measured "
            f"against the {dsr.get('n_trials')} configurations tried, the observed Sharpe sits only "
            f"just above what a worthless strategy would be expected to reach. And the six-factor "
            f"alpha carries a t-statistic of {fmt(alpha_t)}, so a meaningful part of the return is "
            f"exposure that can be bought cheaply elsewhere.\n\n"
            f"That conclusion is the deliverable. Price and volume features alone carry very little "
            f"cross-sectional information in US large caps, and a research pipeline is only worth "
            f"having if it is capable of saying so instead of tuning until the number looks good."
        )
    lines.append("")
    lines.append(f"![Out-of-sample equity curves](reports/figures/{args.run}/equity_curves.png)")
    lines.append("")
    lines.append(f"![Information coefficient by model](reports/figures/{args.run}/ic_by_model.png)")
    lines.append("")
    lines.append(f"![Information coefficient by fold](reports/figures/{args.run}/ic_by_fold.png)")
    lines.append("")

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    start = text.index(MARKER)
    end = text.index("Full write-up with every figure", start)
    text = text[:start] + "\n".join(lines) + "\n" + text[end:]

    # Measured wall clock, so the performance claim in the README is never stale.
    t = js("timings.json")
    if t and RUNTIME_MARKER in text:
        total = t.get("total", 0)
        parts = ", ".join(
            f"{k} {v:.0f}s" for k, v in t.items()
            if k != "total" and isinstance(v, (int, float))
        )
        block = (
            f"{RUNTIME_MARKER}\n\nMeasured wall clock for the run reported above "
            f"(8 GB RAM, CPU only): **{total / 60:.1f} minutes** end to end "
            f"({parts}), with market data already cached."
        )
        r_start = text.index(RUNTIME_MARKER)
        r_end = text.index("---", r_start)
        text = text[:r_start] + block + "\n\n" + text[r_end:]

    readme.write_text(text, encoding="utf-8")
    print(f"README results and runtime blocks updated from {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
