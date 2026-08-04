"""Plotly chart builders for the research console."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from theme import ACCENT, BORDER, MUTED, NEG, PLOTLY_LAYOUT, POS, SERIES, TEXT


def _fig(title: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, height=height, title=dict(text=title, font=dict(size=14)))
    return fig


def equity_curves(curves: dict[str, pd.Series], benchmark: pd.Series | None = None) -> go.Figure:
    fig = _fig("Cumulative out-of-sample growth, net of costs", 420)
    for i, (name, s) in enumerate(sorted(curves.items())):
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.to_numpy(), name=name, mode="lines",
                line=dict(width=2, color=SERIES[i % len(SERIES)]),
                hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    if benchmark is not None and len(benchmark):
        fig.add_trace(
            go.Scatter(
                x=benchmark.index, y=benchmark.to_numpy(), name="S&P 500 buy and hold",
                mode="lines", line=dict(width=1.6, color=MUTED, dash="dash"),
            )
        )
    fig.add_hline(y=1.0, line=dict(color=BORDER, width=1))
    fig.update_yaxes(title="Growth of 1")
    return fig


def drawdown(returns: pd.Series, label: str) -> go.Figure:
    eq = (1 + returns.fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    fig = _fig(f"Drawdown: {label}", 260)
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.to_numpy(), mode="lines", name="drawdown",
            line=dict(color=NEG, width=1.4), fill="tozeroy",
            fillcolor="rgba(255,92,122,0.20)",
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title="%")
    fig.update_layout(showlegend=False)
    return fig


def ic_bars(metrics: pd.DataFrame) -> go.Figure:
    d = metrics.sort_values("rank_mean_ic")
    fig = _fig("Out-of-sample information coefficient by model", 60 + 46 * len(d))
    colors = [POS if v > 0 else NEG for v in d["rank_mean_ic"]]
    fig.add_trace(
        go.Bar(
            x=d["rank_mean_ic"], y=d["model"], orientation="h", marker=dict(color=colors),
            text=[f"{v:+.4f}   t={t:+.2f}" for v, t in zip(d["rank_mean_ic"], d.get("rank_t_stat_nw", d["rank_mean_ic"] * 0), strict=False)],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Mean daily cross-sectional Spearman IC")
    return fig


def ic_by_fold(fold_metrics: pd.DataFrame) -> go.Figure:
    piv = fold_metrics.pivot_table(index="fold", columns="model", values="rank_mean_ic", observed=True)
    fig = _fig("Information coefficient by walk-forward fold", 360)
    for i, col in enumerate(piv.columns):
        fig.add_trace(
            go.Scatter(
                x=piv.index, y=piv[col].to_numpy(), name=col, mode="lines+markers",
                line=dict(width=2, color=SERIES[i % len(SERIES)]), marker=dict(size=8),
            )
        )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="Fold (chronological)", dtick=1)
    fig.update_yaxes(title="Mean IC")
    return fig


def rolling_ic(ic: pd.Series, window: int, label: str) -> go.Figure:
    roll = ic.rolling(window).mean()
    fig = _fig(f"Rolling {window}-day information coefficient: {label}", 300)
    fig.add_trace(
        go.Scatter(
            x=roll.index, y=roll.to_numpy(), mode="lines", name="IC",
            line=dict(color=ACCENT, width=1.8), fill="tozeroy",
            fillcolor="rgba(76,141,255,0.16)",
        )
    )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(showlegend=False)
    return fig


def quantile_ladder(ladders: pd.DataFrame, horizon: int) -> go.Figure:
    fig = _fig(f"Mean {horizon}-day forward return by predicted quintile", 380)
    for i, col in enumerate(ladders.columns):
        fig.add_trace(
            go.Bar(
                x=[f"Q{int(q)}" for q in ladders.index], y=ladders[col].to_numpy() * 100,
                name=col, marker=dict(color=SERIES[i % len(SERIES)]),
                hovertemplate="%{x}: %{y:.3f}%<extra>" + col + "</extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(title="%")
    fig.update_layout(barmode="group")
    return fig


def feature_importance(imp: pd.Series, top: int, label: str) -> go.Figure:
    d = imp.abs().sort_values(ascending=False).head(top).sort_values()
    fig = _fig(f"Feature importance: {label}", 120 + 22 * len(d))
    fig.add_trace(
        go.Bar(
            x=d.to_numpy(), y=d.index, orientation="h",
            marker=dict(color=d.to_numpy(), colorscale=[[0, "#1E3A5F"], [1, ACCENT]], showscale=False),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Mean gain across folds")
    return fig


def monthly_heatmap(table: pd.DataFrame, label: str) -> go.Figure:
    cols = [c for c in table.columns if c != "Year"]
    z = table[cols].to_numpy(dtype=float) * 100
    lim = float(np.nanpercentile(np.abs(z), 96) or 1)
    fig = _fig(f"Monthly returns: {label}", 90 + 30 * len(table))
    fig.add_trace(
        go.Heatmap(
            z=z, x=cols, y=[str(i) for i in table.index],
            colorscale=[[0, NEG], [0.5, "#131B26"], [1, POS]], zmid=0, zmin=-lim, zmax=lim,
            text=[[f"{v:.1f}" if np.isfinite(v) else "" for v in row] for row in z],
            texttemplate="%{text}", textfont=dict(size=10),
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="%", thickness=12, outlinewidth=0),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def cost_curve(capacity: pd.DataFrame) -> go.Figure:
    fig = _fig("Sharpe ratio versus assumed one-way trading cost", 340)
    fig.add_trace(
        go.Scatter(
            x=capacity["cost_bps"], y=capacity["sharpe"], mode="lines+markers",
            line=dict(color="#FF9E5E", width=2.2), marker=dict(size=8),
            hovertemplate="%{x:.0f} bp: Sharpe %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=NEG, width=1, dash="dash"))
    fig.update_xaxes(title="Cost, basis points per unit turnover")
    fig.update_yaxes(title="Annualised Sharpe")
    fig.update_layout(showlegend=False)
    return fig


def exposure_chart(exposure: pd.DataFrame) -> go.Figure:
    fig = _fig("Portfolio exposure through time", 320)
    if "gross_exposure" in exposure:
        fig.add_trace(go.Scatter(x=exposure.index, y=exposure["gross_exposure"], name="gross",
                                 line=dict(color=ACCENT, width=1.8)))
    if "net_exposure" in exposure:
        fig.add_trace(go.Scatter(x=exposure.index, y=exposure["net_exposure"], name="net",
                                 line=dict(color=POS, width=1.6)))
    if "vol_scalar" in exposure:
        fig.add_trace(go.Scatter(x=exposure.index, y=exposure["vol_scalar"], name="vol scalar",
                                 line=dict(color="#FFB454", width=1.4, dash="dot"), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      title="vol scalar", color=MUTED))
    fig.update_yaxes(title="Leverage")
    return fig


def factor_betas(attr: dict) -> go.Figure:
    betas = {k.replace("beta_", "").upper(): v for k, v in attr.items()
             if k.startswith("beta_") and k != "beta_alpha"}
    tstats = {k.replace("tstat_", "").upper(): v for k, v in attr.items() if k.startswith("tstat_")}
    fig = _fig("Fama-French factor loadings", 330)
    keys = list(betas.keys())
    vals = [betas[k] for k in keys]
    fig.add_trace(
        go.Bar(
            x=keys, y=vals, marker=dict(color=[POS if v > 0 else NEG for v in vals]),
            text=[f"t={tstats.get(k, float('nan')):+.1f}" for k in keys],
            textposition="outside", textfont=dict(color=MUTED, size=10),
            hovertemplate="%{x}: beta %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Beta")
    return fig


def volatility_models(vol: pd.DataFrame) -> go.Figure:
    cols = [c for c in vol.columns if c.startswith("qlike_")]
    means = vol[cols].mean().sort_values()
    fig = _fig("Volatility forecast accuracy (QLIKE, lower is better)", 320)
    fig.add_trace(
        go.Bar(
            x=[c.replace("qlike_", "").upper() for c in means.index], y=means.to_numpy(),
            marker=dict(color=SERIES[: len(means)]),
            text=[f"{v:.4f}" for v in means.to_numpy()], textposition="outside",
            textfont=dict(color=TEXT, size=11),
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Mean QLIKE across names")
    return fig


def return_distribution(returns: pd.Series, label: str) -> go.Figure:
    r = returns.dropna() * 100
    fig = _fig(f"Daily return distribution: {label}", 320)
    fig.add_trace(
        go.Histogram(x=r.to_numpy(), nbinsx=70, marker=dict(color=ACCENT, opacity=0.75),
                     hovertemplate="%{x:.2f}%: %{y}<extra></extra>")
    )
    fig.add_vline(x=float(np.percentile(r, 5)), line=dict(color=NEG, width=1.5, dash="dash"),
                  annotation_text="5% VaR", annotation_font_color=NEG)
    fig.update_xaxes(title="Daily return, %")
    fig.update_layout(showlegend=False)
    return fig
