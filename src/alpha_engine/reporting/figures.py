"""Publication-quality figures for the results report and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alpha_engine.utils.io import ensure_dir  # noqa: E402
from alpha_engine.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

INK = "#1b2733"
MUTED = "#7c8b9a"
GRID = "#e3e8ee"
PALETTE = ["#2f6fed", "#e8804a", "#2aa198", "#a05fd3", "#d64d6a", "#6b7f95", "#3fa34d"]
POS, NEG = "#2aa198", "#d64d6a"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.7,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def _save(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    return path


def equity_curves(
    curves: dict[str, pd.Series], benchmark: pd.Series | None, out: Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, s) in enumerate(sorted(curves.items())):
        ax.plot(s.index, s.to_numpy(), label=name, color=PALETTE[i % len(PALETTE)])
    if benchmark is not None and len(benchmark):
        ax.plot(
            benchmark.index, benchmark.to_numpy(), label="S&P 500 buy and hold",
            color=MUTED, linestyle="--", linewidth=1.4,
        )
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.legend(ncol=3, loc="upper left")
    return _save(fig, out)


def drawdown_plot(returns: pd.Series, out: Path, label: str) -> Path:
    eq = (1 + returns.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(dd.index, dd.to_numpy() * 100, 0, color=NEG, alpha=0.28)
    ax.plot(dd.index, dd.to_numpy() * 100, color=NEG, linewidth=1.2)
    ax.set_title(f"Drawdown: {label}")
    ax.set_ylabel("%")
    return _save(fig, out)


def ic_bar(metrics: pd.DataFrame, out: Path) -> Path:
    d = metrics.sort_values("rank_mean_ic")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(d) + 2))
    colors = [POS if v > 0 else NEG for v in d["rank_mean_ic"]]
    ax.barh(d["model"], d["rank_mean_ic"], color=colors, height=0.6)
    for y, (v, t) in enumerate(zip(d["rank_mean_ic"], d["rank_t_stat_nw"], strict=False)):
        ax.text(
            v + (0.0004 if v >= 0 else -0.0004), y, f"{v:+.4f}  (t={t:+.1f})",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK,
        )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title("Out-of-sample rank information coefficient")
    ax.set_xlabel("Mean daily cross-sectional Spearman IC")
    ax.margins(x=0.25)
    return _save(fig, out)


def ic_timeseries(ic: pd.Series, out: Path, label: str, window: int = 63) -> Path:
    roll = ic.rolling(window).mean()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(roll.index, roll.to_numpy(), color=PALETTE[0])
    ax.fill_between(roll.index, roll.to_numpy(), 0, where=roll > 0, color=POS, alpha=0.22)
    ax.fill_between(roll.index, roll.to_numpy(), 0, where=roll <= 0, color=NEG, alpha=0.22)
    ax.set_title(f"Rolling {window}-day information coefficient: {label}")
    ax.set_ylabel("IC")
    return _save(fig, out)


def quantile_ladder(ladders: pd.DataFrame, out: Path, horizon: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    n = ladders.shape[1]
    width = 0.8 / max(n, 1)
    x = np.arange(len(ladders.index))
    for i, col in enumerate(ladders.columns):
        ax.bar(
            x + i * width - 0.4 + width / 2, ladders[col].to_numpy() * 100,
            width=width, label=col, color=PALETTE[i % len(PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{int(q)}" for q in ladders.index])
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title(f"Mean {horizon}-day forward return by predicted quintile")
    ax.set_ylabel("%")
    ax.legend(ncol=3)
    return _save(fig, out)


def feature_importance(imp: pd.Series, out: Path, top: int = 20, label: str = "") -> Path:
    d = imp.abs().sort_values(ascending=False).head(top).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.6))
    ax.barh(d.index, d.to_numpy(), color=PALETTE[0], height=0.65)
    ax.set_title(f"Feature importance{': ' + label if label else ''}")
    ax.set_xlabel("Mean gain across folds")
    return _save(fig, out)


def rolling_sharpe_plot(returns: pd.Series, out: Path, label: str, window: int = 252) -> Path:
    r = returns.dropna()
    rs = r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(rs.index, rs.to_numpy(), color=PALETTE[2])
    ax.set_title(f"Rolling 1-year Sharpe ratio: {label}")
    return _save(fig, out)


def cost_sensitivity(capacity: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(capacity["cost_bps"], capacity["sharpe"], marker="o", color=PALETTE[1])
    ax.axhline(0, color=NEG, linewidth=1, linestyle="--")
    zero = capacity[capacity["sharpe"] <= 0]
    if not zero.empty:
        ax.axvline(zero["cost_bps"].iloc[0], color=NEG, linewidth=1, linestyle=":")
        ax.text(
            zero["cost_bps"].iloc[0], ax.get_ylim()[1] * 0.9,
            f"  break-even ~{zero['cost_bps'].iloc[0]:.0f} bp", color=NEG, fontsize=9,
        )
    ax.set_title("Sharpe ratio versus assumed one-way trading cost")
    ax.set_xlabel("Cost (basis points per unit turnover)")
    ax.set_ylabel("Annualised Sharpe")
    return _save(fig, out)


def monthly_heatmap(table: pd.DataFrame, out: Path, label: str) -> Path:
    cols = [c for c in table.columns if c != "Year"]
    data = table[cols].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(table) + 2))
    lim = np.nanpercentile(np.abs(data), 97) or 1
    im = ax.imshow(data, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="%")
    ax.set_title(f"Monthly returns: {label}")
    return _save(fig, out)


def fold_stability(fold_metrics: pd.DataFrame, out: Path) -> Path:
    piv = fold_metrics.pivot_table(index="fold", columns="model", values="rank_mean_ic", observed=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, col in enumerate(piv.columns):
        ax.plot(piv.index, piv[col].to_numpy(), marker="o", label=col, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("Information coefficient by walk-forward fold")
    ax.set_xlabel("Fold (chronological)")
    ax.set_ylabel("Mean IC")
    ax.legend(ncol=3)
    return _save(fig, out)


def volatility_comparison(vol: pd.DataFrame, out: Path) -> Path:
    cols = [c for c in vol.columns if c.startswith("qlike_")]
    if not cols:
        return out
    means = vol[cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(
        [c.replace("qlike_", "").upper() for c in means.index],
        means.to_numpy(),
        color=[PALETTE[i % len(PALETTE)] for i in range(len(means))],
        width=0.55,
    )
    for i, v in enumerate(means.to_numpy()):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("Volatility forecast accuracy (QLIKE, lower is better)")
    ax.set_ylabel("Mean QLIKE across names")
    return _save(fig, out)


def factor_exposures(attribution: dict, out: Path) -> Path:
    betas = {
        k.replace("beta_", "").upper(): v
        for k, v in attribution.items()
        if k.startswith("beta_") and k != "beta_alpha"
    }
    if not betas:
        return out
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    keys = list(betas.keys())
    vals = [betas[k] for k in keys]
    ax.bar(keys, vals, color=[POS if v > 0 else NEG for v in vals], width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("Fama-French factor loadings of the strategy")
    ax.set_ylabel("Beta")
    return _save(fig, out)
