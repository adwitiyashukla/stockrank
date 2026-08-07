"""Which features carry signal, at which horizon, on the real data?

Run this before committing to a modelling configuration. It computes the
univariate information coefficient of every feature against forward returns at
several horizons, with Newey-West t-statistics that account for the overlap
induced by multi-day labels. Choosing a horizon by looking at this table is
itself a form of selection, so the number of horizons tested is reported and fed
into the deflated Sharpe ratio downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockrank.config import load_config  # noqa: E402
from stockrank.data.loader import load_market_data  # noqa: E402
from stockrank.evaluation.metrics import ic_summary, matrix_ic  # noqa: E402
from stockrank.features.cross_section import normalise  # noqa: E402
from stockrank.features.labels import (  # noqa: E402
    beta_adjusted_forward_return,
    cross_sectional_demean,
    forward_return,
)
from stockrank.features.technical import build_feature_matrices  # noqa: E402
from stockrank.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("diagnostics")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--horizons", default="1,5,10,21,63")
    ap.add_argument("--out", default="reports/factor_diagnostics.csv")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    md = load_market_data(cfg)

    close = md.panel.pivot(index="date", columns="ticker", values="close").astype("float64").sort_index()
    high = md.panel.pivot(index="date", columns="ticker", values="high").astype("float64").sort_index()
    low = md.panel.pivot(index="date", columns="ticker", values="low").astype("float64").sort_index()
    vol = md.panel.pivot(index="date", columns="ticker", values="volume").astype("float64").sort_index()
    dv = md.panel.pivot(index="date", columns="ticker", values="dollar_volume").astype("float64").sort_index()
    mkt = md.market.set_index("date")["mkt_return"].reindex(close.index).ffill().fillna(0.0)

    raw = build_feature_matrices(close, high, low, vol, dv, mkt, cfg.features)
    norm = {k: normalise(v, "zscore", cfg.features.winsorize_q) for k, v in raw.items()}
    realised_vol = raw["vol_63"].replace(0.0, np.nan)

    horizons = [int(h) for h in args.horizons.split(",")]
    rows = []
    for h in horizons:
        fwd = forward_return(close, h, lag=1)
        fwd_mkt = (1 + mkt).rolling(h).apply(np.prod, raw=True).shift(-(1 + h)) - 1
        targets = {
            "excess": cross_sectional_demean(fwd),
            "vol_scaled": cross_sectional_demean(fwd / realised_vol),
            # The target the strategy actually trades: residual return after each
            # name's beta times the market is removed, then cross-sectionally
            # demeaned. Consistent with a dollar- and beta-neutral book.
            "beta_neutral": cross_sectional_demean(
                beta_adjusted_forward_return(fwd, fwd_mkt.reindex(close.index), raw["beta_63"])
            ),
        }
        for tname, tgt in targets.items():
            for fname, fmat in norm.items():
                ic = matrix_ic(fmat, tgt)
                if len(ic) < 200:
                    continue
                s_ = ic_summary(ic, h)
                rows.append(
                    {
                        "horizon": h, "target": tname, "feature": fname,
                        "mean_ic": s_["mean_ic"], "icir": s_["icir"],
                        "t_nw": s_["t_stat_nw"], "hit_rate": s_["hit_rate"], "n_days": s_["n_obs"],
                    }
                )
        logger.info("Horizon %d done", h)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 96)
    print("TOP 15 FEATURES BY |t| (Newey-West), per horizon and target")
    print("=" * 96)
    for h in horizons:
        for tname in ("excess", "vol_scaled", "beta_neutral"):
            sub = out[(out.horizon == h) & (out.target == tname)].copy()
            if sub.empty:
                continue
            sub["abs_t"] = sub["t_nw"].abs()
            top = sub.nlargest(8, "abs_t")[["feature", "mean_ic", "icir", "t_nw"]]
            print(f"\n--- horizon={h}d target={tname} ---")
            print(top.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "=" * 96)
    print("SUMMARY: mean |IC| and count of |t|>3 features by horizon/target")
    print("=" * 96)
    summ = out.assign(abs_t=out.t_nw.abs(), abs_ic=out.mean_ic.abs()).groupby(
        ["horizon", "target"], observed=True
    ).agg(mean_abs_ic=("abs_ic", "mean"), max_abs_ic=("abs_ic", "max"), n_sig=("abs_t", lambda s: int((s > 3).sum())))
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nFull table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
