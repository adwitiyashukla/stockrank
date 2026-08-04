"""Package a completed run as a small, committable demo bundle.

The Streamlit console reads artifacts from disk. To deploy it on Streamlit
Community Cloud, those artifacts have to live in the repository, so this script
copies one run into ``demo_artifacts/`` and trims the only large file
(``predictions.parquet``) to a recent window. Everything else is a few kilobytes.

    python scripts/prepare_demo.py --run baseline --tail-days 900
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COPY_AS_IS = (
    "config.yaml", "data_summary.json", "feature_summary.json", "performance.json",
    "significance.json", "attribution.json", "timings.json", "folds.json",
    "model_metrics.csv", "fold_metrics.csv", "feature_importance.csv",
    "quantile_ladders.csv", "volatility_comparison.csv", "monthly_returns.csv",
    "shap_summary.csv", "RESULTS.md",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="demo_artifacts")
    ap.add_argument("--tail-days", type=int, default=900,
                    help="how many recent trading days of predictions to keep")
    args = ap.parse_args()

    src = Path(args.artifacts) / args.run
    if not src.exists():
        print(f"No run at {src}. Run the pipeline first.")
        return 1
    dst = Path(args.out) / args.run
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in COPY_AS_IS:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
            copied += 1

    for p in sorted(src.glob("backtest_*.parquet")):
        shutil.copy2(p, dst / p.name)
        copied += 1

    # The serialised production model, so a clone can hit /score and /features
    # without first running the pipeline.
    for pattern in ("model_*.joblib", "model_*.json"):
        for p in sorted(src.glob(pattern)):
            shutil.copy2(p, dst / p.name)
            copied += 1

    pred_p = src / "predictions.parquet"
    if pred_p.exists():
        df = pd.read_parquet(pred_p)
        df["date"] = pd.to_datetime(df["date"])
        if args.tail_days > 0:
            cutoff = df["date"].drop_duplicates().nlargest(args.tail_days).min()
            df = df[df["date"] >= cutoff]
        df.to_parquet(dst / "predictions.parquet", index=False)
        copied += 1
        print(f"predictions trimmed to {len(df):,} rows from {df['date'].min().date()}")

    total_mb = sum(f.stat().st_size for f in dst.rglob("*")) / 1e6
    print(f"Copied {copied} files to {dst} ({total_mb:.1f} MB)")
    if total_mb > 45:
        print("WARNING: bundle is large for a git repository; reduce --tail-days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
