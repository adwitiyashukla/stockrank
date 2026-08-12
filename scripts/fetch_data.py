from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockrank.config import load_config
from stockrank.data.loader import load_market_data
from stockrank.utils.logging import get_logger, setup_logging

logger = get_logger("fetch_data")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and cache the market panel")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-cache", action="store_true", help="ignore any cached panel")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    logger.info(
        "Source=%s universe=%s window=%s..%s max_assets=%d",
        cfg.data.source,
        cfg.data.universe,
        cfg.data.start,
        cfg.data.end,
        cfg.data.max_assets,
    )

    t0 = time.time()
    data = load_market_data(cfg, cache=not args.no_cache)
    elapsed = time.time() - t0

    summary = data.summary()
    summary["elapsed_seconds"] = round(elapsed, 1)
    out = Path(cfg.data.cache_dir) / f"data_summary_{cfg.run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("DATA SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        if k == "download":
            print(f"{'download':<32}")
            for kk, vv in v.items():
                if kk == "failed_tickers":
                    print(f"    {kk:<28} {len(vv)} shown: {vv[:12]}")
                else:
                    print(f"    {kk:<28} {vv}")
        else:
            print(f"{k:<32} {v}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
