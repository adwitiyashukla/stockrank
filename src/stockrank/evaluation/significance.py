from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _per_period_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if r.empty or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    if n_trials < 2 or not np.isfinite(sharpe_variance) or sharpe_variance <= 0:
        return 0.0
    gamma = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int = 1, sharpe_variance: float | None = None
) -> dict[str, float]:
    r = pd.Series(returns).dropna()
    t = len(r)
    if t < 60:
        return {"sharpe_per_period": np.nan, "sr_threshold": np.nan, "deflated_sharpe": np.nan}

    sr = _per_period_sharpe(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if sharpe_variance is None:
        sharpe_variance = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (t - 1)

    sr_star = expected_max_sharpe(n_trials, sharpe_variance)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_star) * np.sqrt(t - 1) / denom
    return {
        "sharpe_per_period": sr,
        "sharpe_annual": sr * np.sqrt(252),
        "sr_threshold": sr_star,
        "sr_threshold_annual": sr_star * np.sqrt(252),
        "deflated_sharpe": float(stats.norm.cdf(z)),
        "n_trials": int(n_trials),
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": int(t),
    }


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame, n_splits: int = 8
) -> dict[str, float]:
    R = returns_matrix.dropna(how="all").fillna(0.0)
    n_strategies = R.shape[1]
    if n_strategies < 2 or len(R) < n_splits * 20:
        return {"pbo": np.nan, "n_combinations": 0, "n_strategies": n_strategies}

    if n_splits % 2 == 1:
        n_splits -= 1
    blocks = np.array_split(np.arange(len(R)), n_splits)
    half = n_splits // 2

    logits: list[float] = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_sr = R.iloc[is_idx].apply(_per_period_sharpe)
        oos_sr = R.iloc[oos_idx].apply(_per_period_sharpe)
        if is_sr.isna().all() or oos_sr.isna().all():
            continue

        best = is_sr.idxmax()
        rank = oos_sr.rank(pct=True).get(best, np.nan)
        if not np.isfinite(rank):
            continue
        rank = float(np.clip(rank, 1e-6, 1 - 1e-6))
        logits.append(np.log(rank / (1 - rank)))

    if not logits:
        return {"pbo": np.nan, "n_combinations": 0, "n_strategies": n_strategies}

    arr = np.asarray(logits)
    return {
        "pbo": float((arr <= 0).mean()),
        "median_oos_rank": float(1 / (1 + np.exp(-np.median(arr)))),
        "n_combinations": int(arr.size),
        "n_strategies": int(n_strategies),
    }


def stationary_bootstrap_sharpe(
    returns: pd.Series, n_boot: int = 1000, mean_block: int = 20, seed: int = 0
) -> dict[str, float]:
    r = pd.Series(returns).dropna().to_numpy()
    n = r.size
    if n < 100:
        return {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan, "p_value_sharpe_le_0": np.nan}

    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1)

    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for k in range(1, n):
        idx[:, k] = np.where(new_block[:, k], starts[:, k], (idx[:, k - 1] + 1) % n)

    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sd > 0, samples.mean(axis=1) / sd * np.sqrt(252), np.nan)

    out = out[np.isfinite(out)]
    if out.size == 0:
        return {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan, "p_value_sharpe_le_0": np.nan}
    return {
        "sharpe_ci_low": float(np.percentile(out, 2.5)),
        "sharpe_ci_high": float(np.percentile(out, 97.5)),
        "p_value_sharpe_le_0": float((out <= 0).mean()),
        "n_bootstrap": int(out.size),
    }


def randomisation_test(
    predictions: pd.DataFrame, score_col: str, n_permutations: int = 200, seed: int = 0
) -> dict[str, float]:
    from stockrank.evaluation.metrics import matrix_ic

    rng = np.random.default_rng(seed)
    df = predictions[["date", "ticker", score_col, "target"]].dropna()
    pred_w = df.pivot_table(index="date", columns="ticker", values=score_col, observed=True)
    targ_w = df.pivot_table(index="date", columns="ticker", values="target", observed=True)
    actual = float(matrix_ic(pred_w, targ_w).mean())

    values = pred_w.to_numpy()
    valid = np.isfinite(values)
    beats = 0
    for _ in range(n_permutations):
        keys = np.where(valid, rng.random(values.shape), np.inf)
        order = np.argsort(keys, axis=1)
        shuffled = np.take_along_axis(values, order, axis=1)
        shuffled = np.where(valid, shuffled, np.nan)
        ic = float(matrix_ic(pd.DataFrame(shuffled, index=pred_w.index, columns=pred_w.columns), targ_w).mean())
        if ic >= actual:
            beats += 1
    return {
        "actual_ic": actual,
        "permutation_p_value": (beats + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
    }
