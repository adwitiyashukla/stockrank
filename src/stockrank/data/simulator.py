"""Synthetic equity market simulator with a known ground-truth alpha structure.

Why this exists
---------------
Two reasons, both practical.

1. **Reproducibility.** Anyone who clones this repo can run the entire pipeline
   with no API key, no rate limits and no network access, and get byte-identical
   numbers. That makes CI meaningful and makes the results in ``RESULTS.md``
   verifiable by a reader.

2. **A test bed with ground truth.** Real markets do not tell you what the true
   predictable component was, so you can never prove that a research pipeline is
   free of look-ahead bias - you can only fail to find bias. Here the alpha is
   planted by construction, so two things become checkable:

   * with alpha planted, the pipeline must *recover* an information coefficient
     close to the theoretical ceiling;
   * with ``null_alpha: true`` the true predictable component is exactly zero, so
     any out-of-sample information coefficient that is reliably non-zero is proof
     of leakage somewhere in the code.

The second experiment is the single most useful test in this repository.

Generative model
----------------
Daily log returns follow a three-component factor structure::

    r[i,t] = alpha[i,t] + beta[i] * r_mkt[t] + gamma[i] * r_sector[s(i),t] + e[i,t]

where the market variance follows a GARCH(1,1) recursion with Student-t
innovations (volatility clustering plus fat tails), idiosyncratic variances
follow their own per-asset GARCH(1,1), and the planted alpha is a linear
combination of three *lagged, observable* characteristics::

    alpha[i,t] = (k_mom * z(mom12_1)[i,t-1]
                  + k_rev * z(-ret5)[i,t-1]
                  + k_lv  * z(-vol63)[i,t-1]) * sigma_idio[i]

Because every characteristic is computed strictly from information available at
``t-1``, the planted signal is genuinely predictable rather than contemporaneous.
The strength coefficients are parameterised so that ``k_mom = 0.02`` means the
momentum characteristic on its own carries an expected daily cross-sectional
information coefficient of roughly 0.02, which is in the range reported for real
equity factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from stockrank.config import DataConfig, SimulatorConfig
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252
_CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
_VOWELS = "AEIOU"

SECTOR_NAMES = [
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer",
    "Industrials",
    "Energy",
    "Utilities",
    "Materials",
    "RealEstate",
    "Communication",
]


def _trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Business days minus US federal holidays: a close stand-in for the NYSE calendar."""
    days = pd.bdate_range(start=start, end=end)
    holidays = USFederalHolidayCalendar().holidays(start=days.min(), end=days.max())
    return days.difference(pd.DatetimeIndex(holidays))


def _make_tickers(n: int, rng: np.random.Generator) -> list[str]:
    """Deterministic pronounceable 4-letter tickers, guaranteed unique."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        t = (
            _CONSONANTS[rng.integers(len(_CONSONANTS))]
            + _VOWELS[rng.integers(len(_VOWELS))]
            + _CONSONANTS[rng.integers(len(_CONSONANTS))]
            + _VOWELS[rng.integers(len(_VOWELS))]
        )
        if t not in seen:
            seen.add(t)
            out.append(t)
    return sorted(out)


def _standardised_t(rng: np.random.Generator, df: int, size) -> np.ndarray:
    """Student-t draws rescaled to unit variance so vol targeting stays interpretable."""
    raw = rng.standard_t(df, size=size)
    return raw / np.sqrt(df / (df - 2.0))


def _xs_zscore(v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score over the currently listed names only."""
    out = np.zeros_like(v)
    if valid.sum() < 5:
        return out
    x = v[valid]
    finite = np.isfinite(x)
    if finite.sum() < 5:
        return out
    mu = x[finite].mean()
    sd = x[finite].std()
    if sd <= 0 or not np.isfinite(sd):
        return out
    z = np.zeros_like(x)
    z[finite] = np.clip((x[finite] - mu) / sd, -3.0, 3.0)
    out[valid] = z
    return out


class MarketSimulator:
    """Generate a realistic unbalanced panel of daily equity bars."""

    def __init__(self, data_cfg: DataConfig, sim_cfg: SimulatorConfig, seed: int = 42) -> None:
        self.data_cfg = data_cfg
        self.sim = sim_cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ setup
    def _asset_parameters(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        beta = np.clip(rng.normal(1.0, 0.30, n), 0.25, 2.20)
        sector_load = np.clip(rng.normal(0.85, 0.25, n), 0.10, 1.80)
        # Idiosyncratic vol is lognormal: a few names are much noisier than the median.
        idio_ann = self.sim.ann_idio_vol * np.exp(rng.normal(0.0, 0.35, n) - 0.5 * 0.35**2)
        idio_ann = np.clip(idio_ann, 0.08, 1.20)
        return {
            "beta": beta,
            "sector_load": sector_load,
            "idio_daily": idio_ann / np.sqrt(TRADING_DAYS),
            "p0": np.exp(rng.uniform(np.log(12.0), np.log(320.0), n)),
            "base_volume": np.exp(rng.normal(14.0, 1.1, n)),
        }

    # ------------------------------------------------------------- generation
    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(panel, market)``.

        ``panel`` is tidy long format with one row per (date, ticker).
        ``market`` carries the market factor return and the risk-free rate.
        """
        rng = self.rng
        dates = _trading_calendar(self.data_cfg.start, self.data_cfg.end)
        t_n = len(dates)
        n = int(self.data_cfg.n_assets)
        if t_n < 300:
            raise ValueError("Simulation needs at least ~300 trading days of history")

        tickers = _make_tickers(n, rng)
        n_sectors = min(self.sim.n_sectors, len(SECTOR_NAMES))
        sector_idx = rng.integers(0, n_sectors, n)
        params = self._asset_parameters(n)

        # Staggered listings: ~25% of names appear after the sample starts, which
        # forces every downstream stage to cope with an unbalanced panel.
        listing = np.zeros(n, dtype=int)
        late = rng.random(n) < 0.25
        listing[late] = rng.integers(1, max(2, int(t_n * 0.45)), late.sum())

        a, b = self.sim.garch_alpha, self.sim.garch_beta
        df_t = self.sim.student_t_df

        # --- market factor: GARCH(1,1) with Student-t shocks ------------------
        mkt_var_uncond = (self.sim.ann_market_vol**2) / TRADING_DAYS
        omega_m = mkt_var_uncond * (1.0 - a - b)
        mkt_var = np.empty(t_n)
        mkt_ret = np.empty(t_n)
        mkt_var[0] = mkt_var_uncond
        mkt_drift = 0.07 / TRADING_DAYS
        z_m = _standardised_t(rng, df_t, t_n)
        for t in range(t_n):
            if t > 0:
                mkt_var[t] = omega_m + a * (mkt_ret[t - 1] - mkt_drift) ** 2 + b * mkt_var[t - 1]
            mkt_ret[t] = mkt_drift + np.sqrt(mkt_var[t]) * z_m[t]

        # --- sector factors: constant-vol, mildly autocorrelated --------------
        sec_vol = 0.09 / np.sqrt(TRADING_DAYS)
        sec_ret = rng.normal(0.0, sec_vol, (t_n, n_sectors))
        for t in range(1, t_n):
            sec_ret[t] += 0.05 * sec_ret[t - 1]

        # --- idiosyncratic GARCH state ---------------------------------------
        idio_var_uncond = params["idio_daily"] ** 2
        omega_i = idio_var_uncond * (1.0 - a - b)
        idio_var = idio_var_uncond.copy()
        z_i = _standardised_t(rng, df_t, (t_n, n))

        strengths = (
            (0.0, 0.0, 0.0)
            if self.sim.null_alpha
            else (self.sim.momentum_strength, self.sim.reversal_strength, self.sim.lowvol_strength)
        )
        k_mom, k_rev, k_lv = strengths

        log_ret = np.zeros((t_n, n))
        cum = np.zeros((t_n, n))
        vol_state = np.zeros((t_n, n))
        alpha_true = np.zeros((t_n, n))

        for t in range(t_n):
            listed = listing <= t

            # Characteristics use information through t-1 only.
            if t >= 253:
                mom = cum[t - 22] - cum[t - 253]  # classic 12-1 momentum
                rev = -(cum[t - 1] - cum[t - 6])  # 5-day short-term reversal
                lv = -log_ret[t - 64 : t - 1].std(axis=0)  # low-volatility tilt
                a_t = (
                    k_mom * _xs_zscore(mom, listed)
                    + k_rev * _xs_zscore(rev, listed)
                    + k_lv * _xs_zscore(lv, listed)
                ) * params["idio_daily"]
            else:
                a_t = np.zeros(n)
            alpha_true[t] = a_t

            if t > 0:
                shock = log_ret[t - 1] - alpha_true[t - 1]
                idio_var = omega_i + a * shock**2 + b * idio_var
                idio_var = np.clip(idio_var, 1e-10, None)
            sd_i = np.sqrt(idio_var)
            vol_state[t] = sd_i

            r_t = (
                a_t
                + params["beta"] * mkt_ret[t]
                + params["sector_load"] * sec_ret[t, sector_idx]
                + sd_i * z_i[t]
            )
            r_t = np.where(listed, r_t, 0.0)
            log_ret[t] = r_t
            cum[t] = (cum[t - 1] if t > 0 else 0.0) + r_t

        close = params["p0"] * np.exp(cum)

        # --- OHLC and volume around the simulated closes ----------------------
        total_sd = np.sqrt(
            (params["beta"] * self.sim.ann_market_vol / np.sqrt(TRADING_DAYS)) ** 2 + vol_state**2
        )
        gap = rng.normal(0.0, 0.35, (t_n, n)) * total_sd
        prev_close = np.vstack([close[0:1], close[:-1]])
        open_ = prev_close * np.exp(gap)
        up = np.abs(rng.normal(0.0, 1.0, (t_n, n))) * total_sd * 0.8
        dn = np.abs(rng.normal(0.0, 1.0, (t_n, n))) * total_sd * 0.8
        high = np.maximum(open_, close) * np.exp(up)
        low = np.minimum(open_, close) * np.exp(-dn)
        std_move = np.abs(log_ret) / np.maximum(total_sd, 1e-8)
        volume = params["base_volume"] * np.exp(
            0.45 * rng.normal(0.0, 1.0, (t_n, n)) + 0.55 * np.clip(std_move, 0, 6)
        )

        listed_mask = listing[None, :] <= np.arange(t_n)[:, None]

        panel = pd.DataFrame(
            {
                "date": np.repeat(dates.to_numpy(), n),
                "ticker": np.tile(np.array(tickers), t_n),
                "sector": np.tile(np.array(SECTOR_NAMES[:n_sectors])[sector_idx], t_n),
                "open": open_.ravel(),
                "high": high.ravel(),
                "low": low.ravel(),
                "close": close.ravel(),
                "volume": volume.ravel(),
                "alpha_true": alpha_true.ravel(),
            }
        )
        panel = panel.loc[listed_mask.ravel()].reset_index(drop=True)
        panel["date"] = pd.to_datetime(panel["date"])

        market = pd.DataFrame(
            {
                "date": dates,
                "mkt_return": mkt_ret,
                "mkt_vol": np.sqrt(mkt_var),
                "rf_rate": 0.02 / TRADING_DAYS,
            }
        )

        logger.info(
            "Simulated %d assets x %d days (%d rows), null_alpha=%s",
            n,
            t_n,
            len(panel),
            self.sim.null_alpha,
        )
        return panel, market


def simulate_market(
    data_cfg: DataConfig, sim_cfg: SimulatorConfig, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return MarketSimulator(data_cfg, sim_cfg, seed=seed).generate()
