# Methodology

This document records the design decisions behind the engine, including the ones
that did not work. A research repository that only shows the path that succeeded
is not reproducible research, it is marketing.

---

## 1. The problem being solved

The engine forecasts the **cross-section** of equity returns, not the level of any
individual price. The distinction is fundamental and most retail stock-prediction
projects get it wrong.

Predicting tomorrow's price of AAPL is nearly trivial and completely useless: the
best forecast is today's price, and a model trained on price levels achieves an
R-squared above 0.99 while containing zero information. What is hard and valuable
is ranking: given 250 stocks today, which will outperform the others over the next
month?

That reframing changes everything downstream:

| | Price-level prediction | Cross-sectional ranking |
|---|---|---|
| Target | `close[t+1]` | forward return relative to the cross-section |
| Metric | RMSE, R-squared, MAPE | information coefficient, quantile spread |
| Typical reported score | R-squared > 0.99 | IC of 0.02 to 0.05 |
| Economic value | none | a tradable long/short book |

---

## 2. Data

**Source.** Daily split- and dividend-adjusted OHLCV bars from Yahoo Finance, plus
the Fama-French five factors and momentum from the Ken French Data Library. Both
are free and both are what the underlying series actually are, rather than a
synthetic stand-in.

**Universe: point-in-time S&P 500 membership.** Taking today's index members and
running them back sixteen years is the most common single error in retail
backtests. Every company in that list survived, so the backtest is told in advance
which firms would not go bankrupt or be acquired. Momentum strategies in
particular look far better than they are.

The engine reconstructs membership historically. It scrapes the current
constituent table and the record of index additions and removals, then walks
backwards through the change log undoing each event, which recovers the member set
on any past date. Over 2010 to 2026 this expands the universe from 503 current
names to **813 distinct historical members**, including firms such as Alcoa,
Aetna, Monsanto and First Republic that are no longer in the index.

**Residual limitation, stated plainly.** Yahoo Finance does not serve price
history for most delisted or acquired securities, so roughly 20% of the
historically correct membership cannot be priced. The remaining sample is
therefore still mildly favourable. The exact coverage ratio is computed on every
run and reported in `RESULTS.md`. Eliminating this entirely requires a paid
survivorship-bias-free database such as CRSP.

**Cleaning.** Rows with non-positive prices are dropped. Names showing repeated
single-day moves above 100% are removed, since these almost always indicate an
unadjusted corporate action rather than a real event. A $5 median price floor
removes names where the bid-ask spread would swamp any signal, and a 500-day
history floor removes listings too short to compute a 252-day feature. Finally the
universe is capped at the most liquid 250 names by median dollar volume, which
serves two purposes: it keeps memory inside an 8 GB budget, and it restricts the
strategy to securities that could actually absorb the trades.

---

## 3. Features

36 factors are built from price and volume only, in six families:

- **Momentum**: cumulative returns over 1, 5, 10, 21, 63, 126 and 252 days, plus
  the classic 12-1 spread that skips the most recent month to avoid contaminating
  medium-term momentum with short-term reversal.
- **Volatility**: realised volatility over 10, 21 and 63 days, the ratio between
  short and long windows, the Parkinson high-low range estimator, downside
  deviation, skewness and kurtosis.
- **Trend**: price relative to 10, 21, 50 and 200-day moving averages, a moving
  average crossover, distance from the 52-week high and low, and a Bollinger
  z-score.
- **Technical**: Wilder's RSI and normalised MACD level and histogram.
- **Liquidity**: log dollar volume, its short-versus-long trend, the volume ratio,
  and the Amihud illiquidity measure (average absolute return per dollar traded).
- **Market-relative**: rolling 63-day CAPM beta, residual alpha, idiosyncratic
  volatility and correlation to the market.

**Cross-sectional normalisation.** Every feature is winsorised at the 1st and 99th
percentiles *within each date* and then z-scored *within each date*. This is not
cosmetic. Realised volatility in March 2020 was three times its 2017 level for
almost every name simultaneously; a model fed raw levels spends its capacity
learning the level of the market rather than the relative attractiveness of
individual names. Normalising within the date removes the common component,
renders the inputs stationary essentially for free, and matches what a
dollar-neutral book actually trades.

**Causality.** All feature code lives in `features/technical.py` and contains no
negative shift anywhere. This is enforced by a test that rebuilds every feature on
a panel whose final 60 days have been multiplied by three, and asserts that
nothing stamped before the corruption changed.

---

## 4. The target, and two targets that were rejected

The target is the **cross-sectionally demeaned forward return over 21 trading
days, entered one day after the signal**:

```
fwd[t]    = close[t + 1 + 21] / close[t + 1] - 1
target[t] = fwd[t] - mean_over_names(fwd[t])
```

The one-day execution lag is deliberate insurance. A signal computed from the
close of day `t` cannot realistically be executed at that same close, and building
the lag into the label costs a little performance rather than manufacturing it.

**Horizon chosen empirically.** `scripts/factor_diagnostics.py` screens every
feature at horizons of 1, 5, 10, 21 and 63 days. Mean absolute information
coefficient across features rises monotonically with horizon:

| horizon | mean abs IC | max abs IC |
|---|---|---|
| 1d | 0.0050 | 0.0173 |
| 5d | 0.0079 | 0.0190 |
| 10d | 0.0092 | 0.0248 |
| 21d | 0.0119 | 0.0357 |
| 63d | 0.0179 | 0.0632 |

21 days was chosen over 63 because it yields roughly three times as many
independent observations for the same sample, which matters more for statistical
confidence than the extra raw IC. Note that choosing a horizon by looking at this
table is itself a selection decision, and the number of configurations screened is
fed into the deflated Sharpe ratio.

### Rejected target 1: beta-adjusted residual return

The natural target for a market-neutral book seems to be the residual after
removing each name's beta times the market:

```
target[t] = fwd[t] - beta_hat[t] * fwd_market[t]
```

On the real data this target produced the strongest information coefficients in
the whole study: `beta_63` reached an IC of **-0.084** with t = -4.1 at the 63-day
horizon. That result is spurious.

Rolling beta is estimated with error. Writing the true relation as
`fwd = alpha + beta*fwd_mkt + e` and the estimate as `beta_hat = beta + u`, the
constructed target becomes `alpha - u*fwd_mkt + e`. Because estimation error `u`
is positively correlated with `beta_hat` by construction (regression to the mean),
and because the market rose over most of the sample, the target is mechanically
and negatively correlated with the estimated beta. A model happily learns to
short high-beta names, reports a large IC, and earns nothing, because the
portfolio construction stage neutralises exactly that exposure anyway.

The fix is to keep the target clean and impose neutrality on the **book** instead.

### Rejected target 2: volatility-scaled return

Dividing the forward return by trailing volatility is standard practice for
producing a risk-adjusted target. It also produced impressive numbers:
`downside_vol` reached an IC of -0.054 with t = -3.5.

Also spurious, and for a simpler reason. If the target is `fwd / vol_63`, then any
volatility feature is mechanically anti-correlated with the target through the
denominator. The measured IC rises without any tradable information being added.

Both experiments are reproducible with `scripts/factor_diagnostics.py`.

---

## 5. Validation

**The problem.** A 21-day label formed on the first of the month is still being
realised at month end. Ordinary k-fold cross-validation, and even naive
chronological splitting, place overlapping observations on both sides of the
train/test boundary, so the two sets share the same price path and every metric
downstream is inflated.

**The scheme.** Purged walk-forward validation with an embargo, following Lopez de
Prado:

- Six folds, each training on a rolling five-year window and testing on the
  following year. Training always precedes testing.
- **Purging**: training rows whose label window reaches into the test window are
  removed, a gap of `horizon + 1` = 22 trading days.
- **Embargo**: a further 21 trading days are dropped, because serial correlation
  in the features means observations immediately adjacent to the test period still
  carry information about it.

Early stopping for LightGBM and the neural network uses the most recent slice of
each training window, which is still strictly inside the fold and never touches
test data.

Purged k-fold is also implemented for hyperparameter search inside a fold, but it
is never used for headline numbers because it trains on data that post-dates some
of its own test observations.

---

## 6. Models

| Model | Rationale |
|---|---|
| **Ridge** | The benchmark that must be beaten. With collinear factor exposures and a signal-to-noise ratio around 2%, shrinkage beats flexibility, and this is the regime where linear models are hard to improve on. |
| **ElasticNet** | Adds sparsity, which answers whether a subset of the 36 features carries everything. |
| **LightGBM** | Huber objective, because forward returns have fat tails and squared error lets a handful of earnings gaps dominate the fit. Large `min_child_samples` (200), because at this signal-to-noise ratio small leaves are noise-fitting machines. |
| **GRU** | Sees a 40-day sequence of factor exposures per name rather than a single row, so it can in principle learn paths: momentum that is accelerating versus momentum that is rolling over. Whether the extra capacity pays for itself is an empirical question the repository answers rather than assumes. |
| **Transformer** | Small encoder with learned positional embeddings, available as an alternative sequence head. |
| **Ensemble** | Rank average of members. The members produce scores on incomparable scales, and only the ordering is traded, so converting each to a cross-sectional rank before averaging is both scale free and outlier proof. |

**Memory design for the sequence models.** Materialising every (ticker, window)
pair explicitly would need roughly 30 GB. Instead the features live once in a
dense `(dates, tickers, features)` tensor of about 146 MB, and each training
sample is a view into it assembled at batch time. That is what makes a sequence
model trainable on an 8 GB laptop.

---

## 7. Portfolio construction

A prediction is not a portfolio. Three stages sit in between and each destroys
some of the paper alpha.

**Sizing.** Rank-based long/short: equal weight the top 25 and bottom 25 names.
This is what an IC of 0.02 to 0.05 actually justifies. The model is trusted to
order names, not to say by how much. Mean-variance with Ledoit-Wolf shrinkage and
risk parity are also implemented; unshrunk mean-variance over 250 names and 252
days puts its entire risk budget into the noisiest eigenvector of a
near-singular covariance matrix.

**Beta neutrality.** This is the most important constraint in the stack and the
easiest to get wrong. **Equal dollars long and short does not give a market-neutral
book.** Price-based signals routinely prefer low-volatility names on the long side
and high-beta names on the short side, which leaves the portfolio carrying a large
negative beta. In a rising market that loses money no matter how good the stock
selection is, and an early version of this engine lost 28% a year for exactly that
reason while its information coefficient was positive.

The fix is a projection. Writing `b` for the vector of trailing betas,

```
w_neutral = w - (w . b / b . b) * b
```

is the orthogonal projection onto the zero-net-beta subspace, so it removes market
exposure while disturbing the intended cross-sectional tilt as little as possible.
Realised net beta is recorded on every rebalance and reported.

**Volatility targeting.** Gross exposure is scaled by the ratio of a 10% annual
target to trailing realised volatility, using only past data and capped at 2x in
either direction so the strategy cannot lever into a calm period and get caught by
a regime change.

---

## 8. Costs

Ignoring costs is how backtests lie. Charged explicitly:

- **5 bp commission plus 2 bp slippage** per unit of one-way turnover, applied to
  the change in weights rather than to gross exposure.
- **50 bp annual borrow** on the short leg, every day it is held.
- **A turnover cap**, which blends the target book towards the current book when
  the model wants to trade more than liquidity allows.

The report includes a **break-even cost curve**: net Sharpe as a function of the
assumed one-way cost. This is the single most informative number about whether a
signal is implementable. A strategy that needs sub-3bp execution is a
high-frequency shop's problem, not a research result.

**Not modelled**: market impact beyond linear slippage, auction risk at the close,
borrow availability constraints, and financing spreads that vary by name.

### Daily marking, and a bug worth describing

The first version of the engine booked one lump return per 21-day holding period
and spread it evenly across the days of that period. Every statistic downstream
was computed on the resulting daily series, and the reported Sharpe ratios were
absurd: LightGBM showed 4.58 while its information coefficient was 0.013, two
numbers that cannot both be true.

The cause is that spreading one period return across 21 identical days produces a
series with almost no day-to-day variation. Its standard deviation is the standard
deviation of the period returns divided by 21, but annualising multiplies by the
square root of 252 as though those days were independent observations. The Sharpe
is therefore inflated by roughly the square root of the holding period, which for
21 days is 4.58. Dividing the reported 4.58 by 4.58 gives 1.0, and that is exactly
what the corrected engine produces.

The engine now marks the book to market **daily**: weights are held fixed across
the period and each day's profit and loss is computed from the actual daily
returns of the holdings. Volatility, drawdowns and the Sharpe ratio are all
computed on genuinely independent daily observations.
`tests/test_backtest.py::test_returns_are_not_artificially_smooth` fails if the
old behaviour ever returns.

This is recorded here rather than quietly fixed because the failure mode is
common, it is invisible unless the information coefficient and the Sharpe ratio
are checked against each other, and noticing the inconsistency is the whole
skill.

---

## 9. Is the result real?

A Sharpe ratio computed on the winning configuration of a search is not an
estimate of anything. Four independent checks are applied:

1. **Deflated Sharpe Ratio** (Bailey and Lopez de Prado). Adjusts the observed
   Sharpe for the number of trials, the non-normality of returns and the sample
   length. Reports the probability that the true Sharpe exceeds the
   selection-adjusted threshold.
2. **Probability of backtest overfitting** via combinatorially symmetric cross
   validation. Repeatedly splits the sample, picks the in-sample winner, and
   checks where it ranks out of sample. If the winner routinely lands in the
   bottom half, the selection process is fitting noise.
3. **Stationary bootstrap** confidence intervals for the Sharpe, resampling blocks
   rather than individual days so serial correlation survives the resampling.
4. **Randomisation test**: shuffle the forecasts within each date and recompute
   the IC. Distribution free, and assumes nothing about the shape of the return
   distribution.

Plus a **factor attribution** regression of daily strategy excess returns on the
Fama-French five factors and momentum with Newey-West standard errors. If the
intercept is indistinguishable from zero, the strategy is repackaged beta and can
be replicated with cheap index products.

---

## 10. The leakage control experiment

The tests that give the most confidence are the two that use the synthetic
simulator, which is present in this repository as a **test fixture, not as a
dataset**. No headline result is ever computed on synthetic data.

The simulator generates a panel with GARCH(1,1) volatility, Student-t innovations,
a market factor, sector factors, and a planted alpha built from lagged observable
characteristics. Crucially, it can plant **exactly zero** alpha.

- **Null control** (`configs/leakage_control.yaml`): with no planted signal, the
  true predictable component of the cross-section is zero. Any model that reports
  a reliably non-zero out-of-sample IC has found a bug. This is enforced by
  `tests/test_null_control.py`, which fails if any |t| exceeds 4.
- **Signal recovery** (`configs/signal_recovery.yaml`): with alpha planted, the
  pipeline must actually find it. This guards against the opposite failure, where
  a pipeline that reports zero everywhere passes the null test trivially.

Real markets never tell you what the true predictable component was, so on real
data you can only fail to find leakage; you can never prove its absence. The
simulator is the only place where that proof is available.

---

## 11. What is deliberately not here

- **No fundamentals.** Price and volume only. Earnings, revisions, short interest
  and options-implied data are the obvious next inputs and would likely matter
  more than any modelling change.
- **No intraday data.** Everything is daily close-to-close.
- **No regime switching or online learning.** Each fold refits from scratch.
- **No live trading.** The API serves forecasts and screens; it places no orders
  and has no broker integration.

---

## References

- Bailey, D. and Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio*.
- Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2017). *The Probability of Backtest Overfitting*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, chapters 4 to 7.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*.
- Amihud, Y. (2002). *Illiquidity and Stock Returns*.
- Fama, E. and French, K. (2015). *A Five-Factor Asset Pricing Model*.
- Frazzini, A. and Pedersen, L. (2014). *Betting Against Beta*.
- Ledoit, O. and Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*.
- Politis, D. and Romano, J. (1994). *The Stationary Bootstrap*.
