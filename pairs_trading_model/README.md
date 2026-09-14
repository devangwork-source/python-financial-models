# Statistical Arbitrage: Cointegration-Based Pairs Trading

## Overview
Screens a same-sector universe of NSE stocks for pairs whose price spread is
statistically mean-reverting, trades the spread with a z-score rule, and
tests whether any in-sample edge survives out-of-sample — the same
train/test discipline used in the Portfolio Optimization project.

## Hypothesis
This project tests two things, not one:

1. **Economic hypothesis:** certain same-sector pairs share a long-run
   equilibrium — their price spread is stationary rather than a random walk
   — and that mean-reversion can be captured profitably, net of transaction
   costs, with the edge holding up on unseen (out-of-sample) data rather
   than being an artifact of the fitting window.
2. **Methodological hypothesis:** screening many candidate pairs and
   reporting whichever has the lowest cointegration p-value is a form of
   data snooping — some pairs will look "significant" by chance alone. The
   real claim isn't "this pair is cointegrated," it's "this pair is
   cointegrated *after* accounting for how many pairs were tested," which
   is what the Bonferroni-corrected significance flag checks for.

The second hypothesis matters as much as the first — a pairs-trading
project that skips it is really just p-hacking with extra steps.

## Methodology
1. **Universe:** same-sector candidates only (Banking, IT, Auto NSE stocks)
   — pairs are only tested within a sector, since cross-sector pairs have
   no fundamental reason to be cointegrated.
2. **Screening:** Engle-Granger cointegration test on log prices,
   **train period only** (2020-01-01 to 2023-12-31). Every tested pair is
   reported with its raw p-value and a Bonferroni-corrected significance
   flag.
3. **Hedge ratio:** OLS regression of log(A) on log(B), **train only**,
   then locked in and applied unchanged to the test period — the same
   "estimate on train, evaluate on unseen data" rule Project 1 used for
   portfolio weights.
4. **Half-life of mean reversion:** Ornstein-Uhlenbeck fit on the spread,
   used to sanity-check that the trading window (60-day rolling z-score) is
   a sensible multiple of how fast the spread actually reverts.
5. **Signals:** rolling, causal z-score of the spread (never uses future
   data). Enter at \|z\| > 2, exit at \|z\| < 0.5, stop out at \|z\| > 4 in
   case the relationship is breaking down.
6. **Backtest:** returns-space P&L (`position × (return_A − β·return_B)`),
   transaction costs charged on both legs on every position change.
7. **Validation split:** performance is reported separately for train
   (2020–2023) and test (2024–2025), so any in-sample edge that doesn't
   survive out-of-sample is visible rather than hidden.

## How This Was Validated
Important distinction: the checks below validate that **the code correctly
implements the methodology** — they do not validate that **the strategy
is profitable on real markets**. Those are separate claims.

Because the sandbox this was built in has no network access to pull real
market data, the pipeline was stress-tested against **synthetic data with a
known ground truth**:
- A synthetic pair was constructed with an engineered, known cointegrating
  relationship (shared stochastic trend + a stationary mean-reverting
  spread with a known hedge ratio), alongside synthetic pairs that are
  genuinely independent random walks with no real relationship.
- The screening step correctly flagged the true pair as significant. In one
  test run, two independent random walks were flagged as "cointegrated" by
  chance — a real false positive, which is exactly the failure mode the
  Bonferroni correction exists to catch, not a bug.
- The hedge-ratio estimate recovered the true, known beta within reasonable
  estimation error.
- The half-life calculation returned a sane, small positive number of days
  for genuinely mean-reverting spreads.
- The signal-generation logic was confirmed to use only information
  available up to time *t* when deciding the position applied to returns
  realized after *t* (no look ahead).
- The full pipeline (screen → hedge ratio → signals → backtest → train/test
  split) ran end-to-end on synthetic data with finite, sane Sharpe/drawdown
  output and no crashes.

**What's still open:** whether a real NSE sector pair is genuinely
cointegrated, whether the strategy is profitable after real transaction
costs, and whether any edge survives out-of-sample. That's an empirical
question the synthetic tests can't answer — see Results below.

## Results
*[Run `pairs_trading_model.py` locally and paste the real printed output here — pair selected, hedge ratio, half-life, and the in-sample vs. out-of-sample Sharpe/return/drawdown table.]*

| | Sharpe | Ann. Return | Ann. Vol | Max Drawdown |
|---|---|---|---|---|
| In-sample (train) | — | — | — | — |
| Out-of-sample (test) | — | — | — | — |

**Selected pair:** — · **Cointegration p-value:** — · **Bonferroni-significant:** —
**Hedge ratio (β):** — · **Half-life:** — trading days

## Limitations
- **Multiple testing:** screening N pairs and picking the best-ranked one
  inflates the false-positive rate; the Bonferroni flag makes this visible
  rather than hiding a single lucky-looking result.
- **Single pair, single test window:** one ~15-month out-of-sample test on
  one pair, not a walk-forward study across many periods — a real fund
  would require the edge to survive many such splits.
- **Transaction cost model** is a flat bps-per-leg approximation; it
  ignores bid-ask spread widening in stress periods, short-borrow
  cost/availability, and market-impact costs at size.
- **No margin/leverage financing costs** modeled for the short leg of the
  dollar-neutral position.

## How to Run
```
pip install numpy pandas matplotlib statsmodels yfinance
python pairs_trading_model.py
```
Universe, date ranges, entry/exit thresholds, and transaction costs are all
configurable at the top of the script.
