"""
Project 3: Statistical Arbitrage -- Cointegration-Based Pairs Trading
======================================================================

Pipeline
--------
1. Download daily prices for a same-sector candidate universe (yfinance).
2. Screen every within-sector pair for cointegration on the TRAIN window only
   (Engle-Granger test), and flag which pairs stay significant after a
   Bonferroni correction for the number of pairs tested (multiple-testing /
   data-snooping caution -- see notes at the bottom of this file).
3. Lock in a hedge ratio (OLS on log prices, TRAIN only) for the best pair,
   the same "estimate on train, apply forward" discipline used in Project 1.
4. Build a rolling z-scored spread (causal, no lookahead) and trade it with a
   mean-reversion z-score entry/exit/stop rule.
5. Backtest with transaction costs, then split the resulting return series
   into TRAIN and TEST windows and report Sharpe + max drawdown for both,
   exactly like Project 1's overfitting check.

Requirements: numpy, pandas, matplotlib, statsmodels, yfinance
    pip install numpy pandas matplotlib statsmodels yfinance --break-system-packages   (Linux/Mac)
    py -m pip install numpy pandas matplotlib statsmodels yfinance                     (Windows, match your interpreter -- see Project 1 notes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
# Candidate universe, grouped by sector. We only test pairs WITHIN a sector --
# a bank and an IT stock have no fundamental reason to move together, so
# testing across sectors would just be fishing for spurious relationships.
UNIVERSE = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Auto":    ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"],
}

# Same train/test convention as Project 1, for consistency across the repo.
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-04-01"

ZSCORE_WINDOW = 60          # trading days, rolling lookback for the spread's mean/std
ENTRY_Z = 2.0                # open a position when |z| crosses this
EXIT_Z = 0.5                 # close the position when |z| falls back below this
STOP_Z = 4.0                 # force-close if |z| blows through this (relationship may be breaking down)
TRANSACTION_COST_BPS = 5     # per leg, applied on every position change
INITIAL_CAPITAL = 100_000    # rupees, notional base for the equity curve / Sharpe / drawdown


# ============================================================================
# DATA
# ============================================================================
def download_prices(universe, start, end):
    """
    Downloads adjusted close prices for the full universe and returns a wide
    DataFrame (columns = tickers). Drops any ticker that doesn't have a full
    history back to `start` (same IPO-truncation trap as the YATHARTH.NS bug
    in Project 1) and warns about it.
    """
    all_tickers = [t for tickers in universe.values() for t in tickers]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = all_tickers

    prices = prices.dropna(axis=1, how="all")
    first_valid = prices.apply(lambda col: col.first_valid_index())
    target_start = pd.Timestamp(start)
    bad = [t for t, d in first_valid.items() if d is None or d > target_start + pd.Timedelta(days=10)]
    if bad:
        print(f"WARNING: dropping tickers without full history back to {start} "
              f"(likely IPO'd later, same trap as YATHARTH.NS in Project 1): {bad}")
        prices = prices.drop(columns=bad)

    prices = prices.ffill().dropna()
    return prices


# ============================================================================
# COINTEGRATION SCREENING
# ============================================================================
def screen_pairs(prices_train, universe):
    """
    Engle-Granger cointegration test on log prices, within-sector pairs only.
    Returns a DataFrame sorted by ascending p-value, with a Bonferroni-corrected
    significance flag to guard against data snooping across many pairs.
    """
    log_prices_train = np.log(prices_train)
    rows = []
    for sector, tickers in universe.items():
        tickers = [t for t in tickers if t in log_prices_train.columns]
        for a, b in combinations(tickers, 2):
            pa = log_prices_train[a].dropna()
            pb = log_prices_train[b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < 100:
                continue
            score, pvalue, _ = coint(pa.loc[common], pb.loc[common])
            rows.append({"sector": sector, "stock_a": a, "stock_b": b,
                         "eg_pvalue": pvalue, "eg_stat": score, "n_obs": len(common)})
    result = pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)
    n_tests = len(result)
    result["bonferroni_threshold"] = 0.05 / max(n_tests, 1)
    result["significant_after_correction"] = result["eg_pvalue"] < result["bonferroni_threshold"]
    return result


def estimate_hedge_ratio(logA_train, logB_train):
    """OLS: log(A) = alpha + beta*log(B) + eps, estimated on TRAIN data only and then locked in."""
    X = np.vstack([np.ones(len(logB_train)), logB_train.values]).T
    coef, *_ = np.linalg.lstsq(X, logA_train.values, rcond=None)
    alpha, beta = coef[0], coef[1]
    return beta, alpha


def half_life_mean_reversion(spread_train):
    """
    Ornstein-Uhlenbeck discretisation: spread_t - spread_{t-1} = lambda*spread_{t-1} + eps
    half_life = -ln(2)/lambda (trading days). Used to sanity-check ZSCORE_WINDOW is a
    sensible order of magnitude relative to how fast the spread actually mean-reverts.
    """
    s = spread_train.values
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.vstack([np.ones(len(s_lag)), s_lag]).T
    coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    lam = coef[1]
    if lam >= 0:
        return np.inf
    return -np.log(2) / lam


# ============================================================================
# SPREAD, SIGNALS, BACKTEST
# ============================================================================
def build_spread(logA, logB, beta, alpha):
    return logA - beta * logB - alpha


def rolling_zscore(spread, window):
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std()
    return (spread - mean) / std


def generate_positions(zscore, entry_z, exit_z, stop_z):
    """
    +1 = long the spread (long A, short beta*B) -- entered when z is very negative
    -1 = short the spread (short A, long beta*B) -- entered when z is very positive
     0 = flat
    Exit when |z| reverts back below exit_z, or stop out if |z| blows past stop_z.
    """
    pos = np.zeros(len(zscore))
    state = 0
    z = zscore.values
    for i in range(len(z)):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = 0
            continue
        if state == 0:
            if zi > entry_z:
                state = -1
            elif zi < -entry_z:
                state = 1
        elif state == 1:
            if zi >= -exit_z or zi < -stop_z:
                state = 0
        elif state == -1:
            if zi <= exit_z or zi > stop_z:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=zscore.index)


def run_backtest(logA, logB, beta, positions, cost_bps):
    """
    Returns-space backtest: strategy_return_t = position_{t-1} * (rA_t - beta*rB_t) - cost.
    Using log-return spreads (rather than tracking share counts / rupee notional directly)
    avoids the "weights applied as share counts, not dollar allocation" class of bug from
    Project 1's original script.
    """
    rA = logA.diff()
    rB = logB.diff()
    spread_return = rA - beta * rB

    pos_lagged = positions.shift(1).fillna(0)  # decide on info through t, trade into t+1
    gross_return = pos_lagged * spread_return

    pos_change = pos_lagged.diff().abs().fillna(pos_lagged.abs())
    cost = pos_change * (cost_bps / 10000.0) * (1 + abs(beta))

    strategy_return = (gross_return - cost).fillna(0)
    return strategy_return, pos_lagged


def performance_metrics(returns, periods_per_year=252):
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": np.nan, "ann_return": np.nan, "ann_vol": np.nan,
                "max_drawdown": np.nan, "total_return": np.nan}
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"sharpe": sharpe, "ann_return": ann_return, "ann_vol": ann_vol,
            "max_drawdown": drawdown.min(), "total_return": equity.iloc[-1] - 1}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Downloading prices...")
    prices = download_prices(UNIVERSE, TRAIN_START, TEST_END)
    print(f"Universe after IPO-history check: {list(prices.columns)}\n")

    train_prices = prices.loc[TRAIN_START:TRAIN_END]
    log_prices = np.log(prices)

    print("=" * 78)
    print("STEP 1: Cointegration screen (Engle-Granger, TRAIN period only)")
    print("=" * 78)
    screened = screen_pairs(train_prices, UNIVERSE)
    print(screened.to_string(index=False))
    n_sig_raw = (screened["eg_pvalue"] < 0.05).sum()
    n_sig_corrected = screened["significant_after_correction"].sum()
    print(f"\n{len(screened)} pairs tested. {n_sig_raw} significant at raw p<0.05, "
          f"only {n_sig_corrected} survive the Bonferroni correction "
          f"(threshold {screened['bonferroni_threshold'].iloc[0]:.5f}).")
    if n_sig_corrected == 0:
        print("NOTE: no pair survives the multiple-testing correction. Proceeding with the "
              "single best-ranked pair for demonstration, but this is flagged honestly in "
              "the README as a data-snooping caveat, not hidden.")

    best = screened.iloc[0]
    stock_a, stock_b = best["stock_a"], best["stock_b"]
    print(f"\nSelected pair: {stock_a} / {stock_b}  (p={best['eg_pvalue']:.5f})")

    print("\n" + "=" * 78)
    print("STEP 2: Hedge ratio + half-life (TRAIN only, then locked in)")
    print("=" * 78)
    logA_train = log_prices[stock_a].loc[TRAIN_START:TRAIN_END]
    logB_train = log_prices[stock_b].loc[TRAIN_START:TRAIN_END]
    beta, alpha = estimate_hedge_ratio(logA_train, logB_train)
    spread_train = build_spread(logA_train, logB_train, beta, alpha)
    hl = half_life_mean_reversion(spread_train)
    print(f"Hedge ratio (beta): {beta:.4f}   Intercept (alpha): {alpha:.4f}")
    print(f"Half-life of mean reversion: {hl:.1f} trading days "
          f"(z-score rolling window is set to {ZSCORE_WINDOW} days)")

    print("\n" + "=" * 78)
    print("STEP 3: Signals + backtest (full history, causal rolling z-score)")
    print("=" * 78)
    logA_full = log_prices[stock_a]
    logB_full = log_prices[stock_b]
    spread_full = build_spread(logA_full, logB_full, beta, alpha)  # beta/alpha locked from train
    zscore = rolling_zscore(spread_full, ZSCORE_WINDOW)
    positions = generate_positions(zscore, ENTRY_Z, EXIT_Z, STOP_Z)
    strat_returns, pos_lagged = run_backtest(logA_full, logB_full, beta, positions, TRANSACTION_COST_BPS)

    train_returns = strat_returns.loc[TRAIN_START:TRAIN_END]
    test_returns = strat_returns.loc[TEST_START:TEST_END]
    train_metrics = performance_metrics(train_returns)
    test_metrics = performance_metrics(test_returns)
    n_trades = int((pos_lagged.diff().fillna(pos_lagged) != 0).sum())

    def fmt(m):
        return (f"Sharpe {m['sharpe']:.2f} | Ann.Return {m['ann_return']*100:.2f}% | "
                f"Ann.Vol {m['ann_vol']*100:.2f}% | MaxDD {m['max_drawdown']*100:.2f}% | "
                f"Total Return {m['total_return']*100:.2f}%")

    print(f"IN-SAMPLE  (train, {TRAIN_START} to {TRAIN_END}): {fmt(train_metrics)}")
    print(f"OUT-OF-SAMPLE (test, {TEST_START} to {TEST_END}):  {fmt(test_metrics)}")
    print(f"Position changes (proxy for round-trip trades): {n_trades}")

    print("\n" + "=" * 78)
    print("STEP 4: Plot")
    print("=" * 78)
    equity = INITIAL_CAPITAL * (1 + strat_returns.fillna(0)).cumprod()
    train_end_ts = pd.Timestamp(TRAIN_END)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(logA_full.index, logA_full - logA_full.iloc[0], label=f"{stock_a} (log, normalized)")
    ax.plot(logB_full.index, logB_full - logB_full.iloc[0], label=f"{stock_b} (log, normalized)")
    ax.axvline(train_end_ts, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Price co-movement: {stock_a} vs {stock_b}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(zscore.index, zscore, color="black", linewidth=0.7)
    ax.axhline(ENTRY_Z, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(-ENTRY_Z, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(EXIT_Z, color="green", linestyle=":", linewidth=0.8)
    ax.axhline(-EXIT_Z, color="green", linestyle=":", linewidth=0.8)
    ax.fill_between(zscore.index, zscore, 0, where=(positions.values == 1), color="green", alpha=0.2)
    ax.fill_between(zscore.index, zscore, 0, where=(positions.values == -1), color="red", alpha=0.2)
    ax.axvline(train_end_ts, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Spread z-score & positions")

    ax = axes[1, 0]
    ax.plot(equity.index, equity)
    ax.axvline(train_end_ts, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Strategy equity curve (train | test)")
    ax.set_ylabel("Portfolio value (Rs)")

    ax = axes[1, 1]
    labels = screened["stock_a"] + "-" + screened["stock_b"]
    colors = ["tab:green" if s else "tab:blue" for s in screened["significant_after_correction"]]
    ax.bar(labels, screened["eg_pvalue"], color=colors)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="raw p=0.05")
    ax.axhline(screened["bonferroni_threshold"].iloc[0], color="orange", linestyle="--",
               linewidth=0.8, label="Bonferroni threshold")
    ax.set_title("Cointegration screen (train)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("output.png", dpi=150)
    print("Saved chart to output.png")


if __name__ == "__main__":
    main()

# ============================================================================
# HONEST LIMITATIONS (for README / interview talking points, not hidden)
# ============================================================================
# 1. Multiple-testing / data snooping: screening N pairs and picking the lowest
#    p-value inflates the false-positive rate. The Bonferroni flag above makes
#    this visible instead of silently reporting one lucky-looking pair as "the"
#    finding.
# 2. Single pair, single test window: like Project 1, this is one 15-month-ish
#    out-of-sample test on one pair, not a walk-forward study across many
#    periods/pairs -- a real fund would require the edge to survive many such
#    splits before trusting it.
# 3. Transaction cost model is a flat bps-per-leg approximation; it ignores
#    bid-ask spread widening in stress periods, borrow cost/availability for
#    the short leg, and market-impact costs from trading in size.
# 4. No leverage/margin constraints modeled -- a real dollar/rupee-neutral
#    pairs book also needs margin financing costs on the short leg.
