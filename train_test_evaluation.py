"""
In-sample / out-of-sample evaluation for the pairs trading strategy.

The problem this fixes: if you compute the hedge ratio and check
cointegration using the FULL date range, you're implicitly using
future information to calibrate a strategy you then "test" on that
same range — the backtest is grading its own homework.

The fix: fit everything (cointegration check, hedge ratio) on an
earlier training window only. Freeze those parameters. Apply them,
unchanged, to a later test window the fitting process never saw. If
performance holds up out-of-sample, that's real evidence. If it
collapses, that's also real evidence — and worth reporting honestly
rather than re-tuning until the test period looks good too (which
would just reintroduce the same bias one level up).
"""

import numpy as np
import pandas as pd

from pair_selection import test_cointegration, compute_hedge_ratio, compute_spread, half_life
from signal_generator import generate_signals
from backtest_engine import run_backtest
from performance_metrics import compute_metrics, sharpe_ratio


def split_train_test(pair_prices: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pair_prices.loc[:split_date]
    test = pair_prices.loc[split_date:].iloc[1:]  # avoid double-counting the split date
    return train, test


def select_window(
    train_spread: pd.Series,
    candidates: tuple[int, ...] = (10, 15, 20, 30, 40, 60, 90),
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 5.0,
    min_trades: int = 10,
    n_folds: int = 3,
) -> tuple[int, dict]:
    """
    Pick the z-score window — ON THE TRAIN SPREAD ONLY.

    IMPORTANT: this does NOT simply pick whichever window gives the best
    full-train-period Sharpe. That approach was tried first and it
    overfits — it will always find some window that happens to fit
    quirks of this one sample, with nothing to distinguish a real
    pattern from a coincidence of this specific stretch of history.

    Instead: split the train period into n_folds consecutive chunks,
    score every candidate window on EACH chunk separately, and pick the
    window whose WORST chunk is least bad (maximize the minimum
    per-fold Sharpe). This directly penalizes a window that only looks
    good because of one lucky stretch — a real, stable mean-reversion
    pattern should perform reasonably across most sub-periods, not
    spike in one and fall apart in the rest.

    Candidates with fewer than min_trades on the FULL train period are
    skipped entirely — not enough trades to trust any statistic from.
    If nothing clears that bar, the largest candidate window is used
    as the most conservative fallback.
    """
    scored = {}
    for w in candidates:
        signals = generate_signals(train_spread, window=w, entry_z=entry_z, exit_z=exit_z)
        backtest = run_backtest(signals, cost_bps=cost_bps)
        m = compute_metrics(backtest)

        fold_indices = np.array_split(backtest.index, n_folds)
        fold_sharpes = [sharpe_ratio(backtest.loc[idx, "net_pnl"]) for idx in fold_indices if len(idx) > 0]
        m["fold_sharpes"] = fold_sharpes
        m["worst_fold_sharpe"] = min(fold_sharpes) if fold_sharpes else float("-inf")

        scored[w] = m

    eligible = {w: m for w, m in scored.items() if m["num_trades"] >= min_trades}
    pool = eligible if eligible else scored
    best_window = max(pool, key=lambda w: pool[w]["worst_fold_sharpe"])

    return best_window, scored


def run_train_test_backtest(
    pair_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    split_date: str,
    window: int | None = None,  # None = select via train-only grid search
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 5.0,
) -> dict:
    train, test = split_train_test(pair_prices, split_date)

    # Fit on training data ONLY.
    coint_result = test_cointegration(train[ticker_a], train[ticker_b])
    hedge_ratio = compute_hedge_ratio(train[ticker_a], train[ticker_b])

    # Apply the frozen hedge ratio across the FULL period, so the spread
    # is one continuous series — not recomputed at the split boundary.
    full_spread = compute_spread(pair_prices[ticker_a], pair_prices[ticker_b], hedge_ratio)
    train_spread = full_spread.loc[:split_date]

    hl = half_life(train_spread)  # kept for reporting — see select_window's docstring for why
    window_scores = None
    if window is None:
        window, window_scores = select_window(train_spread, entry_z=entry_z, exit_z=exit_z, cost_bps=cost_bps)

    signals = generate_signals(full_spread, window=window, entry_z=entry_z, exit_z=exit_z)
    backtest = run_backtest(signals, cost_bps=cost_bps)

    train_backtest = backtest.loc[:split_date]
    test_backtest = backtest.loc[split_date:].iloc[1:]

    return {
        "hedge_ratio": hedge_ratio,
        "half_life": hl,
        "window_used": window,
        "window_scores": window_scores,
        "train_cointegration": coint_result,
        "train_metrics": compute_metrics(train_backtest),
        "test_metrics": compute_metrics(test_backtest),
        "train_backtest": train_backtest,
        "test_backtest": test_backtest,
    }


def print_comparison(result: dict, ticker_a: str, ticker_b: str) -> None:
    print(f"--- {ticker_a}/{ticker_b}: in-sample vs out-of-sample ---")
    print(f"Hedge ratio (fit on train only): {result['hedge_ratio']:.4f}")
    print(f"Train cointegration p-value:     {result['train_cointegration']['pvalue']:.4f}")
    print(f"Train half-life (reference only, see note): {result['half_life']:.1f} days")
    print(f"Z-score window used (train-only grid search): {result['window_used']} days")
    if result["window_scores"]:
        print("Candidate windows tried on TRAIN only (full-train sharpe, WORST-FOLD sharpe [selection criterion], trades):")
        for w, m in sorted(result["window_scores"].items()):
            folds_str = ", ".join(f"{s:.2f}" for s in m["fold_sharpes"])
            print(f"  {w:>3}: full={m['sharpe_ratio']:.3f}, worst_fold={m['worst_fold_sharpe']:.3f}, "
                  f"trades={m['num_trades']}, folds=[{folds_str}]")
    print()
    print(f"{'Metric':<24}{'Train (in-sample)':<20}{'Test (out-of-sample)'}")
    for key in ["sharpe_ratio", "annualized_return", "max_drawdown", "num_trades", "win_rate"]:
        train_val = result["train_metrics"][key]
        test_val = result["test_metrics"][key]
        print(f"{key:<24}{train_val:<20.4f}{test_val:.4f}")


if __name__ == "__main__":
    from data_loader import get_pair_data

    ticker_a, ticker_b = "KO", "PEP"
    pair_prices = get_pair_data(ticker_a, ticker_b, start="2015-01-01", end="2024-12-31")

    # ~70/30 split: fit on 2015-2021, test on 2022-2024
    result = run_train_test_backtest(pair_prices, ticker_a, ticker_b, split_date="2021-12-31")
    print_comparison(result, ticker_a, ticker_b)