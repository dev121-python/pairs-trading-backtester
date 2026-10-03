"""
Main entry point — runs the full pairs trading pipeline on real data.

Usage:
    python main.py

Edit TICKER_A / TICKER_B below to test a different pair.
"""

from data_loader import get_pair_data
from pair_selection import select_pair
from signal_generator import generate_signals
from backtest_engine import run_backtest
from performance_metrics import print_report, plot_equity_curve
from train_test_evaluation import run_train_test_backtest, print_comparison

TICKER_A = "V"
TICKER_B = "MA"
START = "2015-01-01"
END = "2024-12-31"
SPLIT_DATE = "2021-12-31"  # everything before this = train, everything after = test


def main():
    print(f"Fetching {TICKER_A}/{TICKER_B} price data ({START} to {END})...")
    pair_prices = get_pair_data(TICKER_A, TICKER_B, start=START, end=END)

    print("\n=== Step 1: Pair selection (full-sample check) ===")
    result = select_pair(pair_prices, TICKER_A, TICKER_B)
    print(f"Hedge ratio: {result['hedge_ratio']:.4f}")
    print(f"Cointegration p-value: {result['pvalue']:.4f}")
    print(f"Cointegrated at 5% level: {result['is_cointegrated']}")
    print(f"Half-life (reference only — can be unreliable, see step 3 note): {result['half_life']:.1f} trading days")

    if not result["is_cointegrated"]:
        print(
            f"\nWARNING: {TICKER_A}/{TICKER_B} did not pass the cointegration test. "
            "The backtest below will still run, but treat its results as unreliable — "
            "try a different candidate pair instead."
        )

    print("\n=== Step 2: Train/test split with train-only window selection (the result that matters) ===")
    split_result = run_train_test_backtest(pair_prices, TICKER_A, TICKER_B, split_date=SPLIT_DATE)
    print_comparison(split_result, TICKER_A, TICKER_B)

    print("\n=== Step 3: Full-sample backtest, using the same window chosen above ===")
    print("(full-sample numbers are optimistic vs. the test-only numbers above — "
          "they include the train period the window was chosen on)")
    signals = generate_signals(result["spread"], window=split_result["window_used"])
    backtest = run_backtest(signals)
    print_report(backtest, TICKER_A, TICKER_B)
    plot_equity_curve(backtest, TICKER_A, TICKER_B, save_path=f"{TICKER_A}_{TICKER_B}_equity_curve.png")


if __name__ == "__main__":
    main()