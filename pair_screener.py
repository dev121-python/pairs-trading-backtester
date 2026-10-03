"""
Screens several candidate pairs at once and reports which ones pass
cointegration — both on the full sample and on the train-only window,
since (as KO/PEP showed) a pair can look cointegated over the full
history while the relationship has actually broken down recently.

Add or remove candidates in CANDIDATE_PAIRS below. These are same-
sector pairs chosen because they share an obvious economic reason to
move together — a necessary starting point, though not a guarantee,
for a real cointegrated relationship.
"""

from data_loader import get_pair_data
from pair_selection import test_cointegration, compute_hedge_ratio
from train_test_evaluation import split_train_test

CANDIDATE_PAIRS = [
    ("PFE", "MRK"),   # pharma
    ("JPM", "BAC"),   # banks
    ("XOM", "CVX"),   # oil majors
    ("V", "MA"),      # payments
    ("HD", "LOW"),    # home improvement retail
]

START = "2015-01-01"
END = "2024-12-31"
SPLIT_DATE = "2021-12-31"


def screen_pair(ticker_a: str, ticker_b: str) -> dict:
    pair_prices = get_pair_data(ticker_a, ticker_b, start=START, end=END)
    train, _ = split_train_test(pair_prices, SPLIT_DATE)

    full_result = test_cointegration(pair_prices[ticker_a], pair_prices[ticker_b])
    train_result = test_cointegration(train[ticker_a], train[ticker_b])
    hedge_ratio = compute_hedge_ratio(train[ticker_a], train[ticker_b])

    return {
        "pair": f"{ticker_a}/{ticker_b}",
        "full_pvalue": full_result["pvalue"],
        "train_pvalue": train_result["pvalue"],
        "hedge_ratio": hedge_ratio,
        "passes_both": full_result["is_cointegrated"] and train_result["is_cointegrated"],
    }


def main():
    print(f"Screening {len(CANDIDATE_PAIRS)} candidate pairs ({START} to {END}, split at {SPLIT_DATE})...\n")
    results = []
    for ticker_a, ticker_b in CANDIDATE_PAIRS:
        try:
            results.append(screen_pair(ticker_a, ticker_b))
        except Exception as e:
            print(f"{ticker_a}/{ticker_b}: failed to screen ({e})")

    print(f"{'Pair':<10}{'Full p-value':<15}{'Train p-value':<16}{'Hedge ratio':<14}{'Passes both'}")
    for r in sorted(results, key=lambda r: r["full_pvalue"]):
        print(
            f"{r['pair']:<10}{r['full_pvalue']:<15.4f}{r['train_pvalue']:<16.4f}"
            f"{r['hedge_ratio']:<14.4f}{r['passes_both']}"
        )

    passing = [r for r in results if r["passes_both"]]
    print(f"\n{len(passing)} of {len(results)} pairs passed cointegration on BOTH full sample and train split.")
    if passing:
        print("Worth backtesting next:", ", ".join(r["pair"] for r in passing))
    else:
        print("None passed both — that's a real and reportable finding too. Consider widening the candidate list.")


if __name__ == "__main__":
    main()