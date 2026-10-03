"""
Data loader for the pairs trading backtester.

Fetches adjusted close prices for two tickers and aligns them onto a
common set of trading days, since a pairs strategy is meaningless on
a day where only one leg actually traded.
"""

import pandas as pd
import yfinance as yf


def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Adjusted close (not raw close) is used so that stock splits and
    dividends don't show up as fake price jumps in the series.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if len(tickers) > 1 else raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices


def align_pair(prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """
    Return a clean two-column DataFrame with only the days both
    tickers actually have a price. Any day missing one leg (holiday
    mismatch, delisting gap, data provider hiccup) is dropped rather
    than filled, since faking a price for a pairs trade is worse
    than just skipping that day.
    """
    pair = prices[[ticker_a, ticker_b]].dropna()
    if len(pair) < 250:
        raise ValueError(
            f"Only {len(pair)} aligned trading days for {ticker_a}/{ticker_b} — "
            "need at least a year of overlapping data to test cointegration meaningfully."
        )
    return pair


def get_pair_data(ticker_a: str, ticker_b: str, start: str, end: str) -> pd.DataFrame:
    """Convenience wrapper: fetch and align a single pair in one call."""
    prices = fetch_price_data([ticker_a, ticker_b], start, end)
    return align_pair(prices, ticker_a, ticker_b)


if __name__ == "__main__":
    # Example: swap these for the pair you actually want to test
    pair_prices = get_pair_data("KO", "PEP", start="2015-01-01", end="2024-12-31")
    print(pair_prices.head())
    print(f"\n{len(pair_prices)} aligned trading days from "
          f"{pair_prices.index[0].date()} to {pair_prices.index[-1].date()}")