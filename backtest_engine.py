"""
Backtest engine for the pairs trading strategy.

This is the piece that has to get lookahead bias right. The rule:
a position decided using information through the close of day t is
only realized starting day t+1 — you cannot trade on a signal using
the same day's price that produced it.

Simplification worth knowing: this executes at day-t's closing price
using a signal computed from day (t-1)'s close, i.e. standard
"close-to-close" backtesting. It's a common and acceptable first
version, but a more realistic build would execute at the next day's
OPEN instead, since you couldn't actually trade at a close you're
only just now observing. Worth mentioning as a known limitation in
your writeup rather than presenting the backtest as more realistic
than it is.
"""

import pandas as pd


def run_backtest(signals: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """
    signals: output of generate_signals() — needs 'spread' and 'position' columns.
    cost_bps: round-trip-style cost in basis points, charged per unit of
              position change (i.e. every time you open, close, or flip).

    Returns the same DataFrame with PnL and equity columns added.
    """
    df = signals.copy()

    # The position we actually HOLD today was decided using yesterday's
    # close (yesterday's z-score) — this shift(1) is the lookahead guard.
    df["lagged_position"] = df["position"].shift(1).fillna(0)

    # Day-over-day change in the spread itself (already hedge-ratio-weighted).
    df["spread_change"] = df["spread"].diff().fillna(0)

    # PnL before costs: position held * how much the spread moved that day.
    df["gross_pnl"] = df["lagged_position"] * df["spread_change"]

    # Cost is charged whenever the position size changes — opening,
    # closing, or flipping from long-spread to short-spread all count.
    position_change = df["lagged_position"].diff().abs()
    position_change.iloc[0] = df["lagged_position"].iloc[0]  # first-ever entry also costs
    df["trade_cost"] = position_change.fillna(0) * (cost_bps / 10000)

    df["net_pnl"] = df["gross_pnl"] - df["trade_cost"]
    df["equity_curve"] = df["net_pnl"].cumsum()

    return df


if __name__ == "__main__":
    from data_loader import get_pair_data
    from pair_selection import select_pair
    from signal_generator import generate_signals

    pair_prices = get_pair_data("KO", "PEP", start="2015-01-01", end="2024-12-31")
    result = select_pair(pair_prices, "KO", "PEP")
    signals = generate_signals(result["spread"])
    backtest = run_backtest(signals)

    n_trades = (backtest["lagged_position"].diff().fillna(0) != 0).sum()
    print(f"Total trades (position changes): {n_trades}")
    print(f"Final cumulative PnL (log-spread units): {backtest['equity_curve'].iloc[-1]:.4f}")
    print(backtest[["zscore", "position", "net_pnl", "equity_curve"]].tail())