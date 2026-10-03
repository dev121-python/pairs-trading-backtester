"""
Signal generator for the pairs trading backtester.

Converts the raw spread into a rolling z-score, then applies simple
entry/exit thresholds with hysteresis (a wider entry band than exit
band) so the strategy doesn't flip positions every time the z-score
wobbles across a single line.

IMPORTANT — this generates signals based on information available at
the CLOSE of each day. The backtest engine must execute these one day
later (next day's open or close) — never on the same day the signal
was computed. Trading on today's close using today's close is exactly
the lookahead bias this whole project is trying to avoid.
"""

import pandas as pd


def compute_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling z-score of the spread: how many standard deviations is
    today's spread from its own recent (trailing) average?

    A rolling, not full-sample, mean/std is used deliberately — the
    "equilibrium" level of a spread can drift over months, so scoring
    against a fixed all-time mean would go stale.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / rolling_std


def generate_signals(
    spread: pd.Series,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    Turn the z-score into a position signal.

    Position convention:
        +1 = long the spread  (long ticker_a, short hedge_ratio * ticker_b)
        -1 = short the spread (short ticker_a, long hedge_ratio * ticker_b)
         0 = flat

    Logic (with hysteresis to avoid flip-flopping near the entry line):
        - Flat -> enter long  when z < -entry_z  (spread unusually low, bet it rises)
        - Flat -> enter short when z >  entry_z  (spread unusually high, bet it falls)
        - In a position -> exit back to flat once |z| < exit_z (reverted close to normal)
        - Otherwise, hold whatever position you're already in
    """
    z = compute_zscore(spread, window)
    position = pd.Series(0, index=spread.index, dtype=int)

    current = 0
    for t in range(len(z)):
        if pd.isna(z.iloc[t]):
            position.iloc[t] = 0
            continue
        if current == 0:
            if z.iloc[t] < -entry_z:
                current = 1
            elif z.iloc[t] > entry_z:
                current = -1
        else:
            if abs(z.iloc[t]) < exit_z:
                current = 0
        position.iloc[t] = current

    return pd.DataFrame({"spread": spread, "zscore": z, "position": position})


if __name__ == "__main__":
    from data_loader import get_pair_data
    from pair_selection import select_pair

    pair_prices = get_pair_data("KO", "PEP", start="2015-01-01", end="2024-12-31")
    result = select_pair(pair_prices, "KO", "PEP")
    signals = generate_signals(result["spread"])

    print(signals.tail(10))
    print(f"\nDays in a position: {(signals['position'] != 0).sum()} / {len(signals)}")