"""
Pair selection for the pairs trading backtester.

Two steps here, and they answer two different questions:

1. Cointegration test (Engle-Granger) — "do these two series actually
   have a long-run equilibrium relationship, or did they just happen
   to look correlated in this sample?" High correlation alone is NOT
   enough; two trending series can be highly correlated while never
   reverting to any stable relationship.

2. Hedge ratio (OLS regression) — "how many shares of B do I need
   per share of A so that the combined position is market-neutral?"
   This ratio defines the spread you'll actually trade.

3. Half-life of mean reversion — how long, in trading days, a
   deviation from the mean typically takes to decay halfway back.
   This is what should set the signal generator's rolling window —
   not a guessed constant.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def test_cointegration(price_a: pd.Series, price_b: pd.Series) -> dict:
    """
    Run the Engle-Granger cointegration test on log prices.

    Log prices are used so that the hedge ratio and spread behave
    consistently regardless of the two tickers' absolute price levels.

    Returns the test statistic, p-value, and critical values. A
    p-value below 0.05 is the conventional (if somewhat arbitrary)
    threshold for treating the pair as cointegrated — but always
    look at the actual spread chart too, not just this number.
    """
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    score, pvalue, crit_values = coint(log_a, log_b)
    return {
        "score": score,
        "pvalue": pvalue,
        "crit_values": crit_values,
        "is_cointegrated": pvalue < 0.05,
    }


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Regress log(price_a) on log(price_b) to get the hedge ratio (beta).

    The spread is then: log(price_a) - beta * log(price_b)
    This beta tells you the ratio of exposure needed so that a move
    in B alone (with no divergence from A) doesn't itself create a
    profit or loss on the spread.
    """
    loga = np.log(price_a)
    logb = np.log(price_b)
    model = sm.OLS(loga,sm.add_constant(logb))
    return model.fit().params.iloc[1] # Return the slope (hedge ratio)


def compute_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    """The traded quantity: log(price_a) - hedge_ratio * log(price_b)."""
    return np.log(price_a) - hedge_ratio * np.log(price_b)




def half_life(spread: pd.Series) -> float:
    """
    Regress delta_spread_t on spread_{t-1} (no constant), then
    half_life = -ln(2) / alpha. A stronger (more negative) alpha means
    faster reversion, hence a shorter half-life.
    """
    spread_lag = spread.shift(1)
    delta_spread = spread.diff()
 
    df = pd.concat([delta_spread, spread_lag], axis=1).dropna()
    df.columns = ["delta_spread", "spread_lag"]
 
    model = sm.OLS(df["delta_spread"], df["spread_lag"]).fit()  # no constant
    alpha = model.params.iloc[0]
 
    if alpha >= 0:
        # No mean reversion detected at all — half-life is undefined/infinite.
        return float("inf")
    return -np.log(2) / alpha


 
def select_pair(pair_prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> dict:
    price_a, price_b = pair_prices[ticker_a], pair_prices[ticker_b]
 
    coint_result = test_cointegration(price_a, price_b)
    hedge_ratio = compute_hedge_ratio(price_a, price_b)
    spread = compute_spread(price_a, price_b, hedge_ratio)
    hl = half_life(spread)
 
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "half_life": hl,
        **coint_result,
    }
 
 

if __name__ == "__main__":
    from data_loader import get_pair_data

    pair_prices = get_pair_data("KO", "PEP", start="2015-01-01", end="2024-12-31")
    result = select_pair(pair_prices, "KO", "PEP")

    print(f"Hedge ratio (beta): {result['hedge_ratio']:.4f}")
    print(f"Cointegration p-value: {result['pvalue']:.4f}")
    print(f"Cointegrated at 5% level: {result['is_cointegrated']}")
    print(f"Half-life of mean reversion: {result['half_life']:.1f} trading days")