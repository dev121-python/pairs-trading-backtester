"""
Performance metrics for the pairs trading backtester.

Raw cumulative PnL means very little on its own — it says nothing
about how much risk was taken to get there, or whether a couple of
lucky days are carrying the whole result. These metrics are the
standard vocabulary for answering "was this actually good, and how
confident should I be that it wasn't luck?"

Note: net_pnl from the backtest engine is a return-like quantity for
a $1-notional-per-leg, dollar-neutral position — not a percent return
on invested capital. That's a fine and standard simplification for a
first version; just be upfront about it in your writeup rather than
implying these are literal portfolio returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """
    Mean excess return divided by volatility, annualized. This is the
    single number quant recruiters will look at first — it rewards
    return per unit of risk taken, not return alone.
    """
    excess = returns - (risk_free / periods_per_year)
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Largest peak-to-trough decline in cumulative PnL. This is what
    tells you how painful the worst stretch of this strategy would
    have felt to actually hold through — a high Sharpe with a brutal
    max drawdown is a strategy most people would have abandoned
    before it recovered.
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return drawdown.min()


def extract_trade_pnls(backtest: pd.DataFrame) -> list[float]:
    """
    Group consecutive days with the same nonzero position into single
    trades, and sum net_pnl within each. This gives a per-TRADE win
    rate, which is a more honest number than a per-DAY win rate (a
    single multi-day trade shouldn't count as many separate wins).
    """
    pos = backtest["lagged_position"]
    trade_id = (pos != pos.shift()).cumsum()
    trade_pnls = []
    for _, group in backtest.groupby(trade_id):
        if group["lagged_position"].iloc[0] != 0:
            trade_pnls.append(group["net_pnl"].sum())
    return trade_pnls


def win_rate(backtest: pd.DataFrame) -> float:
    trade_pnls = extract_trade_pnls(backtest)
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def compute_metrics(backtest: pd.DataFrame) -> dict:
    returns = backtest["net_pnl"]
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(backtest["equity_curve"]),
        "num_trades": len(extract_trade_pnls(backtest)),
        "win_rate": win_rate(backtest),
    }


def print_report(backtest: pd.DataFrame, ticker_a: str, ticker_b: str) -> None:
    m = compute_metrics(backtest)
    print(f"--- Performance report: {ticker_a}/{ticker_b} pairs strategy ---")
    print(f"Annualized return:      {m['annualized_return']:.4f}")
    print(f"Annualized volatility:  {m['annualized_volatility']:.4f}")
    print(f"Sharpe ratio:           {m['sharpe_ratio']:.2f}")
    print(f"Max drawdown:           {m['max_drawdown']:.4f}")
    print(f"Number of trades:       {m['num_trades']}")
    print(f"Win rate:               {m['win_rate']:.1%}")


def plot_equity_curve(backtest: pd.DataFrame, ticker_a: str, ticker_b: str, save_path: str = "equity_curve.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(backtest.index, backtest["equity_curve"], label="Pairs strategy (cumulative PnL)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker_a}/{ticker_b} pairs strategy — equity curve")
    ax.set_ylabel("Cumulative PnL (log-spread units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


if __name__ == "__main__":
    from data_loader import get_pair_data
    from pair_selection import select_pair
    from signal_generator import generate_signals
    from backtest_engine import run_backtest

    ticker_a, ticker_b = "KO", "PEP"
    pair_prices = get_pair_data(ticker_a, ticker_b, start="2015-01-01", end="2024-12-31")
    result = select_pair(pair_prices, ticker_a, ticker_b)
    signals = generate_signals(result["spread"])
    backtest = run_backtest(signals)

    print_report(backtest, ticker_a, ticker_b)
    plot_equity_curve(backtest, ticker_a, ticker_b)