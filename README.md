# Pairs Trading Backtester

A statistical arbitrage backtester for mean-reversion pairs trading, built from scratch in Python — data loading, cointegration-based pair selection, signal generation, a lookahead-safe backtest engine, and honest in-sample/out-of-sample evaluation.

## Why this project

Built as a portfolio piece for quant finance internship applications. Originally considered an AlphaZero-style RL chess engine, but pivoted: pairs trading is a more directly relevant signal for prop trading / stat arb roles, since the core skill (MCTS-style search) doesn't translate the way cointegration, hedge ratios, and risk-adjusted evaluation do.

## Pipeline

```
data_loader.py          → fetch & align price data for a pair
pair_selection.py       → cointegration test (Engle-Granger), hedge ratio (OLS), half-life
signal_generator.py     → rolling z-score of the spread → entry/exit signals (with hysteresis)
backtest_engine.py      → simulates trades with a 1-day execution lag (no lookahead bias)
performance_metrics.py  → Sharpe, max drawdown, win rate, trade-level PnL
train_test_evaluation.py→ train/test split; window selected via fold-based validation on train only
pair_screener.py        → screens multiple candidate pairs for cointegration at once
main.py                 → runs the full pipeline end to end
```

## Key design decisions (and why)

- **Log prices, not raw prices**, for the hedge ratio regression — makes the relationship about percentage moves, which is stable across price levels and over time; a raw-price beta isn't.
- **Correlation is not cointegration.** Two assets can trend together for years while their spread never reverts. The Engle-Granger test checks the property that actually matters: is the *spread* stationary.
- **1-day execution lag in the backtest engine** (`position.shift(1)`): a position is only held on day *t* using information available through day *t-1*. Without this, the backtest trades on information it wouldn't have had — the single most important line in the project.
- **Transaction costs** charged on every position change, not just once — otherwise a strategy that trades constantly looks artificially better than it would in reality.
- **Train/test split, not full-sample fitting.** The hedge ratio and cointegration check are fit on an earlier window only, then frozen and applied forward to a later window the fitting process never saw.

## The overfitting lesson (the most important part of this project)

The z-score window (how many days of history define "normal" for the spread) was first chosen two ways, both wrong, before landing on something defensible:

1. **A guessed constant (20 days).** Produced a near-zero Sharpe on V/MA — not because the pair was bad, but because the window didn't match how slowly the spread actually cycles (multi-year swings, visible directly in the equity curve).
2. **A half-life formula** (`-ln(2)/alpha` from an AR(1) regression on the spread). Theoretically correct, but numerically unreliable on real daily data — produced a "half-life" of ~70,000 days despite the cointegration test being highly significant (p=0.0008). Lesson: statistical significance (confidence that an effect isn't exactly zero) and practical significance (the effect being large enough to matter) are different things.
3. **A naive grid search maximizing full-train-period Sharpe.** This looked like the fix, and initially appeared to work — until the "best" window by this method (10 days, Sharpe 0.65 on train) collapsed to **Sharpe -0.61 out-of-sample.** Classic hyperparameter overfitting: the window that best fit train-period noise, not train-period signal.
4. **The actual fix: fold-based validation.** Split the train period into 3 consecutive chunks, score every candidate window on each chunk separately, and pick the window whose *worst* chunk performed best (maximize the minimum per-fold Sharpe) — rather than the window with the best average. This favors a window that's consistently reasonable over one that's spectacular in one stretch and bad in the rest.

That fourth approach selected a 90-day window, with fold Sharpes of 0.84 / 0.61 / 0.39 — all positive, all in a tight band, unlike the 10-day window's wildly inconsistent 0.72 / 1.17 / 0.23. Out-of-sample, it produced Sharpe 0.87.

## Results — V/MA (validated final configuration)

| | Train (in-sample) | Test (out-of-sample) |
|---|---|---|
| Sharpe ratio | 0.56 | 0.87 |
| Annualized return | 4.2% | 5.3% |
| Max drawdown | -15.3% | -4.5% |
| Number of trades | 27 | 12 |
| Win rate | 81.5% | 83.3% |

Cointegration p-value: 0.0008 (full sample), 0.018 (train only) — both comfortably below the 5% threshold.

**Caveat worth stating plainly:** 12 out-of-sample trades is a small sample. The result is supported by the fold-consistency check above, not by this table alone — a single additional losing trade would move these numbers meaningfully. Reporting trade count alongside Sharpe throughout this project is a deliberate choice, not an oversight.

## Other pairs tested

12 same-sector candidate pairs were screened in total via `pair_screener.py`: PFE/MRK, JPM/BAC, XOM/CVX, V/MA, HD/LOW, WMT/TGT, UNH/CI, GS/MS, C/WFC, MCD/YUM, UPS/FDX, T/VZ. **Only V/MA passed cointegration on both the full sample and the train-only window.** This 1-in-12 pass rate is reported deliberately rather than treated as a shortfall: the search was stopped here rather than widened further, since continuing to screen pairs until a second one passes would itself become a form of overfitting — at a 5% significance threshold, roughly 1 in 20 *unrelated* pairs would be expected to pass by chance alone, so hunting for more "winners" risks manufacturing a false positive rather than finding a real one.

Two instructive negative results came out of this search:
- **KO/PEP** (tested separately, not part of the 12-pair screen): passed on the train window (p=0.026) but failed on the full sample (p=0.107), with a clear structural breakdown in the relationship visible post-2020.
- **MCD/YUM**: the mirror case — passed on the full sample (p=0.028) but failed on the train-only window (p=0.124), meaning the relationship wasn't reliably present during the earlier period used for fitting.

Both are left in the project history rather than hidden, since the negative results are part of demonstrating the methodology works, not just the one pair that happened to pass.

## Running it

```bash
pip install -r requirements.txt
python main.py          # runs the full pipeline on V/MA
python pair_screener.py # screens additional candidate pairs
```

## Known limitations / next steps

- Execution is close-to-close; a more realistic version would execute at the next day's open.
- Only one pair (V/MA) ended up cointegrated across 12 screened candidates; a larger-scale version would screen across more sectors and more pair combinations to find additional validated pairs, ideally testing a pre-registered list rather than searching reactively.
- No portfolio-level position sizing or risk limits — this is a single-pair signal study, not a full trading system.