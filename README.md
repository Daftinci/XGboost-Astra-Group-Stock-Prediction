# XGBoost Astra Group Stock Prediction (ASII & UNTR)

Exploratory project that tries to predict next-day price direction for two Astra Group
companies on the Indonesian Stock Exchange (IDX) using XGBoost:

- **ASII** - Astra International (group holding company)
- **UNTR** - United Tractors

Inputs are 1 year of daily OHLCV from Yahoo Finance and, experimentally, LLM-labeled news
sentiment. **This is educational work. Nothing here is a trading signal** - see the results below.

## Headline result

| | UNTR | ASII |
|---|---|---|
| Edge over baselines? | **No** | **Inconclusive** (one lucky fold) |
| Walk-forward accuracy (125 test days) | 44.8% | 64.8% |
| Best baseline accuracy | 53.6% (persistence) | 63.2% (majority class) |
| Folds beating *both* baselines | 0 / 5 | 2 / 5 |
| Mean ROC-AUC across folds | 0.563 | 0.650 |
| Walk-forward P&L (after costs) | -23.7% | +16.3% |
| Buy & hold over same window | -9.1% | -17.6% |

The ASII number looks good, but a single fold (2026-03-04 to 2026-04-15) produced almost all
of it. The other four folds were flat to negative. Treat it as unproven until more
out-of-sample data confirms or kills it.

## Metrics in detail

### Walk-forward validation (5 expanding-window folds, 25 test days each)

Test period: 2026-03-04 to 2026-09-08. Every fold trains only on the past.

**UNTR**

| Fold | Test window | Persistence | Majority | Model | ROC-AUC |
|---|---|---|---|---|---|
| 0 | 2026-03-04 to 04-15 | 0.36 | 0.52 | 0.52 | 0.631 |
| 1 | 2026-04-16 to 05-21 | 0.56 | 0.28 | 0.28 | 0.357 |
| 2 | 2026-05-22 to 06-29 | 0.60 | 0.36 | 0.36 | 0.594 |
| 3 | 2026-06-30 to 08-03 | 0.68 | 0.68 | 0.68 | 0.794 |
| 4 | 2026-08-04 to 09-08 | 0.48 | 0.44 | 0.40 | 0.438 |
| **Mean** | | **0.536** | **0.456** | **0.448** | **0.563** |

The model beat both baselines in 0/5 folds, tied the best baseline in 2 and lost in 3.

**ASII**

| Fold | Test window | Persistence | Majority | Model | ROC-AUC |
|---|---|---|---|---|---|
| 0 | 2026-03-04 to 04-15 | 0.32 | 0.64 | **0.88** | 0.934 |
| 1 | 2026-04-16 to 05-21 | 0.32 | 0.64 | 0.68 | 0.535 |
| 2 | 2026-05-22 to 06-29 | 0.60 | 0.72 | 0.60 | 0.520 |
| 3 | 2026-06-30 to 08-03 | 0.36 | 0.60 | 0.60 | 0.723 |
| 4 | 2026-08-04 to 09-08 | 0.32 | 0.56 | 0.48 | 0.536 |
| **Mean** | | **0.384** | **0.632** | **0.648** | **0.650** |

The model beat both baselines in 2/5 folds, tied in 1 and lost in 2. Fold 0 (88% accuracy,
ROC-AUC 0.934) carries the average.

### P&L backtest (long/flat, 15 bps per trade)

The 15 bps cost is an assumption, not real IDX fee data.

| Ticker | Strategy | Total return | Max drawdown | Sharpe-like | Trades | Costs paid |
|---|---|---|---|---|---|---|
| UNTR | Model | -23.7% | -32.6% | -1.95 | 20 | 3.0% |
| UNTR | Persistence | -16.9% | -27.9% | -1.34 | 59 | 8.8% |
| UNTR | Buy & hold | -9.1% | -33.5% | -0.40 | 1 | 0.1% |
| ASII | Model | +16.3% | -14.2% | 1.20 | 38 | 5.7% |
| ASII | Persistence | -51.8% | -51.9% | -6.50 | 77 | 11.6% |
| ASII | Buy & hold | -17.6% | -30.6% | -0.67 | 1 | 0.1% |

Acting on the UNTR model would have been worse than doing nothing. Per-fold ASII model P&L:
+22.9%, +1.0%, -2.2%, 0.0%, -4.2% (fold 0 again dominates).

### Single train/test split (45-day test slice)

| | UNTR | ASII |
|---|---|---|
| Persistence accuracy | 57.8% | 37.8% |
| Majority-class accuracy | 55.6% | 60.0% |
| XGBoost accuracy | 55.6% | 57.8% |
| XGBoost ROC-AUC | 0.588 | 0.588 |

On a single split the model does not beat the majority-class baseline for either stock.
Single-split results flipped between "wins" and "loses" depending on the window, which is why
walk-forward is the reference.

### Price-level regression (UNTR, RMSE in IDR)

| Model | Test RMSE |
|---|---|
| Naive persistence baseline | 493.29 |
| XGBoost, price level | 744.25 (loses) |
| XGBoost, return target | 501.13 (still slightly worse) |
| Decomposition, line fit on full train | 1744.40 |
| Decomposition, line fit on last 30 days | 6608.10 |

Predicting price level overfits and classical decomposition fails: daily IDX price behaves
close to a random walk.

### News sentiment

News only reaches back a few months, so over the full year the sentiment feature is ~98%
zeros and models ignore it (0 feature importance). Restricting to the news-covered window
(2026-06-03 to 2026-09-25, 81 trading days, 46/15/20 train/val/test) gives:

| Model | Persistence | Majority | No sentiment | With sentiment |
|---|---|---|---|---|
| UNTR regression (RMSE, lower is better) | 460.60 | n/a | 462.40 | 462.96 |
| ASII direction (accuracy / ROC-AUC) | 35.0% | 55.0% | 55.0% / 0.773 | 55.0% / 0.707 |

Sentiment did not help in either case. ASII has denser coverage (71 of 82 days, 86%) than
UNTR (~50%), so it is worth revisiting as more days accumulate.

## Method

- **Target:** next-day direction (up/down) or return, never price level.
- **Features:** lagged returns (1, 2, 3, 5 days), 5-day volatility, day of week; optional
  sentiment (`positive=+1, negative=-1, neutral/unclear/error=0`). Everything is shifted at
  least one day, so there is no leakage.
- **Model:** XGBoost with early stopping on a validation slice carved from the end of train.
- **Baselines:** persistence (tomorrow = today) and majority class. A model has to beat both.
- **Validation:** chronological splits only; walk-forward with 5 expanding-window folds.

## Project structure

```
data/
  prices/      One CSV per ticker + combined astra_group_1y.xlsx
  news/        Per-ticker raw and sentiment-labeled news CSVs
scripts/
  pull_astra_group_stocks.py   Pull 1y OHLCV for Astra Group tickers (yfinance)
  fetch_news.py                ASII news fetch + 3-class LLM labeling
  load_env.py                  Loads API keys from .env
notebooks/
  Stock UT pred.ipynb                              UNTR walkthrough of the whole methodology (start here)
  Stock Direction Pred.ipynb / - ASII.ipynb        Single-split direction classifier
  Stock Direction Pred - Walk Forward.ipynb        Walk-forward + P&L backtest (UNTR)
  Stock Direction Pred - Walk Forward - ASII.ipynb Walk-forward + P&L backtest (ASII)
  UNTR Sentiment Window Model.ipynb                Regression on the news-covered window
  ASII Direction Pred - Sentiment Window.ipynb     Direction on the news-covered window
  UNTR News Sentiment.ipynb / ASII News Sentiment.ipynb   News pull + LLM labeling
```

## Running it

```bash
# environment (conda env "stock"): pandas numpy xgboost scikit-learn statsmodels
# matplotlib seaborn requests openai yfinance jupyter

python scripts/pull_astra_group_stocks.py          # refresh price data
cd notebooks
python -m jupyter nbconvert --to notebook --execute --inplace "Stock Direction Pred - Walk Forward - ASII.ipynb"
```

The two News Sentiment notebooks and `scripts/fetch_news.py` need API keys. Copy
`.env.example` to `.env` and fill in `SECTORS_API_KEY` and `OPENAI_API_KEY`. `.env` is
gitignored.

## Caveats

- One year of daily data (~250 rows) is small; results are noisy.
- Backtest costs (15 bps/trade) are an assumption, not real IDX fees.
- `TURI.JK` returned no data from yfinance, so it is not included.
- Metrics above come from the notebook outputs as last executed and will shift if the data is
  refreshed.
