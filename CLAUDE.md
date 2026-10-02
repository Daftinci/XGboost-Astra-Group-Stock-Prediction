# Stocks — Astra Group IDX Price/Direction Prediction

Exploratory ML project predicting price movement for Astra Group companies on the
Indonesian Stock Exchange (IDX), using price history and (experimentally) news
sentiment. Everything here is exploratory/educational — no notebook's output should
be treated as a trading signal (see Findings below).

## Environment

- Conda env `stock` (Python, via `C:\Users\Daafiq\.conda\envs\stock\python`).
- Key packages: pandas, numpy, xgboost, scikit-learn, statsmodels, matplotlib, seaborn,
  requests, openai, yfinance, nbformat/nbconvert.
- Run/execute notebooks headlessly with:
  `cd notebooks && python -m jupyter nbconvert --to notebook --execute --inplace "<name>.ipynb"`
- Notebooks needing `SECTORS_API_KEY` / `OPENAI_API_KEY` (the two "News Sentiment"
  notebooks, `scripts/fetch_news.py`) can't be run by Claude in a non-interactive session —
  they use `getpass` prompts. The user has to run those themselves; everything else
  (price-only notebooks) Claude can execute directly.

## Directory layout

```
data/
  prices/                One CSV per ticker (ASII, AALI, UNTR, AUTO, ACST, MPMX; TURI missing) + combined astra_group_1y.xlsx
  news/<TICKER>/         Per-ticker news CSVs, raw + labeled (see "Two sentiment schemes" below)
scripts/
  pull_astra_group_stocks.py   Pulls 1y OHLCV for all Astra Group tickers -> data/prices/
  fetch_news.py                Standalone ASII-only news fetch + 3-class LLM labeling -> data/news/ASII/asii_news_labeled.csv
notebooks/               All notebooks (run with cwd = notebooks/; data paths are ../data/...)
Stock UT pred.ipynb              UNTR tutorial notebook: baseline -> decomposition -> return model -> sentiment feature attempt.
                                  Long, heavily-commented walkthrough of the whole methodology; read this first if new to the project.
UNTR News Sentiment.ipynb        Pulls + labels UNTR news (Sectors + OpenAI, 2-class positive/negative + unclear/error fallback).
ASII News Sentiment.ipynb        Same, for ASII.
UNTR Sentiment Window Model.ipynb   Regression model restricted to the window where UNTR sentiment actually exists.
ASII Direction Pred - Sentiment Window.ipynb   Up/down classifier restricted to the ASII sentiment-covered window.
Stock Direction Pred.ipynb       Generic up/down classifier, TICKER-parameterized (defaults UNTR). Includes a live
                                  "predict the next real trading day" cell at the end.
Stock Direction Pred - ASII.ipynb   ASII run of the above (cloned, TICKER swapped).
Stock Direction Pred - Walk Forward.ipynb        Walk-forward validation (5 expanding-window folds) + P&L backtest, UNTR.
Stock Direction Pred - Walk Forward - ASII.ipynb Same, ASII.
```

## Methodology conventions (keep consistent in new work)

- **No leakage**: every lag/rolling feature is `.shift(1)`+ before use; train/val/test
  splits are always chronological; validation is carved from the *end of train* for
  early stopping; the test set is touched exactly once, at the end.
- **Always beat two baselines**, not one: persistence (tomorrow = today) and
  majority-class (train's most common outcome). A model that doesn't beat *both* isn't
  adding value — several models here tie or lose to majority-class alone.
- **Model returns/direction, not price level.** Price-level regression overfits
  (`Stock UT pred.ipynb`'s first model: train RMSE ~75, test RMSE ~1300 — pure
  memorization). `pct_change()` targets are roughly stationary and play to tree models'
  strengths.
- **Classical decomposition doesn't work here** (tested and rejected in
  `Stock UT pred.ipynb`) — daily IDX price is close to a random walk with no real
  weekly seasonality; a committed multi-day trend line compounds error the way daily
  re-anchoring never does. Don't reach for it again without new evidence of real
  seasonality.
- **Prefer walk-forward validation over a single train/test split** for any real
  claim — single-split results here have flipped between "model wins" and "model
  loses" depending on which window was picked. See Findings.
- **P&L backtests need a cost assumption stated explicitly** (15 bps/trade used here,
  not real IDX fee data — say so if you reuse it) and should be checked fold-by-fold
  before trusting an aggregate return, since one lucky window can carry an entire
  average (see ASII finding below).

## Two sentiment labeling schemes — don't conflate them

1. **UNTR/ASII News Sentiment.ipynb** (Sectors API + OpenAI `gpt-4o-mini`): 2-class
   `positive`/`negative` + `unclear`/`error` fallback, generic financial-news prompt.
   Output: `data/news/<TICKER>/<ticker>_news_sentiment.csv`.
2. **scripts/fetch_news.py**: 3-class `positive`/`neutral`/`negative`, shareholder-perspective
   prompt with explicit thresholds (profit decline >10% YoY, etc.). Output:
   `data/news/ASII/asii_news_labeled.csv`.

Both map to a numeric feature the same way when used: `positive=+1, negative=-1,
neutral/unclear/error=0`.

## Findings so far (don't re-litigate without new data)

- **Sentiment coverage is the binding constraint, not model choice.** News pulls only
  reach back a few months (Sectors API pagination limits / plan tier), while price
  history goes back ~1 year — so a sentiment feature over the *full* history is ~98%
  structurally zero and gets correctly ignored (0 feature importance) by every model
  that's tried it. Fix used: restrict train/val/test to the sentiment-covered window
  only (`*Sentiment Window*.ipynb` notebooks) rather than pretending full-history
  coverage exists.
- **ASII (holding company) has denser sentiment coverage than UNTR** — confirmed: 71/82
  days (86%) vs UNTR's ~50% in a comparable window. Worth revisiting sentiment
  features for ASII again once more days accumulate.
- **UNTR direction prediction: no edge, confirmed two ways.** Walk-forward (5 folds):
  model beat both baselines in 0/5 folds, mean accuracy below majority-class. P&L
  backtest: model-driven strategy lost -23.7% (worse than buy-and-hold's -9.1% and
  worse than a dumb persistence strategy) after 15bps/trade costs — acting on this
  model was worse than doing nothing.
- **ASII direction prediction: looks positive, isn't trustworthy yet.** Single-split
  test and walk-forward both showed a real-looking edge (accuracy, ROC-AUC 0.934, and
  a walk-forward P&L of +16.3% vs buy-hold's -17.6%) — but per-fold breakdown showed
  **one single fold (2026-03-04 to 2026-04-15) drove the entire result** in both
  accuracy and P&L; the other four folds were flat-to-negative. Treat as inconclusive,
  not a real edge, until more out-of-sample folds either confirm or kill it.
- **Bug pattern to watch for**: hardcoded past dates as function defaults (`end=` a
  fixed date) silently truncate any caller that doesn't override them — bit us twice
  (`pull_astra_group_stocks.py`'s 7-day-instead-of-1-year pull, and a `fetch_news`
  rewrite capped at a stale `end` date). Prefer `end=None` resolving to
  `datetime.today()` over a hardcoded date literal.

## Known gotchas

- **`NotebookEdit` tool can corrupt cell source formatting** — observed turning a
  cell's `source` into one-character-per-array-element (still executes fine via
  Jupyter's `"".join()`, but renders as garbled vertical text in VS Code's editor).
  After using `NotebookEdit`, or when editing `.ipynb` files with a raw Python
  `json.load`/`json.dump` script, verify with:
  ```python
  bad = [i for i, c in enumerate(nb["cells"])
         if len(c["source"]) > 5 and sum(1 for x in c["source"] if len(x) <= 1) / len(c["source"]) > 0.5]
  ```
  and normalize via `c["source"] = "".join(c["source"]).splitlines(keepends=True)` if
  anything shows up. Prefer plain Python json read/modify/write over `NotebookEdit`
  for multi-cell or path-reference edits — it's been more reliable in this project.
- **Never hardcode API keys in source** — `fetch_news.py` had `SECTORS_API_KEY` and
  `OPENAI_API_KEY` hardcoded at one point; they were exposed and should be treated as
  rotated/compromised. Always use `os.environ.get(...)` with a `getpass` fallback (see
  the News Sentiment notebooks' pattern), and never let a real secret get typed into a
  cell that gets saved to disk.
- **`TURI.csv` is missing** from `data/prices/` even though
  `pull_astra_group_stocks.py` lists `TURI.JK` — yfinance likely returned empty for it
  on the last pull. Not yet investigated.
