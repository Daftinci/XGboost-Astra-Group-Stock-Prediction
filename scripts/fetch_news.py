from pathlib import Path
import os
import time
import requests
import pandas as pd
from load_env import load_env
from datetime import datetime
from openai import OpenAI

# API KEY HERE
load_env()
API_KEY = os.environ["SECTORS_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


BASE_URL = "https://api.sectors.app/v2/news/"

_LABEL_SYSTEM = """\
You are a financial news sentiment classifier for the Indonesian Stock Exchange (IDX).
Label each article from a SHAREHOLDER'S perspective. Reply with exactly one word.

positive = profit growth, dividend announced, share buyback, insider buying, analyst upgrade
negative = net loss, profit decline >10% YoY, layoffs, license revoked, sustained insider selling
neutral  = routine filings, ambiguous insider trading, mixed results, rights issues, administrative announcements\
"""


def _llm_label(text: str, client: OpenAI) -> str:
    msg = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        messages=[
            {"role": "system", "content": _LABEL_SYSTEM},
            {"role": "user",   "content": text[:1500]},
        ],
    )
    label = msg.choices[0].message.content.strip().lower()
    return label if label in ("positive", "neutral", "negative") else "neutral"


def fetch_news(symbols="ASII", start="2025-09-01", end=None, limit=30, max_articles=300):
    # end defaults to today, not a fixed past date -- a hardcoded end date
    # silently truncates every call that doesn't override it.
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    articles = []
    offset = 0
    while len(articles) < max_articles:
        url = f"{BASE_URL}?limit={limit}&extension=idx&start={start}&end={end}&offset={offset}"
        if symbols:
            url += f"&symbols={symbols}"
        resp = requests.get(url, headers={"Authorization": API_KEY})
        resp.raise_for_status()
        data = resp.json()
        articles.extend(data["results"])
        if not data["pagination"]["has_next"]:
            break
        offset += limit
    return articles[:max_articles]


def get_text(article):
    title = article.get("title") or ""
    body  = article.get("body")  or ""
    return (title + " " + body).strip()


CSV_PATH = str(Path(__file__).resolve().parent.parent / "data" / "news" / "ASII" / "asii_news_labeled.csv")


def load_dataset(random_state=42, save_csv=True, csv_path=CSV_PATH):
    articles = fetch_news()
    # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    client = OpenAI(api_key=OPENAI_API_KEY)

    rows = []
    for a in articles:
        text = get_text(a)
        if text:
            rows.append({
                "timestamp": a.get("timestamp"),
                "title": a.get("title"),
                "source": a.get("source"),
                "text": text,
                "tags": ", ".join(a.get("tags") or []),
            })

    if not rows:
        print("No articles found for the given date range.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df.iloc[:300]

    # Resume from existing progress if available
    done = 0
    if save_csv and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done = existing["label"].notna().sum()
        print(f"Resuming from row {done}/{len(df)} (found existing CSV)...")

    print(f"Labeling {len(df)} articles with Groq...")
    labels = []
    for i, text in enumerate(df["text"], 1):
        if i <= done:
            labels.append(pd.read_csv(csv_path).iloc[i - 1]["label"])
            continue
        label = _llm_label(text, client)
        labels.append(label)
        print(f"  [{i}/{len(df)}] {label}")
        # Save incrementally after each label
        if save_csv:
            df_partial = df.iloc[:i].copy()
            df_partial["label"] = labels
            df_partial.to_csv(csv_path, index=False)
        time.sleep(0.5)
    df["label"] = labels

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows to {csv_path}")
        print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def load_from_csv(csv_path=CSV_PATH):
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    load_dataset()
