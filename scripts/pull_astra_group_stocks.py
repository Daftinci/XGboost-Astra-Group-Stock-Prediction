"""
Pull 1-year historical stock data for Astra Group companies listed on IDX.
Saves each ticker as a CSV, plus one combined Excel file (multi-sheet).
"""

from pathlib import Path
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Astra Group tickers on IDX (Yahoo Finance uses .JK suffix) ---
ASTRA_GROUP_TICKERS = {
    "ASII.JK": "Astra International",
    "AALI.JK": "Astra Agro Lestari",
    "UNTR.JK": "United Tractors",
    "AUTO.JK": "Astra Otoparts",
    "ACST.JK": "Acset Indonusa",
    "MPMX.JK": "Mitra Pinasthika Mustika",
    "TURI.JK": "Tunas Ridean",
}

OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "data" / "prices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

end_date = datetime.today()
start_date = end_date - timedelta(days=365)

all_data = {}

for ticker, name in ASTRA_GROUP_TICKERS.items():
    print(f"Fetching {ticker} ({name})...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"  -> No data returned for {ticker}, skipping.")
        continue

    # Flatten multi-index columns if present (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.insert(1, "Ticker", ticker)
    df.insert(2, "Company", name)

    # Save individual CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('.JK','')}.csv")
    df.to_csv(csv_path, index=False)

    all_data[ticker.replace(".JK", "")] = df

# --- Save combined Excel workbook (one sheet per ticker) ---
excel_path = os.path.join(OUTPUT_DIR, "astra_group_1y.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for sheet_name, df in all_data.items():
        df.to_excel(writer, sheet_name=(sheet_name + "1"), index=False)

print(f"\nDone. Files saved in: {OUTPUT_DIR}/")
print(f"- Combined Excel: {excel_path}")
print(f"- Individual CSVs: {list(all_data.keys())}")
