import os
import requests
import zipfile
import io
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import Optional, List
import json 
import sys

from datetime import date

from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Company, Base, Currency


# --- REPAIRED SECTION: Logic & Data Mapping ---
def get_bis_bulk_data(search_term: str = "Triennial survey") -> Optional[pd.DataFrame]:
    base_url = "https://data.bis.org"
    bulk_page = f"{base_url}/bulkdownload"
    print(f"[*] Scanning BIS Portal for '{search_term}'...")
    response = requests.get(bulk_page, timeout=15)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    target_link = next((l['href'] for l in links if search_term in l.text and "CSV, flat" in l.text), None)
    if not target_link:
        target_link = next((l['href'] for l in links if search_term in l.text and ".zip" in l['href']), None)
    if target_link:
        full_zip_url = target_link if target_link.startswith('http') else base_url + target_link
        print(f"[+] Downloading: {full_zip_url}")
        r = requests.get(full_zip_url, timeout=30)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                return pd.read_csv(f, low_memory=False)
    return None



def clean_bis_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] Starting Data Repair and Column Mapping...")
    df.columns = [c.split(':')[0].strip() for c in df.columns]
    
    mapping = {
        'DER_CURR_LEG1': 'RAW_CURRENCY', # Keep raw string temporarily
        'OBS_VALUE': 'OBS_VALUE', 
        'TIME_PERIOD': 'TIME_PERIOD'
    }
    df = df.rename(columns=mapping)
    
    # Extract Code and Name: "USD: US dollar" -> Code: USD, Name: US dollar
    df['CURRENCY'] = df['RAW_CURRENCY'].astype(str).apply(lambda x: x.split(':')[0].strip())
    df['CURRENCY_NAME'] = df['RAW_CURRENCY'].astype(str).apply(
        lambda x: x.split(':')[1].strip() if ':' in x else x
    )
    
    if 'DER_NET' in df.columns:
        df = df[df['DER_NET'] == 'N']
        
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    return df.dropna(subset=['OBS_VALUE', 'CURRENCY'])




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python discovery.py <params.json>")
        sys.exit(1)

    conf_path = sys.argv[1]
    with open(conf_path, 'r') as f:
        params = json.load(f)
    
    alpha = params.get("alpha", 0.95)
    raw_data = get_bis_bulk_data("Triennial survey")

    # 1. Database Reset
    target_table = Base.metadata.tables.get('currencies')
    if target_table is not None:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS currencies CASCADE"))
            conn.commit()
    
    Base.metadata.create_all(engine)
    print("[+] 'currencies' table initialized.")

    # 2. Data Processing
    df = clean_bis_data(raw_data)
    recent_year = df['TIME_PERIOD'].max()
    snapshot = df[df['TIME_PERIOD'] == recent_year].copy()

    # 3. Isolate Total vs Currencies
    # TO1 is the summary row
    total_mask = snapshot['CURRENCY'].str.contains("TO1", na=False)
    declared_total = snapshot.loc[total_mask, 'OBS_VALUE'].sum()
    
    # Strictly 3-letter codes ONLY (filters out TO1, TOTAL, etc.)
    currencies_only = snapshot[snapshot['CURRENCY'].str.match(r'^[A-Z]{3}$', na=False)].copy()
    
    if currencies_only.empty:
        print("[!] Error: No currencies found after filtering. Check data format.")
        sys.exit(1)

    # 4. Ranking and Stats
    rank = currencies_only.groupby('CURRENCY')['OBS_VALUE'].sum().sort_values(ascending=False)
    calc_total = rank.sum()

    name_lookup = currencies_only.set_index('CURRENCY')['CURRENCY_NAME'].to_dict()

    stats_df = pd.DataFrame({
        "currency_code": rank.index,
        "name": [name_lookup.get(c, "") for c in rank.index], # Map the name back
        "individual_flow": (rank / calc_total).round(6),
        "cumulative_flow": (rank.cumsum() / calc_total).round(6)
    }).reset_index(drop=True)

    print("\n[*] Processed Currency Flow (Top 5):")
    print(stats_df.head(5))

    # 5. Whitelist and Save
    whitelist_stats = stats_df[stats_df['cumulative_flow'] <= alpha].copy()
    params["ticker_whitelist"] = whitelist_stats['currency_code'].tolist()

    # Update JSON params
    params["ticker_whitelist"] = whitelist_stats['currency_code'].tolist()
    print(params)
    # Database Sync
    with SessionLocal() as session:
        for _, row in whitelist_stats.iterrows():
            curr_data = {
                "code": row['currency_code'],
                "name": row['name'], # Now it's not blank!
                "individual_flow": float(row['individual_flow']),
                "last_updated": date.today()
            }
            
            stmt = pg_insert(Currency).values(curr_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["code"],
                set_={
                    "name": stmt.excluded.name,
                    "individual_flow": stmt.excluded.individual_flow,
                    "last_updated": stmt.excluded.last_updated
                }
            )
            session.execute(stmt)
        session.commit()

    # Save back to config
    with open(conf_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\n[SUCCESS] Sync complete. {len(whitelist_stats)} tickers updated.")