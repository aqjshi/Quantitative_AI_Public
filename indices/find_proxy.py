import requests
import time
from datetime import datetime, timedelta
import os
import requests
import sys
import json
import string
import time
import threading
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text
from tqdm import tqdm
import statistics
# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Index, Base

BASE_URL = "https://api.polygon.io/v3/reference/tickers"
AGGS_URL = "https://api.polygon.io/v2/aggs/ticker"

class TokenBucketRateLimiter:
    def __init__(self, rate_per_sec):
        self.delay = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_call = 0

    def wait(self):
        with self.lock:
            now = datetime.now().timestamp()
            if self.next_call > now:
                time.sleep(self.next_call - now)
            self.next_call = max(self.next_call, now) + self.delay

# --- ONLY ADDING NECESSARY METADATA HELPERS ---

def get_ticker_cik_and_name(ticker):
    """Hits the V3 endpoint to grab the CIK for the proxy."""
    url = f"{BASE_URL}/{ticker}"
    try:
        resp = requests.get(url, params={"apiKey": POLY_KEY})
        if resp.status_code == 200:
            res = resp.json().get("results", {})
            return res.get("cik"), res.get("name")
    except: pass
    return None, f"Proxy {ticker}"
def get_aggs(ticker, start_date="2023-03-13", end_date="2023-03-24"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"sort": "asc", "limit": 120, "apiKey": POLY_KEY}
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        if "results" in data:
            return {
                datetime.fromtimestamp(r['t']/1000).strftime('%Y-%m-%d'): {
                    "c": r['c'], 
                    "v": r.get('v', 0)
                } 
                for r in data['results']
            }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return {}

def compute_local_mapping(PROXY_MAP):
    # This will now return the list of payloads for the DB
    upsert_payloads = []
    
    header = f"{'DATE':<12} | {'INDEX':<8} | {'P TICK':<6} | {'IDX PRICE':<10} | {'P PRICE':<10} | {'P VOL':<12} | {'RATIO'}"
    print(f"\n{header}")
    print("-" * 85)

    for idx_ticker, proxy_ticker in PROXY_MAP.items():
        idx_data = get_aggs(idx_ticker)
        proxy_data = get_aggs(proxy_ticker)

        common_dates = sorted(set(idx_data.keys()) & set(proxy_data.keys()))

        if not common_dates:
            if idx_ticker == "I:RUT":
                print(f"{idx_ticker:<21} | {proxy_ticker:<6} | NOTE: RUT often requires specific permissions.")
            else:
                print(f"{idx_ticker:<21} | {proxy_ticker:<6} | NO OVERLAPPING DATA FOUND")
            continue

        ratios = []
        for date in common_dates:
            i_price = idx_data[date]['c']
            p_price = proxy_data[date]['c']
            p_vol = proxy_data[date]['v']
            ratio = i_price / p_price
            ratios.append(ratio)

            print(f"{date:<12} | {idx_ticker:<8} | {proxy_ticker:<6} | {i_price:<10.2f} | {p_price:<10.2f} | {p_vol:<12,.0f} | {ratio:.4f}")
        
        # Calculate median ratio for the multiplier
        median_mult = statistics.median(ratios)
        cik, official_name = get_ticker_cik_and_name(proxy_ticker)
        
        upsert_payloads.append({
            "theoretical_symbol": idx_ticker,
            "tradable_proxy": proxy_ticker,
            "name": official_name,
            "cik": cik,
            "proxy_multiplier": median_mult
        })
        
        print("-" * 85)
    
    return upsert_payloads

# --- THE UPSERT FUNCTION ---

def upsert_to_db(data_list):
    if not data_list: return
    with SessionLocal() as session:
        for row in data_list:
            stmt = pg_insert(Index).values(row)
            # Conflict on theoretical_symbol only to allow shared iShares CIKs
            stmt = stmt.on_conflict_do_update(
                index_elements=["theoretical_symbol"], 
                set_={
                    "name": stmt.excluded.name, 
                    "tradable_proxy": stmt.excluded.tradable_proxy,
                    "cik": stmt.excluded.cik,
                    "proxy_multiplier": stmt.excluded.proxy_multiplier
                }
            )
            session.execute(stmt)
        session.commit()
        print(f"\n[DB SUCCESS] Upserted {len(data_list)} records with live CIKs and Multipliers.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python script.py params.json")

    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    PROXY_MAP = params.get("PROXY_MAP", {})
    
    # Ensure Table Schema is applied (CIK UNIQUE constraint must be gone)
    Base.metadata.create_all(engine)

    # 1. Compute and print the table
    payloads = compute_local_mapping(PROXY_MAP)
    
    # 2. Upload the goddamn data
    upsert_to_db(payloads)