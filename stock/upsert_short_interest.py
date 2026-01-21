import sys
import os
import json
import requests
from sqlalchemy import create_engine
from datetime import datetime

# Project Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import DATABASE_URL, POLY_KEY
from core.models import Base, Company, ShortInterest
import core.sieve as sieve
import random


def fetcher(api_key, end_date, limit):
    """
    Returns a fetcher function configured with a specific end_date.
    """
    def fetcher(ticker):
        url = "https://api.massive.com/stocks/v1/short-interest"
        params = {
            "ticker": ticker,
            "limit": limit,
            "settlement_date.lte": end_date,
            "sort": "settlement_date.desc",
            "apiKey": api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            # print(resp.url)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            print(f" [!] Error fetching {ticker}: {e}")
            return []
    return fetcher

def main():
    table_name="short-interest"
    os.makedirs(f"stock/{table_name}", exist_ok=True)
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    # Extracting the end date (e.g., "2025-12-31")
    # Using train_end as the cutoff for the API query
    end_date_cutoff = params.get("train_end", "2025-12-31").split(' ')[0]
    limit = params.get("limit", 5000)
    alpha = params.get("alpha", .05)
    chunk_size = params.get("chunk_size", 10)
    all_tickers  = params.get("ticker_whitelist", 10) 


    engine = create_engine(DATABASE_URL)
    
    sampling_size = min(100, len(all_tickers))
    discovery_tickers = random.sample(all_tickers, sampling_size)

    # 2. Initialize the dynamic fetcher with our date logic
    fetcher_func = fetcher(POLY_KEY, end_date_cutoff, limit)

    print(f"[*] Phase 1: Policy Discovery (Sampling {sampling_size} tickers)...")
    policy, raw_cache = sieve.generate_sieve_policy(
        identifiers=discovery_tickers, 
        fetcher_func=fetcher_func, 
        alpha=alpha, 
        table_name=table_name,
        output_dir="stock"
    )
    # --- PASS 1: EXPLORATION & AUDIT ---
    print(f"[*] Starting Global Sieve Discovery (End Date: {end_date_cutoff})...")

    sieve.download_saturation(raw_cache, table_name, policy, limit, alpha, output_dir=f"stock/{table_name}")
    del raw_cache



    # 1. Clean up target table
    target_table = Base.metadata.tables.get('short_interest')
    if target_table is not None:
        target_table.drop(engine, checkfirst=True)
        print("[+] Old 'short_interest' table cleared.")
    
    Base.metadata.create_all(engine)



    # --- THE INTERCEPT ---
    print("\n" + "="*60)
    for key in sorted(policy['key_whitelist']):
        print(f"  - {key}")
    print("="*60)
    
    input("\n[?] Review models.py. Press ENTER to start UPSERT...")

    # --- PASS 2: RE-MAP & INGEST ---
    Base.metadata.create_all(engine)

    total_synced = 0
    for i in range(0, len(all_tickers), chunk_size):
        ticker_batch = all_tickers[i:i + chunk_size]
        
        # FETCH: Data only exists in RAM for the duration of this chunk
        batch_cache = {}
        for t in ticker_batch:
            batch_cache[t] = fetcher_func(t)
        
        # UPSERT
        synced = sieve.execute_sieve_upsert(
            engine=engine,
            model=ShortInterest,
            company_model=Company,
            policy=policy,
            data_cache=batch_cache, 
            keep_extraneous=False
        )
        total_synced += synced
        
        batch_cache.clear() 
        print(f"[+] Ingested: {i + len(ticker_batch)}/{len(all_tickers)} | Total: {total_synced}")

    print(f"\n[!] SUCCESS: {total_synced} records updated.")
    sieve.upsert_health(engine, ShortInterest, table_name, policy, chunk_size=10000, output_dir=f"stock/{table_name}")
if __name__ == "__main__":
    main()