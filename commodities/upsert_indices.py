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

# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Instrument, Base

BASE_URL = "https://api.massive.com/v3/reference/tickers"
AGGS_URL = "https://api.polygon.io/v2/aggs/ticker"
# https://api.massive.com/v3/reference/tickers?type=OTHER&search=A&active=true&order=asc&limit=100&sort=ticker&apiKey=0pEnqPfhRk7bYu5LSa7ppMuKfqr21kck
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

class IndexDiscovery:
    def __init__(self, api_key, tree_path, limiter):
        self.api_key = api_key
        self.tree_path = tree_path
        self.limiter = limiter
        self.found_instruments = {}
        self.prefix_leaves = [] 
        self.shadow_regex = re.compile(
            r"(EX-|\d{2,4}$|[A-Z]\d{1,2}$|INTRA-DAY|INAV|HEDGED)", 
            re.IGNORECASE
        )
        self.shadow_keywords = {
            "TR", "NTR", "TOTAL RETURN", "NET RETURN", "GROSS RETURN", 
            "CAPITAL REBATE", "FDS", "FDR", "CURRENCY HEDGED", "EX-DIVIDEND", 
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN" , "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
        }

    def is_shadow_index(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        n_upper = (name or "").upper()
        combined = f"{t_upper} {n_upper}"


        if self.shadow_regex.search(combined):
            return True

        words = set(combined.split())
        if not words.isdisjoint(self.shadow_keywords):
            return True

        return False

    def is_junk(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        n_upper = (name or "").upper()
        clean_t = t_upper.replace("I:", "") # Define this EARLY
        

                  
        # If it's not a master, it must be a 3-4 char Alpha code (e.g., BKX, SOX, HGX)
        if len(clean_t) <= 4 and not any(c.isdigit() for c in clean_t):
            return self.is_shadow_index(t_upper, n_upper)

        
    
        return True
    


    def get_next_prefix(self, prefix):
        if not prefix: 
            return None
        
        last_char = prefix[-1].upper()
        

        if last_char == 'Z':
            return None 
            
        # Standard alphabetic increment (A -> B)
        return prefix[:-1] + chr(ord(last_char) + 1)

    def process_results(self, results, is_active):
        for item in results:
            ticker = item.get('ticker', '')
            name = (item.get('name', '') or '').upper()
            cik = item.get('cik') # Grab CIK for PIT mapping
            
            if ticker and not self.is_junk(ticker, name):
                # Using ticker as key, but storing 'active' status
                self.found_instruments[ticker] = {
                    "theoretical_symbol": ticker, 
                    "name": name,
                    "cik": cik,
                    "is_active": is_active
                }

    def fetch_recursive(self, prefix="", ticker_type="CS", depth=5, active=True):
        if prefix and prefix[0].isdigit(): return
        if depth <= 0: return

        next_prefix = self.get_next_prefix(prefix)
        self.limiter.wait()
        
        params = {
            "type": ticker_type,
            "ticker.gte": prefix if prefix else "A",
            "active": "true" if active else "false", # THE TOGGLE
            "sort": "ticker",
            "limit": 1000,
            "apiKey": self.api_key
        }
        if next_prefix: params["ticker.lt"] = next_prefix

        try:
            resp = requests.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            
            # Pass the active status down
            self.process_results(results, active)

            if len(results) == 1000:
                for char in string.ascii_uppercase:
                    self.fetch_recursive(prefix + char, ticker_type, depth - 1, active)
        except Exception as e:
            print(f" [!] Error at {prefix} (Active={active}): {e}")
    def validate_ticker(self, ticker, start_date="2024-01-01", end_date = "2024-01-10"):
        self.limiter.wait()
        url = f"{AGGS_URL}/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"sort": "asc", "limit": 10, "apiKey": self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200 and resp.json().get("results"):
                return ticker
        except: pass
        return None

    def crawl(self):
        self.fetch_recursive("")


import random

if __name__ == "__main__":
    K = 100  # Number of stocks to sample from EACH group (Alive/Dead)
    limiter = TokenBucketRateLimiter(rate_per_sec=20)
    discovery = IndexDiscovery(POLY_KEY, "expansion_tree.json", limiter)

    # 1. THE CRAWLS
    print("--- Crawling Active Tickers ---")
    discovery.fetch_recursive(active=True)
    alive_pool = [t for t, m in discovery.found_instruments.items() if m['is_active']]

    print("--- Crawling Delisted Tickers ---")
    discovery.fetch_recursive(active=False)
    dead_pool = [t for t, m in discovery.found_instruments.items() if not m['is_active']]

    # 2. THE STRATIFIED SAMPLE
    # Sample K from alive, K from dead (or the max available)
    sampled_alive = random.sample(alive_pool, min(K, len(alive_pool)))
    sampled_dead = random.sample(dead_pool, min(K, len(dead_pool)))
    
    sample_universe = sampled_alive + sampled_dead
    print(f" [+] Stratified Sample Created: {len(sample_universe)} tickers")
    print(f"     (Alive: {len(sampled_alive)}, Dead: {len(sampled_dead)})")

    # 3. VALIDATION (ONLY FOR THE SAMPLE)
    validated_tickers = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Check if they existed in your specific backtest window
        futures = {executor.submit(discovery.validate_ticker, t, "2024-01-01", "2024-01-10"): t for t in sample_universe}
        for future in tqdm(as_completed(futures), total=len(sample_universe), desc="Sampling Proof of Life"):
            res = future.result()
            if res:
                validated_tickers.append(res)

 
    # Separate those who passed vs those who failed
    passed_set = set(validated_tickers)
    failed_set = set(sample_universe) - passed_set

    alive_passed = [t for t in sampled_alive if t in passed_set]
    dead_passed = [t for t in sampled_dead if t in passed_set]

    print("\n" + "="*50)
    print("SURVIVORSHIP ANALYSIS")
    print("="*50)
    
    if alive_passed:
        one_alive = alive_passed[0]
        print(f"ALIVE & VALIDATED: {one_alive}")
        print(f"  - Name: {discovery.found_instruments[one_alive]['name']}")
        print(f"  - Status: Currently Active and traded in Jan 2024")

    print("-" * 30)

    if dead_passed:
        one_dead = dead_passed[0]
        print(f"DEAD BUT VALIDATED: {one_dead}")
        print(f"  - Name: {discovery.found_instruments[one_dead]['name']}")
        print(f"  - Status: DELISTED now, but was ALIVE in Jan 2024 (The 'Lead-Lag' Gold)")
    else:
        print("DEAD BUT VALIDATED: None found in this sample.")

    print("-" * 30)
    
    if failed_set:
        one_failed = list(failed_set)[0]
        meta = discovery.found_instruments[one_failed]
        print(f"REJECTED: {one_failed}")
        print(f"  - Reason: No price action in Jan 2024. (Likely died years ago)")
    
    print("="*50)

       # 4. PREP FOR DB
    final_data = [discovery.found_instruments[t] for t in validated_tickers]
    
    # 5. DB INGEST
    # upsert_to_db(final_data)
    print(f"\n[SUCCESS] Sampled {len(final_data)} tickers. Your DB won't explode.")
    
