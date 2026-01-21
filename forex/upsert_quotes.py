import sys
import os
import json
import requests
from sqlalchemy import create_engine, select
from datetime import timedelta, datetime



import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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




# Project Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import DATABASE_URL, POLY_KEY
from core.models import Base, Company, Quote
import core.sieve as sieve
import random
import queue
from tqdm import tqdm


import psycopg
import multiprocessing

from multiprocessing import Process, Queue


def power_db_worker(result_queue, db_url):
    # Standardize DSN for psycopg
    dsn = db_url.replace("postgresql+psycopg2://", "postgresql://")
    
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            print("[!] DB Firehose Process: Connected (RELIABLE CSV MODE).")
            while True:
                item = result_queue.get()
                if item is None: break 
                
                batch_data, info = item
                try:
                    with conn.cursor() as cur:
                        # CSV/Text is robust. It won't de-sync on a single bad byte.
                        copy_query = """
                                COPY forex_quotes (ticker, t, o, h, l, c, n, v, vw) 
                                FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
                            """
                        with cur.copy(copy_query) as copy:
                            for row in batch_data:
                                # Join by tabs, convert all to string
                                line = "\t".join(map(str, row)) + "\n"
                                copy.write(line)
                except Exception as e:
                    print(f"[!] BATCH ERROR ({info}): {e}")
    except Exception as e:
        print(f"CRITICAL: Firehose died: {e}")
        os._exit(1)



def get_date_chunks(start_str, end_str, day_step=30):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    
    chunks = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=day_step), end)
        chunks.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
    return chunks



def fetcher(api_key, start_date, end_date,limiter, second_level=False):
    def fetch_all_pages(ticker):
        all_results = []
        
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}" 
        params = {
            "adjusted": "true",
            "limit": 50000,
            "sort": "asc",
            "apiKey": api_key
        }
        while url:
            try:
                # CALL THE LIMITER HERE
                limiter.wait()
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                
                results = data.get("results", [])
                all_results.extend(results)
                
                url = data.get("next_url")
                if url:
                    if "apiKey" not in url:
                        url += f"&apiKey={api_key}"
                    params = {} 
                else:
                    break # Window is fully drained
                
            except Exception as e:
                print(f" [!] Error: {e}")
                break
        return all_results
    return fetch_all_pages




def main():
    table_name="forex_quotes"
    os.makedirs(f"forex/{table_name}", exist_ok=True)
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    # Extracting the end date (e.g., "2025-12-31")
    # Using train_end as the cutoff for the API query
    start_date_cutoff = params.get("train_start", "2024-01-01").split(' ')[0]
    end_date_cutoff = params.get("train_end", "2025-12-31").split(' ')[0]
    limit = params.get("limit", 5000)
    alpha = params.get("alpha", .05)
    engine = create_engine(DATABASE_URL)

    # 1. Clean up target table
    limiter = TokenBucketRateLimiter(rate_per_sec=5)

    # 2. Ingestion Loop
    date_windows = get_date_chunks(start_date_cutoff, end_date_cutoff, day_step=30)
    ticker_chunk_size = 10 # Adjust based on your API limits
    ticker_hierarchical_topology  = params.get("ticker_hierarchical_topology", 10) 



    sampling_size = min(5, len(ticker_hierarchical_topology))
    discovery_tickers = random.sample(ticker_hierarchical_topology, sampling_size)

    sample_start, sample_end =  date_windows[0]
    # 2. Initialize the dynamic fetcher with our date logic
    fetcher_func = fetcher(POLY_KEY, sample_start, sample_end, limiter, second_level=False)

    print(f"[*] Phase 1: Policy Discovery (Sampling {sampling_size} tickers)...")
    policy, raw_cache = sieve.generate_sieve_policy(
        identifiers=discovery_tickers, 
        fetcher_func=fetcher_func, 
        alpha=alpha, 
        table_name=table_name,
        output_dir="forex"
    )
    # --- PASS 1: EXPLORATION & AUDIT ---
    print(f"[*] Starting Global Sieve Discovery (End Date: {end_date_cutoff})...")

    sieve.download_saturation(raw_cache, table_name, policy, limit, alpha, output_dir=f"forex/{table_name}")
    del raw_cache

    with engine.connect() as conn:
        # Use .strip() and .upper() on the database side or Python side
        # to ensure 'GOOG ' matches 'GOOG'
        ticker_to_cik = {
            str(row.ticker).strip().upper(): row.cik 
            for row in conn.execute(select(Company.ticker, Company.cik))
        }

    print(f"[*] Map Verified: {len(ticker_to_cik)} tickers loaded.")

    # print(ticker_to_cik.keys())
    
    # print(ticker_to_cik.values())

    target_table = Base.metadata.tables.get('forex_quotes')
    if target_table is not None:
        target_table.drop(engine, checkfirst=True)
        print("[+] Old 'forex_quotes' table cleared.")
    
    Base.metadata.create_all(engine)



    result_queue = multiprocessing.Queue(maxsize=100)
    db_proc = multiprocessing.Process(
        target=power_db_worker, 
        args=(result_queue, DATABASE_URL),
        daemon=True
    )
    db_proc.start()

  
    ticker_hierarchical_topology = params.get("ticker_hierarchical_topology", [])

    for i in tqdm(range(0, len(ticker_hierarchical_topology), ticker_chunk_size)):
        ticker_batch = ticker_hierarchical_topology[i:i + ticker_chunk_size]
        
        # We only call the fetcher ONCE per ticker for the WHOLE date range
        current_fetcher = fetcher(POLY_KEY, start_date_cutoff, end_date_cutoff, limiter, False)
        
        raw_results = {}
        with ThreadPoolExecutor(max_workers=ticker_chunk_size) as executor:
            future_to_ticker = {executor.submit(current_fetcher, t): t for t in ticker_batch}
            for future in as_completed(future_to_ticker):
                raw_results[future_to_ticker[future]] = future.result()
            
         
            binary_batch = []
            for ticker, results in raw_results.items():
                lookup_key = str(ticker).strip().upper()

                if not results:
                    continue

                ticker_rows = 0
                for r in results:
                    try:
                        # 2. Defensive extraction
                        t_val = r.get('t')
                        o_val = r.get('o')
                    
                        row = (
                                str(lookup_key),               # Maps to 'ticker'
                                int(t_val),                    # Maps to 't'
                                float(o_val),                  # Maps to 'o'
                                float(r.get('h', 0.0)),        # Maps to 'h'
                                float(r.get('l', 0.0)),        # Maps to 'l'
                                float(r.get('c', 0.0)),        # Maps to 'c'
                                float(r.get('n', 0.0)),        # Maps to 'n'
                                float(r.get('v', 0.0)),        # Maps to 'v'
                                float(r.get('vw', 0.0))        # Maps to 'vw'
                            )
                        binary_batch.append(row)
                        ticker_rows += 1
                    except Exception as e:
                        print(f"[!] CAST ERROR: {lookup_key}: {e}")
                        continue 
                
                # print(f"[*] Prepared {ticker_rows} rows for {lookup_key}")
            
            if binary_batch:
                # We send the info string so the worker can report it if it fails at the DB level
                result_queue.put((binary_batch, f"{end_date_cutoff} | Tickers: {list(raw_results.keys())}"))
   
    # 5. SHUTDOWN
    print("[*] All fetches complete. Closing pipe...")
    result_queue.put(None)
    db_proc.join()
    
    if db_proc.exitcode != 0:
        print("FATAL: DB Process crashed.")
        sys.exit(1)

    print("\n[SUCCESS] Ingestion complete via Binary Firehose.")

if __name__ == "__main__":
    main()