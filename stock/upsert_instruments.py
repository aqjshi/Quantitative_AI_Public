import os
import requests
import sys
import json
import string
import time
import threading
import re
import datetime
import random
import multiprocessing
import functools

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from multiprocessing import Process, Queue

# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine, DATABASE_URL
from core.models import Instrument, Base, TickerMap, Dividend, UniverseMembership
from core.sieve import TokenBucketRateLimiter, power_db_worker
from core.point_in_time import consolidate_and_link_history, resolve_history, TickerDiscovery
from stock.dividends import fetch_and_map_dividends


# -------------------------------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------------------------------
def load_configuration():
    """Parses command line arguments and loads configuration parameters."""
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            params = json.load(f)
    else:
        params = {}

    config = {
        "case_study": params.get("case_study", []),
        "instrument_universe_size": params.get("instrument_universe_size", 100),
        "start_pit": params.get("train_start", "2004-01-01").split(' ')[0],
        "end_pit": params.get("train_end", datetime.now().strftime("%Y-%m-%d")).split(' ')[0],
        "universe_weights": params.get("universe_weights", {}),
        "rebalance_months": params.get("frequency_reconstruction_in_months", 3),
        "seed_id": 42
    }
    return config

# -------------------------------------------------------------------------
# 2. Discovery Logic
# -------------------------------------------------------------------------
def run_broad_discovery(discovery, universe_weights):
    """Performs the initial recursive fetch for candidate tickers."""
    iterable_weights = universe_weights.items() if isinstance(universe_weights, dict) else universe_weights
    
    for ticker_type, weight in iterable_weights:
        if weight > 0:
            try:
                print(f" [*] Discovering: {ticker_type} (Weight: {weight})")
                discovery.fetch_recursive(active=True, ticker_type=ticker_type)
                discovery.fetch_recursive(active=False, ticker_type=ticker_type)
            except Exception as e:
                print(f" [!] Error processing universe type {ticker_type}: {e}")

def process_case_studies(discovery, case_study_tickers, start_pit, end_pit):
    """
    Performs dense probing for case study tickers and initializes the membership log.
    Returns a set of FIGIs belonging to case studies and the initialized membership log.
    """
    print("--- 1. Running Strategy 1 Sampling (Case Studies) ---")
    
    # A. Probing Case Study Tickers (Dense Search)
    for t in case_study_tickers:
        # Find if we already have it from Broad Discovery
        figi_key = next((f for f, m in discovery.found_instruments.items() if m['ticker'] == t), None)
        
        needs_probe = True
        if figi_key:
            current_meta = discovery.found_instruments[figi_key]
            if current_meta.get('list_date'):
                needs_probe = False
                print(f" [V] {t} already has valid list_date: {current_meta['list_date']}")
            else:
                print(f" [!] {t} exists but missing list_date. initiating DENSE probe...")

        if needs_probe:
            probe_dates = []
            curr = datetime.strptime(start_pit, "%Y-%m-%d")
            end_dt = datetime.strptime(end_pit, "%Y-%m-%d")
            
            # DENSE PROBE (30 DAYS)
            while curr < end_dt:
                probe_dates.append(curr.strftime("%Y-%m-%d"))
                curr += timedelta(days=30) 

            found_start_date = None
            found_figi = None
            found_state = None

            for probe_date in probe_dates:
                state = discovery.get_pit_state(t, probe_date)
                figi = state.get('composite_figi')
                
                if figi not in [None, "NONE", "UNKNOWN"]:
                    found_figi = figi
                    found_state = state
                    found_start_date = state.get('list_date') or probe_date
                    print(f" [+] Found {t} alive on {probe_date} (Effective Start: {found_start_date})")
                    break 

            if found_figi:
                if figi_key and figi_key != found_figi:
                    del discovery.found_instruments[figi_key]
                
                discovery.found_instruments[found_figi] = {
                    "ticker": t, 
                    "is_active": found_state.get('active', True), 
                    "list_date": found_start_date,
                    "name": found_state.get('name', 'Unknown'), 
                    "locale": "us", 
                    "market": "stocks",
                    "cik": found_state.get('cik'), 
                    "composite_figi": found_figi, 
                    "type": "CS"
                }
            elif figi_key:
                print(f" [!] Probe failed for {t}, patching existing record...")
                if not discovery.found_instruments[figi_key].get('list_date'):
                     discovery.found_instruments[figi_key]['list_date'] = start_pit 

    # B. Lock Case Studies into Membership Log
    membership_log = {}
    case_study_figis = set()
    
    for figi, meta in discovery.found_instruments.items():
        if meta['ticker'] in case_study_tickers:
            case_study_figis.add(figi)
            if figi not in membership_log: membership_log[figi] = []
            
            ipo_date_str = meta.get('list_date')
            
            if ipo_date_str and ipo_date_str > start_pit:
                entry_dt = ipo_date_str
            else:
                entry_dt = start_pit
                
            delist_raw = meta.get('delisted_utc')
            exit_dt = delist_raw.split('T')[0] if delist_raw else None
            
            membership_log[figi].append({
                'entry': entry_dt,
                'exit': exit_dt 
            })
        
    print(f" [+] Locked {len(case_study_figis)} Case Study Instruments.")
    return case_study_figis, membership_log

def generate_membership_intervals(discovery, membership_log, case_study_figis, config):
    """
    Generates random sampling intervals for the general universe, rebalancing periodically.
    Updates membership_log in place and filters discovery.found_instruments.
    """
    curr_date = datetime.strptime(config["start_pit"], "%Y-%m-%d")
    final_end_dt = datetime.strptime(config["end_pit"], "%Y-%m-%d")
    full_pool = list(discovery.found_instruments.items())

    while curr_date <= final_end_dt:
        entry_str = curr_date.strftime("%Y-%m-%d")
        next_hop = curr_date + relativedelta(months=config["rebalance_months"])
        exit_dt = next_hop - timedelta(days=1)
        exit_str = exit_dt.strftime("%Y-%m-%d")
        
        alive_candidates = []
        for figi, meta in full_pool:
            if figi in case_study_figis: continue 
            if meta.get('type') != 'CS': continue 

            l_date = meta.get('list_date')
            d_date = meta.get('delisted_utc')
            start_valid = True
            if l_date and l_date > entry_str: start_valid = False
            
            end_valid = True
            if d_date:
                d_date_clean = d_date.split('T')[0]
                if d_date_clean < entry_str: end_valid = False
            elif not meta.get('is_active', False) and not d_date:
                 if not l_date: end_valid = False

            if start_valid and end_valid:
                alive_candidates.append(figi)
        
        sample_k = min(len(alive_candidates), config["instrument_universe_size"])
        if sample_k > 0:
            period_sample = random.sample(alive_candidates, sample_k)
            for figi in period_sample:
                if figi not in membership_log: membership_log[figi] = []
                membership_log[figi].append({
                    'entry': entry_str,
                    'exit': exit_str
                })
        curr_date = next_hop

    # Filter discovery instruments to only those selected
    selected_figis = set(membership_log.keys())
    discovery.found_instruments = {k: v for k, v in discovery.found_instruments.items() if k in selected_figis}
    
    print(f" [+] Final Universe: {len(discovery.found_instruments)} instruments.")
    return membership_log

# -------------------------------------------------------------------------
# 3. History Reconstruction & Refinement
# -------------------------------------------------------------------------
def reconstruct_history(discovery, config):
    """Fetches full historical ticker data for the selected universe."""
    print(f"--- 2. Reconstructing History ---")
    
    universe = list(set(item['ticker'] for item in discovery.found_instruments.values()))
    metadata_map = {}
    for item in discovery.found_instruments.values():
        if item['ticker'] not in metadata_map or item['is_active']:
            metadata_map[item['ticker']] = item

    full_history_map = {} 
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(resolve_history, t, metadata_map.get(t, {}), config["start_pit"], config["end_pit"], discovery): t 
            for t in universe
        }
        for future in tqdm(as_completed(futures), total=len(universe), desc="Stitching DNA"):
            try:
                full_history_map[futures[future]] = future.result()
            except Exception as e:
                print(f" [!] Failed: {e}")
    return full_history_map

def refine_membership_dates(membership_log, full_history_map):
    """Refines membership entry/exit dates based on precise historical data."""
    print("--- 2.5 Refining Membership Dates with Precise History ---")
    
    precise_starts = {}
    precise_ends = {} 

    for ticker_res in full_history_map.values():
        for seg in ticker_res:
             figi = seg.get('figi')
             start = seg.get('valid_from')
             end = seg.get('valid_to')

             if not figi: continue
             
             # 1. Earliest Start Logic
             if start:
                 if figi not in precise_starts:
                     precise_starts[figi] = start
                 else:
                     if start < precise_starts[figi]:
                         precise_starts[figi] = start
            
             # 2. Latest End Logic
             if figi not in precise_ends:
                 precise_ends[figi] = end
             else:
                 current_max = precise_ends[figi]
                 if current_max is not None:
                     if end is None: 
                         precise_ends[figi] = None
                     elif end > current_max:
                         precise_ends[figi] = end

    patched_count = 0
    for figi, intervals in membership_log.items():
        # Patch Entry
        if figi in precise_starts:
            true_start = precise_starts[figi]
            for iv in intervals:
                if iv['entry'] > true_start:
                     iv['entry'] = true_start
                     patched_count += 1

        # Patch Exit
        if figi in precise_ends:
            true_end = precise_ends[figi]
            for iv in intervals:
                if iv['exit'] is None and true_end is not None:
                     iv['exit'] = true_end
                     patched_count += 1
                     if figi == "BBG00YXXW5X3": 
                        print(f" [FIX] Backfilled CCIV exit to {true_end}")
    
    print(f" [+] Refined {patched_count} membership intervals using precise history.")

# -------------------------------------------------------------------------
# 4. Database Ingestion
# -------------------------------------------------------------------------
def prepare_database():
    """Drops existing tables and creates fresh schemas."""
    tables_to_drop = ["universe_membership", "instruments", "ticker_map", "dividends"]
    with engine.connect() as conn:
        conn.execute(text("COMMIT"))
        for table in tables_to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            conn.commit()
            print(f"[+] Cleaned {table}")
    Base.metadata.create_all(engine)

def start_db_worker(db_url):
    """Starts the background DB worker process."""
    result_queue = multiprocessing.Queue(maxsize=100)
    db_proc = multiprocessing.Process(target=power_db_worker, args=(result_queue, db_url), daemon=True)
    db_proc.start()
    return db_proc, result_queue

def ingest_instruments(discovery, result_queue):
    """Prepares and queues instrument data for insertion."""
    print(" [*] Phase A: Inserting Instruments (Static Master Data)...")
    instrument_batch = []
    
    for figi, meta in discovery.found_instruments.items():
        cik_raw = meta.get('cik')
        cik_val = int(cik_raw) if cik_raw and str(cik_raw).isdigit() else None
        
        row = (
            meta.get('ticker'), 
            meta.get('name', '')[:255], 
            meta.get('market', 'stocks'), 
            meta.get('locale', 'us'),
            meta.get('primary_exchange', ''), 
            meta.get('type', 'CS'), 
            meta.get('is_active', False), 
            meta.get('currency_name', 'USD'),
            cik_val, 
            meta.get('composite_figi', 'NONE'), 
            meta.get('share_class_figi', 'NONE')
        )
        instrument_batch.append(row)
    
    inst_copy_sql = """
        COPY instruments (ticker, name, market, locale, primary_exchange, type, active, currency_name, cik, composite_figi, share_class_figi) 
        FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
    """
    if instrument_batch: 
        result_queue.put((instrument_batch, inst_copy_sql, "Instruments Batch"))
    
    # Wait for queue to drain slightly
    while not result_queue.empty(): time.sleep(1)
    time.sleep(2)
    
    # Return mapping for next phases
    figi_id_map = {}
    with engine.connect() as conn:
        result = conn.execute(text("SELECT composite_figi, id FROM instruments"))
        for r in result: figi_id_map[r[0]] = r[1]
    
    return figi_id_map

def ingest_membership(membership_log, figi_id_map, result_queue, seed_id):
    """Queues universe membership data."""
    print(" [*] Phase A.5: Inserting Universe Membership...")
    membership_batch = []
    
    for figi, intervals in membership_log.items():
        inst_id = figi_id_map.get(figi)
        if not inst_id: continue
        
        for iv in intervals:
            membership_batch.append((
                inst_id, 
                iv['entry'], 
                iv['exit'], 
                seed_id
            ))

    mem_copy_sql = """
        COPY universe_membership (instrument_id, entry_date, exit_date, seed_id)
        FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
    """
    if membership_batch:
        chunk_size = 5000
        for i in range(0, len(membership_batch), chunk_size):
            result_queue.put((membership_batch[i:i + chunk_size], mem_copy_sql, f"Membership Chunk {i}"))
            
    while not result_queue.empty(): time.sleep(1)

def ingest_ticker_map(full_history_map, figi_id_map, result_queue):
    """Consolidates history and queues ticker map data. Returns router cache for dividends."""
    print(" [*] Phase C: Inserting Ticker Map...")
    consolidated_batch = consolidate_and_link_history(full_history_map)
    tm_batch = []
    ticker_router_cache = {} 

    for seg in consolidated_batch:
        inst_id = figi_id_map.get(seg['figi'])
        if not inst_id: continue

        t_str = seg['ticker']
        if t_str not in ticker_router_cache:
            ticker_router_cache[t_str] = []
        
        end_date_val = seg['valid_to'] or "2099-12-31"

        ticker_router_cache[t_str].append({
            'start': seg['valid_from'],
            'end': end_date_val,
            'inst_id': inst_id
        })

        tm_batch.append((
            inst_id, seg['previous_ticker'],  seg['ticker'], seg.get('previous_composite_figi'),
            seg['figi'], seg['valid_from'], seg['valid_to'], seg.get('change_event', 'IPO')
        ))

    tm_copy_sql = """
        COPY ticker_map (instrument_id, previous_ticker, ticker, previous_composite_figi, composite_figi, valid_from, valid_to, change_event_type) 
        FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
    """
    chunk_size = 5000
    for i in range(0, len(tm_batch), chunk_size):
        result_queue.put((tm_batch[i:i + chunk_size], tm_copy_sql, f"TickerMap Chunk {i}"))
    
    while not result_queue.empty(): time.sleep(1)
    
    return consolidated_batch, ticker_router_cache

def ingest_dividends(membership_log, figi_id_map, discovery, config, result_queue):
    """
    Fetches and queues dividend data based strictly on Universe Membership intervals.
    Applies Null catchers to ensure open-ended intervals default to the simulation start/end.
    """
    print(" [*] Phase D: Mapping & Ingesting Dividends (Based on Membership)...")
    all_divs = []
    
    # Pre-calculate the "Router" list for each ticker based on Membership, not Ticker Map
    # Structure: { Ticker: [ {start, end, inst_id}, ... ] }
    membership_router = {}

    for figi, intervals in membership_log.items():
        # 1. Get Instrument ID
        inst_id = figi_id_map.get(figi)
        if not inst_id: continue

        # 2. Get Ticker Name (Required for API)
        meta = discovery.found_instruments.get(figi)
        if not meta: continue
        ticker = meta['ticker']

        if ticker not in membership_router:
            membership_router[ticker] = []

        # 3. Build Intervals with Null Catchers
        for iv in intervals:
            # Null Catcher: Begin Date -> Start PIT
            start_date = iv['entry'] if iv['entry'] else config['start_pit']
            
            # Null Catcher: End Date -> End PIT
            end_date = iv['exit'] if iv['exit'] else config['end_pit']

            membership_router[ticker].append({
                'start': start_date,
                'end': end_date,
                'inst_id': inst_id
            })

    # 4. Threaded Execution
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                fetch_and_map_dividends, 
                t, 
                membership_router[t], # Pass the membership-based router
                POLY_KEY, 
                config["start_pit"], 
                config["end_pit"]
            ): t 
            for t in membership_router.keys()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Dividends"):
            ticker = futures[future]
            try:
                data = future.result()
                if data: all_divs.extend(data)
            except Exception as e:
                print(f" [!] Error collecting divs for {ticker}: {e}")

    div_copy_sql = """
        COPY dividends (
             instrument_id, external_id, ticker, record_date,  pay_date, 
            declaration_date, ex_dividend_date, frequency, cash_amount, 
            currency, distribution_type,  historical_adjustment_factor,   split_adjusted_cash_amount 
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
    """
    
    if all_divs:
        chunk_size = 5000
        for i in range(0, len(all_divs), chunk_size):
            result_queue.put((all_divs[i:i + chunk_size], div_copy_sql, f"Dividends Chunk {i}"))
    else:
        print(" [!] No dividends found for the selected universe/dates.")


# -------------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    start_time = time.time()
    
    # 1. Config
    config = load_configuration()

    # 2. Setup Sieve
    limiter = TokenBucketRateLimiter(rate_per_sec=30)
    discovery = TickerDiscovery(POLY_KEY, limiter)
    
    # 3. Discovery Phases
    run_broad_discovery(discovery, config["universe_weights"])
    
    case_study_figis, membership_log = process_case_studies(
        discovery, config["case_study"], config["start_pit"], config["end_pit"]
    )
    
    # Updates membership_log and filters discovery in-place
    generate_membership_intervals(
        discovery, membership_log, case_study_figis, config
    )

    # 4. Reconstruction
    full_history_map = reconstruct_history(discovery, config)
    refine_membership_dates(membership_log, full_history_map)

    # 5. Database Ingestion
    print("--- 3. Database Ingestion ---")
    prepare_database()
    
    db_proc, result_queue = start_db_worker(DATABASE_URL)
    
    try:
        # A. Instruments
        figi_id_map = ingest_instruments(discovery, result_queue)
        
        # B. Membership
        ingest_membership(membership_log, figi_id_map, result_queue, config["seed_id"])
        
        # C. Ticker Map
        # Note: We capture consolidated_batch for reporting, but we ignore ticker_router_cache now
        consolidated_batch, _ = ingest_ticker_map(full_history_map, figi_id_map, result_queue)
        
        # D. Dividends (UPDATED: Now passes membership_log instead of ticker_router_cache)
        ingest_dividends(membership_log, figi_id_map, discovery, config, result_queue)
        
        # Signal End
        result_queue.put(None)
        db_proc.join()
        
    except Exception as e:
        print(f" [!] Critical Error during ingestion: {e}")
        if db_proc.is_alive():
            db_proc.terminate()
        raise

    # 6. Reporting
    end_time = time.time()
    duration = end_time - start_time
    total_unique_tickers = len(discovery.found_instruments)
    
    print("\n" + "="*50)
    print("[SUCCESS] Master Data, Membership & Dividends Ingestion Complete.")
    print(f"Total Unique Tickers:   {total_unique_tickers}")
    print(f"Total Instruments:      {len(discovery.found_instruments)}")
    print(f"Universe Intervals:     {sum(len(v) for v in membership_log.values())} (Rebalancing Events)")
    print(f"Total Mapping Debt:     {len(consolidated_batch)} segments")
    print(f"Yearly Sample Size:     {config['instrument_universe_size']}")
    print(f"Total Processing Time:  {duration:.2f} seconds")
    print(f"Avg Time per Ticker:    {(duration/total_unique_tickers):.4f}s" if total_unique_tickers > 0 else "")
    print("="*50)