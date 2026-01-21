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

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text
from tqdm import tqdm

import psycopg
import multiprocessing

from multiprocessing import Process, Queue
import functools
# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine, DATABASE_URL
from core.models import Instrument, Base, TickerMap, Dividend, UniverseMembership
from core.sieve import TokenBucketRateLimiter, power_db_worker

from dateutil.relativedelta import relativedelta



BASE_URL = "https://api.polygon.io/v3/reference/tickers"

class TickerDiscovery:
    def __init__(self, api_key, limiter, cache_dir="stock/cache/"):
        self.api_key = api_key
        self.limiter = limiter
        self.cache_dir = cache_dir
        self.found_instruments = {} 
        self.shadow_regex = re.compile(r"(EX-|\d{2,4}$|[A-Z]\d{1,2}$|INTRA-DAY|INAV|HEDGED)", re.IGNORECASE)
        self.shadow_keywords = {"TR", "NTR", "TOTAL RETURN", "NET RETURN", "GROSS RETURN", "JAN", "FEB", "MAR", "ETF"}
        self.exchange_whitelist = {
            'XNYS', 'XNAS', 'ARCX', 'BATS', 'XTSE', 'XLON', 
            'XETR', 'XPAR', 'XAMS', 'XJPX', 'XHKG', 'XASX', 'XNSE'
        }
        
        # Ensure physical storage path exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, ticker, date_str):
        """Creates a path like stocks/AAPL/2023-01-01.json"""
        ticker_dir = os.path.join(self.cache_dir, ticker.upper())
        if not os.path.exists(ticker_dir):
            os.makedirs(ticker_dir)
        return os.path.join(ticker_dir, f"{date_str}.json")

    @functools.lru_cache(maxsize=50000) # Increased for larger universes
    def get_pit_state(self, ticker, date_str):
        cache_path = self._get_cache_path(ticker, date_str)

        # 1. Physical Disk Check
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f" [!] Cache read error for {ticker} on {date_str}: {e}")

        # 2. API Fetch (The "Network Tax" paid only once)
        self.limiter.wait()
        url = f"{BASE_URL}/{ticker}"
        params = {"date": date_str, "apiKey": self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                raw_data = resp.json().get("results", {})
                
                # Extract CIK clean
                cik = raw_data.get("cik")
                if cik: cik = str(cik).lstrip('0')
                
                parsed_state = {
                    "ticker": raw_data.get("ticker"),
                    "composite_figi": raw_data.get("composite_figi", "UNKNOWN"),
                    "share_class_figi": raw_data.get("share_class_figi", "UNKNOWN"),
                    "cik": cik,
                    "name": raw_data.get("name"),
                    "active": raw_data.get("active", False),
                    "list_date": raw_data.get("list_date")
                }

                # 3. Store Physically Immediately
                with open(cache_path, 'w') as f:
                    json.dump(parsed_state, f)

                return parsed_state
            
            # Handle 404/Missing data as a "Valid Negative" to avoid re-fetching
            elif resp.status_code == 404:
                negative_state = {"ticker": ticker, "composite_figi": "NONE", "active": False}
                with open(cache_path, 'w') as f:
                    json.dump(negative_state, f)
                return negative_state

        except Exception as e:
            print(f" [!] API Error for {ticker}: {e}")
            
        return {"ticker": ticker, "composite_figi": "NONE", "active": False}

    def is_junk(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        combined = f"{t_upper} {(name or '').upper()}"
        if self.shadow_regex.search(combined): return True
        words = set(combined.split())
        return not words.isdisjoint(self.shadow_keywords)

    def get_next_prefix(self, prefix):
        if not prefix: return None
        last_char = prefix[-1].upper()
        if last_char == 'Z': return None 
        return prefix[:-1] + chr(ord(last_char) + 1)

    def fetch_recursive(self, prefix="", ticker_type="CS", market="stocks",depth=5, active=True):
        if prefix and prefix[0].isdigit(): return
        if depth <= 0: return

        next_prefix = self.get_next_prefix(prefix) 
        self.limiter.wait()
        
        params = {
            "market": market , "type": ticker_type, "active": "true" if active else "false",
            "sort": "ticker", "limit": 1000, "apiKey": self.api_key,
            "ticker.gte": prefix if prefix else "A"
        }
        if next_prefix: params["ticker.lt"] = next_prefix

        try:
            resp = requests.get(BASE_URL, params=params)
            data = resp.json()
            results = data.get("results", [])
            for item in results:
                # 1. Grab the exchange and FIGI
                exchange = item.get('primary_exchange')
                figi = item.get('composite_figi')
                
                # 2. THE QUALITY GATE: 
                # Skip if it's not on our whitelist or if it's 'junk'
                if exchange not in self.exchange_whitelist:
                    continue
                    
                if figi and not self.is_junk(item.get('ticker'), item.get('name')):
                    self.found_instruments[figi] = {**item, "is_active": active}
            if len(results) == 1000:
                for char in string.ascii_uppercase:
                    self.fetch_recursive(prefix + char, ticker_type,market, depth - 1, active)
        except Exception as e:
            print(f" [!] Error at {prefix}: {e}")

def resolve_history(ticker, metadata, start_backtest, end_backtest, discovery_engine, heatbeat_freq_days=180):
    search_start_dt = datetime.strptime(start_backtest, "%Y-%m-%d")
    search_end_dt = datetime.strptime(end_backtest, "%Y-%m-%d")

    # 1. Standard Anchors (Start/End)
    anchors = {search_start_dt, search_end_dt}

    # 2. Metadata Hints (If available)
    if metadata.get('list_date'):
        try:
            ld = datetime.strptime(metadata['list_date'], "%Y-%m-%d")
            if search_start_dt < ld < search_end_dt: anchors.add(ld)
        except: pass
    if metadata.get('delisted_utc'):
        try:
            dd = datetime.strptime(metadata['delisted_utc'].split('T')[0], "%Y-%m-%d")
            if search_start_dt < dd < search_end_dt: anchors.add(dd)
        except: pass


    current_heartbeat = search_start_dt
    while current_heartbeat < search_end_dt:
        # Advance by 365 days (approx 1 year)
        current_heartbeat += timedelta(days=heatbeat_freq_days)
        
        # Stop if we've overshot the end date
        if current_heartbeat >= search_end_dt:
            break
            
        # Add the heartbeat to our anchors
        anchors.add(current_heartbeat)


    sorted_anchors = sorted(list(anchors))
    raw_segments = []

    def probe(d1, d2):
        s1 = discovery_engine.get_pit_state(ticker, d1)
        s2 = discovery_engine.get_pit_state(ticker, d2)

        # 1. BOTH NONE: Skip the interval (Heartbeat handles the risk of missing a 'short' life)
        if s1['composite_figi'] == "NONE" and s2['composite_figi'] == "NONE":
            return

        # 2. METADATA SNAP (The Speed Hack):
        # If it was born inside this window, snap to the IPO date and stop recursing.
        if s1['composite_figi'] == "NONE" and s2['composite_figi'] != "NONE":
            actual_ipo = s2.get('list_date')
            if actual_ipo and d1 < actual_ipo < d2: # Note: strictly less than d2
                probe(actual_ipo, d2) 
                return

        # 3. CONTINUITY: Same asset at both ends? Bridge the gap.
        if s1['composite_figi'] == s2['composite_figi'] and s1['ticker'] == s2['ticker']:
            if s1['composite_figi'] not in ["NONE", "UNKNOWN"]:
                raw_segments.append({
                    "ticker": s1['ticker'], "figi": s1['composite_figi'],
                    "valid_from": d1, "valid_to": d2
                })
            return

        # 4. RECURSION: Only if we haven't found a clean bridge or snap (e.g., Ticker Change)
        t1 = datetime.strptime(d1, "%Y-%m-%d")
        t2 = datetime.strptime(d2, "%Y-%m-%d")
        delta = (t2 - t1).days
        
        if delta <= 1:
            # Base case: we are at the edge of a change
            if s1['composite_figi'] not in ["NONE", "UNKNOWN"]:
                raw_segments.append({
                    "ticker": s1['ticker'], "figi": s1['composite_figi'],
                    "valid_from": d1, "valid_to": d1
                })
            return 
        
        mid_str = (t1 + timedelta(days=delta // 2)).strftime("%Y-%m-%d")
        probe(d1, mid_str)
        probe(mid_str, d2)

    # Probe each anchored interval
    for i in range(len(sorted_anchors) - 1):
        probe(sorted_anchors[i].strftime("%Y-%m-%d"), sorted_anchors[i+1].strftime("%Y-%m-%d"))

    if not raw_segments: return []

    unique_map = {}
    for seg in raw_segments:
        unique_map[f"{seg['valid_from']}_{seg['figi']}"] = seg
    
    cleaned = list(unique_map.values())
    cleaned.sort(key=lambda x: x['valid_from'])

    merged = []
    for seg in cleaned:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg['ticker'] == last['ticker'] and seg['figi'] == last['figi']:
            last['valid_to'] = max(last['valid_to'], seg['valid_to'])
        else:
            merged.append(seg)
    return merged

def consolidate_and_link_history(full_history_map):
    print(" [***] Running Global FIGI-Stitching...")
    all_segments = []
    for ticker_str, segments in full_history_map.items():
        all_segments.extend(segments)

    groups = {}
    for seg in all_segments:
        figi = seg.get('figi')
        if not figi or figi == 'NONE': continue
        if figi not in groups: groups[figi] = []
        groups[figi].append(seg)

    final_timeline = []
    for figi, group in groups.items():
        group.sort(key=lambda x: x['valid_from'])
        
        for i, curr in enumerate(group):
            if i == 0:
                curr['change_event'] = 'IPO'
                curr['previous_ticker'] = None
                curr['previous_composite_figi'] = None
            else:
                prev = group[i-1]
                if curr['ticker'] != prev['ticker']:
                    curr['change_event'] = 'SYMBOL_CHANGE'
                    curr['previous_ticker'] = prev['ticker']
                    curr['previous_composite_figi'] = prev['figi']
                    prev['valid_to'] = curr['valid_from']
                else:
                    curr['change_event'] = 'CONTINUATION'
                    curr['previous_ticker'] = prev['previous_ticker']
                    curr['previous_composite_figi'] = prev.get('previous_composite_figi')
            final_timeline.append(curr)

    return final_timeline


def fetch_and_map_dividends(ticker, tm_lookup, api_key, start_date, end_date):
    """
    Fetches dividends with explicit date range filters to ensure full coverage.
    """
    url = "https://api.massive.com/stocks/v1/dividends"
    
    # Using .gte and .lte to bound the search to your specific backtest window
    params = {
        "ticker": ticker,
        "limit": 1000,
        "sort": "ex_dividend_date",
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
        "apiKey": api_key
    }
    
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200: return []
        
        raw_divs = resp.json().get('results', [])
        mapped_divs = []
        
        for d in raw_divs:
            ex_date_str = d.get('ex_dividend_date')
            if not ex_date_str: continue
            
            owner_id = None
            for interval in tm_lookup:
                if interval['start'] <= ex_date_str <= interval['end']:
                    owner_id = interval['inst_id']
                    break

            if owner_id:
                mapped_divs.append((
                    owner_id,      # instrument_id
                    d.get('id'),                # external_id
                    ticker,                     # ticker snapshot
                    d.get('record_date'),       # record_date
                    d.get('pay_date'),          # pay_date
                    d.get('declaration_date'),  # declaration_date
                    ex_date_str,
                    d.get('frequency'),         # frequency
                    d.get('cash_amount'),
                    d.get('currency'),
                    d.get('distribution_type') ,
                    d.get('cash_amount'),       # amount
                    d.get('split_adjusted_cash_amount') 
                  
                ))
        return mapped_divs
    except Exception as e:
        print(f"Error fetching divs for {ticker}: {e}")
        return []

def snap_to_quarter_start(dt):
    """
    Floors a date to the first day of the quarter.
    Example: Feb 15 -> Jan 01
    """
    quarter_start_month = ((dt.month - 1) // 3) * 3 + 1
    return dt + relativedelta(month=quarter_start_month, day=1)

if __name__ == "__main__":
    # 1. Setup & Configuration
    random.seed(42)
    start_time = time.time()
  
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f: params = json.load(f)
    else: params = {}
    
    case_study = params.get("case_study", [])
    INSTRUMENT_UNIVERSE_SIZE = params.get("instrument_universe_size", 100) 
    start_pit = params.get("train_start", "2004-01-01").split(' ')[0]
    end_pit = params.get("train_end", datetime.now().strftime("%Y-%m-%d")).split(' ')[0]
    universe_weights = params.get("universe_weights", {})
    
    # Quarterly Rebalancing Frequency (90 days / 3 months)
    REBALANCE_MONTHS = params.get("frequency_reconstruction_in_months", 3) 
    SEED_ID = 42 

    limiter = TokenBucketRateLimiter(rate_per_sec=80)
    discovery = TickerDiscovery(POLY_KEY, limiter)
    
    # -------------------------------------------------------------------------
    # 0. BROAD DISCOVERY (Get candidate pool)
    # -------------------------------------------------------------------------
    for ticker_type, weight in universe_weights.items() if isinstance(universe_weights, dict) else universe_weights:
            if weight > 0:
                try:
                    print(f" [*] Discovering: {ticker_type} (Weight: {weight})")
                    discovery.fetch_recursive(active=True, ticker_type=ticker_type)
                    discovery.fetch_recursive(active=False, ticker_type=ticker_type)
                except Exception as e:
                    print(f" [!] Error processing universe type {ticker_type}: {e}")
    
    # -------------------------------------------------------------------------
    # 1. STRATEGY 1 SAMPLING (With Membership Intervals)
    # -------------------------------------------------------------------------
    print("--- 1. Running Strategy 1 Sampling ---")

    # A. Probing Case Study Tickers (Dense Search)
    for t in case_study:
        # Find if we already have it from Broad Discovery
        figi_key = next((f for f, m in discovery.found_instruments.items() if m['ticker'] == t), None)
        
        # Determine if we need to hunt for dates
        needs_probe = True
        if figi_key:
            current_meta = discovery.found_instruments[figi_key]
            if current_meta.get('list_date'):
                needs_probe = False
                print(f" [V] {t} already has valid list_date: {current_meta['list_date']}")
            else:
                print(f" [!] {t} exists but missing list_date. initiating DENSE probe...")

        # If we don't have it, or we have it but it's missing dates -> PROBE
        if needs_probe:
            probe_dates = []
            curr = datetime.strptime(start_pit, "%Y-%m-%d")
            end_dt = datetime.strptime(end_pit, "%Y-%m-%d")
            
            # --- FIX: DENSE PROBE (30 DAYS) FOR CASE STUDIES ---
            # SPACs live and die quickly. Annual probes miss them.
            while curr < end_dt:
                probe_dates.append(curr.strftime("%Y-%m-%d"))
                curr += timedelta(days=30) 
            # ---------------------------------------------------

            found_start_date = None
            found_figi = None
            found_state = None

            # Hunt for the first sign of life
            for probe_date in probe_dates:
                state = discovery.get_pit_state(t, probe_date)
                figi = state.get('composite_figi')
                
                if figi not in [None, "NONE", "UNKNOWN"]:
                    # Found it!
                    found_figi = figi
                    found_state = state
                    # If API list_date is null, use the probe_date
                    found_start_date = state.get('list_date') or probe_date
                    print(f" [+] Found {t} alive on {probe_date} (Effective Start: {found_start_date})")
                    break 

            # Update or Insert the record
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

    # B. Membership Generation Logic
    membership_log = {} 
    
    # 1. Lock Case Study
    case_study_figis = set()
    for figi, meta in discovery.found_instruments.items():
        if meta['ticker'] in case_study:
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

    # 2. Random Sampling Loop
    curr_date = datetime.strptime(start_pit, "%Y-%m-%d")
    final_end_dt = datetime.strptime(end_pit, "%Y-%m-%d")
    full_pool = list(discovery.found_instruments.items())

    while curr_date <= final_end_dt:
        entry_str = curr_date.strftime("%Y-%m-%d")
        next_hop = curr_date + relativedelta(months=REBALANCE_MONTHS)
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
        
        sample_k = min(len(alive_candidates), INSTRUMENT_UNIVERSE_SIZE)
        if sample_k > 0:
            period_sample = random.sample(alive_candidates, sample_k)
            for figi in period_sample:
                if figi not in membership_log: membership_log[figi] = []
                membership_log[figi].append({
                    'entry': entry_str,
                    'exit': exit_str
                })
        curr_date = next_hop

    selected_figis = set(membership_log.keys())
    discovery.found_instruments = {k: v for k, v in discovery.found_instruments.items() if k in selected_figis}
    universe = list(set(item['ticker'] for item in discovery.found_instruments.values()))
    
    metadata_map = {}
    for item in discovery.found_instruments.values():
        if item['ticker'] not in metadata_map or item['is_active']:
            metadata_map[item['ticker']] = item

    print(f" [+] Final Universe: {len(discovery.found_instruments)} instruments ({len(universe)} unique tickers).")

    # -------------------------------------------------------------------------
    # 2. RECONSTRUCTION
    # -------------------------------------------------------------------------
    print(f"--- 2. Reconstructing History ---")
    full_history_map = {} 
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(resolve_history, t, metadata_map.get(t, {}), start_pit, end_pit, discovery): t 
            for t in universe
        }
        for future in tqdm(as_completed(futures), total=len(universe), desc="Stitching DNA"):
            try:
                full_history_map[futures[future]] = future.result()
            except Exception as e:
                print(f" [!] Failed: {e}")

    # -------------------------------------------------------------------------
    # 2.5 REFINEMENT (THE COMPLETE FIX)
    # -------------------------------------------------------------------------
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
            
             # 2. Latest End Logic (Fixes the NULL exit date issue)
             if figi not in precise_ends:
                 precise_ends[figi] = end
             else:
                 current_max = precise_ends[figi]
                 # If we encounter a None (Active) or already have None, it stays None (Active)
                 if current_max is not None:
                     if end is None: 
                         precise_ends[figi] = None
                     elif end > current_max:
                         precise_ends[figi] = end

    # 3. Patch the Membership Log in-place
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
                # If current exit is NULL (but shouldn't be), and we have a valid end date
                if iv['exit'] is None and true_end is not None:
                     iv['exit'] = true_end
                     patched_count += 1
                     # Debug print for CCIV specific confirmation
                     if figi == "BBG00YXXW5X3": 
                        print(f" [FIX] Backfilled CCIV exit to {true_end}")
    
    print(f" [+] Refined {patched_count} membership intervals using precise history.")

    # -------------------------------------------------------------------------
    # 3. DATABASE INGESTION
    # -------------------------------------------------------------------------
    print("--- 3. Database Ingestion ---")
    
    tables_to_drop = ["universe_membership", "instruments", "ticker_map", "dividends"]
    
    with engine.connect() as conn:
        conn.execute(text("COMMIT"))
        for table in tables_to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            conn.commit()
            print(f"[+] Cleaned {table}")

    Base.metadata.create_all(engine)

    result_queue = multiprocessing.Queue(maxsize=100)
    db_proc = multiprocessing.Process(target=power_db_worker, args=(result_queue, DATABASE_URL), daemon=True)
    db_proc.start()

    # --- PHASE A: INSERT INSTRUMENTS (STATIC) ---
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
    
    # NOTE: No dates in this COPY anymore
    inst_copy_sql = """
        COPY instruments (ticker, name, market, locale, primary_exchange, type, active, currency_name, cik, composite_figi, share_class_figi) 
        FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
    """
    if instrument_batch: result_queue.put((instrument_batch, inst_copy_sql, "Instruments Batch"))
    while not result_queue.empty(): time.sleep(1)
    time.sleep(2)

    # --- PHASE B: MAP FIGI -> ID ---
    print(" [*] Phase B: Mapping FIGIs to IDs...")
    figi_id_map = {}
    with engine.connect() as conn:
        result = conn.execute(text("SELECT composite_figi, id FROM instruments"))
        for r in result: figi_id_map[r[0]] = r[1]

    # --- PHASE A.5: INSERT UNIVERSE MEMBERSHIP (DYNAMIC) ---
    print(" [*] Phase A.5: Inserting Universe Membership...")
    membership_batch = []
    
    for figi, intervals in membership_log.items():
        inst_id = figi_id_map.get(figi)
        if not inst_id: continue
        
        for iv in intervals:
            # Row: instrument_id, entry_date, exit_date, seed_id
            membership_batch.append((
                inst_id, 
                iv['entry'], 
                iv['exit'], # Can be None/Null
                SEED_ID
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

    # --- PHASE C: LINK HISTORY ---
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
    
    # --- PHASE D: DIVIDEND INGESTION ---
    print(" [*] Phase D: Mapping & Ingesting Dividends...")
    all_divs = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                fetch_and_map_dividends, t, ticker_router_cache.get(t, []), POLY_KEY, start_pit, end_pit
            ): t 
            for t in ticker_router_cache.keys()
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

    result_queue.put(None)
    db_proc.join()
    
    # 3. Calculate Final Metrics
    end_time = time.time()
    duration = end_time - start_time
    total_unique_tickers = len(universe)
    total_instrument_debt = len(consolidated_batch)
    total_instruments_tracked = len(discovery.found_instruments)
    total_membership_records = len(membership_batch)

    print("\n" + "="*50)
    print("[SUCCESS] Master Data, Membership & Dividends Ingestion Complete.")
    print(f"Total Unique Tickers:   {total_unique_tickers}")
    print(f"Total Instruments:      {total_instruments_tracked}")
    print(f"Universe Intervals:     {total_membership_records} (Rebalancing Events)")
    print(f"Total Mapping Debt:     {total_instrument_debt} segments")
    print(f"Yearly Sample Size:     {INSTRUMENT_UNIVERSE_SIZE}")
    print(f"Total Processing Time:  {duration:.2f} seconds")
    print(f"Avg Time per Ticker:    {(duration/total_unique_tickers):.4f}s" if total_unique_tickers > 0 else "")
    print("="*50)