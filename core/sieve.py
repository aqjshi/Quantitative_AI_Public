import os
import sys
import json
import requests
import pandas as pd
import io
import re
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime, UTC
from collections import Counter
from sqlalchemy import select, inspect
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm
from datetime import datetime, timezone
from core.db import POLY_KEY
import csv


from datetime import timedelta, datetime, date
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psycopg
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



def power_db_worker(result_queue, db_url):
    """
    Generic DB Worker that accepts raw SQL Copy commands.
    Input item format: (batch_data_list, sql_copy_command, info_str)
    """
    # Standardize DSN for psycopg
    dsn = db_url.replace("postgresql+psycopg2://", "postgresql://")
    
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            print("[!] DB Worker: Connected.")
            while True:
                item = result_queue.get()
                if item is None: break 
                
                batch_data, copy_query, info = item
                
                if not batch_data:
                    continue

                try:
                    with conn.cursor() as cur:
                        with cur.copy(copy_query) as copy:
                            for row in batch_data:
                                line = "\t".join(map(str, row)) + "\n"
                                copy.write(line)
                except Exception as e:
                    print(f"[!] BATCH ERROR ({info}): {e}")
    except Exception as e:
        print(f"CRITICAL: DB Worker died: {e}")
        os._exit(1)


def parse_osi(osi: str):
    s = osi[2:] if osi.startswith("O:") else osi

    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    ul = s[:i]
    rest = s[i:]

    if len(rest) < 15:
        raise ValueError(f"Bad OSI: {osi} (rest too short: '{rest}')")

    yymmdd = rest[:6]
    right  = rest[6].upper()
    kcode  = rest[7:15]

    if right not in ("C", "P"):
        raise ValueError(f"Bad OSI right: {osi}")
    if not (yymmdd.isdigit() and kcode.isdigit()):
        raise ValueError(f"Bad OSI digits: {osi}")

    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    exp = date(2000 + yy, mm, dd)
    K = int(kcode) / 1000.0
    return ul, exp, right, K


    
# --- PHASE 1: EXPLORATION & POLICY ---
def get_iso_date(val):
    """
    Standardizes date inputs into a rigid ISO string (YYYY-MM-DD).
    Essential for creating stable dictionary keys for deduplication.
    """
    if val is None:
        return "none"
    
    # Handle already processed date/datetime objects
    if isinstance(val, (date, datetime)):
        return val.strftime('%Y-%m-%d')
    
    # Handle string inputs (e.g. "2025-10-31T00:00:00Z" or "2025-10-31")
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() == "none":
            return "none"
        # Slice first 10 chars to catch 'YYYY-MM-DD' from a full timestamp string
        return val[:10]
    
    return str(val)



def get_top_2000_tickers(limit=2000):
    """
    Fetches active US common stocks directly from Polygon.
    This replaces the 'fetch_target_tickers' Wikipedia logic.
    """
    print(f"[*] Fetching {limit} tickers from Polygon Reference API...")
    
    # We filter for 'type=CS' (Common Stock) and 'active=true'
    url = f"https://api.polygon.io/v3/reference/tickers"
    params = {
        "type": "CS",
        "market": "stocks",
        "active": "true",
        "limit": 1000, # Max per page
        "apiKey": POLY_KEY
    }
    
    tickers = []
    while len(tickers) < limit:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text}")
            break
            
        data = resp.json()
        results = data.get("results", [])
        tickers.extend([r["ticker"] for r in results])
        
        # Pagination
        next_url = data.get("next_url")
        if not next_url:
            break
        url = next_url # next_url already includes the apiKey if using Polygon's format
        
    return tickers[:limit]



def fetch_target_tickers():
    """Baseline: Fetch S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(response.text))[0]
        return [re.split(r"\.", s)[0].upper() for s in df["Symbol"]]
    except Exception as e:
        print(f"[-] Ticker fetch failed: {e}")
        return []
    


def generate_sieve_policy(
    identifiers, 
    fetcher_func, 
    threshold_strategy=None,
    alpha=0.05, 
    table_name="balance-sheet",
    output_dir="stock"
):
    """
    Explores data, calculates key density, and dumps the 
    resulting 'Contract' (Policy) to a JSON file.
    """
    raw_cache = {}
    key_freq = Counter()
    
    if not threshold_strategy:
        threshold_strategy = lambda count, total: (count / total) >= alpha

    print(f"[*] Exploring {len(identifiers)} entities (Alpha: {alpha})...")

    for item in tqdm(identifiers):
        try:
            data = fetcher_func(item)
            if data:
                raw_cache[item] = data
                unique_keys = {k for record in data for k in record.keys()}
                for k in unique_keys:
                    key_freq[k] += 1
        except Exception as e:
            print(f"[!] Error fetching {item}: {e}")
            continue

    total_hits = len(raw_cache)
    if total_hits == 0:
        return None, {}

    whitelist = []
    blacklist = []
    
    for key, count in key_freq.items():
        if threshold_strategy(count, total_hits):
            whitelist.append(key)
        else:
            blacklist.append(key)

    policy = {
        "key_whitelist": whitelist,
        "key_blacklist": blacklist,
        "sample_ticker_whitelist": list(raw_cache.keys()),
        "meta": {
            "alpha": alpha, 
            "samples": total_hits,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "table_name": table_name
        }
    }

    # --- THE DUMP ---
    # Construct the path: e.g., stock/balance-sheets/sieve_policy.json
    policy_path = os.path.join(output_dir, table_name, "sieve_policy.json")
    
    with open(policy_path, 'w') as f:
        json.dump(policy, f, indent=4)
        
    print(f"[+] Sieve Policy dumped to: {policy_path}")
    
    return policy, raw_cache

# --- PHASE 2: UNIVERSAL UPSERT & HEALTH ---

def get_upsert_constraint(engine, model):
    """Dynamically detects the Unique Constraint name for the model."""
    inspector = inspect(engine)
    constraints = inspector.get_unique_constraints(model.__tablename__)
    if constraints:
        return constraints[0]['name']
    return f"uq_{model.__tablename__}_identity"



def execute_sieve_upsert(engine, model, company_model, policy, data_cache, keep_extraneous=False, to_currency_id=None, result_queue=None):
    """
    High-speed vectorized upsert that maintains Sieve Policy whitelisting 
    and extraneous data handling.
    """
    white_keys = set(policy.get("key_whitelist", []))
    model_columns = [c.name for c in model.__table__.columns if c.name != 'id']
    
    # Identify conflict keys for the "ON CONFLICT" clause
    if model.__tablename__ == 'quotes':
        conflict_keys = ['company_cik', 't']
    elif model.__tablename__ == 'short_interest':
        conflict_keys = ['company_cik', 'settlement_date']
    elif model.__tablename__ == 'forex_rates':
        # FIX: Ensure 't' is included in the conflict targets for Forex
        conflict_keys = ['from_currency_id', 'to_currency_id', 't']
    else:
        # Standard for financial statements
        conflict_keys = ['company_cik', 'filing_date', 'period_end', 'timeframe']

    all_rows = []
    
    for symbol, records in data_cache.items():
        # 2. Map ID (Ticker -> CIK OR Currency -> ID)
        with engine.connect() as conn:
            if model.__tablename__ == 'forex_rates':
                res = conn.execute(select(company_model.id).where(company_model.code == symbol)).fetchone()
                actual_id = res.id if res else None
            else:
                res = conn.execute(select(company_model.cik).where(company_model.ticker == symbol)).fetchone()
                actual_id = res.cik if res else None
            
            if not actual_id: continue

        for record in records:
            structured = {col: None for col in model_columns}
            
            # Coordinate Assignment
            if model.__tablename__ == 'forex_rates':
                structured["from_currency_id"] = actual_id
                structured["to_currency_id"] = to_currency_id 
            else:
                structured["company_cik"] = actual_id
            
            # Sieve Whitelisting
            raw_additional = {}
            for k, v in record.items():
                if k in model_columns and k in white_keys:
                    # --- FIX 1: Explicitly JSON-serialize the 'tickers' list ---
                    if k == 'tickers' and isinstance(v, (list, dict)):
                        structured[k] = json.dumps(v)
                    elif k == 't' and isinstance(v, (int, float)):
                        structured[k] = int(v // 1000) if v > 9999999999 else int(v)
                    else:
                        structured[k] = v
                elif k not in ['id', 'company_cik'] and k not in model_columns:
                    raw_additional[k] = v
                if model.__tablename__ == 'forex_rates':
                    # Map 'c' (Close) from API to 'rate' in DB
                    if k == 'c': structured['rate'] = v
                    # Map 'l' (Low) to 'bid' and 'h' (High) to 'ask'
                    if k == 'l': structured['bid'] = v
                    if k == 'h': structured['ask'] = v
            # --- FIX 2: Explicitly JSON-serialize the 'additional_data' blob ---
            if keep_extraneous and "additional_data" in model_columns:
                structured["additional_data"] = json.dumps(raw_additional)

            all_rows.append(structured)

    if not all_rows:
        return 0

    # 1. Sort all_rows by filing_date ASCENDING
    # This ensures that during the loop, the latest date is processed LAST 
    # and therefore overwrites any earlier versions in the dictionary.
    try:
        all_rows.sort(key=lambda x: str(x.get('filing_date') or '1900-01-01'))
    except Exception as e:
        print(f"[*] Sorting warning (Deduplication): {e}")

    # 2. --- THE DEDUPLICATION GATE ---
    deduped_batch = {}
    for row in all_rows:
        # Define the unique coordinates based on the model type
        if model.__tablename__ == 'short_interest':
            # Coordinate: Who + Settlement Date
            identity = (
                row['company_cik'], 
                str(row.get('settlement_date')), 
                "snapshot"
            )
        elif model.__tablename__ == 'quotes':
            # Coordinate: Who + Unix Timestamp
            identity = (
                row['company_cik'], 
                str(row.get('t')), 
                "minute"
            )
        elif model.__tablename__ == 'forex_rates':
            identity = (row['from_currency_id'], row['to_currency_id'], row['t'])
        else:
            # Standard Financial Coordinates: Who + Period + Timeframe
            identity = (
                row['company_cik'], 
                str(row.get('period_end')), 
                str(row.get('timeframe', 'none')).lower()
            )
        
        # The latest data for this specific coordinate survives
        deduped_batch[identity] = row


    if result_queue:
            result_queue.put((list(deduped_batch.values()), model.__tablename__))
            return len(deduped_batch)
    # 3. Convert back to list for the DataFrame
    final_rows = list(deduped_batch.values())

    # 4. Vectorized Prep via Pandas
    df = pd.DataFrame(final_rows)
    df = df.reindex(columns=model_columns)


    # 5. Execute High-Speed COPY
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # Create staging table matching the real model's types
            col_defs = ", ".join([f"{c.name} {c.type}" for c in model.__table__.columns if c.name != 'id'])
            cur.execute(f"CREATE TEMP TABLE stage_table ({col_defs}) ON COMMIT DROP;")

            # Stream buffer to Postgres
            buf = io.StringIO()
            
            # --- THE FIX: Use an empty string for NULL and match it in copy_from ---
            df.to_csv(
                buf, 
                sep="\t", 
                index=False, 
                header=False, 
                na_rep="",             # Use empty string for NULLs
                quoting=csv.QUOTE_NONE, 
                escapechar="\\"        
            )
            
            buf.seek(0)
            # Match the null='' parameter here
            cur.copy_from(buf, "stage_table", sep="\t", columns=model_columns, null="")

            # Atomic Transfer
            cols_str = ", ".join(model_columns)
            conflict_target = ", ".join(conflict_keys)
            update_cols = [c for c in model_columns if c not in conflict_keys]
            update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
            
            cur.execute(f"""
                INSERT INTO {model.__tablename__} ({cols_str})
                SELECT {cols_str} FROM stage_table
                ON CONFLICT ({conflict_target}) 
                DO UPDATE SET {update_str};
            """)
            raw_conn.commit()
            return len(df)
    except Exception as e:
        raw_conn.rollback()
        print(f"[!] Upsert Error: {e}")
        return 0
    finally:
        raw_conn.close()



def download_saturation(data_cache, table_name, policy, limit, alpha, output_dir="stock"):
    """
    PRIORITY: Discovery & Limit Proximity.
    Expresses: 'Did we hit the ceiling?' and 'What keys are dominant?'
    """
    all_records = []
    counts_per_ticker = []
    for ticker, results in data_cache.items():
        all_records.extend(results)
        counts_per_ticker.append(len(results))
    
    df = pd.DataFrame(all_records)
    whitelist = set(policy.get("key_whitelist", []))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Key Density (The Signal)
    existing_white = [c for c in df.columns if c in whitelist]
    fill_rates = df[existing_white].notnull().mean().sort_values(ascending=False)
    sns.barplot(x=fill_rates.values, y=fill_rates.index, ax=ax1, palette="viridis", hue=fill_rates.index, legend=False)
    ax1.set_title(f"Key Density (Alpha={alpha})")

    # 2. Limit Proximity (The 'Depth' Check)
    sns.histplot(counts_per_ticker, bins=20, ax=ax2, color="blue", kde=True)
    ax2.axvline(x=limit, color='red', linestyle='--', label=f'API Limit ({limit})')
    
    # Calculate mode to check if we are 'choking' on the limit
    if counts_per_ticker:
        mode_val = max(set(counts_per_ticker), key=counts_per_ticker.count)
        ax2.set_title(f"Record Depth Distribution (Mode: {mode_val} vs Limit: {limit})")
        if mode_val >= limit * 0.95:
            print(f"[!] WARNING: Mode ({mode_val}) is near limit ({limit}). Data is likely truncated.")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/download_saturation_{table_name}.png")


def upsert_health(engine, model, table_name, policy, chunk_size, output_dir="stock"):
    whitelist = set(policy.get("key_whitelist", []))
    total_records = 0
    null_counts = Counter()
    gaps = []
    
    # Store the last date seen for each company to compute gaps across chunks
    last_dates = {} 

    if table_name == "short-interest":
        date_col = "settlement_date"
    elif table_name == "quotes":
        date_col = "t"

    # if table_name == "short-interest":
    else:
        date_col = "period_end"
    query = f"SELECT * FROM {model.__tablename__} ORDER BY company_cik, {date_col} ASC"
    
    with engine.connect() as conn:
        for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
            total_records += len(chunk)
            
            # --- Incremental Density ---
            existing_white = [c for c in chunk.columns if c in whitelist]
            for col in existing_white:
                null_counts[col] += chunk[col].notnull().sum()
            
            # --- Incremental Gaps ---
            if date_col in chunk.columns:
                chunk[date_col] = pd.to_datetime(chunk[date_col])
                
                for cik, group in chunk.groupby('company_cik'):
                    # Calculate gaps within this chunk
                    group_dates = group[date_col].tolist()
                    
                    # Check if we have a carry-over date from a previous chunk for this CIK
                    if cik in last_dates:
                        first_gap = (group_dates[0] - last_dates[cik]).days
                        gaps.append(first_gap)
                    
                    # Add internal gaps
                    if len(group_dates) > 1:
                        internal_gaps = group[date_col].diff().dt.days.dropna().tolist()
                        gaps.extend(internal_gaps)
                    
                    # Update the 'boundary' date for the next chunk
                    last_dates[cik] = group_dates[-1]

    if total_records == 0:
        print("[-] Upsert Health: No data found.")
        return

    # 2. Finalize Aggregations
    fill_rates = pd.Series({k: v / total_records for k, v in null_counts.items()}).sort_values(ascending=False)
    
    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Density Plot
    sns.barplot(x=fill_rates.values, y=fill_rates.index, ax=ax1, palette="magma", hue=fill_rates.index, legend=False)
    ax1.set_title(f"Post-Upsert Data Density (Total Records: {total_records})")

    # Gap Plot
    if gaps:
        sns.boxenplot(x=pd.Series(gaps), ax=ax2, color="green")
        ax2.set_title("Distribution of Temporal Gaps (Days between records)")
        ax2.set_xlabel("Days Gap")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/upsert_health_{table_name}.png")
    print(f"[+] Chunked Health report saved to: {output_dir}")