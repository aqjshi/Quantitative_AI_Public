import os
import re
import json
import time
import argparse
from typing import Dict, List, Optional
import sys 
import requests 
from tqdm import tqdm 
from datetime import datetime

import queue
import threading
import mmh3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import FRED_KEY_0, engine, DATABASE_URL, Base
from core.utils import to_str, to_int
from core.sieve import TokenBucketRateLimiter
from macros.config import load_configuration
from core.database.async_worker import async_db_worker
from macros.database.database import execute_db_query  # Ensure query execution helper is pulled



from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, FLOAT

def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]




class FredSeriesFiltered(Base):
    __tablename__ = "fred_series_filtered"

    id                              = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    category_id                    = Column(BigInteger)
    depth                           = Column(Integer)
    series_id                            = Column(String) #series_id
    series_id_hash                       = Column(BigInteger)

    realtime_start	                = Column(Date) #"2026-07-08"
    realtime_end	                = Column(Date) #"2026-07-08"
    title	                        = Column(String)#"Real Gross National Income"
    observation_start	            = Column(Date) #"1930-01-01"
    observation_end	                = Column(Date) ##"2025-01-01"
    frequency	                    = Column(String) #"Annual"
    frequency_short	                = Column(String) #"A"
    units	                        = Column(String) #"Percent Change from Preceding Period"
    units_short	                    = Column(String) #"% Chg. from Preceding Period"
    seasonal_adjustment	            = Column(String) #Not Seasonally Adjusted"
    seasonal_adjustment_short	    = Column(String) #"NSA"
    last_updated	                = Column(Date) #"2026-05-28 07:50:15-05"
    popularity	                    = Column(Integer) #8
    group_popularity	            = Column(Integer) #30
    notes	                        = Column(String) #"BEA Account Code: A023RL\n\nFor more information about this series, please see http://www.bea.gov/national/."


class FredObservations(Base):
    __tablename__ = "fred_observations"
    id                              = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    series_id_hash                       = Column(BigInteger)
    realtime_start	                = Column(Date)  #"2026-07-07"
    realtime_end	                = Column(Date)#"2026-07-07"
    date	                        = Column(Date)#"2004-01-01"
    value	                        = Column(Float) #"5.7"
    __table_args__ = (
        UniqueConstraint("series_id_hash", "date", "realtime_start", name="uq_fred_obs_bitemporal"),
    )




def load_raw_file(file_path: str) -> str:
    """Loads raw text content from a SQL/policy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required configuration file missing: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    


def generate_realtime_chunks(start_date_str: str, step_years: int = 10) -> List[tuple]:
    """
    Splits a wide realtime date span into 10-year slices to comply with 
    FRED's maximum 2,000 vintage dates per query limit.
    """
    try:
        start_yr = int(start_date_str.split("-")[0])
    except Exception:
        start_yr = 1990

    current_yr = datetime.now().year
    chunks = []
    curr_yr = start_yr

    while curr_yr < current_yr:
        next_yr = min(curr_yr + step_years - 1, current_yr - 1)
        chunks.append((f"{curr_yr}-01-01", f"{next_yr}-12-31"))
        curr_yr = next_yr + 1

    chunks.append((f"{current_yr}-01-01", "9999-12-31"))
    return chunks


def _execute_fetch_pass(series_id: str, start_date: str, rt_start: str, rt_end: str, api_key: str, limiter: TokenBucketRateLimiter) -> tuple:
    """
    Internal query runner for a given realtime range window.
    Returns (status_code, raw_results, error_message).
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    limit = 100000
    offset = 0
    raw_results = []

    while True:
        params = {
            "series_id": series_id,
            "file_type": "json",
            "api_key": api_key,
            "limit": limit,
            "offset": offset, 
            "observation_start": start_date,
            "realtime_start": rt_start,
            "realtime_end": rt_end
        }

        limiter.wait()

        try:
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 429:
                tqdm.write(" [!] FRED Rate limit hit. Backing off for 5 seconds...")
                time.sleep(5)
                continue

            if resp.status_code == 502:
                tqdm.write(" [!] Error 502 Something Wrong. Retry in 60 Seconds...")
                time.sleep(60)
                continue

            if resp.status_code != 200:
                return resp.status_code, [], resp.text

            data = resp.json()
            batch_results = data.get('observations', [])
            if not batch_results:
                break

            raw_results.extend(batch_results)

            if len(batch_results) < limit:
                break

            offset += limit

        except Exception as e:
            return 500, [], str(e)

    return 200, raw_results, ""


def fetch_observations(series_id: str, series_id_hash: int, start_date: str, api_key: str, limiter: TokenBucketRateLimiter,step_years=5) -> List[tuple]:
    """
    Attempts full-horizon single fetch first. Fallbacks to X-year batching if FRED throws 
    a 2,000 vintage date limit error.
    """
    status_code, raw_results, err_msg = _execute_fetch_pass(
        series_id, start_date, start_date, "9999-12-31", api_key, limiter
    )
    
    # Fallback to 10-year chunking if vintage date overflow occurs
    if status_code == 400 and ("vintage dates" in err_msg or "exceeds" in err_msg):
        tqdm.write(f" [!] Vintage overflow detected for {series_id}. Fallback to {step_years}-year batching...")
        raw_results = []
        seen_keys = set()
        realtime_chunks = generate_realtime_chunks(start_date, step_years)

        for rt_start, rt_end in realtime_chunks:
            code, chunk_results, _ = _execute_fetch_pass(
                series_id, start_date, rt_start, rt_end, api_key, limiter
            )
            if code == 200 and chunk_results:
                for obs in chunk_results:
                    record_key = (obs.get('realtime_start'), obs.get('realtime_end'), obs.get('date'))
                    if record_key not in seen_keys:
                        seen_keys.add(record_key)
                        raw_results.append(obs)

    mapped_entries = []
    for d in raw_results:
        val_str = to_str(d.get('value'))

        try:
            val_float = float(val_str) if val_str not in ['None', '.', ''] else None
        except ValueError:
            val_float = None

        entry = (
            series_id_hash,
            to_str(d.get('realtime_start')),              
            to_str(d.get('realtime_end')),    
            to_str(d.get('date')),    
            val_float if val_float is not None else 'None'
        )
        mapped_entries.append(entry)

    return mapped_entries


def process_observations(series_id: str, series_id_hash: int, start_date: str, result_queue: queue.Queue, api_key: str, limiter: TokenBucketRateLimiter, step_years=5):
    """
    Fetches observations for a given series and pushes PostgreSQL COPY chunks into the streaming queue.
    """
    mapped_data = fetch_observations(series_id, series_id_hash, start_date, api_key, limiter, step_years=step_years)
    if not mapped_data:
        return

    copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_observations (
            series_id_hash BIGINT,
            realtime_start DATE,
            realtime_end DATE,
            date DATE,
            value FLOAT
        );
        TRUNCATE staging_observations;
        
        COPY staging_observations (
            series_id_hash,
            realtime_start,
            realtime_end,
            date,
            value
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');
        
        INSERT INTO fred_observations (
            series_id_hash,
            realtime_start,
            realtime_end,
            date,
            value
        )
        SELECT 
            series_id_hash,
            realtime_start,
            realtime_end,
            date,
            value
        FROM staging_observations;
    """
    
    chunk_size = 5000 
    for i in range(0, len(mapped_data), chunk_size):
        chunk = mapped_data[i:i + chunk_size]
        result_queue.put((chunk, copy_sql, f"Series {series_id} Chunk {i}"))


def apply_filter_unfiltered_series(config: dict) -> List[dict]:
    """
    Dynamically loads modular text blacklists and SQL deduplication policies
    from disk based on the configuration dictionary, compiles the production 
    SQL query, and returns the filtered macro universe series.
    """
    filters_dir = os.path.join(PROJECT_ROOT, "macros", "filters")


    base_query_filename = config.get("fred_base_query", "base_query.sql")

    base_query_path = os.path.join(filters_dir, base_query_filename)

    base_query_template = load_raw_file(base_query_path)


    # Inject dynamic parameters
    try:
        query_string = base_query_template.format(
            base_query_template=base_query_template
        )
    except KeyError:
        query_string = f"""
        {base_query_template}
        """

    # Execute query
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_string))
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        tqdm.write(f" [!] Extraction Fallback Failure: {e}")
        return []





def upsert_filtered_series(series_list: List[dict]):
    """
    Inserts the deduplicated macro core directly into fred_series_filtered.
    Since the table is dropped and recreated clean at start, a flat INSERT handles it safely.
    """
    if not series_list:
        return

    # Stripped ON CONFLICT clause entirely to prevent constraint check errors
    upsert_sql = text("""
        INSERT INTO fred_series_filtered (
            category_id, depth, series_id, series_id_hash, realtime_start, realtime_end, title,
            observation_start, observation_end, frequency, frequency_short, units,
            units_short, seasonal_adjustment, seasonal_adjustment_short, last_updated,
            popularity, group_popularity, notes
        ) VALUES (
            :category_id, :depth, :series_id, :series_id_hash, :realtime_start, :realtime_end, :title,
            :observation_start, :observation_end, :frequency, :frequency_short, :units,
            :units_short, :seasonal_adjustment, :seasonal_adjustment_short, :last_updated,
            :popularity, :group_popularity, :notes
        );
    """)

    try:
        with engine.begin() as conn:
            for row in series_list:
                # Dynamically generate the unique 64-bit integer anchor key
                row['series_id_hash'] = hash_string_to_int64(row['series_id'])
                conn.execute(upsert_sql, row)
        tqdm.write(f"[*] Successfully synchronized {len(series_list)} rows to fred_series_filtered.")
    except Exception as e:
        tqdm.write(f" [!] Ingestion Error during filtered update staging: {e}")


def get_filtered_series() -> List[tuple]:
    """
    Pulls back the locked core from the filtered workspace table.
    """
    query = text("SELECT series_id, series_id_hash FROM fred_series_filtered ORDER BY popularity DESC;")
    try:
        with engine.connect() as conn:
            return conn.execute(query).fetchall()
    except Exception as e:
        tqdm.write(f" [!] Database Retrieval Error: {e}")
        return []


# --- MAIN PIPELINE EXECUTION ---
def main():
    config = load_configuration()
    
    # 1. Total clean database state setup
    tqdm.write("[*] Dropping legacy tables to complete a clean sweep...")
    tables_to_drop = ["fred_series_filtered", "fred_observations"]
    for table_name in tables_to_drop:
        if table_name in Base.metadata.tables:
            Base.metadata.tables[table_name].drop(engine, checkfirst=True)
            tqdm.write(f"  -> Dropped table: {table_name}")
            
    tqdm.write("[*] Rebuilding fresh database schema definitions...")
    Base.metadata.create_all(engine) 

    # 2. Rate limiting adjustments
    rate_limit = config.get("rate_limit_per_sec", 2)
    limiter = TokenBucketRateLimiter(rate_per_sec=rate_limit)
    
    # 3. Initialize background streaming thread
    result_queue = queue.Queue(maxsize=100)
    db_thread = threading.Thread(target=async_db_worker, args=(result_queue,), daemon=True)
    db_thread.start()

    # 4. STEP 1: Apply deduplication policy and extract winners
    tqdm.write("[*] Running deduplication policy over raw macro rows...")
    raw_filtered_winners =  apply_filter_unfiltered_series(config)
    print(f"✅ Filtered macro universe down to {len(raw_filtered_winners)} tier-1 series.")

    if not raw_filtered_winners:
        tqdm.write(" [!] Aborting execution: No target series found matching strategy conditions.")
        result_queue.put(None)
        db_thread.join()
        return

    # 5. STEP 2: Upsert winners cleanly into our locked core table
    tqdm.write("[*] Synchronizing winners into fred_series_filtered skeleton...")
    upsert_filtered_series(raw_filtered_winners)

    # 6. STEP 3: Read back the verified skeleton coordinates
    harvesting_targets = get_filtered_series()
    
    tqdm.write(f"[*] Found {len(harvesting_targets)} target core series. Beginning bitemporal point-in-time harvesting...")
    pbar = tqdm(total=len(harvesting_targets), desc="Harvesting Point-In-Time Matrices", unit="series")

    try:
        # STEP 4: Loop through core series and stream records to the worker queue
        for row in harvesting_targets:
            series_name = str(row[0])
            series_hash = int(row[1])

            process_observations(
                series_id=series_name,
                series_id_hash=series_hash,
                start_date=config["fetch_start"], 
                result_queue=result_queue,
                api_key=FRED_KEY_0,
                limiter=limiter, 
                step_years=config["download_batch_years"]
            )
            pbar.update(1)

    except KeyboardInterrupt:
        tqdm.write("\n [!] Execution halted manually by developer. Graceful shutdown initiated.")
    finally:
        pbar.close()
        tqdm.write("[*] Wrapping up pipeline runs... Waiting for background database queue to flush clean...")
        result_queue.join()  
        result_queue.put(None) 
        db_thread.join()      
        engine.dispose()  
        tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()



