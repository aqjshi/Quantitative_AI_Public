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
from core.db import FRED_KEY_0, engine, DATABASE_URL, Base
from core.sieve import TokenBucketRateLimiter
from macros.config import load_configuration, load_blacklist
from core.database.async_worker import async_db_worker
from macros.database.database import execute_db_query  # Ensure query execution helper is pulled



from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB

def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]




class FredSeriesUnfiltered(Base):
    __tablename__ = "fred_series_unfiltered"

    id                              = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    category_id                     = Column(BigInteger)
    depth                           = Column(Integer)
    series_id                            = Column(String)
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

def to_str(val) -> str:
    if val is None:
        return 'None'
    # CRITICAL: Neutralize newlines and tabs so they don't break the COPY frame
    return str(val).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def to_int(val) -> str:
    if val is None or str(val).strip() == '':
        return 'None'
    try:
        # Extrapolate floating points (like "8.0") cleanly into base integers
        return str(int(float(val)))
    except (ValueError, TypeError):
        return 'None'
    

def fetch_category_series(category_id: int, api_key: str, depth: int, seed:int , limiter: TokenBucketRateLimiter) -> List[tuple]:
    url = "https://api.stlouisfed.org/fred/category/series"
    limit = 1000
    offset = 0
    raw_results = []

    while True:
        params = {
            "category_id": category_id,
            "file_type": "json",
            "api_key": api_key,
            "limit": limit,
            "offset": offset
        }
        
        # Guard the API right before execution line
        limiter.wait()
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            
            if resp.status_code == 429:  # Rate limit safety net
                tqdm.write(" [!] FRED Rate limit hit. Backing off for 5 seconds...")
                time.sleep(5)
                continue
                
            if resp.status_code != 200:
                tqdm.write(f" [!] API Error {resp.status_code} on category {category_id} {resp.url} ")
                break

            data = resp.json()
            batch_results = data.get('seriess', [])  # FRED's collection key is explicitly 'seriess'
            if not batch_results:
                break

            raw_results.extend(batch_results)

            # If the response array size is smaller than our limit frame, we reached the leaf edge
            if len(batch_results) < limit:
                break

            offset += limit  # Slide query offset pointer forward

        except Exception as e:
            tqdm.write(f" [!] Fetch Exception Error: {e}")
            break

    mapped_entries = []
    for d in raw_results:
   
        seasonal_str = d.get('seasonal_adjustment', '')
    
        series_id = to_str(d.get('id'))
        series_id_hash = hash_string_to_int64(series_id, seed=seed)
        
        # Handle Date conversions gracefully or allow Postgres parsing
        # Timestamp truncations for 'last_updated' are cleanly intercepted
        updated_raw = d.get('last_updated', 'None')
        if updated_raw and updated_raw != 'None' and ' ' in updated_raw:
            updated_raw = updated_raw.split(' ')[0] # Isolate YYYY-MM-DD

        entry = (
            category_id,                
            depth,                      
                       
            series_id,                       
            series_id_hash,                  
            to_str(d.get('realtime_start')),               
            to_str(d.get('realtime_end')),            
            to_str(d.get('title')),             
            to_str(d.get('observation_start')),            
            to_str(d.get('observation_end')),            
            to_str(d.get('frequency')),          
            to_str(d.get('frequency_short')),          
            to_str(d.get('units')),              
            to_str(d.get('units_short')),            
            to_str(seasonal_str),        
            to_str(d.get('seasonal_adjustment_short')),   
            to_str(updated_raw),      
            to_int(d.get('popularity')),    
            to_int(d.get('group_popularity')),    
            to_str(d.get('notes'))                    
        )
        mapped_entries.append(entry)
        
    return mapped_entries
def process_category_series(category_id: int, result_queue: queue.Queue, api_key: str, depth: int, seed: int, limiter: TokenBucketRateLimiter):
    """
    Fetches the children series of a specific category and formats the clean PostgreSQL COPY commands.
    """
    mapped_data = fetch_category_series(category_id, api_key, depth, seed, limiter)
    if not mapped_data:
        return

    # FIXED: Explicit standalone structure mapping. 
    # This prevents Postgres from inheriting the primary key 'id' position from the parent model.
    copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_series (
            category_id BIGINT,
            depth INTEGER,

            series_id VARCHAR,
            series_id_hash BIGINT,
            realtime_start DATE,
            realtime_end DATE,
            title VARCHAR,
            observation_start DATE,
            observation_end DATE,
            frequency VARCHAR,
            frequency_short VARCHAR,
            units VARCHAR,
            units_short VARCHAR,
            seasonal_adjustment VARCHAR,
            seasonal_adjustment_short VARCHAR,
            last_updated DATE,
            popularity INTEGER,
            group_popularity INTEGER,
            notes TEXT
        );
        TRUNCATE staging_series;
        
        COPY staging_series (
            category_id, depth,  series_id, series_id_hash,
            realtime_start, realtime_end, title, observation_start, observation_end,
            frequency, frequency_short, units, units_short,
            seasonal_adjustment, seasonal_adjustment_short, last_updated,
            popularity, group_popularity, notes
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');

        -- Safe Upsert execution: On conflict of unique series_ids, update metrics.
        -- Explicitly list target columns, skipping the autoincrementing PK 'id' entirely!
        INSERT INTO fred_series_unfiltered (
            category_id, depth, series_id, series_id_hash,
            realtime_start, realtime_end, title, observation_start, observation_end,
            frequency, frequency_short, units, units_short,
            seasonal_adjustment, seasonal_adjustment_short, last_updated,
            popularity, group_popularity, notes
        )
        SELECT 
            category_id, depth, series_id, series_id_hash,
            realtime_start, realtime_end, title, observation_start, observation_end,
            frequency, frequency_short, units, units_short,
            seasonal_adjustment, seasonal_adjustment_short, last_updated,
            popularity, group_popularity, notes
        FROM staging_series;
    """
    
    chunk_size = 1000
    for i in range(0, len(mapped_data), chunk_size):
        chunk = mapped_data[i:i + chunk_size]
        result_queue.put((chunk, copy_sql, f"Category {category_id} Chunk {i}"))


def get_target_categories(target_depth: int) -> List[tuple]:
    """
    Natively queries the local fred_categories layout using a direct connection link.
    Bypasses external database wrappers and returns clean (id, depth) tuples.
    """
    query = text("""
        SELECT id, depth 
        FROM fred_categories 
        WHERE depth <= :target_depth;
    """)
    
    try:
        # Use a context manager to borrow a connection from the SQLAlchemy pool
        with engine.connect() as conn:
            result = conn.execute(query, {"target_depth": target_depth})
            # fetchall() safely returns a list of Row objects that act like tuples
            return result.fetchall()
            
    except Exception as e:
        tqdm.write(f" [!] Database Extraction Error: {e}")
        return []
    
# --- MAIN PIPELINE EXECUTION ---
def main():
    config = load_configuration()
    black_list = load_blacklist()
    
    # 1. Total clean database state setup
    tqdm.write("[*] Dropping legacy tables to complete a clean sweep...")
    tables_to_drop = ["fred_series_unfiltered"]
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

    target_depth = config.get("upsert_series_lte_depth", 2)             # Safe source boundary layer
    
    raw_categories = get_target_categories(target_depth)
    
    # Clean extraction from DB query wrapper depending on database handler formatting outputs
    category_ids = [row[0] if isinstance(row, (tuple, list)) else row.id for row in raw_categories]
    
    if not category_ids:
        tqdm.write(" [!] Aborting execution: No target category categories found matching depth constraints.")
        result_queue.put(None)
        db_thread.join()
        return
    
    tqdm.write(f"[*] Found {len(category_ids)} target categories. Beginning flat harvesting sweep...")
    pbar = tqdm(total=len(category_ids), desc="Harvesting Series Masters", unit="category")

    try:
        # A simple, flat linear loop over your predefined targets
        for row in raw_categories:
            # Handle row unpacking safely regardless of tuple or object-row types returned
            cat_id = int(row[0] if isinstance(row, (tuple, list)) else row.id)
            cat_depth = int(row[1] if isinstance(row, (tuple, list)) else row.depth)

            # ---------------------------------------------------------
            # THE INTEGRATION: O(1) Blacklist Firewall
            # ---------------------------------------------------------
            if cat_id in black_list:
                # Instantly drop the toxic category. 
                # Advance the progress bar so the loop doesn't freeze tracking.
                pbar.update(1)
                continue
            # ---------------------------------------------------------

            process_category_series(
                category_id=cat_id,
                result_queue=result_queue,
                api_key=FRED_KEY_0,
                depth=cat_depth,
                seed=config["random_seed"],
                limiter=limiter
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
        engine.dispose()  # Clean socket link cleanup pool dump
        tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()