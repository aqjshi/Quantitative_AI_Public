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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import FRED_KEY_0, engine, DATABASE_URL, Base
from core.sieve import TokenBucketRateLimiter
from macros.config import load_configuration
from core.database.async_worker import async_db_worker


from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB


class FredCategory(Base):
    __tablename__ = "fred_categories"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    name = Column(String)
    parent_id = Column(BigInteger)
    depth = Column(Integer)


def to_str(val):
    if val is None:
        return 'None' # Matches the 'NULL None' in your COPY command
    return str(val)


def fetch_category_children(category_id: int, api_key: str, depth: int, limiter: TokenBucketRateLimiter) -> List[tuple]:


    url = "https://api.stlouisfed.org/fred/category/children"


    params = {
        "category_id": category_id,
        "file_type": "json",
        "api_key": api_key
    }
    
    raw_results = []
    current_url = url
    current_params = params
  

    while current_url:
        try:
            limiter.wait()

            resp = requests.get(current_url, params=current_params, timeout=15)
            
            if resp.status_code == 429:  # Rate limit hit
                time.sleep(5)
                continue
                
            if resp.status_code != 200:
                print(f" [!] API Error {resp.status_code} on {current_url}")
                break

            data = resp.json()
            batch_results = data.get('categories', [])
            if batch_results:
                raw_results.extend(batch_results)

            # --- PAGINATION CHECK ---
            next_url = data.get('next_url')
            if next_url:
                # Security/Auth: next_url doesn't always include the apiKey
                current_url = f"{next_url}&api_key={api_key}" if "api_key" not in next_url else next_url
                current_params = None  # Don't send params again; they are in the URL
            else:
                current_url = None

        except Exception as e:
            print(f" [!] Fetch Error: {e}")
            break
    mapped_entries = []


    for d in raw_results:
        
        entry = (
            to_str(d.get('id')),    
            to_str(d.get('name')),       
            to_str(d.get('parent_id')),         
            depth
        )
        mapped_entries.append(entry)
        
    return mapped_entries

def process_category_level(category_id: int, result_queue: queue.Queue, api_key: str, depth: int, limiter: TokenBucketRateLimiter):
    """
    Fetches the children of a specific category and formats the clean PostgreSQL COPY commands.
    """
    # 3. Thread the BATCHES, not the tickers
    mapped_data = fetch_category_children(category_id, api_key, depth, limiter)
    if not mapped_data:
        return []

    # 3. Prepare SQL COPY Command
    copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_categories (LIKE fred_categories INCLUDING ALL);
        TRUNCATE staging_categories;
        
        COPY staging_categories (
            id,
            name,
            parent_id, 
            depth
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');

        -- SIMPLE INSERT: No conflict handling
        INSERT INTO fred_categories 
        SELECT * FROM staging_categories;

    """
    
 # Chunk data out to prevent database lockups
    chunk_size = 1000
    for i in range(0, len(mapped_data), chunk_size):
        chunk = mapped_data[i:i + chunk_size]
        result_queue.put((chunk, copy_sql, f"Category {category_id} Chunk {i}"))
        
    # Return child IDs back to main thread so the BFS queue keeps moving
    return [int(item[0]) for item in mapped_data]


# --- MAIN PIPELINE EXECUTION ---
# --- MAIN PIPELINE EXECUTION ---
def main():
    config = load_configuration()
    
    # 1. Total clean database state setup
    tqdm.write("[*] Dropping legacy tables to complete a clean sweep...")
    tables_to_drop = ["fred_categories"]
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

    # 4. Read BFS boundaries from parameters
    seed_node = config.get("fred_seed_category_id", 0) # Ultimate root is 0
    max_depth = config.get("upsert_categories_bfs_depth", 2) # Safe source boundary layer
    
    # 🌟 EXTRACT WHITELIST VALUES FROM CONFIG
    # Converts the dict values into a high-performance lookup set of integers
    whitelist_dict = config.get("upsert_categories_depth_1_whitelist", {})
    whitelist_ids = set(int(v) for v in whitelist_dict.values()) if whitelist_dict else set()
    
    if whitelist_ids:
        tqdm.write(f"[*] Loaded Depth 1 Whitelist Firewall containing {len(whitelist_ids)} active sectors.")

    # Set up BFS engine trackers
    bfs_queue = queue.Queue()
    bfs_queue.put((seed_node, 0)) # Format: (category_id, current_depth)
    seen_categories = {seed_node}

    tqdm.write(f"[*] Starting Scoped BFS mapping loop out to depth layer: {max_depth}...")
    pbar = tqdm(total=bfs_queue.qsize(), desc="Mapping Category Tree Nodes", unit="node")

    try:
        while not bfs_queue.empty():
            current_cat, current_depth = bfs_queue.get()
            
            # If the node exceeds our target depth limit, skip it and advance the bar
            if current_depth >= max_depth:
                bfs_queue.task_done()
                pbar.update(1)
                continue
                
            child_ids = process_category_level(
                category_id=current_cat,
                result_queue=result_queue,
                api_key=FRED_KEY_0,
                depth=current_depth + 1,
                limiter=limiter
            )
            
            # Track how many brand new, unseen nodes we expand into the queue
            new_nodes_count = 0
            for child_id in child_ids:
                if child_id not in seen_categories:
                    
                    # 🌟 EARLY PRUNING FIREWALL
                    # If the next layer is Depth 1, verify if the child ID is explicitly authorized
                    if (current_depth + 1) == 1 and whitelist_ids:
                        if child_id not in whitelist_ids:
                            # Prune early! Skip enqueuing entirely so we never perform downstream fetches
                            continue
                    
                    seen_categories.add(child_id)
                    bfs_queue.put((child_id, current_depth + 1))
                    
                    # Only expand the progress bar total if the child falls within the processing depth
                    if (current_depth + 1) < max_depth:
                        new_nodes_count += 1
            
            # Dynamically increase the progress bar's max ceiling based on newly discovered work
            if new_nodes_count > 0:
                pbar.total += new_nodes_count
                pbar.refresh() # Updates the layout instantly so the ETA recalculates
                    
            bfs_queue.task_done()
            pbar.update(1) # Complete the current iteration loop item

    except KeyboardInterrupt:
        tqdm.write("\n [!] Execution halted manually by developer. Graceful shutdown initiated.")
    finally:
        pbar.close() 
        tqdm.write("[*] Wrapping up pipeline runs... Waiting for background database queue to flush clean...")
        result_queue.join()  
        result_queue.put(None) 
        db_thread.join()      
        
        # FIX 2: Safely close and dispose of the connection pool links
        engine.dispose()
        tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()