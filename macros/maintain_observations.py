import os
import time
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

from core.db import FRED_KEY_0, engine
from core.utils import to_str, to_int
from core.sieve import TokenBucketRateLimiter
from macros.config import load_configuration
from core.database.async_worker import async_db_worker

from sqlalchemy import text

def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]



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


def maintain_observations(series_id: str, series_id_hash: int, start_date: str, result_queue: queue.Queue, api_key: str, limiter: TokenBucketRateLimiter, step_years=5):
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
            series_id_hash, realtime_start, realtime_end, date, value
        )
        SELECT series_id_hash, realtime_start, realtime_end, date, value
        FROM staging_observations
        ON CONFLICT (series_id_hash, date, realtime_start) 
        DO UPDATE SET 
            realtime_end = EXCLUDED.realtime_end,
            value = EXCLUDED.value;
    """
    
    chunk_size = 5000 
    for i in range(0, len(mapped_data), chunk_size):
        chunk = mapped_data[i:i + chunk_size]
        result_queue.put((chunk, copy_sql, f"Series {series_id} Chunk {i}"))








def get_filtered_series(series_list: List[str]) -> List[tuple]:
    """
    Pulls back the locked core from the filtered workspace table matching the input list.
    """
    query = text("""
        SELECT series_id, series_id_hash 
        FROM fred_series_filtered 
        WHERE series_id = ANY(:series_list)
        ORDER BY popularity DESC;
    """)
    try:
        with engine.connect() as conn:
            return conn.execute(query, {"series_list": series_list}).fetchall()
    except Exception as e:
        tqdm.write(f" [!] Database Retrieval Error: {e}")
        return []


def get_latest_state(series_id_hashes: List[int]) -> Optional[str]:
    """Largest realtime_start on record for the given series hashes."""
    if not series_id_hashes:
        return None

    query = text("""
        SELECT MAX(realtime_start) 
        FROM fred_observations 
        WHERE series_id_hash = ANY(:series_id_hashes);
    """)
    try:
        with engine.connect() as conn:
            value = conn.execute(query, {"series_id_hashes": series_id_hashes}).scalar()
            return None if value is None else str(value)
    except Exception as e:
        tqdm.write(f" [!] Error fetching latest state watermark: {e}")
        return None

    
# --- MAIN PIPELINE EXECUTION ---
def main():
    config = load_configuration()
    raw_filtered_winners  = []
    exo_config = sys.argv[2]
    file_path = os.path.join(exo_config)
    with open(file_path, "r") as f:
        raw_filtered_winners = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if len(raw_filtered_winners) == 0:
        return
    

    # 2. Rate limiting adjustments
    rate_limit = config.get("rate_limit_per_sec", 2)
    limiter = TokenBucketRateLimiter(rate_per_sec=rate_limit)
    
    # 3. Initialize background streaming thread
    result_queue = queue.Queue(maxsize=100)
    db_thread = threading.Thread(target=async_db_worker, args=(result_queue,), daemon=True)
    db_thread.start()




    # 6. STEP 3: Read back the verified skeleton coordinates
    harvesting_targets = get_filtered_series(raw_filtered_winners)
    # print(harvesting_targets)
    # Extract integer hashes from fetched target tuples (row[1] is series_id_hash)
    target_hashes = [row[1] for row in harvesting_targets]
    
    latest_state = get_latest_state(target_hashes)
    tqdm.write(f"[*] Watermark for target set: {latest_state}")

    tqdm.write(f"[*] Found {len(harvesting_targets)} target core series. Beginning harvest...")
    pbar = tqdm(total=len(harvesting_targets), desc="Harvesting Point-In-Time Matrices", unit="series")

    try:
        # STEP 4: Loop through core series and stream records to the worker queue
        for row in harvesting_targets:
            series_name = str(row[0])
            series_hash = int(row[1])

            maintain_observations(
                series_id=series_name,
                series_id_hash=series_hash,
                start_date=latest_state, 
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






# def main():


#     watermark = get_watermark()

#     rows_before = count_observations()
#     open_dupes_before = count_open_duplicates()

 
#     ensure_maintenance_index()

#     source = f"{seed_file} ({len(series_ids)} ids requested)"

#     harvesting_targets = resolve_targets(series_ids)

#     tqdm.write(f"[*] Target set: {source}")
#     tqdm.write(f"[*] Maintaining {len(harvesting_targets)} series "
#                f"(fred_series_filtered read only, never rewritten).")

#     limiter = TokenBucketRateLimiter(rate_per_sec=config.get("rate_limit_per_sec", 2))

#     result_queue = queue.Queue(maxsize=100)
#     db_thread = threading.Thread(target=async_db_worker, args=(result_queue,), daemon=True)
#     db_thread.start()

#     total_fetched = 0
#     total_absent = 0
#     series_with_updates = 0
#     updated_series: List[tuple] = []

#     pbar = tqdm(total=len(harvesting_targets), desc="Maintaining vintages", unit="series")
#     try:
#         for row in harvesting_targets:
#             series_name = str(row[0])
#             series_hash = int(row[1])

#             mapped_data = fetch_new_vintages(
#                 series_id=series_name,
#                 series_id_hash=series_hash,
#                 observation_start=config["fetch_start"],
#                 watermark=watermark,
#                 api_key=FRED_KEY_0,
#                 limiter=limiter,
#             )

#             if mapped_data:
#                 absent = count_absent_vintages(series_hash, mapped_data)
#                 total_fetched += len(mapped_data)
#                 total_absent += absent
#                 if absent:
#                     series_with_updates += 1
#                     updated_series.append((series_name, absent))

#                 queue_new_observations(series_name, mapped_data, result_queue)

#             pbar.set_postfix(new=total_absent, series=series_with_updates)
#             pbar.update(1)

#     except KeyboardInterrupt:
#         tqdm.write("\n [!] Halted manually. Flushing what has already been queued.")
#     finally:
#         pbar.close()
#         tqdm.write("[*] Waiting for the background writer to drain...")
#         result_queue.join()
#         result_queue.put(None)
#         db_thread.join()

#     rows_after = count_observations()
#     open_dupes_after = count_open_duplicates()
#     new_watermark = get_watermark()

   

#     if updated_series:
#         tqdm.write("\n  most-revised series this pass:")
#         for name, count in sorted(updated_series, key=lambda x: -x[1])[:10]:
#             tqdm.write(f"     {name:<20} {count:>7,} vintages not previously on record")
#     else:
#         tqdm.write("\n  nothing outstanding -- every fetched vintage was already on record.")

#     engine.dispose()
#     tqdm.write("\n[*] Connection pools disposed. Maintenance run finished cleanly.")


# if __name__ == "__main__":
#     main()
