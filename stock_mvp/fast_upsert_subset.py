import os
import sys
import time
from datetime import datetime, timedelta, date
from tqdm import tqdm
from sqlalchemy import text
from dateutil.relativedelta import relativedelta
import mmh3  # MurmurHash3 library
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd 
import requests 
import cProfile
import pstats
import io
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import POLY_KEY, engine, DATABASE_URL, Base
from core.sieve import TokenBucketRateLimiter

from stock_mvp.database.database import execute_db_query

from stock_mvp.fundamentals.instruments import Instrument

from stock_mvp.fundamentals.dividends import ingest_dividends
from stock_mvp.fundamentals.cash_flow import ingest_cash_flow
from stock_mvp.fundamentals.balance_sheets import ingest_balance_sheets
from stock_mvp.fundamentals.income_statement import ingest_income_statements
from stock_mvp.fundamentals.short_interest import ingest_short_interest
from stock_mvp.fundamentals.bar import ingest_bars
from stock_mvp.fresh_upsert_subset import process_ticker_phase1, fetch_pit_ticker_state, fetch_ticker_details, hash_string_to_int64, safe_int_str, bulk_ingest_instruments, async_db_worker
BASE_POLYGON_URL = "https://api.polygon.io/v3/reference/tickers"



import traceback

def async_db_worker(task_queue):
    while True:
        item = task_queue.get()

        if item is None:
            task_queue.task_done()
            break

        try:
            payload = item[0]
            sql = item[1]
            description = item[2]

      

            execute_db_query(sql, payload, description)

        
        except Exception as db_err:
            tqdm.write(
                f"\n[!!! DB ERROR !!!] {description}\n"
                f"Exception: {type(db_err).__name__}: {db_err}\n"
            )
            traceback.print_exc()

        finally:
            task_queue.task_done()




class FlatFileAggsEngine:
    @staticmethod
    def fetch_and_parse_bars(
        date_str: str,
        target_tickers: list[str],
        agg_type: str = "minute_aggs_v1",  # or "day_aggs_v1"
        temp_dir: str = "./options/tmp_aggs",
    ) -> list[tuple]:
        """
        Static wrapper:
        1. Downloads the compressed S3 flat file using Boto3.
        2. Filters target tickers and standardizes timestamps via DuckDB.
        3. Cleans up the temporary local file.
        """
        year, month = date_str.split("-")[0], date_str.split("-")[1]
        object_key = f"us_options_opra/{agg_type}/{year}/{month}/{date_str}.csv.gz"

        os.makedirs(temp_dir, exist_ok=True)
        local_file_path = os.path.join(temp_dir, f"{date_str}_{agg_type}.csv.gz")

        # 1. Download file via Boto3
        try:
            S3_CLIENT.download_file(POLYGON_S3_BUCKET, object_key, local_file_path)
        except S3_CLIENT.exceptions.NoSuchKey:
            return []  # Market holiday or weekend
        except Exception as e:
    
            tqdm.write(f"\n[!] S3 download failed for {object_key}: {e}")
            return []

        tickers_tuple = (
            tuple(target_tickers)
            if len(target_tickers) > 1
            else f"('{target_tickers[0]}')"
        )

        # 2. Vectorized filter and transform via DuckDB
        query = f"""
            SELECT 
                ticker,
                CAST(window_start / 1000000000 AS BIGINT) AS t,  -- Nanoseconds to natural seconds
                open AS o,
                high AS h,
                low AS l,
                close AS c,
                volume AS v,
                close AS vw,
                transactions AS n
            FROM read_csv_auto('{local_file_path}', compression='gzip', header=True)
            WHERE regexp_extract(ticker, '^O:([A-Z]+)', 1) IN {tickers_tuple}
        """

        try:
            con = duckdb.connect()
            df = con.execute(query).df()
            con.close()

            if df.empty:
                return []

            tsv_rows = [
                (
                    str(r.ticker),
                    str(r.underlying_ticker),
                    str(int(r.t)),
                    str(float(r.o)),
                    str(float(r.h)),
                    str(float(r.l)),
                    str(float(r.c)),
                    str(int(r.v)),
                    str(float(r.vw)),
                    str(int(r.n))
                )
                for r in df.itertuples(index=False)
            ]
            return tsv_rows
        except Exception as e:
            tqdm.write(f"\n[!] Parsing error on {local_file_path}: {e}")
            return []
        finally:
            # 3. Clean up the downloaded temporary archive
            if os.path.exists(local_file_path):
                os.remove(local_file_path)




def process_ticker_phase2(
    ticker,
    true_ticker_hash,
    start_str,
    end_str,
    config,
    result_queue
):
    try:
        tqdm.write(
            f"[BAR START] {ticker} | "
            f"{start_str} -> {end_str} | "
            f"hash={true_ticker_hash}"
        )

        bars_config = config.copy()

        ingest_bars(
            ticker=ticker,
            ticker_hash=true_ticker_hash,
            start=start_str,
            end=end_str,
            config=bars_config,
            result_queue=result_queue,
            api_key=POLY_KEY
        )

        tqdm.write(
            f"[BAR QUEUED/RETURNED] {ticker}"
        )

        return f"[V] Completed bar ingestion for {ticker}"

    except Exception as e:
        import traceback

        traceback.print_exc()

        return (
            f"[!] Bar Ingestion failed for {ticker}: "
            f"{type(e).__name__}: {e}"
        )




def run_pipeline():
    tqdm.write("[*] Starting direct Ticker-based MVP pipeline.")
    policy_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(PROJECT_ROOT, "pricing", "simulation_policy_big.json")
    )
    with open(policy_path) as f:
        config = json.load(f)
    
    skip_phase1 = config["skip_phase1"]
    if not skip_phase1:

        tqdm.write("[*] Force-dropping tables via direct SQL CASCADE sweep...")
        tables_to_drop = ["instruments", "balance_sheets", "cash_flow", "dividends", "income_statements", "short_interest", "quotes"]

        with engine.begin() as conn:
            for table_name in tables_to_drop:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
                tqdm.write(f" -> Dropped table: {table_name}")

        tqdm.write("[*] Rebuilding fresh database schema definitions...")
        Base.metadata.create_all(engine)


    else:
        tqdm.write("[*] 'skip_phase1' is ACTIVE: Skipping database purge and Phase 1 API discovery.")
        tables_to_drop = ["balance_sheets", "cash_flow", "dividends", "income_statements", "short_interest", "quotes"]
        
        with engine.begin() as conn:
            for table_name in tables_to_drop:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
                tqdm.write(f" -> Dropped table: {table_name}")

        tqdm.write("[*] Rebuilding fresh database schema definitions...")
        Base.metadata.create_all(engine)
        
        
      

    limiter = TokenBucketRateLimiter(rate_per_sec=config["rate_limit_per_sec"])

    # today_date = pd.to_datetime("2026-01-01")
    today_date = pd.to_datetime(datetime.today())


    today_str = today_date.strftime('%Y-%m-%d')
    today_month_start_str = today_date.strftime('%Y-%m-01')
    today_month_start = pd.to_datetime(today_month_start_str)
    unique_tickers = sorted(set(config["tickers_in"]))
    ticker_to_hash_map = {}

    result_queue = queue.Queue(maxsize=50)
    db_thread = threading.Thread(target=async_db_worker, args=(result_queue,), daemon=True)
    db_thread.start()

    try:
        if not skip_phase1:

            tqdm.write(f"[*] Launching Phase 1 parallel allocation pool ({config['num_workers']} workers)...")
            global_instruments_map = {}

            with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
                futures = {
                    executor.submit(
                        process_ticker_phase1, 
                        ticker, config, today_month_start_str, today_month_start, limiter, POLY_KEY, config.get("string_hash_seed", 42)
                    ): ticker for ticker in unique_tickers
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Ingesting Asset Master"):
                    ticker = futures[future]
                    try:
                        local_map = future.result()
                        global_instruments_map.update(local_map)
                    except Exception as exc:
                        tqdm.write(f" [!!!] Worker thread crashed processing instrument mappings for {ticker}: {exc}")

            # Flush Asset Master in a single bulk transaction
            tqdm.write(f"[*] Flushing {len(global_instruments_map)} deduplicated instrument snapshots to DB...")
            bulk_ingest_instruments(global_instruments_map, result_queue)
            result_queue.join()
            tqdm.write("[*] Phase 1 complete. Asset master cleanly populated.")

        with engine.connect() as conn:
            query = text("SELECT DISTINCT ticker, ticker_hash FROM instruments")
            for row in conn.execute(query).fetchall():
                ticker_to_hash_map[row[0]] = row[1]

        # Phase 2: Sync Market Bars
        tqdm.write(f"[*] Launching Phase 2 parallel allocation pool ({config['num_workers']} workers)...")
        with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
            futures = {}
            for ticker in unique_tickers:
                true_hash = ticker_to_hash_map.get(ticker)
                if not true_hash:
                    tqdm.write(f" [!] Skipping {ticker}: No local instrument metadata found.")
                    continue
                futures[executor.submit(process_ticker_phase2, ticker, true_hash, config["fetch_start"], today_str, config, result_queue)] = ticker

            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2: Syncing Market Bars"):
                ticker = futures[future]
                try:
                    completion_log = future.result()
                    if completion_log:
                        tqdm.write(completion_log)
                except Exception as exc:
                    tqdm.write(f" [!!!] Worker thread crashed processing data for {ticker}: {exc}")

        # Financial Timeseries Modules
        tqdm.write("[*] Processing macro financial modules in optimized chunks...")
        fin_config = config.copy()
        financial_modules = [ingest_balance_sheets, ingest_cash_flow, ingest_income_statements, ingest_short_interest, ingest_dividends]

        ticker_items = list(ticker_to_hash_map.items())
        chunked_groups = [dict(ticker_items[i:i + config['fundamentals_batch_size']]) for i in range(0, len(ticker_items), config['fundamentals_batch_size'])]

        for func_obj in financial_modules:
            for group in chunked_groups:
                try:
                    func_obj(ticker_to_hash_map=group, start=config["fetch_start"], end=today_str, result_queue=result_queue, api_key=POLY_KEY)
                except Exception as e:
                    tqdm.write(f" [!] Financial Module {func_obj.__name__} encountered chunk error: {e}")

    finally:
        print("[*] Wrapping up pipeline runs... Waiting for background database queue to flush clean...")
        result_queue.join()
        result_queue.put(None)
        db_thread.join()
        tqdm.write("[*] Pipeline execution run loop finished cleanly.")



if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Execute main workload
    run_pipeline()

    profiler.disable()

    # Format and print top performance bottlenecks sorted by cumulative time
    print("\n" + "=" * 80)
    print("                      PROFILING REPORT (Top 50 Hotspots)")
    print("=" * 80 + "\n")
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(50)

