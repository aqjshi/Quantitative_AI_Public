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





sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, engine, DATABASE_URL, Base
from core.sieve import TokenBucketRateLimiter

from stock_mvp.database.database import execute_db_query
from stock_mvp.config import load_configuration

from stock_mvp.fundamentals.instruments import Instrument

from stock_mvp.fundamentals.dividends import ingest_dividends
from stock_mvp.fundamentals.cash_flow import ingest_cash_flow
from stock_mvp.fundamentals.balance_sheets import ingest_balance_sheets
from stock_mvp.fundamentals.income_statement import ingest_income_statements
from stock_mvp.fundamentals.short_interest import ingest_short_interest
from stock_mvp.fundamentals.bar import ingest_bars

BASE_POLYGON_URL = "https://api.polygon.io/v3/reference/tickers"

def fetch_pit_ticker_state(ticker, date_str, api_key, limiter, max_attempts=5):
    """Fetches point-in-time ticker state using direct HTTP calls."""
    params = {
        "ticker": ticker,
        "active": "true",
        "date": date_str,
        "limit": 1000,
        "apiKey": api_key
    }

    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            resp = requests.get(BASE_POLYGON_URL, params=params, timeout=(5, 15))
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", []) or []

                # Handle pagination if multiple pages exist
                while "next_url" in data and data["next_url"]:
                    limiter.wait()
                    next_url = data["next_url"]
                    if "apiKey" not in next_url:
                        next_url = f"{next_url}&apiKey={api_key}"
                    
                    next_resp = requests.get(next_url, timeout=(5, 15))
                    if next_resp.status_code == 200:
                        data = next_resp.json()
                        results.extend(data.get("results", []) or [])
                    else:
                        break
                return results

            elif resp.status_code == 429:
                time.sleep(attempt ** 2 + 2)
            else:
                return []
        except Exception:
            if attempt == max_attempts:
                return []
            time.sleep(attempt)

    return []

def fetch_ticker_details(ticker, target_date, api_key, limiter):
    """Fetches deep ticker metadata for a specific date (handles delisted fallbacks)."""
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    params = {
        "date": target_date,
        "apiKey": api_key
    }
    limiter.wait()
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("results", {}) or {}
        else:
            return {"active": False, "delisted_fallback": True}
    except Exception:
        return {}



def safe_int_str(val):
    if val is None or str(val).upper() in ["NONE", "NULL", "\\N"]:
        return '\\N'
    try:
        return str(int(float(val))) # Handles float strings like "4275929952280.0" cleanly
    except (ValueError, TypeError):
        return '\\N'
            

def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]

def bulk_ingest_instruments(instruments_dict, result_queue):
    """
    Executes a SINGLE bulk copy/upsert operation for all Phase 1 instruments.
    Completely eliminates lock contention and queue flooding.
    """
    if not instruments_dict:
        return

    instrument_batch = list(instruments_dict.values())

    inst_upsert_sql = """
        DROP TABLE IF EXISTS temp_instruments;
        CREATE TEMPORARY TABLE temp_instruments (
            composite_figi VARCHAR(20),
            composite_figi_hash BIGINT,
            ticker VARCHAR(20),
            ticker_hash BIGINT,
            name VARCHAR(255),
            sic_code INT,
            sic_description VARCHAR(255),
            total_employees INT,
            market_cap BIGINT,
            share_class_shares_outstanding BIGINT,
            weighted_shares_outstanding BIGINT,
            city VARCHAR(100),
            postal_code VARCHAR(20),
            state VARCHAR(50),
            market VARCHAR(20),
            locale VARCHAR(20),
            primary_exchange VARCHAR(20),
            type VARCHAR(20),
            active BOOLEAN,
            currency_name VARCHAR(20),
            cik BIGINT,
            share_class_figi VARCHAR(20),
            point_in_time_date DATE,
            upsert_date DATE
        );
        
        COPY temp_instruments (
            composite_figi, composite_figi_hash, ticker, ticker_hash, name, 
            sic_code, sic_description, total_employees, market_cap, 
            share_class_shares_outstanding, weighted_shares_outstanding, 
            city, postal_code, state, market, locale, primary_exchange, 
            type, active, currency_name, cik, share_class_figi, 
            point_in_time_date, upsert_date
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL '\\N');
        
        INSERT INTO instruments (
            composite_figi, composite_figi_hash, ticker, ticker_hash, name, 
            sic_code, sic_description, total_employees, market_cap, 
            share_class_shares_outstanding, weighted_shares_outstanding, 
            city, postal_code, state, market, locale, primary_exchange, 
            type, active, currency_name, cik, share_class_figi, 
            point_in_time_date, upsert_date
        )
        SELECT 
            composite_figi, composite_figi_hash, ticker, ticker_hash, name, 
            sic_code, sic_description, total_employees, market_cap, 
            share_class_shares_outstanding, weighted_shares_outstanding, 
            city, postal_code, state, market, locale, primary_exchange, 
            type, active, currency_name, cik, share_class_figi, 
            point_in_time_date, upsert_date 
        FROM temp_instruments
        ON CONFLICT (ticker, point_in_time_date) 
        DO UPDATE SET 
            composite_figi = EXCLUDED.composite_figi,
            composite_figi_hash = EXCLUDED.composite_figi_hash,
            name = EXCLUDED.name,
            sic_code = EXCLUDED.sic_code,
            sic_description = EXCLUDED.sic_description,
            total_employees = EXCLUDED.total_employees,
            market_cap = EXCLUDED.market_cap,
            share_class_shares_outstanding = EXCLUDED.share_class_shares_outstanding,
            weighted_shares_outstanding = EXCLUDED.weighted_shares_outstanding,
            city = EXCLUDED.city,
            postal_code = EXCLUDED.postal_code,
            state = EXCLUDED.state,
            market = EXCLUDED.market,
            locale = EXCLUDED.locale,
            primary_exchange = EXCLUDED.primary_exchange,
            type = EXCLUDED.type,
            active = EXCLUDED.active,
            currency_name = EXCLUDED.currency_name,
            cik = EXCLUDED.cik,
            share_class_figi = EXCLUDED.share_class_figi,
            upsert_date = EXCLUDED.upsert_date;
    """

    result_queue.put((instrument_batch, inst_upsert_sql, "Bulk Upsert Asset Master"))


def async_db_worker(task_queue):
    """
    BACKGROUND CONSUMER THREAD: Consumes database payloads asynchronously
    so the network pipeline never drops to 0 Mbps.
    """
    while True:
        item = task_queue.get()
        if item is None:  # Poison pill exit token
            task_queue.task_done()
            break
        try:
            # item[1] = copy_sql, item[0] = row data payload matrix, item[2] = info log string
            execute_db_query(item[1], item[0], item[2])
        except Exception as db_err:
            print(f"\n!!!! [DATABASE ERROR] Failed during {item[2]}: {db_err} !!!!\n")
        finally:
            task_queue.task_done()

def process_ticker_phase1(ticker, config, today_str, today_date, limiter, api_key, seed=42):
    hb_months = config.get("reconstruction_heartbeat_freq_months", 12)
    raw_start = datetime.strptime(config["fetch_start"], "%Y-%m-%d")
    start_dt = raw_start.replace(day=1)
    grid_dates = []
    curr = start_dt
    while curr <= today_date.replace(day=1):
        grid_dates.append(curr.strftime("%Y-%m-%d"))
        curr += relativedelta(months=hb_months)
    if not grid_dates or grid_dates[-1] != today_str:
        grid_dates.append(today_str)

    ticker_type = config.get("ticker_type", "CS")
    local_instruments = {}

    for date_str in grid_dates:
        # Use function-based API call
        results = fetch_pit_ticker_state(ticker, date_str, api_key, limiter)

        for item in results:
            if isinstance(item, dict) and item.get('ticker') == ticker and item.get('composite_figi'):
                figi = item['composite_figi']
                
                # Use function-based API call for detailed metadata
                deep_meta = fetch_ticker_details(ticker, date_str, api_key, limiter)
                hydrated_item = {**item, **deep_meta}

                if deep_meta.get("delisted_fallback") is True:
                    hydrated_item["active"] = False

                actual_type = hydrated_item.get('type') or ticker_type
                if actual_type != ticker_type:
                    continue

                figi_hash = hash_string_to_int64(figi, seed=seed)
                ticker_hash = hash_string_to_int64(ticker, seed=seed)
                address_block = hydrated_item.get("address", {}) if isinstance(hydrated_item.get("address"), dict) else {}

                raw_cik = hydrated_item.get('cik')
                clean_cik = str(raw_cik).lstrip("0") if raw_cik else '\\N'

                row = (
                    figi, figi_hash, ticker, ticker_hash,
                    hydrated_item.get('name', 'None')[:255] if hydrated_item.get('name') else 'None',
                    hydrated_item.get('sic_code', -1) if hydrated_item.get('sic_code') is not None else -1,
                    hydrated_item.get('sic_description', 'None')[:255] if hydrated_item.get('sic_description') else 'None',
                    safe_int_str(hydrated_item.get('total_employees')),
                    safe_int_str(hydrated_item.get('market_cap')),
                    safe_int_str(hydrated_item.get('share_class_shares_outstanding')),
                    safe_int_str(hydrated_item.get('weighted_shares_outstanding')),
                    address_block.get('city', 'None')[:100] if address_block.get('city') else 'None',
                    address_block.get('postal_code', 'None')[:20] if address_block.get('postal_code') else 'None',
                    address_block.get('state', 'None')[:50] if address_block.get('state') else 'None',
                    hydrated_item.get('market', 'stocks'), hydrated_item.get('locale', 'us'),
                    hydrated_item.get('primary_exchange', 'None'), actual_type, hydrated_item.get('is_active', True),
                    hydrated_item.get('currency_name', 'usd'),
                    clean_cik,
                    hydrated_item.get('share_class_figi', '\\N' if hydrated_item.get('share_class_figi') is None else hydrated_item.get('share_class_figi')),
                    date_str, today_str
                )

                local_instruments[(ticker, date_str)] = row

    return local_instruments



def process_ticker_phase2(ticker, true_ticker_hash, start_str, end_str, config, result_queue):
    try:
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
    except Exception as e:
        return f" [!] Bar Ingestion failed for {ticker} over full history window: {e}"

    return f" [V] Completed processing time-series data for {ticker}"


def run_pipeline():
    tqdm.write("[*] Starting Incremental Maintenance Run (Non-Destructive Sync)...")
    config = load_configuration()

    # Rebuild schema definitions safely if any new tables were added (Without dropping existing data)
    Base.metadata.create_all(engine) 

    limiter = TokenBucketRateLimiter(rate_per_sec=config["rate_limit_per_sec"])

    # Establish dynamic maintenance lookback window in months
    today_date = pd.to_datetime(datetime.today())
    today_str = today_date.strftime('%Y-%m-%d')
    today_month_start_str = today_date.strftime('%Y-%m-01')
    today_month_start = pd.to_datetime(today_month_start_str)
            
    lookback_months = config.get("maintenance_lookback_months", 12)
    start_date = today_date - relativedelta(months=lookback_months)
    start_str = start_date.strftime('%Y-%m-%d')

    # Override config fetch start dynamically for this maintenance pass
    config["fetch_start"] = start_str
    
    tqdm.write(f"[*] Maintenance Window Active: {start_str} to {today_str} ({lookback_months} Months Lookback)")

    unique_tickers = sorted(set(config["case_study"]))
    ticker_to_hash_map = {}

    result_queue = queue.Queue(maxsize=50)
    db_thread = threading.Thread(target=async_db_worker, args=(result_queue,), daemon=True)
    db_thread.start()

    try:
        # Phase 1: Ingest Asset Master Snapshots (Over maintenance window)
        tqdm.write(f"[*] Launching Phase 1 parallel allocation pool ({config['num_workers']} workers)...")
        global_instruments_map = {}

        with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
            futures = {
                executor.submit(
                    process_ticker_phase1, 
                    ticker, config, today_month_start_str, today_month_start, limiter, POLY_KEY, config.get("string_hash_seed", 42)
                ): ticker for ticker in unique_tickers
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Syncing Asset Master"):
                ticker = futures[future]
                try:
                    local_map = future.result()
                    global_instruments_map.update(local_map)
                except Exception as exc:
                    tqdm.write(f" [!!!] Worker thread crashed processing instrument mappings for {ticker}: {exc}")

        # Flush Asset Master updates to DB
        tqdm.write(f"[*] Upserting {len(global_instruments_map)} instrument snapshots to DB...")
        bulk_ingest_instruments(global_instruments_map, result_queue)
        result_queue.join()
        tqdm.write("[*] Phase 1 complete. Asset master sync complete.")

        # Resolve local ticker-to-hash mappings directly from the database
        with engine.connect() as conn:
            query = text("SELECT DISTINCT ticker, ticker_hash FROM instruments")
            for row in conn.execute(query).fetchall():
                ticker_to_hash_map[row[0]] = row[1]

        # Phase 2: Sync Market Bars over lookback window
        tqdm.write(f"[*] Launching Phase 2 parallel allocation pool ({config['num_workers']} workers)...")
        with ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
            futures = {}
            for ticker in unique_tickers:
                true_hash = ticker_to_hash_map.get(ticker)
                if not true_hash:
                    tqdm.write(f" [!] Skipping {ticker}: No local instrument metadata found.")
                    continue
                futures[executor.submit(process_ticker_phase2, ticker, true_hash, start_str, today_str, config, result_queue)] = ticker

            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2: Syncing Market Bars"):
                ticker = futures[future]
                try:
                    completion_log = future.result()
                    if completion_log:
                        tqdm.write(completion_log)
                except Exception as exc:
                    tqdm.write(f" [!!!] Worker thread crashed processing data for {ticker}: {exc}")

        # Step 3: Sync Financial Timeseries Modules over lookback window
        tqdm.write("[*] Syncing macro financial modules in optimized chunks...")
        fin_config = config.copy()
        financial_modules = [ingest_balance_sheets, ingest_cash_flow, ingest_income_statements, ingest_short_interest, ingest_dividends]

        ticker_items = list(ticker_to_hash_map.items())
        chunked_groups = [
            dict(ticker_items[i:i + config['fundamentals_batch_size']]) 
            for i in range(0, len(ticker_items), config['fundamentals_batch_size'])
        ]

        for func_obj in financial_modules:
            for group in chunked_groups:
                try:
                    func_obj(ticker_to_hash_map=group, start=start_str, end=today_str, result_queue=result_queue, api_key=POLY_KEY)
                except Exception as e:
                    tqdm.write(f" [!] Financial Module {func_obj.__name__} encountered chunk error: {e}")

    finally:
        print("[*] Wrapping up maintenance runs... Flushing background database queue...")
        result_queue.join()
        result_queue.put(None)
        db_thread.join()
        tqdm.write("[*] Daily maintenance pipeline finished cleanly.")



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


