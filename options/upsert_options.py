import sys
import os
import json
import requests
from sqlalchemy import create_engine, select, MetaData

from typing import List, Dict, Any, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
import pandas as pd

from datetime import timedelta, datetime, date
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import DATABASE_URL, POLY_KEY, SessionLocal, engine
from core.models import Base, Company, Quote, engine, OptionQuote
import core.sieve as sieve
from core.sieve import  TokenBucketRateLimiter, parse_osi
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
                                COPY option_quotes (
                                    company_cik, osi, is_call, contract_expiry, t, 
                                    underlying_c, underlying_SOD, o, h, l, c, v, vw, n
                                ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None')
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

def fetcher(api_key, start_date, end_date, limiter, second_level=False):
    def fetch_all_pages(ticker):
        all_results = []
        # Initial URL
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
        
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








def iter_batches(start_date: str, end_date: str, batch_days: int = 14):
    cur = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    while cur <= end:
        batch_end = min(cur + timedelta(days=batch_days - 1), end)
        yield cur, batch_end
        cur = batch_end + timedelta(days=1)
    
def run_for_day(
    ticker: str,
    day_str: str,
    k_pct: float = 0.30,         
    max_contracts: int = 250,
    max_dte: int = 90,
    min_oi: int = 0,
    min_trades: int = 0,
    anchor_S: Optional[float] = None,  
):
    d = pd.to_datetime(day_str).date()
    
    # 计算最大到期日
    max_expiry_date = (d + timedelta(days=max_dte)).strftime('%Y-%m-%d')
    
    cik = get_company_id(ticker)
    client = RESTClient(POLY_KEY)

    log = dict(
        ticker=ticker, day=d, sod_open=None, kmin=None, kmax=None,
        selected_contracts=0, written_rows=0, skipped_existing=0,
        skipped_illiquid=0, duration_sec=None, status="started", error=None
    )

    # 使用传入的anchor_S（batch第一天的SOD）
    if anchor_S is not None:
        S = float(anchor_S)
    else:
        S = get_sod_from_database(ticker, d)

    if S is None or S <= 0:
        tqdm.write(f"[SOD SKIP] {ticker} {d}: no valid S anchor (batch or daily).")
        log.update(status="no_sod")
        return log
    

    underlying_minute_prices = get_underlying_minute_prices(cik, t)
    kmin = S * (1.0 - k_pct)
    kmax = S * (1.0 + k_pct)
    log.update(sod_open=S, kmin=kmin, kmax=kmax)

    raw_contracts = [] 
    try:
        # 修复：使用as_of参数来获取历史合约
        raw_contracts_generator = client.list_options_contracts(
            underlying_ticker=ticker,
            as_of=day_str,  # 关键修复：使用as_of参数指定历史日期
            expiration_date_gte=day_str,  # 可以同时使用expiration_date_gte和lte来过滤到期日
            expiration_date_lte=max_expiry_date,
            limit=1000
        )
        raw_contracts = list(raw_contracts_generator) 
    except Exception as e:
        tqdm.write(f"[FETCH ERROR] {ticker} {d}: Failed to list options contracts: {e}")
        log.update(status="contract_fetch_error", error=str(e))
        return log

    tqdm.write(f"Contracts fetched for {ticker} on {d}: {len(raw_contracts)} (max_dte={max_dte}, max_expiry={max_expiry_date})")

    # 转换并过滤合约
    contracts = []
    utc_20_00 = pd.Timedelta(hours=20)
    for raw_c in raw_contracts:
        try:
            exp_date_str = getattr(raw_c, 'expiration_date', None)
            if not exp_date_str:
                continue

            exp_date = pd.to_datetime(exp_date_str, utc=True) + utc_20_00
            exp_date_only = exp_date.date()  # 只取日期部分用于DTE计算
            
            # BUG FIX: 在本地计算并过滤DTE
            dte = (exp_date_only - d).days
            if dte <= 0 or dte > max_dte:  # 本地LTE过滤
                continue
            
            contracts.append({
                "osi": raw_c.ticker,  
                "strike": raw_c.strike_price,
                "expiry_ts": int(exp_date.timestamp()),
                "ctype": "C" if raw_c.contract_type == "call" else "P",
                "eod_oi": getattr(raw_c, 'open_interest', 0) or 0,
                "exp_date": exp_date_only  # 保存用于调试
            })
        except Exception as e:
            continue
    
    tqdm.write(f"Contracts after DTE filter ({max_dte} days): {len(contracts)}")

    # SOD-based fetch filter
    items = []
    for c in contracts:
        try:
            K = float(c["strike"])
            
            # 已经通过DTE过滤，这里只需要检查K范围
            if not (kmin <= K <= kmax):
                continue
            
            if min_oi > 0:
                oi = c.get("eod_oi", 0) or 0
                if oi < min_oi:
                    continue
            
            logm = float(np.log(K / S))
            items.append({
                "osi": c["osi"], 
                "ctype": c.get("ctype", "C"), 
                "K": K, 
                "dte": (c["exp_date"] - d).days, 
                "logm": logm
            })
        except Exception:
            continue

    if not items:
        tqdm.write(f"[INFO] {ticker} {d}: no contracts within DTE≤{max_dte} and K∈[{kmin:.2f},{kmax:.2f}].")
        log.update(status="empty_after_filter")
        return log

    dfc = pd.DataFrame(items).drop_duplicates(subset=["osi"])
    dfc["atm_rank"] = dfc["logm"].abs()
    dfc.sort_values(["atm_rank", "dte"], inplace=True)
    dfc = dfc.head(max_contracts)

    tqdm.write(
        f"Plan: ticker={ticker}, day={d}, SOD={S:.4f}, K∈[{kmin:.2f},{kmax:.2f}], "
        f"max_dte={max_dte}, max_contracts={max_contracts}, selected={len(dfc)}"
    )
    log.update(selected_contracts=int(len(dfc)))

    # Fetch minutes and upsert
    total_rows = 0
    skipped_existing = 0
    skipped_illiquid = 0
    skipped_parsing_error = 0  # 新增：统计解析错误的合约

    for r in dfc.itertuples(index=False):


        bars = minute_aggs_polygon(r.osi, d, client=client) 
        if not bars:
            continue

        # Fine filter: drop "zombie" contracts whose total daily transactions are too low
        if min_trades > 0:
            day_trades = sum((b.get("n") or 0) for b in bars)
            if (day_trades or 0) < min_trades:
                skipped_illiquid += 1
                continue

        try:   
            _, exp_ts, _, _ = parse_osi(r.osi)
        except ValueError as e:
            tqdm.write(f"[EXPIRY-PARSE-SKIP] {ticker} {d}: Skipping {r.osi} due to expiry parsing error: {e}")
            skipped_parsing_error += 1
            continue

        out_rows: List[Dict[str, Any]] = []
        for b in bars:
            t_ms = b.get("t")
            if t_ms is None:
                continue

            t_ms = b.get("t")
            time_entry_ts = int(int(t_ms) / 1000)
            
            # ------------------ LOOKUP THE UNDERLYING PRICE (S_t) ------------------
            underlying_close = underlying_minute_prices.get(time_entry_ts)
        
            if underlying_close is None:
                # Skip if the underlying price for this minute is missing.
                # This ensures data integrity.
                continue    
                        # Inside the loop where you build out_rows:
            out_rows.append((
                cik,                   # company_cik
                r.osi,                 # osi
                (r.ctype == "C"),      # is_call (boolean)
                exp_ts,                # contract_expiry
                time_entry_ts,         # t
                underlying_close,      # underlying_c
                S,                     # underlying_SOD
                b.get("o"),            # o
                b.get("h"),            # h
                b.get("l"),            # l
                b.get("c"),            # c
                b.get("v"),            # v
                b.get("vw"),           # vw
                b.get("n")             # n
            ))

        upsert_minute_rows(out_rows) 
        total_rows += len(out_rows)

    tqdm.write(f"[DONE] {ticker} {d}: total minute rows upserted = {total_rows}, skipped_parsing_error = {skipped_parsing_error}")

    # 更新日志
    log.update(
        status="success", 
        written_rows=total_rows, 
        skipped_existing=skipped_existing, 
        skipped_illiquid=skipped_illiquid,
        skipped_parsing_error=skipped_parsing_error  # 新增
    )
    return log


def main():
    table_name="options"
    os.makedirs(f"options/{table_name}", exist_ok=True)
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)
    
    tickers = params.get("ticker", [])
    start_day = params.get("train_start", "").split(' ')[0]
    end_day   = params.get("train_end", "").split(' ')[0]
    max_contracts = params.get("max_contracts", 250)
    k_pct         = params.get("k_pct", 0.30)   
    max_dte       = params.get("max_dte", 90)   
    min_oi        = params.get("min_oi", 0)
    min_trades    = params.get("min_trades", 0)
    engine = create_engine(DATABASE_URL)

    results = []

    for ticker in tickers:
        
        for batch_start, batch_end in iter_batches(start_day, end_day, batch_days=14):
            tqdm.write(f"\n[BATCH] {ticker} {batch_start} → {batch_end}")

    
            anchor_S = get_sod_from_database(ticker, batch_start)
            if anchor_S is None or anchor_S <= 0:
                tqdm.write(f"[BATCH-SKIP] {ticker} {batch_start}: failed to get SOD anchor; skip this batch.")
                continue

            
            days_dt = pd.date_range(start=batch_start, end=batch_end, freq='B')
            days_str = [d.strftime('%Y-%m-%d') for d in days_dt]


            for day_str in tqdm(days_str, desc=f"Processing {ticker} {batch_start}"):
                tqdm.write(f"--- Running for {ticker} on {day_str} (BATCH_S={anchor_S:.4f}, k_pct={k_pct}) ---")
    
                result_log = run_for_day(
                    ticker=ticker,
                    day_str=day_str,
                    k_pct=k_pct,
                    max_contracts=max_contracts,
                    max_dte=max_dte,
                    min_oi=min_oi,
                    min_trades=min_trades,
                    anchor_S=anchor_S,  
                )
                results.append(result_log)
      
                
                sep_len = len(f"--- Running for {ticker} on {day_str} ---")
                tqdm.write("-" * sep_len + "\n")

if __name__ == "__main__":
    main()

