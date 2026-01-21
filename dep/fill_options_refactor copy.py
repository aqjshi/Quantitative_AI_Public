import sys
import os
import json
import requests
from sqlalchemy import create_engine, select
from datetime import timedelta, datetime



import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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




# Project Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import DATABASE_URL, POLY_KEY
from core.models import Base, Company, Quote
import core.sieve as sieve
import random
import queue
from tqdm import tqdm
from multiprocessing import Process, Queue



POLY_BASE = "https://api.polygon.io"

SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)


engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
Base.metadata.create_all(engine)

from massive import RESTClient
metadata = MetaData()



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


def _expiry_ts_from_osi(osi: str) -> int:
    """Parse OSI and convert to UTC timestamp at 20:00 on expiration date (to match schema convention)."""
    _, exp, _, _ = parse_osi(osi)
    return int(pd.Timestamp(f"{exp} 20:00", tz="UTC").timestamp())


def _val(d: Any, *keys):
    """Unified getter for dict (REST) or SDK Agg object."""
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None
    for k in keys:
        if hasattr(d, k):
            v = getattr(d, k)
            if v is not None:
                return v
    return None




def get_company_id(ticker: str) -> int:
    with Session(engine) as s:
        obj = s.execute(select(Company).where(Company.ticker == ticker)).scalars().one_or_none()
        if obj:
            return obj.id
        obj = Company(ticker=ticker)
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return obj.id
    
def sod_ts_from_date(d: date) -> int:
    """
    Calculates the Start-of-Day (SOD) market timestamp (13:30 UTC)
    for a given date and returns it as a UTC Unix timestamp in seconds.
    """
    # Combine the date with a min time, set timezone to UTC, 
    # then replace the time with market open.
    sod_dt_utc = pd.Timestamp(
        datetime.combine(d, datetime.min.time()), tz='UTC'
    ).replace(hour=13, minute=30, second=0, microsecond=0)
    
    # Return as a standard (seconds) Unix timestamp
    return int(sod_dt_utc.timestamp())

def minute_aggs_polygon(osi: str, d: date, client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Fetch minute bars (13:30–20:00 UTC) and return a unified list of dicts {t,o,h,l,c,v,vw,n}, with `t` in milliseconds.
    No forward filling is done here (per spec, forward-fill happens in the visualization layer).
    """
    # print("hit")
    rows: List[Dict[str, Any]] = []
    t0 = pd.Timestamp(f"{d} 13:30", tz="UTC")
    t1 = pd.Timestamp(f"{d} 20:00", tz="UTC")

    for a in client.list_aggs(osi, 1, "minute", d.isoformat(), d.isoformat(),
                                adjusted=True, sort="asc", limit=1000):
        t_ms = _val(a, "timestamp")
        if t_ms is None:
            continue
        ts = pd.to_datetime(int(t_ms), unit="ms", utc=True)
        if not (t0 <= ts <= t1):
            continue
        rows.append({
            "t": int(t_ms),
            "o": _val(a, "open"),
            "h": _val(a, "high"),
            "l": _val(a, "low"),
            "c": _val(a, "close"),
            "v": _val(a, "volume"),
            "vw": _val(a, "vwap"),
            "n": _val(a, "transactions"),
        })
    return rows

def get_sod_from_database(ticker: str, d: date) -> Optional[float]:
    """
    Fetches the Start-of-Day (SOD) price for a ticker from the 'quotes' table.
    
    The SOD price is defined as the 'close_price' from the 'quotes' table
    at the exact market open time (13:30 UTC).
    """
    try:
        # 1. Get the company ID for the ticker
        company_id = get_company_id(ticker)
        
        # 2. Get the target SOD timestamp (13:30 UTC)
        sod_timestamp = sod_ts_from_date(d)
        
        # 3. Query the 'quotes' table
        # (Using 'Quote.close_price' from your models.py)
        stmt = (
            select(Quote.close_price)
            .where(
                Quote.company_id == company_id,
                Quote.time_entry_ts == sod_timestamp
            )
            .limit(1)
        )

        with Session(engine) as session:
            result = session.execute(stmt).scalar_one_or_none()
            
            if result is not None:
                return float(result)
            else:
                # This else is important for debugging
                tqdm.write(f"[DB-SOD-MISS] No SOD price found for {ticker} on {d} at ts {sod_timestamp}.")
                return None

    except Exception as e:
        tqdm.write(f"[DB-SOD-ERROR] Failed to fetch SOD for {ticker} on {d}: {e}")
        return None


def get_underlying_minute_prices(ticker: str, d: date) -> Dict[int, float]:
    """
    Fetches all minute-level close prices for the underlying stock on a given date.
    
    Returns: A dictionary mapping timestamp (int, seconds) to close price (float).
    """
    company_id = get_company_id(ticker)
    
    # Define market hours for the day (13:30 to 20:00 UTC)
    start_ts = int(pd.Timestamp(f"{d} 13:30", tz="UTC").timestamp())
    end_ts   = int(pd.Timestamp(f"{d} 20:00", tz="UTC").timestamp())
    
    # Query Quote.close_price and Quote.time_entry_ts
    stmt = (
        select(Quote.time_entry_ts, Quote.close_price)
        .where(
            Quote.company_id == company_id,
            Quote.time_entry_ts >= start_ts,
            Quote.time_entry_ts <= end_ts
        )
    )

    prices = {}
    with Session(engine) as session:
        # Use fetchall() to get all minutes
        results = session.execute(stmt).fetchall()
        for ts, price in results:
            prices[ts] = float(price)
            
    if not prices:
        tqdm.write(f"[DB-UNDERLYING-MISS] No minute prices found for {ticker} on {d}.")

    return prices



# ────────── UPSERT ──────────
# expected to be 50 seconds
def upsert_minute_rows(rows: List[Dict[str, Any]]):
    """Upsert on (option_name, time_entry_ts); only minute OHLCV columns are written/updated."""
    if not rows:
        return

    # === 新增：数据去重逻辑 ===
    seen = set()
    unique_rows = []
    
    for row in rows:
        # 创建唯一标识符 (option_name, time_entry_ts)
        key = (row['option_name'], row['time_entry_ts'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
        else:
            # 可选：记录重复数据用于调试
            tqdm.write(f"[DUPLICATE-SKIP] Duplicate entry: {key}")
    
    if not unique_rows:
        tqdm.write("[DUPLICATE-INFO] All rows were duplicates, skipping upsert")
        return
    
    # 使用去重后的数据
    rows = unique_rows
    
    stmt = pg_insert(OptionQuote.__table__).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[OptionQuote.option_name, OptionQuote.time_entry_ts],
        set_={
            "contract_type": stmt.excluded.contract_type,
            "company_id": stmt.excluded.company_id,
            "contract_expiry": stmt.excluded.contract_expiry,
            "strike": stmt.excluded.strike,

            "option_open": stmt.excluded.option_open,
            "option_close": stmt.excluded.option_close,
            "option_high": stmt.excluded.option_high,
            "option_low": stmt.excluded.option_low,
            "option_volume": stmt.excluded.option_volume,
            "option_volume_weighted": stmt.excluded.option_volume_weighted,
            "option_transactions": stmt.excluded.option_transactions,
        },
    )
    with engine.begin() as conn:
        conn.execute(stmt)


# ────────── UTILITIES ──────────
def minutes_exist_for_osi_day(osi: str, d: date) -> bool:
    """For --resume: skip if minute data for this OSI already exists on this day."""
    start_ts = int(pd.Timestamp(f"{d} 13:30", tz="UTC").timestamp())
    end_ts   = int(pd.Timestamp(f"{d} 20:00", tz="UTC").timestamp())
    with engine.connect() as conn:
        cnt = conn.execute(
            text("""
            SELECT 1 FROM option_quotes
            WHERE option_name = :osi
              AND time_entry_ts BETWEEN :t0 AND :t1
            LIMIT 1
            """),
            {"osi": osi, "t0": start_ts, "t1": end_ts}
        ).fetchone()
    return cnt is not None

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
    resume: bool = False,
    anchor_S: Optional[float] = None,  
):
    d = pd.to_datetime(day_str).date()
    
    # 计算最大到期日
    max_expiry_date = (d + timedelta(days=max_dte)).strftime('%Y-%m-%d')
    
    cid = get_company_id(ticker)
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
    

    underlying_minute_prices = get_underlying_minute_prices(ticker, d)
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
        if resume and minutes_exist_for_osi_day(r.osi, d):
            skipped_existing += 1
            continue

        # 添加OSI解析检查
        try:
            # 先尝试解析OSI，如果失败则跳过
            test_parsing = _expiry_ts_from_osi(r.osi)
        except ValueError as e:
            tqdm.write(f"[OSI-PARSE-SKIP] {ticker} {d}: Skipping {r.osi} due to parsing error: {e}")
            skipped_parsing_error += 1
            continue

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
            exp_ts = _expiry_ts_from_osi(r.osi) 
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
            out_rows.append(
                dict(
                    option_name=r.osi,
                    contract_type=r.ctype,
                    company_id=cid,
                    contract_expiry=exp_ts,
                    strike=float(r.K),
                    time_entry_ts=int(int(t_ms) / 1000),

                    underlying_close=underlying_close,
                    
       
                    underlying_SOD=S,
                    option_open=b.get("o"),
                    option_close=b.get("c"),
                    option_high=b.get("h"),
                    option_low=b.get("l"),
                    option_volume=b.get("v"),
                    option_volume_weighted=b.get("vw"),
                    option_transactions=b.get("n"),
                )
            )

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
    parser = argparse.ArgumentParser(description="Run data fetching based on parameters from a JSON file.")
    parser.add_argument("params_file", help="Path to the JSON file containing all run parameters (e.g., params.json)")
    args = parser.parse_args()
    params_file_path = args.params_file

    with open(params_file_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    
    tickers = params.get("ticker", [])
    start_dt_full = params.get("train_start", "")
    end_dt_full   = params.get("train_end", "")
    max_contracts = params.get("max_contracts", 250)
    k_pct         = params.get("k_pct", 0.30)   
    max_dte       = params.get("max_dte", 90)   
    min_oi        = params.get("min_oi", 0)
    min_trades    = params.get("min_trades", 0)

    
    start_day = start_dt_full.split()[0]
    end_day   = end_dt_full.split()[0]

    results = []
    # pr = cProfile.Profile()
    # pr.enable()
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
            # return BUG: Or it will quit at the first batch
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(70)
    # print(s.getvalue())
if __name__ == "__main__":
    main()

