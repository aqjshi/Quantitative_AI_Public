import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Date, Float, Integer, BigInteger,
    UniqueConstraint, text as sql_text
)
from sqlalchemy import create_engine, select, text, Table, Column, Integer, String, Float, Date, BigInteger, MetaData


from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from models import Base, Company, OptionQuote
from datetime import datetime, date

from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
# ────────── CONFIG ──────────
load_dotenv()

POLY_KEY  = os.getenv("POLYGON_API_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

POLY_BASE = "https://api.polygon.io"
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)



engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
Base.metadata.create_all(engine)


# Whether Polygon SDK is available
try:
    from polygon import RESTClient as PolyRESTClient
    HAS_POLY_SDK = True
except Exception:
    HAS_POLY_SDK = False


# ────────── Optional: ingestion log table (for post-run auditing) ──────────
metadata = MetaData()

ingestion_log = Table(
    "ingestion_log_polygon",
    metadata,
    Column("ticker", String(16), nullable=False),
    Column("day", Date, nullable=False),
    Column("sod_open", Float),
    Column("kmin", Float),
    Column("kmax", Float),
    Column("selected_contracts", Integer),
    Column("written_rows", Integer),
    Column("skipped_existing", Integer),
    Column("skipped_illiquid", Integer),
    Column("duration_sec", Float),
    Column("status", String(16)),
    Column("error", String(512)),
)

# Do not enforce creation (compatible with restricted environments); create on first use when --log-ingestion is set
def ensure_ingestion_log():
    try:
        metadata.create_all(engine, tables=[ingestion_log])
    except Exception:
        pass


# ────────── HELPERS ──────────
def parse_osi(osi: str):
    """
    Parse OCC/OSI code:
      O:<UL><YYMMDD><C|P><strike*1000 with 8-digit zero padding>
    Example: O:AAPL251003C00255000
             UL=AAPL, YYMMDD=251003, Right=C, K=255.000
    Returns: (ul:str, exp:date, right:'C'|'P', K:float)
    """
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

import random
from threading import Lock
from collections import deque

_request_lock = Lock()
_request_times = deque(maxlen=120)  # track recent timestamps of requests
MAX_REQUESTS_PER_MIN = 100  # adjust for your API tier


def _throttle_request():
    """Simple global rate limiter for Polygon API."""
    with _request_lock:
        now = time.time()
        # remove timestamps older than 60s
        while _request_times and now - _request_times[0] > 60:
            _request_times.popleft()

        if len(_request_times) >= MAX_REQUESTS_PER_MIN:
            sleep_for = 60 - (now - _request_times[0])
            print(f"[THROTTLE] Sleeping {sleep_for:.1f}s to respect rate limit.")
            time.sleep(sleep_for)

        _request_times.append(time.time())


def _backoff_get(url: str, params: dict, retries: int = 5, base: float = 30.0) -> dict:
    for attempt in range(retries + 1):
        try:
            _throttle_request()  # <--- enforce rate limit before sending
            # Add a small random jitter before the request on subsequent attempts
            if attempt > 0:
                 time.sleep(random.uniform(0.1, 1.0))

            r = requests.get(url, params=params, timeout=60) # Increased timeout for safety
            
            if r.status_code == 429:
                delay = 60 + random.uniform(5, 15)  # back off for ~1 minute
                print(f"[429] Throttled. Sleeping {delay:.2f}s before retry.")
                time.sleep(delay)
                continue
                
            r.raise_for_status()
            return r.json()
            
        except Exception as e:
            if attempt == retries:
                # Check if the error is due to an extended 429 response before re-raising
                if "too many 429 error responses" in str(e):
                    # Re-raise with a clear message suggesting a manual cooldown
                    raise requests.exceptions.HTTPError(
                        f"Failed after {retries} retries due to persistent 429 errors. "
                        "Consider a longer manual cooldown or reducing batch size."
                    ) from e
                raise

            # Non-429 error (e.g., connection, timeout) uses a gentler backoff
            delay = (base * 0.5 * (2 ** attempt)) + random.uniform(1, 5) 
            print(f"[HTTP] {e} — retry {attempt+1}/{retries} in {delay:.2f}s")
            time.sleep(delay)
            
    return {}


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


# ────────── POLYGON FETCHERS ──────────
def get_sod_from_polygon(ticker: str, d: date, client: Optional[Any] = None) -> Optional[float]:
    """Get daily open as SOD. Prefer SDK, fall back to REST."""
    if client is not None:
        for a in client.list_aggs(ticker, 1, "day", d.isoformat(), d.isoformat(),
                                  adjusted=True, sort="asc", limit=1):
            o = _val(a, "open")
            return float(o) if o is not None else None
        return None

    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{d}/{d}"
    js = _backoff_get(url, {"adjusted": "true", "sort": "asc", "limit": 1, "apiKey": POLY_KEY})
    rows = js.get("results", []) or []
    if not rows:
        return None
    return float(rows[0].get("o")) if rows[0].get("o") is not None else None


def minute_aggs_polygon(osi: str, d: date, client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Fetch minute bars (13:30–20:00 UTC) and return a unified list of dicts {t,o,h,l,c,v,vw,n}, with `t` in milliseconds.
    No forward filling is done here (per spec, forward-fill happens in the visualization layer).
    """
    rows: List[Dict[str, Any]] = []
    t0 = pd.Timestamp(f"{d} 13:30", tz="UTC")
    t1 = pd.Timestamp(f"{d} 20:00", tz="UTC")

    if client is not None:
        for a in client.list_aggs(osi, 1, "minute", d.isoformat(), d.isoformat(),
                                  adjusted=True, sort="asc", limit=50000):
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

    url = f"{POLY_BASE}/v2/aggs/ticker/{osi}/range/1/minute/{d}/{d}"
    js = _backoff_get(url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLY_KEY})
    raw = js.get("results", []) or []
    for r in raw:
        t_ms = r.get("t")
        if t_ms is None:
            continue
        ts = pd.to_datetime(int(t_ms), unit="ms", utc=True)
        if t0 <= ts <= t1:
            rows.append({
                "t": int(t_ms),
                "o": r.get("o"),
                "h": r.get("h"),
                "l": r.get("l"),
                "c": r.get("c"),
                "v": r.get("v"),
                "vw": r.get("vw"),
                "n": r.get("n"),
            })
    return rows


# ────────── CONTRACT DISCOVERY ──────────
def discover_contracts_from_db(company_id: int, d: date, want_oi: bool = True) -> List[Dict[str, Any]]:
    """
    Read the contract set (distinct option_name) for the given day from DB (option_quotes).
    Returns: {osi, ctype, strike, expiry_ts, eod_oi?}
    If expiry_ts is missing, derive it from OSI.
    """
    start_ts = int(pd.Timestamp(datetime(d.year, d.month, d.day, 0, 0), tz="UTC").timestamp())
    next_ts  = start_ts + 86400
    with Session(engine) as s:
        cols = [
            OptionQuote.option_name,
            OptionQuote.contract_type,
            OptionQuote.strike,
            OptionQuote.contract_expiry,
        ]

        q = (
            s.query(*cols)
            .filter(
                OptionQuote.company_id == company_id,
                OptionQuote.time_entry_ts >= start_ts,
                OptionQuote.time_entry_ts <  next_ts,
            )
            .distinct(OptionQuote.option_name)
        )
        raw = q.all()

    out = []
    for row in raw:
        osi, ctype, strike, expiry_ts = row[:4]
        eod_oi = row[4] if want_oi and len(row) > 4 else None
        if not osi or strike is None:
            continue
        osi_norm = osi if str(osi).startswith("O:") else f"O:{osi}"
        if expiry_ts is None:
            try:
                expiry_ts = _expiry_ts_from_osi(osi_norm)
            except Exception:
                continue
        out.append({
            "osi": osi_norm,
            "ctype": ctype or "C",
            "strike": float(strike),
            "expiry_ts": int(expiry_ts),
            "eod_oi": None if eod_oi is None else int(eod_oi),
        })
    return out


def discover_contracts_from_av(ticker: str, d: date) -> List[Dict[str, Any]]:
    """Fallback when DB has no results: discover contracts for the day via AV HISTORICAL_OPTIONS (for OSI/strike/expiry only)."""
    if not ALPHA_KEY:
        return []
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": ticker,
        "date": d.isoformat(),
        "apikey": ALPHA_KEY,
    }
    js = _backoff_get(AV_BASE, params)
    data = js.get("data", []) or []
    out = []
    for c in data:
        osi = c.get("contractID")
        if not osi:
            continue
        try:
            ul, exp, right, K = parse_osi(osi)
        except Exception:
            continue
        out.append({
            "osi": osi if str(osi).startswith("O:") else f"O:{osi}",
            "ctype": "C" if (c.get("type","").lower()=="call") else "P",
            "strike": float(K),
            "expiry_ts": int(pd.Timestamp(f"{exp} 20:00", tz="UTC").timestamp()),
            "eod_oi": None,
        })
    return out


# ────────── UPSERT ──────────
def upsert_minute_rows(rows: List[Dict[str, Any]]):
    """Upsert on (option_name, time_entry_ts); only minute OHLCV columns are written/updated."""
    if not rows:
        return
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


# ────────── CORE ──────────
def run_for_day(
    ticker: str,
    day_str: str,
    k_pct: float = 0.10,          
    max_contracts: int = 250,
    max_dte: int = 30,            
    min_oi: int = 0,
    min_trades: int = 0,
    resume: bool = False,
    anchor_S: Optional[float] = None,   
):
    d = pd.to_datetime(day_str).date()
    cid = get_company_id(ticker)
    client = PolyRESTClient(POLY_KEY) if HAS_POLY_SDK else None

    log = dict(
        ticker=ticker, day=d, sod_open=None, kmin=None, kmax=None,
        selected_contracts=0, written_rows=0, skipped_existing=0,
        skipped_illiquid=0, duration_sec=None, status="started", error=None
    )

    # 1) SOD
    if anchor_S is not None:
        S = float(anchor_S)
    else:
        S = get_sod_from_polygon(ticker, d, client=client)

    if S is None or S <= 0:
        print(f"[SOD SKIP] {ticker} {d}: no valid S anchor (batch or daily).")
        log.update(status="no_sod")
        return

    kmin = S * (1.0 - k_pct)
    kmax = S * (1.0 + k_pct)
    log.update(sod_open=S, kmin=kmin, kmax=kmax)

    # 2) Contract list: prefer DB, fall back to AV
    contracts = discover_contracts_from_db(cid, d, want_oi=True)
    if not contracts:
        alt = discover_contracts_from_av(ticker, d)
        if alt:
            contracts = alt
            print(f"[INFO] Using AV contracts list as fallback: {len(contracts)}")
        else:
            print(f"[INFO] {ticker} {d}: no contracts list in DB or AV; abort.")
            log.update(status="no_contracts")
            return

    # 3) SOD-based fetch filter: 0 < DTE ≤ max_dte; K ∈ [0.9S, 1.1S]; coarse OI filter; prefer near-ATM
    items = []
    for c in contracts:
        try:
            K = float(c["strike"])
            exp = pd.to_datetime(int(c["expiry_ts"]), unit="s", utc=True).date()
            dte = (exp - d).days
            if dte <= 0 or dte > max_dte:
                continue
            if not (kmin <= K <= kmax):
                continue
            if min_oi > 0:
                oi = c.get("eod_oi", 0) or 0
                if oi < min_oi:
                    continue
            logm = float(np.log(K / S))
            items.append({"osi": c["osi"], "ctype": c.get("ctype", "C"), "K": K, "dte": dte, "logm": logm})
        except Exception:
            continue

    if not items:
        print(f"[INFO] {ticker} {d}: no contracts within DTE≤{max_dte} and K∈[{kmin:.2f},{kmax:.2f}].")
        log.update(status="empty_after_filter")
        return

    dfc = pd.DataFrame(items).drop_duplicates(subset=["osi"])
    dfc["atm_rank"] = dfc["logm"].abs()
    dfc.sort_values(["atm_rank", "dte"], inplace=True)
    dfc = dfc.head(max_contracts)

    print(
        f"Plan: ticker={ticker}, day={d}, SOD={S:.4f}, K∈[{kmin:.2f},{kmax:.2f}], "
        f"max_dte={max_dte}, max_contracts={max_contracts}, selected={len(dfc)}"
    )
    log.update(selected_contracts=int(len(dfc)))

    # 4) Fetch minutes and upsert (no forward-filling; visualization layer will do that)
    total_rows = 0
    skipped_existing = 0
    skipped_illiquid = 0

    for r in dfc.itertuples(index=False):
        if resume and minutes_exist_for_osi_day(r.osi, d):
            skipped_existing += 1
            continue

        bars = minute_aggs_polygon(r.osi, d, client=client)
        if not bars:
            continue

        # Fine filter: drop “zombie” contracts whose total daily transactions are too low
        if min_trades > 0:
            day_trades = sum((b.get("n") or 0) for b in bars)
            if (day_trades or 0) < min_trades:
                skipped_illiquid += 1
                continue

        exp_ts = _expiry_ts_from_osi(r.osi)
        out_rows: List[Dict[str, Any]] = []
        for b in bars:
            t_ms = b.get("t")
            if t_ms is None:
                continue
            out_rows.append(
                dict(
                    option_name=r.osi,
                    contract_type=r.ctype,
                    company_id=cid,
                    contract_expiry=exp_ts,
                    strike=float(r.K),

                    time_entry_ts=int(int(t_ms) / 1000),

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
        print(f"[OK] {r.osi}: wrote {len(out_rows)} minute bars")

    print(f"[DONE] {ticker} {d}: total minute rows upserted = {total_rows}")
    log.update(
        written_rows=int(total_rows),
        skipped_existing=int(skipped_existing),
        skipped_illiquid=int(skipped_illiquid),
        status="ok"
    )
import argparse

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
    t_start_total = time.time()

    for ticker in tickers:
        print(f"\n========== TICKER {ticker} ==========")
        # now hardcore batch_days=14
        for batch_start, batch_end in iter_batches(start_day, end_day, batch_days=14):
            print(f"\n[BATCH] {ticker} {batch_start} → {batch_end}")

           
            try:
                client = PolyRESTClient(POLY_KEY) if HAS_POLY_SDK else None
            except Exception:
                client = None
            anchor_S = get_sod_from_polygon(ticker, batch_start, client=client)
            if anchor_S is None or anchor_S <= 0:
                print(f"[BATCH-SKIP] {ticker} {batch_start}: failed to get SOD anchor; skip this batch.")
                continue

            
            days_dt = pd.date_range(start=batch_start, end=batch_end, freq='B')
            days_str = [d.strftime('%Y-%m-%d') for d in days_dt]

            for day_str in days_str:
                print(f"--- Running for {ticker} on {day_str} (BATCH_S={anchor_S:.4f}, k_pct={k_pct}) ---")
                try:
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
                except Exception as e:
                    print(f"[!!! ERROR !!!] Unhandled exception for {ticker} on {day_str}: {e}")
                    results.append(dict(ticker=ticker, day=day_str, status="error", error=str(e), duration_sec=0))

               
                sep_len = len(f"--- Running for {ticker} on {day_str} ---")
                print("-" * sep_len + "\n")

    t_end_total = time.time()
    print("\n" + "="*30)
    print("--- JOB COMPLETE ---")
    print(f"Total time: {t_end_total - t_start_total:.2f} seconds")

    if results:
        df_results = pd.DataFrame(results)
        print(f"Total runs: {len(df_results)}")

        if "status" in df_results.columns:
            status_counts = df_results['status'].value_counts(dropna=False)
            print("\nStatus Summary:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

        ok_runs = df_results[df_results.get('status') == 'ok'] if 'status' in df_results.columns else pd.DataFrame()
        if not ok_runs.empty:
            print("\nSuccess Summary (status='ok'):")
            if 'selected_contracts' in ok_runs.columns:
                print(f"  Total contracts selected: {ok_runs['selected_contracts'].sum()}")
            if 'written_rows' in ok_runs.columns:
                print(f"  Total rows written: {ok_runs['written_rows'].sum()}")
            if 'skipped_existing' in ok_runs.columns:
                print(f"  Total skipped (resume): {ok_runs['skipped_existing'].sum()}")
            if 'skipped_illiquid' in ok_runs.columns:
                print(f"  Total skipped (illiquid): {ok_runs['skipped_illiquid'].sum()}")
            if 'duration_sec' in ok_runs.columns and ok_runs['duration_sec'].notnull().any():
                print(f"  Avg duration (sec): {ok_runs['duration_sec'].mean():.2f}")

    print("="*30)


if __name__ == "__main__":
    main()