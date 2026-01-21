import os
import time
import argparse
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typing import List, Tuple
# ORM models
from models import Base, NEWS_SENTIMENT

# ---------------- Config ----------------
load_dotenv()
from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"





SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)
WINDOW_DEFAULT_START = "2024-01-01 13:30:00"
WINDOW_DEFAULT_END   = "2025-10-01 20:00:00"

# -------------- Helpers ----------------
def to_unix_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def parse_naive_to_unix(s: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS' (assume UTC) -> unix seconds."""
    return to_unix_utc(datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))

def parse_time_published_to_unix(tp: str) -> int:
    """
    Alpha Vantage time format like '20251022T020047' (UTC).
    Convert to UNIX seconds.
    """
    try:
        dt = datetime.strptime(tp, "%Y%m%dT%H%M%S")
        return to_unix_utc(dt)
    except Exception:
        return None

def iter_month_windows(start_s: str, end_s: str) -> List[Tuple[int, int]]:
   
    start_dt = pd.to_datetime(start_s)
    end_dt   = pd.to_datetime(end_s)
    if end_dt < start_dt:
        return []

   
    cur = pd.Timestamp(start_dt.year, start_dt.month, 1, start_dt.hour, start_dt.minute, start_dt.second)
    windows: List[Tuple[int, int]] = []

    while cur <= end_dt:
        nxt_month = (cur + pd.offsets.MonthBegin(1))
        
        batch_end = min(nxt_month - pd.Timedelta(seconds=1), end_dt)
        
        batch_start = max(cur, start_dt)

        windows.append((to_unix_utc(batch_start.to_pydatetime()), to_unix_utc(batch_end.to_pydatetime())))
        cur = nxt_month

    return windows

# -------------- Fetch ----------------
def fetch_news_sentiment_for_ticker(api_key: str, ticker: str, start_ts: int, end_ts: int, max_items: int = 1000) -> list[dict]:
  
    def fmt_min(ts: int) -> str:
        return datetime.utcfromtimestamp(ts).strftime("%Y%m%dT%H%M")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": fmt_min(start_ts),
        "time_to": fmt_min(end_ts),
        "limit": str(max_items),
        "apikey": api_key,
    }

   
    for attempt in range(3):
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and "Note" in j:
            if attempt < 2:
                time.sleep(12)
                continue
            else:
                print(f"[AV] Rate limit Note for {ticker}; return empty for this window.")
                return []
        feed = j.get("feed", []) or []

        out = []
        tgt = ticker.upper()
        for item in feed:
            ts = parse_time_published_to_unix(item.get("time_published", ""))
            if ts is None:
                continue
            tlist = item.get("ticker_sentiment") or []
            has_tgt = any((isinstance(t, dict) and str(t.get("ticker","")).upper() == tgt) for t in tlist)
            if has_tgt:
                out.append(item)
        return out

    return []


# -------------- Upsert ----------------
def _dedupe_rows_by_url(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        u = r.get("url")
        if not u:
            continue
        seen[u] = r
    return list(seen.values())

def upsert_news(engine, items: list[dict]) -> int:
    """
    New addition: Pre-batch duplicate removal
    """
    if not items:
        return 0

    rows = []
    for it in items:
        ts = parse_time_published_to_unix(it.get("time_published", ""))
        if ts is None:
            continue

        rows.append({
            "title": it.get("title"),
            "url": it.get("url"),
            "summary": it.get("summary"),
            "time_published_ts": int(ts),
            "source": it.get("source"),
            "source_domain": it.get("source_domain"),
            "authors": it.get("authors"),                 # list
            "topics": it.get("topics"),                   # list of objects
            "ticker_sentiment": it.get("ticker_sentiment"),  # list of objects
            "overall_sentiment_score": _num_or_none(it.get("overall_sentiment_score")),
            "overall_sentiment_label": it.get("overall_sentiment_label"),
        })

   
    rows = [r for r in rows if r.get("url")]
    if not rows:
        return 0

    
    rows = _dedupe_rows_by_url(rows)

    affected = 0
    CHUNK = 500 
    with engine.begin() as conn:
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i:i+CHUNK]

            
            chunk = _dedupe_rows_by_url(chunk)
            if not chunk:
                continue

            stmt = pg_insert(NEWS_SENTIMENT.__table__).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=[NEWS_SENTIMENT.url],
                set_={
                    "title": stmt.excluded.title,
                    "summary": stmt.excluded.summary,
                    "time_published_ts": stmt.excluded.time_published_ts,
                    "source": stmt.excluded.source,
                    "source_domain": stmt.excluded.source_domain,
                    "authors": stmt.excluded.authors,
                    "topics": stmt.excluded.topics,
                    "ticker_sentiment": stmt.excluded.ticker_sentiment,
                    "overall_sentiment_score": stmt.excluded.overall_sentiment_score,
                    "overall_sentiment_label": stmt.excluded.overall_sentiment_label,
                }
            )
            res = conn.execute(stmt)
            affected += res.rowcount or 0

    return affected

def _num_or_none(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None

# -------------- Plot ----------------
def plot_news_sentiment(engine, start_ts: int, end_ts: int, ticker: str, outfile: str):
    sql = text("""
        SELECT time_published_ts, ticker_sentiment
        FROM news_sentiment
        WHERE time_published_ts BETWEEN :s AND :e
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"s": start_ts, "e": end_ts}).fetchall()

    if not rows:
        print(f"[Plot] No news rows in window for {ticker}; skip plot.")
        return

    tgt = ticker.upper()
    agg = {}
    for ts, tlist in rows:
        if not isinstance(tlist, (list, tuple)):
            try:
                import json
                tlist = json.loads(tlist)
            except Exception:
                tlist = []

        rel_sum = 0.0
        sent_sum = 0.0
        for t in tlist or []:
            try:
                if isinstance(t, dict) and str(t.get("ticker","")).upper() == tgt:
                    rel_sum  += float(t.get("relevance_score") or 0.0)
                    sent_sum += float(t.get("ticker_sentiment_score") or 0.0)
            except Exception:
                continue
        if rel_sum == 0.0 and sent_sum == 0.0:
            continue
        if ts not in agg:
            agg[ts] = [0.0, 0.0]
        agg[ts][0] += rel_sum
        agg[ts][1] += sent_sum

    if not agg:
        print(f"[Plot] No {ticker} ticker sentiments found in window; skip plot.")
        return

    xs = sorted(agg.keys())
    y_rel = [agg[t][0] for t in xs]
    y_sco = [agg[t][1] for t in xs]
    x_dt = [datetime.utcfromtimestamp(t) for t in xs]

    plt.figure(figsize=(10, 4))
    plt.plot(x_dt, y_rel, label="relevance_score (sum)")
    plt.plot(x_dt, y_sco, label="ticker_sentiment_score (sum)")
    title = f"{datetime.utcfromtimestamp(start_ts)} - {datetime.utcfromtimestamp(end_ts)} news for {ticker} (2-line chart)"
    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    print(f"[Plot] Saved {outfile} for {ticker} with {len(xs)} points.")

import json 

# -------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", nargs="?", help="path to params.json (preferred)")
    parser.add_argument("--start", default=WINDOW_DEFAULT_START, help="fallback start (if no params_file)")
    parser.add_argument("--end",   default=WINDOW_DEFAULT_END,   help="fallback end (if no params_file)")
    parser.add_argument("--limit", type=int, default=1000, help="Alpha Vantage 'limit' parameter per request")
    args = parser.parse_args()

    if not ALPHA_VANTAGE_API_KEY:
        print("[CONFIG] Missing ALPHA_VANTAGE_API_KEY.")
        return

    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    
    if args.params_file:
        with open(args.params_file, "r", encoding="utf-8") as f:
            j = json.load(f)

        tickers: List[str] = j.get("ticker", []) or []
        start_s: str = j.get("train_start", WINDOW_DEFAULT_START)
        end_s:   str = j.get("train_end",   WINDOW_DEFAULT_END)
        out_dir = j.get("output_dir", ".")
        limit   = int(j.get("news_limit", args.limit))

        os.makedirs(out_dir, exist_ok=True)

        
        for tkr in tickers:
            print(f"\n=== NEWS for {tkr} | {start_s} ~ {end_s} (monthly windows) ===")
            month_windows = iter_month_windows(start_s, end_s)
            total_items = 0
            for (ts0, ts1) in month_windows:
                print(f"[AV] {tkr} window {datetime.utcfromtimestamp(ts0)} ~ {datetime.utcfromtimestamp(ts1)}")
                feed = fetch_news_sentiment_for_ticker(ALPHA_VANTAGE_API_KEY, tkr, ts0, ts1, max_items=limit)
                total_items += len(feed)
                affected = upsert_news(engine, feed)
                print(f"[DB] Upserted/updated: {affected} rows for {tkr} in this window.")
                
                time.sleep(1.0)

            
            start_ts = parse_naive_to_unix(start_s)
            end_ts   = parse_naive_to_unix(end_s)
            outfile = os.path.join(out_dir, f"{tkr}_news_{start_s.replace(' ','_').replace(':','-')}_to_{end_s.replace(' ','_').replace(':','-')}.png")
            print(f"[PLOT] Building {outfile} (total fetched items ~ {total_items})")
            plot_news_sentiment(engine, start_ts, end_ts, ticker=tkr, outfile=outfile)

        print("\n[ALL DONE] news sentiment ingestion & plots finished.")
        return

   
    start_ts = parse_naive_to_unix(args.start)
    end_ts   = parse_naive_to_unix(args.end)

    print("[AV] Fetching NEWS_SENTIMENT for AAPL (single-shot fallback)...")
    feed = fetch_news_sentiment_for_ticker(ALPHA_VANTAGE_API_KEY, "AAPL", start_ts, end_ts, max_items=args.limit)
    print(f"[AV] Got {len(feed)} items with AAPL in ticker_sentiment.")
    affected = upsert_news(engine, feed)
    print(f"[DB] Upserted/updated rows: {affected}")
    plot_news_sentiment(engine, start_ts, end_ts, ticker="AAPL", outfile="news_sentiment.png")

if __name__ == "__main__":
    main()