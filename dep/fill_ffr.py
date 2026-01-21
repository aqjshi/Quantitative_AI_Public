

import os
import argparse
from urllib.parse import quote_plus
from datetime import datetime, timezone, timedelta
import numpy as np
import requests
import pandas as pd
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

# --- ORM models (from your models.py, no modifications needed) ---
from models import Base, FEDERAL_FUNDS_RATE, OptionQuote  # keep your original names

from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
# ────────── CONFIG ──────────
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"





SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
Base.metadata.create_all(engine)


def to_epoch_sec(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def parse_local_naive_to_utc_epoch(s: str) -> int:
    # Parse "YYYY-MM-DD HH:MM:SS" as naive local time then treat it as UTC (for deterministic behavior)
    # If you want local tz, adjust here. We keep UTC to match DB usage in your other scripts.
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return to_epoch_sec(dt)

# ----------------------- Alpha Vantage client -----------------------
def fetch_ffr_daily(api_key: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: date (datetime UTC midnight), value (float), ts (epoch seconds at 20:00:00 UTC)
    We set the point-in-time as 20:00:00 UTC for consistency with your EOD convention.
    """
    params = {
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "daily",
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    j = resp.json()

    data = j.get("data", []) or []
    if not data:
        return pd.DataFrame(columns=["date", "value", "ts"])

    df = pd.DataFrame(data)
    # AV returns strings: {"date": "2025-10-01", "value": "5.33"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # anchor at 20:00:00 UTC same as your EOD convention
    df['datetime_anchor'] = df["date"].dt.floor("D").dt.tz_localize(None).dt.tz_localize(timezone.utc) + pd.Timedelta(hours=20)

    # Calculate 'ts' using the .dt.tz_convert and integer division method which is more reliable
    df["ts"] = (df['datetime_anchor'].astype(np.int64) // 10**9).astype(int)
    df.dropna(subset=["date", "value", "ts"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    return df[["date", "value", "ts"]]

# ----------------------- DB upserts -----------------------
def upsert_ffr(engine, df_ffr: pd.DataFrame, batch: int = 5000) -> int:
    if df_ffr.empty:
        return 0
    rows = [{"time_entry_ts": int(r.ts), "risk_free_rate_r": float(r.value)} for r in df_ffr.itertuples(index=False)]
    affected = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i+batch]
            stmt = pg_insert(FEDERAL_FUNDS_RATE.__table__).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=[FEDERAL_FUNDS_RATE.time_entry_ts],
                set_={"risk_free_rate_r": stmt.excluded.risk_free_rate_r},
            )
            res = conn.execute(stmt)
            affected += res.rowcount or 0
    return affected

def query_ffr_series(engine, start_ts: int, end_ts: int):
    sql = text("""
        SELECT time_entry_ts, risk_free_rate_r
        FROM federal_funds_rate
        WHERE time_entry_ts BETWEEN :s AND :e
        ORDER BY time_entry_ts
    """)
    with engine.connect() as conn:
        return conn.execute(sql, {"s": start_ts, "e": end_ts}).fetchall()

def map_ffr_to_optionquotes_simple_loop(engine, start_ts: int, end_ts: int, batch_size: int = 20_000):
    """
    One-pass simple loop:
      - Pull FFR (<= end_ts)
      - Pull option_quotes rows [start_ts, end_ts] with NULL risk_free_rate_r
      - Walk both lists in a single forward pass to assign the latest FFR <= option_ts
      - Batch update option_quotes via VALUES join
    """
    # 1) Get all FFR up to end_ts (we need historical to do "last known <= t")
    with engine.connect() as conn:
        ffr_rows = conn.execute(text("""
            SELECT time_entry_ts, risk_free_rate_r
            FROM federal_funds_rate
            WHERE time_entry_ts <= :e
            ORDER BY time_entry_ts
        """), {"e": end_ts}).fetchall()

    if not ffr_rows:
        print("[FFR] No FFR rows available, skip mapping.")
        return 0

    # Convert to two lists for fast pointer walk
    ffr_ts = [r[0] for r in ffr_rows]
    ffr_val = [float(r[1]) for r in ffr_rows]

    # 2) Stream option rows in time order
    total_updated = 0
    with engine.begin() as conn:
        offset = 0
        while True:
            opt_rows = conn.execute(text(f"""
                SELECT id, time_entry_ts
                FROM option_quotes
                WHERE time_entry_ts BETWEEN :s AND :e
                  AND risk_free_rate_r IS NULL
                ORDER BY time_entry_ts
                LIMIT {batch_size} OFFSET {offset}
            """), {"s": start_ts, "e": end_ts}).fetchall()
            if not opt_rows:
                break

            # 3) One-pass assign: maintain pointer j to latest FFR <= option_ts
            updates = []
            j = 0
            for oid, ots in opt_rows:
                # advance j while next ffr_ts <= ots
                while j + 1 < len(ffr_ts) and ffr_ts[j + 1] <= ots:
                    j += 1
                # if earliest FFR already > ots, we cannot map (no past rate); skip
                if ffr_ts[0] > ots:
                    continue
                updates.append({"id": int(oid), "r": ffr_val[j]})

            if updates:
                # Batch update via VALUES
                values_clause = ", ".join(f"(:id{i}, :r{i})" for i in range(len(updates)))
                params = {}
                for i, u in enumerate(updates):
                    params[f"id{i}"] = u["id"]
                    params[f"r{i}"] = u["r"]
                sql = text(f"""
                    WITH src(id, r) AS (
                        VALUES {values_clause}
                    )
                    UPDATE option_quotes q
                    SET risk_free_rate_r = src.r
                    FROM src
                    WHERE q.id = src.id
                """)
                conn.execute(sql, params)
                total_updated += len(updates)

            # Advance offset window
            offset += batch_size

    print(f"[FFR→Options] Updated rows: {total_updated}")
    return total_updated

# ----------------------- Plotting -----------------------
def plot_ffr_png(engine, start_ts: int, end_ts: int, outfile: str):
    rows = query_ffr_series(engine, start_ts, end_ts)
    if not rows:
        print("[Plot] No FFR rows in window; skip plotting.")
        return
    ts = [datetime.utcfromtimestamp(r[0]) for r in rows]
    val = [float(r[1]) for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(ts, val)
    plt.title("01/01/2024 13:30:00 -10/01/2025 20:00:00 Federal funds rate linegraph")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Federal Funds Rate (%)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    print(f"[Plot] Saved {outfile} ({len(val)} points).")

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01 13:30:00")
    parser.add_argument("--end",   default="2025-10-01 20:00:00")
    parser.add_argument("--map-optionquotes", action="store_true",
                        help="Map FFR into option_quotes.risk_free_rate_r within the time window")
    args = parser.parse_args()

    if not ALPHA_VANTAGE_API_KEY:
        print("[CONFIG] Missing ALPHA_VANTAGE_API_KEY.")
        return

    # Prepare engine (and ensure tables exist if needed)
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    # 1) Fetch & upsert FFR
    df_ffr = fetch_ffr_daily(ALPHA_VANTAGE_API_KEY)
    if df_ffr.empty:
        print("[FFR] Empty response from Alpha Vantage.")
        return
    affected = upsert_ffr(engine, df_ffr)
    print(f"[FFR] Upserted/updated rows: {affected}")

    # 2) Time window (UTC)
    start_ts = parse_local_naive_to_utc_epoch(args.start)
    end_ts   = parse_local_naive_to_utc_epoch(args.end)

    # 3) Plot
    plot_ffr_png(engine, start_ts, end_ts, outfile="ffr.png")

    # 4) Optionally map into option_quotes
    if args.map_optionquotes:
        map_ffr_to_optionquotes_simple_loop(engine, start_ts, end_ts)

if __name__ == "__main__":
    main()
