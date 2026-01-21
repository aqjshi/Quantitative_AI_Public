import os
import argparse
from urllib.parse import quote_plus
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ORM models (use your existing models.py)
from models import Base, TREASURY_YIELD, OptionQuote

# ----------------------- Config -----------------------
from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
# ────────── CONFIG ──────────
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"





SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)


WINDOW_DEFAULT_START = "2024-01-01 13:30:00"
WINDOW_DEFAULT_END   = "2025-10-01 20:00:00"

MATURITIES = ["3month", "2year", "5year", "7year", "10year"]
COLMAP = {
    "3month": "_3month",
    "2year":  "_2year",
    "5year":  "_5year",
    "7year":  "_7year",
    "10year": "_10year",
}

def parse_utc_epoch(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

# ----------------------- Fetch AV -----------------------
def fetch_treasury_yield_daily(maturity: str, api_key: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: date (UTC), value (float), ts (epoch seconds at 20:00:00 UTC)
    """
    params = {
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": maturity,
        "apikey": api_key,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    data = j.get("data", []) or []
    if not data:
        return pd.DataFrame(columns=["date", "value", "ts"])

    df = pd.DataFrame(data)
    # AV returns: {"date":"2025-10-01","value":"4.50"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ts"] = (df["date"].dt.floor("D") + pd.Timedelta(hours=20)).astype("int64") // 10**9
    df.dropna(subset=["date", "value", "ts"], inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.sort_values("ts", inplace=True)
    return df[["date", "value", "ts"]]

def merge_curves_one_pass(curve_map: dict) -> pd.DataFrame:
    """
    curve_map: maturity -> DataFrame(date,value,ts)
    Output: one row per ts with columns: ts,_3month,_2year,_5year,_7year,_10year
    One-pass merge by ts using outer union + pivot-like reconstruction.
    """
    # Collect all timestamps
    all_ts = set()
    for df in curve_map.values():
        all_ts.update(df["ts"].tolist())
    if not all_ts:
        return pd.DataFrame(columns=["ts", *_ordered_cols()])

    all_ts = sorted(all_ts)
    # Build mapping ts -> dict of cols
    rows = []
    # Convert each maturity df to dict(ts->value) for O(1) lookup
    dicts = {mat: dict(zip(df["ts"], df["value"])) for mat, df in curve_map.items()}
    for ts in all_ts:
        row = {"ts": int(ts)}
        for mat in MATURITIES:
            col = COLMAP[mat]
            row[col] = dicts.get(mat, {}).get(ts)  # None if missing
        rows.append(row)
    return pd.DataFrame(rows)

def _ordered_cols():
    return [COLMAP[m] for m in MATURITIES]

# ----------------------- Upsert -----------------------
def upsert_treasury(engine, df_ty: pd.DataFrame, batch: int = 5000) -> int:
    """
    将合并后的曲线 df_ty（列: ts,_3month,_2year,_5year,_7year,_10year）UPSERT 进 treasury_yield。
    注意：列名以下划线开头时，itertuples 的属性访问会失效，因此改用位置访问。
    """
    if df_ty.empty:
        return 0

    # 保证列齐全且顺序固定
    cols = ["ts", "_3month", "_2year", "_5year", "_7year", "_10year"]
    for c in cols:
        if c not in df_ty.columns:
            df_ty[c] = pd.NA
    df_ty = df_ty[cols]

    def _num_or_none(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    # 使用位置解包，避免以下划线列名带来的属性访问问题
    rows = []
    for ts, v3, v2, v5, v7, v10 in df_ty.itertuples(index=False, name=None):
        rows.append({
            "time_entry_ts": int(ts),
            "_3month": _num_or_none(v3),
            "_2year":  _num_or_none(v2),
            "_5year":  _num_or_none(v5),
            "_7year":  _num_or_none(v7),
            "_10year": _num_or_none(v10),
        })

    if not rows:
        return 0

    affected = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i+batch]
            stmt = pg_insert(TREASURY_YIELD.__table__).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=[TREASURY_YIELD.time_entry_ts],
                set_={
                    "_3month": stmt.excluded._3month,
                    "_2year":  stmt.excluded._2year,
                    "_5year":  stmt.excluded._5year,
                    "_7year":  stmt.excluded._7year,
                    "_10year": stmt.excluded._10year,
                }
            )
            res = conn.execute(stmt)
            affected += res.rowcount or 0
    return affected

# ----------------------- OptionQuotes mapping (optional) -----------------------
def map_treasury_to_optionquotes(engine, start_ts: int, end_ts: int, benchmark: str = "3mo", batch_size: int = 20000):
    """
    One-pass simple loop: choose a benchmark curve (e.g., 3mo) as risk_free_rate_r
    Map to option_quotes.risk_free_rate_r for rows within [start,end] where it's NULL.
    """
    col = COLMAP[benchmark]
    with engine.connect() as conn:
        ty_rows = conn.execute(text(f"""
            SELECT time_entry_ts, {col}
            FROM treasury_yield
            WHERE time_entry_ts <= :e
              AND {col} IS NOT NULL
            ORDER BY time_entry_ts
        """), {"e": end_ts}).fetchall()
    if not ty_rows:
        print(f"[TY] No treasury rows for benchmark {benchmark}; skip mapping.")
        return 0

    ty_ts  = [r[0] for r in ty_rows]
    ty_val = [float(r[1]) for r in ty_rows]

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

            updates = []
            j = 0
            for oid, ots in opt_rows:
                while j + 1 < len(ty_ts) and ty_ts[j + 1] <= ots:
                    j += 1
                if ty_ts[0] > ots:
                    continue
                updates.append({"id": int(oid), "r": ty_val[j]})

            if updates:
                values_clause = ", ".join(f"(:id{i}, :r{i})" for i in range(len(updates)))
                params = {}
                for i, u in enumerate(updates):
                    params[f"id{i}"] = u["id"]
                    params[f"r{i}"]  = u["r"]
                sql = text(f"""
                    WITH src(id, r) AS (VALUES {values_clause})
                    UPDATE option_quotes q
                    SET risk_free_rate_r = src.r
                    FROM src
                    WHERE q.id = src.id
                """)
                conn.execute(sql, params)
                total_updated += len(updates)

            offset += batch_size

    print(f"[TY→Options] Updated rows (benchmark={benchmark}): {total_updated}")
    return total_updated

# ----------------------- Plot -----------------------
def plot_ty_png(engine, start_ts: int, end_ts: int, outfile: str = "ty.png"):
    sql = text("""
        SELECT time_entry_ts, _3month, _2year, _5year, _7year, _10year
        FROM treasury_yield
        WHERE time_entry_ts BETWEEN :s AND :e
        ORDER BY time_entry_ts
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"s": start_ts, "e": end_ts}).fetchall()
    if not rows:
        print("[Plot] No treasury rows in window; skip plotting.")
        return

    ts = [datetime.utcfromtimestamp(r[0]) for r in rows]
    y3  = [r[1] for r in rows]
    y2  = [r[2] for r in rows]
    y5  = [r[3] for r in rows]
    y7  = [r[4] for r in rows]
    y10 = [r[5] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(ts, y3,  label="3mo")
    plt.plot(ts, y2,  label="2yr")
    plt.plot(ts, y5,  label="5yr")
    plt.plot(ts, y7,  label="7yr")
    plt.plot(ts, y10, label="10yr")
    plt.title("01/01/2024 13:30:00 -10/01/2025 20:00:00 3mo, 2yr, 5yr, 7yr, 10yr 5 lines linegraph")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    print(f"[Plot] Saved {outfile} ({len(ts)} points).")

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=WINDOW_DEFAULT_START)
    parser.add_argument("--end",   default=WINDOW_DEFAULT_END)
    parser.add_argument("--map-optionquotes", action="store_true",
                        help="Map chosen benchmark into option_quotes.risk_free_rate_r for rows within window (NULL-only)")
    parser.add_argument("--benchmark", choices=["3mo","2yr","5yr","7yr","10yr"], default="3mo",
                        help="Benchmark curve used if --map-optionquotes is set (default 3mo)")
    args = parser.parse_args()

    if not ALPHA_VANTAGE_API_KEY:
        print("[CONFIG] Missing ALPHA_VANTAGE_API_KEY.")
        return

    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    # 1) Fetch all maturities
    curve_map = {}
    for mat in MATURITIES:
        print(f"[AV] Fetching TREASURY_YIELD {mat} ...")
        df = fetch_treasury_yield_daily(mat, ALPHA_VANTAGE_API_KEY)
        curve_map[mat] = df

    # 2) Merge to single DF with columns _3month.._10year
    merged = merge_curves_one_pass(curve_map)
    if merged.empty:
        print("[TY] Empty merged series. Nothing to upsert.")
        return

    # 3) Upsert
    affected = upsert_treasury(engine, merged)
    print(f"[TY] Upserted/updated rows: {affected}")

    # 4) Plot
    start_ts = parse_utc_epoch(args.start)
    end_ts   = parse_utc_epoch(args.end)
    plot_ty_png(engine, start_ts, end_ts, outfile="ty.png")

    # 5) Optional map into option quotes
    if args.map_optionquotes:
        bench = {"3mo":"3month","2yr":"2year","5yr":"5year","7yr":"7year","10yr":"10year"}[args.benchmark]
        map_treasury_to_optionquotes(engine, start_ts, end_ts, benchmark=bench)

if __name__ == "__main__":
    main()
