import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import get_context
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine
from pricing.database.identity import _prepare_global_identity
from pricing.database.fetcher import _fetch_clean_universe
from pricing.demo import async_f2 as kernel
from pricing.demo.async_f2 import AsyncF2, PAYLOAD_COLS, POLICY_PATH


def get_latest_state() -> dict:
    """Newest stamp already on record, per corporate entity.

    Same idea as indices/maintain_subset.py::get_latest_state -- what is already
    stored decides where the next pass begins -- but keyed per figi_group rather
    than as one scalar. A single global MAX would hand an entity that stopped
    early a block of sessions it already has, because entities do not all end on
    the same date: a delisted name stops at its last bar while a live one runs to
    the most recent session.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT figi_group, max(stamp) FROM async_f2 GROUP BY figi_group"
        )).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def main():
    policy = json.load(open(POLICY_PATH))
    universe = sorted(set(policy["in_set"] + policy["intermediate_set"]
                          + policy["hold_out"] + policy["benchmark_set"]))

    # create, never drop: this pass appends to what async_f2.py built.
    AsyncF2.__table__.create(engine, checkfirst=True)

    marks = get_latest_state()
    if marks:
        # The fetch begins at the newest stored session. _fetch_clean_universe
        # then walks back context_depth_months for quotes and
        # fundamental_context_depth_months for fundamentals on its own, so an
        # entity lagging the leader by days or weeks still gets its own missing
        # sessions rebuilt inside that padding, and the 4-quarter TTM minimum
        # still has its runway.
        start = pd.Timestamp(max(marks.values()))
        tqdm.write(f"[*] Watermark {start.date()} across {len(marks)} entities on record.")
    else:
        start = pd.to_datetime(policy["fetch_start"])
        tqdm.write(f"[*] async_f2 empty. Seeding from fetch_start: {start.date()}")

    db_fetch_end = datetime.today()
    global_identity = _prepare_global_identity(
        universe, pd.Timestamp(db_fetch_end),
        policy["reconstruction_heartbeat_freq_months"],
    )
    full_quotes_df, full_fundamentals_df = _fetch_clean_universe(
        global_identity, start, pd.Timestamp(db_fetch_end),
        policy["context_depth_months"],
        policy["forecast_depth_months"],
        policy["fundamental_context_depth_months"],
        policy["fundamental_proxy_filing_date_delta_days"],
        policy["reconstruction_heartbeat_freq_months"],
    )
    if full_quotes_df.empty or full_fundamentals_df.empty:
        tqdm.write("[*] No new panel data in window. Nothing to maintain.")
        engine.dispose()
        return

    full_quotes_df = full_quotes_df.drop_duplicates(
        subset=['ticker_hash', 't', 'adjusted'], keep='last'
    )
    full_fundamentals_df['period_end_t'] = (
        pd.to_datetime(full_fundamentals_df['period_end_t']).astype('datetime64[ns]')
    )

    entities = (global_identity.sort_values("latest")
                               .drop_duplicates(subset=["composite_figi_hash"], keep="last"))

    kernel._SHARED["quotes"] = {k: v for k, v in full_quotes_df.groupby('figi_group')}
    kernel._SHARED["fund"] = {k: v for k, v in full_fundamentals_df.groupby('figi_group')}
    kernel._SHARED["policy"] = policy
    ticker_by_hash = dict(zip(global_identity['ticker_hash'], global_identity['ticker']))

    tasks = [int(h) for h in entities["composite_figi_hash"]]
    engine.dispose()

    written = 0
    pbar = tqdm(total=len(tasks), desc="Maintaining f2", unit="entity")
    with get_context("fork").Pool(processes=min(10, os.cpu_count() or 1)) as pool:
        for figi, panel in pool.imap_unordered(kernel._worker, tasks):
            if not panel.empty:
                # Strictly past this entity's own high-water mark, so a re-run is
                # idempotent without needing a unique constraint to conflict on.
                mark = marks.get(figi)
                if mark is not None:
                    panel = panel[panel["stamp"] > mark]
            if not panel.empty:
                out = pd.DataFrame({
                    "stamp": panel["stamp"],
                    "ticker": panel["ticker_hash"].map(ticker_by_hash),
                    "ticker_hash": panel["ticker_hash"],
                    "figi_group": figi,
                    **{c: panel[c] for c in PAYLOAD_COLS},
                })
                out.to_sql("async_f2", engine, if_exists="append", index=False,
                           method="multi", chunksize=5000)
                written += len(out)
            pbar.set_postfix(rows=written)
            pbar.update(1)
    pbar.close()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM async_f2")).scalar()
        covered = conn.execute(text("SELECT count(DISTINCT figi_group) FROM async_f2")).scalar()

    tqdm.write(f"[*] Appended {written:,} rows. async_f2 now holds {total:,} "
               f"rows across {covered} corporate entities.")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
