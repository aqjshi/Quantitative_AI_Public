import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine
from pricing.database.identity import _prepare_global_identity
from pricing.database.fetcher import _fetch_clean_universe
from pricing.demo.async_short_interest import (
    AsyncShortInterest, POLICY_PATH, reconstruct_pit_short_interest_series,
)


def get_latest_state() -> dict:
    """Newest stamp already on record, per corporate entity.

    Same idea as indices/maintain_subset.py::get_latest_state -- what is already
    stored decides where the next pass begins -- but keyed per figi_group rather
    than as one scalar, because entities do not all end on the same date.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT figi_group, max(stamp) FROM async_short_interest GROUP BY figi_group"
        )).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def main():
    policy = json.load(open(POLICY_PATH))
    universe = sorted(set(policy["in_set"] + policy["intermediate_set"]
                          + policy["hold_out"] + policy["benchmark_set"]))

    # create, never drop: this pass appends to what async_short_interest.py built.
    AsyncShortInterest.__table__.create(engine, checkfirst=True)

    marks = get_latest_state()
    if marks:
        # _fetch_clean_universe walks back context_depth_months for quotes and
        # fundamental_context_depth_months for fundamentals from this date, so an
        # entity lagging the leader still gets its own missing bars rebuilt, and
        # the 4-quarter maturity checkpoint keeps its runway.
        start = pd.Timestamp(max(marks.values()))
        tqdm.write(f"[*] Watermark {start.date()} across {len(marks)} entities on record.")
    else:
        start = pd.to_datetime(policy["fetch_start"])
        tqdm.write(f"[*] async_short_interest empty. Seeding from fetch_start: {start.date()}")

    db_fetch_end = datetime.today()
    current_time = pd.to_datetime(db_fetch_end)

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

    written = 0
    pbar = tqdm(total=len(entities), desc="Maintaining short interest", unit="entity")
    for row in entities.itertuples():
        aligned = reconstruct_pit_short_interest_series(
            row, current_time, full_quotes_df, full_fundamentals_df
        )
        if not aligned.empty:
            out = pd.DataFrame({
                "stamp": aligned["t"].astype("datetime64[ns]").astype("int64"),
                "ticker": row.ticker,
                "ticker_hash": aligned["ticker_hash"],
                "figi_group": row.composite_figi_hash,
                "adj_price": aligned["p_adjusted"],
                "adj_volume": aligned["adj_volume"],
                "unadj_price": aligned["p_nominal"],
                "unadj_volume": aligned["v"],
                "market_cap": aligned["market_cap"],
                "basic_shares_outstanding": aligned["basic_shares_outstanding"],
                "pct_short_of_market_cap": aligned["si_to_cap_ratio"] * 100.0,
            })
            # Strictly past this entity's own high-water mark, so a re-run is
            # idempotent without needing a unique constraint to conflict on.
            mark = marks.get(row.composite_figi_hash)
            if mark is not None:
                out = out[out["stamp"] > mark]
            if not out.empty:
                out.to_sql("async_short_interest", engine, if_exists="append",
                           index=False, method="multi", chunksize=5000)
                written += len(out)
        pbar.set_postfix(rows=written)
        pbar.update(1)
    pbar.close()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM async_short_interest")).scalar()
        covered = conn.execute(text("SELECT count(DISTINCT figi_group) FROM async_short_interest")).scalar()

    tqdm.write(f"[*] Appended {written:,} rows. async_short_interest now holds "
               f"{total:,} rows across {covered} corporate entities.")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
