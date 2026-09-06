import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from multiprocessing import get_context
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine
from pricing.demo import async_si_flag as kernel
from pricing.demo.async_si_flag import AsyncSiFlag, POLICY_PATH


def get_latest_state() -> dict:
    """Newest stamp already on record, per corporate entity.

    Same shape as maintain_async_f2.py::get_latest_state.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT figi_group, max(stamp) FROM async_si_flag GROUP BY figi_group"
        )).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def main():
    policy = json.load(open(POLICY_PATH))

    # create, never drop: this pass appends to what async_si_flag.py built.
    AsyncSiFlag.__table__.create(engine, checkfirst=True)

    marks = get_latest_state()
    if marks:
        watermark = max(marks.values())

        # WARM-UP IS MANDATORY HERE, unlike the other two maintain passes.
        #
        # f2 is point-in-time -- each timestamp is independent, so a new bar can
        # be valued from its own lookback alone. si_regime_filter is not: it is
        # path-dependent in two separate ways.
        #
        #   1. STEP 1 rolls mean/std/var over si_filter_long_offset_days. A bar
        #      evaluated without that much history behind it gets a different
        #      mean_long than the full-history run gave it, so its gates flip.
        #   2. STEP 4 carries a quarantine deadline forward. A trigger fired
        #      before the watermark can still hold the cage open after it, and
        #      recomputing from the watermark alone would silently release it.
        #
        # Reading back long_offset_days + quarantine_delta_months covers both. A
        # trigger older than that window has a deadline of at most
        # (start + quarantine_delta_months), which lands before the watermark, so
        # it cannot reach the bars being written.
        warmup_start = (pd.Timestamp(watermark)
                        - pd.Timedelta(days=policy["si_filter_long_offset_days"])
                        - pd.DateOffset(months=policy["quarantine_delta_months"]))
        tqdm.write(f"[*] Watermark {pd.Timestamp(watermark).date()} across {len(marks)} entities.")
        tqdm.write(f"[*] Replaying from {warmup_start.date()} so rolling windows and the "
                   f"quarantine cage reconstruct identically.")
    else:
        warmup_start = pd.Timestamp.min
        tqdm.write("[*] async_si_flag empty. Seeding from the full short interest history.")

    si = pd.read_sql(text(
        "SELECT figi_group, ticker, ticker_hash, stamp, pct_short_of_market_cap AS pct "
        "FROM async_short_interest WHERE stamp >= :floor ORDER BY figi_group, stamp"
    ), engine, params={"floor": int(warmup_start.value) if marks else 0})

    if si.empty:
        tqdm.write("[*] No short interest bars in window. Nothing to maintain.")
        engine.dispose()
        return
    tqdm.write(f"[*] {len(si):,} bars in replay window across "
               f"{si['figi_group'].nunique()} entities.")

    kernel._SHARED["si"] = {int(k): v for k, v in si.groupby("figi_group")}
    kernel._SHARED["policy"] = policy
    tasks = sorted(kernel._SHARED["si"].keys())
    del si
    engine.dispose()

    written = 0
    pbar = tqdm(total=len(tasks), desc="Maintaining si flags", unit="entity")
    with get_context("fork").Pool(processes=min(10, os.cpu_count() or 1)) as pool:
        for figi, out in pool.imap_unordered(kernel._worker, tasks):
            if not out.empty:
                # The warm-up bars were only there to prime the rolling windows
                # and the cage; they are already stored, so only what is strictly
                # past this entity's own mark gets appended.
                mark = marks.get(figi)
                if mark is not None:
                    out = out[out["stamp"] > mark]
            if not out.empty:
                out.to_sql("async_si_flag", engine, if_exists="append", index=False,
                           method="multi", chunksize=5000)
                written += len(out)
            pbar.set_postfix(rows=written)
            pbar.update(1)
    pbar.close()

    with engine.connect() as conn:
        stats = conn.execute(text(
            "SELECT count(*), count(DISTINCT figi_group), sum(si_flag) FROM async_si_flag"
        )).fetchone()

    tqdm.write(f"[*] Appended {written:,} rows. async_si_flag now holds {stats[0]:,} rows "
               f"across {stats[1]} corporate entities, si_flag=1 on {stats[2]:,} bars.")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
