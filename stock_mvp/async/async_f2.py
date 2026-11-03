import os
import sys
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import get_context
from sqlalchemy import Column, BigInteger, String, Numeric, Float, Index, text


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine, Base
from pricing.database.identity import _prepare_global_identity
from pricing.database.fetcher import _fetch_clean_universe
from pricing.math.f2 import fundamental_f2

POLICY_PATH = os.path.join(PROJECT_ROOT, "pricing", "simulation_policy_big.json")

# pricing/dataset.py:554
PROFILE_COLS = {
    "inc": ['revenue'], "bs": ['total_assets'], "cf": ['cfo'],
    "div": ['div_amount'], "si": ['short_interest'],
}


class AsyncF2(Base):
    __tablename__ = "async_f2"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stamp = Column(BigInteger, index=True, nullable=False)
    ticker = Column(String(20))
    ticker_hash = Column(BigInteger, index=True, nullable=False)
    figi_group = Column(BigInteger, index=True)

    # --- valuation scalars ---
    derived_market_value = Column(Numeric(38, 6))
    adjusted_derived_price = Column(Numeric(28, 8))
    market_cap = Column(Numeric(38, 6))
    basic_shares_outstanding = Column(Numeric(24, 4))
    split_factor_anchor = Column(Float)

    ttm_revenue = Column(Numeric(38, 6))
    lassets = Column(Numeric(38, 6))
    tassets = Column(Numeric(38, 6))
    latassets = Column(Float)
    ebita_multiple = Column(Float)

    # --- explicit features package (f2 payload tail) ---
    tdebt_to_ttm_cfo = Column(Float)
    capex_to_topex = Column(Float)
    rd_to_topex = Column(Float)
    sga_to_topex = Column(Float)
    opportunity_cost = Column(Float)
    gross_profit_to_revenue = Column(Float)
    sloan_ratio = Column(Float)
    retained_earnings_to_tassets = Column(Float)
    net_ppe_to_tassets = Column(Float)
    accounts_receivable_to_revenue = Column(Float)

    __table_args__ = (
        Index("idx_async_f2_hash_stamp", "ticker_hash", "stamp"),
    )


PAYLOAD_COLS = [
    "derived_market_value", "adjusted_derived_price", "market_cap",
    "basic_shares_outstanding", "split_factor_anchor", "ttm_revenue",
    "lassets", "tassets", "latassets", "ebita_multiple",
    "tdebt_to_ttm_cfo", "capex_to_topex", "rd_to_topex", "sga_to_topex",
    "opportunity_cost", "gross_profit_to_revenue", "sloan_ratio",
    "retained_earnings_to_tassets", "net_ppe_to_tassets",
    "accounts_receivable_to_revenue",
]


def reconstruct_pit_f2_series(
    quotes_slice: pd.DataFrame,
    fundamentals_slice: pd.DataFrame,
    context_depth_months: int,
    fundamental_context_depth_months: int,
    proxy_days: int,
    stale_days: int,
) -> pd.DataFrame:
    """Walk one corporate entity across its trading calendar, evaluating f2 at
    every session.

    The kernel call mirrors pricing/dataset.py:697-711 exactly -- same argument
    order, same five dropna profile splits, same point-in-time fundamental window
    (period_end_t in [current_time - fundamental_context_depth_months,
    current_time)). Nothing about the valuation is reimplemented here; this only
    supplies the slices and collects the returned payload.

    Both frames arrive already keyed on figi_group and already pruned to each
    ticker_hash's own lifespan, so the FB->META rename is one continuous entity
    and the recycled FB symbol cannot leak back in.
    """
    if quotes_slice.empty or fundamentals_slice.empty:
        return pd.DataFrame()

    q = quotes_slice.sort_values('t')
    q['session'] = q['t'].dt.normalize()

    # f2 reads only the terminal close of each leg (pricing/math/f2.py:127-128),
    # so collapsing the intraday tape to one bar per session per adjustment flag
    # is lossless and ~13x cheaper to slice. Same compression as
    # pricing/demo/anchor_premium_sweep.py:203-208.
    daily = (
        q.groupby(['session', 'adjusted'], as_index=False)
         .last()[['t', 'c', 'adjusted', 'ticker_hash']]
         .sort_values('t')
         .reset_index(drop=True)
    )
    if daily.empty:
        return pd.DataFrame()

    f_sorted = fundamentals_slice.sort_values('period_end_t')
    profiles = {k: f_sorted.dropna(subset=v) for k, v in PROFILE_COLS.items()}
    profile_times = {k: v['period_end_t'].values for k, v in profiles.items()}

    def pit(key, lo, hi):
        """The profile's rows with period_end_t in [lo, hi) -- dataset.py:527-529."""
        a, b = np.searchsorted(profile_times[key], [lo, hi], side='left')
        return profiles[key].iloc[a:b]

    q_times = daily['t'].values
    f_times = f_sorted['period_end_t'].values

    # One anchor per trading session: the last bar of the day.
    grid = daily.groupby(daily['t'].dt.normalize())['t'].max().sort_values()

    rows = []
    for current_time in grid:
        ctx_lo = current_time - pd.DateOffset(months=context_depth_months)
        lo, hi = np.searchsorted(
            q_times, [ctx_lo.to_datetime64(), current_time.to_datetime64()], side='right'
        )
        context_slice = daily.iloc[lo:hi]
        if context_slice.empty:
            continue

        fund_lo = current_time - pd.DateOffset(months=fundamental_context_depth_months)
        f_lo, f_hi = np.searchsorted(
            f_times, [fund_lo.to_datetime64(), current_time.to_datetime64()], side='left'
        )
        if f_hi <= f_lo:
            continue

        lo_t, hi_t = fund_lo.to_datetime64(), current_time.to_datetime64()
        metrics = fundamental_f2(
            current_time,
            proxy_days,
            stale_days,
            context_slice,
            pit("bs", lo_t, hi_t),
            pit("cf", lo_t, hi_t),
            pit("inc", lo_t, hi_t),
            pit("div", lo_t, hi_t),
            pit("si", lo_t, hi_t),
        )
        if not metrics:
            continue

        # The hash of the bar actually being priced, so the FB era carries
        # Facebook's hash and the META era carries Meta's.
        metrics["stamp"] = current_time.value
        metrics["ticker_hash"] = int(context_slice['ticker_hash'].iloc[-1])
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# Populated in main() before the pool forks, so workers inherit the two large
# frames copy-on-write rather than receiving them through a pickle.
_SHARED = {}


# --- Process Initializer for Windows Compatibility ---
def _init_worker(shared_quotes, shared_fund, shared_policy):
    """Initializes global memory inside each spawned worker process on Windows."""
    global _SHARED
    _SHARED = {
        "quotes": shared_quotes,
        "fund": shared_fund,
        "policy": shared_policy,
    }

def _worker(figi):
    """Tasks are bare FIGI hashes."""
    policy = _SHARED["policy"]
    panel = reconstruct_pit_f2_series(
        _SHARED["quotes"].get(figi, pd.DataFrame()),
        _SHARED["fund"].get(figi, pd.DataFrame()),
        policy["context_depth_months"],
        policy["fundamental_context_depth_months"],
        policy["fundamental_proxy_filing_date_delta_days"],
        policy["fundamental_period_end_stale_delta_days"],
    )
    return figi, panel

def main():
    policy = json.load(open(POLICY_PATH))
    # universe = sorted(set(policy["in_set"] + policy["intermediate_set"]
    #                       + policy["hold_out"] + policy["benchmark_set"]))
    universe = sorted(["NVDA", "META", "FB"])

    db_fetch_start = pd.to_datetime(policy["fetch_start"]).to_pydatetime()
    db_fetch_end = datetime.today()

    AsyncF2.__table__.drop(engine, checkfirst=True)
    AsyncF2.__table__.create(engine)
    tqdm.write(f"[*] Rebuilt async_f2. Universe: {len(universe)} tickers.")

    # Identical intake to pricing/demo/async_short_interest.py: _fetch_clean_universe
    # returns quotes and fundamentals already carrying figi_group and already pruned
    # to each ticker_hash's earliest/latest lifespan. That pruning is what keeps FB
    # and META one entity while shutting out the recycled FB symbol.
    global_identity = _prepare_global_identity(
        universe, pd.Timestamp(db_fetch_end),
        policy["reconstruction_heartbeat_freq_months"],
    )
    full_quotes_df, full_fundamentals_df = _fetch_clean_universe(
        global_identity,
        pd.to_datetime(db_fetch_start),
        pd.Timestamp(db_fetch_end),
        policy["context_depth_months"],
        policy["forecast_depth_months"],
        policy["fundamental_context_depth_months"],
        policy["fundamental_proxy_filing_date_delta_days"],
        policy["reconstruction_heartbeat_freq_months"],
    )

    # A ticker_hash can sit in several identity rows with overlapping lifespans
    # (XOM carries three), so the bounds join fans the same bar out more than once.
    full_quotes_df = full_quotes_df.drop_duplicates(
        subset=['ticker_hash', 't', 'adjusted'], keep='last'
    )
    full_fundamentals_df['period_end_t'] = (
        pd.to_datetime(full_fundamentals_df['period_end_t']).astype('datetime64[ns]')
    )

    # One row per corporate entity. Sorting by 'latest' before dropping duplicates
    # keeps the ticker current at the end of the window, so the FB/META entity is
    # labelled META rather than the retired FB.
    entities = (global_identity.sort_values("latest")
                               .drop_duplicates(subset=["composite_figi_hash"], keep="last"))
    tqdm.write(f"[*] {len(entities)} corporate entities from "
               f"{global_identity['ticker_hash'].nunique()} ticker hashes.")

    _SHARED["quotes"] = {k: v for k, v in full_quotes_df.groupby('figi_group')}
    _SHARED["fund"] = {k: v for k, v in full_fundamentals_df.groupby('figi_group')}
    _SHARED["policy"] = policy
    ticker_by_hash = dict(zip(global_identity['ticker_hash'], global_identity['ticker']))

    # f2 spends its time in extract_latest_valid_row's per-row .apply, once per
    # profile per session, so a single pass costs ~55s per entity. Entities are
    # independent, so they fan out the same way anchor_premium_sweep.py:477 does.
    # 'fork' lets the workers read the two big frames out of inherited memory
    # instead of re-fetching or pickling them; the engine is disposed first and
    # workers stay pure-compute, so no DB handle is ever carried across the fork.
    tasks = [int(h) for h in entities["composite_figi_hash"]]
    engine.dispose()

    written = 0
    pbar = tqdm(total=len(tasks), desc="Evaluating f2", unit="entity")

    num_workers = min(10, os.cpu_count() or 1)
    with get_context("spawn").Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(_SHARED["quotes"], _SHARED["fund"], policy)
    ) as pool:
        for figi, panel in pool.imap_unordered(_worker, tasks):
   
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

    tqdm.write(f"[*] async_f2 holds {total:,} rows across {covered} corporate entities.")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
