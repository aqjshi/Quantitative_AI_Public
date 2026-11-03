import os
import sys
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import Column, BigInteger, String, Numeric, Index, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine, Base
from pricing.database.identity import _prepare_global_identity
from pricing.database.fetcher import _fetch_clean_universe

POLICY_PATH = os.path.join(PROJECT_ROOT, "pricing", "simulation_policy_big.json")


class AsyncShortInterest(Base):
    __tablename__ = "async_short_interest"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stamp = Column(BigInteger, index=True, nullable=False)
    ticker = Column(String(20))
    ticker_hash = Column(BigInteger, index=True, nullable=False)
    figi_group = Column(BigInteger, index=True)

    adj_price = Column(Numeric(18, 4))
    adj_volume = Column(Numeric(24, 10))
    unadj_price = Column(Numeric(18, 4))
    unadj_volume = Column(Numeric(24, 10))

    market_cap = Column(Numeric(28, 4))
    basic_shares_outstanding = Column(Numeric(24, 4))
    pct_short_of_market_cap = Column(Numeric(18, 8))

    __table_args__ = (
        Index("idx_async_si_hash_stamp", "ticker_hash", "stamp"),
    )


def reconstruct_pit_short_interest_series(
    row: tuple,
    current_time: pd.Timestamp,
    fundamental_slice: pd.DataFrame,
    full_fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """pricing/dataset.py reconstruct_pit_short_interest_regime, lines 124-254.

    Copied verbatim through the alignment and ratio math. Two things are removed
    because this writes a table rather than feeding the filter:

      - si_regime_filter and the .iloc[-1] scalar extraction. The original returns
        one (ratio, flag, bypass) triple for current_time; here the whole aligned
        frame is returned so every 30-minute bar is persisted.
      - the si_filter_* and quarantine parameters, which only fed that filter.

    Everything that makes the reconstruction correct is kept as written: the
    figi_group masking that bridges FB into META, the 4-quarter maturity
    checkpoint, the p_adjusted forward-fill, share_adjustment_factor, and the
    merge_asof alignment of shares outstanding and short interest.
    """
    # 1. Defensively isolate quotes up to current_time
    hist_quotes_mask = (fundamental_slice['figi_group'] == row.composite_figi_hash) & (fundamental_slice['t'] <= current_time)
    asset_quotes_hist = fundamental_slice[hist_quotes_mask].copy()

    if asset_quotes_hist.empty:
        return pd.DataFrame()

    q_unadj_h = asset_quotes_hist[asset_quotes_hist['adjusted'] == False].sort_values('t')
    q_adj_h = asset_quotes_hist[asset_quotes_hist['adjusted'] == True].sort_values('t')

    f_hist = full_fundamentals_df[
        (full_fundamentals_df['figi_group'] == row.composite_figi_hash) &
        (full_fundamentals_df['period_end_t'] <= current_time)
    ].sort_values('period_end_t')

    if q_unadj_h.empty or f_hist.empty:
        return pd.DataFrame()

    # 4-QUARTER MATURITY CHECKPOINT: ENSURE HISTORICAL DATA DEPTH
    historical_quarters = f_hist[
        (f_hist['timeframe'] == 'quarterly') &
        (f_hist['basic_shares_outstanding'].notna())
    ]
    if historical_quarters['period_end_t'].nunique() < 4:
        return pd.DataFrame()

    q_unadj_h['p_nominal'] = q_unadj_h['c']

    adj_price_map_h = q_adj_h.set_index('t')['c'].to_dict()
    q_unadj_h['p_adjusted'] = q_unadj_h['t'].map(adj_price_map_h).ffill().fillna(q_unadj_h['p_nominal'])

    q_unadj_h['share_adjustment_factor'] = np.where(
        q_unadj_h['p_adjusted'] > 0, q_unadj_h['p_nominal'] / q_unadj_h['p_adjusted'], 1.0
    )

    # 3. Extract and Align point-in-time fundamentals
    f_sub_h = f_hist[(f_hist['timeframe'] == 'quarterly') | (f_hist['timeframe'].isna())].copy()
    f_sub_h['basic_shares_outstanding'] = f_sub_h['basic_shares_outstanding'].bfill().ffill()

    si_source_h = f_hist[f_hist['short_interest'].notna()].copy()
    if si_source_h.empty:
        return pd.DataFrame()

    q_date_map_h = q_unadj_h[['t', 'share_adjustment_factor']].copy()
    # .dt.date yields date objects, which pandas 3 reads back as datetime64[s]
    # while period_end_t is datetime64[ns]. merge_asof rejects mixed resolutions,
    # so both join keys are pinned. dataset.py:199 has the same issue unpinned.
    q_date_map_h['normalized_date'] = pd.to_datetime(q_date_map_h['t'].dt.date).astype('datetime64[ns]')
    q_date_map_h = q_date_map_h.drop_duplicates(subset=['normalized_date'], keep='last')

    si_source_h['period_end_t'] = pd.to_datetime(si_source_h['period_end_t']).astype('datetime64[ns]')
    si_source_h = pd.merge_asof(
        si_source_h,
        q_date_map_h[['normalized_date', 'share_adjustment_factor']].rename(
            columns={'share_adjustment_factor': 'si_native_factor', 'normalized_date': 't_target'}
        ),
        left_on='period_end_t', right_on='t_target', direction='nearest'
    )

    q_aligned_h = pd.merge_asof(
        q_unadj_h, f_sub_h[['period_end_t', 'basic_shares_outstanding']].rename(columns={'period_end_t': 'available_t'}),
        left_on='t', right_on='available_t', direction='backward'
    )
    q_aligned_h = pd.merge_asof(
        q_aligned_h, si_source_h[['period_end_t', 'short_interest', 'si_native_factor']].rename(columns={'period_end_t': 'available_t'}),
        left_on='t', right_on='available_t', direction='backward'
    )

    if 'si_native_factor' in q_aligned_h.columns:
        q_aligned_h['share_adjustment_factor'] = q_aligned_h['si_native_factor'].fillna(q_aligned_h['share_adjustment_factor'])

    q_aligned_h['share_adjustment_factor'] = q_aligned_h['share_adjustment_factor'].fillna(1.0)
    q_aligned_h['short_interest'] = q_aligned_h['short_interest'].fillna(0.0)
    q_aligned_h['basic_shares_outstanding'] = q_aligned_h['basic_shares_outstanding'].ffill().bfill().fillna(1e-6)

    q_aligned_h['adjusted_short_interest_shares'] = q_aligned_h['short_interest'] * q_aligned_h['share_adjustment_factor']
    q_aligned_h['market_cap'] = q_aligned_h['p_adjusted'] * q_aligned_h['basic_shares_outstanding']
    q_aligned_h['short_interest_market_cap'] = q_aligned_h['adjusted_short_interest_shares'] * q_aligned_h['p_adjusted']

    q_aligned_h['si_to_cap_ratio'] = q_aligned_h['short_interest_market_cap'] / q_aligned_h['market_cap'].replace(0, np.nan)
    q_aligned_h['si_to_cap_ratio'] = q_aligned_h['si_to_cap_ratio'].fillna(0.0)

    # The original quarantines the asset when padding produced NaNs. Here the
    # affected bars are dropped instead, so one bad bar does not discard a
    # ticker's whole history.
    q_aligned_h = q_aligned_h[q_aligned_h['si_to_cap_ratio'].notna()]

    adj_vol_map_h = q_adj_h.set_index('t')['v'].to_dict()
    q_aligned_h['adj_volume'] = q_aligned_h['t'].map(adj_vol_map_h)

    return q_aligned_h


def main():
    policy = json.load(open(POLICY_PATH))
    # universe = sorted(set(policy["in_set"] + policy["intermediate_set"]
    #                       + policy["hold_out"] + policy["benchmark_set"]))
    universe = sorted(["NVDA", "META", "FB"])

    db_fetch_start = pd.to_datetime(policy["fetch_start"]).to_pydatetime()
    db_fetch_end = datetime.today()
    current_time = pd.to_datetime(db_fetch_end)

    AsyncShortInterest.__table__.drop(engine, checkfirst=True)
    AsyncShortInterest.__table__.create(engine)
    tqdm.write(f"[*] Rebuilt async_short_interest. Universe: {len(universe)} tickers.")

    # _fetch_clean_universe returns quotes and fundamentals together, both already
    # carrying figi_group and both already pruned to each ticker_hash's own
    # earliest/latest lifespan. That pruning is what separates FB from META: the
    # FB symbol was recycled, so short_interest rows exist under Facebook's hash
    # four years past the 2022 rename. Unpruned they interleave with META's at the
    # same period_end_t and merge_asof takes the wrong one -- 532 shares instead
    # of 31.5 million. Pruned, FB stops at 2022-05-31 and META starts 2022-06-15.
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

    # A ticker_hash can appear in several identity rows -- XOM carries three
    # composite_figi_hash values with overlapping lifespans -- so the bounds join
    # inside _fetch_clean_universe fans out and emits the same bar more than once.
    # The copies are byte-identical, and (ticker_hash, t, adjusted) is the natural
    # key of the quotes table, so collapsing on it restores one row per bar.
    full_quotes_df = full_quotes_df.drop_duplicates(
        subset=['ticker_hash', 't', 'adjusted'], keep='last'
    )

    # merge_asof needs both join keys at one resolution. Quotes arrive as
    # datetime64[ns]; period_end_t comes back as datetime64[s] because pandas 3
    # infers the unit from the source date objects.
    full_fundamentals_df['period_end_t'] = (
        pd.to_datetime(full_fundamentals_df['period_end_t']).astype('datetime64[ns]')
    )

    # One row per corporate entity, not per ticker: FB and META share a FIGI group
    # and must be reconstructed once, together.
    # One row per corporate entity. Sorting by 'latest' before dropping duplicates
    # keeps the ticker that is current at the end of the window, so the FB/META
    # entity is labelled META rather than the retired FB.
    entities = (global_identity.sort_values("latest")
                               .drop_duplicates(subset=["composite_figi_hash"], keep="last"))
    tqdm.write(f"[*] {len(entities)} corporate entities from "
               f"{global_identity['ticker_hash'].nunique()} ticker hashes.")

    written = 0
    pbar = tqdm(total=len(entities), desc="Reconstructing short interest", unit="entity")
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
            out.to_sql("async_short_interest", engine, if_exists="append", index=False, method="multi", chunksize=5000)
            written += len(out)
        pbar.set_postfix(rows=written)
        pbar.update(1)
    pbar.close()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM async_short_interest")).scalar()
        covered = conn.execute(text("SELECT count(DISTINCT figi_group) FROM async_short_interest")).scalar()
        stats = conn.execute(text("""
            SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY pct_short_of_market_cap)::numeric, 4),
                   round(percentile_cont(0.99) WITHIN GROUP (ORDER BY pct_short_of_market_cap)::numeric, 4),
                   round(max(pct_short_of_market_cap), 4)
            FROM async_short_interest
        """)).fetchone()

    tqdm.write(f"[*] async_short_interest holds {total:,} rows across {covered} corporate entities.")
    tqdm.write(f"[*] pct_short_of_market_cap  p50={stats[0]}%  p99={stats[1]}%  max={stats[2]}%")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
