import os
import sys
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import get_context
from sqlalchemy import Column, BigInteger, String, Numeric, SmallInteger, Index, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine, Base

POLICY_PATH = os.path.join(PROJECT_ROOT, "pricing", "simulation_policy_big.json")


class AsyncSiFlag(Base):
    __tablename__ = "async_si_flag"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stamp = Column(BigInteger, index=True, nullable=False)
    ticker = Column(String(20))
    ticker_hash = Column(BigInteger, index=True, nullable=False)
    figi_group = Column(BigInteger, index=True)

    pct_short_of_market_cap = Column(Numeric(18, 8))

    # 1 while the asset sits inside the quarantine cage -- this is the value
    # pricing/dataset.py consumes as si_flag.
    si_flag = Column(SmallInteger, index=True)
    # 1 only on the bars that actually breached a gate. The cage is raw_trigger
    # held open for quarantine_delta_months, so raw_trigger <= si_flag always.
    raw_trigger = Column(SmallInteger)

    __table_args__ = (
        Index("idx_async_si_flag_hash_stamp", "ticker_hash", "stamp"),
    )


def si_regime_filter(q_unadj, si_filter_shock_sigma=4.0, si_filter_vol_saturation_sigma=4.0,
                     si_filter_gated_si_drift=1.4, si_filter_min_sid=0.05, si_filter_max_sid=0.5,
                     si_filter_short_offset_days=60,
                     si_filter_long_offset_days=365,
                     si_filter_drift_offset_days=126,
                     quarantine_delta_months=3):
    """pricing/math/filter.py si_regime_filter, lines 208-331.

    Copied verbatim through STEP 1-4. pricing/math/si_regime_filter.py holds a
    byte-identical copy; filter.py is the one pricing/dataset.py:29 actually
    imports, so that is the one taken here.

    One thing changes, at the very end. The original throws the per-bar result
    away: it computes quarantine_state for every bar, leaves
    `df_time['regime_eviction_flag'] = quarantine_state.astype(int)` commented
    out, and instead overwrites a single composite diagnostic float
    ([status][gates].[trigger count]) onto last_idx, because dataset.py:250 only
    ever reads `.iloc[-1]`. This table needs the flag on every bar, so that
    commented line is restored and the last-row composite is dropped. The gate
    arithmetic above it is untouched.
    """
    if q_unadj.empty or 'si_to_cap_ratio' not in q_unadj.columns:
        return q_unadj

    original_index = q_unadj.index
    df_time = q_unadj.sort_values(by='t').set_index('t')
    sid = df_time['si_to_cap_ratio']

    short_offset = f"{si_filter_short_offset_days}D"
    long_offset = f"{si_filter_long_offset_days}D"

    # --- STEP 1: Compute Macro Horisons (Quarter vs Full Trading Year) ---
    mean_long = sid.rolling(window=long_offset, min_periods=60).mean()
    std_long = sid.rolling(window=long_offset, min_periods=60).std()
    var_long = sid.rolling(window=long_offset, min_periods=60).var()

    mean_short = sid.rolling(window=short_offset, min_periods=20).mean()
    var_short = sid.rolling(window=short_offset, min_periods=20).var()

    # --- STEP 2: Asymmetric Logic Gates ---
    df_time['f1_shock'] = sid > (mean_long + si_filter_shock_sigma * std_long)

    df_time['vol_ratio'] = var_short / var_long.replace(0, np.nan)
    df_time['f2_vol_saturation'] = df_time['vol_ratio'] > si_filter_vol_saturation_sigma

    # Unidirectional: a short collapse drives this negative and cannot breach.
    df_time['mean_drift_sigma_distance'] = (mean_short - mean_long) / std_long.replace(0, np.nan).fillna(1e-6)
    df_time['f3_mean_drift'] = df_time['mean_drift_sigma_distance'] > si_filter_gated_si_drift

    df_time['macro_trend_velocity'] = mean_long.diff(periods=si_filter_drift_offset_days)
    df_time['f5_structural_growth'] = df_time['macro_trend_velocity'] > 0

    df_time['f4_above_floor'] = sid > si_filter_min_sid
    df_time['si_filter_max_siding_breach'] = sid > si_filter_max_sid

    # --- STEP 3: Raw Signal Synthesis ---
    is_upward_regime = mean_short >= mean_long

    df_time['raw_trigger'] = (
        (df_time['f1_shock'] & is_upward_regime) |
        (df_time['f2_vol_saturation'] & is_upward_regime) |
        (df_time['f3_mean_drift'] & df_time['f5_structural_growth']) |
        df_time['si_filter_max_siding_breach']
    ) & df_time['f4_above_floor']

    # --- STEP 4: MANDATORY COOLDOWN HYSTERESIS LOCK ---
    timestamps = df_time.index.to_numpy()
    raw_signals = df_time['raw_trigger'].fillna(False).to_numpy()
    quarantine_state = np.zeros(len(raw_signals), dtype=bool)

    active_quarantine_deadline = pd.Timestamp.min

    for i in range(len(raw_signals)):
        current_time = pd.Timestamp(timestamps[i])

        if raw_signals[i]:
            target_deadline = current_time + pd.DateOffset(months=quarantine_delta_months)
            if target_deadline > active_quarantine_deadline:
                active_quarantine_deadline = target_deadline

        if current_time <= active_quarantine_deadline:
            quarantine_state[i] = True
        else:
            quarantine_state[i] = False

    df_time['regime_eviction_flag'] = quarantine_state.astype(int)

    df_res = df_time.reset_index()
    df_res.index = original_index
    return df_res


_SHARED = {}


def _worker(figi):
    panel = _SHARED["si"].get(figi)
    if panel is None or panel.empty:
        return figi, pd.DataFrame()

    p = _SHARED["policy"]
    q_unadj = pd.DataFrame({
        "t": pd.to_datetime(panel["stamp"]),
        # async_short_interest stores the ratio as a percentage; the filter's
        # floor/ceiling gates (si_filter_min_sid, si_filter_max_sid) are defined
        # on the raw ratio, so it goes back to fractional here.
        "si_to_cap_ratio": panel["pct"].astype(float) / 100.0,
    })

    res = si_regime_filter(
        q_unadj,
        si_filter_shock_sigma=p["si_filter_shock_sigma"],
        si_filter_vol_saturation_sigma=p["si_filter_vol_saturation_sigma"],
        si_filter_gated_si_drift=p["si_filter_gated_si_drift"],
        si_filter_min_sid=p["si_filter_min_sid"],
        si_filter_max_sid=p["si_filter_max_sid"],
        si_filter_short_offset_days=p["si_filter_short_offset_days"],
        si_filter_long_offset_days=p["si_filter_long_offset_days"],
        si_filter_drift_offset_days=p["si_filter_drift_offset_days"],
        quarantine_delta_months=p["quarantine_delta_months"],
    )
    if res.empty:
        return figi, pd.DataFrame()

    out = pd.DataFrame({
        "stamp": res["t"].astype("datetime64[ns]").astype("int64"),
        "ticker": panel["ticker"].to_numpy(),
        "ticker_hash": panel["ticker_hash"].to_numpy(),
        "figi_group": figi,
        "pct_short_of_market_cap": panel["pct"].to_numpy(),
        "si_flag": res["regime_eviction_flag"].astype(int).to_numpy(),
        "raw_trigger": res["raw_trigger"].fillna(False).astype(int).to_numpy(),
    })
    return figi, out


def main():
    policy = json.load(open(POLICY_PATH))

    AsyncSiFlag.__table__.drop(engine, checkfirst=True)
    AsyncSiFlag.__table__.create(engine)
    tqdm.write("[*] Rebuilt async_si_flag.")

    # The filter only needs (t, si_to_cap_ratio), and async_short_interest already
    # holds exactly that -- point-in-time reconstructed, lifespan-pruned, FB/META
    # bridged. Reading it back is the whole intake, so no quotes or fundamentals
    # are re-fetched here.
    tqdm.write("[*] Reading async_short_interest...")
    si = pd.read_sql(text(
        "SELECT figi_group, ticker, ticker_hash, stamp, pct_short_of_market_cap AS pct "
        "FROM async_short_interest ORDER BY figi_group, stamp"
    ), engine)
    tqdm.write(f"[*] {len(si):,} bars across {si['figi_group'].nunique()} corporate entities.")

    _SHARED["si"] = {int(k): v for k, v in si.groupby("figi_group")}
    _SHARED["policy"] = policy
    tasks = sorted(_SHARED["si"].keys())
    del si
    engine.dispose()

    written = 0
    flagged = 0
    pbar = tqdm(total=len(tasks), desc="Applying si_regime_filter", unit="entity")
    with get_context("fork").Pool(processes=min(10, os.cpu_count() or 1)) as pool:
        for figi, out in pool.imap_unordered(_worker, tasks):
            if not out.empty:
                out.to_sql("async_si_flag", engine, if_exists="append", index=False,
                           method="multi", chunksize=5000)
                written += len(out)
                flagged += int(out["si_flag"].sum())
            pbar.set_postfix(rows=written, flagged=flagged)
            pbar.update(1)
    pbar.close()

    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT count(*), count(DISTINCT figi_group), sum(si_flag), sum(raw_trigger),
                   count(DISTINCT figi_group) FILTER (WHERE si_flag = 1)
            FROM async_si_flag
        """)).fetchone()

    tqdm.write(f"[*] async_si_flag holds {stats[0]:,} rows across {stats[1]} corporate entities.")
    tqdm.write(f"[*] si_flag=1 on {stats[2]:,} bars ({100.0 * stats[2] / stats[0]:.2f}%), "
               f"raw_trigger=1 on {stats[3]:,} bars.")
    tqdm.write(f"[*] {stats[4]} of {stats[1]} entities were caught by the filter at least once.")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
