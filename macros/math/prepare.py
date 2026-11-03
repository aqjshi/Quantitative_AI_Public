import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from millify import millify
from typing import List, Dict, Tuple, Any, Optional, Union



def fmt_val(val: float) -> str:
    """Helper to format values with millify rounded to 3 decimal places, handling NaNs & Inf safely."""
    if pd.isna(val) or val is None or np.isinf(val):
        return "NaN" if pd.isna(val) or val is None else ("Inf" if val > 0 else "-Inf")
    return str(millify(val, precision=3))



def filter_active_universe(
    train_obs_context: List[dict],
    metadata_lookup: dict,
    current_start: pd.Timestamp,
    config: dict
) -> Tuple[List[dict], List[str]]:
    """
    Iterates through observation contexts and prunes out low-variance variables,
    columns failing CV limits, immature series, stale/dead series, OR series whose 
    earliest realtime_start vintage date is after current_start (backfilled series).
    Prints formatted telemetry with strict column allocations and millified stats.
    """
    if not train_obs_context:
        return train_obs_context, []

    df_raw = pd.DataFrame(train_obs_context)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['realtime_start'] = pd.to_datetime(df_raw['realtime_start'])

    # Deduplicate bitemporal observations
    sort_cols = ['series_id', 'date', 'realtime_start']
    df_raw = df_raw.sort_values(by=sort_cols).drop_duplicates(subset=['series_id', 'date'], keep='last')

    # Resample grid to evaluation frequency
    resample_freq = config.get("resample_frequency", "1W")
    df_pivot = df_raw.pivot(index='date', columns='series_id', values='value')
    
    # Calculate un-filled missing observations prior to forward-filling
    unfilled_resampled = df_pivot.resample(resample_freq).last()
    df_resampled = unfilled_resampled.ffill()

   
    max_stale_cfg = config.get("fred_frequency_short_max_staleness_months", {"D": 3, "W": 3, "M": 3, "Q": 9, "A": 24})

    audit_rows = []
    active_sids = []
    for col_sid in df_resampled.columns:
      
        
        # Unfilled series to count missing observations
        raw_series = unfilled_resampled[col_sid]
        total_obs_count = int(raw_series.notna().sum())
        missing_obs_count = int(raw_series.isna().sum())

        series_vals = df_resampled[col_sid].dropna().values
        
    
        diff_vals = np.diff(series_vals)
        v = np.var(diff_vals) if len(diff_vals) > 0 else np.var(series_vals)
        mean_val = float(np.mean(series_vals))
        std_val = float(np.std(series_vals))

        u_count = len(np.unique(series_vals))
   

        meta = (metadata_lookup.get(col_sid)or {})
        real_name = meta.get('series_id', meta.get('name', meta.get('title', 'UNKNOWN NAME')))
        freq = str(meta.get('frequency_short', 'M')).upper().strip()
        units = str(meta.get('units_short', 'UNK')).strip()

    
        req_max_stale = max_stale_cfg.get(freq, 3)

        series_records = df_raw[df_raw['series_id'] == col_sid]
        
    
        min_d = series_records['date'].min()
        max_d = series_records['date'].max()
        min_realtime = series_records['realtime_start'].min()


        req_min_months = (config["train_start_back_months"] - config["eval_start_back_months"]) * config["req_min_coverage_ratio"]
        obs_months = (max_d - min_d).days / 30.4375
        
        # Dynamic calendar month float calculation
        staleness_months = float((current_start - max_d).days / 30.4375)
        staleness_months = max(0.0, staleness_months)
    

        reasons = []
        
        # Bitemporal Availability Gate
        if min_realtime > current_start:
            reasons.append(f"unobservable ({min_realtime.strftime('%Y-%m-%d')} > {current_start.strftime('%Y-%m-%d')})")

        if not (u_count >= 5): reasons.append(f"unique ({u_count} < 5)")
        if np.isnan(v): reasons.append("var is NaN")
        
        # Maturity and Expiry/Staleness gates
        if not (obs_months >= req_min_months): 
            reasons.append(f"maturity ({obs_months:.2f} mo < {req_min_months} mo)")
        if not (staleness_months <= req_max_stale): 
            reasons.append(f"stale ({staleness_months:.1f} mo > {req_max_stale} mo)")

        is_kicked = 1 if len(reasons) > 0 else 0

        if is_kicked == 0:
            active_sids.append(col_sid)

        audit_rows.append({
            'series_id': real_name,
            'freq': freq,
            'units': units,
            'last_obs': max_d.strftime('%Y-%m-%d') if pd.notna(max_d) else 'N/A',
            'stale_mo': staleness_months,
            'max_stale': float(req_max_stale),
            'obs': total_obs_count,
            'mis_obs': missing_obs_count,
            'mean': fmt_val(mean_val),
            'std': fmt_val(std_val),
            'kicked': is_kicked,
            'reason': ", ".join(reasons) if is_kicked else 'PASSED'
        })

    # Header and Formatting Layout
    print(f" UNIVERSE EXPIRY & STALENESS AUDIT MATRIX (ANCHOR: {current_start.strftime('%Y-%m-%d')})")

    header_str = (
        f"{'SERIES ID':<20} "
        f"{'FREQ':<5} "
        f"{'UNIT':<8} "
        f"{'LAST OBS':<10} "
        f"{'STALE MO':<10} "
        f"{'MAX STALE':<10} "
        f"{'OBS':<8} "
        f"{'MIS OBS':<8} "
        f"{'MEAN':<12} "
        f"{'STD':<12} "
        f"{'KICKED':<6} "
        f"{'REASON'}"
    )
    print(header_str)

    for r in audit_rows:
        row_str = (
            f"{r['series_id'][:20]:<20} "
            f"{r['freq'][:5]:<5} "
            f"{r['units'][:8]:<8} "
            f"{r['last_obs']:<10} "
            f"{r['stale_mo']:<10.2f} "
            f"{r['max_stale']:<10.1f} "
            f"{r['obs']:<8d} "
            f"{r['mis_obs']:<8d} "
            f"{r['mean']:<12} "
            f"{r['std']:<12} "
            f"{r['kicked']:<6d} "
            f"{r['reason']}"
        )
        print(row_str)

    print(f" TOTAL EVALUATED: {len(audit_rows)} | KEPT: {len(active_sids)} | KICKED: {len(audit_rows) - len(active_sids)}\n")

    active_sid_set = {str(sid) for sid in active_sids}
    
    filtered_obs_context = [ 
        row for row in train_obs_context 
        if str(row.get('series_id', row.get('series_id_hash'))) in active_sid_set
    ]
    
    return filtered_obs_context, active_sids