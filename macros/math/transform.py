import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from millify import millify
from typing import List, Dict, Tuple, Any, Optional, Union
import os, sys , json, re, csv





   
def transxf_fred(x: pd.Series, tcode: int, code9_trillion_denom:Optional[float]=1) -> pd.Series:
    """Forward transform a single series using standard FRED-MD codes (1-7)."""
    s = x.astype(float).copy()
    small = 1e-6

    if tcode == 1:
        # Level: x(t)
        return s

    elif tcode == 2:
        # First difference: x(t) - x(t-1)
        return s.diff()

    elif tcode == 3:
        # Second difference: (x(t) - x(t-1)) - (x(t-1) - x(t-2))
        return s.diff().diff()

    elif tcode == 4:
        # Natural log: ln(x)
        return (
            np.log(s) if s.min() > small else pd.Series(np.nan, index=s.index)
        )

    elif tcode == 5:
        # First difference of natural log: ln(x) - ln(x-1)
        return (
            np.log(s).diff()
            if s.min() > small
            else pd.Series(np.nan, index=s.index)
        )

    elif tcode == 6:
        # Second difference of natural log
        return (
            np.log(s).diff().diff()
            if s.min() > small
            else pd.Series(np.nan, index=s.index)
        )

    elif tcode == 7:
        # First difference of percent change
        pct_change = s.pct_change()
        return pct_change.diff()

    elif tcode == 8:
        # First difference scaled to fractional rate: (x(t) - x(t-1)) / 100.0
        return s.diff() / 100.0
    elif tcode == 9:
        # Edge Case Safeguard: Prevent division by zero/None
        denom = (
            code9_trillion_denom
            if code9_trillion_denom and code9_trillion_denom > 0
            else 1.0
        )
        return s.diff() / denom
    elif tcode == 10:
        # Fractional Level: x(t) / 100.0 (e.g., 5.25% -> 0.0525)
        return s / 100.0        
    else:
        raise ValueError(f"Invalid transformation code: {tcode}")


def reverse_transxf_fred(
    y: pd.Series,
    tcode: int,
    init_val1: float = None,
    init_val2: float = None,
    code9_trillion_denom: Optional[float] = 1.0
) -> pd.Series:
    """Reverse transform a single series for standard FRED-MD codes (1-7)

    with exact index-aligned reconstruction.
    """
    s = y.copy()

    if tcode == 1:
        return s

    elif tcode == 2:
        # Fixed-grid reversal: seed + cumsum over complete input index
        if init_val1 is None:
            raise ValueError("init_val1 is required for tcode 2 reversal.")
        
        # Directly accumulate filled deltas without re-aggregating slices or masks
        return init_val1 + s.fillna(0.0).cumsum()

    elif tcode == 3:
        if init_val1 is None or init_val2 is None:
            raise ValueError(
                "init_val1 and init_val2 are required to invert tcode 3."
            )

        diff2 = s.fillna(0.0)
        first_diff = (init_val2 - init_val1) + diff2.cumsum()
        res = init_val1 + first_diff.cumsum()

        # Alignment offset fix to eliminate residual shifts
        offset = init_val2 - res.iloc[1] if len(res) > 1 else 0.0
        return res + offset

    elif tcode == 4:
        return np.exp(s)

    elif tcode == 5:
        if init_val1 is None:
            raise ValueError("init_val1 is required for tcode 5 reversal.")
        return init_val1 * np.exp(s.fillna(0.0).cumsum())

    elif tcode == 6:
        if init_val1 is None or init_val2 is None:
            raise ValueError(
                "init_val1 and init_val2 are required to invert tcode 6."
            )

        log_diff2 = s.fillna(0.0)
        init_log1, init_log2 = np.log(init_val1), np.log(init_val2)
        first_log_diff = (init_log2 - init_log1) + log_diff2.cumsum()

        log_level = init_log1 + first_log_diff.cumsum()
        offset = init_log2 - log_level.iloc[1] if len(log_level) > 1 else 0.0
        return np.exp(log_level + offset)

    elif tcode == 7:
        pct_diff = s.fillna(0.0)
        init_pct = (init_val2 - init_val1) / init_val1
        pct_change = init_pct + pct_diff.cumsum()
        return init_val2 * (1 + pct_change).cumprod()

    elif tcode == 8:
        if init_val1 is None:
            raise ValueError("init_val1 is required for tcode 8 reversal.")
        return init_val1 + (s.fillna(0.0) * 100.0).cumsum()

    elif tcode == 9:
        if init_val1 is None:
            raise ValueError("init_val1 is required for tcode 9 reversal.")
        denom = (
            code9_trillion_denom
            if code9_trillion_denom and code9_trillion_denom > 0
            else 1.0
        )
        return init_val1 + (s.fillna(0.0) * denom).cumsum()

    elif tcode == 10:
        return s * 100.0
    else:
        raise ValueError(f"Invalid transformation code: {tcode}")








def compute_precomputed_rev_vars(
    train_obs_fit: List[dict],
    column_order: List[str],
    transform_metadata: Dict[str, Dict],
    metadata_lookup: Dict[str, Dict],
    resample_frequency: str = "1ME",
    config: Dict ={}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes historical revision error variance (sigma^2_rev) and 
    publication latency stats (mean & std) for each series in column_order.
    
    Returns:
        rev_vars (np.ndarray): Scaled revision variances per series.
        min_latency_mean (np.ndarray): Mean publication lag per series in frequency units.
        min_latency_std (np.ndarray): Std dev of publication lag per series in frequency units.
    """
    n_cols = len(column_order)
    df_raw = pd.DataFrame(train_obs_fit)
    if df_raw.empty:
        return np.zeros(n_cols), np.zeros(n_cols), np.zeros(n_cols)

    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["realtime_start"] = pd.to_datetime(
        df_raw.get("realtime_start", df_raw["date"])
    )
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_raw = df_raw.dropna(subset=["value"])

    # -------------------------------------------------------------------------
    # 1. LATENCY COMPUTATION (min realtime_start - observation date per date)
    # -------------------------------------------------------------------------
    latency_group = df_raw.groupby(["series_id", "date"])["realtime_start"].min().reset_index()
    
    latency_group["latency_days"] = (latency_group["realtime_start"] - latency_group["date"]).dt.days
    latency_group["latency_days"] = np.maximum(latency_group["latency_days"], 0)

    freq_days_map = {"1MS": 30.4375, "MS": 30.4375,"1ME": 30.4375,  "M": 30.4375, "1W": 7.0, "W": 7.0, "1D": 1.0, "D": 1.0}
    days_per_unit = freq_days_map.get(resample_frequency, 30.4375)
    latency_group["min_latency"] = latency_group["latency_days"] / days_per_unit

    latency_stats = latency_group.groupby("series_id")["min_latency"].agg(
        mean="mean", 
        std=lambda x: x.std(ddof=1) if len(x) > 1 else 0.0
    ).to_dict(orient="index")

    # -------------------------------------------------------------------------
    # 2. REVISION VARIANCE COMPUTATION
    # -------------------------------------------------------------------------
    df_sorted = df_raw.sort_values(by=["series_id", "date", "realtime_start"])

    df_first = df_sorted.drop_duplicates(subset=["series_id", "date"], keep="first")
    df_final = df_sorted.drop_duplicates(subset=["series_id", "date"], keep="last")

    def process_vintage_to_stationary(df_v: pd.DataFrame) -> pd.DataFrame:
        if df_v.empty:
            return pd.DataFrame()

        df_v = df_v.copy()
        df_v["resampled_date"] = (
            df_v["date"].dt.to_period("M").dt.to_timestamp(how="end")
        )


        series_frames = []
        for sid, group in df_v.groupby("series_id"):
            resampled = (
                group.set_index("date")["value"]
                .resample(resample_frequency)
                .last()
            )

            resampled.name = str(sid)
            series_frames.append(resampled)

        if not series_frames:
            return pd.DataFrame()

        df_pivot = pd.concat(series_frames, axis=1)
        df_stat = pd.DataFrame(index=df_pivot.index)

        for col in df_pivot.columns:
            col_str = str(col)
            tf_meta = transform_metadata.get(col_str, {})
            code = tf_meta.get("code", 1)
            denom = tf_meta.get("code9_trillion_denom", 1.0)

            raw_levels = df_pivot[col]
            native_levels = raw_levels.dropna()

        
            native_stat_s = transxf_fred(native_levels, code, denom)
            df_stat[col_str] = native_stat_s

        return df_stat

    stat_first = process_vintage_to_stationary(df_first)
    stat_final = process_vintage_to_stationary(df_final)

    rev_vars = []
    lat_means = []
    lat_stds = []

    for col in column_order:
        col_str = str(col)
        
        # Pull series metadata to inspect reporting frequency
        meta = metadata_lookup.get(col_str, {})
        freq_short = meta.get("frequency_short")
        cap_months = config["fred_frequency_short_max_staleness_months"].get(freq_short, 3.0)

        # Pull raw latency stats
        l_stat = latency_stats.get(col_str, {"mean": 0.0, "std": 0.0})
        raw_mean = float(l_stat["mean"])
        raw_std = float(l_stat["std"])

        # Apply clipping guardrail if latency is artificially inflated by database backfilling
        if raw_mean > cap_months:
            clean_mean = cap_months
            clean_std = 0.0  # Zero out contaminated standard deviation from backfill jumps
        else:
            clean_mean = raw_mean
            clean_std = raw_std

        lat_means.append(clean_mean)
        lat_stds.append(clean_std)

        # Pull Revision Variance
        if col_str in stat_first.columns and col_str in stat_final.columns:
            s_first = stat_first[col_str]
            s_final = stat_final[col_str]

            aligned = pd.concat(
                [s_first, s_final], axis=1, keys=["first", "final"]
            ).dropna()

            
            revision_errors = aligned["final"] - aligned["first"]
            raw_var = float(revision_errors.var(ddof=1))
            
            # Compute true standard deviation directly from stationary first-vintage values
            series_std = float(aligned["first"].std(ddof=1))
            std_denom = (series_std ** 2) if series_std > 1e-6 else 1.0
            
            scaled_var = raw_var / std_denom if np.isfinite(raw_var) else 0.0
            
            # Clip to prevent extreme ratio blowups on near-constant series
            rev_vars.append(float(np.clip(scaled_var, 0.0, 2.0)))
        
        else:
            rev_vars.append(0.0)

    # Sanitize final array bounds
    clean_rev_vars = np.maximum(np.nan_to_num(np.array(rev_vars, dtype=float), nan=0.0), 0.0)
    clean_lat_mean = np.nan_to_num(np.array(lat_means, dtype=float), nan=0.0)
    clean_lat_std = np.nan_to_num(np.array(lat_stds, dtype=float), nan=0.0)

    return clean_rev_vars, clean_lat_mean, clean_lat_std

