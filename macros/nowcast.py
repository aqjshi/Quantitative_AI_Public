import os
from typing import Dict, List, Optional, Tuple, Any
import sys 
import pandas as pd
import numpy as np
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macros.math.transform import transxf_fred

def transform_with_metadata(
    obs_context: List[Dict],
    train_start: pd.Timestamp,
    current_time: pd.Timestamp,
    transform_metadata: Dict[str, Dict],
    resample_frequency: str = "1ME"
) -> pd.DataFrame:
    df_raw = pd.DataFrame(obs_context)
    if df_raw.empty:
        return pd.DataFrame()

    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["realtime_start"] = pd.to_datetime(
        df_raw.get("realtime_start", df_raw["date"])
    )
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_raw = df_raw.dropna(subset=["value"])

    train_start = pd.to_datetime(train_start)
    current_time = pd.to_datetime(current_time)

    df_raw = df_raw[
        (df_raw["date"] >= train_start) & (df_raw["date"] <= current_time)
    ]

  
    df_raw["resampled_date"] = (
        df_raw["date"].dt.to_period("M").dt.to_timestamp(how="end")
    )


    # DEDUPLICATION: Keep latest realtime_start per series & resampled monthly bucket
    df_raw = df_raw.sort_values(
        by=["series_id", "resampled_date", "realtime_start"]
    ).drop_duplicates(subset=["series_id", "resampled_date"], keep="last")

    series_frames = []
    for sid, group in df_raw.groupby("series_id"):
        resampled = (
            group.set_index("resampled_date")["value"]
            .resample(resample_frequency)
            .last()
        )

        resampled.name = sid
        series_frames.append(resampled)

    df_pivot_levels = pd.concat(series_frames, axis=1)
    df_pivot_stat = pd.DataFrame(index=df_pivot_levels.index)

    for col in df_pivot_levels.columns:
        col_str = str(col)
        tf_meta = transform_metadata.get(col_str, {})
        code = tf_meta.get("code", 1)
        denom = tf_meta.get("code9_trillion_denom", 1.0)

        raw_levels = df_pivot_levels[col]
        native_levels = raw_levels.dropna()

      
        df_pivot_stat[col_str] = transxf_fred(native_levels, code, denom)

    return df_pivot_stat

def nowcast(
    mf_fit: dict,
    train_obs_orc: List[dict],
    transform_metadata: Dict[str, Dict],
    train_start: pd.Timestamp,
    current_time: pd.Timestamp,
    exog_params: Dict,
    config: Dict
) -> Dict:
    resample_freq = config.get('resample_frequency', '1ME')
    
    df_stationary = transform_with_metadata(
        obs_context=train_obs_orc,
        train_start=train_start,
        current_time=current_time,
        transform_metadata=transform_metadata,
        resample_frequency=resample_freq
    )
    
    wide = df_stationary.loc[df_stationary.index <= current_time].copy()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    cols = mf_fit["cols"]
    mean = mf_fit["mean"]
    std = mf_fit["std"]
    train_result = mf_fit["train_result"]

    wide_scaled = (wide[cols] - mean) / std
    last_obs_date = wide_scaled.index[-1]

    # 1. EXTEND PANEL WITH NaNs FOR OUT-OF-SAMPLE FORECAST HORIZON
    if current_time > last_obs_date:
        forecast_dates = pd.date_range(
            start=last_obs_date + pd.tseries.frequencies.to_offset(resample_freq),
            end=current_time,
            freq=resample_freq
        )
        if len(forecast_dates) > 0:
            nan_df = pd.DataFrame(np.nan, index=forecast_dates, columns=cols)
            wide_scaled = pd.concat([wide_scaled, nan_df])

    # -----------------------------------------------------------------
    # INTERNAL CAP: Give up if current_time extends past valid panel horizon
    # -----------------------------------------------------------------
    valid_rows = wide_scaled.dropna(how='all')
    max_real_obs_date = valid_rows.index[-1]
    max_allowed_horizon = max_real_obs_date + pd.tseries.frequencies.to_offset(resample_freq)

    # If current_time is beyond the 1-step forecast cap, output pure NaNs
    if current_time > max_allowed_horizon:
        nan_fitted = pd.DataFrame(np.nan, index=wide_scaled.index, columns=cols)
        return {
            "fitted_values": nan_fitted,
            "monthly_signal": nan_fitted,  
            "latent_state": pd.DataFrame(),
            "current_state": np.full(train_result.model.k_states, np.nan),
            "nowcast_latest": {col: np.nan for col in cols}
        }

    # 2. Dynamic Bitemporal Metrics Lookup
    raw_df = pd.DataFrame(train_obs_orc)
    release_lookup = {}
    
    # if not raw_df.empty and 'realtime_start' in raw_df.columns:
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df['realtime_start'] = pd.to_datetime(raw_df['realtime_start'])
    
    valid_df = raw_df[
        (raw_df['realtime_start'] <= current_time) & 
        (raw_df['date'] <= current_time)
    ]
    
    # if not valid_df.empty:
    aggs = valid_df.groupby(['series_id', 'date'])['realtime_start'].agg(
        t_first='min',
        t_latest='max'
    ).reset_index()
    
    release_lookup = {
        (str(r.series_id), r.date): (r.t_first, r.t_latest)
        for r in aggs.itertuples()
    }

    # 3. Construct 3D Time-Varying Covariance Matrix R_3d
    rev_vars = np.array(exog_params.get("revision_variances", np.zeros(len(cols))), dtype=float)

    nobs = len(wide_scaled)
    k_endog = len(cols)
    obs_dates = wide_scaled.index

    ssm_obs_cov = train_result.model.ssm['obs_cov']
    base_R_diag = np.maximum(ssm_obs_cov.diagonal().copy(), 1e-4)
    R_3d = np.zeros((k_endog, k_endog, nobs))

    for t_idx, t_obs in enumerate(obs_dates):
        for i_idx, col in enumerate(cols):
            col_str = str(col)
            releases = release_lookup.get((col_str, t_obs))
            
            if releases is not None:
                _, t_latest = releases
            else:
                # Fallback: Estimate release age assuming standard publication date = t_obs
                t_latest = t_obs

            # Calculate age in months relative to current_time
            age_days = (current_time - t_latest).days
            age_months = max(age_days / 30.4375, 0.0)
            
            # Fresh data = full revision variance penalty
            # Old data = decayed revision variance penalty (approaches 0 as age -> infinity)
            penalty = rev_vars[i_idx] + (age_months)

            R_3d[i_idx, i_idx, t_idx] = base_R_diag[i_idx] + penalty

    # 4. Clone Model and Run Kalman Smoother Across Extended Window
    point_in_time_model = train_result.model.clone(wide_scaled)
    point_in_time_model.ssm['obs_cov'] = R_3d

    updated_result = point_in_time_model.smooth(train_result.params)

    # 5. Extract Un-scaled Predictions (In-Sample + Extended Horizon)
    fitted_scaled = updated_result.fittedvalues
    fitted = (fitted_scaled * std) + mean

    current_state = updated_result.states.smoothed.iloc[-1].values
    nowcast_latest = fitted.iloc[-1].to_dict()

    return {
        "fitted_values": fitted,
        "monthly_signal": fitted,  
        "latent_state": updated_result.factors.smoothed,
        "current_state": current_state,
        "nowcast_latest": nowcast_latest
    }


