import os
from typing import Dict, List, Optional, Tuple, Any
import sys 
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import timedelta,  datetime
import numpy as np
import os
from tqdm import tqdm
import re


from datetime import datetime
import cProfile
import pstats
import json

from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, FLOAT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import FRED_KEY_0, engine, DATABASE_URL, Base
from macros.config import load_configuration, initialize_run_environment

from macros.database.database import fetch_fold_data
from macros.math.prepare import filter_active_universe
from macros.math.transform import  reverse_transxf_fred, compute_precomputed_rev_vars
from macros.ADF import classify_ADF
from macros.math.kalman import fit_mixed_frequency_kalman, parse_model_diagnostics
from macros.nowcast import nowcast
from macros.display.residuals import analyze_global_backtest
from macros.validation import evaluate_and_log_fold
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class MacrosNowcastMetadata(Base):
    __tablename__ = "macros_nowcast_metadata"
    run_hash                                    = Column(String, primary_key=True)
    train_start                                 = Column(Date)
    eval_start                                  = Column(Date)
    eval_end                                    = Column(Date)
    current_start                               = Column(Date, primary_key=True, index=True)
    resample_frequency 	                    = Column(String)
    dynamic_factor_n_factors 	            = Column(Integer)
    dynamic_factor_lag_order 	            = Column(Integer)

    train_start_back_months 	            = Column(Integer)
    n_observations                          = Column(Integer)

    dynamic_factor_statespace_maxiter 	    = Column(Integer)
    em_iterations                           = Column(Integer)

  
    factor_rho                              = Column(ARRAY(Float))
    log_likelihood                          = Column(Float)
    aic                                     = Column(Float)
    bic                                     = Column(Float)
    hqic                                    = Column(Float)
  
    realtime_start                              = Column(DateTime, default=datetime.now)
    series_ids                              = Column(ARRAY(String))  # Fixed typed ARRAY
    kalman_parameters                       = Column(ARRAY(Float))  # Fixed typed ARRAY



# must assume trained and evaled, otherwise no point in using with relative error
class MacrosNowcastDiagnostic(Base):
    __tablename__ = "macros_nowcast_diagnostic"

    run_hash            = Column(String, primary_key=True)
    train_start         = Column(Date)
    eval_start          = Column(Date)  # Added primary key flag
    eval_end            = Column(Date)
    current_start       = Column(Date, primary_key=True, index=True)
    series_id           = Column(String, primary_key=True, index=True)
    
    eval_nsamples       = Column(Integer)
    signal_share        = Column(Float)
    pit_km_rmse             = Column(Float)
    pit_base_rmse           = Column(Float)
    pit_rrmse               = Column(Float)
    pit_skill               = Column(Float)

    oracle_km_rmse             = Column(Float)
    oracle_base_rmse           = Column(Float)
    oracle_rrmse               = Column(Float)
    oracle_skill               = Column(Float)

    transformation_code = Column(Integer)
    nonstationary_mean  = Column(Float)
    nonstationary_std   = Column(Float)
    stationary_mean     = Column(Float)
    stationary_std      = Column(Float)
    revision_variance   = Column(Float)
    latency_mean_months = Column(Float)
    latency_std_months  = Column(Float)

    realtime_start          = Column(DateTime, default=datetime.now)



class MacrosNowcast(Base):
    __tablename__ = "macros_nowcast"
    run_hash                        = Column(String, primary_key=True)
    series_id                       = Column(String, primary_key=True, index=True)
    eval_end                        = Column(Date)
    current_start                   = Column(Date, primary_key=True, index=True)
    date                            = Column(Date, primary_key=True, index=True)
    value                                               = Column(Float)   
    seed_date                                           = Column(Date)
    seed_realtime_start                                 = Column(Date)
    seed_value                                          = Column(Float)   
    realtime_start                                      = Column(DateTime, default=datetime.now)

    
 
def upsert_metadata(
    run_hash: str,
    realtime_start: datetime,
    train_start_back_months: int,
    resample_frequency: str,
    dynamic_factor_n_factors: int,
    dynamic_factor_lag_order: int,
    dynamic_factor_statespace_maxiter: int,
    train_start: Any,
    eval_start: Any,
    eval_end: Any,
    current_start: Any,
    series_ids: List[str],
    n_observations: int,
    factor_rhos: List[float],
    log_likelihood: float,
    aic: float,
    bic: float,
    hqic: float,
    em_iterations: int,
    kalman_parameters: List[float],
):
    """Inserts or updates global fold-level model metadata."""
    upsert_sql = text("""
        INSERT INTO macros_nowcast_metadata (
            run_hash, train_start, eval_start, eval_end, current_start, resample_frequency, 
            dynamic_factor_n_factors, dynamic_factor_lag_order, 
            train_start_back_months, n_observations,
            dynamic_factor_statespace_maxiter, em_iterations, 
            factor_rho, log_likelihood, aic, bic, hqic,
            realtime_start, series_ids, kalman_parameters
        ) VALUES (
            :run_hash, :train_start, :eval_start, :eval_end, :current_start, :resample_frequency, 
            :dynamic_factor_n_factors, :dynamic_factor_lag_order, 
            :train_start_back_months, :n_observations,
            :dynamic_factor_statespace_maxiter, :em_iterations, 
            :factor_rho, :log_likelihood, :aic, :bic, :hqic,
            :realtime_start, :series_ids, :kalman_parameters
        )
       
    """)

    # Ensure factor_rhos is a clean list of python floats
    clean_factor_rhos = (
        factor_rhos.tolist() if hasattr(factor_rhos, "tolist") else list(factor_rhos)
    ) if factor_rhos is not None else []

    payload = {
        "run_hash": run_hash,
        "realtime_start": realtime_start,
        "train_start_back_months": train_start_back_months,
        "resample_frequency": resample_frequency,
        "dynamic_factor_n_factors": dynamic_factor_n_factors,
        "dynamic_factor_lag_order": dynamic_factor_lag_order,
        "dynamic_factor_statespace_maxiter": dynamic_factor_statespace_maxiter,
        "train_start": train_start,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "current_start": current_start,
        "series_ids": series_ids,
        "n_observations": n_observations,
        "factor_rho": clean_factor_rhos,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "hqic": hqic,
        "em_iterations": em_iterations,
        "kalman_parameters": kalman_parameters,
    }

    try:
        with engine.begin() as conn:
            conn.execute(upsert_sql, payload)
        tqdm.write(f"[*] Successfully synchronized fold metadata for [{eval_end}] to macros_nowcast_metadata.")
    except Exception as e:
        tqdm.write(f" [!] Ingestion Error during metadata staging: {e}")

def upsert_diagnostics(
    run_hash: str, 
    train_start: datetime,
    eval_start: datetime,
    eval_end: datetime,
    current_start: datetime,
    realtime_start: datetime, 
    series_dict: List[Dict[str, Any]]
):
    """Batch-inserts series-level transformation stats, skill metrics, and signal share."""
    if not series_dict:
        return

    upsert_sql = text("""
        INSERT INTO macros_nowcast_diagnostic (
            run_hash, train_start, eval_start, eval_end, current_start, series_id, realtime_start,
            eval_nsamples, signal_share,
            pit_km_rmse, pit_base_rmse, pit_rrmse, pit_skill,
            oracle_km_rmse, oracle_base_rmse, oracle_rrmse, oracle_skill,
            transformation_code, nonstationary_mean, nonstationary_std,
            stationary_mean, stationary_std, revision_variance,
            latency_mean_months, latency_std_months
        ) VALUES (
            :run_hash, :train_start, :eval_start, :eval_end, :current_start, :series_id, :realtime_start,
            :eval_nsamples, :signal_share,
            :pit_km_rmse, :pit_base_rmse, :pit_rrmse, :pit_skill,
            :oracle_km_rmse, :oracle_base_rmse, :oracle_rrmse, :oracle_skill,
            :transformation_code, :nonstationary_mean, :nonstationary_std,
            :stationary_mean, :stationary_std, :revision_variance,
            :latency_mean_months, :latency_std_months
        )
    """)

    for row in series_dict:
        row["run_hash"] = run_hash
        row["train_start"] = train_start
        row["eval_start"] = eval_start
        row["eval_end"] = eval_end
        row["realtime_start"] = realtime_start
        row["current_start"] = current_start

    try:
        with engine.begin() as conn:
            conn.execute(upsert_sql, series_dict)  # Vectorized bulk execute
        tqdm.write(f"[*] Successfully synchronized {len(series_dict)} rows to macros_nowcast_diagnostic.")
    except Exception as e:
        tqdm.write(f" [!] Ingestion Error during diagnostic staging: {e}")


def upsert_nowcast(
    run_hash: str, 
    eval_end: datetime, 
    current_start: datetime,
    realtime_start: datetime, 
    series_dict: List[Dict[str, Any]]
):
    """Batch-inserts point-in-time time series predictions including seed tracking metadata."""
    if not series_dict:
        return
    upsert_sql = text("""
        INSERT INTO macros_nowcast (
            run_hash, series_id, eval_end, current_start, date, value,
            seed_date, seed_realtime_start, seed_value, realtime_start
        ) VALUES (
            :run_hash, :series_id, :eval_end, :current_start, :date, :value,
            :seed_date, :seed_realtime_start, :seed_value, :realtime_start
        )
    """)

    for row in series_dict:
        row["run_hash"] = run_hash
        row["realtime_start"] = realtime_start
        row["eval_end"] = eval_end
        row["current_start"] = current_start

    try:
        with engine.begin() as conn:
            conn.execute(upsert_sql, series_dict)
        tqdm.write(f"[*] Successfully synchronized {len(series_dict)} rows to macros_nowcast.")
    except Exception as e:
        tqdm.write(f" [!] Ingestion Error during nowcast staging: {e}")



def _process_single_stride(
    stride_dt: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    current_start: pd.Timestamp,
    df_obs_orc: pd.DataFrame,
    mf_fit: dict,
    transform_metadata: Dict[str, Dict],
    train_start: pd.Timestamp,
    exog_params: Dict,
    config: Dict,
    column_order: List[str]
) -> pd.DataFrame:
    """Worker function executing a single stride evaluation in parallel."""
  
    stride_obs = df_obs_orc[
        (df_obs_orc['realtime_start'] <= stride_dt) & 
        (df_obs_orc['date'] <= stride_dt)
    ]
    
    train_obs_at_stride = stride_obs.to_dict(orient='records')
    
    mf_eval = nowcast(
        mf_fit=mf_fit,
        train_obs_orc=train_obs_at_stride,  
        transform_metadata=transform_metadata,
        train_start=train_start,
        current_time=stride_dt,  
        exog_params=exog_params,        
        config=config
    )

    fitted_values = mf_eval["fitted_values"]
    if not fitted_values.empty:
        fitted_values.index = pd.to_datetime(fitted_values.index).normalize()

    raw_obs_df = pd.DataFrame(train_obs_at_stride)
    if raw_obs_df.empty:
        records = [{
            f"{c}": np.nan,
            f"{c}__seed_date": None,
            f"{c}__seed_realtime_start": None,
            f"{c}__seed_value": np.nan
        } for c in column_order]
        flat_record = {}
        for r in records:
            flat_record.update(r)
        return pd.DataFrame(flat_record, index=[stride_dt])

    raw_obs_df['date'] = pd.to_datetime(raw_obs_df['date'])
    raw_obs_df['realtime_start'] = pd.to_datetime(raw_obs_df['realtime_start'])
    raw_obs_df = raw_obs_df.sort_values(['date', 'realtime_start'])

    stride_row = {}

    for col in column_order:
        col_str = str(col)
        tm = transform_metadata.get(col_str, {})
        code = tm.get("code", 1)
        denom = tm.get("code9_trillion_denom", 1.0)

        col_raw = raw_obs_df[raw_obs_df['series_id'] == col_str]
        valid_past = col_raw[col_raw['value'].notna()].copy()

        if valid_past.empty:
            stride_row[col_str] = np.nan
            stride_row[f"{col_str}__seed_date"] = None
            stride_row[f"{col_str}__seed_realtime_start"] = None
            stride_row[f"{col_str}__seed_value"] = np.nan
            continue

        valid_past['resampled_date'] = (valid_past['date'] + pd.offsets.MonthEnd(0)).dt.normalize()

        # Filter strictly for observations published on or before stride_dt
        valid_past_asof = valid_past[
            (valid_past['resampled_date'] <= stride_dt) & 
            (valid_past['realtime_start'] <= stride_dt)
        ]

        if valid_past_asof.empty:
            stride_row[col_str] = np.nan
            stride_row[f"{col_str}__seed_date"] = None
            stride_row[f"{col_str}__seed_realtime_start"] = None
            stride_row[f"{col_str}__seed_value"] = np.nan
            continue

        # Get latest published level, its realtime release date, and month-end observation date
        latest_obs = valid_past_asof.sort_values(by=['resampled_date', 'realtime_start']).iloc[-1]
        seed_date = latest_obs['resampled_date']
        seed_realtime_start = latest_obs['realtime_start'].date()
        last_known_level = float(latest_obs['value'])

        stride_row[f"{col_str}__seed_date"] = seed_date.date()
        stride_row[f"{col_str}__seed_realtime_start"] = seed_realtime_start
        stride_row[f"{col_str}__seed_value"] = last_known_level

        target_horizon_date = (stride_dt + pd.offsets.MonthEnd(0)).normalize()

        if seed_date >= target_horizon_date:
            stride_row[col_str] = last_known_level
        else:
            gap_deltas = fitted_values.loc[
                (fitted_values.index > seed_date) & 
                (fitted_values.index <= target_horizon_date), 
                col_str
            ]

            if gap_deltas.empty or gap_deltas.isna().any():
                stride_row[col_str] = np.nan
            else:
                reconstructed_series = reverse_transxf_fred(
                    y=gap_deltas,
                    tcode=code,
                    init_val1=last_known_level,
                    init_val2=last_known_level,
                    code9_trillion_denom=denom
                )
                if not reconstructed_series.empty and reconstructed_series.notna().all():
                    stride_row[col_str] = float(reconstructed_series.iloc[-1])
                else:
                    stride_row[col_str] = np.nan

    return pd.DataFrame(stride_row, index=[stride_dt])



def main():
    config = load_configuration()
    run_hash, meta_dir = initialize_run_environment(config)



    # 1. Total clean database state setup
    tqdm.write("[*] Dropping legacy tables to complete a clean sweep...")
    tables_to_drop = ["macros_nowcast_metadata", "macros_nowcast_diagnostic", "macros_nowcast"]
    for table_name in tables_to_drop:
        if table_name in Base.metadata.tables:
            Base.metadata.tables[table_name].drop(engine, checkfirst=True)
            tqdm.write(f"  -> Dropped table: {table_name}")
            
    # tqdm.write("[*] Rebuilding fresh database schema definitions...")
    Base.metadata.create_all(engine) 


    context_len_months = config["train_start_back_months"]
    current_start = pd.to_datetime(config["train_start"]) 
    max_lookback = pd.to_datetime(config["train_earliest_lookback"]) 
    today = pd.to_datetime(datetime.today()).normalize()
    end_backtest = (today + pd.offsets.MonthEnd(0)).normalize()

    walk_stride_months = config.get("walk_stride_months", 6)
    production_forward_months = config.get("production_forward_months", 3)  
    eval_start_back_months = config.get("eval_start_back_months", 12)  
    eval_end_back_months = config.get("eval_end_back_months", 6)  
    resample_frequency = config.get("resample_frequency", "1ME")
    global_metadata_lookup: Dict[int, dict] = {}
    all_eval_dfs = []
    realtime_start = datetime.now().date()
    

    while current_start <= end_backtest:
        out_dir = os.path.join(meta_dir, f"{current_start.strftime('%Y-%m-%d')}")
        os.makedirs(out_dir, exist_ok=True)
        train_start = max(current_start - relativedelta(months=context_len_months), max_lookback)
        eval_start = max(train_start, current_start - relativedelta(months=eval_start_back_months))
        eval_end = max(train_start, current_start - relativedelta(months=eval_end_back_months))
        production_end = current_start + relativedelta(months=production_forward_months)

        exo_config = config.get("exogeneity_set", [])
        file_path = os.path.join("macros", "exogeneity", exo_config)
        
        with open(file_path, "r") as f:
            exo_series_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
        # 2. Fetch Data
        with engine.connect() as conn:
            Z_1, train_obs_orc = fetch_fold_data(
                exogeneity_set=exo_series_list,
                current_start=production_end,
                train_start=train_start,
                conn=conn
            )
    

        print(f"[train, eval_start, eval_end, current_start, production_end] : [{train_start.date()}, {eval_start.date()}, {eval_end.date()}, {current_start.date()}, {production_end.date()}]")
        train_obs_fit = [
            r for r in train_obs_orc 
            if pd.to_datetime(r.get('realtime_start')) <= eval_start
            and pd.to_datetime(r.get('date')) <= eval_start
        ]
  
        train_metadata_lookup = {}
        with engine.connect() as conn:
            anchor_meta_query = text("""
                SELECT 
                    series_id_hash, 
                    series_id, 
                    frequency_short, 
                    units_short,
                    category_id,
                    title, 
                    notes
                FROM fred_series_filtered
                WHERE series_id IN :anchor_series_ids
            """)
            try:
                res = conn.execute(anchor_meta_query, {"anchor_series_ids": tuple(exo_series_list)})
                for row in res.fetchall():
                    r_dict = dict(row._mapping)
                    sid = r_dict.get("series_id")
                
                    train_metadata_lookup[sid] = r_dict
            except Exception as e:
                print(f" [!] Anchor metadata hydration skipped: {e}")

        global_metadata_lookup.update(train_metadata_lookup)

        fit_obs_context, active_sids = filter_active_universe(
            train_obs_context=train_obs_fit,
            metadata_lookup=train_metadata_lookup,
            current_start=eval_start, 
            config=config
        )

        df_stationary_fit, transform_metadata = classify_ADF(
            obs_context=fit_obs_context,
            train_start=train_start,
            current_time=eval_start, 
            resample_frequency=resample_frequency,
            metadata_lookup=train_metadata_lookup
        )
        column_order = list(df_stationary_fit.columns)  

        clean_rev_vars, clean_lat_mean, clean_lat_std = compute_precomputed_rev_vars(
            train_obs_fit=train_obs_fit,
            column_order=column_order,
            transform_metadata=transform_metadata,
            metadata_lookup=train_metadata_lookup,
            resample_frequency=resample_frequency, 
            config=config
        )


        
        summary = pd.DataFrame(
            [
                {
                    "series_id": str(col),
                    "freq": (meta := transform_metadata.get(str(col), {})).get("frequency_short", "M"),
                    "units": str(meta.get("units_short", ""))[:15],
                    "code": meta.get("code", 1),
                    "non_na": df_stationary_fit[col].notna().sum(),
                    "na": df_stationary_fit[col].isna().sum(),
                    "mean": df_stationary_fit[col].mean(),
                    "std": df_stationary_fit[col].std(),
                    "rev_var": clean_rev_vars[idx],
                    "lat_mean": clean_lat_mean[idx],
                    "lat_std": clean_lat_std[idx],
                }
                for idx, col in enumerate(df_stationary_fit.columns)
            ]
        ).set_index("series_id")

        if config.get("verbose", True):
            pd.set_option("display.max_rows", None)
            pd.set_option("display.width", 1000)
            print(f" PANEL SUMMARY & LATENCY METRICS ({resample_frequency})")
            print(summary.to_string())
            print("=" * 110 + "\n")


        exog_params = {
            "revision_variances": clean_rev_vars
        }

        # Fit training model
        mf_fit = fit_mixed_frequency_kalman(
            df_stationary=df_stationary_fit,
            transform_metadata=transform_metadata,
            train_start=train_start,
            eval_start=eval_start,  
            config=config
        )
   


        diag = parse_model_diagnostics(mf_fit)

        # Extract parameters array safely
        raw_params = mf_fit.get("params", mf_fit.get("kalman_parameters", []))
        kalman_params_list = raw_params.tolist() if isinstance(raw_params, np.ndarray) else list(raw_params)
        upsert_metadata(
            run_hash=run_hash,
            realtime_start=realtime_start,
            train_start_back_months=context_len_months,
            resample_frequency=resample_frequency,
            dynamic_factor_n_factors=config.get("dynamic_factor_n_factors", 1),
            dynamic_factor_lag_order=config.get("dynamic_factor_lag_order", 1),
            dynamic_factor_statespace_maxiter=config.get("dynamic_factor_statespace_maxiter", 500),
            train_start=train_start.date(),
            eval_start=eval_start.date(),
            eval_end=eval_end.date(),
            current_start=current_start.date(), # <--- Fixed date type conversion
            series_ids=[str(c) for c in column_order],
            n_observations=len(df_stationary_fit),
            factor_rhos=mf_fit["factor_rhos"],
            log_likelihood=diag["log_likelihood"],
            aic=diag["aic"],
            bic=diag["bic"],
            hqic=diag["hqic"],
            em_iterations=diag["em_iterations"],
            kalman_parameters=kalman_params_list
        )


    
        # 3. Build Full Prediction Index (Evaluation Horizon + Production Horizon)
        effective_prod_end = current_start + relativedelta(months=production_forward_months)
        

        eval_dates = pd.date_range(start=eval_start, end=current_start, freq=resample_frequency)
        

        prod_dates = pd.date_range(
            start=current_start + pd.tseries.frequencies.to_offset(resample_frequency), 
            end=effective_prod_end, 
            freq=resample_frequency
        )
        # Union of evaluation dates and valid production horizon
        bitemp_index = sorted(list(set(eval_dates).union(set(prod_dates))))

    
        # Strict safety filter: drop any stride date beyond end_backtest
        bitemp_index = [d for d in bitemp_index if d <= end_backtest]

        prediction_records = []
        df_obs_orc = pd.DataFrame(train_obs_orc)
        if not df_obs_orc.empty:
            df_obs_orc['realtime_start'] = pd.to_datetime(df_obs_orc['realtime_start'])
            df_obs_orc['date'] = pd.to_datetime(df_obs_orc['date'])
        max_workers = min(os.cpu_count() or 4, len(bitemp_index))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                    executor.submit(
                        _process_single_stride,
                        stride_dt=stride_dt,
                        eval_start=eval_start,
                        eval_end=eval_end,
                        current_start=current_start,
                        df_obs_orc=df_obs_orc,
                        mf_fit=mf_fit,
                        transform_metadata=transform_metadata,
                        train_start=train_start,
                        exog_params=exog_params,
                        config=config,
                        column_order=column_order
                    )
                    for stride_dt in bitemp_index
                ]
            for future in as_completed(futures):
                prediction_records.append(future.result())
        # Sort index back to chronological order after parallel execution
        prediction_records.sort(key=lambda df: df.index[0])
    
        kalman_df_levels = pd.concat(prediction_records)
        nowcast_payload = []
        
        for dt, row in kalman_df_levels.iterrows():
            obs_date = dt.date() 
            for col in column_order:
                col_str = str(col)
                val = row.get(col_str)
                s_date = row.get(f"{col_str}__seed_date")
                s_rts = row.get(f"{col_str}__seed_realtime_start")
                s_val = row.get(f"{col_str}__seed_value")

                # Only ingest if the forecasted value is valid
                if pd.notna(val):
                    nowcast_payload.append({
                        "series_id": col_str,
                        "date": obs_date,
                        "value": float(val),
                        "seed_date": s_date if pd.notna(s_date) else None,
                        "seed_realtime_start": s_rts if pd.notna(s_rts) else None,
                        "seed_value": float(s_val) if pd.notna(s_val) else None
                    })

        # -----------------------------------------------------------------
        # UPSERT NOWCAST VALUES (WITH EVAL_END)
        # -----------------------------------------------------------------
        upsert_nowcast(
            run_hash=run_hash,
            eval_end=eval_end.date(),
            current_start=current_start.date(),
            realtime_start=realtime_start,
            series_dict=nowcast_payload
        )
        # eval_kalman_levels = kalman_df_levels.loc[kalman_df_levels.index <= current_start]
        
        eval_df = evaluate_and_log_fold(
            kalman_df_raw=kalman_df_levels,
            train_obs=train_obs_orc,
            train_metadata_lookup=train_metadata_lookup,
            train_columns=column_order,
            eval_start=eval_start,
            eval_end=eval_end,
            current_start=current_start,
            bitemp_index=bitemp_index,
            config=config, 
            out_dir=out_dir
        )

        analyze_global_backtest(
                all_eval_dfs=[eval_df],
                global_metadata_lookup=train_metadata_lookup,
                meta_dir=out_dir,
                eval_start=eval_start,
                eval_end=eval_end,
                current_start=current_start,
                production_end=production_end,
                today=today,
                config=config
            )
        # -----------------------------------------------------------------
        # UPSERT FOLD DIAGNOSTICS & SKILL METRICS
        # -----------------------------------------------------------------
        raw_obs_df = pd.DataFrame(train_obs_fit)
        diagnostic_payload = []
        signal_share_map = mf_fit.get("signal_share", {})

        for idx, col in enumerate(column_order):
            col_str = str(col)
            tm = transform_metadata.get(col_str, {})
            
            s_vals = raw_obs_df[raw_obs_df['series_id'] == col_str]['value'].dropna().astype(float)
            nonstat_mean = float(s_vals.mean()) if not s_vals.empty else 0.0
            nonstat_std = float(s_vals.std()) if not s_vals.empty else 1.0
            
            stat_s = df_stationary_fit[col_str].dropna()
            stat_mean = float(stat_s.mean()) if not stat_s.empty else 0.0
            stat_std = float(stat_s.std()) if not stat_s.empty else 1.0

            rev_var = float(clean_rev_vars[idx]) if idx < len(clean_rev_vars) else 0.0
            lat_mean = float(clean_lat_mean[idx]) if idx < len(clean_lat_mean) else 0.0
            lat_std = float(clean_lat_std[idx]) if idx < len(clean_lat_std) else 0.0
            sig_share = float(signal_share_map.get(col_str, 0.0))

            # --- ENFORCE EXACT SAMPLE INTERSECTION (PIT vs ORACLE) ---
            col_eval = eval_df[
                (eval_df['series_id'] == col_str) & 
                (pd.to_datetime(eval_df['date']) >= eval_start) &
                (pd.to_datetime(eval_df['date']) <= eval_end) &
                (eval_df['orc_val'].notna()) &       # Replaces is_oracle_target
                (eval_df['pit_val'].notna()) &       # Replaces is_pit_target
                (eval_df['kalman_val'].notna()) &    # Ensures prediction exists
                (eval_df['filled_val'].notna())      # Ensures baseline exists
            ]
            n_eval_samples = len(col_eval)

            if n_eval_samples > 0:
                # Oracle Metrics (evaluated against final revisions)
                orc_km_mse = col_eval['km_sq_err_oracle'].mean()
                orc_base_mse = col_eval['base_sq_err_oracle'].mean()
                orc_km_rmse = float(np.sqrt(orc_km_mse))
                orc_base_rmse = float(np.sqrt(orc_base_mse))
                orc_rrmse = (orc_km_rmse / orc_base_rmse) if orc_base_rmse > 1e-6 else (1.0 if orc_km_rmse <= 1e-6 else 0.0)
                orc_skill = float(1.0 - orc_rrmse)

                # Point-in-time Metrics (evaluated against first release at stride time)
                pit_km_mse = col_eval['km_sq_err_pit'].mean()
                pit_base_mse = col_eval['base_sq_err_pit'].mean()
                pit_km_rmse = float(np.sqrt(pit_km_mse))
                pit_base_rmse = float(np.sqrt(pit_base_mse))
                pit_rrmse = (pit_km_rmse / pit_base_rmse) if pit_base_rmse > 1e-6 else (1.0 if pit_km_rmse <= 1e-6 else 0.0)
                pit_skill = float(1.0 - pit_rrmse)
            else:
                orc_km_rmse = orc_base_rmse = orc_rrmse = orc_skill = 0.0
                pit_km_rmse = pit_base_rmse = pit_rrmse = pit_skill = 0.0

            diagnostic_payload.append({
                "series_id": col_str,
                "current_start": current_start.date(),
                "signal_share": sig_share,
                "transformation_code": int(tm.get("code", 1)),
                "nonstationary_mean": nonstat_mean,
                "nonstationary_std": nonstat_std,
                "stationary_mean": stat_mean,
                "stationary_std": stat_std,
                "revision_variance": rev_var,
                "latency_mean_months": lat_mean,
                "latency_std_months": lat_std,
                "eval_nsamples": n_eval_samples,
                # Point-in-time metrics
                "pit_km_rmse": pit_km_rmse,
                "pit_base_rmse": pit_base_rmse,
                "pit_rrmse": float(pit_rrmse),
                "pit_skill": pit_skill,
                # Oracle metrics
                "oracle_km_rmse": orc_km_rmse,
                "oracle_base_rmse": orc_base_rmse,
                "oracle_rrmse": float(orc_rrmse),
                "oracle_skill": orc_skill
            })


        upsert_diagnostics(
            run_hash=run_hash,
            train_start=train_start.date(),
            eval_start=eval_start.date(),
            eval_end=eval_end.date(),
            current_start=current_start.date(),
            realtime_start=realtime_start,
            series_dict=diagnostic_payload
        )


        all_eval_dfs.append(eval_df)


            
        current_start += relativedelta(months=walk_stride_months)
        # break
    # 6. Global Aggregate Diagnostics across all processed folds
    # analyze_global_backtest(
    #     all_eval_dfs=all_eval_dfs,
    #     global_metadata_lookup=global_metadata_lookup,
    #     meta_dir=meta_dir,
    #     config=config
    # )

if __name__ == "__main__":


    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
        
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime') 
    stats.print_stats(20)