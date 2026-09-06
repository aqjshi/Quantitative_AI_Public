import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta



from macros.display.residuals import analyze_global_backtest

def evaluate_and_log_fold(
    kalman_df_raw: pd.DataFrame,
    train_obs: List[dict],
    train_metadata_lookup: dict,
    train_columns: List[str],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    current_start: pd.Timestamp,
    bitemp_index:Any, 
    config: dict, 
    
    out_dir: str
) -> pd.DataFrame:

    obs_map: Dict[str, List[tuple]] = {}
    for row in train_obs:
        val = row.get('value')
        if val is None or pd.isna(val):
            continue
        sid_key = str(row['series_id'])
        if sid_key not in obs_map:
            obs_map[sid_key] = []
        
        obs_dt = pd.to_datetime(row['date'])
        rts_dt = pd.to_datetime(row.get('realtime_start', row['date']))
        rte_val = row.get('realtime_end', '9999-12-31')
        
        obs_map[sid_key].append((obs_dt, rts_dt, rte_val, float(val)))
    series_vintages_dict = obs_map

    km_lookup_df = kalman_df_raw.copy() if not kalman_df_raw.empty else pd.DataFrame()
    if not km_lookup_df.empty:
        km_lookup_df.columns = [str(c).strip() for c in km_lookup_df.columns]
        km_lookup_df.index = pd.to_datetime(km_lookup_df.index)
    
    merged_records = []
 
    for raw_col in train_columns:
        meta = train_metadata_lookup.get(raw_col, {})
        sid = meta.get('series_id')
        freq = meta.get('frequency_short')
        units = meta.get('units_short')
        category_id = meta.get("category_id")
        
        series_vintages = series_vintages_dict.get(sid, [])
        
        orc_vals, orc_obs_dates, orc_rts, orc_rte = [], [], [], []
        pit_vals, pit_obs_dates, pit_rts, pit_rte = [], [], [], []
        ff_vals, ff_obs_dates, ff_rts, ff_rte = [], [], [], []
        km_vals = []

        for stride_dt in bitemp_index:
          
            interval_start_dt = stride_dt.replace(day=1)
       
            orc_matches = [
                v for v in series_vintages 
                if interval_start_dt <= v[0] <= stride_dt
            ] 
            if orc_matches:
             
                best_orc = max(orc_matches, key=lambda x: (x[0], x[1])) # Max realtime_start
                # --- DEBUG PRINT FOR NROU ---
                # if sid.upper() == 'NROU':
                #     print(f"[FOUND NROU] stride_dt={stride_dt.date()} | match_obs={best_orc[0].date()} | rts={best_orc[1].date()} | val={best_orc[3]}")
                orc_obs_dates.append(best_orc[0])
                orc_rts.append(best_orc[1])
                orc_rte.append(str(best_orc[2]))
                orc_vals.append(best_orc[3])
            else:
                orc_vals.append(np.nan)
                orc_obs_dates.append(stride_dt)
                orc_rts.append(stride_dt)
                orc_rte.append('9998-12-31')

            # -------------------------------------------------------------
            # 2. Oracle wrt to current time
            # -------------------------------------------------------------
            pit_matches = [
                v for v in series_vintages 
                if interval_start_dt <= v[0] <= stride_dt
                and (v[1] <= current_start)
            ]
            if pit_matches:
                # Pick the latest observation date, and break ties with the latest revision (max realtime_start)
                best_pit = max(pit_matches, key=lambda x: (x[0], x[1]))
                
                pit_obs_dates.append(best_pit[0])
                pit_rts.append(best_pit[1])
                pit_rte.append(str(best_pit[2]))
                pit_vals.append(best_pit[3])   
            else:
                pit_vals.append(np.nan)
                pit_obs_dates.append(stride_dt)
                pit_rts.append(current_start)
                pit_rte.append('9998-12-31')

            # -------------------------------------------------------------
            # 3. BASELINE FORWARD-FILL TARGET: Latest observation as of stride_dt
            # -------------------------------------------------------------
            ff_matches = [
                v for v in series_vintages 
                if (v[0] <= stride_dt) and (v[1] <= stride_dt)
            ]
            if ff_matches:
                best_ff = max(ff_matches, key=lambda x: (x[0], x[1]))
                ff_obs_dates.append(best_ff[0])
                ff_rts.append(best_ff[1])
                ff_rte.append(str(best_ff[2]))
                ff_vals.append(best_ff[3])
            else:
                ff_vals.append(np.nan)
                ff_obs_dates.append(stride_dt)
                ff_rts.append(stride_dt)
                ff_rte.append('9998-12-31')

            # -------------------------------------------------------------
            # 4. KALMAN LOOKUP
            # -------------------------------------------------------------
            val_km = np.nan
            if not km_lookup_df.empty:
                if stride_dt in km_lookup_df.index:
                    val_km = km_lookup_df.loc[stride_dt, sid]
                else:
                    valid_idx = km_lookup_df.index[km_lookup_df.index <= stride_dt]
                    if not valid_idx.empty:
                        val_km = km_lookup_df.loc[valid_idx[-1], sid]
            km_vals.append(val_km)

        sub_df = pd.DataFrame({
            'date': bitemp_index,
            'series_id': sid,
            'freq': freq,
            'units': units,
            'category_id': category_id,
            'orc_val': orc_vals,
            'orc_obs': orc_obs_dates,
            'orc_realtime_start': orc_rts,
            'orc_realtime_end': orc_rte,
            'pit_val': pit_vals,
            'pit_obs': pit_obs_dates,
            'pit_realtime_start': pit_rts,
            'pit_realtime_end': pit_rte,
            'filled_val': ff_vals,
            'filled_date': ff_obs_dates,
            'filled_realtime_start': ff_rts,
            'filled_realtime_end': ff_rte,  
            'kalman_val': km_vals,
        })

        act_arr = np.nan_to_num(sub_df['orc_val'].to_numpy(), nan=np.nan, posinf=np.nan, neginf=np.nan)
        pit_arr = np.nan_to_num(sub_df['pit_val'].to_numpy(), nan=np.nan, posinf=np.nan, neginf=np.nan)
        km_arr = np.nan_to_num(sub_df['kalman_val'].to_numpy(), nan=np.nan, posinf=np.nan, neginf=np.nan)
        base_arr = np.nan_to_num(sub_df['filled_val'].to_numpy(), nan=np.nan, posinf=np.nan, neginf=np.nan)

        valid_km_orc = np.isfinite(act_arr) & np.isfinite(km_arr)
        valid_km_pit = np.isfinite(pit_arr) & np.isfinite(km_arr)
        valid_base_orc = np.isfinite(act_arr) & np.isfinite(base_arr)
        valid_base_pit = np.isfinite(pit_arr) & np.isfinite(base_arr)

        sub_df['km_err_oracle'] = np.where(valid_km_orc, act_arr - km_arr, np.nan)
        sub_df['km_sq_err_oracle'] = np.where(valid_km_orc, sub_df['km_err_oracle'] ** 2, np.nan)
        
        sub_df['km_err_pit'] = np.where(valid_km_pit, pit_arr - km_arr, np.nan)
        sub_df['km_sq_err_pit'] = np.where(valid_km_pit, sub_df['km_err_pit'] ** 2, np.nan)

        sub_df['base_err_oracle'] = np.where(valid_base_orc, act_arr - base_arr, np.nan)
        sub_df['base_sq_err_oracle'] = np.where(valid_base_orc, sub_df['base_err_oracle'] ** 2, np.nan)
        
        sub_df['base_err_pit'] = np.where(valid_base_pit, pit_arr - base_arr, np.nan)
        sub_df['base_sq_err_pit'] = np.where(valid_base_pit, sub_df['base_err_pit'] ** 2, np.nan)
    
        merged_records.append(sub_df)

    eval_df = pd.concat(merged_records, ignore_index=True)
    eval_df['current_start'] = current_start

    os.makedirs(out_dir, exist_ok=True)
    csv_out_path = os.path.join(out_dir, "fold_predictions.csv")
    
    export_cols = [
        'date', 'series_id', 'category_id', 'freq', 'units',
        'orc_val', 'pit_val', 'filled_val', 'kalman_val', 
        'km_err_oracle', 'km_err_pit', 'base_err_oracle', 'base_err_pit'
    ]
    
    present_cols = [c for c in export_cols if c in eval_df.columns]
    eval_df[present_cols].to_csv(csv_out_path, index=False)
    print(f"[+] Exported fold prediction levels to: {csv_out_path}")



    return eval_df


