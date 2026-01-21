import json
from models import SubsecondQuote,  Quote, Company, Base
from sqlalchemy.orm import sessionmaker
from db import engine
import matplotlib.pyplot as plt


from datetime import datetime, timedelta, timezone , time as dt_time
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import random
import os
from scipy.stats import beta
import io
import matplotlib.pyplot as plt
# memory to database
def upsert_historical_emulate_quotes_via_copy_batched(
    data_df: pd.DataFrame, # Now accepts a DataFrame directly
    ticker: str,       # And the company_id separately
    batch_size: int = 5000
):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        ticker_comp = session.query(Company).filter(Company.ticker == ticker).first()
    company_id = ticker_comp.id
    # Create a copy to avoid modifying the original DataFrame passed in
    df_to_upsert = data_df.copy()
    # print("data_df")
    # print(data_df.head())

    df_to_upsert.rename(columns={'VOLUME': 'volume'}, inplace=True)

    # Add company_id column
    df_to_upsert['company_id'] = company_id
    df_to_upsert['time_entry_ts'] = df_to_upsert.index

    df_to_upsert.rename(columns={'CLOSE': 'close_price'}, inplace=True)
    # print("df_to_upsert")
    # print(df_to_upsert.head())
    # Select and reorder columns to match the database table
    columns = [
        "company_id",
        "time_entry_ts",
        "close_price", # Assuming 'CLOSE' from the merged_data is the close_price
        "volume"
    ]
    



    # Ensure all required columns are present before proceeding
    missing_cols = [col for col in columns if col not in df_to_upsert.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    df_to_upsert = df_to_upsert[columns]

    # Cast to appropriate dtypes for database copy
    dtypes = {
        "company_id": "int64",
        "time_entry_ts": "int64",
        "close_price": "float64",
        "volume": "int64"
    }
    df_to_upsert = df_to_upsert.astype(dtypes)

    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()

    try:
        cur.execute("""
            CREATE TEMP TABLE historical_emulate_quote_stage (
              company_id      bigint,
              time_entry_ts   bigint,
              close_price     double precision,
              volume          bigint
            ) ON COMMIT DROP;
        """)

        total_rows = len(df_to_upsert)
        for i in range(0, total_rows, batch_size):
            batch_df = df_to_upsert.iloc[i : i + batch_size]
            # print(f"Processing subsecond batch {i // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size} "
            #       f"({len(batch_df)} rows) for company_id {company_id}")

            buf = io.StringIO()
            batch_df.to_csv(buf, sep="\t", index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cur.copy_from(
                buf,
                "historical_emulate_quote_stage",
                sep="\t",
                columns=columns
            )
            buf.close()

        cur.execute("""
            INSERT INTO historical_emulate_quotes (
              company_id, time_entry_ts,
              close_price, volume
            )
            SELECT
              company_id, time_entry_ts,
              close_price, volume
            FROM historical_emulate_quote_stage
            ON CONFLICT (company_id, time_entry_ts) DO NOTHING;
        """)

        raw_conn.commit()
        # print(f"Successfully upserted {total_rows} historical_emulate_quote quotes for company_id {company_id}.")

    except Exception as e:
        raw_conn.rollback()
        print(f"Error during batched historical_emulate_quote upsert for company_id {company_id}: {e}")
        raise

    finally:
        cur.close()
        raw_conn.close()


def modified_astb(
    forecast_close_prices: pd.Series,
    sma_UB: pd.Series,
    sma_MB: pd.Series,
    sma_LB: pd.Series,
    current_t_s: int,
    end_t_s: int,
) -> Dict[str, Any]:
    """
    Derives S and T points and A/B values from the SMA series themselves,
    using the provided start and end timestamps.
    """
    full_results = {}



    if sma_MB.empty or sma_UB.empty or sma_LB.empty:
        return {k: None for k in [
            's_sma', 't_sma',
            's_time', 's_value', 't_time', 't_value',
            'a_time', 'b_time', 'a_value', 'b_value'
        ]}

    # Convert timestamps to datetime objects for .loc lookup
    a_idx_dt = pd.to_datetime(current_t_s, unit='s')
    b_idx_dt = pd.to_datetime(end_t_s, unit='s')

    # Find first crossing of MB above UB (T candidate) and below LB (S candidate)
    s_candidates = sma_MB[sma_MB <= sma_LB]
    t_candidates = sma_MB[sma_MB >= sma_UB]

    # If never crosses, fallback to min/max of MB
    s_idx = s_candidates.index[0] if not s_candidates.empty else sma_MB.idxmin()
    t_idx = t_candidates.index[0] if not t_candidates.empty else sma_MB.idxmax()

    # Ensure S and T are ordered in time
    s_extrema_idx = min(s_idx, t_idx)
    t_extrema_idx = max(s_idx, t_idx)

    # Populate results for S and T
    full_results['s_time'] = int(s_extrema_idx.timestamp())
    # full_results['s_sma'] = sma_MB.get(s_extrema_idx)
    # full_results['s_value'] = forecast_close_prices.get(s_extrema_idx)
    full_results['s_value'] = sma_MB.get(s_extrema_idx)

    full_results['t_time'] = int(t_extrema_idx.timestamp())
    # full_results['t_sma'] = sma_MB.get(t_extrema_idx)
    # full_results['t_value'] = forecast_close_prices.get(t_extrema_idx)
    full_results['t_value'] = sma_MB.get(t_extrema_idx)



    # Populate results for A and B using the provided timestamps
    full_results['a_time'] = current_t_s
    full_results['b_time'] = end_t_s

    full_results['a_value'] = forecast_close_prices.get(a_idx_dt)
    full_results['b_value'] = forecast_close_prices.get(b_idx_dt)

    # # Check and print if the B value was found
    # if full_results['b_value'] is not None:
    #     print(f"B value found at {b_idx_dt}: {full_results['b_value']}")
    # else:
    #     print(f"B value NOT found for theoretical end time: {b_idx_dt}")

    return full_results

def solve_segment(
    segment_data: pd.Series, 
    vol_segment_data: pd.Series, 
    forecast_depth: int = 45
) -> Dict[str, Any]:
    features = {}
    features['duration'] = (len(segment_data) / forecast_depth)
    features['start'] = segment_data.iloc[0]    # for reconstruction 
    features['end'] = segment_data.iloc[-1]     # for reconstruction 
    features['drift'] = (features['end'] - features['start'])  / features['start']  # to be solved
    features['windedness'] = segment_data.diff().dropna().abs().sum() / features['start'] # to be solved

    features['range'] = ( segment_data.max()  - segment_data.min() )  / features['start']  # to be solved
    epsilon = 1e-5 # A small constant to prevent division by zero

    # --- Robust price_slope calculation ---
    if len(segment_data) < 120:
        features['price_slope'] = 0
    else:
        half_time = int(len(segment_data) / 2)
        first_half_slice = segment_data.iloc[0:half_time]
        second_half_slice = segment_data.iloc[half_time:]
        first_half_mean = first_half_slice.mean() + epsilon
        second_half_mean = second_half_slice.mean() + epsilon

        # Guarded calculation
 
        price_slope_val = (second_half_mean - first_half_mean) / first_half_mean
        
        # Final robustness check to catch any remaining inf or NaN values
        if np.isinf(price_slope_val) or np.isnan(price_slope_val):
            features['price_slope'] = 0.0
        else:
            features['price_slope'] = price_slope_val
            
    # --- Robust vol_slope calculation ---
    if len(vol_segment_data) < 120:
        features['vol_slope'] = 0
    else:
        half_time = int(len(vol_segment_data) / 2)
        first_half_slice = vol_segment_data.iloc[0:half_time]
        second_half_slice = vol_segment_data.iloc[half_time:]
        first_half_mean = first_half_slice.mean() + epsilon
        second_half_mean = second_half_slice.mean() + epsilon
        
    
        vol_slope_val = (second_half_mean - first_half_mean) / first_half_mean
            
        # Final robustness check to catch any remaining inf or NaN values
        if np.isinf(vol_slope_val) or np.isnan(vol_slope_val):
            features['vol_slope'] = 0.0
        else:
            features['vol_slope'] = vol_slope_val
            
    return features

# solver of astb into ab as st tb
def solve_astb(
    df_horizon_segment: pd.DataFrame, 
    forecast_depth: int = 45
) -> Dict[str, Any]:
    full_results = {}

    close_series = pd.to_numeric(df_horizon_segment['CLOSE'], errors='coerce').dropna()
    volume_series  = pd.to_numeric(df_horizon_segment['VOLUME'], errors='coerce').dropna()
    if close_series.empty:
        default_features = {k: np.nan for k in [
            'duration', 'start', 'end', 'drift',
            'windedness', 'range', 
            'price_slope', 'vol_slope'
        ]}
        
        full_results = {
            'ab_' + k: v for k, v in default_features.items()
        }
        for prefix in ['as_', 'st_', 'tb_']:
            full_results.update({prefix + k: v for k, v in default_features.items()})

        full_results.update({
            's_time': np.nan, 's_value': np.nan,
            't_time': np.nan, 't_value': np.nan
        })
        return full_results

    ab_features = solve_segment(close_series, volume_series, forecast_depth)
    for k, v in ab_features.items():
        full_results[f'ab_{k}'] = round(v, 3)

    extrema1 = close_series.idxmin()
    extrema2 = close_series.idxmax()
    
    s_extrema_idx =  extrema1 if extrema1 < extrema2 else extrema2
    t_extrema_idx =  extrema1 if extrema1 > extrema2 else extrema2


    full_results['s_time'] = s_extrema_idx
    full_results['s_value'] = close_series[s_extrema_idx]
    full_results['t_time'] = t_extrema_idx
    full_results['t_value'] = close_series[t_extrema_idx]



    a_idx = close_series.index[0]
    b_idx = close_series.index[-1]
    full_results['a_time'] = a_idx
    full_results['b_time'] = b_idx


    as_segment_data = close_series.loc[a_idx:s_extrema_idx]
    volume_as_segment_data = volume_series.loc[a_idx:s_extrema_idx]
    as_features = solve_segment(as_segment_data, volume_as_segment_data, forecast_depth)
    for k, v in as_features.items():
        full_results[f'as_{k}'] = round(v, 3)

    st_segment_data = close_series.loc[s_extrema_idx:t_extrema_idx]
    volume_st_segment_data = volume_series.loc[s_extrema_idx:t_extrema_idx]
    st_features = solve_segment(st_segment_data, volume_st_segment_data, forecast_depth)
    for k, v in st_features.items():
        full_results[f'st_{k}'] = round(v, 3)

    tb_segment_data = close_series.loc[t_extrema_idx:b_idx]
    volume_tb_segment_data = volume_series.loc[t_extrema_idx:b_idx]
    tb_features = solve_segment(tb_segment_data, volume_tb_segment_data, forecast_depth)
    for k, v in tb_features.items():
        full_results[f'tb_{k}'] = round(v, 3)
    
    return full_results


# get the subsecond quotes
def process_subsecond_quotes(comp: Company, intended_hours_ts: List[int], step_size=1) -> pd.DataFrame:
    # 1. Determine the query range in microseconds, converting to standard ints
    start_ts_us = int(intended_hours_ts[0] * 1_000_000)
    end_ts_us = int(intended_hours_ts[-1] * 1_000_000)

    # 2. Fetch the raw sub-second data from the database
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        rows = (
            session.query(
                SubsecondQuote.time_entry_ts,
                SubsecondQuote.close_price,
                SubsecondQuote.volume
            )
            .filter_by(company_id=comp.id)
            .filter(SubsecondQuote.time_entry_ts.between(start_ts_us, end_ts_us))
            .order_by(SubsecondQuote.time_entry_ts)
            .all()
        )

    # 3. Handle case where no data is returned.
    if not rows:
        print("No raw sub-second data found in the specified range. Returning an empty DataFrame.")
        return pd.DataFrame(columns=['CLOSE', 'VOLUME'])
    
    # 4. Load raw data into a DataFrame and convert to DatetimeIndex
    raw_df = pd.DataFrame(rows, columns=["uts_us", "CLOSE", "VOLUME"])
    raw_df['uts_us'] = pd.to_datetime(raw_df['uts_us'], unit='us')
    raw_df = raw_df.set_index('uts_us')
    
    # 5. Resample the data to a 1-second frequency ('1S')
    # Use aggregation methods: 'sum' for volume, 'last' for close price
    resampled_df = raw_df.resample('1s').agg({
        'CLOSE': 'last', # Use the last close price for the second
        'VOLUME': 'sum'  # Sum up all volumes within the second
    })

    # 6. Fill missing values
    # Forward-fill and back-fill the close prices to fill gaps
    resampled_df['CLOSE'] = resampled_df['CLOSE'].ffill().bfill()
    # Fill any seconds with no volume with 0
    resampled_df['VOLUME'] = resampled_df['VOLUME'].diff().fillna(0)

    # 7. Convert the index to integer timestamps (seconds) as your other functions expect
    final_df = resampled_df.copy()
    # final_df.index = final_df.index.astype(int) // 10**9 # Convert DatetimeIndex to seconds

    return final_df

# get the minute aggregates
def process_minute_quotes(comp: Company, intended_hours_ts: List[int]):
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        rows = (
            session.query(
                Quote.time_entry_ts,
                Quote.open_price, Quote.high_price,
                Quote.low_price, Quote.close_price,
                Quote.volume
            )
            .filter_by(company_id=comp.id)
            .filter(Quote.time_entry_ts.between(intended_hours_ts[0], intended_hours_ts[-1]))
            .order_by(Quote.time_entry_ts)
            .all()
        )

    if not rows:
        return pd.DataFrame(columns=["uts", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]).set_index("uts")

    minute_bars_df = pd.DataFrame(rows, columns=["uts", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    minute_bars_df.set_index("uts", inplace=True)
    # print(minute_bars_df.head())
    return minute_bars_df

# helper of emulate brownian bridge
def emulate_brownian_bridge_segment(
    start_unix_s: int,
    end_unix_s: int,
    start_value: float,
    end_value: float,
    computational_density_per_unit: int, 
    local_sigma: float,
):
    # Calculate exact duration
    bridge_interval_duration_in_s = end_unix_s- start_unix_s

    if computational_density_per_unit <= 0:
        raise ValueError("computational_density_per_unit must be a positive integer.")
    
    if bridge_interval_duration_in_s <= 0:
        index_dt = pd.to_datetime([start_unix_s], unit='s')
        return pd.DataFrame({"CLOSE": [start_value]}, index=index_dt)


    num_intervals = int(bridge_interval_duration_in_s / computational_density_per_unit)
    num_interpolated_points = num_intervals + 1 # num_points = num_intervals + 1 for endpoint=True
    
    if num_interpolated_points < 2:
        index_dt = pd.to_datetime([start_unix_s], unit='s')
        return pd.DataFrame({"CLOSE": [start_value]}, index=index_dt)

    t = np.linspace(0, 1, num_interpolated_points, endpoint=True)
    dt_norm = 1.0 / (num_interpolated_points - 1) # Normalized time step

    dW = np.random.standard_normal(size=num_interpolated_points) * np.sqrt(dt_norm)
    dW[0] = 0.0
    W = np.cumsum(dW)

    bridge_values = start_value + t * (end_value - start_value) + local_sigma * (W - t * W[-1]) * np.sqrt(bridge_interval_duration_in_s )
    current_time_stamps_us = np.arange(start_unix_s, end_unix_s + computational_density_per_unit, computational_density_per_unit, dtype=np.int64)

    if current_time_stamps_us[-1] != end_unix_s:
         current_time_stamps_us = np.append(current_time_stamps_us, end_unix_s)


    index_dt = pd.date_range(start=pd.to_datetime(start_unix_s, unit='s'),
                             end=pd.to_datetime(end_unix_s, unit='s'),
                             freq=f'{computational_density_per_unit}s', 
                             inclusive='both') 


    if len(index_dt) != len(bridge_values):
        num_points_actual = len(index_dt)
        t_actual = np.linspace(0, 1, num_points_actual, endpoint=True)
        dt_norm_actual = 1.0 / (num_points_actual - 1) if num_points_actual > 1 else 1.0
        dW_actual = np.random.standard_normal(size=num_points_actual) * np.sqrt(dt_norm_actual)
        dW_actual[0] = 0.0
        W_actual = np.cumsum(dW_actual)
        bridge_values = start_value + t_actual * (end_value - start_value) + local_sigma * (W_actual - t_actual * W_actual[-1]) * np.sqrt(bridge_interval_duration_in_s )


    final_df = pd.DataFrame({
        "CLOSE": bridge_values
    }, index=index_dt)
    
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    return final_df

# turns minute aggregates into subsecond
def emulate_brownian_bridge(
    minute_aggregate: pd.DataFrame,
    output_density_per_unit: int,
    bridge_length_in_bars: int,
    num_samples: int = 20
) -> pd.DataFrame:
    brownian_bridge_segments = [] 

    if len(minute_aggregate) < 2:
        print("Warning: Not enough aggregate data to create bridges.")
        return pd.DataFrame(columns=['CLOSE'], index=pd.DatetimeIndex([])) # Return empty with correct index type
        
    if not isinstance(minute_aggregate.index, pd.DatetimeIndex):
         minute_aggregate.index = pd.to_datetime(minute_aggregate.index, unit='us') # assuming it's still raw uts

    if len(minute_aggregate) > 1:
        aggregate_bar_unit_duration_us = (minute_aggregate.index[1] - minute_aggregate.index[0]).total_seconds()
    else:
        print("Warning: Single aggregate bar, cannot determine duration for bridges.")
        return pd.DataFrame(columns=['CLOSE'], index=pd.DatetimeIndex([]))

    if aggregate_bar_unit_duration_us <= 0:
        print("Warning: Aggregate data bars have non-positive duration.")
        return pd.DataFrame(columns=['CLOSE'], index=pd.DatetimeIndex([]))
    bridge_normalization_const = np.sqrt(60)
    # Iterate through minute_aggregate to create bridges
    for i in (range(0, len(minute_aggregate) - bridge_length_in_bars, bridge_length_in_bars)):
        overall_start_dt = minute_aggregate.index[i]
        overall_end_dt = minute_aggregate.index[i + bridge_length_in_bars]
        if i ==0:
            overall_start_val = minute_aggregate.iloc[i]['OPEN']
        else:
            overall_start_val = minute_aggregate.iloc[i]['CLOSE']
        overall_end_val_contestant1 = minute_aggregate.iloc[i + bridge_length_in_bars]['OPEN']
        overall_end_val_contestant2 = minute_aggregate.iloc[i]['CLOSE']

        
        if np.abs(overall_end_val_contestant1-overall_end_val_contestant2) >.05:
            overall_end_val = overall_end_val_contestant2  
        else: 
            overall_end_val = overall_end_val_contestant1
        # Select the segment of minute_aggregate data relevant for this bridge
        segment = minute_aggregate.iloc[i : i + bridge_length_in_bars + 1] # +1 to include end_value bar for sigma

        # Calculate local_sigma based on the segment's HIGH/LOW range
        # Ensure segment is not empty and has valid HIGH/LOW
        if not segment.empty and 'HIGH' in segment.columns and 'LOW' in segment.columns:
            # A common way to estimate local_sigma from OHLC
            local_sigma = (segment["HIGH"].max() - segment["LOW"].min()) / overall_start_val / (2 * np.sqrt(max(bridge_length_in_bars, 1)))
            # Add a small floor to sigma to prevent division by zero or overly flat bridges
            local_sigma = max(local_sigma, 1e-6) # Ensure sigma is positive
        else:
            local_sigma = 0.01 # Default sigma if segment data is bad
            
        overall_bridge_duration_us = (overall_end_dt - overall_start_dt).total_seconds() 
        
        if overall_bridge_duration_us <= 0:
            continue

        current_segment_start_dt = overall_start_dt
        current_segment_start_val = overall_start_val
    
        # Iterate through individual minute bars within the bridge_length_in_bars
        for j in range(bridge_length_in_bars):
            bar_for_segment = minute_aggregate.iloc[i + j]
            bar_high = bar_for_segment['HIGH']
            bar_low = bar_for_segment['LOW']

            # Calculate the end timestamp for this sub-segment (i.e., the end of the current minute bar)
            sub_segment_end_dt = minute_aggregate.index[i + j + 1] 
            
            sub_segment_duration_us = (sub_segment_end_dt - current_segment_start_dt).total_seconds()
            
            if sub_segment_duration_us <= 0:
                continue

            # Linear interpolation for target value across the overall bridge for this sub-segment's end
            linear_target_val = overall_start_val + \
                                (sub_segment_end_dt - overall_start_dt).total_seconds() / overall_bridge_duration_us * \
                                (overall_end_val - overall_start_val)
            
            # This is the target for the end of the small bridge segment
            intermediate_target_val = linear_target_val 

            segment_samples = []


            for _ in range(num_samples):
                sample_df = emulate_brownian_bridge_segment(
                    start_unix_s=int(current_segment_start_dt.timestamp() ),
                    end_unix_s=int(sub_segment_end_dt.timestamp() ),
                    start_value=current_segment_start_val,
                    end_value=        intermediate_target_val,
                    computational_density_per_unit=output_density_per_unit,
                    local_sigma=local_sigma * overall_end_val / bridge_normalization_const # Scale sigma to daily/annualized volatility if local_sigma is for a different period
                )
                if not sample_df.empty:
                    segment_samples.append(sample_df)

            if not segment_samples:
                # If no samples could be generated for this segment, fall back to linear interpolation
                # This could happen if durations are too small or other issues.
                fallback_timestamps = pd.to_datetime(np.linspace(current_segment_start_dt.timestamp(), sub_segment_end_dt.timestamp(), int(sub_segment_duration_us / output_density_per_unit), endpoint=True) * 1_000_000, unit='us')
                fallback_values = np.linspace(current_segment_start_val, intermediate_target_val, len(fallback_timestamps))
                emulated_segment_df = pd.DataFrame({'CLOSE': fallback_values}, index=fallback_timestamps)
                emulated_segment_df = emulated_segment_df[~emulated_segment_df.index.duplicated(keep='first')]
                brownian_bridge_segments.append(emulated_segment_df)
                current_segment_start_ts = sub_segment_end_dt
                current_segment_start_val = intermediate_target_val # Use target as next start for fallback
                continue # Skip to next sub-segment

            # Filter paths to find ones that respect the historical High and Low
            valid_paths = [
                path for path in segment_samples 
                if path['CLOSE'].max() <= bar_high and path['CLOSE'].min() >= bar_low
            ]

            # Select a path: prioritize valid paths, otherwise pick a random sample
            if valid_paths:
                emulated_segment_df = random.choice(valid_paths)
            else:
                # Failsafe: if no path was valid, pick a random one from all samples.
                # This means it might violate High/Low, but ensures a path is generated.
                emulated_segment_df = random.choice(segment_samples)
            
            brownian_bridge_segments.append(emulated_segment_df)
            
            # Update start for the next segment
            current_segment_start_dt = sub_segment_end_dt # The end of current is start of next
            if not emulated_segment_df.empty:
                current_segment_start_val = emulated_segment_df['CLOSE'].iloc[-1] # Use the actual end value of the chosen path

    if not brownian_bridge_segments:
        return pd.DataFrame(columns=['CLOSE'], index=pd.DatetimeIndex([]))

    # Concatenate all segments, sort by index, and remove duplicates
    brownian_bridge = pd.concat(brownian_bridge_segments).sort_index() # Index is already DatetimeIndex from segment
    brownian_bridge = brownian_bridge[~brownian_bridge.index.duplicated(keep='first')]
    return brownian_bridge

# after fitting  volume, apply to any given series given its minute volume aggregate
def apply_beta_interpolation(minute_agg_test: pd.DataFrame, beta_params_per_second: Dict[int, Dict[str, float]]) -> pd.Series:
    modeled_subsecond_volume_data = []
    np.random.seed(42) # Ensure reproducibility of random samples
    # print("minute_agg_test")
    # print(minute_agg_test.head())
    for minute_start, row in minute_agg_test.iterrows():
        total_minute_volume = row['VOLUME']
        
        if total_minute_volume == 0:
            for s in range(60):
                modeled_subsecond_volume_data.append({'timestamp': minute_start + pd.Timedelta(seconds=s), 'VOLUME': 0})
            continue

        sampled_proportions = []
        for s in range(60):
            alpha_s = beta_params_per_second[s]['alpha']
            beta_s = beta_params_per_second[s]['beta']
            
            if alpha_s > 0 and beta_s > 0: # Check for valid Beta parameters
                sampled_prop = beta.rvs(alpha_s, beta_s)
            else:
                sampled_prop = 1/60 # Fallback to uniform if parameters are invalid
            
            sampled_proportions.append(sampled_prop)
        
        sum_sampled_props = sum(sampled_proportions)
        if sum_sampled_props > 0:
            normalized_sampled_proportions = [p / sum_sampled_props for p in sampled_proportions]
        else:
            normalized_sampled_proportions = [1/60] * 60 # Fallback to uniform

        for i, prop in enumerate(normalized_sampled_proportions):
            volume_for_second = prop * total_minute_volume
            modeled_subsecond_volume_data.append({'timestamp': minute_start + pd.Timedelta(seconds=i), 'VOLUME': volume_for_second})
            
    modeled_subsecond_volume_df = pd.DataFrame(modeled_subsecond_volume_data).set_index('timestamp')
    modeled_subsecond_volume_series = modeled_subsecond_volume_df['VOLUME'].resample('1s').sum().fillna(0)

    return modeled_subsecond_volume_series



def fit_beta_distributions(subsecond_train: pd.DataFrame, ticker: str, config_filepath: str) -> Dict[int, Dict[str, float]]:
    # Default parameters for Beta distribution, biased towards 0
    default_alpha = 0.5 
    default_beta = 2.0  
    
    # Initialize the specific ticker's beta params with defaults or loaded values
    current_ticker_beta_params = {s: {'alpha': default_alpha, 'beta': default_beta} for s in range(60)}

    # --- Load the entire configuration file ---
    # This will hold all tickers' beta parameters, and other configurations
    full_config_data: Dict[str, Any] = {}
    if os.path.exists(config_filepath):
        try:
            with open(config_filepath, 'r') as f:
                full_config_data = json.load(f)
            print(f"Loaded existing configuration from {config_filepath}.")
            
            # If the ticker's data already exists in the config, load it
            if f"{ticker}" in full_config_data:
                loaded_ticker_params = full_config_data[f"{ticker}"]
                # Convert string keys back to int for the loaded ticker's params
                loaded_ticker_params = {int(k): v for k, v in loaded_ticker_params.items()}
                current_ticker_beta_params.update(loaded_ticker_params)
                print(f"Loaded existing Beta parameters for ticker '{ticker}' from config.")
            else:
                print(f"Ticker '{ticker}' not found in config. Will create new entry.")

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {config_filepath}. Starting with an empty config (might overwrite existing non-JSON content).")
        except Exception as e:
            print(f"Error loading configuration from {config_filepath}: {e}. Starting with an empty config.")
    else:
        print(f"Configuration file {config_filepath} not found. A new one will be created.")

    print("\n--- Training Beta Distribution Parameters ---")
    if not subsecond_train.empty:
        subsecond_train_temp = subsecond_train.copy() 
        subsecond_train_temp['minute_start'] = subsecond_train_temp.index.floor('1min')
        subsecond_train_temp['second_of_minute'] = (subsecond_train_temp.index - subsecond_train_temp['minute_start']).dt.seconds
        
        minute_second_volume = subsecond_train_temp.groupby(['minute_start', 'second_of_minute'])['VOLUME'].sum().unstack(fill_value=0)
        minute_total_volume_for_norm = minute_second_volume.sum(axis=1)

        valid_minutes_for_norm = minute_total_volume_for_norm[minute_total_volume_for_norm > 0].index
        minute_proportions = minute_second_volume.loc[valid_minutes_for_norm].divide(
                                minute_total_volume_for_norm.loc[valid_minutes_for_norm], axis=0).fillna(0)
        
        print(f"Attempting to fit Beta distributions for {len(minute_proportions)} training minutes...")
        for s in range(60): # Iterate through all 60 seconds
            if s in minute_proportions.columns: # Check if this second had data in any training minute
                proportions_for_second_s = minute_proportions[s].values
                proportions_for_fitting = proportions_for_second_s[(proportions_for_second_s > 1e-9) & (proportions_for_second_s < 1 - 1e-9)]
                
                if len(proportions_for_fitting) > 1: # Need at least 2 unique points for fit
                    try:
                        alpha_fit, beta_fit, _, _ = beta.fit(proportions_for_fitting, floc=0, fscale=1)
                        
                        # Apply biasing logic
                        final_alpha = max(alpha_fit, 0.1)  
                        final_beta = max(beta_fit, 1.5)    
                        
                        current_ticker_beta_params[s] = {'alpha': final_alpha, 'beta': final_beta}

                    except (RuntimeError, ValueError) as e:
                        print(f"Warning: Beta fit failed for second {s} ({type(e).__name__}: {e}). Retaining existing/default value.")
                        # current_ticker_beta_params[s] will retain its value (default or loaded from config)
                    except Exception as e:
                        print(f"Warning: Beta fit failed for second {s} (Unexpected Error: {e}). Retaining existing/default value.")
                        # current_ticker_beta_params[s] will retain its value
                else:
                    print(f"Not enough non-zero/non-one data points to fit Beta for second {s}. Retaining existing/default value.")
                    # current_ticker_beta_params[s] will retain its value
            else:
                print(f"Second {s} not found in training data columns. Retaining existing/default value.")
                # current_ticker_beta_params[s] will retain its value

        print("Finished fitting Beta parameters.")
    else:
        print(f"Warning: subsecond_train is empty. No Beta parameters fitted. Retaining existing/default Beta parameters for all seconds.")
    
    # --- Update the full configuration data and save back to file ---
    full_config_data[f"{ticker}"] = {str(k): v for k, v in current_ticker_beta_params.items()} # Convert keys to str for JSON
    
    try:
        # It's good practice to create the directory if it doesn't exist, even for single file.
        # This will operate on the parent directory of config_filepath.
        os.makedirs(os.path.dirname(config_filepath) or '.', exist_ok=True)
        
        with open(config_filepath, 'w') as f:
            json.dump(full_config_data, f, indent=4)
        print(f"Updated configuration file '{config_filepath}' with Beta parameters for ticker '{ticker}'.")
    except Exception as e:
        print(f"Error saving updated configuration to {config_filepath}: {e}")

    return current_ticker_beta_params
    

def plot_volume_modeling(
    ticker: str,
    subsecond_test: pd.DataFrame,
    interpolated_1s_close_series: pd.DataFrame,
    interpolated_1s_volume_series: pd.Series
):
    """
    Generates and saves a plot comparing actual, modeled, and input minute volumes.
    Args:
        ticker (str): The stock ticker.
        subsecond_test (pd.DataFrame): Actual subsecond data (ground truth for price and volume).
        minute_agg_test (pd.DataFrame): Minute aggregate data (input to model).
        modeled_volume_series (pd.Series): The 1-second modeled volume series.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Ground Truth Price (from the subsecond test set)
    ax.plot(subsecond_test.index, subsecond_test['CLOSE'], label='Actual Subsecond Price', color='black', linewidth=0.2)
    ax.plot(interpolated_1s_close_series.index, interpolated_1s_close_series['CLOSE'], label='Interpolated Aggregate Price (Close)', color='dodgerblue', linewidth=0.8, zorder=10)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    fig.suptitle(f'Volume Modeling for {ticker} (Beta Distribution Sampling Volume, BBMM Price Interpolation)', fontsize=16, fontweight='bold')

    ax2 = ax.twinx()

    # Prepare actual and input minute volumes for consistent plotting
    actual_subsecond_volume_resampled = subsecond_test['VOLUME'].resample('1s').sum().fillna(0)

    # Align all volume series for consistent plotting
    all_volume_indices = actual_subsecond_volume_resampled.index \
                            .union(interpolated_1s_volume_series.index) 

    actual_subsecond_volume_resampled = actual_subsecond_volume_resampled.reindex(all_volume_indices, fill_value=0)
    modeled_volume_series_aligned = interpolated_1s_volume_series.reindex(all_volume_indices, fill_value=0)

    # Plot Actual Subsecond Volume (Ground Truth)
    ax2.bar(actual_subsecond_volume_resampled.index, actual_subsecond_volume_resampled,
            label='Actual Subsecond Volume (1s)', color='grey', alpha=0.6, width=pd.Timedelta(seconds=0.8))

    # Plot Modeled Real-time Volume (from aggregate data, sampled from Beta)
    ax2.bar(modeled_volume_series_aligned.index, modeled_volume_series_aligned,
            label='Modeled Real-time Volume (1s from Agg - Beta Sampled)', color='red', alpha=0.3, width=pd.Timedelta(seconds=0.6))
    


    ax2.set_ylabel('Volume', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    max_volume = max(actual_subsecond_volume_resampled.max(),
                     modeled_volume_series_aligned.max(),
                     interpolated_1s_volume_series.max())
    ax2.set_ylim(0, max_volume * 1.5)

    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc='upper left')

    plt.gcf().autofmt_xdate()
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"beta_volume_emulation_{ticker}.png")
    plt.show()

def compare_series(gt_series, emu_series, title):
    if gt_series.empty or emu_series.empty:
        print(f"Skipping comparison for {title} due to empty series.")
        return

    aligned_gt, aligned_emulated = gt_series.align(emu_series, join='inner')
    if aligned_gt.empty:
        print(f"No overlapping data for comparison for {title}.")
        return

    # Ensure numeric types and handle potential NaNs from alignment
    aligned_gt = pd.to_numeric(aligned_gt, errors='coerce').dropna()
    aligned_emulated = pd.to_numeric(aligned_emulated, errors='coerce').dropna()

    # Re-align after dropping NaNs, if necessary, to ensure exact same indices
    final_common_index = aligned_gt.index.intersection(aligned_emulated.index)
    aligned_gt = aligned_gt.loc[final_common_index]
    aligned_emulated = aligned_emulated.loc[final_common_index]

    if aligned_gt.empty: # Check again after final cleaning
        print(f"No valid overlapping data points after cleaning. Metrics cannot be calculated.")
        return

    # Randomly sample for MAE/RMSE calculation (as per your current code)
    num_samples = max(1, len(aligned_gt) // 2)
    # Ensure num_samples doesn't exceed available data points
    if num_samples > len(aligned_gt):
        num_samples = len(aligned_gt)
    sampled_indices = random.sample(list(aligned_gt.index), num_samples)

    sampled_gt = aligned_gt.loc[sampled_indices]
    sampled_emu = aligned_emulated.loc[sampled_indices]

    # Calculate MAE
    mae = np.mean(np.abs(sampled_gt - sampled_emu))
    print(f"Comparison for {title}: MAE = {mae:.6f}")

    # Calculate RMSE
    mse = np.mean((sampled_gt - sampled_emu)**2)
    rmse = np.sqrt(mse)
    print(f"Comparison for {title}: RMSE = {rmse:.6f}")

    print("--------------------------------" + "-" * len(title))





