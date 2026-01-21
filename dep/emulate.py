import json
from models import Company, Base
import sys
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from db import engine
from typing import List
import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm

est_timezone = pytz.timezone('America/New_York')

from emulate_helper import process_minute_quotes, emulate_brownian_bridge,  upsert_historical_emulate_quotes_via_copy_batched
import holidays 
import multiprocessing
NYSE_HOLIDAYS = holidays.NYSE() 

from sqlalchemy import text

def fast_slow_optimal_facilities_outlier_detection_algorithm(
    df: pd.DataFrame,
    cols_to_clean: List[str],
    fastperiod: int = 22,
    fast_thresh: float = .006,
    slowperiod: int = 59,
    slow_thresh: float = .06
):
    '''
    Primary goal is to increase data integrity to prior knowledge of original data distributions, adapted to Quantitative Data
    This function removes outliers from specified price columns using a combination of
    fast and slow rolling median filters. 
    '''
    cleaned_df = df.copy()
    original_df_for_metrics = df.copy()


    outlier_counts = {}
    nan_fill_rates = {}
    smoothness_scores = {}

    for col in cols_to_clean:
        # Calculate the rolling median for the current column using the fast period
        fast_rolling_median = cleaned_df[col].rolling(
            window=fastperiod, min_periods=1, center=True
        ).median()

        # Define the UB LB for acceptable values for the fast period
        fast_upper_bound = fast_rolling_median * (1 + fast_thresh)
        fast_lower_bound = fast_rolling_median * (1 - fast_thresh)

        # Create a mask for outliers based on the fast period: values outside the bounds
        fast_outlier_mask = (cleaned_df[col] > fast_upper_bound) | \
                            (cleaned_df[col] < fast_lower_bound)

        # Calculate the rolling median for the current column using the slow period
        slow_rolling_median = cleaned_df[col].rolling(
            window=slowperiod, min_periods=1, center=True
        ).median()

        # Define the UB LB for acceptable values for the slow period
        slow_upper_bound = slow_rolling_median * (1 + slow_thresh)
        slow_lower_bound = slow_rolling_median * (1 - slow_thresh)

        # Create a mask for outliers based on the slow period: values outside the bounds
        slow_outlier_mask = (cleaned_df[col] > slow_upper_bound) | \
                            (cleaned_df[col] < slow_lower_bound)

        # A point is considered an outlier if it's flagged by EITHER the fast OR the slow filter
        combined_outlier_mask = fast_outlier_mask | slow_outlier_mask

        # Count outliers removed for the current column
        num_outliers_removed = combined_outlier_mask.sum()
        outlier_counts[f'{col}_outliers_removed'] = num_outliers_removed

        # if num_outliers_removed > 0:
        #     print(f"Removed {num_outliers_removed} outliers from column '{col}' using combined fast/slow filters.")

        # Replace outliers with NaN in the cleaned DataFrame for the current column
        cleaned_df[col] = cleaned_df[col].mask(combined_outlier_mask)

        # --- Calculate Data Loss (NaN Fill Rate) ---
        original_non_nan_count = original_df_for_metrics[col].dropna().count()
        nan_after_cleaning_count = cleaned_df[col].isna().sum()
        if original_non_nan_count > 0:
            nan_fill_rates[f'{col}_nan_fill_rate'] = nan_after_cleaning_count / original_non_nan_count
        else:
            nan_fill_rates[f'{col}_nan_fill_rate'] = 0.0 # No original data, so no loss

        # Calculate Local Smoothness Score (Mean Absolute Percentage Change - MAPC) ---
        cleaned_col_data = cleaned_df[col].dropna()
        if len(cleaned_col_data) > 1:
            percentage_changes = cleaned_col_data.pct_change().abs().dropna()
            smoothness_scores[f'{col}_local_smoothness_score'] = percentage_changes.mean()
        else:
            smoothness_scores[f'{col}_local_smoothness_score'] = np.inf # Use infinity to denote very bad smoothness

    # --- Calculate OHLC Consistency Violations (after all columns are cleaned) ---
    ohlc_consistency_violations = 0
    required_ohlc_cols = ["OPEN", "HIGH", "LOW", "CLOSE"]
    if all(col in cleaned_df.columns for col in required_ohlc_cols):
        # Drop rows where any of the OHLC columns are NaN for this check
        temp_ohlc_df = cleaned_df[required_ohlc_cols].dropna()

        # Count violations where LOW is greater than OPEN or CLOSE
        ohlc_consistency_violations += (temp_ohlc_df["LOW"] > temp_ohlc_df["OPEN"]).sum()
        ohlc_consistency_violations += (temp_ohlc_df["LOW"] > temp_ohlc_df["CLOSE"]).sum()

        # Count violations where HIGH is less than OPEN or CLOSE
        ohlc_consistency_violations += (temp_ohlc_df["HIGH"] < temp_ohlc_df["OPEN"]).sum()
        ohlc_consistency_violations += (temp_ohlc_df["HIGH"] < temp_ohlc_df["CLOSE"]).sum()

    # --- Consolidate All Scores ---
    all_scores = {
        'total_outliers_removed': sum(outlier_counts.values()),
        **outlier_counts, # Individual column outlier counts
        **nan_fill_rates, # Individual column NaN fill rates
        **smoothness_scores, # Individual column smoothness scores
        'ohlc_consistency_violations': ohlc_consistency_violations
    }

    return cleaned_df, all_scores


def init_schema(drop_train_item: bool = True,  drop_test_item: bool = True, drop_emulate: bool = True):
    if drop_train_item:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS train_item"))
            conn.commit()
        print("Dropped existing train_item table")
    if drop_test_item:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS test_item"))
            conn.commit()
        print("Dropped existing test_item table")
    if drop_emulate:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS historical_emulate_quotes"))
            conn.commit()
        print("Dropped existing test_item table")

    Base.metadata.create_all(bind=engine)
    print("All tables ensured")


def string_to_unix_s(
    time_str: str,
    timezone_str: str,
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
) -> int:
    tz = pytz.timezone(timezone_str)
    localized_dt = tz.localize(datetime.strptime(time_str, datetime_format))
    utc_dt = localized_dt.astimezone(pytz.utc)
    return int(pd.Timestamp(utc_dt).timestamp())


def interpolate_second_from_minute(
    ticker: str,
    intended_hours_ts: List[int]
):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        ticker_comp = session.query(Company).filter(Company.ticker == ticker).first()

    db_query_start_ts_us = intended_hours_ts[0]
    db_query_end_ts_us = intended_hours_ts[-1]


    minute_aggregate = process_minute_quotes(ticker_comp, [db_query_start_ts_us, db_query_end_ts_us])
    minute_aggregate.index = pd.to_datetime(minute_aggregate.index, unit='s')

    minute_aggregate, _ = fast_slow_optimal_facilities_outlier_detection_algorithm(minute_aggregate, ["OPEN", "HIGH", "LOW", "CLOSE"], fastperiod=25, fast_thresh=.005, slowperiod=59, slow_thresh=.05)
    minute_aggregate['OPEN'] = minute_aggregate['OPEN'].ffill().bfill()
    minute_aggregate['HIGH'] = minute_aggregate['HIGH'].ffill().bfill()
    minute_aggregate['LOW'] = minute_aggregate['LOW'].ffill().bfill()
    minute_aggregate['CLOSE'] = minute_aggregate['CLOSE'].ffill().bfill()
    
    minute_aggregate['VOLUME'] = minute_aggregate['VOLUME'].fillna(0)


 
    emulated_brownian_bridge_length_1 = emulate_brownian_bridge(minute_aggregate=minute_aggregate, output_density_per_unit=1, bridge_length_in_bars=1) 
    emulated_brownian_bridge_length_1['VOLUME'] = 0
    


    merged_data = emulated_brownian_bridge_length_1.copy()

    merged_data['CLOSE'] = merged_data['CLOSE'].ffill().bfill().round(3)
    merged_data['VOLUME'] = merged_data['VOLUME'].fillna(0)
    merged_data.index = merged_data.index.astype(np.int64) // 10**9
    merged_data.index.name = "unix_s"

    return merged_data



def build_and_persist_inner(ticker: str,
                            start_idx: int,
                            end_idx: int):
    linspace = list(range(start_idx, end_idx + 1, 1))

    interpolated_second_data = interpolate_second_from_minute(
        ticker,
        linspace
    )


    if interpolated_second_data.empty:
        return


    upsert_historical_emulate_quotes_via_copy_batched(
        interpolated_second_data, ticker, 50
    )


def worker(args):
    ticker, start_idx, end_idx = args
    build_and_persist_inner(ticker, start_idx, end_idx)

def build_and_persist(
    start_unix_s: int,
    end_unix_s: int,

    tuning_companies: List[str] = [],
    output_filepath: str="exe_params.json",
    hour_s: int = 28800
):

    Base.metadata.create_all(bind=engine)

    set_of_batches = []
    batch_start = start_unix_s
    while batch_start < end_unix_s:
        batch_end = min(batch_start + hour_s - 1, end_unix_s)
        set_of_batches.append((batch_start, batch_end))
        batch_start += hour_s


    for ticker in tuning_companies:
        print(f"Doing ticker {ticker}")
        args_list = [(ticker, start_idx, end_idx) for start_idx, end_idx in set_of_batches]

        with multiprocessing.Pool(processes=8) as pool:
            list(tqdm(pool.imap_unordered(worker, args_list), total=len(args_list)))

def prepare_data(
    timezone_str: str, 
    train_start: str, 
    train_end: str, 

 

    tuning_companies: list, 
    output_filepath: str, 
    batch_size: int = 409600
):

    print("Build and Persist Train Items")
    train_start_utc_s = string_to_unix_s(train_start, timezone_str) 
    train_end_utc_s = string_to_unix_s(train_end, timezone_str) 

    # Check for successful conversions
    if None in [train_start_utc_s, train_end_utc_s]:
        print("Skipping train data preparation due to conversion errors.")
        return
    
    build_and_persist(
        train_start_utc_s,
        train_end_utc_s,
  
        tuning_companies,
        output_filepath,

        batch_size
    )





def main():
    config_filepath = sys.argv[1]

    with open(config_filepath, 'r', encoding='utf-8') as f:
        known = json.load(f)
        tickers = known["ticker"]
        timezone_str = known["timezone_str"]
        train_start = known["train_start"]
        train_end = known["train_end"]

    

 
    print("tickers",  tickers)

    init_schema(drop_train_item=False, drop_test_item=False, drop_emulate=True)

    # # # persist all slices
    prepare_data(       
        timezone_str, 
        train_start,
        train_end,
        tickers, 
        config_filepath, 
        )

if __name__ == '__main__':
    main()

