import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone, time, timedelta
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from models import Company, HistoricalEmulateQuote, TrainItem,  Base
import random
from indicators import LOCAL_FUNCS, parse_param_to_inputs
from db import engine, DATABASE_URL
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from bisect import bisect_left
from urllib.parse import urlencode, parse_qs
from typing import Dict, List, Tuple, Any
import sys
import json
import holidays 
NYSE_HOLIDAYS = holidays.NYSE() 
from pandas_market_calendars import get_calendar
from emulate_helper import solve_astb, solve_segment, modified_astb
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from multiprocessing import Pool, cpu_count


def tz_to_unix(dt_obj: datetime) -> int:
    if isinstance(dt_obj, pd.Timestamp):
        return dt_obj.value
    else:

        if dt_obj.tzinfo is None:

            dt_obj = pytz.timezone('US/Eastern').localize(dt_obj)
            
        return pd.Timestamp(dt_obj).value

def init_schema(drop_train_item: bool = True):
    if drop_train_item:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS train_item"))
            conn.commit()
        print("Dropped existing train_item table")
    Base.metadata.create_all(bind=engine)
    print("All tables ensured")

    
def process_company(comp: Company, after_hours_ts: List[int]):
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        rows = (
            session.query(
                HistoricalEmulateQuote.time_entry_ts,
                HistoricalEmulateQuote.close_price,
                HistoricalEmulateQuote.volume
            )
            .filter_by(company_id=comp.id)
            .filter(HistoricalEmulateQuote.time_entry_ts.between(after_hours_ts[0], after_hours_ts[-1]))
            .order_by(HistoricalEmulateQuote.time_entry_ts)
            .all()
        )


    company_df = pd.DataFrame(rows, columns=["uts", "CLOSE", "VOLUME"]).set_index("uts")

    return company_df.reindex(after_hours_ts)


def unix_to_dt(unix_us: int) -> pd.Timestamp:
    return pd.to_datetime(unix_us, unit='s')

def dt_to_unix_s(dt: pd.Timestamp) -> int:
    return int(dt.timestamp())

def unix_in_market_hours(unix_s: int, timezone='US/Eastern') -> bool:
    dt_obj = unix_to_dt(unix_s).tz_localize(pytz.utc).tz_convert(timezone)

    # 2. Check for Weekends (Saturday = 5, Sunday = 6)
    if dt_obj.weekday() >= 5: # Monday is 0, Sunday is 6
        return False

    # 3. Check for NYSE Holidays
    # The 'holidays' library needs a date object (year, month, day)
    if dt_obj.date() in NYSE_HOLIDAYS:
        return False

    # 4. Check for Time of Day (9:30 AM to 4:00 PM)
    market_open = time(9, 30, 0)  # 9:30:00 AM
    market_close = time(16, 0, 0) # 4:00:00 PM

    return market_open <= dt_obj.time() < market_close
    

def get_cv_forecast_segment(df: pd.DataFrame, current_unix_s: int, forecast_seconds: int) -> Tuple[pd.DataFrame, int]:
    end_dt = current_unix_s + (forecast_seconds )
    segment = df.loc[current_unix_s :end_dt].dropna(subset=['CLOSE', 'VOLUME'])

    seg_length = (end_dt -current_unix_s  ) 
    
    return segment, seg_length

def get_cv_context_segment(df: pd.DataFrame, current_unix_s: int, context_seconds: int) -> pd.DataFrame:
    start_dt = current_unix_s - (context_seconds)
    segment = df.loc[start_dt:current_unix_s].dropna(subset=['CLOSE', 'VOLUME'])
    seg_length = (current_unix_s - start_dt ) 
    return segment, seg_length

def close_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], index=pd.DatetimeIndex([]))

    df.index = pd.to_datetime(df.index, unit='s')
    df = df.sort_index()
    # Resample to the specified minute_step_seconds
    ohlc = df['CLOSE'].resample(f'1min', origin='start').ohlc()
    volume = df['VOLUME'].resample(f'1min', origin='start').sum()
    ohlcv = pd.DataFrame({
        'OPEN': ohlc['open'], 
        'HIGH': ohlc['high'], 
        'LOW': ohlc['low'],
        'CLOSE': ohlc['close'], 
        'VOLUME': volume
    })
    
    # Drop any rows where OHLCV data could not be formed (e.g., no data in a bucket)
    ohlcv = ohlcv.dropna()
    
    return ohlcv

def _execute_upsert_chunk(chunk: List[Dict], engine=None):

    if not chunk:
        print("chunk empty")
        return
        
    Session = sessionmaker(bind=engine)
    with Session() as session:
        

        table    = TrainItem.__table__ 

        stmt = pg_insert(table).values(chunk)
    
        set={
                'tensor': stmt.excluded.tensor,    
                # timeline 
                'a_time': stmt.excluded.a_time, # a_time  # 1752414329000000000 aka id
                's_time': stmt.excluded.s_time, # s_time  # 1752414914000000000
                't_time': stmt.excluded.t_time, # t_time # 1752415257000000000
                'b_time': stmt.excluded.b_time, # b_time # 1752417929000000000
                'a_value': stmt.excluded.a_value,
                's_value': stmt.excluded.s_value,      # 12.622
                't_value' : stmt.excluded.t_value,     # 12.744
        
                'b_value': stmt.excluded.b_value,
            }
    
        
        
        update_stmt = stmt.on_conflict_do_update(
            index_elements=['time_entry_ts'],
            set_=set
        )
        session.execute(update_stmt)
        session.commit()

def build_single_ticker(
    start_dt: datetime,
    end_dt: datetime,
    padded_start_dt: datetime,
    padded_end_dt: datetime,
    forecast_depth: int = 30, # In minutes
    context_depth: int = 30,  # In minutes
    burn_in: int = 400,
    params: List[str] = [], # Retained for strict structure
    ticker: str = "", 
    engine=None
):
    filtered_params = [p for p in params if ticker in p]
    filtered_timestamps_s_range = list(range(int(start_dt.timestamp()), int(end_dt.timestamp()) + 1, 1))

    if not unix_in_market_hours(filtered_timestamps_s_range[0]) and not unix_in_market_hours(filtered_timestamps_s_range[-1]):
        return None
    padded_timestamps_s_range = list(range(int(padded_start_dt.timestamp()), int(end_dt.timestamp()) + 1, 1))

    Session = sessionmaker(bind=engine)
    company_data_for_ticker: pd.DataFrame = pd.DataFrame() 

    with Session() as session:
        comp  = session.query(Company).filter(Company.ticker == ticker).first()
        company_data_for_ticker = process_company(comp, padded_timestamps_s_range)

    context_s_duration = context_depth * 60  
    forecast_s_duration = forecast_depth * 60 

    start_of_loaded_data_s  = company_data_for_ticker.index[0]
    end_of_loaded_data_s = company_data_for_ticker.index[-1]

    start_analysis_window_s = start_of_loaded_data_s - context_s_duration 
    end_analysis_window_s = end_of_loaded_data_s + forecast_s_duration

    df_raw_seconds = company_data_for_ticker.sort_index()[~company_data_for_ticker.index.duplicated(keep='first')]

    raw_segment_for_context, _ = get_cv_context_segment(df_raw_seconds, end_analysis_window_s, (end_analysis_window_s - start_analysis_window_s))
    if raw_segment_for_context.empty :
        print("empty seg1")
        return None
    
    ohlcv_minute_data = close_to_ohlcv(raw_segment_for_context)

    indicator_df = {}
    for p_str in filtered_params:
        if "CURRENT_PRICE" in p_str:
            indicator_df[p_str]  =  raw_segment_for_context["CLOSE"]
        else: 
            input_dict, key = parse_param_to_inputs(p_str)
            fn_name = key[0]
            fn = LOCAL_FUNCS.get(fn_name)
            indicator_df[p_str]  =  fn(ohlcv_minute_data, input_dict, key).ffill().bfill().round(3)

    forecast_burn_in_minutes = 3


    target_df = {}

    # Calculate the Simple Moving Average (SMA) bands.
    smooth_fn = LOCAL_FUNCS.get("SMA")
    minute_close_prices = ohlcv_minute_data['CLOSE'] # <--- ADD THIS
    ub_input_dict, ub_key = parse_param_to_inputs(f"ticker={ticker}&function=SMA&time_period={forecast_burn_in_minutes}&series_type=HIGH")
    mb_input_dict, mb_key = parse_param_to_inputs(f"ticker={ticker}&function=SMA&time_period={forecast_burn_in_minutes}&series_type=CLOSE")
    lb_input_dict, lb_key = parse_param_to_inputs(f"ticker={ticker}&function=SMA&time_period={forecast_burn_in_minutes}&series_type=LOW")
    # Upper Band (UB), Middle Band (MB), and Lower Band (LB)
    target_df['UB'] = smooth_fn(ohlcv_minute_data, ub_input_dict, ub_key)
    target_df['MB'] = smooth_fn(ohlcv_minute_data, mb_input_dict, mb_key)
    target_df['LB'] = smooth_fn(ohlcv_minute_data, lb_input_dict, lb_key)



    valid_current_t_s = [t for t in filtered_timestamps_s_range if unix_in_market_hours(t)]
    print(f"DEBUG: {ticker} {len(valid_current_t_s)} item  | start: {start_analysis_window_s} {unix_to_dt(start_analysis_window_s)} |  end: {end_analysis_window_s} {unix_to_dt(end_analysis_window_s)} | ")
    if not valid_current_t_s:
        return None
        
    output_dict_by_timestamp = {}
    # Process each valid timestamp
    for current_t_s in valid_current_t_s:
        current_t_dt_s = pd.to_datetime(current_t_s, unit='s')
        current_t_dt_floor = pd.to_datetime(current_t_s, unit='s').floor('min')
        current_t_dt_ceil = pd.to_datetime(current_t_s, unit='s').ceil('min')
        end_t_dt_ceil = pd.to_datetime(current_t_s + forecast_depth * 60 + 60, unit='s').ceil('min')

        if raw_segment_for_context.empty or len(raw_segment_for_context.index) < context_s_duration:
            continue 
        try:
            tensor = {
                k: v.loc[(current_t_dt_s if "CURRENT_PRICE" in k else current_t_dt_floor)].tolist()
                for k, v in indicator_df.items()
            }
        except KeyError:
            continue

        if current_t_s not in output_dict_by_timestamp:
            output_dict_by_timestamp[current_t_s] = {}
        output_dict_by_timestamp[current_t_s]['tensor'] = tensor

     

        astb_results = modified_astb(
            raw_segment_for_context['CLOSE'][current_t_dt_floor:end_t_dt_ceil], 
            target_df['UB'][current_t_dt_ceil:end_t_dt_ceil], 
            target_df['MB'][current_t_dt_ceil:end_t_dt_ceil], 
            target_df['LB'][current_t_dt_ceil:end_t_dt_ceil], 
            current_t_s, 
            current_t_s+ forecast_depth * 60
        )

        # 6. Assign the calculated forecast results to the output dictionary.
        if astb_results:
            for key, value in astb_results.items():
                output_dict_by_timestamp[current_t_s][key] = value

    return output_dict_by_timestamp



# Helper function to wrap build_single_ticker for use with multiprocessing.Pool
def _build_single_ticker_wrapper(args):
    worker_engine = create_engine(DATABASE_URL, poolclass=NullPool)
    
    # Call the original function with the new engine
    return build_single_ticker(*args, engine=worker_engine)


def _execute_upsert_chunk_wrapper(args):

    worker_engine = create_engine(DATABASE_URL, poolclass=NullPool)
    
    # Call the original function with the new engine
    return _execute_upsert_chunk(*args, engine=worker_engine)


def batch_and_persist(
    start_dt: datetime,
    end_dt: datetime,
    forecast_depth: int = 30, # In minutes
    context_depth: int = 30,  # In minutes
    burn_in: int = 400,
    params: List[str] = [], # Retained for strict structure
    tickers: str = "",
    batch_size: int =200
):
    current_processing_start = start_dt
    
    batch_data_retrieval_ranges = [] 
    print("start_dt", start_dt)
    print("end_dt", end_dt)
    while current_processing_start < end_dt:
        current_processing_end = current_processing_start + timedelta(minutes=batch_size)
        
        if current_processing_end > end_dt:
            current_processing_end = end_dt

        padded_start_dt = max(start_dt, current_processing_start - timedelta(minutes=burn_in))
        
        padded_end_dt = min(end_dt, current_processing_end + timedelta(minutes=forecast_depth))
        batch_data_retrieval_ranges.append(
            (padded_start_dt, padded_end_dt, current_processing_start, current_processing_end)
        )
        
        current_processing_start = current_processing_end
    if not batch_data_retrieval_ranges and start_dt <= end_dt:
        padded_start_dt = max(start_dt, start_dt - timedelta(minutes=burn_in))
        padded_end_dt = min(end_dt, end_dt + timedelta(minutes=forecast_depth))
        batch_data_retrieval_ranges = [(padded_start_dt, padded_end_dt, start_dt, end_dt)]

    # This is the main multiprocessing block
    # Use a pool to parallelize the inner loop
    with Pool(12) as pool:
        # Use imap_unordered for a more efficient way to process the results
        # Iterate over the results to ensure the main process waits for workers to complete
        for _ in pool.imap_unordered(_process_batch_and_persist, 
                             [(batch_range, tickers, forecast_depth, context_depth, burn_in, params) 
                              for batch_range in batch_data_retrieval_ranges]):
            pass # The loop is just to wait for the workers to finish
        
# This is the new inner function that contains the logic you selected.
def _process_batch_and_persist(args):
    """
    Processes a single batch of tickers and persists the data.
    This function is designed to be called by a multiprocessing pool.
    """
    (padded_data_start, padded_data_end, actual_process_start, actual_process_end), tickers, forecast_depth, context_depth, burn_in, params = args

    upsert_items = {}
    # Prepare arguments for each ticker to be run in a separate process
    ticker_args = [
        (
            actual_process_start, 
            actual_process_end,
            padded_data_start,
            padded_data_end,
            forecast_depth,
            context_depth,
            burn_in, 
            params, 
            ticker, 
        
        ) for ticker in tickers
    ]

    results = []
    for args in ticker_args:
        results.append(_build_single_ticker_wrapper(args))
    
    # The results list will contain the output dictionary from each call
    for output_dict_by_timestamp in results:
        if output_dict_by_timestamp:
            for ts_s, data_for_ts in output_dict_by_timestamp.items():
                if ts_s not in upsert_items:
                    upsert_items[ts_s] = {
                        "time_entry_ts": int(ts_s)
                    }
         
                try:
                    ticker = next(iter(data_for_ts['tensor'].keys())).split('&')[0].split('=')[1]
                except (StopIteration, KeyError, IndexError):
                    print("Could not determine ticker from output, skipping.")
                    continue
                
            
                
                for feature_key, feature_val in data_for_ts.items():
                    if feature_key == "tensor":
                        if feature_key not in upsert_items[ts_s]:
                            upsert_items[ts_s]["tensor"] = {}
                        for param_key, param_val in feature_val.items():
                            # Fix 1: Handle NaN values for the 'tensor' feature
                            
                            processed_val = None
                            if isinstance(param_val, (np.float32, np.float64)):
                                if np.isnan(param_val):
                                    processed_val = None
                                else:
                                    processed_val = float(param_val)
                            else:
                                processed_val = param_val

                            upsert_items[ts_s]["tensor"][param_key] = processed_val

                    elif feature_key in ['a_time', 'b_time', 's_time', 't_time'] :
                        # Fix 2: Handle NaN values for 'a_time' and 'b_time'
                          # Always handle as float (or None for NaN)
                        if isinstance(feature_val, (np.float64, np.float32)):
                            processed_val = None if np.isnan(feature_val) else float(feature_val)
                        elif isinstance(feature_val, (np.int64, np.int32)):
                            processed_val = int(feature_val)
                        elif isinstance(feature_val, list):
                            processed_val = [
                                None if (isinstance(x, (np.float64, np.float32)) and np.isnan(x))
                                else int(x) if isinstance(x, (np.integer, float, np.floating, np.integer))
                                else x
                                for x in feature_val
                            ]
                        else:
                            try:
                                processed_val = int(feature_val)
                            except Exception:
                                processed_val = feature_val

                        if feature_key not in upsert_items[ts_s]:
                            upsert_items[ts_s][feature_key] = {}
                        upsert_items[ts_s][feature_key][str(ticker)] = processed_val
                    else :
                        # Always handle as float (or None for NaN)
                        if isinstance(feature_val, (np.float64, np.float32)):
                            processed_val = None if np.isnan(feature_val) else float(feature_val)
                        elif isinstance(feature_val, (np.int64, np.int32)):
                            processed_val = float(feature_val)
                        elif isinstance(feature_val, list):
                            processed_val = [
                                None if (isinstance(x, (np.float64, np.float32)) and np.isnan(x))
                                else float(x) if isinstance(x, (np.integer, float, np.floating, np.integer))
                                else x
                                for x in feature_val
                            ]
                        else:
                            try:
                                processed_val = float(feature_val)
                            except Exception:
                                processed_val = feature_val

                        if feature_key not in upsert_items[ts_s]:
                            upsert_items[ts_s][feature_key] = {}
                        upsert_items[ts_s][feature_key][str(ticker)] = processed_val

    final_upsert_chunk = list(upsert_items.values())
    if final_upsert_chunk:
        # print(final_upsert_chunk[1])
        chunk_args = [final_upsert_chunk]

        _execute_upsert_chunk_wrapper(chunk_args)



def prepare_data(
        
    timezone_str: str, 
    train_start: str, 
    train_end: str, 
    forecast_depth: int,
    context_depth: int,
    burn_in:int,
    params: list, 
    tuning_companies: list
):
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    tz = pytz.timezone(timezone_str)
    train_start_dt = tz.localize(datetime.strptime(train_start, DATETIME_FORMAT))
    train_end_dt = tz.localize(datetime.strptime(train_end, DATETIME_FORMAT))

    print("Build and Persist Train Items")
    batch_and_persist(
        train_start_dt, train_end_dt,     
        forecast_depth,
        context_depth, 
        burn_in, 
        params, 
        tuning_companies
    )



def main():
    config_filepath = sys.argv[1]

    with open(config_filepath, 'r', encoding='utf-8') as f:
        known = json.load(f)
        tickers = known["ticker"]
        timezone_str = known["timezone_str"]
        train_start = known["train_start"]
        train_end = known["train_end"]
        forecast_depth = known["forecast_depth"]
        burn_in = known["burn_in"]
        context_depth = known["context_depth"]
        params = known["params"]
        drop_train_item = known["drop_train_item"]




    drop_train_item_arg  = True if drop_train_item == "True" else False

    print("tickers",  tickers)
    init_schema(drop_train_item=drop_train_item_arg)

    # # # persist all slices
    prepare_data(       
        timezone_str, 
        train_start,
        train_end,
        forecast_depth,
        context_depth,
        burn_in,
        params,
        tickers
        )

if __name__ == '__main__':
    main()

