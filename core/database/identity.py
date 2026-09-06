import sys
import os 

from datetime import timedelta,  datetime
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import DATABASE_URL, engine, ticker_to_identity_map, figi_hash_to_identity_map


def _prepare_global_identity(
    tickers_list: list, 
    end: datetime, 
    reconstruction_heartbeat_freq_months: int
) -> pd.DataFrame:
    """Fetches mappings and enforces chronological bounding logic on historical lifespans."""
    formatted_end = end.strftime('%Y-%m-%d %H:%M:%S')
    global_tickers = ticker_to_identity_map(tickers_list, lookback_date=formatted_end)
    if global_tickers.empty:
        return pd.DataFrame()
        
    global_hashes = global_tickers['composite_figi_hash'].dropna().unique().tolist()
    if not global_hashes:
        return pd.DataFrame()
        
    global_identity = figi_hash_to_identity_map(global_hashes, lookback_date=formatted_end, drop_unknown=True)
    if global_identity.empty:
        return pd.DataFrame()
        
    global_identity['earliest'] = pd.to_datetime(global_identity['earliest'])
    global_identity['latest'] = pd.to_datetime(global_identity['latest'])

    # Apply corporate lifespan adjustments and heartbeat padding
    global_identity['latest'] = global_identity['latest'] + pd.DateOffset(months=reconstruction_heartbeat_freq_months)
    global_identity['latest'] = global_identity['latest'] - pd.Timedelta(days=1)
    
    return global_identity
