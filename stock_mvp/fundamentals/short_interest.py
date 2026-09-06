import requests
import os 
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy.orm import relationship, declarative_base
from core.db import Base

class ShortInterest(Base):
    __tablename__ = 'short_interest'
    
    id = Column(BigInteger, primary_key=True)
    ticker_hash = Column(BigInteger, index=True, nullable=False)
    
    # Core Data
    settlement_date = Column(Date, nullable=False, index=True)
    short_interest = Column(BigInteger)
    avg_daily_volume = Column(BigInteger)
    days_to_cover = Column(Numeric(10, 4)) # Changed from Float to Numeric
    
    # --- Standardized Metadata Block ---
    # Now congruent with IncomeStatement, BalanceSheet, and CashFlow
    ticker = Column(String(20)) 
    __table_args__ = (
        UniqueConstraint('ticker_hash', 'settlement_date', name='uq_short_interest_identity'),
    )

def to_str(val):
    if val is None:
        return 'None' # Matches the 'NULL None' in your COPY command
    return str(val)

def fetch_and_map_short_interest_batch(ticker_to_hash_map, api_key, start_date, end_date):

    url = "https://api.massive.com/stocks/v1/short-interest"
    tickers_list = list(ticker_to_hash_map.keys())
    if not tickers_list:
        return []
    
    params = {
        # assume tickers is sorted
        "ticker.any_of": ",".join(tickers_list),
        "limit": 50000,
        "settlement_date.gte": start_date,
        "settlement_date.lte": end_date,
        "sort": "settlement_date.desc",
        "apiKey": api_key
    }
    
    raw_results = []
    current_url = url
    current_params = params

    while current_url:
        try:
            # Note: params are only sent on the FIRST call. 
            # Polygon's next_url already includes all necessary filters.
            resp = requests.get(current_url, params=current_params, timeout=15)
            
            if resp.status_code == 429:  # Rate limit hit
                time.sleep(5)
                continue
                
            if resp.status_code != 200:
                print(f" [!] API Error {resp.status_code} on {current_url}")
                break

            data = resp.json()
            batch_results = data.get('results', [])
            if batch_results:
                raw_results.extend(batch_results)

            # --- PAGINATION CHECK ---
            next_url = data.get('next_url')
            if next_url:
                # Security/Auth: next_url doesn't always include the apiKey
                current_url = f"{next_url}&apiKey={api_key}" if "apiKey" not in next_url else next_url
                current_params = None  # Don't send params again; they are in the URL
            else:
                current_url = None

        except Exception as e:
            print(f" [!] Fetch Error: {e}")
            break

    # 2. SIEVE & MAP
    mapped_entries = []
    for d in raw_results:
        raw_tickers = d.get('tickers', [])
        ticker = raw_tickers[0] if raw_tickers else d.get('ticker')
        
        # FIX: Dynamically resolve the corresponding hash for the ticker in this data row
        ticker_hash = ticker_to_hash_map.get(ticker)
        if not ticker_hash:
            continue  # Skip if it belongs to an unmapped asset

   
        settlement_dt = d.get('settlement_date')
  

        entry = (
            ticker_hash,
            to_str(settlement_dt),
            to_str(d.get('short_interest')),
            to_str(d.get('avg_daily_volume')),
            to_str(d.get('days_to_cover')),
            ticker
        )
        mapped_entries.append(entry)
            
    return mapped_entries



def ingest_short_interest(ticker_to_hash_map, start, end, result_queue, api_key):
    """
    Unified Ingestion Interface: Synchronous pipeline mapping logic.
    Accepts scalar inst_id and ticker strings directly from the main loop.
    """
    all_mapped_data = fetch_and_map_short_interest_batch(
        ticker_to_hash_map,
        api_key, 
        start, 
        end
    )

    # Prepare SQL COPY Command (exactly matching the 6-tuple positions)
    short_int_copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_si (LIKE short_interest INCLUDING ALL);
        TRUNCATE staging_si;

        COPY staging_si (
            ticker_hash, settlement_date, short_interest, avg_daily_volume, 
            days_to_cover, ticker
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');

        INSERT INTO short_interest (
            ticker_hash, settlement_date, short_interest, avg_daily_volume, 
            days_to_cover, ticker
        )
        SELECT 
            ticker_hash, settlement_date, short_interest, avg_daily_volume, 
            days_to_cover, ticker
        FROM staging_si
        ON CONFLICT (ticker_hash, settlement_date) 
        DO UPDATE SET 
            short_interest = EXCLUDED.short_interest,
            avg_daily_volume = EXCLUDED.avg_daily_volume,
            days_to_cover = EXCLUDED.days_to_cover,
            ticker = EXCLUDED.ticker;
    """

    # Queue Results
    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put((
                all_mapped_data[i:i + chunk_size], 
                short_int_copy_sql, 
                f"Short Interest Chunk - {i}"
            ))
    else:
        print(f" [!] No short interest data found for selected universe/dates")