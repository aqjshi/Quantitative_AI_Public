import requests
import os 
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.dialects.postgresql import JSONB


from sqlalchemy.orm import relationship, declarative_base
from core.db import Base

class Dividend(Base):
    __tablename__ = "dividends"
    id = Column(BigInteger, primary_key=True)
    ticker_hash = Column(BigInteger, nullable=False, index=True)
    external_id = Column(String(128), index=True)

    # Core Data
    record_date = Column(Date)
    pay_date = Column(Date)
    declaration_date = Column(Date)
    ex_dividend_date = Column(Date, nullable=False)
    frequency = Column(Integer)
    cash_amount = Column(Numeric(20, 4))
    currency = Column(String(20))
    distribution_type = Column(String(20))
    # Metadata Block
    ticker = Column(String(20)) 
    # --- ADD THESE TWO COLUMNS ---
    historical_adjustment_factor = Column(Float) 
    split_adjusted_cash_amount = Column(Numeric(20, 4))

    __table_args__ = (
        UniqueConstraint('ticker_hash', 'ex_dividend_date', 'distribution_type', 'cash_amount', name='uq_dividend_identity'),
    )

def to_str(val):
    if val is None:
        return 'None' # Matches the 'NULL None' in your COPY command
    return str(val)


def fetch_and_map_dividends_batch(ticker_to_hash_map, api_key, start_date, end_date):
    url = "https://api.massive.com/stocks/v1/dividends"
    
    tickers_list   = list(ticker_to_hash_map.keys())
    if not tickers_list:
        return []
    

    params = {
        "ticker.any_of":",".join(tickers_list),
        "limit": 5000,
        "sort": "ex_dividend_date",
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
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
    mapped_entries = []



    seen_keys = set()  # FIX 1: Upstream in-memory dedup tracking

    for d in raw_results:
        raw_tickers = d.get('tickers', [])
        ticker = raw_tickers[0] if raw_tickers else d.get('ticker')
        
        ticker_hash = ticker_to_hash_map.get(ticker)
        if not ticker_hash:
            continue

        ex_date_str = to_str(d.get('ex_dividend_date'))
        dist_type = to_str(d.get('distribution_type'))
        cash_amt = to_str(d.get('cash_amount'))

        # FIX 1 CHECK: Drop duplicate records inside raw payload batch
        dedup_key = (ticker_hash, ex_date_str, dist_type, cash_amt)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        entry = (
            ticker_hash,                                    # 1
            to_str(d.get('id')),                            # 2: external_id
            ticker,                                         # 3: ticker
            to_str(d.get('record_date')),                   # 4
            to_str(d.get('pay_date')),                      # 5
            to_str(d.get('declaration_date')),              # 6
            ex_date_str,                                    # 7: ex_dividend_date
            to_str(d.get('frequency')),                     # 8
            cash_amt,                                       # 9: cash_amount
            to_str(d.get('currency')),                      # 10
            dist_type,                                      # 11: distribution_type
            to_str(d.get('historical_adjustment_factor')),  # 12
            to_str(d.get('split_adjusted_cash_amount'))     # 13
        )
        mapped_entries.append(entry)

    return mapped_entries

def ingest_dividends(ticker_to_hash_map, start, end, result_queue, api_key):
    all_mapped_data = fetch_and_map_dividends_batch(
        ticker_to_hash_map,  
        api_key, 
        start, 
        end
    )

    div_copy_sql = """
        DROP TABLE IF EXISTS staging_div;
        CREATE TEMPORARY TABLE staging_div (
            ticker_hash BIGINT,
            external_id VARCHAR(128),
            ticker VARCHAR(20),
            record_date DATE,
            pay_date DATE,
            declaration_date DATE,
            ex_dividend_date DATE,
            frequency INT,
            cash_amount NUMERIC(20, 4),
            currency VARCHAR(20),
            distribution_type VARCHAR(20),
            historical_adjustment_factor FLOAT,
            split_adjusted_cash_amount NUMERIC(20, 4)
        );

        COPY staging_div (
            ticker_hash, external_id, ticker, record_date, pay_date, 
            declaration_date, ex_dividend_date, frequency, cash_amount, 
            currency, distribution_type,
            historical_adjustment_factor, split_adjusted_cash_amount
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');
        
        INSERT INTO dividends (
            ticker_hash, external_id, ticker, record_date, pay_date, 
            declaration_date, ex_dividend_date, frequency, cash_amount, 
            currency, distribution_type,
            historical_adjustment_factor, split_adjusted_cash_amount
        )
        SELECT 
            ticker_hash, external_id, ticker, record_date, pay_date, 
            declaration_date, ex_dividend_date, frequency, cash_amount, 
            currency, distribution_type,
            historical_adjustment_factor, split_adjusted_cash_amount
        FROM staging_div
        ON CONFLICT (ticker_hash, ex_dividend_date, distribution_type, cash_amount) 
        DO UPDATE SET
            external_id = EXCLUDED.external_id,
            record_date = EXCLUDED.record_date,
            pay_date = EXCLUDED.pay_date,
            declaration_date = EXCLUDED.declaration_date,
            frequency = EXCLUDED.frequency,
            currency = EXCLUDED.currency,
            historical_adjustment_factor = EXCLUDED.historical_adjustment_factor,
            split_adjusted_cash_amount = EXCLUDED.split_adjusted_cash_amount;
    """
    
    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put((
                all_mapped_data[i:i + chunk_size], 
                div_copy_sql, 
                f"Dividends Chunk {i}"
            ))
    else:
        print(" [!] No Dividends data found for the selected universe/dates.")