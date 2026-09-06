import os
import sys
import time
import requests
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index
)
from core.db import Base
import mmh3  # MurmurHash3 library


def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]

class Quote(Base):
    __tablename__ = "quotes"
  
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker_hash = Column(BigInteger, index=True, nullable=False)
    t = Column(BigInteger, index=True, nullable=False) 

    # --- TRADE DATA ---
    o = Column(Numeric(18, 4))
    h = Column(Numeric(18, 4))
    l = Column(Numeric(18, 4))
    c = Column(Numeric(18, 4))
    v = Column(Numeric(24, 10), default=0) 
    vw = Column(Numeric(18, 4))
    adjusted = Column(Boolean, nullable=False)
    __table_args__ = (
        Index('ix_instr_time', 'ticker_hash', 't'),
        UniqueConstraint('ticker_hash', 't', 'adjusted', name='uq_bar_canonical'),
        {'extend_existing': True}
    ) 

    

def resolve_ticker_hash(ticker, unix_ts_ns, router):
    if ticker not in router:
        return None
        
    ts_date = datetime.fromtimestamp(unix_ts_ns / 1_000_000_000, tz=timezone.utc).date()
    
    for interval in router[ticker]:
        if interval['start'] <= ts_date <= interval['end']:
            return interval['ticker_hash']
            
    return None


def fetch_and_map_bars_generator(ticker, ticker_hash, api_key, start, end,  config):
    """
    GENERATOR: Streams clean tuple payloads, mapped directly to the pre-resolved
    database hash token passed from the orchestration layer.
    """
    multiplier = config.get("multiplier", 1)
    timespan = config.get("timespan", "day")

    for adjust_flag in [True, False]:
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true" if adjust_flag else "false", 
            "limit": 50000,
            "sort": "asc",
            "apiKey": api_key
        }

        current_url = url
        current_params = params 

        while current_url:
            try:
                resp = requests.get(current_url, params=current_params, timeout=(5, 45))
                if resp.status_code == 429:
                    time.sleep(5)
                    continue

                
                # Server Error (500, 502, 503, 504) -> Pause 60 seconds and retry
                if 500 <= resp.status_code < 600:
                    print(f" [!] Server Error {resp.status_code} on {ticker}. Pausing 60s before retrying...")
                    time.sleep(60)
                    continue

                # Non-200 client/other errors -> Break out
                if resp.status_code != 200:
                    print(f" [!] API Error {resp.status_code} on {ticker}")
                    break


                data = resp.json()
                results = data.get('results', [])
                
                if not results:
                    break
                
                adjust_flag_str = "TRUE" if adjust_flag else "FALSE"
            
                page_tuples = []
                for d in results:
                    unix_t_millis = d.get('t') 
                    if not unix_t_millis:
                        continue
                    
                    unix_t_ns = unix_t_millis * 1_000_000 
                    
                    o_val = str(d['o']) if d.get('o') is not None else "None"
                    h_val = str(d['h']) if d.get('h') is not None else "None"
                    l_val = str(d['l']) if d.get('l') is not None else "None"
                    c_val = str(d['c']) if d.get('c') is not None else "None"
                    v_val = str(int(d['v'])) if d.get('v') is not None else "0"
                    vw_val = str(d['vw']) if d.get('vw') is not None else "None"

                    entry = (
                        str(int(ticker_hash)), 
                        str(unix_t_ns),
                        o_val, h_val, l_val, c_val, v_val, vw_val,
                        adjust_flag_str
                    )
                    page_tuples.append(entry)

                if page_tuples:
                    yield page_tuples, len(page_tuples), adjust_flag

                next_url = data.get('next_url')
                if next_url:
                    current_url = next_url if "apiKey" in next_url else f"{next_url}&apiKey={api_key}"
                    current_params = None
                else:
                    current_url = None

            except Exception as e:
                print(f" [!] Fetch Error {ticker} (Adjusted={adjust_flag}): {e}")
                break


            
def ingest_bars(ticker, ticker_hash, start, end, config, result_queue, api_key):
    """Consumes page matrices and pushes them down to the thread-safe worker."""
    for page_tuples, row_count, adjust_flag in fetch_and_map_bars_generator(ticker, ticker_hash, api_key, start, end, config):
        
        copy_sql = """
            CREATE TEMPORARY TABLE IF NOT EXISTS staging_quotes (LIKE quotes INCLUDING ALL);
            TRUNCATE staging_quotes;
            
            COPY staging_quotes (
                ticker_hash, t, o, h, l, c, v, vw, adjusted
            ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');
            
            INSERT INTO quotes (
                ticker_hash, t, o, h, l, c, v, vw, adjusted
            )
            SELECT 
                ticker_hash, t, o, h, l, c, v, vw, adjusted
            FROM staging_quotes
            ON CONFLICT (ticker_hash, t, adjusted) DO NOTHING;
        """
        
        info_str = f"Streaming: {ticker} | {row_count} rows loaded (Adjusted={adjust_flag})"
        result_queue.put((page_tuples, copy_sql, info_str))



