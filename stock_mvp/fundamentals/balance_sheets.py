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
from core.db import Base, engine 

class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
        
          
    id                                              = Column(BigInteger, primary_key=True)
    ticker                                          = Column(String(20)) 
    ticker_hash                                     = Column(BigInteger, index=True, nullable=False)
    filing_date                                     = Column(Date, nullable=False, index=True) 
    period_end                                      = Column(Date, nullable=False)          
    timeframe                                       = Column(String(30)) 
    accounts_payable                                = Column(Numeric(20, 2))     
    accrued_and_other_current_liabilities           = Column(Numeric(20, 2))
    accumulated_other_comprehensive_income          = Column(Numeric(20, 2))
    additional_paid_in_capital                      = Column(Numeric(20, 2))
    cash_and_equivalents                            = Column(Numeric(20, 2))
    commitments_and_contingencies                   = Column(Numeric(20, 2))  
    common_stock                                    = Column(Numeric(20, 2))
    debt_current                                    = Column(Numeric(20, 2))
    deferred_revenue_current                        = Column(Numeric(20, 2))
    goodwill                                        = Column(Numeric(20, 2))
    intangible_assets_net                           = Column(Numeric(20, 2))
    inventories                                     = Column(Numeric(20, 2))
    long_term_debt_and_capital_lease_obligations    = Column(Numeric(20, 2))
    noncontrolling_interest                         = Column(Numeric(20, 2))
    other_assets                                    = Column(Numeric(20, 2))
    other_current_assets                            = Column(Numeric(20, 2))
    other_equity                                    = Column(Numeric(20, 2))
    other_noncurrent_liabilities                    = Column(Numeric(20, 2))
    preferred_stock                                 = Column(Numeric(20, 2))
    property_plant_equipment_net                    = Column(Numeric(20, 2))
    receivables                                     = Column(Numeric(20, 2))
    retained_earnings_deficit                       = Column(Numeric(20, 2))
    short_term_investments                          = Column(Numeric(20, 2))  
    total_assets                                    = Column(Numeric(20, 2))
    total_current_assets                            = Column(Numeric(20, 2))      
    total_current_liabilities                       = Column(Numeric(20, 2))
    total_equity                                    = Column(Numeric(20, 2))
    total_equity_attributable_to_parent             = Column(Numeric(20, 2))
    total_liabilities                               = Column(Numeric(20, 2))
    total_liabilities_and_equity                    = Column(Numeric(20, 2))
    treasury_stock                                  = Column(Numeric(20, 2))
    fiscal_quarter                                  = Column(Integer)
    fiscal_year                                     = Column(Integer, nullable=False)
    cik                                             = Column(BigInteger, nullable=True)
    
    additional_data = Column(JSONB, server_default='{}')
    UniqueConstraint('ticker_hash', 'period_end', 'filing_date', 'timeframe', name='uq_balance_sheets_canonical')
    



def to_str(val):
    if val is None:
        return 'None' # Matches the 'NULL None' in your COPY command
    return str(val)



def fetch_and_map_balance_sheets_batch(ticker_to_hash_map, api_key, start_date, end_date):
    """
    SPEEDUP: Accepts a dictionary of {ticker: ticker_hash} and queries 
    multiple tickers simultaneously in huge paginated chunks.
    """
    url = "https://api.massive.com/stocks/financials/v1/balance-sheets"
    
    # Combine tickers into a comma-separated filter string
    tickers_list = list(ticker_to_hash_map.keys())
    if not tickers_list:
        return []
    # UPSTREAM FIX 1: Fetch valid primary CIKs active in your instruments table
    # -------------------------------------------------------------------------
    valid_ciks = set()
    with engine.connect() as conn:
        query = text("SELECT DISTINCT cik FROM instruments WHERE cik IS NOT NULL")
        valid_ciks = {row[0] for row in conn.execute(query).fetchall()}


    params = {
        "tickers.any_of": ",".join(tickers_list),
        "limit": 50000,
        "period_end.gte": start_date,
        "period_end.lte": end_date,
        "sort": "period_end.desc",
        "apiKey": api_key
    }
    
    raw_results = []
    current_url = url
    current_params = params

    while current_url:
        try:
            resp = requests.get(current_url, params=current_params, timeout=20)
            if resp.status_code == 429:
                time.sleep(5)
                continue
            if resp.status_code != 200:
                print(f" [!] API Error {resp.status_code} on {current_url}")
                break
                
            data = resp.json()
            batch_results = data.get('results', [])
            
            if batch_results:
                raw_results.extend(batch_results)

            next_url = data.get('next_url')
            if next_url:
                current_url = f"{next_url}&apiKey={api_key}" if "apiKey" not in next_url else next_url
                current_params = None  
            else:
                current_url = None
        except Exception as e:
            print(f" [!] Fetch Error: {e}")
            break

    mapped_entries = []
    for d in raw_results:
        raw_tickers = d.get('tickers', [])
        if not isinstance(raw_tickers, list):
            raw_tickers = [raw_tickers] if raw_tickers else []
            
        # Fallback to single field if tickers array is empty
        if not raw_tickers and d.get('ticker'):
            raw_tickers = [d.get('ticker')]

        # =====================================================================
        # FIXED: CHRONOLOGICAL PORTFOLIO MULTI-MATCH ENGINE
        # Scan all connected asset symbols to locate your targeted portfolio tracker
        # =====================================================================
        matched_ticker = None
        ticker_hash = None
        
        for sym in raw_tickers:
            if sym in ticker_to_hash_map:
                matched_ticker = sym
                ticker_hash = ticker_to_hash_map[sym]
                break # Matched asset found, break lookup loop safely

        
        if not ticker_hash:
            continue  # Skip if it belongs to an unmapped asset


        raw_cik = d.get('cik')
        
        # -------------------------------------------------------------------------
        # UPSTREAM FIX 1 CHECK: Drop orphaned secondary trusts / untracked CIKs
        # -------------------------------------------------------------------------
        if raw_cik is not None and int(raw_cik) not in valid_ciks:
            continue


        p_end = d.get('period_end')
        raw_q = d.get('fiscal_period') or d.get('fiscal_quarter')
        f_quarter = int(str(raw_q).replace('Q', '')) if raw_q else None

        
        # 2. Build Ordered Tuple (Must match bs_copy_sql EXACTLY)
        entry = (
            matched_ticker,   
            ticker_hash,                                       
            to_str(d.get('filing_date')),                       
            to_str(p_end),                                       
            to_str(d.get('timeframe')),                            
            to_str(d.get('accounts_payable')),      
            to_str(d.get('accrued_and_other_current_liabilities')), 
            to_str(d.get('accumulated_other_comprehensive_income')),
            to_str(d.get('additional_paid_in_capital')), 
            to_str(d.get('cash_and_equivalents')),
            to_str(d.get('commitments_and_contingencies')),     
            to_str(d.get('common_stock')),
            to_str(d.get('debt_current')), 
            to_str(d.get('deferred_revenue_current')),
            to_str(d.get('goodwill')),
            to_str(d.get('intangible_assets_net')),  
            to_str(d.get('inventories')),
            to_str(d.get('long_term_debt_and_capital_lease_obligations')),  
            to_str(d.get('noncontrolling_interest')),  
            to_str(d.get('other_assets')), 
            to_str(d.get('other_current_assets')),
            to_str(d.get('other_equity')), 
            to_str(d.get('other_noncurrent_liabilities')),  
            to_str(d.get('preferred_stock')), 
            to_str(d.get('property_plant_equipment_net')),
            to_str(d.get('receivables')),
            to_str(d.get('retained_earnings_deficit')),
            to_str(d.get('short_term_investments')),         
            to_str(d.get('total_assets')),   
            to_str(d.get('total_current_assets')),               
            to_str(d.get('total_current_liabilities')), 
            to_str(d.get('total_equity')),              
            to_str(d.get('total_equity_attributable_to_parent')), 
            to_str(d.get('total_liabilities')),                   
            to_str(d.get('total_liabilities_and_equity')),
            to_str(d.get('treasury_stock')),  
            to_str(f_quarter),                                  
            to_str(d.get('fiscal_year')),    
            d.get('cik'),                                    
            json.dumps({})                                 
            )
            

        mapped_entries.append(entry)
            
    return mapped_entries






def ingest_balance_sheets(ticker_to_hash_map, start, end, result_queue, api_key):
    """
    Processes the entire portfolio universe in large parallel database chunks.
    """
    all_mapped_data = fetch_and_map_balance_sheets_batch(
        ticker_to_hash_map, api_key,
        start, end
    )

    # UPDATED SQL: Added the NULL 'None' specification
    bs_copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_bs (LIKE balance_sheets INCLUDING ALL);
        TRUNCATE staging_bs;
        
        COPY staging_bs (
            ticker, ticker_hash, filing_date, period_end,timeframe,
            accounts_payable     , 
            accrued_and_other_current_liabilities ,
            accumulated_other_comprehensive_income,
            additional_paid_in_capital ,
            cash_and_equivalents,
            commitments_and_contingencies ,    
            common_stock,
            debt_current ,
            deferred_revenue_current,
            goodwill,
            intangible_assets_net  ,
            inventories,
            long_term_debt_and_capital_lease_obligations , 
            noncontrolling_interest  ,
            other_assets ,
            other_current_assets,
            other_equity ,
            other_noncurrent_liabilities  ,
            preferred_stock ,
            property_plant_equipment_net,
            receivables,
            retained_earnings_deficit,
            short_term_investments       ,  
            total_assets   ,
            total_current_assets       ,        
            total_current_liabilities ,
            total_equity             ,
            total_equity_attributable_to_parent ,
            total_liabilities               ,    
            total_liabilities_and_equity,
            treasury_stock  ,
            fiscal_quarter, fiscal_year, 
            cik, additional_data
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');
        
        -- SIMPLE INSERT: No conflict handling
        INSERT INTO balance_sheets 
        SELECT * FROM staging_bs;
    """
    
    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put((
                all_mapped_data[i:i + chunk_size], 
                bs_copy_sql, 
                f"Balance Sheets Bulk Chunk Group {i}"
            ))
    else:
        print(" [!] No balance sheet data found for the selected universe/dates.")
