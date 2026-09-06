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
from core.db import Base, engine 

class IncomeStatement(Base):
    """
    Standardized Financial Manifold: Income Statement
    """
    __tablename__ = 'income_statements'
        
    id = Column(BigInteger, primary_key=True)

    ticker = Column(String(20))
    # REFACTORED: Link to Instrument via Surrogate ID
    ticker_hash = Column(BigInteger, index=True, nullable=False)

    filing_date = Column(Date, nullable=False, index=True) 
    period_end  = Column(Date, nullable=False)   
    timeframe = Column(String(30)) 
    

    basic_earnings_per_share     = Column(Numeric(20, 2))
    basic_shares_outstanding = Column(Numeric(20, 2))
    consolidated_net_income_loss = Column(Numeric(20, 2))
    cost_of_revenue = Column(Numeric(20, 2))
    depreciation_depletion_amortization = Column(Numeric(20, 2))
    diluted_earnings_per_share = Column(Numeric(20, 2))
    diluted_shares_outstanding = Column(Numeric(20, 2))
    discontinued_operations = Column(Numeric(20, 2))
    ebitda = Column(Numeric(20, 2))
    equity_in_affiliates = Column(Numeric(20, 2))
    extraordinary_items = Column(Numeric(20, 2))
    gross_profit = Column(Numeric(20, 2))
    income_before_income_taxes = Column(Numeric(20, 2))
    income_taxes = Column(Numeric(20, 2))
    interest_expense = Column(Numeric(20, 2))
    interest_income = Column(Numeric(20, 2))
    net_income_loss_attributable_common_shareholders = Column(Numeric(20, 2))
    noncontrolling_interest = Column(Numeric(20, 2))
    operating_income = Column(Numeric(20, 2))
    other_income_expense = Column(Numeric(20, 2))
    other_operating_expenses = Column(Numeric(20, 2))
    preferred_stock_dividends_declared = Column(Numeric(20, 2))
    research_development = Column(Numeric(20, 2))
    revenue = Column(Numeric(20, 2))
    selling_general_administrative = Column(Numeric(20, 2))
    total_operating_expenses = Column(Numeric(20, 2))
    total_other_income_expense = Column(Numeric(20, 2))


    fiscal_quarter = Column(Integer)       
    fiscal_year = Column(Integer, nullable=False)
    cik = Column(BigInteger, nullable=True)
    additional_data = Column(JSONB, server_default='{}')
    UniqueConstraint('ticker_hash', 'period_end', 'filing_date', 'timeframe', name='uq_income_statements_canonical')


def to_str(val):
    if val is None:
        return 'None' # Matches the 'NULL None' in your COPY command
    return str(val)

def fetch_and_map_income_statements_batch(ticker_to_hash_map, api_key, start_date, end_date):
    url = "https://api.massive.com/stocks/financials/v1/income-statements"
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
            resp = requests.get(current_url, params=current_params, timeout=15)
            
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


        raw_q = d.get('fiscal_quarter') or d.get('fiscal_period')
        f_quarter = int(str(raw_q).replace('Q', '')) if raw_q else None


        entry = (
            matched_ticker ,
            ticker_hash ,
            to_str(d.get('filing_date')),
            to_str(d.get('period_end')),
            to_str(d.get('timeframe')), 
    
            to_str(d.get('basic_earnings_per_share')),
            to_str(d.get('basic_shares_outstanding')),
            to_str(d.get('consolidated_net_income_loss')),
            to_str(d.get('cost_of_revenue')),
            to_str(d.get('depreciation_depletion_amortization')),
            to_str(d.get('diluted_earnings_per_share')),
            to_str(d.get('diluted_shares_outstanding')),
            to_str(d.get('discontinued_operations')),
            to_str(d.get('ebitda')),
            to_str(d.get('equity_in_affiliates')),
            to_str(d.get('extraordinary_items')),
            to_str(d.get('gross_profit')),
            to_str(d.get('income_before_income_taxes')),
            to_str(d.get('income_taxes')),
            to_str(d.get('interest_expense')),
            to_str(d.get('interest_income')),
            to_str(d.get('net_income_loss_attributable_common_shareholders')),
            to_str(d.get('noncontrolling_interest')),
            to_str(d.get('operating_income')),
            to_str(d.get('other_income_expense')),
            to_str(d.get('other_operating_expenses')),
            to_str(d.get('preferred_stock_dividends_declared')),
            to_str(d.get('research_development')),
            to_str(d.get('revenue')),
            to_str(d.get('selling_general_administrative')),
            to_str(d.get('total_operating_expenses')),
            to_str(d.get('total_other_income_expense')),
            f_quarter,   
            to_str(d.get('fiscal_year')),
            d.get('cik'),                            
            json.dumps({})                                               # additional_data
        )
        mapped_entries.append(entry)
        
    return mapped_entries



def ingest_income_statements(ticker_to_hash_map, start, end, result_queue, api_key):
    """
    REFACTORED: Perfectly matches the execution matrix of ingest_cash_flow.
    """
    all_mapped_data = fetch_and_map_income_statements_batch(
        ticker_to_hash_map,
        api_key, 
        start, end
    )

    stmt_copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_is (LIKE income_statements INCLUDING ALL);
        TRUNCATE staging_is;
        COPY staging_is (
            ticker ,
            ticker_hash ,
            filing_date,
            period_end,
            timeframe, 
            basic_earnings_per_share,
            basic_shares_outstanding,
            consolidated_net_income_loss,
            cost_of_revenue,
            depreciation_depletion_amortization,
            diluted_earnings_per_share,
            diluted_shares_outstanding,
            discontinued_operations,
            ebitda,
            equity_in_affiliates,
            extraordinary_items,
            gross_profit,
            income_before_income_taxes,
            income_taxes,
            interest_expense,
            interest_income,
            net_income_loss_attributable_common_shareholders,
            noncontrolling_interest,
            operating_income,
            other_income_expense,
            other_operating_expenses,
            preferred_stock_dividends_declared,
            research_development,
            revenue,
            selling_general_administrative,
            total_operating_expenses,
            total_other_income_expense,
            fiscal_quarter,   
            fiscal_year,
            cik,         
            additional_data
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL 'None');
        INSERT INTO income_statements 
        SELECT * FROM staging_is;
    """
    
    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put((
                all_mapped_data[i:i + chunk_size], 
                stmt_copy_sql, 
                f"Income Statement Chunk {i}"
            ))
    else:
        print(f" [!] No Income Statement data found for ticker selected universe/dates.")

