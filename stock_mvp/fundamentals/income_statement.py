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
    __table_args__ = (UniqueConstraint('ticker_hash', 'period_end', 'filing_date', 'timeframe', name='uq_income_statements_canonical'),
    )
    
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


def to_tsv_val(val):
    """Safely converts Python types to PostgreSQL COPY format with NULL handling."""
    if val is None or val == "" or val == "None":
        return r"\N"
    return str(val).replace("\t", " ").replace("\n", " ").replace("\r", " ")



def fetch_and_map_income_statements_batch(ticker_to_hash_map, api_key, start_date, end_date):
    url = "https://api.massive.com/stocks/financials/v1/income-statements"
    tickers_list = list(ticker_to_hash_map.keys())
    if not tickers_list:
        return []

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

    # FIX 2: Use in-memory deduplication like balance_sheets does
    deduped_entries = {}
    
    for d in raw_results:
        raw_tickers = d.get('tickers', [])
        if not isinstance(raw_tickers, list):
            raw_tickers = [raw_tickers] if raw_tickers else []
            
        if not raw_tickers and d.get('ticker'):
            raw_tickers = [d.get('ticker')]

        matched_ticker = None
        ticker_hash = None
        
        for sym in raw_tickers:
            if sym in ticker_to_hash_map:
                matched_ticker = sym
                ticker_hash = ticker_to_hash_map[sym]
                break

        if not ticker_hash:
            continue 

        raw_cik = d.get('cik')
        if raw_cik is not None and int(raw_cik) not in valid_ciks:
            continue

        p_end = d.get('period_end')
        f_date = d.get('filing_date')
        t_frame = d.get('timeframe')

        # Canonical key for deduplication
        dedup_key = (ticker_hash, str(p_end), str(f_date), str(t_frame))

        entry = (
            to_tsv_val(matched_ticker), # FIX 1: Wrapped in to_tsv_val
            to_tsv_val(ticker_hash),    # FIX 1: Wrapped in to_tsv_val
            to_tsv_val(f_date),
            to_tsv_val(p_end),
            to_tsv_val(t_frame), 
    
            to_tsv_val(d.get('basic_earnings_per_share')),
            to_tsv_val(d.get('basic_shares_outstanding')),
            to_tsv_val(d.get('consolidated_net_income_loss')),
            to_tsv_val(d.get('cost_of_revenue')),
            to_tsv_val(d.get('depreciation_depletion_amortization')),
            to_tsv_val(d.get('diluted_earnings_per_share')),
            to_tsv_val(d.get('diluted_shares_outstanding')),
            to_tsv_val(d.get('discontinued_operations')),
            to_tsv_val(d.get('ebitda')),
            to_tsv_val(d.get('equity_in_affiliates')),
            to_tsv_val(d.get('extraordinary_items')),
            to_tsv_val(d.get('gross_profit')),
            to_tsv_val(d.get('income_before_income_taxes')),
            to_tsv_val(d.get('income_taxes')),
            to_tsv_val(d.get('interest_expense')),
            to_tsv_val(d.get('interest_income')),
            to_tsv_val(d.get('net_income_loss_attributable_common_shareholders')),
            to_tsv_val(d.get('noncontrolling_interest')),
            to_tsv_val(d.get('operating_income')),
            to_tsv_val(d.get('other_income_expense')),
            to_tsv_val(d.get('other_operating_expenses')),
            to_tsv_val(d.get('preferred_stock_dividends_declared')),
            to_tsv_val(d.get('research_development')),
            to_tsv_val(d.get('revenue')),
            to_tsv_val(d.get('selling_general_administrative')),
            to_tsv_val(d.get('total_operating_expenses')),
            to_tsv_val(d.get('total_other_income_expense')),
            to_tsv_val(d.get("fiscal_quarter")),            
            to_tsv_val(d.get('fiscal_year')),
            to_tsv_val(raw_cik),             
            json.dumps({}) 
        )
        
        # Overwrite duplicates cleanly before hitting the COPY command
        deduped_entries[dedup_key] = entry
        
    return list(deduped_entries.values())

def ingest_income_statements(
    ticker_to_hash_map, start, end, result_queue, api_key
):
    all_mapped_data = fetch_and_map_income_statements_batch(
        ticker_to_hash_map, api_key, start, end
    )

    stmt_copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_is (LIKE income_statements INCLUDING DEFAULTS);
        TRUNCATE staging_is;

        COPY staging_is (
            ticker,
            ticker_hash,
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
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL '\\N');

        INSERT INTO income_statements 
        SELECT DISTINCT ON (ticker_hash, period_end, filing_date, timeframe) * 
        FROM staging_is
        ORDER BY ticker_hash, period_end, filing_date, timeframe, id DESC
        ON CONFLICT (ticker_hash, period_end, filing_date, timeframe) 
        DO UPDATE SET
            revenue = EXCLUDED.revenue,
            gross_profit = EXCLUDED.gross_profit,
            ebitda = EXCLUDED.ebitda,
            operating_income = EXCLUDED.operating_income,
            total_operating_expenses = EXCLUDED.total_operating_expenses,
            basic_shares_outstanding = EXCLUDED.basic_shares_outstanding,
            net_income_loss_attributable_common_shareholders = EXCLUDED.net_income_loss_attributable_common_shareholders,
            research_development = EXCLUDED.research_development,
            selling_general_administrative = EXCLUDED.selling_general_administrative,
            total_other_income_expense = EXCLUDED.total_other_income_expense,
            additional_data = EXCLUDED.additional_data;
    """

    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put(
                (
                    all_mapped_data[i : i + chunk_size],
                    stmt_copy_sql,
                    f"Income Statement Chunk {i}",
                )
            )
    else:
        print(" [!] No valid Income Statement data found for universe/dates.")