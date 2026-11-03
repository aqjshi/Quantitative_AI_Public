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

class CashFlow(Base):
    __tablename__ = 'cash_flow'
    __table_args__ = (UniqueConstraint('ticker_hash', 'period_end', 'filing_date', 'timeframe', name='uq_cash_flow_canonical'),
    )

    
    id = Column(BigInteger, primary_key=True)
    ticker = Column(String(20))
    # REFACTORED: Link to Instrument via Surrogate ID
    ticker_hash = Column(BigInteger, index=True, nullable=False)

    filing_date = Column(Date, nullable=False, index=True) 
    period_end  = Column(Date, nullable=False)   
    timeframe = Column(String(30)) 

    cash_from_operating_activities_continuing_operations            = Column(Numeric(20, 2))
    change_in_cash_and_equivalents                                  = Column(Numeric(20, 2))
    change_in_other_operating_assets_and_liabilities_net            = Column(Numeric(20, 2))
    depreciation_depletion_and_amortization                         = Column(Numeric(20, 2))
    dividends                                                       = Column(Numeric(20, 2))
    effect_of_currency_exchange_rate                                = Column(Numeric(20, 2))
    income_loss_from_discontinued_operations                        = Column(Numeric(20, 2))
    long_term_debt_issuances_repayments                             = Column(Numeric(20, 2))
    net_cash_from_financing_activities                              = Column(Numeric(20, 2))
    net_cash_from_financing_activities_continuing_operations        = Column(Numeric(20, 2))
    net_cash_from_financing_activities_discontinued_operations      = Column(Numeric(20, 2))
    net_cash_from_investing_activities                              = Column(Numeric(20, 2))
    net_cash_from_investing_activities_continuing_operations        = Column(Numeric(20, 2))
    net_cash_from_investing_activities_discontinued_operations      = Column(Numeric(20, 2))
    net_cash_from_operating_activities                              = Column(Numeric(20, 2))
    net_cash_from_operating_activities_discontinued_operations      = Column(Numeric(20, 2))
    net_income                                                      = Column(Numeric(20, 2))
    noncontrolling_interests                                        = Column(Numeric(20, 2))
    other_cash_adjustments                                          = Column(Numeric(20, 2))
    other_financing_activities                                      = Column(Numeric(20, 2))
    other_investing_activities                                      = Column(Numeric(20, 2))
    other_operating_activities                                      = Column(Numeric(20, 2))
    purchase_of_property_plant_and_equipment                        = Column(Numeric(20, 2))
    sale_of_property_plant_and_equipment                            = Column(Numeric(20, 2))
    short_term_debt_issuances_repayments                            = Column(Numeric(20, 2))



    fiscal_quarter = Column(Integer)       
    fiscal_year = Column(Integer, nullable=False)
    cik = Column(BigInteger, nullable=True)
     
    additional_data = Column(JSONB, server_default='{}')
 

def to_tsv_val(val):
    """Safely converts Python types to PostgreSQL COPY format with NULL handling."""
    if val is None or val == "" or val == "None":
        return r"\N"
    return str(val).replace("\t", " ").replace("\n", " ").replace("\r", " ")




def fetch_and_map_cash_flow_batch(
    ticker_to_hash_map, api_key, start_date, end_date
):
    url = "https://api.massive.com/stocks/financials/v1/cash-flow-statements"
    tickers_list = list(ticker_to_hash_map.keys())
    if not tickers_list:
        return []

    with engine.connect() as conn:
        query = text("SELECT DISTINCT cik FROM instruments WHERE cik IS NOT NULL")
        valid_ciks = {row[0] for row in conn.execute(query).fetchall()}

    params = {
        "tickers.any_of": ",".join(tickers_list),
        "limit": 50000,
        "period_end.gte": start_date,
        "period_end.lte": end_date,
        "sort": "period_end.desc",
        "apiKey": api_key,
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
            batch_results = data.get("results", [])
            if batch_results:
                raw_results.extend(batch_results)

            next_url = data.get("next_url")
            if next_url:
                current_url = (
                    f"{next_url}&apiKey={api_key}"
                    if "apiKey" not in next_url
                    else next_url
                )
                current_params = None
            else:
                current_url = None

        except Exception as e:
            print(f" [!] Fetch Error: {e}")
            break

    # In-memory dictionary for canonical deduplication
    deduped_entries = {}

    for d in raw_results:

        raw_tickers = d.get("tickers", [])
        if not isinstance(raw_tickers, list):
            raw_tickers = [raw_tickers] if raw_tickers else []
        if not raw_tickers and d.get("ticker"):
            raw_tickers = [d.get("ticker")]

        matched_ticker = None
        ticker_hash = None
        for sym in raw_tickers:
            if sym in ticker_to_hash_map:
                matched_ticker = sym
                ticker_hash = ticker_to_hash_map[sym]
                break

        if not ticker_hash:
            continue

        raw_cik = d.get("cik")
        if raw_cik is not None and int(raw_cik) not in valid_ciks:
            continue

        p_end = d.get("period_end")
        f_date = d.get("filing_date")
        t_frame = d.get("timeframe")

        dedup_key = (ticker_hash, str(p_end), str(f_date), str(t_frame))


  
        entry = (
            to_tsv_val(matched_ticker),
            to_tsv_val(ticker_hash),
            to_tsv_val(f_date),
            to_tsv_val(p_end),
            to_tsv_val(t_frame),
            to_tsv_val(d.get("cash_from_operating_activities_continuing_operations")),
            to_tsv_val(d.get("change_in_cash_and_equivalents")),
            to_tsv_val(d.get("change_in_other_operating_assets_and_liabilities_net")),
            to_tsv_val(d.get("depreciation_depletion_and_amortization")),
            to_tsv_val(d.get("dividends")),
            to_tsv_val(d.get("effect_of_currency_exchange_rate")),
            to_tsv_val(d.get("income_loss_from_discontinued_operations")),
            to_tsv_val(d.get("long_term_debt_issuances_repayments")),
            to_tsv_val(d.get("net_cash_from_financing_activities")),
            to_tsv_val(d.get("net_cash_from_financing_activities_continuing_operations")),
            to_tsv_val(d.get("net_cash_from_financing_activities_discontinued_operations")),
            to_tsv_val(d.get("net_cash_from_investing_activities")),
            to_tsv_val(d.get("net_cash_from_investing_activities_continuing_operations")),
            to_tsv_val(d.get("net_cash_from_investing_activities_discontinued_operations")),
            to_tsv_val(d.get("net_cash_from_operating_activities")),
            
            to_tsv_val(d.get("net_cash_from_operating_activities_discontinued_operations")),
            to_tsv_val(d.get("net_income")),
            to_tsv_val(d.get("noncontrolling_interests")),
            to_tsv_val(d.get("other_cash_adjustments")),
            to_tsv_val(d.get("other_financing_activities")),
            to_tsv_val(d.get("other_investing_activities")),
            to_tsv_val(d.get("other_operating_activities")),
            to_tsv_val(d.get("purchase_of_property_plant_and_equipment")),
            to_tsv_val(d.get("sale_of_property_plant_and_equipment")),
            to_tsv_val(d.get("short_term_debt_issuances_repayments")),
            to_tsv_val(d.get("fiscal_quarter")),
            to_tsv_val(d.get("fiscal_year")),
            to_tsv_val(raw_cik),
            json.dumps(d),
        )

        deduped_entries[dedup_key] = entry

    return list(deduped_entries.values())







def ingest_cash_flow(ticker_to_hash_map, start, end, result_queue, api_key):
    all_mapped_data = fetch_and_map_cash_flow_batch(
        ticker_to_hash_map, api_key, start, end
    )

    # Use INCLUDING DEFAULTS to omit unique constraints from the staging table
    cf_copy_sql = """
        CREATE TEMPORARY TABLE IF NOT EXISTS staging_cf (LIKE cash_flow INCLUDING DEFAULTS);
        TRUNCATE staging_cf;
        
        COPY staging_cf (
            ticker,  
            ticker_hash,                 
            filing_date,  
            period_end,      
            timeframe,     
            cash_from_operating_activities_continuing_operations, 
            change_in_cash_and_equivalents, 
            change_in_other_operating_assets_and_liabilities_net, 
            depreciation_depletion_and_amortization, 
            dividends, 
            effect_of_currency_exchange_rate, 
            income_loss_from_discontinued_operations, 
            long_term_debt_issuances_repayments, 
            net_cash_from_financing_activities, 
            net_cash_from_financing_activities_continuing_operations, 
            net_cash_from_financing_activities_discontinued_operations, 
            net_cash_from_investing_activities, 
            net_cash_from_investing_activities_continuing_operations, 
            net_cash_from_investing_activities_discontinued_operations, 
            net_cash_from_operating_activities, 
            net_cash_from_operating_activities_discontinued_operations, 
            net_income, 
            noncontrolling_interests, 
            other_cash_adjustments, 
            other_financing_activities, 
            other_investing_activities, 
            other_operating_activities, 
            purchase_of_property_plant_and_equipment, 
            sale_of_property_plant_and_equipment, 
            short_term_debt_issuances_repayments, 
            fiscal_quarter, 
            fiscal_year, 
            cik,             
            additional_data
        ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL '\\N');

        INSERT INTO cash_flow 
        SELECT DISTINCT ON (ticker_hash, period_end, filing_date, timeframe) * 
        FROM staging_cf
        ORDER BY ticker_hash, period_end, filing_date, timeframe, id DESC
        ON CONFLICT (ticker_hash, period_end, filing_date, timeframe) 
        DO UPDATE SET
            net_cash_from_operating_activities = EXCLUDED.net_cash_from_operating_activities,
            purchase_of_property_plant_and_equipment = EXCLUDED.purchase_of_property_plant_and_equipment,
            net_income = EXCLUDED.net_income,
            depreciation_depletion_and_amortization = EXCLUDED.depreciation_depletion_and_amortization,
            dividends = EXCLUDED.dividends,
            net_cash_from_investing_activities = EXCLUDED.net_cash_from_investing_activities,
            net_cash_from_financing_activities = EXCLUDED.net_cash_from_financing_activities,
            additional_data = EXCLUDED.additional_data;
    """

    if all_mapped_data:
        chunk_size = 5000
        for i in range(0, len(all_mapped_data), chunk_size):
            result_queue.put(
                (
                    all_mapped_data[i : i + chunk_size],
                    cf_copy_sql,
                    f"Cash Flow Chunk {i}",
                )
            )
    else:
        print(" [!] No valid Cash Flow data found for the selected universe/dates.")




