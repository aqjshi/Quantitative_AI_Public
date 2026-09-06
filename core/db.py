import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy import text




load_dotenv()
from sqlalchemy.orm import declarative_base


MACHINE_TYPE = os.getenv("MACHINE_TYPE", "win")


ALPHA_KEY       = os.getenv("ALPHA_VANTAGE_API_KEY", "")
AV_BASE_URL     = "https://www.alphavantage.co/query"
POLY_KEY  = os.getenv("POLYGON_API_KEY")

FRED_KEY_0  = os.getenv("FRED_KEY_0")
FRED_KEY_1 = os.getenv("FRED_KEY_1")
FRED_KEY_2  = os.getenv("FRED_KEY_2")
FRED_KEY_3  = os.getenv("FRED_KEY_3")
FRED_KEY_4  = os.getenv("FRED_KEY_4")


FRED_KEY_5  = os.getenv("FRED_KEY_5")
FRED_KEY_6 = os.getenv("FRED_KEY_6")
FRED_KEY_7  = os.getenv("FRED_KEY_7")
FRED_KEY_8  = os.getenv("FRED_KEY_8")
FRED_KEY_9  = os.getenv("FRED_KEY_9")


FRED_KEY_10  = os.getenv("FRED_KEY_10")
FRED_KEY_11 = os.getenv("FRED_KEY_11")
FRED_KEY_12  = os.getenv("FRED_KEY_12")
FRED_KEY_13  = os.getenv("FRED_KEY_13")
FRED_KEY_14  = os.getenv("FRED_KEY_14")


FRED_KEY_15  = os.getenv("FRED_KEY_15")
FRED_KEY_16 = os.getenv("FRED_KEY_16")
FRED_KEY_17  = os.getenv("FRED_KEY_17")
FRED_KEY_18  = os.getenv("FRED_KEY_18")
FRED_KEY_19  = os.getenv("FRED_KEY_19")


FRED_KEY_20  = os.getenv("FRED_KEY_20")
FRED_KEY_21 = os.getenv("FRED_KEY_21")
FRED_KEY_22  = os.getenv("FRED_KEY_22")
FRED_KEY_23  = os.getenv("FRED_KEY_23")
FRED_KEY_24  = os.getenv("FRED_KEY_24")

FRED_KEY_25  = os.getenv("FRED_KEY_25")
FRED_KEY_26 = os.getenv("FRED_KEY_26")
FRED_KEY_27  = os.getenv("FRED_KEY_27")
FRED_KEY_28  = os.getenv("FRED_KEY_28")
FRED_KEY_29  = os.getenv("FRED_KEY_29")


POLY_BASE = "https://api.polygon.io"
POLY_INCOME_BASE = "https://api.massive.com/stocks/financials/v1/income-statements"
SQL_USER   = os.getenv("SQL_USER")
SQL_PWD    = os.getenv("SQL_PWD")
SQL_HOST   = os.getenv("SQL_HOST")
SQL_PORT   = os.getenv("SQL_PORT")
SQL_DB_NAME= os.getenv("SQL_DB_NAME")
if SQL_PORT and str(SQL_PORT).lower() != "none": 
    host_section = f"{SQL_HOST}:{SQL_PORT}" 
else: 
    host_section = SQL_HOST 
DATABASE_URL = f"postgresql://{SQL_USER}:{SQL_PWD}@{host_section}/{SQL_DB_NAME}"

# DATABASE_URL = f"postgresql://{SQL_USER}:{SQL_PWD}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
engine        = create_engine(DATABASE_URL, echo=False)
SessionLocal  = sessionmaker(bind=engine)
Base = declarative_base() # <--- Define it once here
# print(f"DEBUG: SQL_USER={SQL_USER}, SQL_PORT={SQL_PORT}, SQL_DB_NAME={SQL_DB_NAME}")


def resolve_identity_matrix(case_study_tickers):
    # Ensure tickers are a clean, unique list
    ticker_list = list(set(case_study_tickers))
    
    query = text("""
        SELECT 
            ticker,
            ticker_hash,
            composite_figi,
            composite_figi_hash,
            point_in_time_date
        FROM instruments
        WHERE ticker = ANY(:tlist)
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tlist": ticker_list})
        
    # Sort and clean index to align with your pipeline configuration
    df = df.sort_values(["ticker", "point_in_time_date"]).reset_index(drop=True)
    return df




def get_ticker_from_id(inst_id):
    """Queries the instruments table to map an integer ID back to a string ticker."""
    query = text("""
        SELECT ticker 
        FROM instruments 
        WHERE ticker_hash = :inst_id 
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            res = conn.execute(query, {"inst_id": int(inst_id)}).fetchone()
            if res:
                return res[0]
    except Exception as e:
        print(f"[WARNING] Database lookup failed for id {inst_id}: {e}")
    return "UNKNOWN"





def ticker_to_identity_map(tickers: list, lookback_date: str) -> pd.DataFrame:
    """
    Maps tickers to identity horizons. Creates independent segment blocks 
    ONLY when the underlying FIGI changes, ignoring regular temporal logging gaps.
    """
    ticker_list = list(set(tickers))
    if not ticker_list:
        return pd.DataFrame()

    query = text("""
    WITH ordered_timeline AS (
        SELECT 
            ticker,
            ticker_hash,
            composite_figi,
            composite_figi_hash,
            sic_code,
            point_in_time_date::date as point_in_time_date,
            -- Look backward to see what the previous row's FIGI was
            LAG(composite_figi_hash) OVER (
                PARTITION BY ticker_hash
                ORDER BY point_in_time_date ASC
            ) as prev_figi_hash
        FROM instruments
        WHERE ticker = ANY(:tlist)
          AND point_in_time_date <= CAST(:target_date AS DATE)
          AND composite_figi IS NOT NULL
    ),
    detected_shifts AS (
        SELECT *,
            CASE 
                WHEN prev_figi_hash IS NULL THEN 0
                -- ALTERED: Only trigger a brand new segment if the physical FIGI changed
                WHEN composite_figi_hash != prev_figi_hash THEN 1
                ELSE 0
            END as is_new_segment
        FROM ordered_timeline
    ),
    island_groups AS (
        SELECT *,
            SUM(is_new_segment) OVER (
                PARTITION BY ticker_hash
                ORDER BY point_in_time_date ASC
            ) as segment_id
        FROM detected_shifts
    )
    SELECT 
        ticker,
        ticker_hash,
        composite_figi,
        composite_figi_hash,
        (ARRAY_AGG(sic_code ORDER BY CASE WHEN sic_code IS NOT NULL THEN point_in_time_date END DESC NULLS LAST))[1] as sic_code,
        MIN(point_in_time_date) as earliest,
        MAX(point_in_time_date) as latest
    FROM island_groups
    GROUP BY 
        ticker, 
        ticker_hash, 
        composite_figi, 
        composite_figi_hash,
        segment_id
    ORDER BY 
        ticker, 
        earliest ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query, 
            conn, 
            params={
                "tlist": ticker_list, 
                "target_date": pd.to_datetime(lookback_date).strftime('%Y-%m-%d')
            }
        )
        
    return df

def figi_hash_to_identity_map(figi_hashes: list, lookback_date: str, drop_unknown: bool = False) -> pd.DataFrame:
    """
    Identifies contiguous chronological runs for unique FIGIs.
    If the FIGI is UNKNOWN, it treats each unique ticker as an independent track
    to avoid multiplexing collisions and false continuity breaks.
    """
    figi_hash_list = list(set([int(h) for h in figi_hashes]))
    if not figi_hash_list:
        return pd.DataFrame()
    
    query_str = """
    WITH ordered_timeline AS (
        SELECT 
            ticker,
            ticker_hash,
            composite_figi,
            composite_figi_hash,
            sic_code,
            point_in_time_date::date as point_in_time_date,
            
            -- TRACK MULTIPLE HISTORIES SEPARATELY FOR UNKNOWN FIGIs:
            -- If FIGI is UNKNOWN, we partition by the ticker_hash itself so different 
            -- companies sharing the same placeholder ID don't bleed into each other's window.
            LAG(ticker_hash) OVER (
                PARTITION BY 
                    CASE 
                        WHEN upper(composite_figi) = 'UNKNOWN' THEN ticker_hash
                        ELSE composite_figi_hash 
                    END
                ORDER BY point_in_time_date ASC
            ) as prev_ticker_hash
        FROM instruments
        WHERE composite_figi_hash = ANY(:flist)
          AND point_in_time_date <= CAST(:target_date AS DATE)
          AND composite_figi IS NOT NULL
          {unknown_filter}
    ),
    detected_shifts AS (
        SELECT *,
            CASE 
                WHEN prev_ticker_hash IS NULL THEN 0
                -- If ticker hash changes within our tracking partition, break the segment
                WHEN ticker_hash != prev_ticker_hash THEN 1
                ELSE 0
            END as is_new_segment
        FROM ordered_timeline
    ),
    island_groups AS (
        SELECT *,
            SUM(is_new_segment) OVER (
                PARTITION BY 
                    CASE 
                        WHEN upper(composite_figi) = 'UNKNOWN' THEN ticker_hash
                        ELSE composite_figi_hash 
                    END
                ORDER BY point_in_time_date ASC
            ) as segment_id
        FROM detected_shifts
    )
    SELECT 
        ticker,
        ticker_hash,
        composite_figi,
        composite_figi_hash,
        (ARRAY_AGG(sic_code ORDER BY CASE WHEN sic_code IS NOT NULL THEN point_in_time_date END DESC NULLS LAST))[1] as sic_code,
        MIN(point_in_time_date) as earliest,
        MAX(point_in_time_date) as latest
    FROM island_groups
    GROUP BY 
        ticker, 
        ticker_hash, 
        composite_figi, 
        composite_figi_hash,
        segment_id
    ORDER BY 
        composite_figi_hash, 
        earliest ASC
    """
    
    unknown_filter = ""
    if drop_unknown:
        unknown_filter = """
          AND upper(composite_figi) != 'UNKNOWN'
          AND upper(ticker) != 'UNKNOWN'
        """
        
    formatted_query = query_str.format(unknown_filter=unknown_filter)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(formatted_query), 
            conn, 
            params={
                "flist": figi_hash_list, 
                "target_date": pd.to_datetime(lookback_date).strftime('%Y-%m-%d')
            }
        )
        
    return df

