import psycopg
import sys
import os 
from typing import List, Any, Tuple, Dict
from tqdm import tqdm
import random
from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, engine, DATABASE_URL, Base
from core.utils import _to_list


def execute_db_query(query, batch_data, info=""):
    """Replaces the entire power_db_worker loop. Executes database commands directly and synchronously."""
    # Convert pool URL to raw psycopg format if necessary
    dsn = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
    
    try:

        with psycopg.connect(dsn, autocommit=True) as conn:
            if "COPY" in query and ";" in query:
                with conn.transaction():
                    with conn.cursor() as cur:
                        parts = [p.strip() for p in query.split(";") if p.strip()]
                        copy_cmd, pre_cmds, post_cmds = "", [], []
                        found_copy = False
                        for p in parts:
                            if p.upper().startswith("COPY"):
                                copy_cmd = p
                                found_copy = True
                            elif not found_copy: pre_cmds.append(p)
                            else: post_cmds.append(p)

                        for cmd in pre_cmds: cur.execute(cmd)
                        if "FROM STDIN" not in copy_cmd.upper(): copy_cmd += " FROM STDIN"
                        with cur.copy(copy_cmd) as copy:
                            for row in batch_data:
                                copy.write("\t".join(map(str, row)) + "\n")
                        for cmd in post_cmds: cur.execute(cmd)
                        
            elif query.strip().upper().startswith("COPY"):
                with conn.cursor() as cur:
                    with cur.copy(query) as copy:
                        for row in batch_data:
                            copy.write("\t".join(map(str, row)) + "\n")
            else:
                with conn.cursor() as cur:
                    cur.execute(query, batch_data)
        return True
    except Exception as e:
        print(f"\n!!!! [DATABASE ERROR] Failed during {info}: {e} !!!!\n")
        return False
    
def get_unfiltered_series(limit: int) -> List[dict]:
    """Queries fred_series_filtered and returns a list of dictionaries."""
    query_string = "SELECT * FROM fred_series_filtered LIMIT :limit;"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_string), {"limit": limit})
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        tqdm.write(f" [!] Extraction Fallback Failure: {e}")
        return []



def get_vintage_stratified_set(
    sample_stratification: str, 
    exogeneity_set: List[str],
    endogenous_set: List[str],
    endogenous_set_size: int,
    seen_set : List[str],
    max_set_size: int, 
    random_seed: int, 
    observation_start_date: datetime, 
    observation_end_date: datetime, 
    as_of_date: datetime, 
    conn: Any
) -> List[dict]:
    """
    Bitemporal Stratification Selector:
    1. Grab all available series set X from as_of_date
    2. Let set Y =  set X  -  seen set - exogeneity set - endogenous set
    3. Let set Z_1 = exogeneity set  + endogenous set from Z_0 + uniform sampling with respect to sample_stratification from set Y until we reach max_set_size
    4. From set Z_1's explore set, sample endogenous_set_size, let this be the next batch's (Z_2) endogenous set. 
    5. Return Set Z_1 and Set Z_2 endogenous set. 
    """
    strat_column = str(sample_stratification).strip()
    if strat_column not in ["category_id"]:
        raise ValueError(f"Unsafe or unsupported stratification dimension targeted: '{strat_column}'")

    # 1. Calculate explore slots needed to reach max_set_size
    fixed_anchors_count = len(exogeneity_set) + len(endogenous_set)
    needed_for_z1_y = max(0, max_set_size - fixed_anchors_count)

    raw_exclusions = list(set(_to_list(seen_set) + _to_list(exogeneity_set) + _to_list(endogenous_set)))
    excluded_series_ids = [
        str(x).strip() for x in raw_exclusions 
        if isinstance(x, str) and len(str(x).strip()) > 1 and not str(x).endswith(".txt")
    ]
    if not excluded_series_ids:
        excluded_series_ids = ["__EMPTY_EXCLUSION_GUARD__"]

    random.seed(random_seed)
    pg_seed = (random_seed % 10000) / 10000.0 if random_seed != 0 else 0.5
    pg_seed = max(-1.0, min(1.0, pg_seed))

    # 2. SQL Query: Fetch candidates ONLY for Z_1's explore set
    query_string = f"""
        WITH historically_active_pool AS (
            SELECT DISTINCT obs.series_id_hash, unf.series_id
            FROM fred_observations obs
            INNER JOIN fred_series_unfiltered unf ON obs.series_id_hash = unf.series_id_hash
            WHERE obs.realtime_start <= :as_of_date
              AND (obs.realtime_end >= :as_of_date OR obs.realtime_end IS NULL)
              AND obs.date >= :start_date
              AND obs.date <= :end_date
              AND unf.series_id NOT IN :excluded_series_ids
        ),
        active AS (
            SELECT emb.*, hap.series_id_hash
            FROM fred_series_filtered emb
            INNER JOIN historically_active_pool hap ON emb.series_id = hap.series_id
            WHERE emb.{strat_column} IS NOT NULL
        ),
        RankedSeriesPerCluster AS (
            SELECT ae.*,
                   ROW_NUMBER() OVER(
                       PARTITION BY ae.{strat_column} 
                       ORDER BY RANDOM()
                   ) as slot_row_num
            FROM active ae
        )
        SELECT * 
        FROM RankedSeriesPerCluster 
        ORDER BY slot_row_num ASC, RANDOM()
        LIMIT :needed_for_z1_y;
    """

    try:
        conn.execute(text("SELECT SETSEED(:seed);"), {"seed": pg_seed})
        result = conn.execute(
            text(query_string), 
            {
                "start_date": observation_start_date,
                "end_date": observation_end_date,
                "as_of_date": as_of_date,
                "excluded_series_ids": tuple(excluded_series_ids),
                "needed_for_z1_y": needed_for_z1_y
            }
        )
        y_for_z1 = [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f" [!] Stratified Extraction Failure: {e}")
        y_for_z1 = []

    # 3. Sample Z_2 endogenous bridge set DIRECTLY FROM Z_1's explore pool
    sample_count = min(len(y_for_z1), endogenous_set_size)
    y_for_z2_endogenous = random.sample(y_for_z1, sample_count) if sample_count > 0 else []

    # 4. Construct Z_1 set
    z0_exogeneity_dicts = [{"series_id": s, "type": "exogenous_anchor"} for s in exogeneity_set]
    z0_endogenous_dicts = [{"series_id": s, "type": "endogenous_anchor"} for s in endogenous_set]

    z1_set = z0_exogeneity_dicts + z0_endogenous_dicts + y_for_z1

    return z1_set, y_for_z2_endogenous
   

def get_vintage_unstratified_set(
    exogeneity_set: List[str],
    observation_start_date: datetime, 
    observation_end_date: datetime, 
    as_of_date: datetime, 
    conn: Any
) -> Tuple[List[dict], List[dict]]:
    """
    Fetches vintage metadata strictly for the requested exogenous series set,
    verifying that observations were legally observable as of `as_of_date`
    within the target date window without random sampling.
    Prints a summary audit showing which series passed or failed observability.
    """
    if not exogeneity_set:
        return [], []

    # Clean input series list
    clean_exo_set = [
        str(s).strip() for s in exogeneity_set 
        if s and isinstance(s, str) and not str(s).endswith(".txt")
    ]

    if not clean_exo_set:
        return []

    query_string = """
        WITH historically_active_pool AS (
            SELECT DISTINCT obs.series_id_hash, unf.series_id
            FROM fred_observations obs
            INNER JOIN fred_series_unfiltered unf ON obs.series_id_hash = unf.series_id_hash
            WHERE obs.realtime_start <= :as_of_date
              AND (obs.realtime_end >= :as_of_date OR obs.realtime_end IS NULL)
              AND obs.date >= :start_date
              AND obs.date <= :end_date
              AND unf.series_id IN :exogeneity_set
        )
        SELECT emb.*, hap.series_id_hash
        FROM fred_series_filtered emb
        INNER JOIN historically_active_pool hap ON emb.series_id = hap.series_id
        ORDER BY emb.series_id ASC;
    """

    try:
        result = conn.execute(
            text(query_string), 
            {
                "start_date": observation_start_date,
                "end_date": observation_end_date,
                "as_of_date": as_of_date,
                "exogeneity_set": tuple(clean_exo_set)
            }
        )
        z1_exogeneity_dicts = [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f" [!] Unstratified Vintage Extraction Failure: {e}")
        z1_exogeneity_dicts = []

    # -------------------------------------------------------------------------
    # PASS / FAIL AUDIT REPORTING
    # -------------------------------------------------------------------------
    passed_ids = {item["series_id"] for item in z1_exogeneity_dicts if "series_id" in item}
    print(f" EXOGENEITY VINTAGE OBSERVABILITY AUDIT (AS OF: {as_of_date.strftime('%Y-%m-%d')})")
    for s_id in clean_exo_set:
        if s_id not in passed_ids:
            print(f"{s_id[:25]:<25} | {'N/A':<22} | {'FAILED':<10} | Unobservable/Missing in DB as of {as_of_date.strftime('%Y-%m-%d')}")

    print(f" TOTAL REQUESTED: {len(clean_exo_set)} | PASSED: {len(passed_ids)} | FAILED: {len(clean_exo_set) - len(passed_ids)}\n")

    return z1_exogeneity_dicts


def get_vintage_observations(
    series_id_hashes: List[int], 
    observation_start_date: datetime, 
    observation_end_date: datetime,  
    as_of_date: datetime
) -> List[dict]:
    """
    Pulls the latest known observation published ON OR BEFORE as_of_date
    without lookahead bias.
    """
    query_string = """
        WITH valid_records AS (
            SELECT 
                series_id_hash, 
                date, 
                value, 
                realtime_start, 
                realtime_end
            FROM fred_observations
            WHERE series_id_hash = ANY(:series_id_hashes)
              AND date >= :start_date
              AND date <= :end_date
              AND realtime_start <= :as_of_date
        )
        SELECT 
            series_id_hash, 
            date, 
            value, 
            realtime_start, 
            realtime_end
        FROM valid_records
        ORDER BY series_id_hash ASC, date ASC;
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(query_string), 
                {
                    "series_id_hashes": series_id_hashes,
                    "start_date": observation_start_date,
                    "end_date": observation_end_date,
                    "as_of_date": as_of_date
                }
            )
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f" [!] Point-in-Time Observation Extraction Failure: {e}")
        return []
    




def fetch_fold_data(
    exogeneity_set: List[str],
    current_start: pd.Timestamp, 
    train_start: pd.Timestamp, 
    conn: Any
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    if len(exogeneity_set) == 0:
        return None, None
    Z_1 = get_vintage_unstratified_set(
        exogeneity_set=exogeneity_set, 
        observation_start_date=train_start,
        observation_end_date=current_start,
        as_of_date=current_start, 
        conn=conn
    )

    if not Z_1:
        return  None, None

    missing_hash_series_ids = [
        s["series_id"] for s in Z_1 
        if isinstance(s, dict) and "series_id" in s and ("series_id_hash" not in s or s["series_id_hash"] is None)
    ]

    if missing_hash_series_ids:
        hash_query = text("""
            SELECT series_id, series_id_hash 
            FROM fred_series_filtered
            WHERE series_id IN :series_ids
        """)
        try:
            res = conn.execute(hash_query, {"series_ids": tuple(missing_hash_series_ids)})
            series_id_to_hash = {row.series_id: row.series_id_hash for row in res.fetchall()}
            for item in Z_1:
                if isinstance(item, dict) and "series_id" in item and ("series_id_hash" not in item or item["series_id_hash"] is None):
                    item["series_id_hash"] = series_id_to_hash.get(item["series_id"])
        except Exception as e:
            print(f" [!] series_id hash resolution failed for anchors: {e}")



    hash_to_series_id = {
        int(s["series_id_hash"]): s["series_id"]
        for s in Z_1
        if isinstance(s, dict) and s.get("series_id_hash") is not None and s.get("series_id")
    }
    # print(hash_to_series_id)
    train_series_id_hashes = [int(s["series_id_hash"]) for s in Z_1]


    train_obs_oracle = get_vintage_observations(
        series_id_hashes=train_series_id_hashes,
        observation_start_date=train_start,
        observation_end_date=current_start,
        as_of_date=datetime(9999, 12, 31)
    )


    for record in train_obs_oracle:
        if isinstance(record, dict) and "series_id_hash" in record:
            nh = int(record["series_id_hash"])
            record["series_id"] = hash_to_series_id.get(nh)
    return Z_1, train_obs_oracle



