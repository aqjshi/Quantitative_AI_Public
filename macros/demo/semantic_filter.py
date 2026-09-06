import os

from typing import Dict, List, Optional
import sys  
from tqdm import tqdm 

import csv 



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from core.db import FRED_KEY_0, engine, DATABASE_URL, Base
from macros.config import load_configuration


from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, text, DateTime, String, Text,
    JSON, Boolean, Numeric, Date, Index
)


def load_filter_terms(file_path: str) -> List[str]:
    """Reads a text filter file, strips comments/empty lines, and returns active regex terms."""
    if not os.path.exists(file_path):
        tqdm.write(f" [!] Warning: Filter file not found: {file_path}")
        return []
    
    terms = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return terms


def load_raw_file(file_path: str) -> str:
    """Loads raw text content from a SQL/policy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required configuration file missing: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def apply_filter_unfiltered_series(config: dict) -> List[dict]:
    """
    Dynamically loads modular text blacklists and SQL deduplication policies
    from disk based on the configuration dictionary, compiles the production 
    SQL query, and returns the filtered macro universe series.
    """
    filters_dir = os.path.join(PROJECT_ROOT, "macros", "filters")

    # Extract configuration settings
    filter_filenames = config.get("fred_semantic_filter_files", [])
    
    dedup_policy_filename = config.get("fred_dedup_policy", "deduplication.sql")
    if isinstance(dedup_policy_filename, list):
        dedup_policy_filename = dedup_policy_filename[0] if dedup_policy_filename else "deduplication.sql"

    base_query_filename = config.get("fred_base_query", "base_query.sql")
    if isinstance(base_query_filename, list):
        base_query_filename = base_query_filename[0] if base_query_filename else "base_query.sql"
    
    seasonal_whitewords = config.get("fred_seasonal_adjustment_short_whitewords", ["NSA", "NSAAR"])
    units_blackwords = config.get("units_blackwords", ["%Percent Change%", "%Growth Rate%"])
    title_blackwords = config.get("title_blackwords", [])

    # Compile Master Regex
    all_terms = []
    if isinstance(filter_filenames, str):
        filter_filenames = [filter_filenames]

    for filename in filter_filenames:
        file_path = os.path.join(filters_dir, filename)
        all_terms.extend(load_filter_terms(file_path))

    master_blacklist_regex = "|".join(all_terms) if all_terms else "a^"

    # Load SQL templates
    dedup_policy_path = os.path.join(filters_dir, dedup_policy_filename)
    base_query_path = os.path.join(filters_dir, base_query_filename)

    deduplication_policy_sql = load_raw_file(dedup_policy_path)
    base_query_template = load_raw_file(base_query_path)

    # Build SQL conditions
    formatted_whitewords = ", ".join([f"'{w}'" for w in seasonal_whitewords])
    seasonal_clause = f"seasonal_adjustment_short IN ({formatted_whitewords})"

    units_clauses = [f"units NOT ILIKE '{bw}'" for bw in units_blackwords]
    units_condition = " AND ".join(units_clauses) if units_clauses else "1=1"

    title_clauses = [f"title NOT ILIKE '{bw}'" for bw in title_blackwords]
    title_blackwords_condition = " AND ".join(title_clauses) if title_clauses else "1=1"

    # Inject dynamic parameters
    try:
        query_string = base_query_template.format(
            deduplication_policy=deduplication_policy_sql,
            master_blacklist_regex=master_blacklist_regex,
            seasonal_whitelist_clause=seasonal_clause,
            units_blackwords_clause=units_condition,
            title_blackwords_clause=title_blackwords_condition
        )
    except KeyError:
        query_string = f"""
        WITH DeduplicatedFRED AS (
            SELECT 
                id, category_id, name, title, units, units_short,
                frequency_short, seasonal_adjustment_short, popularity,
                {deduplication_policy_sql}
            FROM fred_series_unfiltered
            WHERE 
                {seasonal_clause}
                AND {units_condition}
                AND {title_blackwords_condition}
                AND NOT title ~* (
                    '\\y(' ||
                    '{master_blacklist_regex}' ||
                    ')\\y'
                )
        )
        SELECT * 
        FROM DeduplicatedFRED 
        WHERE title_rank = 1 
        ORDER BY popularity DESC;
        """

    # Execute query
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_string))
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        tqdm.write(f" [!] Extraction Fallback Failure: {e}")
        return []


def export_to_tsv(data: List[dict], output_filepath: str):
    """
    Exports filtered series dictionary records to a tab-delimited CSV file (TSV)
    containing: id, popularity, frequency_short, units_short, seasonal_adjustment_short, name
    """
    fieldnames = ["id", "popularity", "frequency_short", "units_short", "seasonal_adjustment_short", "name", "title"]
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
        
    print(f"📁 Tab-delimited file written successfully ({len(data)} rows): {output_filepath}")


# --- MAIN PIPELINE EXECUTION ---
def main():
    config = load_configuration()
    
    # 1. Fetch filtered series winners
    raw_filtered_winners = apply_filter_unfiltered_series(config)
    print(f"✅ Filtered macro universe down to {len(raw_filtered_winners)} tier-1 series.")

    # 2. Define output filepath inside macros/demo/
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_tsv_path = os.path.join(demo_dir, "filtered_macro_series.tsv")

    # 3. Save tab-delimited CSV output
    if raw_filtered_winners:
        export_to_tsv(raw_filtered_winners, output_tsv_path)

if __name__ == "__main__":
    main()