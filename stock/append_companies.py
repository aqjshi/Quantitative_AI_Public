import os
import requests
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Company, Quote, Base

def fetch_ratios_batch(ticker_chunk):
    """Queries Massive for Market Cap and Volume for specific tickers."""
    ticker_str = ",".join(ticker_chunk)
    url = "https://api.massive.com/stocks/financials/v1/ratios"
    params = {"ticker.any_of": ticker_str, "limit": 1000, "apiKey": POLY_KEY}
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        print(f" [!] Batch fetch failed: {e}")
    return []

def enrich_names(whale_list):
    """Fetches official names/CIKs from Polygon for the manual list."""
    def fetch_name(whale):
        ticker = whale['ticker']
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLY_KEY}"
        try:
            res = requests.get(url, timeout=5).json().get("results", {})
            whale['name'] = res.get("name", ticker)
            if not whale['cik'] and res.get("cik"):
                whale['cik'] = int(res.get("cik"))
            return whale
        except:
            whale['name'] = ticker
            return whale

    print(f"[*] Enriching metadata for {len(whale_list)} companies...")
    final_enriched = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_name, w) for w in whale_list]
        for f in as_completed(futures):
            res = f.result()
            if res and res.get('cik'):
                final_enriched.append(res)
    return final_enriched

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sync_companies.py <params.json>")
        sys.exit(1)

    conf_path = sys.argv[1]
    
    # 1. Load existing JSON
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            try:
                params = json.load(f)
            except json.JSONDecodeError:
                params = {}
    else:
        params = {}

    manual_additions = params.get("adding_companies", [])
    
    # 2. FETCH DATA - Try Massive first, then fallback to Polygon for missing ones
    print(f"[*] Processing {len(manual_additions)} manual additions...")
    raw_data = fetch_ratios_batch(manual_additions)
    found_tickers = {r.get("ticker") for r in raw_data}
    
    formatted_data = []
    for r in raw_data:
        formatted_data.append({
            "ticker": r.get("ticker"),
            "cik": int(r.get("cik")) if r.get("cik") else None,
            "volume": int(r.get("average_volume", 0)),
            "market_cap": r.get("market_cap") or 0
        })

    # FALLBACK: If a manual ticker wasn't in Massive, add a skeleton entry for Polygon to enrich
    for ticker in manual_additions:
        if ticker not in found_tickers:
            formatted_data.append({"ticker": ticker, "cik": None, "volume": 0, "market_cap": 0})

    # 3. Final Enrichment (Polygon) - This will now catch LYFT even if Massive missed it
    final_to_upsert = enrich_names(formatted_data)

    # 4. DB Operations
    with SessionLocal() as session:
        print(f"[*] Upserting {len(final_to_upsert)} companies to DB...")
        upserted_ciks = []
        for row in final_to_upsert:
            upserted_ciks.append(row['cik'])
            stmt = pg_insert(Company).values(row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["cik"],
                set_={k: getattr(stmt.excluded, k) for k in ["ticker", "name", "volume", "market_cap"]}
            )
            session.execute(stmt)
        
        # B. Drop companies with NO quotes, BUT PROTECT the ones we just added
        print("[*] Cleaning up old companies without quote data...")
        subq = select(Quote.company_cik).distinct().scalar_subquery()
        
        # We delete if: (No quotes exist) AND (It wasn't one of the companies we just manually added/updated)
        delete_stmt = delete(Company).where(
            Company.cik.not_in(subq),
            Company.cik.not_in(upserted_ciks) # Protection layer
        )
        session.execute(delete_stmt)
        session.commit()

        final_tickers = session.execute(select(Company.ticker)).scalars().all()

    # 5. Update JSON
    params["ticker_whitelist"] = sorted(list(set(final_tickers)))
    with open(conf_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\n[SUCCESS] Database cleaned and JSON updated.")
    print(f"[+] Total Whitelisted Tickers: {len(params['ticker_whitelist'])}")