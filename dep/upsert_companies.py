import os
import requests
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Company, Base

def get_all_active_tickers():
    """Fetches every active CS ticker from Polygon."""
    print("[*] Stage 1: Discovering ALL active tickers in the market...")
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "CS",
        "active": "true",
        "limit": 1000,
        "apiKey": POLY_KEY
    }
    
    all_tickers = []
    while url:
        resp = requests.get(url, params=params if "apiKey" not in url else None)
        if resp.status_code != 200: break
        data = resp.json()
        all_tickers.extend([r['ticker'] for r in data.get("results", [])])
        
        next_url = data.get("next_url")
        url = f"{next_url}&apiKey={POLY_KEY}" if next_url else None
    
    print(f"[+] Found {len(all_tickers)} active tickers.")
    return all_tickers

def fetch_ratios_batch(ticker_chunk):
    """Queries Massive for Market Cap and Volume in bulk (ticker.any_of)."""
    ticker_str = ",".join(ticker_chunk)
    url = "https://api.massive.com/stocks/financials/v1/ratios"
    params = {
        "ticker.any_of": ticker_str,
        "limit": 1000,
        "apiKey": POLY_KEY 
    }
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        print(f" [!] Batch fetch failed: {e}")
    return []

def get_enriched_whales(all_tickers, target_k):
    """Sorts the entire market by Market Cap in-memory."""
    chunk_size = 100
    ticker_chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]
    
    raw_whale_data = []
    print(f"[*] Stage 2: Scanning {len(ticker_chunks)} batches for Market Significance...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_ratios_batch, chunk) for chunk in ticker_chunks]
        for future in as_completed(futures):
            results = future.result()
            for r in results:
                raw_whale_data.append({
                    "ticker": r.get("ticker"),
                    "cik": int(r.get("cik")) if r.get("cik") else None,
                    "volume": int(r.get("average_volume", 0)),
                    "market_cap": r.get("market_cap") or 0
                })

    raw_whale_data.sort(key=lambda x: x['market_cap'], reverse=True)

    # 2. Reproducible Deduplication: 1 Ticker per CIK
    unique_ciks = {} # Map CIK -> Best Whale Data
    
    for whale in raw_whale_data:
        cik = whale.get('cik')
        if not cik: continue # Skip if no identity
        
        if cik not in unique_ciks:
            unique_ciks[cik] = whale
        else:
            # If CIK exists, keep the one with higher trading volume
            if whale['volume'] > unique_ciks[cik]['volume']:
                unique_ciks[cik] = whale

    # 3. Convert back to list and take top K
    final_candidates = list(unique_ciks.values())
    final_candidates.sort(key=lambda x: x['market_cap'], reverse=True)
    
    return final_candidates[:target_k]

def enrich_names(whale_list):
    """Final identity check for the Top K winners."""
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

    print(f"[*] Stage 3: Adding names to the top {len(whale_list)} Whales...")
    final_enriched = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_name, w) for w in whale_list]
        for f in as_completed(futures):
            res = f.result()
            if res and res.get('cik'):
                final_enriched.append(res)
    return final_enriched

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python discovery.py <params.json> <target_k>")
        sys.exit(1)

    conf_path = sys.argv[1]
    k_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    
    # 1. Database Setup
    if len(sys.argv) > 3 and sys.argv[3] == "1":
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # 2. Logic Flow
    full_market = get_all_active_tickers()
    top_whale_candidates = get_enriched_whales(full_market, k_limit)
    final_whales = enrich_names(top_whale_candidates)

    # 3. DB Injection
    print(f"[*] Stage 4: Injecting {len(final_whales)} Whales into DB...")
    with SessionLocal() as session:
        for row in final_whales:
            stmt = pg_insert(Company).values(row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["cik"],
                set_={k: getattr(stmt.excluded, k) for k in ["ticker", "name", "volume", "market_cap"]}
            )
            session.execute(stmt)
        session.commit()

    # --- THE PRESERVATION GATE ---
    # Load the existing JSON to preserve train_start, alpha, etc.
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            try:
                params = json.load(f)
            except json.JSONDecodeError:
                params = {}
    else:
        params = {}

    # Only update the whitelist key
    params["ticker_whitelist"] = [w['ticker'] for w in final_whales]

    # Save it back with all other parameters intact
    with open(conf_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\n[SUCCESS] Manifold synced. {len(final_whales)} tickers updated in {conf_path} (other params preserved).")