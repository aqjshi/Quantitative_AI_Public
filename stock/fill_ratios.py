import sys
import os
import json
from sqlalchemy import create_engine, inspect
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import DATABASE_URL, POLY_KEY
from core.models import Base, Company, Ratios # Import the model you'll be editing
import core.sieve as sieve

def main():
    
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    all_tickers = params.get("ticker", [])
    params["train_start"] = params.get("train_start").split(' ')[0]
    params["train_end"] = params.get("train_end").split(' ')[0]
    params["timezone_str"] = params.get("timezone_str").split(' ')[0]

    
    
    engine = create_engine(DATABASE_URL)

    print(f"[*] Starting Global Sieve Discovery for {len(all_tickers)} tickers...")
    print("[*] Selective Reset: Dropping only 'ratios'...")

    # 1. Access the specific table object from the metadata
    target_table = Base.metadata.tables.get('ratios')

    if target_table is not None:
        # 2. Drop only this table
        target_table.drop(engine, checkfirst=True)
        print("[+] Old 'ratios' table cleared.")

    # 3. Create everything (it will only create what is missing)
    Base.metadata.create_all(engine)

    
    # Fetching everyone into memory
    policy, data_cache = sieve.generate_sieve_policy(all_tickers, POLY_KEY,  fundamental_name= "ratios", limit=5000, alpha=.05)
    
    if not policy or not data_cache:
        print("[-] Discovery failed.")
        return

    # Show the health plots (The Histogram and Density Map)
    sieve.plot_sieve_health(data_cache, policy, plot_name="Ratios Discovery", output_dir= "stock")

    # --- THE INTERCEPT ---
    print("\n" + "="*60)
    print("PROPOSED BASIS VECTORS (Density >= 5%):")
    for key in sorted(policy['key_whitelist']):
        print(f"  - {key}")
    print("="*60)
    
    print("\n[!] ACTION REQUIRED:")
    print("1. Review the histogram and the list above.")
    print("2. Update your 'ratios' class in core/models.py with any missing columns.")
    print("3. Save models.py.")
    input("\n[?] Ready to ingest PLEASE DROP THE TABLES IN SQL FIRST? Press ENTER to re-map the model and start UPSERT...")

    # --- PASS 2: RE-MAP & INGEST ---
    print("[*] Re-mapping model and creating tables...")
    # This ensures SQLAlchemy sees your new columns without restarting the process
    Base.metadata.create_all(engine)
    
    tickers_with_data = list(data_cache.keys())
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    total_synced = 0

    print(f"[*] Ingesting {len(tickers_with_data)} companies...")

    for i in range(0, len(tickers_with_data), chunk_size):
        ticker_batch = tickers_with_data[i:i + chunk_size]
        batch_cache = {t: data_cache[t] for t in ticker_batch}
        
        synced = sieve.execute_sieve_upsert(
            engine=engine,
            model=Ratios, # The logic now uses the UPDATED class
            company_model=Company,
            policy=policy,
            data_cache=batch_cache
        )
        total_synced += synced
        print(f"[+] Chunk {i//chunk_size + 1} complete. Total: {total_synced}")

    print(f"\n[!] SUCCESS: Global Sync complete. {total_synced} records updated.")

if __name__ == "__main__":
    main()