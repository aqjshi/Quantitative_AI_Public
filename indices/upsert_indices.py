import os
import requests
import sys
import json
import string
import time
import threading
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text
from tqdm import tqdm

# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, SessionLocal, engine
from core.models import Index, Base

BASE_URL = "https://api.polygon.io/v3/reference/tickers"
AGGS_URL = "https://api.polygon.io/v2/aggs/ticker"

class TokenBucketRateLimiter:
    def __init__(self, rate_per_sec):
        self.delay = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_call = 0

    def wait(self):
        with self.lock:
            now = datetime.now().timestamp()
            if self.next_call > now:
                time.sleep(self.next_call - now)
            self.next_call = max(self.next_call, now) + self.delay

class IndexDiscovery:
    def __init__(self, api_key, tree_path, limiter, test_tickers=None):
        self.api_key = api_key
        self.tree_path = tree_path
        self.limiter = limiter
        self.test_tickers = test_tickers or []
        self.found_indices = {}
        self.prefix_leaves = [] 
        self.shadow_keywords = ["TR", "NTR", "TOTAL RETURN", "NET RETURN", "GROSS RETURN"]

    def is_shadow_index(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        n_upper = (name or "").upper()
        return any(k in t_upper or k in n_upper for k in self.shadow_keywords)

    def is_junk(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        n_upper = (name or "").upper()
        clean_t = t_upper.replace("I:", "") # Define this EARLY
        
        # 1. WHREAT PROTECTION
        if t_upper in self.test_tickers:
            return False
        

        BLACKLIST = [
            "CHINA",
            "DOW JONES U.S.", "RUSSELL 1000", "S&P 100", "Q-50", "NASDAQ CANADA", "NASDAQ SWITZERLAND", 
            "NASDAQ GERMANY", "NASDAQ UK", "NASDAQ HONG KONG", 
            "NASDAQ INDIA", "NASDAQ JAPAN", "NASDAQ KOREA", 
            "NASDAQ SINGAPORE", "NASDAQ TAIWAN", "NASDAQ NORWAY", 
            "NASDAQ MEXICO", "NASDAQ N AMERICA", "DOW JONES SHANGHAI", 
            "DOW JONES SHENZHEN", "IPOX CANADA",
            "UTILITY", "COMMODITY", "COMPUTER", "PROPERTY", "INTERNET", "OPTION",
            "SECTOR", "SELECT", "COMPOSITE", "REIT", "WEALTH", "ESTATE", "SERVICES", 
            "BUYWRITE", "OTM", "DELTA", "CONDITIONAL", "MANAGED VOLATILITY",
            "IRON CONDOR", "TAIL RISK", "DYNAMIC", "PUTWRITE", "PROTECT", 
            "ENHANCED", "RISK MANAGED", "SETTLEMENT", "SKEWDEX", "TAILDEX",
            "VOLDEX", "TAIL HEDGE", "INAV", "ETF", "UCITS", "INVERSE", "LEVERAGED", 
            "TOTAL",  "COMMUNITY", "BANK", "BUTTERFLY",  "COLLAR", "IRON", "LEADERS",  "GROWTH", 
            "FUTURES", "MARK INDEX",
            "MINI", "NANOS", "SMID", "MICRO",
            "INTEREST RATE", "T NOTE", "BOND",
            "SHORT", "ANNUAL", "NATURAL",  "HEALTH",
            "EAFE", "EMERGING", "INSURANCE", "TELECOMMUNICATIONS", "FINANCIAL", "CAPITAL", "BIOTECHNOLOGY",
            "EQUAL WEIGHTED", "DISPERSION", "INNOVATORS", "JUMBO",
            "TEST", "INDICATOR", "AFTER HOURS", "PRE MARKET",
            "NEXT",  "GENERATION",  "DEVELOPED",  "REDUCED", "COVERED", "INDICATIVE", "ASK", "BID", "OPTIONS", "EURO",
            "HIGH", "MICRO", "VIX", "VOLATILITY", "SKEW", "RISK", "VVIX", "VSTN", "VSTF",
            "EX-TECHNOLOGY", "EX-HEALTH", "BUFFERED", "BUFFER", "ZERO", "APR", "INDEX SERIES", "APR", "DEC", "FEB", "AUG", "JAN"
        ]
        
        if any(s in n_upper for s in BLACKLIST):
            return True

        # 3. THE "REGIONAL & COUNTRY" KILLER
        REGIONAL_TRASH = [
            "NORWAY", "MEXICO",
            "COLOMBIA", "GREECE", "ITALY", "NETHERLANDS", "BRAZIL", "ISRAEL",
            "BELGIUM", "CHILE", "CZECH", "SPAIN", "HUNGARY", 
            "INDONESIA", "IRELAND", "LAT AMERICA", 
            "MOROCCO", "MALAYSIA", "NEW ZEALAND", "PERU", 
            "PHILIPPINES", "POLAND", "PORTUGAL", "SWEDEN", 
            "THAILAND", "SOUTH AFRICA", "DENMARK", "FINLAND",
            "GLOBAL", "INTERNATIONAL",
            "ASIA", "BRITISH POUND", "YEN", "FRANC", "DOLLAR",
            "CAD", "MONTHLY CURRENCY HEDGED", 
        ]
        if any(r in n_upper for r in REGIONAL_TRASH):
            return True

        # 4. THE "FACTOR & PRODUCT" KILLER
        FACTOR_TRASH = [
            
            "MORNINGSTAR", "ACHIEVERS", "BUYBACK", "WIDE MOAT", "DIVIDEND", 
            "ESG", "ISLAMIC", "SHARIA", "CLEAN EDGE", "WATER", "CYBER", 
            "CLOUD", "BITCOIN", "CRYPTO", "ETHEREUM", "ALTCOIN", "SMARTPHONE",
            "INNOVATIVE", "GREEN", "FACTOR TILT", "EX-ENERGY", "EX-FINANCIALS"
        ]
        if any(f in n_upper for f in FACTOR_TRASH):
            return True
        blacklist_tickers = ["DJS"]
        if clean_t in blacklist_tickers:
            return True
        # 5. ELITE ANCHORS
        # Only keep it if it's a PURE master or a very short Alpha-root (like DJI, SPX, BKX)
        MASTERS = ["DOW JONES INDUSTRIAL", "S&P 500", "NASDAQ 100", "NASDAQ COMPOSITE", "RUSSELL 2000", "CBOE VOLATILITY"]
        is_elite = any(m in n_upper for m in MASTERS)
        
        clean_t = t_upper.replace("I:", "")
        if is_elite:
            # Re-run shadow check to ensure we don't get 'S&P 500 TOTAL RETURN'
            return self.is_shadow_index(t_upper, n_upper)
            
        # If it's not a master, it must be a 3-4 char Alpha code (e.g., BKX, SOX, HGX)
        if len(clean_t) <= 4 and not any(c.isdigit() for c in clean_t):
            return self.is_shadow_index(t_upper, n_upper)
        
   
        return True
    def process_results(self, results):
        for item in results:
            ticker = item.get('ticker', '')
            name = (item.get('name', '') or '').upper()
            if ticker and not self.is_junk(ticker, name):
                self.found_indices[ticker] = {"theoretical_symbol": ticker, "name": name}

    def fetch_recursive(self, prefix="I:"):
        self.limiter.wait()
        params = {"market": "indices", "search": prefix, "active": "true", "limit": 1000, "apiKey": self.api_key}
        try:
            resp = requests.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if len(results) < 1000:
                self.prefix_leaves.append(prefix) 
                self.process_results(results)
            else:
                for char in string.ascii_uppercase + string.digits:
                    self.fetch_recursive(prefix + char)
        except Exception as e:
            print(f" [!] Error crawling {prefix}: {e}")

    def inject_and_verify_tests(self):
        for t in self.test_tickers:
            self.limiter.wait()
            try:
                r = requests.get(f"{BASE_URL}/{t}?apiKey={self.api_key}").json()
                if "results" in r:
                    res = r["results"]
                    self.found_indices[t] = {"theoretical_symbol": res["ticker"], "name": res["name"]}
                    print(f"[*] DJI/SPX/NDX Persistence Check: {t} is LOCKED.")
            except: continue

    def validate_ticker(self, ticker):
        if ticker in self.test_tickers:
            return ticker
        self.limiter.wait()
        url = f"{AGGS_URL}/{ticker}/range/1/day/2024-01-01/2024-01-10"
        params = {"sort": "asc", "limit": 10, "apiKey": self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200 and resp.json().get("results"):
                return ticker
        except: pass
        return None

    def load_or_crawl(self):
        if os.path.exists(self.tree_path):
            with open(self.tree_path, 'r') as f:
                data = json.load(f)
                for pref in tqdm(data.get("safe_prefixes", []), desc="Loading Prefix Tree"):
                    self.limiter.wait()
                    try:
                        r = requests.get(BASE_URL, params={"market":"indices","search":pref,"active":"true","limit":1000,"apiKey":self.api_key}).json()
                        self.process_results(r.get("results", []))
                    except: continue
        else:
            self.fetch_recursive("I:")
            self.save_expansion_tree()
        self.inject_and_verify_tests()

    def save_expansion_tree(self):
        os.makedirs(os.path.dirname(self.tree_path), exist_ok=True)
        with open(self.tree_path, 'w') as f:
            json.dump({"market": "indices", "safe_prefixes": self.prefix_leaves}, f, indent=4)

def upsert_to_db(data_list):
    if not data_list: return
    with SessionLocal() as session:
        for row in data_list:
            # Map the tradable symbol before inserting
            theo = row['theoretical_symbol']
            row['tradable_proxy'] = "PENDING"
            stmt = pg_insert(Index).values(row)
            # FIXED: Match the unique constraint in your Index model
            stmt = stmt.on_conflict_do_update(
                index_elements=["theoretical_symbol"], 
                set_={"name": stmt.excluded.name, "tradable_proxy": stmt.excluded.tradable_proxy}
            )
            session.execute(stmt)
        session.commit()

if __name__ == "__main__":
    limiter = TokenBucketRateLimiter(rate_per_sec=20)
    tree_path = "indices/expansion_tree.json"
    test_tickers = ["I:DJI", "I:SPX", "I:NDX", "I:RUT",  "I:VIX", "I:TNX", "I:SOX"]

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS indices CASCADE"))
        conn.commit()
    Base.metadata.create_all(engine)

    discovery = IndexDiscovery(POLY_KEY, tree_path, limiter, test_tickers=test_tickers)
    discovery.load_or_crawl()
    
    all_raw = list(discovery.found_indices.keys())
    validated_tickers = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(discovery.validate_ticker, t): t for t in all_raw}
        for future in tqdm(as_completed(futures), total=len(all_raw), desc="Final Sync"):
            res = future.result()
            if res: validated_tickers.append(res)

    final_data = [discovery.found_indices[t] for t in validated_tickers]
    
    found_symbols = {item['theoretical_symbol'] for item in final_data}
    missing = set(test_tickers) - found_symbols
    if missing:
        print(f"\n[!!!] FATAL ERROR: {missing} was lost. Aborting DB sync.")
        sys.exit(1)

    upsert_to_db(final_data)
    print(f"\n[SUCCESS] Sync complete. Cleaned {len(final_data)} primary roots.")