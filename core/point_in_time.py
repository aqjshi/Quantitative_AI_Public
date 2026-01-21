import datetime
from datetime import datetime, timedelta
import string
import re
import os 
import functools
import json
import requests
from dateutil.relativedelta import relativedelta


BASE_URL = "https://api.polygon.io/v3/reference/tickers"


class TickerDiscovery:
    def __init__(self, api_key, limiter, cache_dir="stock/cache/"):
        
        self.api_key = api_key
        self.limiter = limiter
        self.cache_dir = cache_dir
        self.found_instruments = {} 
        self.shadow_regex = re.compile(r"(EX-|\d{2,4}$|[A-Z]\d{1,2}$|INTRA-DAY|INAV|HEDGED)", re.IGNORECASE)
        self.shadow_keywords = {"TR", "NTR", "TOTAL RETURN", "NET RETURN", "GROSS RETURN", "JAN", "FEB", "MAR", "ETF"}
        self.exchange_whitelist = {
            'XNYS', 'XNAS', 'ARCX', 'BATS', 'XTSE', 'XLON', 
            'XETR', 'XPAR', 'XAMS', 'XJPX', 'XHKG', 'XASX', 'XNSE'
        }
        
        # Ensure physical storage path exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, ticker, date_str):
        """Creates a path like stocks/AAPL/2023-01-01.json"""
        ticker_dir = os.path.join(self.cache_dir, ticker.upper())
        if not os.path.exists(ticker_dir):
            os.makedirs(ticker_dir)
        return os.path.join(ticker_dir, f"{date_str}.json")

    @functools.lru_cache(maxsize=50000) # Increased for larger universes
    def get_pit_state(self, ticker, date_str):
        cache_path = self._get_cache_path(ticker, date_str)

        # 1. Physical Disk Check
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f" [!] Cache read error for {ticker} on {date_str}: {e}")

        # 2. API Fetch (The "Network Tax" paid only once)
        self.limiter.wait()
        url = f"{BASE_URL}/{ticker}"
        params = {"date": date_str, "apiKey": self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                raw_data = resp.json().get("results", {})
                
                # Extract CIK clean
                cik = raw_data.get("cik")
                if cik: cik = str(cik).lstrip('0')
                
                parsed_state = {
                    "ticker": raw_data.get("ticker"),
                    "composite_figi": raw_data.get("composite_figi", "UNKNOWN"),
                    "share_class_figi": raw_data.get("share_class_figi", "UNKNOWN"),
                    "cik": cik,
                    "name": raw_data.get("name"),
                    "active": raw_data.get("active", False),
                    "list_date": raw_data.get("list_date")
                }

                # 3. Store Physically Immediately
                with open(cache_path, 'w') as f:
                    json.dump(parsed_state, f)

                return parsed_state
            
            # Handle 404/Missing data as a "Valid Negative" to avoid re-fetching
            elif resp.status_code == 404:
                negative_state = {"ticker": ticker, "composite_figi": "NONE", "active": False}
                with open(cache_path, 'w') as f:
                    json.dump(negative_state, f)
                return negative_state

        except Exception as e:
            print(f" [!] API Error for {ticker}: {e}")
            
        return {"ticker": ticker, "composite_figi": "NONE", "active": False}

    def is_junk(self, ticker, name=""):
        t_upper = (ticker or "").upper()
        combined = f"{t_upper} {(name or '').upper()}"
        if self.shadow_regex.search(combined): return True
        words = set(combined.split())
        return not words.isdisjoint(self.shadow_keywords)

    def get_next_prefix(self, prefix):
        if not prefix: return None
        last_char = prefix[-1].upper()
        if last_char == 'Z': return None 
        return prefix[:-1] + chr(ord(last_char) + 1)

    def fetch_recursive(self, prefix="", ticker_type="CS", market="stocks",depth=5, active=True):
        if prefix and prefix[0].isdigit(): return
        if depth <= 0: return

        next_prefix = self.get_next_prefix(prefix) 
        self.limiter.wait()
        
        params = {
            "market": market , "type": ticker_type, "active": "true" if active else "false",
            "sort": "ticker", "limit": 1000, "apiKey": self.api_key,
            "ticker.gte": prefix if prefix else "A"
        }
        if next_prefix: params["ticker.lt"] = next_prefix

        try:
            resp = requests.get(BASE_URL, params=params)
            data = resp.json()
            results = data.get("results", [])
            for item in results:
                # 1. Grab the exchange and FIGI
                exchange = item.get('primary_exchange')
                figi = item.get('composite_figi')
                
                # 2. THE QUALITY GATE: 
                # Skip if it's not on our whitelist or if it's 'junk'
                if exchange not in self.exchange_whitelist:
                    continue
                    
                if figi and not self.is_junk(item.get('ticker'), item.get('name')):
                    self.found_instruments[figi] = {**item, "is_active": active}
            if len(results) == 1000:
                for char in string.ascii_uppercase:
                    self.fetch_recursive(prefix + char, ticker_type,market, depth - 1, active)
        except Exception as e:
            print(f" [!] Error at {prefix}: {e}")





def resolve_history(ticker, metadata, start_backtest, end_backtest, discovery_engine, heatbeat_freq_days=180):
    search_start_dt = datetime.strptime(start_backtest, "%Y-%m-%d")
    search_end_dt = datetime.strptime(end_backtest, "%Y-%m-%d")

    # 1. Standard Anchors (Start/End)
    anchors = {search_start_dt, search_end_dt}

    # 2. Metadata Hints (If available)
    if metadata.get('list_date'):
        try:
            ld = datetime.strptime(metadata['list_date'], "%Y-%m-%d")
            if search_start_dt < ld < search_end_dt: anchors.add(ld)
        except: pass
    if metadata.get('delisted_utc'):
        try:
            dd = datetime.strptime(metadata['delisted_utc'].split('T')[0], "%Y-%m-%d")
            if search_start_dt < dd < search_end_dt: anchors.add(dd)
        except: pass


    current_heartbeat = search_start_dt
    while current_heartbeat < search_end_dt:
        # Advance by 365 days (approx 1 year)
        current_heartbeat += timedelta(days=heatbeat_freq_days)
        
        # Stop if we've overshot the end date
        if current_heartbeat >= search_end_dt:
            break
            
        # Add the heartbeat to our anchors
        anchors.add(current_heartbeat)


    sorted_anchors = sorted(list(anchors))
    raw_segments = []

    def probe(d1, d2):
        s1 = discovery_engine.get_pit_state(ticker, d1)
        s2 = discovery_engine.get_pit_state(ticker, d2)

        # 1. BOTH NONE: Skip the interval (Heartbeat handles the risk of missing a 'short' life)
        if s1['composite_figi'] == "NONE" and s2['composite_figi'] == "NONE":
            return

        # 2. METADATA SNAP (The Speed Hack):
        # If it was born inside this window, snap to the IPO date and stop recursing.
        if s1['composite_figi'] == "NONE" and s2['composite_figi'] != "NONE":
            actual_ipo = s2.get('list_date')
            if actual_ipo and d1 < actual_ipo < d2: # Note: strictly less than d2
                probe(actual_ipo, d2) 
                return

        # 3. CONTINUITY: Same asset at both ends? Bridge the gap.
        if s1['composite_figi'] == s2['composite_figi'] and s1['ticker'] == s2['ticker']:
            if s1['composite_figi'] not in ["NONE", "UNKNOWN"]:
                raw_segments.append({
                    "ticker": s1['ticker'], "figi": s1['composite_figi'],
                    "valid_from": d1, "valid_to": d2
                })
            return

        # 4. RECURSION: Only if we haven't found a clean bridge or snap (e.g., Ticker Change)
        t1 = datetime.strptime(d1, "%Y-%m-%d")
        t2 = datetime.strptime(d2, "%Y-%m-%d")
        delta = (t2 - t1).days
        
        if delta <= 1:
            # Base case: we are at the edge of a change
            if s1['composite_figi'] not in ["NONE", "UNKNOWN"]:
                raw_segments.append({
                    "ticker": s1['ticker'], "figi": s1['composite_figi'],
                    "valid_from": d1, "valid_to": d1
                })
            return 
        
        mid_str = (t1 + timedelta(days=delta // 2)).strftime("%Y-%m-%d")
        probe(d1, mid_str)
        probe(mid_str, d2)

    # Probe each anchored interval
    for i in range(len(sorted_anchors) - 1):
        probe(sorted_anchors[i].strftime("%Y-%m-%d"), sorted_anchors[i+1].strftime("%Y-%m-%d"))

    if not raw_segments: return []

    unique_map = {}
    for seg in raw_segments:
        unique_map[f"{seg['valid_from']}_{seg['figi']}"] = seg
    
    cleaned = list(unique_map.values())
    cleaned.sort(key=lambda x: x['valid_from'])

    merged = []
    for seg in cleaned:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg['ticker'] == last['ticker'] and seg['figi'] == last['figi']:
            last['valid_to'] = max(last['valid_to'], seg['valid_to'])
        else:
            merged.append(seg)
    return merged

def consolidate_and_link_history(full_history_map):
    print(" [***] Running Global FIGI-Stitching...")
    all_segments = []
    for ticker_str, segments in full_history_map.items():
        all_segments.extend(segments)

    groups = {}
    for seg in all_segments:
        figi = seg.get('figi')
        if not figi or figi == 'NONE': continue
        if figi not in groups: groups[figi] = []
        groups[figi].append(seg)

    final_timeline = []
    for figi, group in groups.items():
        group.sort(key=lambda x: x['valid_from'])
        
        for i, curr in enumerate(group):
            if i == 0:
                curr['change_event'] = 'IPO'
                curr['previous_ticker'] = None
                curr['previous_composite_figi'] = None
            else:
                prev = group[i-1]
                if curr['ticker'] != prev['ticker']:
                    curr['change_event'] = 'SYMBOL_CHANGE'
                    curr['previous_ticker'] = prev['ticker']
                    curr['previous_composite_figi'] = prev['figi']
                    prev['valid_to'] = curr['valid_from']
                else:
                    curr['change_event'] = 'CONTINUATION'
                    curr['previous_ticker'] = prev['previous_ticker']
                    curr['previous_composite_figi'] = prev.get('previous_composite_figi')
            final_timeline.append(curr)

    return final_timeline




def snap_to_quarter_start(dt):
    """
    Floors a date to the first day of the quarter.
    Example: Feb 15 -> Jan 01
    """
    quarter_start_month = ((dt.month - 1) // 3) * 3 + 1
    return dt + relativedelta(month=quarter_start_month, day=1)


