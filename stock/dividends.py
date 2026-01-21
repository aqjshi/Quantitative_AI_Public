import requests
import os 
import json

def get_cache_path(ticker, start_date, end_date, cache_dir):
    """
    Creates a path based on ticker and the date range to ensure 
    we don't load a partial year's cache for a full year's request.
    """
    ticker_dir = os.path.join(cache_dir, ticker.upper())
    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)
    # File name includes start and end to prevent date-range mismatches
    return os.path.join(ticker_dir, f"{start_date}_to_{end_date}.json")

def fetch_and_map_dividends(ticker, tm_lookup, api_key, start_date, end_date, cache_dir="stock/cache_dividends/"):
    """
    Checks cache for raw data. If missing, fetches from API and saves to cache.
    Then performs mapping to instrument_ids.
    """
    cache_path = get_cache_path(ticker, start_date, end_date, cache_dir)
    raw_divs = []

    # 1. ATTEMPT TO LOAD FROM CACHE
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                raw_divs = json.load(f)
                # print(f" [cache] Loaded {ticker} from disk")
        except Exception as e:
            print(f" [!] Cache read error for {ticker}: {e}")

    # 2. FETCH FROM API IF CACHE MISSING
    if not raw_divs:
        url = "https://api.massive.com/stocks/v1/dividends"
        params = {
            "ticker": ticker,
            "limit": 1000,
            "sort": "ex_dividend_date",
            "ex_dividend_date.gte": start_date,
            "ex_dividend_date.lte": end_date,
            "apiKey": api_key
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                raw_divs = resp.json().get('results', [])
                
                # SAVE TO CACHE
                with open(cache_path, 'w') as f:
                    json.dump(raw_divs, f)
            else:
                return []
        except Exception as e:
            print(f"Error fetching divs for {ticker}: {e}")
            return []

    # 3. MAP RAW DATA TO INSTRUMENT IDs
    # This part always runs so that tm_lookup (current DB state) is applied
    mapped_divs = []
    for d in raw_divs:
        ex_date_str = d.get('ex_dividend_date')
        if not ex_date_str: continue
        
        owner_id = None
        for interval in tm_lookup:
            if interval['start'] <= ex_date_str <= interval['end']:
                owner_id = interval['inst_id']
                break

        if owner_id:
            mapped_divs.append((
                owner_id,
                d.get('id'),
                ticker,
                d.get('record_date'),
                d.get('pay_date'),
                d.get('declaration_date'),
                ex_date_str,
                d.get('frequency'),
                d.get('cash_amount'),
                d.get('currency'),
                d.get('distribution_type'),
                d.get('cash_amount'),
                d.get('split_adjusted_cash_amount')
            ))
            
    return mapped_divs