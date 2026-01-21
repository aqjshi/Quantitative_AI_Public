

def fetcher(api_key, end_date, limit):
    """
    Returns a fetcher function configured with a specific end_date.
    """

    def fetcher(ticker):
        url = "https://api.massive.com/stocks/financials/v1/cash-flow-statements"
        params = {
            "tickers": ticker,
            "limit": limit,
            "timeframe" : "quarterly",
            "period_end.lte": end_date,
            "sort": "period_end.desc",
            "apiKey": api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            print(f" [!] Error fetching {ticker}: {e}")
            return []
    return fetcher




import requests

def fetch_and_map_cash_flow(ticker, tm_lookup, api_key, start_date, end_date):
    """
    Fetches dividends with explicit date range filters to ensure full coverage.
    """
    url = "https://api.massive.com/stocks/v1/dividends"
    
    # Using .gte and .lte to bound the search to your specific backtest window
    params = {
        "ticker": ticker,
        "limit": 1000,
        "sort": "ex_dividend_date",
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
        "apiKey": api_key
    }
    
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200: return []
        
        raw_divs = resp.json().get('results', [])
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
                    owner_id,      # instrument_id
                    d.get('id'),                # external_id
                    ticker,                     # ticker snapshot
                    d.get('record_date'),       # record_date
                    d.get('pay_date'),          # pay_date
                    d.get('declaration_date'),  # declaration_date
                    ex_date_str,
                    d.get('frequency'),         # frequency
                    d.get('cash_amount'),
                    d.get('currency'),
                    d.get('distribution_type') ,
                    d.get('cash_amount'),       # amount
                    d.get('split_adjusted_cash_amount') 
                  
                ))
        return mapped_divs
    except Exception as e:
        print(f"Error fetching divs for {ticker}: {e}")
        return []
    

