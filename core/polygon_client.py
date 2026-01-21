import os
import pandas as pd
from dataclasses import dataclass, field
from massive import RESTClient
from typing import List, Optional

@dataclass
class polygon_client:
    api_key: str
    client: RESTClient = field(init=False)

    def __post_init__(self):
        # Initialize the actual SDK client
        self.client = RESTClient(api_key=self.api_key)

    def fetch_fx_aggregates(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "minute", 
        multiplier: int = 1
    ) -> pd.DataFrame:
        """
        Fetches FX bar data (Aggregates) from Polygon.
        Automatically handles pagination via the SDK's generator.
        """
        # Polygon FX tickers are prefixed with C:
        ticker = f"C:{symbol}USD"
        aggs = []
        
        try:
            # The Massive SDK list_aggs returns a generator handling all pagination
            for a in self.client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=interval,
                from_=start_date,
                to=end_date,
                limit=50000,
                adjusted=True
            ):
                aggs.append({
                    "timestamp_ms": a.timestamp,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                    "vwap": getattr(a, 'vwap', None) # Safety check for vwap
                })

            if not aggs:
                return pd.DataFrame()

            df = pd.DataFrame(aggs)
            
            # Convert milliseconds to Unix Seconds (Standard for your DB)
            df['time_entry_ts'] = df['timestamp_ms'] // 1000
            
            # Optional: Set a datetime index for easier Pandas manipulation later
            df['dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
            
            return df

        except Exception as e:
            print(f"[POLY-API ERROR] Failed fetching {ticker}: {e}")
            return pd.DataFrame()
        
    def fetch_stock_aggregates(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "minute", 
        multiplier: int = 1,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetches Stock bar data (Aggregates) from Polygon/Massive.
        """
        aggs = []
        
        try:
            # For Stocks, the ticker is passed directly (e.g., "AAPL")
            for a in self.client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=interval,
                from_=start_date,
                to=end_date,
                limit=50000,
                adjusted=adjusted
            ):
                aggs.append({
                    "timestamp_ms": a.timestamp,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                    "vwap": getattr(a, 'vwap', None),
                    "transactions": getattr(a, 'transactions', None) # Stocks include tx count
                })

            if not aggs:
                return pd.DataFrame()

            df = pd.DataFrame(aggs)
            df['time_entry_ts'] = df['timestamp_ms'] // 1000
            df['dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
            
            return df

        except Exception as e:
            print(f"[POLY-API ERROR] Failed fetching stock {ticker}: {e}")
            return pd.DataFrame()
        
    def fetch_list_financials_income_statements(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetches Financial Income Statements from Polygon/Massive.
        Maps results to a DataFrame and creates a unix timestamp based on period_end.
        """
        aggs = []
        
        try:
            # 1. Call the client using the correct keyword arguments
            # We map start_date/end_date to period_end filters
            results = self.client.list_financials_income_statements(
                tickers=ticker,
                period_end_gte=start_date,
                period_end_lte=end_date,
                limit=limit,
                sort="period_end"
            )

            # 2. Iterate through the generator/list
            for item in results:
                # item is usually an object or dict containing the keys you provided
                # We convert to a dict to ensure pandas handles it easily
                data = item if isinstance(item, dict) else item.__dict__
                aggs.append(data)

            if not aggs:
                return pd.DataFrame()

            # 3. Create DataFrame
            df = pd.DataFrame(aggs)

            # 4. Standardize Timestamps
            # Financials use 'period_end' (e.g., "2025-06-28") instead of ms timestamps
            df['dt'] = pd.to_datetime(df['period_end'], utc=True)
            
            # Create a unix timestamp (seconds) for database compatibility
            df['time_entry_ts'] = df['dt'].view('int64') // 10**9
            
            return df

        except Exception as e:
            print(f"[POLY-API ERROR] Failed fetching financials for {ticker}: {e}")
            return pd.DataFrame()


    def get_ticker_details(self, symbol: str):
        """Returns metadata about the FX pair."""
        return self.client.get_ticker_details(f"C:{symbol}USD")