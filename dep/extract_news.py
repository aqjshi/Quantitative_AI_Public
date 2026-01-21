import pandas as pd
from sqlalchemy import text
import sys
import pyarrow

from db import engine


def fetch_and_save_news():
    """
    Fetches news sentiment data (assuming Unix timestamps)
    and saves it to a dynamic Parquet file.
    """
    
    # The SQL query you specified
    sql_query = text("select * from news_sentiment order by time_published_ts ASC; ")

    print("Connecting to database and executing query...")

    with engine.connect() as connection:
        df = pd.read_sql_query(sql_query, connection)

    if df.empty:
        print("Query returned no data. No Parquet file created.")
        return

    print(f"Successfully fetched {len(df)} rows.")

    # --- Create the dynamic filename ---

    # Convert the 'time_published_ts' column from Unix time (seconds)
    # into a proper datetime object.
    df['time_published_ts'] = pd.to_datetime(df['time_published_ts'], unit='s')

    # Get the first and last timestamps
    first_ts = df['time_published_ts'].iloc[0]
    last_ts = df['time_published_ts'].iloc[-1]

    # Format timestamps for the filename
    ts_format = "%Y%m%d-%H%M%S"
    first_ts_str = first_ts.strftime(ts_format)
    last_ts_str = last_ts.strftime(ts_format)

    # Construct the final filename
    filename = f"NEWS_{first_ts_str}_{last_ts_str}.parquet"

    # --- Save the DataFrame to Parquet ---
    df.to_parquet(filename, index=False)

    print(f"\nSuccessfully saved data to: {filename}")

from playwright.sync_api import sync_playwright

def scrape_vanguard_the_hard_way(ticker):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker}")
        
        # You have to wait for the JS to load the 'Net Assets' element
        page.wait_for_selector(".fund-stats") 
        
        # Now you grab the content
        content = page.content()
        # NOW you can pass 'content' to Selectolax for fast parsing
        browser.close()
        return content
# --- Run the script ---
if __name__ == "__main__":
    fetch_and_save_news()