import pandas as pd
from sqlalchemy import text
import sys
from db import engine



def fetch_and_save_quotes():
    """
    Fetches quote data for company_id = 5 (assuming Unix timestamps)
    and saves it to a dynamic CSV file.
    """
    
    # The SQL query you specified
    sql_query = text("SELECT * FROM quotes WHERE company_id = 5 ORDER BY time_entry_ts ASC;")

    print("Connecting to database and executing query...")

    try:
        # Use pandas to read the SQL query directly into a DataFrame
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, connection)

        if df.empty:
            print("Query returned no data. No CSV file created.")
            return

        print(f"Successfully fetched {len(df)} rows.")

        # --- Create the dynamic filename ---

        # ** MODIFICATION HERE **
        # Convert the 'time_entry_ts' column from Unix time (seconds)
        # into a proper datetime object.
        df['time_entry_ts'] = pd.to_datetime(df['time_entry_ts'], unit='s')

        # Get the first and last timestamps from the DataFrame
        first_ts = df['time_entry_ts'].iloc[0]
        last_ts = df['time_entry_ts'].iloc[-1]

        # Format the timestamps into a string safe for filenames
        # (e.g., YYYYMMDD-HHMMSS)
        # Note: We also add .utc.strftime to handle potential timezone awareness
        # issues, formatting it as Coordinated Universal Time.
        # If you want local time, remove .utc
        ts_format = "%Y%m%d-%H%M%S"
        first_ts_str = first_ts.strftime(ts_format)
        last_ts_str = last_ts.strftime(ts_format)

        # Construct the final filename
        filename = f"AMD_{first_ts_str}_{last_ts_str}.csv"

        # --- Save the DataFrame to CSV ---
        # Set datetime format in CSV to be unambiguous (ISO 8601)
        df.to_csv(filename, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')

        print(f"\nSuccessfully saved data to: {filename}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your database connection, table/column names, and file permissions.")

# --- Run the script ---
if __name__ == "__main__":
    fetch_and_save_quotes()