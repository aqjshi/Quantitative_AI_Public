import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_lasso_plots(file_path):
    # 1. Setup Output Directory
    output_dir = "lasso"
    os.makedirs(output_dir, exist_ok=True)

    # Extract ETF name from filename (e.g., "SOXX.csv" -> "SOXX")
    src_etf = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"[*] Loading {file_path}...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[!] Error: File {file_path} not found.")
        return

    # 2. Iterate over columns
    # Filter out timestamp columns if they exist to avoid plotting time
    skip_cols = ['t', 'timestamp', 'timestamp_utc']
    
    for column in data.columns:
        if column in skip_cols:
            continue
            
        if pd.api.types.is_numeric_dtype(data[column]):
            # Optional: Skip columns that are all zeros (sparse lasso weights)
            if data[column].sum() == 0:
                continue

            # Plot the distribution
            plt.figure(figsize=(10, 6))
            data[column].hist(bins=50, edgecolor='k', alpha=0.7)
            
            plt.title(f"Distribution of {column} in {src_etf}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.5)

            # 3. Save to subdir
            # Clean column name just in case (remove _w or _p if you want just the ticker, 
            # but keeping unique col name is safer)
            safe_col_name = column.replace('/', '_') 
            save_path = os.path.join(output_dir, f"{src_etf}_{safe_col_name}.png")
            
            plt.savefig(save_path)
            plt.close() # Close to free memory
            
            print(f"    Saved: {save_path}")

if __name__ == "__main__":
    # Example usage
    generate_lasso_plots("SOXX.csv")
    generate_lasso_plots("QQQ.csv")
    generate_lasso_plots("SPY.csv")