#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text

# Force matplotlib to use a non-interactive backend for headless/SSH environments
plt.switch_backend('Agg')

# ─────────────────────────────────────────────────────────────
# 1. FIXED PROJECTS ROOT ENVIRONMENT RESOLUTION
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine

# ─────────────────────────────────────────────────────────────
# 2. DATA INGESTION ENGINE
# ─────────────────────────────────────────────────────────────
def fetch_popularity_data() -> pd.DataFrame:
    """
    Connects to PostgreSQL and pulls distinct popularity scores per series.
    """
    query_string = """
        SELECT DISTINCT
            name_hash,
            popularity
        FROM fred_series_unfiltered
        WHERE popularity IS NOT NULL
    """
    print("\n[+] Initializing Database Extraction Pipeline...")
    print(" -> Fetching distinct series popularity scores...")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query_string), conn)
        
        if df.empty:
            print(" [!] Verification Failure: Empty dataframe returned.")
            return pd.DataFrame()
            
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        return df.dropna(subset=['popularity'])
        
    except Exception as e:
        print(f" [!] Database query crash encountered: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# 3. HISTOGRAM PLOTTING ENGINE
# ─────────────────────────────────────────────────────────────
def plot_popularity_distribution(df: pd.DataFrame, out_dir: str):
    """
    Renders and exports a histogram of FRED series popularity scores.
    """
    print(" -> Generating popularity score histogram...")
    
    popularity = df['popularity']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Render Histogram
    ax.hist(
        popularity, 
        bins=50, 
        color='#2980b9', 
        edgecolor='#2c3e50', 
        alpha=0.75,
        linewidth=0.8,
        label=f'Series Count (N={len(popularity):,})'
    )
    
    # Statistics Overlays
    mean_val = popularity.mean()
    median_val = popularity.median()
    
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#27ae60', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    # Chart Styling
    ax.set_title('FRED Series Popularity Score Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Popularity Score', fontsize=11)
    ax.set_ylabel('Unique Series Count', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Directory verification & file output
    os.makedirs(out_dir, exist_ok=True)
    save_plot_path = os.path.join(out_dir, "popularity_distribution.png")
    
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Popularity histogram successfully exported to: {save_plot_path}")

# ─────────────────────────────────────────────────────────────
# 4. RUN EXECUTION GATEWAY
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Path target: <PROJECT_ROOT>/macros/demo
    demo_dir = os.path.join(PROJECT_ROOT, "macros", "demo")
    
    raw_df = fetch_popularity_data()
    
    if not raw_df.empty:
        plot_popularity_distribution(df=raw_df, out_dir=demo_dir)
        
    print("[+] Process complete.\n")