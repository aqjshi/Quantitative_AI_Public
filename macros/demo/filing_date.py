#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
# 2. CORE DATA INGESTION ENGINE
# ─────────────────────────────────────────────────────────────
def fetch_raw_bitemporal_pool() -> pd.DataFrame:
    """
    Connects to PostgreSQL and extracts the entire bitemporal matrix index,
    including row values and human-readable names for diagnostic tracking.
    """
    query_string = """
        SELECT 
            obs.name_hash, 
            obs.date, 
            obs.realtime_start,
            obs.value,
            unf.name,
            COALESCE(emb.frequency_short, 'M') as frequency_short
        FROM fred_series_observations obs
        INNER JOIN fred_series_unfiltered unf ON obs.name_hash = unf.name_hash
        INNER JOIN fred_series_filtered emb ON unf.name = emb.name;
    """
    print("\n[+] Initializing Database Extraction Pipeline...")
    print(" -> Pulling raw bitemporal coordinate pairs from disk...")
    
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text(query_string), conn)
        
        if df.empty:
            print(" [!] Verification Failure: Empty dataframe returned.")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        df['freq'] = df['frequency_short'].str.strip().str.upper()
        return df
        
    except Exception as e:
        print(f" [!] Database query crash encountered: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# 3. UNIFIED ALGORITHM: INITIAL RELEASE + GMM OVERLAY ENGINE
# ─────────────────────────────────────────────────────────────
def process_and_plot_gmm_initial_release(df: pd.DataFrame, out_dir: str):
    """
    1. Executes structural deduplication on value collisions.
    2. Isolates the true minimum initial advance release window.
    3. Fits a 3-Component GMM for each frequency tier.
    4. Prints text telemetry and database specimens to console.
    5. Plots raw histograms alongside individual components and total GMM curves.
    """
    print("\n -> Executing structural deduplication on value collisions...")
    working_df = df.copy()
    
    # Force timeline ordering per series-date track
    working_df = working_df.sort_values(by=['name_hash', 'date', 'realtime_start'])
    
    # Drop consecutive identical value blocks (vintages with no structural updates)
    working_df['value_changed'] = working_df.groupby(['name_hash', 'date'])['value'].shift(1) != working_df['value']
    deduped_df = working_df[working_df['value_changed']]

    # Isolate initial advance release date
    initial_df = deduped_df.groupby(['name_hash', 'date']).agg({
        'realtime_start': 'min',
        'freq': 'first',
        'name': 'first'
    }).reset_index()

    # Calculate exact delta boundaries
    initial_df['lag_days'] = (initial_df['realtime_start'] - initial_df['date']).dt.days
    
    # Clip hyper-outliers and look-aheads
    MAX_LAG = 730
    initial_df = initial_df[(initial_df['lag_days'] >= 0) & (initial_df['lag_days'] <= MAX_LAG)]

    # Setup the multi-panel visual grid
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('True Advance Release Lags mapped with 3-Component Gaussian Mixture Models (GMM)', 
                 fontsize=14, fontweight='bold', y=1.03)

    freq_configs = [
        ('M', 'Monthly True Advance Lags', '#2980b9', 0),
        ('Q', 'Quarterly True Advance Lags', '#e67e22', 1),
        ('A', 'Annual True Advance Lags', '#27ae60', 2)
    ]

    print("\n" + "="*95)
    print("         GAUSSIAN MIXTURE MODEL (GMM) MACRO REGIME CLUSTERING & SAMPLING DEEP-DIVE")
    print("="*95)

    # Establish uniform mathematical bin spaces to make scale translations rock solid
    NUM_BINS = 40
    bin_edges = np.linspace(0, MAX_LAG, NUM_BINS + 1)
    bin_width = MAX_LAG / NUM_BINS # 18.25 Days

    for freq_code, title, hist_color, ax_idx in freq_configs:
        ax = axs[ax_idx]
        sub_df = initial_df[initial_df['freq'] == freq_code].copy()
        
        if len(sub_df) < 10:
            ax.text(0.5, 0.5, f'Insufficient samples\nfor Tier: {freq_code}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='#7f8c8d')
            ax.set_title(title, fontsize=11, fontweight='semibold')
            continue
            
        lags_vector = sub_df['lag_days'].to_numpy()
        N = len(lags_vector)
        
        # ─── 1. RENDER RAW COUNT HISTOGRAM ───
        ax.hist(lags_vector, bins=bin_edges, color=hist_color, alpha=0.4, 
                edgecolor='#2c3e50', linewidth=0.5, label=f'Raw Data (N={N:,})')
        
        # ─── 2. FIT GAUSSIAN MIXTURE MODEL ───
        X_fit = lags_vector.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42, max_iter=250)
        sub_df['mode_cluster'] = gmm.fit_predict(X_fit)
        
        # Extract components and sort them chronologically by mean position
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        sorted_indices = np.argsort(means)
        
        print(f"\n⚡ FREQUENCY TIER HORIZON: [{freq_code}] (Total Records: {N:,})")
        print("-" * 95)
        
        # Generate smooth x space vector to draw continuous GMM distributions
        x_pdf = np.linspace(0, MAX_LAG, 1000)
        total_count_pdf = np.zeros_like(x_pdf)
        
        component_colors = ['#c0392b', '#8e44ad', '#16a085']
        
        for rank, orig_idx in enumerate(sorted_indices):
            m_val = means[orig_idx]
            s_val = stds[orig_idx]
            w_val = weights[orig_idx]
            
            print(f" -> Mode {rank + 1}: Mean Lag = {m_val:5.1f}d | StdDev = {s_val:4.1f}d | Weight = {w_val*100:4.1f}%")
            
            # Calculate normal curve distribution
            pdf_component = (1.0 / (s_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - m_val) / s_val) ** 2)
            
            # Translate Probability Space directly back to Count Space
            count_scaled_pdf = w_val * pdf_component * N * bin_width
            total_count_pdf += count_scaled_pdf
            
            # Draw individual component curve
            ax.plot(x_pdf, count_scaled_pdf, color=component_colors[rank], linestyle='--', 
                    linewidth=1.5, label=f'Mode {rank+1} ({int(m_val)}d)')
            
            # Sample specimens from this specific cluster group
            mode_rows = sub_df[sub_df['mode_cluster'] == orig_idx]
            if not mode_rows.empty:
                sample_size = min(3, len(mode_rows)) # Pull 3 crisp samples to save terminal space
                samples = mode_rows.sample(n=sample_size, random_state=42)
                for _, row in samples.iterrows():
                    target_dt = row['date'].strftime('%Y-%m-%d')
                    realtime_dt = row['realtime_start'].strftime('%Y-%m-%d')
                    print(f"    │  • Name: {row['name']:<30} | Target: {target_dt} | Released: {realtime_dt} | Lag: {row['lag_days']}d")
        
        # ─── 3. DRAW COMBINED MIXTURE ENVELOPE ───
        ax.plot(x_pdf, total_count_pdf, color='#2c3e50', linestyle='-', 
                linewidth=2.2, label='Total GMM Mix')
        
        # Standardize panel visual annotations
        median_lag = np.median(lags_vector)
        ax.axvline(median_lag, color='#d35400', linestyle=':', linewidth=1.5, label=f'Median: {int(median_lag)}d')
        
        ax.set_title(title, fontsize=11, fontweight='semibold')
        ax.set_xlabel('Lag Duration from Target Period Start Line (Days)', fontsize=9)
        ax.set_ylabel('Unique Asset/Record Count', fontsize=9)
        ax.grid(True, which="both", linestyle=':', alpha=0.3)
        ax.set_xlim(-10, MAX_LAG + 10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        print("-" * 95)

    print("="*95 + "\n")

    plt.tight_layout()
    save_plot_path = os.path.join(out_dir, "initial_release_lags_gmm.png")
    plt.savefig(save_plot_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"[+] Multi-component GMM profile plots exported successfully to: {save_plot_path}")

# ─────────────────────────────────────────────────────────────
# 4. RUN EXECUTION GATEWAY
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    script_local_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Extract data pool
    raw_df = fetch_raw_bitemporal_pool()
    
    if not raw_df.empty:
        # Run calculation, fit GMM layers, output metrics and generate the plot asset
        process_and_plot_gmm_initial_release(df=raw_df, out_dir=script_local_dir)
        
    print("[+] Diagnostic pipelines executed completely. Standalone process complete.\n")