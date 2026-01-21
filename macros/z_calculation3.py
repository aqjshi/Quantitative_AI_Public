import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred
import os
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter
import json
import subprocess
import shutil
import glob
import sys


def calculate_grid_dims(n_items):
    """Calculates square-ish nrows and ncols for a given number of items."""
    # We add +1 to account for the 'Proprietary View' text box
    total_slots = n_items + 1
    ncols = math.ceil(math.sqrt(total_slots))
    nrows = math.ceil(total_slots / ncols)
    return nrows, ncols

def plot_insolvency_matrix(df_raw, indicator_map, output_file='plots/insolvency_plot.png'):
    """Generates a dynamically sized grid of economic indicators from a DataFrame."""
    print("\nGenerating Dynamic Insolvency Matrix...")
    
    # Use the number of columns to define the grid
    n_cols_to_plot = len(df_raw.columns)
    nrows, ncols = calculate_grid_dims(n_cols_to_plot)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    
    # Flatten axes for easy iteration, handle the case of a single plot
    if n_cols_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_color = '#1f77b4' 

    # 1. Iterate over columns of the DataFrame
    for i, col_name in enumerate(df_raw.columns):
        ax = axes[i]
        df_raw[col_name].plot(ax=ax, color=plot_color, linewidth=1.5)
        
        # Get the descriptive name from the map if it exists
        display_name = indicator_map.get(col_name, col_name)
        ax.set_title(f"{col_name} ({display_name})", fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("")

    # 2. Handle the 'Proprietary Text' in the last slot
    last_idx = len(axes) - 1
    axes[last_idx].axis('off')
    axes[last_idx].text(0.5, 0.5, 'Insolvency Matrix\n(Proprietary Risk View)', 
                        ha='center', va='center', alpha=0.2, fontsize=14, fontweight='bold')

    # 3. Hide remaining unused slots
    for j in range(i + 1, last_idx):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close() # Always close to free up memory




    
def plot_log_returns_matrix(df_returns, output_file='plots/log_returns_plot.png'):
    """Accepts PRE-CALCULATED shock data and plots it."""
    print("\nGenerating Dynamic Log Returns Matrix...")
    
    # Check if df is empty to prevent the fmin error
    if df_returns.empty:
        print("Error: The shock dataframe is empty. Nothing to plot.")
        return

    nrows, ncols = calculate_grid_dims(len(df_returns.columns))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten()
    
    plot_color = '#e67e22' # Shock Orange

    for i, label in enumerate(df_returns.columns):
        ax = axes[i]
        suffix = "(Diff)" if label in ['STLFSI', 'WEI'] else "(Log Return)"
        
        # Plotting logic
        df_returns[label].plot(ax=ax, color=plot_color, linewidth=1, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        
        ax.set_title(f"{label} {suffix}", fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("")

    # Proprietary Text
    last_idx = len(axes) - 1
    axes[last_idx].axis('off')
    axes[last_idx].text(0.5, 0.5, 'Volatility Matrix\n(Risk Shocks 2005-2025)', 
                        ha='center', va='center', alpha=0.3, fontsize=14, fontweight='bold')

    for j in range(len(df_returns.columns), last_idx):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Clean Log Returns Plot saved to {output_file}")


# --- 0. CONFIGURATION ---
API_KEY = '6799bbe3b5be9dc92751b055763e402b'
START_DATE = "2015-01-01"
END_DATE = "2025-12-01"

# Hyperparameters
WINDOWS = range(8, 54, 4)   # Search space for windows
LAGS = range(-26, 27, 2)    # Search space for lags

# Output Structure
BASE_DIR = "quant_pipeline_master"
SUBDIRS = [
    'data', 
    'plots/shocks', 
    'plots/original_series', 
    'plots/stability_frames', # For the video (low res)
    'plots/stability_static', # For the report (high res)
    'plots/wavelet_proofs',   # Static proofs
    'logs', 
    'frames/stacked',         # For Matrix Video
    'frames/wavelet',         # For Wavelet Video
    'frames/temp_sync',       # FFmpeg buffer
    'videos'
]

def setup_environment():
    if os.path.exists(BASE_DIR):
        print(f"Cleaning up: {BASE_DIR}")
        try:
            shutil.rmtree(BASE_DIR)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")
    for d in SUBDIRS:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    print(f"Environment initialized at ./{BASE_DIR}")

# --- 1. DATA INGESTION ---
def fetch_and_sync(api_key, indicators, start, end):
    print("\n--- 1. FETCHING & SYNCING ---")
    fred = Fred(api_key=api_key)
    series_list = []
    
    for label, sym in indicators.items():
        try:
            s = fred.get_series(sym, observation_start=start, observation_end=end)
            s.name = label
            s = s.resample('W-FRI').mean().interpolate(method='linear').ffill().bfill()
            series_list.append(s)
            print(f"  -> Loaded {label} ({len(s)} wks)")
        except Exception as e:
            print(f"  !! Failed {label}: {e}")
            
    df = pd.concat(series_list, axis=1)
    
    # Anti-Zero-Variance Jitter
    for col in df.columns:
        noise = np.random.normal(0, df[col].std() * 0.001 if df[col].std() > 0 else 1e-5, size=len(df))
        df[col] = df[col] + noise
        
    df = df.dropna().ffill().bfill()
    df.to_csv(f"{BASE_DIR}/data/raw_synced.csv")
    return df



def transform_and_normalize(df):
    print("\n--- 2. TRANSFORMATION (Z-SCORE) ---")
    shocks = pd.DataFrame(index=df.index)
    diff_cols = ['STLFSI', 'WEI']
    
    for col in df.columns:
        if col in diff_cols:
            # For Indices already in % or rate form
            raw = df[col].diff()
        else:
            # For raw levels (ICSA, Fuel, Retail)
            # Use .bfill() instead of fillna(method='bfill') to fix the warning
            denom = df[col].shift(1).replace(0, np.nan).bfill()
            
            # Use a vectorized clip to handle zero/negative ratios safely
            ratio = (df[col] / denom).clip(lower=1e-6)
            raw = np.log(ratio)
            
        # --- CRITICAL SAFETY SCRUBBING ---
        # 1. Replace any Infs created by math errors with NaN
        raw = raw.replace([np.inf, -np.inf], np.nan)
        
        # 2. Fill the first row's NaN and any math errors with 0
        raw = raw.fillna(0)
        
        # 3. Robust Z-Score calculation
        mu, std = raw.mean(), raw.std()
        if not np.isnan(std) and std > 1e-9: 
            shocks[col] = (raw - mu) / std
        else:
            shocks[col] = 0.0
            
    # --- FINAL SQUASHING ---
    # This prevents the "Overflow Error" in the stability score (score = tr - val * 2)
    # By capping shocks at +/- 4, the max diff is 8, and score max is ~16.
    shocks = shocks.clip(-4, 4)
    
    shocks.to_csv(f"{BASE_DIR}/data/z_shocks.csv")
    return shocks


# --- 3. PLOTTING HELPERS ---
def plot_static_wavelet_proof(df, target, predictor, optimal_lag, label):
    """Generates the static proof of the relationship."""
    pred_shifted = df[predictor].shift(optimal_lag)
    target_series = df[target]
    
    visual_windows = np.arange(2, 60, 2) 
    cwt_matrix = np.zeros((len(visual_windows), len(df)))
    
    for i, w in enumerate(visual_windows):
        roll = target_series.rolling(w).corr(pred_shifted).replace([np.inf, -np.inf], np.nan).fillna(0)
        cwt_matrix[i, :] = roll.fillna(0).values
        
    plt.figure(figsize=(12, 6))
    img = plt.imshow(cwt_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[0, len(df), visual_windows[0], visual_windows[-1]], origin='lower')
    plt.colorbar(img, label="Rolling Correlation")
    plt.title(f"Wavelet Proof: {label} (Shift {optimal_lag}w) vs {target}", fontweight='bold')
    plt.ylabel("Window Size")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/wavelet_proofs/PROOF_{label}_vs_{target}.png", dpi=150)
    plt.close()

# --- 4. CORE SEARCH ENGINE ---
def run_walk_forward_search(df, target_var):
    years = sorted(df.index.year.unique())
    predictors = [c for c in df.columns if c != target_var]
    manifest = [] # This will now store YEARLY snapshots

    print(f"  Target: {target_var}...")

    for pred in predictors:
        # We still track a global surface to find the "All-Time Best" for the static plot
        global_surface = np.zeros((len(WINDOWS), len(LAGS)))
        
        for split_year in range(years[0] + 2, years[-1]):
            yearly_surface = np.zeros((len(WINDOWS), len(LAGS)))
            test_mask = df.index.year == (split_year + 1)
            train_mask = df.index.year <= split_year
            if not test_mask.any(): continue

            for i, w in enumerate(WINDOWS):
                for j, l in enumerate(LAGS):
                    full_corr = df[target_var].rolling(w, min_periods=4).corr(df[pred].shift(l))
                    tr, val = full_corr[train_mask].mean(), full_corr[test_mask].mean()
                    tr, val = np.nan_to_num(tr), np.nan_to_num(val)
                    
                    score = abs(val) - (abs(tr - val) * 1.5) 
                    v_capped = max(0, score)
                    yearly_surface[i, j] = v_capped
                    global_surface[i, j] += v_capped

            # --- LOG YEARLY PEAK TO MANIFEST ---
            i_y, j_y = np.unravel_index(yearly_surface.argmax(), yearly_surface.shape)
            manifest.append({
                "target": target_var,
                "predictor": pred,
                "year": split_year + 1,
                "optimal_window": int(WINDOWS[i_y]),
                "optimal_lag": int(LAGS[j_y]),
                "stability_score": float(yearly_surface[i_y, j_y])
            })

            # --- PLOT YEARLY FRAME ---
            plt.figure(figsize=(6, 5))
            # FIX: Local normalization prevents the "Black Frame" issue
            v_max = max(0.05, np.max(yearly_surface)) 
            sns.heatmap(gaussian_filter(yearly_surface, sigma=0.8), 
                        xticklabels=LAGS, yticklabels=WINDOWS,
                        cmap='magma', cbar=True, vmin=0, vmax=v_max)
            
            plt.title(f"Stability: {pred} vs {target_var} ({split_year+1})")
            plt.xlabel("Lag (Weeks)")
            plt.ylabel("Window Size (Weeks)")
            
            fname = f"{target_var}_{pred}_{split_year+1}.png"
            plt.savefig(os.path.join(BASE_DIR, "plots/stability_frames", fname))
            plt.close()

        # Save Global Static Heatmap (The 'Average' through time)
        avg_surface = global_surface / (len(years) - 3)
        plt.figure(figsize=(8, 6))
        sns.heatmap(gaussian_filter(avg_surface, sigma=1.2), xticklabels=LAGS, yticklabels=WINDOWS, cmap='viridis')
        plt.title(f"GLOBAL STABILITY: {target_var} vs {pred}")
        plt.xlabel("Lag (Weeks)")
        plt.ylabel("Window (Weeks)")
        plt.savefig(os.path.join(BASE_DIR, "plots/stability_static", f"GLOBAL_{target_var}_{pred}.png"))
        plt.close()

    return manifest


# --- 3. VIDEO ENGINE ---
def encode_video(frame_pattern, output_name, fps=3):
    temp_dir = os.path.join(BASE_DIR, "frames/temp_sync")
    # Clean buffer
    for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
    
    files = sorted(glob.glob(frame_pattern))
    if not files: return
    
    # Re-index for FFmpeg
    for i, f in enumerate(files):
        shutil.copy(f, os.path.join(temp_dir, f"img_{i+1:04d}.png"))
    
    out_path = os.path.join(BASE_DIR, "videos", output_name)
    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(temp_dir, "img_%04d.png"),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"    -> Rendered: {output_name}")

def main():
    setup_environment()
    indicators = {'WEI': 'WEI', 'ICSA': 'ICSA', 'FUEL': 'GASREGW', 'RETAIL': 'RSAFS', 'STLFSI': 'STLFSI4'}
    
    # 1. Pipeline Execution
    df_raw = fetch_and_sync(API_KEY, indicators, START_DATE, END_DATE)
    df = transform_and_normalize(df_raw)

    # --- INSERTED GRAPHING LOGIC (With Safety Directory Check) ---
    print("\n--- 2.5 GENERATING BASELINE MATRICES ---")
    os.makedirs(os.path.join(BASE_DIR, "plots/original_series"), exist_ok=True)
    
    # Graph the Raw Series (Insolvency Matrix)
    plot_insolvency_matrix(df_raw, indicators, 
                           output_file=os.path.join(BASE_DIR, "plots/original_series/insolvency_view.png"))
    
    # Graph the Z-Shocks (Log Returns/Volatility Matrix)
    plot_log_returns_matrix(df, 
                            output_file=os.path.join(BASE_DIR, "plots/shocks/volatility_view.png"))
    # -------------------------------------------------------------
    
    print("\n--- 3. RUNNING STABILITY SEARCH ---")
    opt_map = {}
    for col in df.columns:
        opt_map[col] = run_walk_forward_search(df, target_var=col)

    print("\n--- 4. ENCODING STABILITY VIDEOS ---")
    for target, preds in opt_map.items():
        for p in preds:
            pattern = os.path.join(BASE_DIR, "plots/stability_frames", f"{target}_{p['predictor']}_*.png")
            encode_video(pattern, f"stability_{target}_vs_{p['predictor']}.mp4")



def generate_wavelet_video(df, target, predictor, lag):
    print(f"--- Generating Wavelet Video: {predictor} vs {target} ---")
    frames_dir = os.path.join(BASE_DIR, "frames/wavelet")
    
    p_series = df[predictor].shift(lag).fillna(0)
    t_series = df[target]
    
    # Render frames (stride 2 for speed)
    for idx in range(len(df)//2, len(df), 2):
        window_data = []
        for w in WINDOWS:
            roll = t_series.iloc[:idx].rolling(w).corr(p_series.iloc[:idx])
            window_data.append(roll.values[-50:])
        
        plt.figure(figsize=(10, 6))
        plt.imshow(window_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   extent=[0, 50, WINDOWS[0], WINDOWS[-1]], origin='lower')
        plt.colorbar(label="Coherence")
        plt.title(f"Wavelet Flow | {predictor} (Lag {lag}) vs {target} | {df.index[idx].date()}")
        plt.ylabel("Window Size")
        plt.savefig(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        plt.close()
    
    encode_video(os.path.join(frames_dir, "*.png"), f"wavelet_flow_top_signal_pred{predictor}_lag{lag}_target{target}.mp4", fps=10)

def generate_stacked_matrix_video(df, opt_map):
    print("\n--- Generating Stacked Matrix Animation ---")
    frames_dir = os.path.join(BASE_DIR, 'frames/stacked')
    
    # Determine safe start index
    all_lags = [res['optimal_lag'] for m in opt_map.values() for res in m]
    max_lag = max([abs(l) for l in all_lags]) if all_lags else 52
    start_idx = 52 + max_lag
    
    frame_counter = 0
    for i in range(start_idx, len(df), 2):
        current_date = df.index[i]
        
        # Baseline (Standard Correlation)
        baseline_corr = df.iloc[i-52 : i].corr()

        # Optimized (Stacked)
        opt_corr = pd.DataFrame(0.0, index=df.columns, columns=df.columns)
        for target, predictors_list in opt_map.items():
            for res in predictors_list:
                pred = res['predictor']
                lag = res['optimal_lag']
                win = res['optimal_window']
                
                s1 = df[target].iloc[i-win : i]
                s2 = df[pred].shift(lag).iloc[i-win : i]
                
                if len(s1.dropna()) > 4:
                    val = s1.corr(s2)
                    opt_corr.loc[target, pred] = val if not np.isnan(val) else 0.0
        
        np.fill_diagonal(opt_corr.values, 1.0)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        sns.heatmap(baseline_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1, cbar=False, fmt=".2f")
        ax1.set_title("Standard Correlation (Fixed 52w)", fontsize=14)
        sns.heatmap(opt_corr, annot=True, cmap='viridis', vmin=-1, vmax=1, ax=ax2, cbar=False, fmt=".2f")
        ax2.set_title("Optimized Lead-Lag Matrix", fontsize=14, color='green')
        plt.suptitle(f"Market Dynamics | {current_date.strftime('%Y-%m-%d')}", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f"frame_{frame_counter:04d}.png"))
        plt.close()
        frame_counter += 1
        
    encode_video(os.path.join(frames_dir, "*.png"), "stacked_matrix_dynamics.mp4", fps=8)

# --- 6. MAIN ORCHESTRATOR ---
def main():
    setup_environment()
    indicators = {'WEI': 'WEI',     'CHURN': 'EUANDH',  'ICSA': 'ICSA', 'FUEL': 'GASREGW', 'RETAIL': 'RSAFS', 'STLFSI': 'STLFSI4'}
    
    # 1. Pipeline Execution
    df_raw = fetch_and_sync(API_KEY, indicators, START_DATE, END_DATE)
    df = transform_and_normalize(df_raw)

    # --- INSERTED GRAPHING LOGIC HERE ---
    # Graph the Raw Series (Insolvency Matrix)
    plot_insolvency_matrix(df_raw, indicators, output_file=f"{BASE_DIR}/plots/original_series/insolvency_view.png")
    
    # Graph the Z-Shocks (Log Returns/Volatility Matrix)
    plot_log_returns_matrix(df, output_file=f"{BASE_DIR}/plots/shocks/volatility_view.png")
    # ------------------------------------

    print("\n--- 3. RUNNING STABILITY SEARCH ---")
    
    print("\n--- 3. RUNNING STABILITY SEARCH ---")
    all_results_flat = []
    # We'll use a simplified map for the Stacked Video (using all-time best)
    global_opt_map = {} 

    for col in df.columns:
        manifest = run_walk_forward_search(df, target_var=col)
        all_results_flat.extend(manifest)
        
        # Extract the highest-scoring parameters for the Stacked Matrix Video
        col_preds = [m for m in manifest if m['target'] == col]
        # Group by predictor and find best avg across years
        for p in indicators.keys():
            if p == col: continue
            best_p = sorted([m for m in col_preds if m['predictor'] == p], key=lambda x: x['stability_score'])[-1]
            if col not in global_opt_map: global_opt_map[col] = []
            global_opt_map[col].append(best_p)

    # 4. Save Logs
    pd.DataFrame(all_results_flat).to_csv(f"{BASE_DIR}/logs/optimal_parameters_BY_YEAR.csv", index=False)
    print(f"  -> Logs saved to {BASE_DIR}/logs/")

    # 5. Stability Films
    print("\n--- 4. ENCODING STABILITY FILMS ---")
    for col in df.columns:
        for pred in [c for c in df.columns if c != col]:
            pattern = os.path.join(BASE_DIR, "plots/stability_frames", f"{col}_{pred}_*.png")
            encode_video(pattern, f"stability_{col}_vs_{pred}.mp4", fps=2)

    # 6. Wavelet & Matrix Dynamics
    print("\n--- 5. RENDERING DYNAMIC FLOWS ---")
    top_signal = sorted(all_results_flat, key=lambda x: x['stability_score'])[-1]
    
    # Animates the "Optimal Island" moving through the scalogram
    generate_wavelet_video(df, top_signal['target'], top_signal['predictor'], top_signal['optimal_lag'])
    
    top2_signal = sorted(all_results_flat, key=lambda x: x['stability_score'])[-2]
    
    # Animates the "Optimal Island" moving through the scalogram
    generate_wavelet_video(df, top2_signal['target'], top2_signal['predictor'], top2_signal['optimal_lag'])
    
    top3_signal = sorted(all_results_flat, key=lambda x: x['stability_score'])[-3]
    
    # Animates the "Optimal Island" moving through the scalogram
    generate_wavelet_video(df, top3_signal['target'], top3_signal['predictor'], top3_signal['optimal_lag'])
    

    # Animates the bivariate vs multivariate matrix comparison
    generate_stacked_matrix_video(df, global_opt_map)

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()