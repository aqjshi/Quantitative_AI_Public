import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
import sys

from models import TrainItem
from db import  DATABASE_URL,  engine
import random
import json

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from train_helper import  TransposeCNN, sort_json_recursively, LazyTrainDataset, generate_stacked_exposure_plot, generate_stacked_plot, CustomCollate, print_in_chunks, reverse_transform_Y, run_training_epoch

from sqlalchemy.orm import sessionmaker
from itertools import product
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import copy
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run_validation_aggregate(model, loader, tickers, output_keys_sorted, decoded_output_keys, forecast_depth_seconds, device, roll_num):
    model.eval()
    all_pred_tensors = []
    all_true_tensors = []
    all_decoded_preds = []
    all_decoded_trues = []
    all_entry_times = []
    all_initial_prices = []

    loop = tqdm(loader, desc=f"Roll {roll_num+1} Validation", unit="batch", leave=False)
    
    with torch.no_grad():
        for time_entry_ts, X, true_y, initial_prices_batch in loop:
            X = X.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(X)

            all_pred_tensors.append(logits.cpu())
            all_true_tensors.append(true_y.cpu())

            for i in range(logits.size(0)):
                pred_tensor_single = logits[i]
                true_tensor_single = true_y[i]
                initial_prices_single = initial_prices_batch[i]
                time_entry_ts_single = time_entry_ts[i]

                initial_times = {
                    'a_time': {ticker: time_entry_ts_single for ticker in initial_prices_single.index}
                }
    
                decoded_preds = reverse_transform_Y(
                    pred_tensor_single, output_keys_sorted, initial_prices_single, 
                    initial_times, forecast_depth_seconds
                )
                all_decoded_preds.append(decoded_preds)
                
                decoded_trues = reverse_transform_Y(
                    true_tensor_single, output_keys_sorted, initial_prices_single, 
                    initial_times, forecast_depth_seconds
                )
                all_decoded_trues.append(decoded_trues)
                all_entry_times.append(time_entry_ts_single)
                all_initial_prices.append(initial_prices_single)
  

    final_preds_tensor = torch.cat(all_pred_tensors, dim=0)
    final_trues_tensor = torch.cat(all_true_tensors, dim=0)
    tensor_mae = torch.mean(torch.abs(final_preds_tensor - final_trues_tensor))

    differences = final_preds_tensor - final_trues_tensor
    euclidean_distances = torch.linalg.norm(differences, ord=2, dim=1)
    avg_euclidean_distance = torch.mean(euclidean_distances)
    mae = avg_euclidean_distance
    
    preds_df = pd.DataFrame(all_decoded_preds)
    trues_df = pd.DataFrame(all_decoded_trues)

    items_per_group = int(len(output_keys_sorted) / len(tickers))

    print("\n" + "="*20 + f" Roll {roll_num+1} Validation Results " + "="*20)
    total_samples = len(final_trues_tensor)
    num_samples_to_print = min(3, total_samples)
    random_indices = random.sample(range(total_samples), num_samples_to_print)

    for index in random_indices:
        true_str = final_trues_tensor[index].numpy()
        pred_str = final_preds_tensor[index].numpy()
        
        print(f"--- Sample Index: {index} ---")
        print("Tensor True:")
        print_in_chunks(true_str, items_per_group)
        print("Tensor Pred:")
        print_in_chunks(pred_str, items_per_group)
        print("-" * 50)


    print("\n--- Tensor Performance Metrics ---")
    print(f"Tensor MAE (L1 Norm):        {tensor_mae.item():.6f}")
    print(f"Avg. Euclidean Dist (L2 Norm): {avg_euclidean_distance.item():.6f}")
    COL_WIDTH_KEY = 20
    COL_WIDTH_MAE = 35 
    COL_WIDTH_TRUE = 30
    COL_WIDTH_PRED = 30
    if not preds_df.empty:
        trues_df = trues_df[preds_df.columns]
        decoded_mae = (trues_df - preds_df).abs().mean().sort_index()

        print("\n--- Decoded MAE Analysis (Random Sample) ---")
        
        # Select the first random index for the detailed decoded view
        selected_index = random_indices[0]
        
        # Retrieve the context for that specific sample
        entry_time_ts = all_entry_times[selected_index]
        initial_prices_series = all_initial_prices[selected_index]
        entry_datetime = datetime.fromtimestamp(entry_time_ts)

        # Print the context
        print(f"Displaying Decoded Details for Sample Index: {selected_index}")
        print(f"Prediction Time: {entry_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Initial Prices at Prediction Time:")
        with pd.option_context('display.precision', 2):
            print(initial_prices_series.to_string())
        print("-" * (COL_WIDTH_KEY + COL_WIDTH_TRUE + COL_WIDTH_PRED + COL_WIDTH_MAE + 3))
        
        # Print the table header
        print(f"{'METRIC':<{COL_WIDTH_KEY}}|{'Sample Decoded Ground Truth':^{COL_WIDTH_TRUE}}|{'Sample Decoded Predictions':^{COL_WIDTH_PRED}}|{'MAE':>{COL_WIDTH_MAE}}")
        print("-" * (COL_WIDTH_KEY + COL_WIDTH_TRUE + COL_WIDTH_PRED + COL_WIDTH_MAE + 3))

        with pd.option_context('display.precision', 4):
            # Use the selected index to get the correct row from the DataFrames
            for key in decoded_output_keys:
                true_val = trues_df.iloc[selected_index].get(key, float('nan'))
                pred_val = preds_df.iloc[selected_index].get(key, float('nan'))
                mae_val = decoded_mae.get(key, float('nan'))    
                print(f"{key:<{COL_WIDTH_KEY}}|{true_val:^{COL_WIDTH_TRUE}.4f}|{pred_val:^{COL_WIDTH_PRED}.4f}|{mae_val:>{COL_WIDTH_MAE}.4f}")

    print("=" * (COL_WIDTH_KEY + COL_WIDTH_TRUE + COL_WIDTH_PRED + COL_WIDTH_MAE + 3) + "\n")
    
    
    print("\n--- Subpopulation Analysis based on True Time Duration ---")
    time_thresholds = [300, 900]  

    for ticker in tickers:
        print(f"\n===== Analyzing Ticker: {ticker} =====")
        maes_by_threshold = {}
        samples_by_threshold = {}
        
        true_duration = trues_df[f'{ticker}_t_time'] - trues_df[f'{ticker}_s_time']

        for threshold in time_thresholds:
            mask = true_duration < threshold
            num_samples = mask.sum()
            samples_by_threshold[threshold] = num_samples

            if num_samples > 0:
                trues_subpop = trues_df[mask]
                preds_subpop = preds_df[mask]
                maes_by_threshold[threshold] = (trues_subpop - preds_subpop).abs().mean()
            else:

                maes_by_threshold[threshold] = pd.Series(dtype='float64')

        header = f"{'METRIC':<{COL_WIDTH_KEY}}|"
        separator = "-" * (COL_WIDTH_KEY + 1)

        for threshold in time_thresholds:
            header_text = f"MAE (< {threshold}s, n={samples_by_threshold[threshold]})"
            header += f"{header_text:^{COL_WIDTH_MAE}}|"
            separator += "-" * (COL_WIDTH_MAE + 1)
        
        print(header)
        print(separator)


        for key in decoded_output_keys:
            if key.startswith(ticker):
                row_str = f"{key:<{COL_WIDTH_KEY}}|"
                for threshold in time_thresholds:
                    # Get the specific MAE value for the key from the stored Series
                    mae_val = maes_by_threshold[threshold].get(key, float('nan'))
                    row_str += f"{mae_val:^{COL_WIDTH_MAE}.4f}|"
                print(row_str)
    
    print("\n" + "="*80 + "\n")

    return mae, decoded_mae








def cluster_manager(predictions: list, population_thresh: int, confidence_thresh: float, HSP_seconds: int):

    if len(predictions) < population_thresh:
        return [] # Not enough data to form a meaningful cluster

    prices = np.array([p['price'] for p in predictions]).reshape(-1, 1)
    timestamps = np.array([p['time'].timestamp() for p in predictions]).reshape(-1, 1)

    price_std = prices.std()
    if price_std == 0: return [] # Avoid division by zero if all prices are the same


    price_bw = price_std * (len(prices)**(-1/5))

    kde_price = KernelDensity(kernel='gaussian', bandwidth=price_bw).fit(prices)
    price_range = np.linspace(prices.min(), prices.max(), 200).reshape(-1, 1)
    log_density_price = kde_price.score_samples(price_range)
    
    # Find peaks in the price density using a prominence threshold
    price_peaks_idx, _ = find_peaks(log_density_price, prominence=0.1)
    
    if not price_peaks_idx.any():
        return []

    identified_clusters = []

    # --- Step 3: For each price peak, run a 1D KDE on Time ---
    for peak_idx in price_peaks_idx:
        peak_price = price_range[peak_idx][0]
        
        # Filter predictions that belong to this price peak
        price_tolerance = price_bw 
        price_cluster_preds = [p for p in predictions if abs(p['price'] - peak_price) <= price_tolerance]

        if len(price_cluster_preds) < population_thresh:
            continue

        # --- Step 4: 1D KDE on Time for the price-filtered subset ---
        time_cluster_stamps = np.array([p['time'].timestamp() for p in price_cluster_preds]).reshape(-1, 1)
        
        time_std = time_cluster_stamps.std()
        if time_std == 0: continue
        time_bw = time_std * (len(time_cluster_stamps)**(-1/5))

        kde_time = KernelDensity(kernel='gaussian', bandwidth=time_bw).fit(time_cluster_stamps)
        time_range = np.linspace(time_cluster_stamps.min(), time_cluster_stamps.max(), 200).reshape(-1, 1)
        log_density_time = kde_time.score_samples(time_range)

        time_peaks_idx, _ = find_peaks(log_density_time, prominence=0.1)

        for t_peak_idx in time_peaks_idx:
            peak_time_ts = time_range[t_peak_idx][0]
            peak_time = datetime.fromtimestamp(peak_time_ts)
            
            # --- Step 5: Calculate Confidence and Finalize Cluster ---
            num_samples = len(price_cluster_preds)
            peak_height = np.exp(log_density_price[peak_idx])

            # Normalize components to [0, 1] before combining
            norm_samples = min(num_samples / (2 * population_thresh), 1.0) # Example normalization
            norm_height = min(peak_height / 1.0, 1.0) # Assuming max height is around 1.0
            confidence = 0.5 * norm_samples + 0.5 * norm_height
            
            if confidence > confidence_thresh:
                start_time = peak_time - timedelta(seconds=HSP_seconds)
                identified_clusters.append({
                    'start_time': start_time,
                    'end_time': peak_time,
                    'target_price': peak_price,
                    'confidence': confidence,
                    'status': 'PENDING' # Initial state
                })
    return identified_clusters





def cull_manager(prediction_cache: dict, cycle_duration_seconds: int) -> dict:
    culled_cache = {}
    for ticker, predictions in prediction_cache.items():

        for pred in predictions:
            pred['remaining_lifespan'] -= cycle_duration_seconds
        
        culled_cache[ticker] = [p for p in predictions if p['remaining_lifespan'] > 0]
        
    return culled_cache





def generate_raw_pred_plot(plot_data, tickers, roll_num, plot_title):
    num_tickers = len(tickers)
    fig, axes = plt.subplots(num_tickers, 1, figsize=(20, 8 * num_tickers), sharex=True)
    if num_tickers == 1: axes = [axes]

    for ax, ticker in zip(axes, tickers):
        if plot_data[ticker]['true_s']:
            true_s_points = sorted(plot_data[ticker]['true_s'], key=lambda x: x['time'])
            ax.plot([p['time'] for p in true_s_points], [p['price'] for p in true_s_points],
                    color='black', alpha=0.9, linewidth=2, label='True Price Path', zorder=10)


        all_plot_points = []
        if 'prediction_snapshots' in plot_data[ticker]:
            for snapshot_data in plot_data[ticker]['prediction_snapshots'].values():
                all_plot_points.extend(snapshot_data['plot_data'])

        if all_plot_points:
            ax.scatter(
                [p['time'] for p in all_plot_points], 
                [p['price'] for p in all_plot_points],
                s=5,
                alpha=0.3,
                label=f'All Raw Preds' # A single legend entry
            )
        
        ax.set_title(f'Raw Prediction Snapshots: {ticker}', fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Event Time', fontsize=14)
    fig.suptitle(f'Raw Predictions (Roll {roll_num+1})', fontsize=20, y=0.99)
    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_format)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.savefig(plot_title)
    plt.close()

def generate_actions_plot(plot_data: dict, tickers: list, roll_num: int, plot_title: str):

    num_tickers = len(tickers)
    fig, axes = plt.subplots(num_tickers, 1, figsize=(20, 8 * num_tickers), sharex=True)
    if num_tickers == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        if plot_data[ticker]['true_s']:
            true_points = sorted(plot_data[ticker]['true_s'], key=lambda x: x['time'])
            ax.plot([p['time'] for p in true_points], [p['price'] for p in true_points],
                    color='black', alpha=0.9, linewidth=2, label='True Price Path', zorder=10)

        if plot_data[ticker]['actions']:
            actions = plot_data[ticker]['actions']
            

            hlines_y = [a['target_price'] for a in actions]
            hlines_xmin = [a['start_time'] for a in actions]
            hlines_xmax = [a['end_time'] for a in actions]
            

            alphas = np.clip([a['confidence'] for a in actions], 0.3, 1.0) 
            
            scatter_x = [a['end_time'] for a in actions]
            scatter_y = [a['target_price'] for a in actions]

            # Plot all horizontal lines and scatter points
            ax.hlines(y=hlines_y, xmin=hlines_xmin, xmax=hlines_xmax,
                      colors='red', linestyles='-', lw=5, alpha=alphas,
                      label='Declared Action')
            
            ax.scatter(scatter_x, scatter_y, color='red', s=5, 
                       zorder=11)
        
        # --- Formatting ---
        ax.set_title(f'Confident Declared Actions: {ticker}', fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Handle legend to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Overall Figure Formatting ---
    axes[-1].set_xlabel('Event Time', fontsize=14)
    fig.suptitle(f'Confident Actions (Roll {roll_num+1})', fontsize=20, y=0.99)
    
    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_format)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    plt.savefig(plot_title)
    plt.close()


def run_test(model, loader, tickers, output_keys_sorted,  forecast_depth_seconds, device, roll_num, set="validation",
             HSP_seconds=180,
             sample_freq_n=1, 
             pred_lifespan_seconds=600, # Lifespan in seconds
             query_freq_seconds=60,      # Query frequency in seconds
             population_thresh=100,       # Min points to form a cluster
             confidence_thresh=0.6):      # Min confidence to declare a position
    
    model.eval()
    print("hit")
    prediction_cache = {ticker: [] for ticker in tickers}
    # In run_test(), initialize these dictionaries at the top
    raw_pred_plot = {ticker: {'true_s': [], 'prediction_snapshots': {}} for ticker in tickers}
    actions_plot = {ticker: {'true_s': [], 'actions': []} for ticker in tickers}
    print("\n" + "="*20 + f" {set} simulation roll {roll_num+1} " + "="*20)
    loop = tqdm(loader, desc=f"Roll {roll_num+1} Testing", unit="batch", leave=False)

    next_processing_time = None

    with torch.no_grad():
        for time_entry_ts, X, true_y, initial_prices_batch in loop:
            X = X.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(X)

            for i in range(0, len(time_entry_ts), sample_freq_n):
                current_sim_time = datetime.fromtimestamp(time_entry_ts[i])
                if next_processing_time is None:
                    next_processing_time = current_sim_time + timedelta(seconds=query_freq_seconds)


                initial_times = {'a_time': {ticker: time_entry_ts[i] for ticker in initial_prices_batch[i].index}}
                decoded_preds = reverse_transform_Y(logits[i], output_keys_sorted, initial_prices_batch[i], initial_times, forecast_depth_seconds)
                decoded_trues = reverse_transform_Y(true_y[i], output_keys_sorted, initial_prices_batch[i], initial_times, forecast_depth_seconds)

                for ticker in tickers:
         
                    for p_type in ['s']: # ignore leg 2  for now
                        if f'{ticker}_{p_type}_time' in decoded_preds:
                            pred_time = datetime.fromtimestamp(decoded_preds[f'{ticker}_{p_type}_time'])
                            if pred_time > current_sim_time:
                                raw_pred = {
                                    'time': pred_time,
                                    'price': decoded_preds[f'{ticker}_{p_type}_value'],
                                    'p_type': p_type,
                                    'remaining_lifespan': pred_lifespan_seconds 
                                }
                                prediction_cache[ticker].append(raw_pred)
         
                        
                        if f'{ticker}_{p_type}_time' in decoded_trues:
                            true_data = {
                                'time': datetime.fromtimestamp(decoded_trues[f'{ticker}_{p_type}_time']), 
                                'price': decoded_trues[f'{ticker}_{p_type}_value']
                            }
            
                            actions_plot[ticker][f'true_{p_type}'].append(true_data)

        
                if current_sim_time >= next_processing_time:
                    # First, cull expired predictions from the cache
                    cycle_duration = (current_sim_time - (next_processing_time - timedelta(seconds=query_freq_seconds))).total_seconds()
                    prediction_cache = cull_manager(prediction_cache, cycle_duration)

                    for ticker in tickers:
                        active_predictions = prediction_cache[ticker]
                        
                        # --- 1. STORE RAW SNAPSHOT FOR THE RAW PLOT ---
                        if active_predictions:
                            snapshot_full = copy.deepcopy(active_predictions)
                            # Create a deduplicated version for efficient plotting
                            unique_points = {(p['time'], p['price']) for p in snapshot_full}
                            snapshot_for_plot = [dict(zip(['time', 'price'], t)) for t in unique_points]
                            
                            # Store in the raw_pred_plot dictionary
                            raw_pred_plot[ticker]['prediction_snapshots'][next_processing_time] = {
                                'full_data': snapshot_full,
                                'plot_data': snapshot_for_plot
                            }
                        
        
                        declared_positions = cluster_manager(
                            predictions=active_predictions,
                            population_thresh=population_thresh,
                            confidence_thresh=confidence_thresh,
                            HSP_seconds=HSP_seconds
                        )
                        
        
                        if declared_positions:
                            actions_plot[ticker]['actions'].extend(declared_positions)
                    
                    next_processing_time += timedelta(seconds=query_freq_seconds)
    for ticker in tickers:
        raw_pred_plot[ticker]['true_s'] = actions_plot[ticker]['true_s'] 

    # Correctly call the renamed plotting function

    generate_raw_pred_plot(raw_pred_plot, tickers, roll_num, f'asset_pricing/{set}_pred_roll_{roll_num+1}.png')
    print("saved plot")
    generate_actions_plot(actions_plot, tickers, roll_num, f'asset_pricing/{set}_exposure_roll_{roll_num+1}.png')
    print("saved plot")
    return actions_plot # Or return both plot dictionaries if needed



        

    

def train_and_save_artifacts(config: Dict):
    Session = sessionmaker(bind=engine)
    with Session() as session:
        train_ids_query = session.query(TrainItem.id).order_by(TrainItem.time_entry_ts.asc())
        all_train_item_ids = [r[0] for r in train_ids_query.all()]
    input_keys_sorted = sorted(config["params"])
    tickers = sorted(config["ticker"])
    metrics = ['duration', 'drift', 'direction']
    time_points = ['as', 'st']
    forecast_depth_seconds = config['forecast_depth'] * 60
    
    output_keys_sorted = [
        f"{ticker}_{point}_{metric}" 
        for ticker, metric, point in product(tickers, metrics, time_points)
    ]
    output_keys_sorted = sorted(output_keys_sorted)

    decoded_points = ['s', 't']
    decoded_metrics = ['value', 'time']
    decoded_output_keys =  [
        f"{ticker}_{point}_{metric}" 
        for ticker, point, metric  in product(tickers, decoded_points, decoded_metrics)
    ]

    decoded_output_keys = sorted(decoded_output_keys)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Found {len(all_train_item_ids)} TrainItem records. Using device: {device}")



    model = TransposeCNN(in_channels=1, base_filters=32, input_shape=(len(input_keys_sorted),1), output_size =len(output_keys_sorted), dropout_rate=0.0).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    criterion = torch.nn.L1Loss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    epochs_per_roll = 4     # config.get("epochs_per_roll", 10)
    l1_reg_strength = config.get("l1_reg_strength", 0.0)
    



    norm = ['RSI', 'TRENDMODE', 'HT_SINE_SINE', 'ATR' , 'HT_SINE_LEAD_SINE', 'CURRENT_PRICE']

    total_size = len(all_train_item_ids)
    train_size = int(total_size * config['initial_train_percentage'])
    val_size = int(total_size * config['validation_percentage'])
    test_size = int(total_size * config['test_percentage'])
    
    roll_step_size = test_size

    for roll_num in range(config['num_rolls']):
        train_start = roll_num * roll_step_size
        train_end = train_start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > total_size:
                print("Not enough data for the next full roll. Stopping.")
                break
        
        current_train_ids = all_train_item_ids[train_start:train_end]
        current_validation_ids = all_train_item_ids[train_end:val_end]
        current_test_ids = all_train_item_ids[val_end:test_end]
        
        print(f"\n{'='*20} Starting Roll {roll_num + 1}/{config['num_rolls']} {'='*20}")
        print(f"Train set size: {len(current_train_ids)} ({train_start}:{train_end})")
        print(f"Validation set size: {len(current_validation_ids)} ({train_end}:{val_end})")
        print(f"Test set size: {len(current_test_ids)} ({val_end}:{test_end})")

     
        train_ds = LazyTrainDataset(current_train_ids, TrainItem)
        collate_fn = CustomCollate(
            train_ds, 
            input_keys_sorted, 
            output_keys_sorted, 
            norm, 
            forecast_depth_seconds
        )

        train_loader = DataLoader(
            dataset=list(range(len(train_ds))), 
            batch_size=64, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True, 
            persistent_workers=True, 
            collate_fn=collate_fn
        )
        best_mae = float('inf')
        patience = 2
        patience_counter = 0
        best_model_state = None
        for epoch in range(1, epochs_per_roll + 1):
            avg_train_loss = run_training_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                roll_num=roll_num,
                epoch=epoch,
                epochs_per_roll=epochs_per_roll,
                l1_reg_strength=l1_reg_strength
            )
            print(f"Roll {roll_num+1}, Epoch {epoch}/{epochs_per_roll} completed. Average Loss: {avg_train_loss:.4f}")

            # --- Validation Step ---
            val_ds = LazyTrainDataset(current_validation_ids, TrainItem)
            val_collate_fn = CustomCollate(
                val_ds, input_keys_sorted, output_keys_sorted, norm, forecast_depth_seconds
            )
            val_loader = DataLoader(
                dataset=list(range(len(val_ds))), batch_size=64, shuffle=False, 
                num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=val_collate_fn
            )
            
            mae, decoded_mae = run_validation_aggregate(
                model=model, loader=val_loader, tickers=tickers,
                output_keys_sorted=output_keys_sorted, decoded_output_keys=decoded_output_keys,
                forecast_depth_seconds=forecast_depth_seconds, device=device, roll_num=roll_num
            )

            if mae < best_mae:
                print(f"Validation MAE improved from {best_mae:.4f} to {mae:.4f}. Saving model state.")
                best_mae = mae
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                print(f"Validation MAE did not improve, best: {best_mae:.4f}, current: {mae:.4f}. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break 
        
       
        print(f"\nFinished training for Roll {roll_num + 1}. Best Validation MAE: {best_mae:.4f}")
        
        if best_model_state:
            print("Loading best model state for test evaluation.")
            model.load_state_dict(best_model_state)

        val_result = run_test(
            model=model, loader=val_loader, tickers=tickers,
            output_keys_sorted=output_keys_sorted, 
            forecast_depth_seconds=forecast_depth_seconds, device=device, roll_num=roll_num,  set="val")    


        test_ds = LazyTrainDataset(current_test_ids, TrainItem)
        test_collate_fn = CustomCollate(
            test_ds, input_keys_sorted, output_keys_sorted, norm, forecast_depth_seconds
        )
        test_loader = DataLoader(
            dataset=list(range(len(test_ds))), batch_size=128, shuffle=False,
            num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=test_collate_fn
        )
        

        test_result = run_test(
            model=model, loader=test_loader, tickers=tickers,
            output_keys_sorted=output_keys_sorted, 
            forecast_depth_seconds=forecast_depth_seconds, device=device, roll_num=roll_num,  set="test")      
    return


def main():
    config_filepath = sys.argv[1]
    
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    sorted_data = sort_json_recursively(config)
    
    with open(sorted_data["config_filepath"], 'w') as f:
        json.dump(sorted_data, f, indent=4)

    train_and_save_artifacts(
        config=sorted_data
    )
    

if __name__ == '__main__':
    main()


