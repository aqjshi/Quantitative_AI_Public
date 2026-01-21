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

import pandas as pd
from train_helper import  TransposeCNN, LazyTrainDataset, CustomCollate, print_in_chunks

from sqlalchemy.orm import sessionmaker
from itertools import product

from datetime import datetime
from urllib.parse import parse_qs, unquote

class LazyTrainDataset(Dataset):
    def __init__(self, item_ids: List[int], item_class):
        self.item_ids           = item_ids
        self.item_class         = item_class
        self.db_url             = DATABASE_URL 
        self.engine             = None
        self.Session            = None


    def __len__(self):
        return len(self.item_ids)

    def _init_db(self):
        if self.engine is None:
            self.engine         = create_engine(DATABASE_URL)
            self.Session        = sessionmaker(bind=self.engine)

    def __getitem__(self, index: int, input_keys_sorted: list, output_keys_sorted: list, norm: list, forecast_depth_seconds: int):
        self._init_db()

        with self.Session() as session:
            item_id = self.item_ids[index]
            itm = session.get(self.item_class, item_id)
            # print(f"DEBUG: itm = {itm}")

            if not itm or not itm.tensor:
                return None, torch.zeros(len(input_keys_sorted)), torch.zeros(len(output_keys_sorted)), None
            
            time_entry_ts = itm.time_entry_ts
            # print(f"DEBUG: time_entry_ts = {time_entry_ts}")

            # Cast all values to float
            processed_tensor = {
                key: float(value[0]) if isinstance(value, (list, np.ndarray)) else float(value)
                for key, value in itm.tensor.items()
            }
            
            tensor_series = pd.Series(processed_tensor)
            # print(f"DEBUG: Initial tensor_series = \n{tensor_series}")


            multi_index = pd.MultiIndex.from_tuples(
                [(parse_qs(unquote(k))['ticker'][0], k) for k in tensor_series.index],
                names=['ticker', 'original_key']
            )
            tensor_series.index = multi_index
            
            # Separate current prices and indicators
            current_prices = tensor_series[tensor_series.index.get_level_values('original_key').str.contains("CURRENT_PRICE", na=False)]
            current_prices.index = current_prices.index.get_level_values('ticker')
            
            indicators = tensor_series[~tensor_series.index.get_level_values('original_key').str.contains("CURRENT_PRICE", na=False)]
            
        
            if norm:
                norm_pattern = '|'.join(norm)
                indicators_as_is = indicators[indicators.index.get_level_values('original_key').str.contains(norm_pattern, na=False)]
                indicators_to_normalize = indicators[~indicators.index.get_level_values('original_key').str.contains(norm_pattern, na=False)]
            else:
                indicators_as_is = pd.Series(dtype=float)
                indicators_to_normalize = indicators

            transformed_x_parts = []

            if not indicators_to_normalize.empty:
                mapped_prices = indicators_to_normalize.index.get_level_values('ticker').map(current_prices)
                if not mapped_prices.isnull().all():
                    normalized_values = np.log(indicators_to_normalize.values / mapped_prices.values, where=((indicators_to_normalize.values > 0) & (mapped_prices.values > 0)), out=np.full_like(indicators_to_normalize.values, 0.0))
                    normalized_series = pd.Series(normalized_values, index=indicators_to_normalize.index.get_level_values('original_key'))
                    transformed_x_parts.append(normalized_series)

           
            as_is_series = pd.Series(indicators_as_is.values, index=indicators_as_is.index.get_level_values('original_key'))
            transformed_x_parts.append(as_is_series)
            current_price_series = pd.Series(current_prices.values, index=[f"ticker={ticker}&function=CURRENT_PRICE" for ticker in current_prices.index])
            transformed_x_parts.append(current_price_series)

  
            final_x_series = pd.concat(transformed_x_parts)
            x_values = final_x_series.reindex(input_keys_sorted, fill_value=0.0).values


            final_x_series = pd.concat(transformed_x_parts)
            x_values = final_x_series.reindex(input_keys_sorted, fill_value=0.0).values


            X_tensor = torch.tensor(x_values, dtype=torch.float32)
            
            points_data = {
                'a': {'time': getattr(itm, 'a_time', {}), 'value': getattr(itm, 'a_value', {})},
                's': {'time': getattr(itm, 's_time', {}), 'value': getattr(itm, 's_value', {})}, #, 'sma': getattr(itm, 's_sma', {})},
                't': {'time': getattr(itm, 't_time', {}), 'value': getattr(itm, 't_value', {})}, #, 'sma': getattr(itm, 't_sma', {})},
                
                'b': {'time': getattr(itm, 'b_time', {}), 'value': getattr(itm, 'b_value', {})},
            }
            
            df_y = pd.DataFrame({(point, metric): data for point, metrics in points_data.items() for metric, data in metrics.items()})


            y_dict = {}
            a_time = df_y.get(('a', 'time'), 0) 
            s_time = df_y.get(('s', 'time'), 0) 
            t_time = df_y.get(('t', 'time'), 0) 
            b_time = df_y.get(('b', 'time'), 0)
            y_dict['as_duration'] = (s_time - a_time) / forecast_depth_seconds
            y_dict['st_duration'] = (t_time - a_time) / forecast_depth_seconds
           
            s_value, a_value = df_y.get(('s', 'value')), df_y.get(('a', 'value'))
            b_value, t_value = df_y.get(('b', 'value')), df_y.get(('t', 'value'))
            y_dict['as_drift'] = np.log(s_value / a_value, where=((s_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            y_dict['st_drift'] = np.log(t_value / a_value, where=((t_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            final_y_series_data = {
                f"{ticker}_{key}": series.get(ticker, 0.0)
                for key, series in y_dict.items()
                for ticker in df_y.index
            }
            final_y_series = pd.Series(final_y_series_data)
            
            y_values = final_y_series.reindex(output_keys_sorted, fill_value=0.0).values
            Y_tensor = torch.tensor(y_values, dtype=torch.float32)

        return time_entry_ts, X_tensor, Y_tensor, current_prices


def reverse_transform_Y(y_tensor, output_keys_sorted, initial_prices, initial_times, forecast_depth_seconds):
    reverse_transformed_values = {}
    y_values_dict = dict(zip(output_keys_sorted, y_tensor.tolist()))
    

    for key, value in y_values_dict.items():
        ticker = key.split('_')[0]
        time_point = key.split('_')[1]

        if 'duration' in key:
            time_difference = value * forecast_depth_seconds

            start_time = initial_times.get('a_time', {}).get(ticker)
                
    
            final_time = start_time + time_difference
            reverse_transformed_values[f"{ticker}_{time_point[1]}_time"] = final_time

        elif 'drift' in key:
            initial_price = initial_prices.get(ticker)
            
      
            original_value = np.exp(value) * initial_price
            reverse_transformed_values[f"{ticker}_{time_point[1]}_value"] = original_value

    return reverse_transformed_values
    
def sort_json_recursively(obj):
    if isinstance(obj, dict):
        # If it's a dictionary, sort by key and recursively sort each value
        return {k: sort_json_recursively(obj[k]) for k in sorted(obj)}
    
    if isinstance(obj, list):
        # If it's a list, recursively sort each item first
        sorted_list_items = [sort_json_recursively(item) for item in obj]
        
        # Then, sort the list itself. This handles lists of mixed types
        # by converting non-string items to strings for sorting purposes.
        return sorted(sorted_list_items, key=lambda x: str(x))

    # Return numbers, strings, booleans, etc., as they are
    return obj


def run_validation(model, loader, tickers, output_keys_sorted, decoded_output_keys, forecast_depth_seconds, device, roll_num):
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
        
  
        print(f"{'METRIC':<{COL_WIDTH_KEY}}|{'Sample Decoded Ground Truth':^{COL_WIDTH_TRUE}}|{'Sample Decoded Predictions':^{COL_WIDTH_PRED}}|{'MAE':>{COL_WIDTH_MAE}}")
        print("-" * (COL_WIDTH_KEY + COL_WIDTH_TRUE + COL_WIDTH_PRED + COL_WIDTH_MAE + 3))

        with pd.option_context('display.precision', 4):
    
            for key in decoded_output_keys:
                true_val = trues_df.iloc[selected_index].get(key, float('nan'))
                pred_val = preds_df.iloc[selected_index].get(key, float('nan'))
                mae_val = decoded_mae.get(key, float('nan'))    
                print(f"{key:<{COL_WIDTH_KEY}}|{true_val:^{COL_WIDTH_TRUE}.4f}|{pred_val:^{COL_WIDTH_PRED}.4f}|{mae_val:>{COL_WIDTH_MAE}.4f}")

    print("=" * (COL_WIDTH_KEY + COL_WIDTH_TRUE + COL_WIDTH_PRED + COL_WIDTH_MAE + 3) + "\n")
    
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
                    mae_val = maes_by_threshold[threshold].get(key, float('nan'))
                    row_str += f"{mae_val:^{COL_WIDTH_MAE}.4f}|"
                print(row_str)
    
    print("\n" + "="*80 + "\n")

    return mae, decoded_mae




    
def train_epoch(model, train_loader,  optimizer, criterion, scaler, device, roll_num, epoch, epochs_per_roll, l1_reg_strength):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Roll {roll_num+1}, Epoch {epoch}/{epochs_per_roll}", unit="batch")
    weight_matrix = torch.ones(4, 5) 
    weight_matrix[2, 3] = 10.0
    weight_matrix = weight_matrix.to(device)
    
    for _, X, normalized_gt_Y, initial_prices in loop:
        X = X.to(device, non_blocking=True)
        normalized_gt_Y = normalized_gt_Y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(X)
    
            
            loss = criterion(logits, normalized_gt_Y)
            if l1_reg_strength > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_reg_strength * l1_norm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        total_train_loss += batch_loss * X.size(0)
        loop.set_postfix(loss=f"{batch_loss:.4f}")
    
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    return avg_train_loss

def train(config: Dict):
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
    epochs_per_roll = 4  
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
        for epoch in range(1, epochs_per_roll + 1):
            avg_train_loss = train_epoch(
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

 
            val_ds = LazyTrainDataset(current_validation_ids, TrainItem)
            val_collate_fn = CustomCollate(
                val_ds, input_keys_sorted, output_keys_sorted, norm, forecast_depth_seconds
            )
            val_loader = DataLoader(
                dataset=list(range(len(val_ds))), batch_size=64, shuffle=False, 
                num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=val_collate_fn
            )
            
            mae, decoded_mae = run_validation(
                model=model, loader=val_loader, tickers=tickers,
                output_keys_sorted=output_keys_sorted, decoded_output_keys=decoded_output_keys,
                forecast_depth_seconds=forecast_depth_seconds, device=device, roll_num=roll_num
            )

        


def main():
    config_filepath = sys.argv[1]
    
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    sorted_data = sort_json_recursively(config)
    
    with open(sorted_data["config_filepath"], 'w') as f:
        json.dump(sorted_data, f, indent=4)

    train(
        config=sorted_data
    )
    

if __name__ == '__main__':
    main()


