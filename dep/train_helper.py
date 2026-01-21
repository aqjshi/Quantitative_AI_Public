import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
from sqlalchemy.orm import sessionmaker
from db import  DATABASE_URL,  engine
from tqdm import tqdm
from urllib.parse import parse_qs, unquote
from sqlalchemy import create_engine
import pandas as pd
import random 
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.use('Agg')


def save_plane_as_html(plane: List[Tuple], current_time: float, asset: str, plane_type: str) -> str:
    TIME_GRID_RESOLUTION = 20 
    PRICE_GRID_POINTS = 15 

    if len(plane) < 5:
        return f"Skipping {plane_type} plane visualization: Too few points ({len(plane)})."

    df = pd.DataFrame(plane, columns=['id', 'lifespan', 'price', 'time'])
    
    # --- Data Preparation ---
    time_data_seconds = df['time'].values - current_time
    price_data = df['price'].values
    lifespan_data = df['lifespan'].values
    time_data_minutes = time_data_seconds / 60.0
    
    # --- 1. Standardization (Normalization) ---
    time_mean = time_data_minutes.mean()
    time_std = time_data_minutes.std()
    price_mean = price_data.mean()
    price_std = price_data.std()

    time_std_safe = time_std if time_std > 1e-6 else 1.0
    price_std_safe = price_std if price_std > 1e-6 else 1.0

    # Normalize BOTH time and price data
    time_data_norm = (time_data_minutes - time_mean) / time_std_safe
    price_data_norm = (price_data - price_mean) / price_std_safe
    
    # --- CRITICAL CHANGE FOR 2D KDE ---
    # Combine the standardized data into a single 2xN array
    data_2d_norm = np.vstack([time_data_norm, price_data_norm])
    kde_2d = gaussian_kde(data_2d_norm, bw_method='scott') 

    # X Grid (Time) Setup
    x_min_norm, x_max_norm = time_data_norm.min(), time_data_norm.max()
    x_range_norm = x_max_norm - x_min_norm
    x_min_adj_norm = x_min_norm - 0.05 * x_range_norm
    x_max_adj_norm = x_max_norm + 0.05 * x_range_norm
    X_norm_1D = np.linspace(x_min_adj_norm, x_max_adj_norm, TIME_GRID_RESOLUTION)

    y_min_norm, y_max_norm = price_data_norm.min(), price_data_norm.max()
    y_range_norm = y_max_norm - y_min_norm
    y_min_adj_norm = y_min_norm - 0.05 * y_range_norm
    y_max_adj_norm = y_max_norm + 0.05 * y_range_norm
    Y_norm_1D = np.linspace(y_min_adj_norm, y_max_adj_norm, PRICE_GRID_POINTS)


    X_norm_matrix, Y_norm_matrix = np.meshgrid(X_norm_1D, Y_norm_1D)
  
    Z_2D_points = np.vstack([X_norm_matrix.ravel(), Y_norm_matrix.ravel()])
    Z_combined = kde_2d(Z_2D_points).reshape(X_norm_matrix.shape)
    

    X_final = X_norm_matrix * time_std_safe + time_mean 
    Y_final = Y_norm_matrix * price_std_safe + price_mean 


    fig = go.Figure()

    fig.add_trace(go.Surface(

        x=X_final[0, :], # Use a 1D slice for X
        y=Y_final[:, 0], # Use a 1D slice for Y
        z=Z_combined, 
        colorscale='Viridis',
        opacity=0.6,
        name='2D KDE Surface' # Renamed to reflect 2D KDE
    ))

    Z_floor = Z_combined.min() + (Z_combined.max() - Z_combined.min()) * 0.05 
    
    fig.add_trace(go.Scatter3d(
        x=time_data_minutes, 
        y=price_data,
        z=np.full_like(time_data_minutes, Z_floor), 
        mode='markers',
        marker=dict(
            size=5,
            color=lifespan_data, 
            colorscale='Plasma',
            colorbar=dict(title='Lifespan (s)'), 
            opacity=1.0
        ),
        name='Raw Predictions'
    ))
    
    plot_title = f"3D Prediction Plane: {asset} ({plane_type}) - PROPER 2D KDE"
    
    # 5. Set Layout and Title
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='Time (Relative Minutes)',
            yaxis_title='Price ($)',
            zaxis_title='Joint Density (2D KDE Value)', # Renamed Z-axis title
            aspectmode='manual',
            aspectratio=dict(x=5.0, y=1.0, z=0.5) 
        )
    )
    
    output_filename = f"snapshot/time={int(current_time)}_asset={asset}_type={plane_type}.html"
    fig.write_html(output_filename, auto_open=False, include_plotlyjs='cdn')

    return output_filename



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
                # Ensure mapped_prices is not empty and has the same index as indicators_to_normalize
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
            # y_dict['tb_duration'] = (b_time - t_time) / forecast_depth_seconds
      
            s_value, a_value = df_y.get(('s', 'value')), df_y.get(('a', 'value'))
            b_value, t_value = df_y.get(('b', 'value')), df_y.get(('t', 'value'))
            # s_sma, t_sma =  df_y.get(('s', 'sma')), df_y.get(('t', 'sma'))
            y_dict['as_drift'] = np.log(s_value / a_value, where=((s_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            y_dict['st_drift'] = np.log(t_value / a_value, where=((t_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            # y_dict['tb_drift'] = np.log(t_value / a_value, where=((t_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            # y_dict['as_direction'] = np.log(s_sma / a_value, where=((s_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            # y_dict['st_direction'] = np.log(t_sma / a_value, where=((t_value > 0) & (a_value > 0)), out=np.full(len(df_y), 0.0))
            final_y_series_data = {
                f"{ticker}_{key}": series.get(ticker, 0.0)
                for key, series in y_dict.items()
                for ticker in df_y.index
            }
            final_y_series = pd.Series(final_y_series_data)
            
            y_values = final_y_series.reindex(output_keys_sorted, fill_value=0.0).values
            Y_tensor = torch.tensor(y_values, dtype=torch.float32)

        return time_entry_ts, X_tensor, Y_tensor, current_prices


def print_in_chunks(data_array, chunk_size=10):

    for j in range(0, len(data_array), chunk_size):
        chunk = data_array[j:j + chunk_size]
   
        formatted_chunk = [f"{x:.3f}" for x in chunk]
    
        print(f"    {' '.join(formatted_chunk)}")


def simple_collate(batch):
    return list(zip(*batch))





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
        # elif 'direction' in key:
        #     initial_price = initial_prices.get(ticker)
            
      
        #     original_value = np.exp(value) * initial_price
        #     reverse_transformed_values[f"{ticker}_{time_point[1]}_sma"] = original_value

    return reverse_transformed_values
    
def run_training_epoch(model, train_loader,  optimizer, criterion, scaler, device, roll_num, epoch, epochs_per_roll, l1_reg_strength):
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


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers, batch norm, and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path (to match dimensions if stride or channels change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The skip connection
        out = F.leaky_relu(out, 0.2)
        return out


class TransposeCNN(nn.Module):
    def __init__(self, in_channels, base_filters, input_shape, output_size, dropout_rate):
        super(TransposeCNN, self).__init__()
        
        # You can optionally store the input_shape for reference
        self.input_shape = input_shape
        
        # --- Encoder (Downsampling Path) ---
        self.initial_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(base_filters)
        
        self.enc_block1 = ResidualBlock(base_filters, base_filters * 2, stride=2)
        self.enc_block2 = ResidualBlock(base_filters * 2, base_filters * 4, stride=2)
        
        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(base_filters * 4, base_filters * 8, stride=2)
        
        # --- Decoder (Upsampling Path) ---
        self.tconv1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec_block1 = ResidualBlock(base_filters * 8, base_filters * 4) # 4+4 from concat

        self.tconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec_block2 = ResidualBlock(base_filters * 4, base_filters * 2) # 2+2 from concat
        
        self.tconv3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec_block3 = ResidualBlock(base_filters * 2, base_filters) # 1+1 from concat

        # --- Output Head ---
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, output_size * 4)) # Pool to an intermediate size
        
        self.output_head = nn.Sequential(
            nn.Linear(output_size * 4, output_size * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(output_size * 2, output_size)
        )
        

    def forward(self, x):
        x = x.view(x.size(0), 1, 1, -1)
        
        # The rest of your forward pass can now proceed correctly
        x1 = F.leaky_relu(self.bn_initial(self.initial_conv(x)), 0.2)
        x2 = self.enc_block1(x1)
 
        x3 = self.enc_block2(x2)
  
        b = self.bottleneck(x3)

        d1 = self.tconv1(b)

        d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d1, x3], dim=1) # Skip connection from encoder

        d1 = self.dec_block1(d1)
      
        
        d2 = self.tconv2(d1)

        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)

        d2 = torch.cat([d2, x2], dim=1)

        d2 = self.dec_block2(d2)


        d3 = self.tconv3(d2)

        d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=False)

        d3 = torch.cat([d3, x1], dim=1) # Skip connection from encoder
      
        d3 = self.dec_block3(d3)

        out = self.final_conv(d3)

        out = self.adaptive_pool(out)

        out = out.view(out.size(0), -1) 
        out = self.output_head(out)

        return out


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

def generate_stacked_exposure_plot(trade_actions, tickers, roll_num, plot_filename):
    """
    Generates a plot visualizing each triggered trade as a position with
    take-profit (green) and stop-loss (red) zones using a fast, vectorized approach.
    """
    print(f"Generating exposure plot: {plot_filename}...")
    num_tickers = len(tickers)
    fig, axes = plt.subplots(num_tickers, 1, figsize=(20, 8 * num_tickers), sharex=True)
    if num_tickers == 1: axes = [axes]

    for ax, ticker in zip(axes, tickers):
        # --- 1. Plot True Prices as Context (same as before) ---
        true_s_points = sorted(trade_actions[ticker]['true_s'], key=lambda x: x['time'])
        true_t_points = sorted(trade_actions[ticker]['true_t'], key=lambda x: x['time'])
        if true_s_points or true_t_points:
            all_true_points = sorted(true_s_points + true_t_points, key=lambda x: x['time'])
            if all_true_points:
                ax.plot([p['time'] for p in all_true_points], [p['price'] for p in all_true_points],
                        color='black', alpha=0.7, linewidth=1.5, label='True Price Path')

        # --- 2. Prepare Data for Vectorized Plotting ---
        profit_rects = {'y': [], 'width': [], 'left': [], 'height': [], 'color': []}
        loss_rects = {'y': [], 'width': [], 'left': [], 'height': [], 'color': []}
        
        num_actions = len(trade_actions[ticker]['actions'])
        print(f"Ticker {ticker}: Preparing {num_actions} actions for plotting...")

        for action in trade_actions[ticker]['actions']:
            s_point = action['s_point']
            t_point = action['t_point']
            
            entry_price = s_point['price']
            exit_price = t_point['price']
            duration = t_point['time'] - s_point['time']
            
            profit_target = abs(exit_price - entry_price)
            stop_loss_distance = profit_target

            if action['type'] == 'bull':
                # Profit rectangle (green)
                profit_rects['y'].append(entry_price)
                profit_rects['width'].append(duration)
                profit_rects['left'].append(s_point['time'])
                profit_rects['height'].append(profit_target)
                profit_rects['color'].append('green')
                # Loss rectangle (red)
                loss_rects['y'].append(entry_price - stop_loss_distance)
                loss_rects['width'].append(duration)
                loss_rects['left'].append(s_point['time'])
                loss_rects['height'].append(stop_loss_distance)
                loss_rects['color'].append('red')

            elif action['type'] == 'bear':
                # Profit rectangle (green)
                profit_rects['y'].append(exit_price)
                profit_rects['width'].append(duration)
                profit_rects['left'].append(s_point['time'])
                profit_rects['height'].append(profit_target)
                profit_rects['color'].append('green')
                # Loss rectangle (red)
                loss_rects['y'].append(entry_price)
                loss_rects['width'].append(duration)
                loss_rects['left'].append(s_point['time'])
                loss_rects['height'].append(stop_loss_distance)
                loss_rects['color'].append('red')

        # --- 3. Draw All Rectangles in Two Optimized Calls ---
        if profit_rects['y']:
            ax.barh(y=profit_rects['y'], width=profit_rects['width'], left=profit_rects['left'], 
                    height=profit_rects['height'], color=profit_rects['color'], alpha=0.2, 
                    align='edge', edgecolor='none')
        
        if loss_rects['y']:
            ax.barh(y=loss_rects['y'], width=loss_rects['width'], left=loss_rects['left'], 
                    height=loss_rects['height'], color=loss_rects['color'], alpha=0.2, 
                    align='edge', edgecolor='none')

        # --- 4. Finalize Plot (same as before) ---
        ax.set_title(f'Ticker: {ticker} - Triggered Positions ({num_actions} actions)', fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    axes[-1].set_xlabel('Event Time', fontsize=14)
    fig.suptitle(f'Trade Exposure Simulation (Roll {roll_num+1})', fontsize=20, y=0.99)
    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_format)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    plt.savefig(plot_filename)
    print(f"Saved exposure plot: {plot_filename}")
    plt.close()

def generate_stacked_plot(plot_data, tickers, roll_num, plot_title):
    """
    Generates and saves a single stacked plot with subplots for each ticker.
    """
    print("Generating stacked plot...")
    num_tickers = len(tickers)
    # Create a figure with subplots; one row for each ticker
    fig, axes = plt.subplots(num_tickers, 1, figsize=(20, 8 * num_tickers), sharex=True)
    # Ensure axes is always a list for consistent iteration, even with one ticker
    if num_tickers == 1: axes = [axes]

    for ax, ticker in zip(axes, tickers):
        # Plot True Prices as a solid black line for context
        true_s_points = sorted(plot_data[ticker]['true_s'], key=lambda x: x['time'])
        true_t_points = sorted(plot_data[ticker]['true_t'], key=lambda x: x['time'])
        if true_s_points and true_t_points:
            # Combine and sort all true points to form a continuous path
            all_true_points = sorted(true_s_points + true_t_points, key=lambda x: x['time'])
            ax.plot([p['time'] for p in all_true_points], [p['price'] for p in all_true_points],
                    color='black', alpha=0.7, linewidth=1.5, label='True Price Path')

        # Plot Predicted 's' price points
        s_points = plot_data[ticker]['s']
        if s_points:
            ax.scatter([p['time'] for p in s_points], [p['price'] for p in s_points],
                       c='red',
                       s=2, alpha=0.6,
                       edgecolors='w', linewidth=0.5, label='Predicted Start Price ("s")')

        # Plot Predicted 't' price points
        t_points = plot_data[ticker]['t']
        if t_points:
            ax.scatter([p['time'] for p in t_points], [p['price'] for p in t_points],
                       c='green',
                       s=2, alpha=0.6,
                       edgecolors='w', linewidth=0.5, label='Predicted Terminal Price ("t")')

 


        ax.set_title(f'Ticker: {ticker}', fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    # Format the shared X-axis
    axes[-1].set_xlabel('Event Time', fontsize=14)
    fig.suptitle(f'Test Set Predictions (Roll {roll_num+1})', fontsize=20, y=0.99)

    # Use a date formatter for the x-axis to make it readable
    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_format)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

    plot_filename = plot_title
    plt.savefig(plot_filename)
    print(f"Saved stacked plot: {plot_filename}")
    plt.close()

class ViTMultiHead(nn.Module):
    def __init__(self, img_shape=(27, 8), in_channels=1, patch_size=(9, 4), 
                 embed_dim=128, num_layers=4, nhead=4, dropout_rate=0.1):
        super().__init__()
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_h, patch_w = patch_size
        img_h, img_w = img_shape
        

        num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = in_channels * patch_h * patch_w

        self.patch_embedding = nn.Linear(patch_dim, embed_dim)


        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout_rate,
            batch_first=True, 
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_ab = nn.Linear(embed_dim, 1)
        self.head_st = nn.Linear(embed_dim, 1)
        self.head_tb = nn.Linear(embed_dim, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        patch_h, patch_w = self.patch_size

     
        x = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        x = x.contiguous().view(N, -1, C * patch_h * patch_w)


        x = self.patch_embedding(x)


        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_encoding

    
        x = self.transformer_encoder(x)

        cls_output = x[:, 0]

        ab_output = self.head_ab(cls_output)
        st_output = self.head_st(cls_output)
        tb_output = self.head_tb(cls_output)

        return ab_output, st_output, tb_output
    


class CustomCollate:
    def __init__(self, train_ds, input_keys_sorted, output_keys_sorted, norm, forecast_depth_seconds):
        self.train_ds = train_ds
        self.input_keys_sorted = input_keys_sorted
        self.output_keys_sorted = output_keys_sorted
        self.norm = norm
        self.forecast_depth_seconds = forecast_depth_seconds

    def __call__(self, batch):
        items = [self.train_ds.__getitem__(
            index,
            self.input_keys_sorted,
            self.output_keys_sorted,
            self.norm,
            self.forecast_depth_seconds
        ) for index in batch]
        items = [item for item in items if item is not None and item[0] is not None]
        if not items:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])


        # Stack the tensors.
        time_entry_ts_list, X_tensor_list, Y_tensor_list, current_prices_list = zip(*items)
        
        X_tensor_batch = torch.stack(X_tensor_list)
        Y_tensor_batch = torch.stack(Y_tensor_list)


        return time_entry_ts_list, X_tensor_batch, Y_tensor_batch, current_prices_list
    