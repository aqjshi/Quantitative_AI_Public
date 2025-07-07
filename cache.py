from datetime import datetime, timedelta, time as dt_time
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple, Union
from datetime import timezone
import numpy as np
import pandas as pd
from models import Conv2DMultiBinary

from indicators import ( LOCAL_FUNCS, AGAUSSIAN, PERCENTILE,
    HUGE, CONSTANT, parse_param_to_inputs, load_indicator_input, load_indicator_output
)

from av_client import av_client, process_REALTIME_BULK_QUOTES

import torch
from urllib.parse import urlencode, parse_qs
import copy


import matplotlib.pyplot as plt
import seaborn as sns
import pytz

est_timezone = pytz.timezone('America/New_York')

class RealTimeInferenceEngine:
    def __init__(self, tickers, accuracy_model_path, profit_model_path, accuracy_basefilters, profit_basefilters, context_depth, params, hot_burn_in=250, device='cuda'):
        # --- This __init__ method remains the same ---
        self.tickers = tickers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hot_burn_in = hot_burn_in
        print("Loading dual models...")
        num_outputs = len(self.tickers)
        self.profit_model = Conv2DMultiBinary(in_channels=1, base_filters=profit_basefilters, num_outputs=num_outputs).to(self.device)
        self.profit_model.load_state_dict(torch.load(profit_model_path, map_location=self.device))
        self.profit_model.eval()
        print(f"✅ Profit Model loaded from {profit_model_path}")
        self.accuracy_model = Conv2DMultiBinary(in_channels=1, base_filters=accuracy_basefilters, num_outputs=num_outputs).to(self.device)
        self.accuracy_model.load_state_dict(torch.load(accuracy_model_path, map_location=self.device))
        self.accuracy_model.eval()
        print(f"✅ Accuracy Model loaded from {accuracy_model_path}")
        self.ohlcv_funcs = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        self.context_depth = context_depth
        self.params = sorted(params)
        self.hot_raw_ohlcv_data_buffer = {}
        for ticker in self.tickers:
            self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=TIMESTAMP"] = deque(maxlen=hot_burn_in)
            for func in self.ohlcv_funcs:
                self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function={func}"] = deque(maxlen=hot_burn_in)
        self.param_to_idx = {param: i for i, param in enumerate(self.params)}
        print(f"Engine initialized for {len(self.tickers)} tickers.")

    def cold_start(self):
        print("--- Starting Cold Start via API ---")
        av = av_client()
        raw_ohlcv_data = {}
        current_date = datetime.now()
        current_month_yyyy_mm = current_date.strftime("%Y-%m")


        for ticker in self.tickers:
            print(f"Fetching historical data for {ticker}...")
            try:
                series_data = av.fetch_intraday(
                    symbol=ticker, 
                    month=current_month_yyyy_mm,
                    interval="1min",
                    outputsize="full", # Ensure you get enough historical data
                    extended_hours=True,
                    entitlement="realtime"
                )

                if not series_data:
                    print(f"⚠️ Warning: No data returned from API for {ticker}.")
                    continue
                
                # 1. Create DataFrame directly from the fetched dictionary
                df = pd.DataFrame.from_dict(series_data, orient='index')
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S') 
                
                # Localize the naive index to 'America/New_York' (EST/EDT)
                df.index = df.index.tz_localize(est_timezone, ambiguous='infer') # 'infer' handles DST
                
                # 2. Rename columns
                df = df.rename(columns={
                    '1. open': 'OPEN', '2. high': 'HIGH', 
                    '3. low': 'LOW', '4. close': 'CLOSE', '5. volume': 'VOLUME'
                })
                
                # 3. Convert index to datetime and sort (Alpha Vantage gives newest first)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index() # Sort to have oldest first (ascending time)

                # 4. Convert OHLCV data to numeric types
                # Use .apply(pd.to_numeric) for robustness, handling errors
                for col in self.ohlcv_funcs:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' will turn non-numeric to NaN
                
                # Set volume to integer, handling NaNs from conversion if any
                df['VOLUME'] = df['VOLUME'].fillna(0).astype(float) # Fill NaN volumes with 0, then convert to int

                if not df.empty:
                    latest_data_time = df.index[-1] # Most recent time in the fetched data
                    print("latest_data_time ", latest_data_time)
                    
                    current_utc_minute_rounded = datetime.now(df.index.tz).replace(second=0, microsecond=0)
                    start_time_for_range = latest_data_time - timedelta(minutes=self.hot_burn_in - 1)
                    
                    full_time_index = pd.date_range(
                        start=start_time_for_range,
                        end=latest_data_time,
                        freq='1min',
                        name='timestamp', # Name the index for clarity
                        tz=df.index.tz # Ensure timezone consistency
                    )
                    print(f"length of linspace", len(full_time_index))
                    
                    reindexed_df = df.reindex(full_time_index)

                    
                    reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].ffill()
                    
                    # Fill VOLUME with 0 for missing entries
                    reindexed_df['VOLUME'] = reindexed_df['VOLUME'].fillna(0).astype(float)
                    



                    # Handle any remaining NaNs (e.g., if the very first values were NaN and couldn't be ffilled)
                    # One common approach is to backfill remaining NaNs, or drop rows if you prefer
                    reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].bfill()
                    reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = reindexed_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].fillna(0).astype(float)
       
       
                    timestamp_list = reindexed_df.index[:self.hot_burn_in].tolist()
                    self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=TIMESTAMP"] = deque(timestamp_list, maxlen=self.hot_burn_in)

                    for func_name in self.ohlcv_funcs:
                        data_list = reindexed_df[func_name][:self.hot_burn_in].tolist()
                        self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function={func_name}"] = deque(data_list, maxlen=self.hot_burn_in)
                
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=TIMESTAMP"])
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=OPEN"])
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=HIGH"])
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=LOW"])
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=CLOSE"])
                    # print(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=VOLUME"])
                else:
                    print(f"⚠️ Warning: DataFrame for {ticker} is empty after initial fetch. Cannot reindex.")

            except Exception as e:
                print(f"❌ Failed to process data for {ticker}. Error: {e}")
        # print(self.hot_raw_ohlcv_data_buffer.keys())

        
        # print("Calculating mean for each series:")
        # for key in self.hot_raw_ohlcv_data_buffer.keys():
        #     if re.search(r"TIMESTAMP", key):
        #         continue
        #     else:
        #         data_series = self.hot_raw_ohlcv_data_buffer[key]
        #         mean_value = statistics.mean(data_series)
        #         print(f"  {key}: Mean = {mean_value:.4f}")

        print("✅ Cold start complete. Tensors and buffers are ready.")


    # def _calculate_param_slice(self, param_str: str, local_hot_raw_ohlcv_data_buffer: Dict[str, deque]) -> pd.Series:
    #     """
    #     Reconstructs an OHLCV DataFrame for a ticker and calculates one indicator series.
    #     """
    #     # 1. Extract ticker from the param string
    #     ticker = param_str.split("ticker=")[1].split("&")[0]
    #     input_dict, key = parse_param_to_inputs(param_str)
    #     # --- DIAGNOSIS PRINTING STARTS HERE ---
    #     # print(f"\n--- Diagnosing _calculate_param_slice for {param_str} (Ticker: {ticker}) ---")
    #     # print(f"Keys in local_hot_raw_ohlcv_data_buffer for {ticker}:")
        
    #     # # Print all relevant keys for the current ticker and their deque lengths
    #     # re_pattern = f"ticker={ticker}" # This correctly forms the pattern string, e.g., "ticker=NVDA"
    #     # all_ticker_keys = [k for k in local_hot_raw_ohlcv_data_buffer.keys() if re.search(re_pattern, k)]

    #     # for k in all_ticker_keys:
    #     #     deque_len = len(local_hot_raw_ohlcv_data_buffer[k])
    #     #     print(f"  {k}: Length = {deque_len}, MaxLen = {local_hot_raw_ohlcv_data_buffer[k].maxlen}")
    #     #     # Print actual data for OHLCV to see NaNs or zeros
    #     #     if 'function=TIMESTAMP' in k or 'function=OPEN' in k or 'function=HIGH' in k or 'function=LOW' in k or 'function=CLOSE' in k or 'function=VOLUME' in k:
    #     #         print(f"    Data (first 5, last 5): {list(local_hot_raw_ohlcv_data_buffer[k])[:5]} ... {list(local_hot_raw_ohlcv_data_buffer[k])[-5:]}")
    #     #         if any(pd.isna(x) for x in local_hot_raw_ohlcv_data_buffer[k]):
    #     #             print(f"    *** WARNING: {k} contains NaN values! ***")
    #     #         if not local_hot_raw_ohlcv_data_buffer[k]:
    #     #             print(f"    *** WARNING: {k} is empty! ***")

        
    #     # --- DIAGNOSIS PRINTING ENDS HERE ---


    #     timestamp_key = f"ticker={ticker}&function=TIMESTAMP"
    #     timestamps_list = list(local_hot_raw_ohlcv_data_buffer.get(timestamp_key, deque()))
        
    #     # Check if timestamps are available and valid
    #     if not timestamps_list:
    #         print(f"Warning: No TIMESTAMP data found for {ticker} in buffer. Cannot create DataFrame.")
    #         return pd.DataFrame(dtype=np.float64) # Return an empty DataFrame




    #     data_for_df_columns = {}
    #     for func in self.ohlcv_funcs + ["TIMESTAMP"]: # e.g., "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
    #         buffer_key = f"ticker={ticker}&function={func}"
    #         data_list = list(local_hot_raw_ohlcv_data_buffer.get(buffer_key, deque()))
    #         data_for_df_columns[func] = data_list 



    #     all_lengths = [len(timestamps_list)] + [len(lst) for lst in data_for_df_columns.values()]
    #     if not all_lengths or min(all_lengths) == 0:
    #         print(f"WARNING: Data for {ticker} is empty or has inconsistent lengths. Lengths: {all_lengths}")
    #         return pd.DataFrame(dtype=np.float64)
        
        
    #     min_len = min(all_lengths)

    #     timestamps_list = timestamps_list[-min_len:]
    #     for col_name in data_for_df_columns:
    #         data_for_df_columns[col_name] = data_for_df_columns[col_name][-min_len:]

        
    #     # Convert timestamps to datetime objects immediately
    #     df = pd.DataFrame(data_for_df_columns)
    #     print(df.shape)
    #     df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    #     df = df.set_index('TIMESTAMP')#.sort_index() # Set timestamp as index and sort
    #     print(df.index)                
        

    #     fn = LOCAL_FUNCS[key[0]]
        
    #     # Pass the DataFrame and parsed inputs to the indicator function
    #     result_series = fn(df, input_dict, key)
    #     # print(result_series.mean())
    #     return     result_series
    #     #    



    def _calculate_param_slice(self, param_str: str, local_hot_raw_ohlcv_data_buffer: Dict[str, deque]) -> pd.Series:
        """
        Reconstructs an OHLCV DataFrame for a ticker and calculates one indicator series,
        using an implicit 1-minute spacing.
        """
        # 1. Extract ticker from the param string
        ticker = param_str.split("ticker=")[1].split("&")[0]
        input_dict, key = parse_param_to_inputs(param_str)

        data_for_df_columns = {}
        ohlcv_deque_lengths = []
        for func in self.ohlcv_funcs:
            buffer_key = f"ticker={ticker}&function={func}"
            data_list = list(local_hot_raw_ohlcv_data_buffer.get(buffer_key, deque()))
            data_for_df_columns[func] = data_list
            ohlcv_deque_lengths.append(len(data_list))

        # Determine the minimum length among all OHLCV deques
        if not ohlcv_deque_lengths or min(ohlcv_deque_lengths) == 0:
            print(f"WARNING: OHLCV data for {ticker} is empty or has inconsistent lengths. Lengths: {ohlcv_deque_lengths}")
            return pd.Series(dtype=np.float64) # Return an empty Series

        min_len = min(ohlcv_deque_lengths)

        # Slice all OHLCV data to the minimum length, taking the most recent items
        for col_name in self.ohlcv_funcs:
            data_for_df_columns[col_name] = data_for_df_columns[col_name][-min_len:]

        # Create DataFrame without explicitly using TIMESTAMP for index
        # pandas will create a default integer index (0, 1, 2, ...)
        df = pd.DataFrame(data_for_df_columns)
        
        # Check for NaNs immediately after DataFrame creation
        if df.isnull().values.any():
            # print(f"WARNING: DataFrame for {ticker} contains NaNs before indicator calculation:\n{df.isnull().sum()}")
            df = df.ffill().bfill()
            if df.isnull().values.any(): # If still NaNs after fillna (e.g., all NaNs)
                 print(f"CRITICAL WARNING: DataFrame for {ticker} still contains NaNs after fillna. Filling remaining with 0.")
                 df = df.fillna(0) # As a last resort, fill with 0

        # print(f"DataFrame for {ticker} shape: {df.shape}")
        # print(f"DataFrame for {ticker} head:\n{df.head()}")
        # print(f"DataFrame for {ticker} tail:\n{df.tail()}")

        fn = LOCAL_FUNCS[key[0]]
        
        # Pass the DataFrame and parsed inputs to the indicator function
        # The indicator function will operate on the sequential data,
        # implicitly treating each row as a new 1-minute interval.
        result_series = fn(df, input_dict, key)

        # Ensure the result series is also handled for potential NaNs from indicator calc
        if result_series.isnull().values.any():
            # print(f"WARNING: Indicator '{param_str}' returned NaNs. Filling with previous non-NaN value or 0.")
            result_series = result_series.ffill().bfill() # Fill NaNs for safety

        return result_series
    def _update_buffer_with_new_bar(self, ticker: str, quote: Dict[str, Any]):
        """
        PRIVATE HELPER: Rolls the buffer forward by one minute. Pops the oldest bar
        and appends a new, complete one. Called only when update=True.
        """
        print(f"--- New Bar for {ticker} ---")
        buffer = self.hot_raw_ohlcv_data_buffer
        
        # 1. Pop the oldest data point from the left
        buffer[f"ticker={ticker}&function=TIMESTAMP"].popleft()
        for func_name in self.ohlcv_funcs:
            buffer[f"ticker={ticker}&function={func_name}"].popleft()

        # 2. Append the new data to the right
        buffer[f"ticker={ticker}&function=TIMESTAMP"].append(datetime.now(timezone.utc).replace(second=0, microsecond=0))
        for func_name, av_key in [('OPEN', 'open'), ('HIGH', 'high'), ('LOW', 'low'), ('CLOSE', 'close'), ('VOLUME', 'volume')]:
            buffer[f"ticker={ticker}&function={func_name}"].append(float(quote[av_key]))

    def _update_buffer_intra_bar(self, ticker: str, quote: Dict[str, Any]):
        """
        PRIVATE HELPER: Updates the most recent bar in-place with a new tick.
        Called only when update=False.
        """
        print(f"--- Intra-Bar Update for {ticker} ---")
        buffer = self.hot_raw_ohlcv_data_buffer
        
        if len(buffer[f"ticker={ticker}&function=TIMESTAMP"]) == 0:
            print(f"Warning: Buffer for {ticker} is empty, cannot perform intra-bar update.")
            return

        # Update HIGH, LOW, CLOSE, and VOLUME of the last element [-1]
        buffer[f"ticker={ticker}&function=HIGH"][-1] = max(buffer[f"ticker={ticker}&function=HIGH"][-1], float(quote['high']))
        buffer[f"ticker={ticker}&function=LOW"][-1] = min(buffer[f"ticker={ticker}&function=LOW"][-1], float(quote['low']))
        buffer[f"ticker={ticker}&function=CLOSE"][-1] = float(quote['close'])
        buffer[f"ticker={ticker}&function=VOLUME"][-1] += float(quote['volume'])
        # OPEN and TIMESTAMP are intentionally not touched.

    def _prepare_inference_tensor(self) -> np.ndarray:
        """
        PRIVATE HELPER: Calculates all indicators and assembles the final
        feature tensor for the models.
        """
        all_new_indicator_values = {}

        # --- Loop through each parameter to calculate and normalize it individually ---
        for param_str in self.params:
            # 1. Calculate the indicator series from the buffer
            result_series = self._calculate_param_slice(param_str, self.hot_raw_ohlcv_data_buffer)
            unnormalized_series = result_series.tail(self.context_depth)
            
            # 2. Get the last value for normalization
            # Use a try-except block to gracefully handle cases where the series might be empty
            try:
                last_value = unnormalized_series.iloc[-1]
            except IndexError:
                last_value = np.nan # Default to NaN if the series is empty

            # 3. Perform normalization
            # if pd.isna(last_value):
            #     # If the last value is NaN, the whole normalized series should be NaN
            #     normalized_series = pd.Series(np.nan, index=unnormalized_series.index)
            # else:
            fn_name = param_str.split("function=")[1].split("&")[0] if "function=" in param_str else param_str
            if fn_name in AGAUSSIAN or fn_name in CONSTANT:
                normalized_series = unnormalized_series - last_value
            elif fn_name in PERCENTILE:
                normalized_series = (unnormalized_series - last_value) / 100.0
            else:  # HUGE functions
                normalized_series = (unnormalized_series - last_value) / 10000000.0
            



            
            all_new_indicator_values[param_str] = normalized_series

        # --- Assemble Final Tensor from the calculated series ---
        ordered_series_list = []
        for param_str in self.params:
            series = all_new_indicator_values.get(param_str)
            
            # Pad with NaNs if the series is shorter than the required context depth
            if series is None or len(series) < self.context_depth:
                padded_series = pd.Series([np.nan] * self.context_depth)
                if series is not None:
                    padded_series.iloc[-len(series):] = series.values
                series = padded_series
                
            ordered_series_list.append(series)
            
        # Concatenate all series into a single DataFrame and then convert to a NumPy array
        inference_df = pd.concat(ordered_series_list, axis=1)
        inference_df.columns = self.params
        
        # Final check to fill any remaining NaNs before creating the tensor
        if inference_df.isnull().values.any():
            inference_df.ffill(inplace=True)
            inference_df.bfill(inplace=True)
            inference_df.fillna(0, inplace=True) # Fill any remaining NaNs with 0

        return inference_df.values.astype(np.float32)
    def update_hot_tensor_and_return_inference_item(self, processing_tickers: List[str], new_quote_data: Dict[str, Dict], update: bool) -> np.ndarray:
        """
        PUBLIC METHOD: Main entry point to update the real-time buffer and get a new inference item.
        """
        # Step 1: Update the buffer for each ticker based on the 'update' flag
        for ticker_symbol in processing_tickers:
            if ticker_symbol in new_quote_data:
                data_for_ticker = new_quote_data[ticker_symbol]
                if update:
                    self._update_buffer_with_new_bar(ticker_symbol, data_for_ticker)
                else:
                    self._update_buffer_intra_bar(ticker_symbol, data_for_ticker)
        
        # Step 2: After all buffers are updated, prepare the feature tensor
        inference_item = self._prepare_inference_tensor()
        
        return inference_item


    def infer(self, inference_item: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference through both models and returns their logits.
        """
        with torch.no_grad():
            
            # --- THE FIX IS HERE ---
            # 1. Convert the incoming NumPy array to a PyTorch tensor FIRST.
            x_tensor = torch.from_numpy(inference_item)
            
            # 2. Now you can use PyTorch methods like .unsqueeze() and .to() on the new tensor.
            x = x_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get logits from both models
            profit_logits = self.profit_model(x).squeeze()
            accuracy_logits = self.accuracy_model(x).squeeze()
            
            return profit_logits, accuracy_logits
        
    def visualize_inference_tensor(self, inference_item: np.ndarray, file_name: str):
        """
        Generates and saves a discrete heatmap of the feature tensor.

        Args:
            inference_item (np.ndarray): The (time_steps, num_features) tensor to plot.
            file_name (str): The name of the image file to save (e.g., 'cold_start_heatmap.png').
        """
        print(f"\nGenerating heatmap of the feature tensor, shape: {inference_item.shape}...")
        
        # We transpose the matrix so features are on the Y-axis and time is on the X-axis
        data_to_plot = inference_item.T  # Shape becomes (num_features, time_steps)

        # Set a dynamic figure size. Height depends on the number of features.
        # Adjust the numbers as needed for your screen/preference.
        height = max(10, len(self.params) / 4) # at least 10 inches tall
        width = 15
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Use seaborn to create the heatmap.
        # 'coolwarm' is great for data centered around zero, like your normalized features.
        sns.heatmap(data_to_plot, 
                    ax=ax,
                    yticklabels=self.params, # Use feature names as Y-axis labels
                    xticklabels=False,      # Hide the X-axis labels (too many time steps)
                    cmap='coolwarm',        # Color scheme
                    cbar=True)              # Show the color bar legend

        ax.set_title('Feature Tensor Heatmap at Cold Start', fontsize=16)
        ax.set_xlabel(f'Time Steps ({self.context_depth} mins)')
        ax.set_ylabel('Features')
        
        # Ensure labels aren't cut off
        plt.tight_layout()
        
        try:
            plt.savefig(file_name, dpi=150) # Save the plot to a file
            print(f"✅ Heatmap saved successfully to '{file_name}'")
        except Exception as e:
            print(f"❌ Failed to save heatmap. Error: {e}")
        
        plt.close() # Close the plot to free up memory
    def visualize_ohlcv_buffers(self, file_name: str = 'ohlcv_buffers_visualization.png'):
        """
        Generates and saves line plots of the raw OHLCV data stored in the
        engine's buffers after cold start.
        """
        print(f"\nGenerating visualization of the OHLCV data buffers...")

        num_tickers = len(self.tickers)
        if num_tickers == 0:
            print("No tickers to visualize.")
            return

        # Create a tall figure with one subplot for each ticker
        fig, axes = plt.subplots(
            nrows=num_tickers,
            ncols=1,
            figsize=(15, 6 * num_tickers), # 6 inches of height per ticker
            squeeze=False # Ensures axes is always a 2D array
        )
        axes = axes.flatten() # Flatten to a 1D array for easy iteration

        for i, ticker in enumerate(self.tickers):
            ax = axes[i]
            
            # --- Extract Data for this Ticker ---
            # Convert deques to lists for plotting
            try:
                close_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=CLOSE"])
                open_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=OPEN"])
                high_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=HIGH"])
                low_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=LOW"])
                volume_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=VOLUME"])
            except KeyError as e:
                print(f"Data not found for {ticker} in buffer, skipping plot. Error: {e}")
                continue

            if not close_data:
                ax.set_title(f'{ticker} OHLCV Buffer Data - NO DATA FOUND', color='red')
                continue
                
            # --- Plot Price Data (OHLC) ---
            ax.plot(close_data, label='Close', color='blue', linewidth=2)
            ax.plot(open_data, label='Open', color='black', linestyle='--', alpha=0.7)
            ax.plot(high_data, label='High', color='green', linestyle=':', alpha=0.6)
            ax.plot(low_data, label='Low', color='red', linestyle=':', alpha=0.6)
            
            ax.set_title(f'{ticker} OHLCV Buffer Data', fontsize=16)
            ax.set_ylabel('Price ($)', color='blue')
            ax.grid(True, linestyle='--', alpha=0.6)

            # --- Plot Volume Data on a Second Y-Axis ---
            ax2 = ax.twinx() # Create a second y-axis that shares the same x-axis
            ax2.bar(range(len(volume_data)), volume_data, label='Volume', color='grey', alpha=0.2)
            ax2.set_ylabel('Volume', color='grey')
            
            # --- Create a single, combined legend for both axes ---
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

        # Adjust layout to prevent titles and labels from overlapping
        plt.tight_layout(pad=3.0)
        
        try:
            plt.savefig(file_name, dpi=150)
            print(f"✅ OHLCV buffer visualization saved successfully to '{file_name}'")
        except Exception as e:
            print(f"❌ Failed to save OHLCV visualization. Error: {e}")
            
        plt.close() # Close the plot to free up memory
        