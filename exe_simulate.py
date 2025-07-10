import time
import threading
from datetime import datetime, timedelta
from av_client import av_client
from cache import process_REALTIME_BULK_QUOTES,RealTimeInferenceEngine
import sys


import time
import pytz
import json
import time
from datetime import datetime, timedelta, time as dt_time, timezone
import os 
import csv

def is_market_open() -> bool:
    """Checks if the current time is within regular NYSE trading hours."""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
    return market_open <= now.time() <= market_close

class PaperTrader:
    """
    Manages a paper trading portfolio, simulating trades and logging them to a CSV file.
    """
    def __init__(self, output_dir: str, forecast_depth_minutes: int):
        self.filepath = os.path.join(output_dir, "paper_portfolio_transactions.csv")
        self.fieldnames = [
            "transaction_id", "time_UTC", "ticker", "current_price", 
            "transaction_type", "quantity", "order_type", "parent_id", 
            "child_tp_price", "child_tp_id", "child_sl_price", "child_sl_id", 
            "child_maturity_id", "child_maturity_UTC"
        ]
        self.next_transaction_id = 1
        self.open_positions = {}  # Key: ticker, Value: position details
        self.forecast_depth = timedelta(minutes=forecast_depth_minutes)
        self.lock = threading.Lock()

        # Initialize CSV file with headers
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        print(f"📄 Paper portfolio initialized at: {self.filepath}")

    def _get_next_id(self):
        """Safely gets the next unique transaction ID."""
        with self.lock:
            tx_id = self.next_transaction_id
            self.next_transaction_id += 1
            return tx_id

    def _log_transaction(self, trade_data: dict):
        """Appends a single transaction record to the CSV file."""
        with self.lock, open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(trade_data)

    def place_paper_trade(self, ticker: str, quantity: int, entry_price: float, tp_float: float, sl_float: float):
        """Simulates placing a bracket order and logs the entry."""
        with self.lock:
            if ticker in self.open_positions:
                print(f"  --> ℹ️ Skipping trade for {ticker}: Position already open.")
                return

        now_utc = datetime.now(timezone.utc)
        parent_id = self._get_next_id()
        tp_id = self._get_next_id()
        sl_id = self._get_next_id()
        maturity_id = self._get_next_id()

        tp_price = entry_price * (1 + tp_float)
        sl_price = entry_price * (1 - sl_float)
        maturity_utc = now_utc + self.forecast_depth

        # Log the parent "BUY" transaction
        entry_log = {
            "transaction_id": parent_id, "time_UTC": now_utc.isoformat(), "ticker": ticker,
            "current_price": entry_price, "transaction_type": "buy", "quantity": quantity,
            "order_type": "entry", "parent_id": None, "child_tp_price": tp_price,
            "child_tp_id": tp_id, "child_sl_price": sl_price, "child_sl_id": sl_id,
            "child_maturity_id": maturity_id, "child_maturity_UTC": maturity_utc.isoformat()
        }
        self._log_transaction(entry_log)
        print(f"  --> 📝 Logged [BUY] for {quantity} {ticker} @ {entry_price:.2f} (TX_ID: {parent_id})")

        # Store the open position in memory
        self.open_positions[ticker] = {
            "parent_id": parent_id, "quantity": quantity, "entry_price": entry_price,
            "tp_price": tp_price, "sl_price": sl_price, "maturity_utc": maturity_utc,
            "tp_id": tp_id, "sl_id": sl_id, "maturity_id": maturity_id
        }

    def check_positions(self, current_prices: dict):
        """Checks all open positions against current prices for TP/SL/Maturity triggers."""
        now_utc = datetime.now(timezone.utc)
        with self.lock:
            # Iterate over a copy of keys to allow safe modification
            for ticker in list(self.open_positions.keys()):
                pos = self.open_positions[ticker]
                current_price = current_prices.get(ticker, {}).get("price")

                if current_price is None:
                    continue # Skip if no price data for this ticker

                triggered_event = None
                
                # Check for triggers
                if current_price >= pos['tp_price']:
                    triggered_event = ("tp", pos['tp_id'], pos['tp_price'])
                elif current_price <= pos['sl_price']:
                    triggered_event = ("sl", pos['sl_id'], pos['sl_price'])
                elif now_utc >= pos['maturity_utc']:
                    triggered_event = ("maturity", pos['maturity_id'], current_price)

                if triggered_event:
                    order_type, tx_id, exit_price = triggered_event
                    
                    # Log the closing "SELL" transaction
                    exit_log = {
                        "transaction_id": tx_id, "time_UTC": now_utc.isoformat(), "ticker": ticker,
                        "current_price": exit_price, "transaction_type": "sell", "quantity": pos['quantity'],
                        "order_type": order_type, "parent_id": pos['parent_id'], "child_tp_price": None,
                        "child_tp_id": None, "child_sl_price": None, "child_sl_id": None,
                        "child_maturity_id": None, "child_maturity_UTC": None
                    }
                    self._log_transaction(exit_log)
                    print(f"  --> 📝 Logged [SELL] for {pos['quantity']} {ticker} @ {exit_price:.2f} (Reason: {order_type.upper()}, TX_ID: {tx_id})")

                    # Remove from open positions
                    del self.open_positions[ticker]



def main():
    config_filepath = sys.argv[1]
    DEBUG = sys.argv[2]
    with open(config_filepath, 'r', encoding='utf-8') as f:
        known = json.load(f)
        tickers = known["ticker"]
        params = known["params"]
        accuracy_model_path = known["accuracy_model_path"]
        profit_model_path = known["profit_model_path"]
        output_dir = known["output_dir"]


        ACCURACY_base_filters  = known["ACCURACY_base_filters"]


    
        PROFIT_base_filters  = known["PROFIT_base_filters"]

        forecast_depth = known["forecast_depth"]
        context_depth = known["context_depth"]
        order_quantity = known["order_quantity"]
        take_profit = known["take_profit"]
        stop_loss = known["stop_loss"]
        maximum_exposure = known["maximum_exposure"]
        frequency_limiter_seconds = known["frequency_limiter_seconds"]

    tickers = sorted(tickers)
    params = sorted(params)

    # --- Engine and State Initialization ---
    inference_engine = RealTimeInferenceEngine(
        tickers, accuracy_model_path, profit_model_path,
        ACCURACY_base_filters, PROFIT_base_filters,
        context_depth, params
    )
    inference_engine.cold_start()
    
    # === NEW: Initialize the PaperTrader ===
    paper_trader = PaperTrader(output_dir, forecast_depth)
    
    av = av_client()
    last_permanent_update_minute = -1
    
    trade_cooldown_expiry = {ticker: datetime.now(timezone.utc) for ticker in tickers}
    print(f"Trade frequency limiter initialized to {frequency_limiter_seconds} seconds per ticker.")

    # --- Main Real-Time Loop ---
    print("\n--- Starting High-Frequency Inference & Paper Trading Loop ---")
    try:
        while is_market_open() or DEBUG == "TRUE":
            now_utc = datetime.now(timezone.utc)
            
            # --- Fetch Data and Check Positions ---
            trading_data = process_REALTIME_BULK_QUOTES(av, tickers, extended_hours=True)
            if not trading_data:
                print("No data fetched, skipping cycle.")
                time.sleep(1)
                continue
            
            # Check for TP/SL/Maturity triggers before making new trades
            paper_trader.check_positions(trading_data)

            # --- State Update ---
            update_flag = (now_utc.second >= 55 and now_utc.minute != last_permanent_update_minute)
            if update_flag: last_permanent_update_minute = now_utc.minute

            # --- Perform Inference ---
            inference_item = inference_engine.update_hot_tensor_and_return_inference_item(tickers, trading_data, update=update_flag)
            inference_engine.visualize_inference_tensor(inference_item, "ouput_inference_tensor.png")
            inference_engine.visualize_ohlcv_buffers( "ouput_ohlcv_buffer.png")
            profit_logits, accuracy_logits = inference_engine.infer(inference_item)

            # --- Paper Trading Logic ---
            print(f"\n--- Cycle at {now_utc.strftime('%H:%M:%S')} UTC ---")
            for i, ticker in enumerate(inference_engine.tickers):
                profit_signal = profit_logits[i].item()
                accuracy_signal = accuracy_logits[i].item()
                
                print(f"  Signals for {ticker}: Profit={profit_signal:.4f}, Accuracy={accuracy_signal:.4f}")
                
                if now_utc < trade_cooldown_expiry[ticker]:
                    continue

                if profit_signal > 0 and accuracy_signal > 0:
                    entry_price = round((trading_data[ticker]["close"]+.05), 2)
                    print(f"  >>> Agreement found! Simulating trade for {ticker} at entry price {entry_price}")
                    
                    paper_trader.place_paper_trade(
                        ticker=ticker,
                        quantity=order_quantity,
                        entry_price=entry_price,
                        tp_float=take_profit,
                        sl_float=stop_loss
                    )
                    
                    expiry_time = now_utc + timedelta(seconds=frequency_limiter_seconds)
                    trade_cooldown_expiry[ticker] = expiry_time
                    print(f"  --> ⏳ {ticker} is now in cooldown until {expiry_time.strftime('%H:%M:%S')} UTC.")
            
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("\n--- Shutdown Sequence Complete ---")
        print(f"Final paper portfolio saved at: {paper_trader.filepath}")

if __name__ == "__main__":
    main()


