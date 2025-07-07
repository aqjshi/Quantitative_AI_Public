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

class Simulator:
    def __init__(self, max_exposure_limit: float = 2000.00):
        self.nextint = None
        self.stale_order_tracker = {}
        self.max_exposure = max_exposure_limit
        self.current_exposure = 0.0
        self.open_positions = {}
        self.pending_orders = {}
        self.order_placement_lock = threading.Lock()
        self.exposure_lock = threading.Lock()
        self.position_lock = threading.Lock()
        


    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCappedPrice: float):
        pass

    def execDetails(self, reqId: int):
        pass
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        pass


    def nextValidId(self, orderId: int):
        pass

    def place_bracket_order_with_maturity(self, ticker: str, take_profit_float: float, stop_loss_float: float, entry_price: float, quantity: int, maturity_minutes: int = 15):
        pass



    def liquidate_specific_position(self, ticker: str, quantity: int, tp_id: int, sl_id: int):
        pass


    def liquidate_all_positions(self):
        pass

    def manage_expired_positions(self, forecast_depth_minutes: int):
        """
        Checks for any open positions that have exceeded their forecast depth
        and liquidates them immediately.
        """
        pass

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
        ACCURACY_lr  = known["ACCURACY_lr"]
        ACCURACY_batch_size  = known["ACCURACY_batch_size"]
        ACCURACY_dropout_rate = known["ACCURACY_dropout_rate"]
        ACCURACY_l1_reg_strength = known["ACCURACY_l1_reg_strength"]
        ACCURACY_l2_reg_strength = known["ACCURACY_l2_reg_strength"]


    
        PROFIT_base_filters  = known["PROFIT_base_filters"]
        PROFIT_lr  = known["PROFIT_lr"]
        PROFIT_batch_size  = known["PROFIT_batch_size"]
        PROFIT_dropout_rate = known["PROFIT_dropout_rate"]
        PROFIT_l1_reg_strength = known["PROFIT_l1_reg_strength"]
        PROFIT_l2_reg_strength = known["PROFIT_l2_reg_strength"]

        context_depth = known["context_depth"]
        order_quantity = known["order_quantity"]
        maximum_exposure = known["maximum_exposure"]
        frequency_limiter_seconds = known["frequency_limiter_seconds"]

    tickers = sorted(tickers)
    params = sorted(params)


 # --- Engine and State Initialization ---
    inference_engine = RealTimeInferenceEngine(tickers, 
                                               accuracy_model_path, 
                                               profit_model_path, 
                                               ACCURACY_base_filters,
                                               PROFIT_base_filters,
                                            context_depth, params)
    inference_engine.cold_start()
    
    av = av_client()
    last_permanent_update_minute = -1
    
    trade_cooldown_expiry = {ticker: datetime.now(timezone.utc) for ticker in tickers}
    print(f"Trade frequency limiter initialized to {frequency_limiter_seconds} seconds per ticker.")

    # --- Main Real-Time Loop ---
    print("\n--- Starting High-Frequency Inference & Trading Loop ---")
    try:
        while (is_market_open() or (DEBUG=="TRUE")):
            now_utc = datetime.now(timezone.utc)
            
            # --- State Update and Data Fetching ---
            update_flag = (now_utc.second >= 55 and now_utc.minute != last_permanent_update_minute)
            if update_flag: last_permanent_update_minute = now_utc.minute

            trading_data = process_REALTIME_BULK_QUOTES(
                av, tickers, extended_hours=True
            )

            
            if not (trading_data) :
                print("No data fetched, skipping cycle.")
                time.sleep(1)
                continue
            
            # --- Perform Inference ---
            inference_item = inference_engine.update_hot_tensor_and_return_inference_item(processing_tickers=tickers, new_quote_data=trading_data, update=update_flag)



            profit_logits, accuracy_logits = inference_engine.infer(inference_item)

            # --- Trading Logic ---
            print(f"\n--- Cycle at {now_utc.strftime('%H:%M:%S')} UTC ---")
            for i, ticker in enumerate(inference_engine.tickers):
                profit_signal = profit_logits[i].item()
                accuracy_signal = accuracy_logits[i].item()
                
                print(f"  Signals for {ticker}: Profit={profit_signal:.4f}, Accuracy={accuracy_signal:.4f}")
                
                if now_utc < trade_cooldown_expiry[ticker]: continue
                

                if profit_signal > 0 and accuracy_signal > 0:
                    print(f"  >>> ✅ Agreement found! Attempting trade for {ticker}")
                    entry_price = round(trading_data[ticker]["close"], 2)
                    
                    '''
                    KEY EDIT POINT INSTEAD OF ACTUALLY PLACING A TRADE SAVE TO CSV 
                    id, current time, ticker, action: {buy, sell}, price at time, profit loss, cumulative profit loss

                    '''
                    # parent_order_id = app.place_bracket_order_with_maturity(
                    #     ticker=ticker, take_profit_float=take_profit, stop_loss_float=stop_loss,
                    #     entry_price=entry_price, quantity=order_quantity
                    # )
                    
                    # if parent_order_id:
                    #     print(f"    --> ✅ Bracket order submitted with Parent ID: {parent_order_id}")
                    #     expiry_time = now_utc + timedelta(seconds=frequency_limiter_seconds)
                    #     trade_cooldown_expiry[ticker] = expiry_time
                    #     print(f"    --> ⏳ {ticker} is now in cooldown until {expiry_time.strftime('%H:%M:%S')} UTC.")
                    # else:
                    #     print(f"    --> ℹ️ Order for {ticker} was not submitted (risk limits, etc.).")


                    
            inference_engine.visualize_inference_tensor(inference_item, "ouput_inference_tensor.png")
            inference_engine.visualize_ohlcv_buffers( "ouput_ohlcv_buffer.png")
            time.sleep(4)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("\n--- Initiating Shutdown Sequence ---")


if __name__ == "__main__":
    main()