import time
import threading
from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.execution import Execution
from ibapi.contract import ComboLeg
from datetime import datetime, timedelta
from av_client import av_client
from cache import process_REALTIME_BULK_QUOTES,RealTimeInferenceEngine
from ibapi.order import Order
from ibapi.order_condition import TimeCondition
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

class TradeApp(EWrapper, EClient):
    def __init__(self, max_exposure_limit: float = 2000.00):
        EClient.__init__(self, self)
        self.nextOrderId = None
        self.stale_order_tracker = {}
        self.max_exposure = max_exposure_limit
        self.current_exposure = 0.0
        self.open_positions = {}
        self.pending_orders = {}
        self.order_placement_lock = threading.Lock()
        self.exposure_lock = threading.Lock()
        self.position_lock = threading.Lock()
        


    def orderStatus(self, orderId: OrderId, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCappedPrice: float):
        print(f"OrderStatus - Id: {orderId}, Status: {status}, Filled: {filled}")
        # When a position is closed (TP/SL is filled), remove it from our tracker
        with self.position_lock:
            for symbol, data in list(self.open_positions.items()):
                # Check if this orderId corresponds to a TP or SL of an open position
                if orderId == data.get('tp_id') or orderId == data.get('sl_id'):
                    if status == "Filled":
                        print(f"Position for {symbol} closed by order {orderId}. Removing from active positions.")
                        del self.open_positions[symbol]
                        break
                # Also handle the parent buy order being filled
                elif orderId in self.pending_orders and status == "Filled":
                    # This means the initial BUY order (parent) was filled
                    details = self.pending_orders.pop(orderId) # Remove from pending
                    # Update open_positions with actual fill data if needed, or simply mark it as open
                    # For this test, we care that it enters open_positions
                    print(f"Parent BUY order {orderId} for {symbol} filled. Position now active.")
                    # If execDetails is not called quickly enough or you need to ensure it's in open_positions immediately
                    # you could add a fallback here:
                    if symbol not in self.open_positions:
                         self.open_positions[symbol] = {
                            'shares': filled,
                            'value': filled * avgFillPrice, # Approximate value
                            'entry_time': datetime.now(timezone.utc),
                            'tp_id': details['tp_id'],
                            'sl_id': details['sl_id']
                        }


    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        super().execDetails(reqId, contract, execution)
        print(f"ExecDetails - Symbol: {contract.symbol}, Side: {execution.side}, Shares: {execution.shares}, Price: {execution.price}")

        with self.exposure_lock, self.position_lock:
            executed_value = execution.shares * execution.price
            
            if execution.side == "BOT":
                self.current_exposure += executed_value
                parent_order_id = execution.orderId
                if parent_order_id in self.pending_orders:
                    details = self.pending_orders.pop(parent_order_id)
                    self.open_positions[contract.symbol] = {
                        'shares': execution.shares,
                        'value': executed_value,
                        'entry_time': datetime.now(timezone.utc),
                        'tp_id': details['tp_id'],
                        'sl_id': details['sl_id']
                    }

            elif execution.side == "SLD":
                if contract.symbol in self.open_positions:
                    position = self.open_positions[contract.symbol]
                    avg_cost = position['value'] / position['shares'] if position['shares'] > 0 else 0
                    value_to_remove = execution.shares * avg_cost
                    self.current_exposure -= value_to_remove
                    position['shares'] -= execution.shares
                    position['value'] -= value_to_remove
                    if position['shares'] <= 0:
                        del self.open_positions[contract.symbol]
        

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158, 2107]:
            print(f"Error reqId:{reqId}, Code:{errorCode}, Msg:{errorString}")


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        print(f"Connection successful. Next valid order ID: {orderId}")
    next_order_id_event = threading.Event()

    def place_bracket_order_with_maturity(self, ticker: str, take_profit_float: float, stop_loss_float: float, entry_price: float, quantity: int, maturity_minutes: int = 15):
        with self.order_placement_lock:
            if self.nextOrderId is None:
                print("Error: nextOrderId is not set.")
                return None

            potential_exposure_increase = entry_price * quantity
            with self.exposure_lock:
                if self.current_exposure + potential_exposure_increase > self.max_exposure:
                    print(f"🚫 EXPOSURE LIMIT REACHED. Order for {quantity} {ticker} BLOCKED.")
                    return None

            mycontract = Contract()
            mycontract.symbol = ticker
            mycontract.secType = "STK"
            mycontract.exchange = "SMART"
            mycontract.currency = "USD"
            mycontract.primaryExchange = "NASDAQ"

            tp_price = round(entry_price * (1 + take_profit_float), 2)
            sl_price = round(entry_price * (1 - stop_loss_float), 2)
            sl_limit_price = round(sl_price - 0.03, 2)


            oca_group_name = f"OCA_{ticker}_{self.nextOrderId}"
            
            parent = Order()
            parent.orderId = self.nextOrderId
            parent.action = "BUY"
            parent.orderType = "LMT"
            parent.lmtPrice = entry_price
            parent.totalQuantity = quantity
            parent.eTradeOnly = False
            parent.firmQuoteOnly = False
            parent.transmit = False

            profit_taker = Order()
            profit_taker.orderId = parent.orderId + 1
            profit_taker.parentId = parent.orderId
            profit_taker.action = "SELL"
            profit_taker.orderType = "LMT"
            profit_taker.lmtPrice = tp_price
            profit_taker.totalQuantity = quantity
            profit_taker.ocaGroup = oca_group_name
            profit_taker.ocaType = quantity
            profit_taker.eTradeOnly = False
            profit_taker.firmQuoteOnly = False
            profit_taker.transmit = False

            stop_loss_order = Order()
            stop_loss_order.orderId = parent.orderId + 2
            stop_loss_order.parentId = parent.orderId
            stop_loss_order.action = "SELL"
            stop_loss_order.orderType = "STP LMT"
            stop_loss_order.auxPrice = sl_price
            stop_loss_order.lmtPrice = sl_limit_price
            stop_loss_order.totalQuantity = quantity
            stop_loss_order.ocaGroup = oca_group_name
            stop_loss_order.ocaType = quantity
            stop_loss_order.eTradeOnly = False
            stop_loss_order.firmQuoteOnly = False
            stop_loss_order.transmit = True 

            print(f"Placing Bracket Order for {quantity} {ticker}: Entry={entry_price}, TP={tp_price}, SL={sl_price}")
            
            self.placeOrder(parent.orderId, mycontract, parent)
            self.placeOrder(profit_taker.orderId, mycontract, profit_taker)
            self.placeOrder(stop_loss_order.orderId, mycontract, stop_loss_order)


            self.pending_orders[parent.orderId] = {'tp_id': profit_taker.orderId, 'sl_id': stop_loss_order.orderId}
            
            self.nextOrderId += 3
            return parent.orderId

    def orderStatus(self, orderId: OrderId, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCappedPrice: float):
        print(f"OrderStatus - Id: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPrice: {avgFillPrice}")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        print(f"OpenOrder - Id: {orderId}, Symbol: {contract.symbol}, Action: {order.action}, Total Qty: {order.totalQuantity}, Status: {orderState.status}")



    # This is the most reliable way to liquidate a position
    def liquidate_specific_position(self, ticker: str, quantity: int, tp_id: int, sl_id: int):
        """
        Liquidates one position by canceling its bracket orders and submitting a new MKT order.
        """
        print(f"\nPOSITION EXPIRED! Liquidating {quantity} shares of {ticker} with a MKT order...")
        
        # Step 1: Cancel the outstanding child orders to prevent conflicts
        print(f"  -> Cancelling Take Profit (ID: {tp_id}) and Stop Loss (ID: {sl_id}).")
        self.cancelOrder(tp_id) 
        self.cancelOrder(sl_id)
        time.sleep(0.5) # A brief pause for cancellations to process

        # Step 2: Place a new Market (MKT) order to sell immediately
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = Order()
        order.action = "SELL"
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.transmit = True

        # Use a new, unique order ID
        liquidate_order_id = self.nextOrderId
        self.placeOrder(liquidate_order_id, contract, order)
        self.nextOrderId += 1
        print(f"  -> MKT SELL order (ID: {liquidate_order_id}) submitted for {ticker}.")

        # Proactively remove the position from tracking to prevent re-liquidation
        with self.position_lock:
            if ticker in self.open_positions:
                del self.open_positions[ticker]


    def liquidate_all_positions(self):
        print("\n" + "="*40)
        print(" INITIATING EMERGENCY LIQUIDATION ")
        print("="*40)

        print("--> Step 1: Cancelling all open orders...")
        self.reqGlobalCancel()
        time.sleep(1) # Give the cancellation request a moment to process

        # Step 2: Liquidate all known positions
        print(f"--> Step 2: Liquidating {len(self.open_positions)} known positions...")
        
        with self.exposure_lock:
            if not self.open_positions:
                print("   - No open positions to liquidate.")
                return

            # Create a copy of the keys to safely iterate
            for symbol in list(self.open_positions.keys()):
                position_data = self.open_positions[symbol]
                quantity_to_sell = position_data.get('shares', 0)

                if quantity_to_sell > 0:
                    print(f"   - Submitting MKT SELL for {quantity_to_sell} shares of {symbol}...")

                    # Define the contract to sell
                    contract = Contract()
                    contract.symbol = symbol
                    contract.secType = "STK"
                    contract.exchange = "SMART"
                    contract.currency = "USD"
                    
                    # Create a market order for immediate execution
                    order = Order()
                    order.action = "SELL"
                    order.orderType = "MKT"
                    order.totalQuantity = quantity_to_sell
                    order.eTradeOnly = False
                    order.firmQuoteOnly = False

                    # Place the order
                    self.placeOrder(self.nextOrderId, contract, order)
                    self.nextOrderId += 1
                else:
                    print(f"   - Skipping {symbol}, quantity is zero.")
        
        print("\n\Liquidation orders have been submitted.")
        print("Monitor your TWS terminal to confirm all positions are closed.")
        time.sleep(3) # Allow time for orders to be sent before disconnecting

    def manage_expired_positions(self, forecast_depth_minutes: int):
        """
        Checks for any open positions that have exceeded their forecast depth
        and liquidates them immediately.
        """
        now_utc = datetime.now(timezone.utc)
        
        # Iterate over a copy of the items to allow safe modification during the loop
        with self.position_lock:
            # print(f"DEBUG: Checking {len(self.open_positions)} open positions for expiry.")
            for ticker, data in list(self.open_positions.items()): # Use list() to iterate over a copy
                entry_time = data['entry_time']
                expiry_time = entry_time + timedelta(minutes=forecast_depth_minutes)
                
                # print(f"DEBUG: Ticker: {ticker}, Entry: {entry_time.isoformat()}, Expiry: {expiry_time.isoformat()}, Now: {now_utc.isoformat()}")

                if now_utc > expiry_time:
                    print(f"Position for {ticker} (entered at {entry_time.isoformat()}) has expired (expiry at {expiry_time.isoformat()}).")
                    self.liquidate_specific_position(
                        ticker=ticker,
                        quantity=data['shares'],
                        tp_id=data['tp_id'],
                        sl_id=data['sl_id']
                    )
                else:
                    print(f"\Position for {ticker} is still active. Expires in {(expiry_time - now_utc).total_seconds():.0f} seconds.")


def main():
    
    config_filepath = sys.argv[1]
    DEBUG = sys.argv[2]
    with open(config_filepath, 'r', encoding='utf-8') as f:
        known = json.load(f)
        tickers = known["ticker"]
        params = known["params"]
    
        forecast_depth = known["forecast_depth"]
        context_depth = known["context_depth"]
        profit_model_path = known["profit_model_path"]
        accuracy_model_path = known["accuracy_model_path"]
        order_quantity = known["order_quantity"]
        take_profit = known["take_profit"]
        stop_loss = known["stop_loss"]
        maximum_exposure = known["maximum_exposure"]
        frequency_limiter_seconds = known["frequency_limiter_seconds"]

    tickers = sorted(tickers)
    params = sorted(params)

    # --- App and Threading Setup ---
    app = TradeApp(max_exposure_limit=maximum_exposure)
    app.connect("127.0.0.1", 7496, clientId=0)
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    print("Connecting to TWS...")
    time.sleep(2)

    if app.isConnected():
        app.liquidate_all_positions()

    if app.nextOrderId is None:
        print("Could not connect to TWS or get nextValidId. Exiting.")
        return

    # --- Engine and State Initialization ---
    inference_engine = RealTimeInferenceEngine(tickers, profit_model_path, accuracy_model_path, context_depth, params)
    inference_engine.cold_start()
    
    av = av_client() # Initialize Alpha Vantage client

    # --- Manual Order Placement for Test ---
    # Pick a ticker from your config.json (e.g., "SPY")
    test_ticker = tickers[0] 
    test_quantity = order_quantity # Use the configured quantity
    
    # Fetch a real-time price to use as entry_price
    print(f"\nFetching current price for {test_ticker} for test order...")
    current_market_data = process_REALTIME_BULK_QUOTES(av, [test_ticker], extended_hours=True)
    if not current_market_data or test_ticker not in current_market_data:
        print(f"Could not fetch current market data for {test_ticker}. Cannot place test order. Exiting.")
        app.disconnect()
        sys.exit(1)
    
    current_price = round(current_market_data[test_ticker]["close"], 2)
    print(f"Current price for {test_ticker}: {current_price}")

    # Manually place the bracket order
    # Set a very short forecast_depth (e.g., 5 minutes) in config.json for quick expiry
    print("\n--- Placing a MANUAL TEST ORDER to trigger expiration ---")
    parent_order_id = app.place_bracket_order_with_maturity(
        ticker=test_ticker,
        take_profit_float=take_profit, # From config
        stop_loss_float=stop_loss,     # From config
        entry_price=current_price,
        quantity=test_quantity,
        maturity_minutes=forecast_depth # This parameter is actually unused in your current place_bracket_order_with_maturity, but it matches the idea of forecast_depth
    )

    if parent_order_id:
        print(f"Manual test bracket order submitted for {test_ticker} with Parent ID: {parent_order_id}")
        print(f"Waiting for {forecast_depth} minutes for position to expire...")
    else:
        print(f"Manual test order for {test_ticker} could not be placed. Exiting.")
        app.disconnect()
        sys.exit(1)

    # --- Main Real-Time Loop (Now primarily for managing expired positions) ---
    print("\n--- Starting Monitoring Loop for Expired Positions ---")
    try:
        start_time_of_loop = datetime.now(timezone.utc)
        while (is_market_open() or (DEBUG and datetime.now(timezone.utc) < start_time_of_loop + timedelta(minutes=forecast_depth + 2))): # Run a bit longer than forecast_depth
            now_utc = datetime.now(timezone.utc)
            
            # --- Call the position manager ---
            # This is the function you want to test!
            app.manage_expired_positions(forecast_depth)
            
            # --- Original AI Trading Logic (DISABLED for this test) ---
            # If you want to keep the inference loop, but disable active trading:
            # trading_data = process_REALTIME_BULK_QUOTES(av, tickers, extended_hours=True)
            # if trading_data:
            #     inference_item = inference_engine.update_hot_tensor_and_return_inference_item(
            #         processing_tickers=tickers, new_quote_data=trading_data, update=True
            #     )
            #     profit_logits, accuracy_logits = inference_engine.infer(inference_item)
            #     for i, ticker in enumerate(inference_engine.tickers):
            #         profit_signal = profit_logits[i].item()
            #         accuracy_signal = accuracy_logits[i].item()
            #         # print(f"  Signals for {ticker}: Profit={profit_signal:.4f}, Accuracy={accuracy_signal:.4f}")
            #         # Original trading condition:
            #         # if profit_signal > 0 and accuracy_signal > 0 and now_utc >= trade_cooldown_expiry[ticker]:
            #         #     ... (Do NOT place new orders here during this specific test)


            time.sleep(5) # Check every 5 seconds or so
            print(f"Loop running... Current time: {now_utc.strftime('%H:%M:%S UTC')}, Open Positions: {len(app.open_positions)}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("\n--- Initiating Shutdown Sequence ---")
        if app.isConnected():
            app.liquidate_all_positions() # Clean up any remaining positions/orders
            app.disconnect()
            print("Disconnected from TWS.")
        else:
            print("Not connected to TWS.")
        print("Script finished.")

if __name__ == "__main__":
    main()