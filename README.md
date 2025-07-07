# Quantitative AI Public Project

**Developer:** Qingjian Shi
**Contact:** aqj.shi@gmail.com

This project provides an automated trading system. This uses a 2 party veto system in which one profit party and one accuracy party must both agree in which a trade is profitable in which a trade is made. This allows a breakthrough from 68% converged accuracy in the validation set,  and higher profitability ratio in the profit model by pruning the negative false positive predictions.






**Please read the instructions below carefully to set it up and run it.**

## Prerequisites & Setup

To run this application, you will need the following:

1.  **Interactive Brokers (IBKR) Trader Workstation (TWS) API Client:**
    * TWS is Interactive Brokers' proprietary trading platform. This application connects to the TWS API to execute trades and retrieve live market data.
    * **Installation:** Ensure you have the TWS application installed and running on your machine. You can download it from the official Interactive Brokers website.
    * **API Port Configuration:** You must enable API access and configure the correct port within the TWS application.
        * Open TWS.
        * Go to **File > Global Configuration**.
        * Navigate to **API > Settings**.
        * Ensure "Enable ActiveX and Socket Clients" is checked.
        * Verify that the "Socket Port" is set to `7496` (or `7497` for paper trading accounts if applicable).
        * Ensure "Read-Only API" is unchecked if you intend to place trades.
        * Add your machine's IP address to "Trusted IPs" if you encounter connection issues, although `127.0.0.1` (localhost) is usually sufficient.

2.  **Alpha Vantage API Client with Real-Time Data Stream Subscription:**
    * Alpha Vantage is a provider of financial market data. This application uses Alpha Vantage for historical and potentially some real-time data.
    * **Subscription:** You need an Alpha Vantage API key. Ensure your subscription plan includes access to real-time data streams.
    * **API Configuration for Real-Time Data:** You must configure your API key to allow for real-time data streaming. This is typically done through the Alpha Vantage dashboard or their "AlphaX terminal" if they provide specific real-time data access tools. Refer to Alpha Vantage's documentation for details on enabling real-time data access for your API key.

## Environment Configuration

Before running the program, you need to set up your API keys as environment variables:

1.  **Rename `production.env` to `.env`:**
    * In the root directory of this project, you will find a file named `production.env`. Rename this file to `.env`.

2.  **Add Your API Keys to `.env`:**
    * Open the newly renamed `.env` file.
    * Add your `ALPHA_VANTAGE_API_KEY` and any other required API keys (e.g., for TWS if it uses an API key, though TWS typically uses connection via socket port) in the following format:

    ```
    ALPHA_VANTAGE_API_KEY="YOUR_ALPHA_VANTAGE_API_KEY_HERE"
    # Add other keys as needed, e.g., for TWS if applicable
    ```
    * Replace `"YOUR_ALPHA_VANTAGE_API_KEY_HERE"` with your actual key.

## Installation Steps

Open your terminal or command prompt and follow these steps:

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the Virtual Environment:**
    * **Linux/Mac:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows Powershell:**
        ```powershell
        venv\scripts\activate.ps1
        ```

3.  **Install Dependencies:**
    * With the virtual environment activated, install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Program

Once all prerequisites are met and dependencies are installed:

1.  **Run the Main Program:**
    ```bash
    python exe.py exe_params.json
    ```
    * This command starts the main execution script, using `exe_params.json` for its parameters.

## Automation and Monitoring

After initiation, the program is designed to automate the trading process by fetching real-time data, making trading decisions, and executing orders. It is intended to run autonomously.

However, please be aware that despite automation, bugs or unexpected market conditions might occasionally leave positions unclosed. It is highly recommended to monitor the program's activity and your brokerage account regularly.
**PORTFOLIO LIQUIDITY WARNING**
If your portfolio allows for larger position sizes, accomodate it by editing exe_paras.json in position size order_quantity, we assume you ahve infinite position size such that the model will not factor in fixed transaction fee of $2.00[...] when putting in an order, as profitability sclaes linearly with reasonable position size (around 5- 200 shares).



**Disclaimer 1**
I have the $250 AV API key, which allows for 4 calls per second, if you purchase a cheaper plan, you will have to edit exe_params specifially  
```
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("\n--- Initiating Shutdown Sequence ---")
        if app.isConnected():
            app.liquidate_all_positions()
        app.disconnect()
        print("Disconnected from TWS.")
``` 
into a higher sleep time to accomodate for your lower frequency data availability, hoever the code still works, it will jsut harass AV client for rejections until your allocation becomes available. 


**Disclaimer 2**
I take no responsibility for the profitability, negligence, or any losses/gains incurred from the use of this software. Always exercise caution and understand the risks involved in automated trading.

**Follow updates on blendingwaves.com for further information.**
For Model Backtest Results check the config file, for trained models have a paired config file that display performance diagnostics. This is STRICTLY for academic purposes, and historical performance may not be the most accurate in reflecting real world performance. 
<<<<<<< HEAD

=======
>>>>>>> 98152812c4b100fed7a61c158cf7b596b5d509fd
