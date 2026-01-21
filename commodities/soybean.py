from ib_insync import *
import datetime

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

def get_last_year_soybean_contracts():
    # 1. Critical Step: Include 'includeExpired=True' to see past contracts
    contract = Future(symbol='ZS', exchange='CBOT', currency='USD', includeExpired=True)
    
    # 2. Request all possible contract details
    details = ib.reqContractDetails(contract)
    
    if not details:
        print("No contracts found. Check your permissions.")
        return []

    # 3. Define our 'Last Year' boundary
    one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
    today = datetime.datetime.now().strftime('%Y%m%d')

    last_year_contracts = []

    for d in details:
        expiry = d.contract.lastTradeDateOrContractMonth
        
        # 4. Filter: Contract must have expired in the last year OR be active
        # (Using string comparison for YYYYMMDD format)
        if expiry >= one_year_ago:
            last_year_contracts.append(d.contract)

    # 5. Sort by expiration date
    last_year_contracts.sort(key=lambda x: x.lastTradeDateOrContractMonth)
    
    return last_year_contracts

# Execute and Print
contracts_list = get_last_year_soybean_contracts()
first_contract = contracts_list[-1]

print("--- Available Attributes in the Contract Object ---")
for key, value in vars(first_contract).items():
    print(f"{key:<30}: {value}")

ib.reqMarketDataType(1) 

# 2. Request the market data "ticker" for your soybean contract
ticker = ib.reqMktData(first_contract)

# 3. CRITICAL: You must wait for the data to travel across the internet
ib.sleep(2) 

# 4. Now print the price
print(f"--- Market Data for {first_contract.localSymbol} ---")
print(f"Last Price: {ticker.last}")
print(f"Bid: {ticker.bid} | Ask: {ticker.ask}")
print(f"Volume: {ticker.volume}")
print(f"\n--- Soybean Contracts Found (Last 12 Months) ---")
print(f"{'Local Symbol':<15} | {'Expiry Date':<12} | {'ConId'}")
print("-" * 45)

    
for c in contracts_list:
    print(f"{c.localSymbol:<15} | {c.lastTradeDateOrContractMonth:<12} | {c.conId}")

ib.disconnect()