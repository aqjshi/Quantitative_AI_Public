# class Quote(Base):
#     __tablename__ = "quotes"

#     id = Column(BigInteger, primary_key=True, autoincrement=True)
#     ticker_hash = Column(BigInteger, index=True, nullable=False)
    
#     # Use BigInteger for nanoseconds since epoch
#     t = Column(BigInteger, index=True, nullable=False) 

#     # --- TRADE DATA ---
#     o = Column(Numeric(18, 8))
#     h = Column(Numeric(18, 8))
#     l = Column(Numeric(18, 8))
#     c = Column(Numeric(18, 8))
#     v = Column(Numeric(24, 10), default=0) 
#     vw = Column(Numeric(18, 8))

#     # --- NBBO DATA ---
#     asof_bid_price = Column(Numeric(18, 8)) 
#     asof_ask_price = Column(Numeric(18, 8))
#     asof_bid_size = Column(BigInteger) 
#     asof_ask_size = Column(BigInteger)

#     is_stale = Column(Boolean, default=False) 

#     # Composite Index for lightning fast Lead-Lag lookups
#     __table_args__ = (
#         Index('ix_instr_time', 'ticker_hash', 't'),
#     )

#     @property
#     def mid_price(self):
#         if self.asof_bid_price is not None and self.asof_ask_price is not None:
#             return (self.asof_bid_price + self.asof_ask_price) / 2
#         return self.c


def fetch_and_map_bars_batch(tickers, membership_router, api_key, config ):
    mapped_entries = []
    # Use the config keys correctly
    start_date = config["fetch_start"]
    end_date = config["train_end"]
    multiplier = config["multiplier"]
    timespan = config["timespan"]

    for ticker in tickers:
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        params = {
            "adjusted": "true",
            "limit": 50000,
            "sort": "asc",
            "apiKey": api_key
        }

        
        current_url = url
        current_params = params # Only sent on first request

        while current_url:
            try:
                resp = requests.get(current_url, params=current_params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                if resp.status_code != 200:
                    print(f" [!] API Error {resp.status_code} on {ticker}")
                    break

                data = resp.json()
                results = data.get('results', [])
                
                # Process results immediately to save RAM
                for d in results:
                    unix_t_millis = d.get('t') 
                    
                    # FIX: Convert Millis to Nanos for the resolver/DB consistency
                    unix_t_ns = unix_t_millis * 1_000_000 
                    
                    inst_id = resolve_ticker_hash(ticker, unix_t_ns, membership_router)

                    if inst_id:
                        entry = (
                            inst_id, 
                            d.get('c'), d.get('h'), d.get('l'), d.get('n'), 
                            d.get('o'), 
                            unix_t_ns, # Store Nanos in DB 't' column
                            d.get('v'), d.get('vw')
                        )
                        mapped_entries.append(entry)

                # Pagination
                next_url = data.get('next_url')
                if next_url:
                    current_url = f"{next_url}&apiKey={api_key}" if "apiKey" not in next_url else next_url
                    current_params = None
                else:
                    current_url = None

            except Exception as e:
                print(f" [!] Fetch Error {ticker}: {e}")
                break
                
    return mapped_entries



def fetch_and_map_quotes_batch(tickers, membership_router, api_key, config):
    mapped_entries = []


    def to_ms(date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1_000)

    # Note: If your config already has tounix, use that, 
    # but here is the manual conversion for safety:
    start_ms = to_ms(config["fetch_start"])
    end_ms = to_ms(config["train_end"])
    # 1. LOOP through tickers here
    for ticker in tickers:
        url = f"https://api.massive.com/v3/quotes/{ticker}"
        
        params = {
            "timestamp.lte": end_ms,
            "timestamp.gte": start_ms,
            "adjusted": "true",
            "limit": 50000,
            "sort": "asc",
            "apiKey": api_key
        }

        current_url = url
        current_params = params

        while current_url:
            try:
                resp = requests.get(current_url, params=current_params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                if resp.status_code != 200:
                    print(f" [!] API Error {resp.status_code} on {ticker}")
                    break

                data = resp.json()
                results = data.get('results', [])
                
                for d in results:
                    unix_t_ns = d.get('sip_timestamp')
                    
                    # Nanos are native here, so resolver works fine
                    inst_id = resolve_ticker_hash(ticker, unix_t_ns, membership_router)

                    if inst_id:
                        # Indices for Reference in Merge:
                        # 0:id, 1:ticker, 2:ask_ex, 3:ask_p, 4:ask_s, 5:bid_ex, 6:bid_p, 7:bid_s, 8:seq, 9:ts
                        entry = (
                            inst_id, ticker,
                            d.get('ask_exchange'), d.get('ask_price'), d.get('ask_size'),
                            d.get('bid_exchange'), d.get('bid_price'), d.get('bid_size'),
                            d.get('sequence_number'), 
                            unix_t_ns, # Index 9
                            d.get('tape')
                        )
                        mapped_entries.append(entry)

                next_url = data.get('next_url')
                if next_url:
                    current_url = f"{next_url}&apiKey={api_key}" if "apiKey" not in next_url else next_url
                    current_params = None
                else:
                    current_url = None

            except Exception as e:
                print(f" [!] Fetch Error {ticker}: {e}")
                break
                
    return mapped_entries


def ingest_merged_data(tickers, figi_id_map, discovery, config, result_queue, api_key):
    """
    1. Builds Router from DB.
    2. Fetches Bars.
    3. Fetches Quotes.
    4. Merges (Zipper).
    5. Pushes to DB Queue.
    """
    # --- STEP 1: Build the Router ---
    # We need a temporary DB session to read the TickerMap
    session = SessionLocal() 
    try:
        router = build_ticker_router(session, tickers)
    finally:
        session.close()

    # --- STEP 2: Parallel Fetching ---
    bars_data = []
    quotes_data = []
    
    # We fetch Bars and Quotes in parallel to save network time
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 2a. Fetch Bars
        future_bars = executor.submit(
            fetch_and_map_bars_batch, 
            tickers, router, api_key, config
        )
        
        # 2b. Fetch Quotes
        future_quotes = executor.submit(
            fetch_and_map_quotes_batch,
            tickers, router, api_key, config
        )
        
        bars_data = future_bars.result()
        quotes_data = future_quotes.result()

    if not bars_data:
        return 

    # Sort is CRITICAL for groupby to work
    bars_data.sort(key=lambda x: (x[0], x[6]))    # Sort by InstID, then Time
    quotes_data.sort(key=lambda x: (x[0], x[9]))  # Sort by InstID, then Time

    from itertools import groupby

    # FIX: Fully realize these into dictionaries immediately
    # GroupBy requires the data to be sorted by the key first (which we did above)
    bars_by_inst = {k: list(v) for k, v in groupby(bars_data, key=lambda x: x[0])}
    quotes_by_inst = {k: list(v) for k, v in groupby(quotes_data, key=lambda x: x[0])}

    merged_rows = []

    # Iterate through INSTRUMENTS (the Keys)
    for inst_id, bars in bars_by_inst.items():
        quotes = quotes_by_inst.get(inst_id, [])
        q_idx = 0
        n_quotes = len(quotes)
        
        for b in bars:
            bar_ts = b[6] # 't' (nanoseconds)
            
            # ZIPPER LOGIC
            # Advance quote pointer while the NEXT quote is still <= current bar time
            while q_idx + 1 < n_quotes and quotes[q_idx+1][9] <= bar_ts:
                q_idx += 1
            
            # Determine if the current quote pointer is valid
            best_quote = None
            if n_quotes > 0 and quotes[q_idx][9] <= bar_ts:
                best_quote = quotes[q_idx]
            
            # formatting helpers for None values
            bid_p = best_quote[6] if best_quote and best_quote[6] is not None else ''
            ask_p = best_quote[3] if best_quote and best_quote[3] is not None else ''
            bid_s = best_quote[7] if best_quote and best_quote[7] is not None else ''
            ask_s = best_quote[4] if best_quote and best_quote[4] is not None else ''

            # Construct Row (Tab Separated)
            # Schema: ticker_hash, t, o, h, l, c, v, vw, asof_bid_p, asof_ask_p, asof_bid_s, asof_ask_s
            row_str = (
                f"{b[0]}\t{b[6]}\t"
                f"{b[5]}\t{b[2]}\t{b[3]}\t{b[1]}\t"
                f"{b[7]}\t{b[8]}\t"
                f"{bid_p}\t{ask_p}\t{bid_s}\t{ask_s}\n"
            )
            merged_rows.append(row_str)

    # --- STEP 4: Push to Queue ---
    if merged_rows:
        copy_sql = """
            COPY quotes (
                ticker_hash, t, o, h, l, c, v, vw, 
                asof_bid_price, asof_ask_price, asof_bid_size, asof_ask_size
            ) FROM STDIN WITH (FORMAT text, DELIMITER '\t', NULL '')
        """
        result_queue.put((merged_rows, copy_sql, f"Merged Batch {tickers[0]}..."))
