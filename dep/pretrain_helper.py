from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from scipy.optimize import newton, brent, brentq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
load_dotenv()

SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)

def load_minutes(engine: Engine, ticker: str, day: str, max_dte: int) -> pd.DataFrame:
    """Loads and cleans options data for a day, applying initial DTE filtering, including volume."""
    sql = text("""
        SELECT contract_type, strike, contract_expiry, time_entry_ts, option_close, option_volume_weighted, 
               option_volume, option_transactions
        FROM option_quotes
        WHERE date(to_timestamp(time_entry_ts) AT TIME ZONE 'UTC') = DATE :d
          AND option_name LIKE :osi_prefix
          AND option_close IS NOT NULL
    """)
    df = pd.read_sql(sql, engine, params={"d": day, "osi_prefix": f"O:{ticker.upper()}%"})
    if df.empty: return df
    
    day_date = pd.to_datetime(day).date()
    
    df = df.assign(
        ts_utc=pd.to_datetime(df["time_entry_ts"], unit="s", utc=True),
        strike=pd.to_numeric(df["strike"], errors="coerce"),
        option_close=pd.to_numeric(df["option_close"], errors="coerce"),
        option_volume=pd.to_numeric(df["option_volume"], errors="coerce").fillna(1),
        option_transactions = pd.to_numeric(df["option_transactions"], errors="coerce").fillna(1),
        option_volume_weighted = pd.to_numeric(df["option_volume_weighted"], errors="coerce"),

        option_type=df["contract_type"].map({"C":"call","P":"put"}),
        expiry_date=pd.to_datetime(df["contract_expiry"], unit="s", utc=True).dt.date
    )
    
    df["dte"] = (df["expiry_date"] - day_date).apply(lambda x: x.days).astype(int)
    # Aggressive filtering based on max DTE parameter (e.g., 90 days)
    df = df[(df["dte"] > 0) & (df["dte"] <= max_dte)].dropna(subset=["strike"])
    
    return df




def load_underlying(engine: Engine, day: str, company_id: int, volume_profiling_burn_in_days: int) -> pd.Series:
    day_dt = pd.to_datetime(day)
    # Go back N days to capture enough history for the profile
    # We add buffer to ensure we get 'volume_profiling_burn_in_days' trading days
    start_dt = day_dt - timedelta(days=volume_profiling_burn_in_days) 
    start_ts_unix = int(start_dt.timestamp())
    end_ts_unix = int((day_dt + timedelta(days=1)).timestamp())

    # 2. Fetch Raw Data (Price + Volume)
    # Note: Added 'volume' to the SELECT statement
    sql = text("""
        SELECT time_entry_ts, close_price, volume
        FROM quotes
        WHERE
            company_id = :cid
            AND time_entry_ts >= :start_ts
            AND time_entry_ts < :end_ts
        ORDER BY time_entry_ts ASC
    """)
    
    df = pd.read_sql(sql, engine, params={
        "cid": company_id,
        "start_ts": start_ts_unix,
        "end_ts": end_ts_unix
    })
    if df.empty:
        print(f"[WARN] No underlying minute quotes found for company_id={company_id} on {day}.")
        return pd.Series(dtype=float)

    # 3. Preprocess Raw Data
    df['ts_utc'] = pd.to_datetime(df['time_entry_ts'], unit='s', utc=True)
    # Convert to Eastern Time for correct 9:30-16:00 alignment
    df['ts_et'] = df['ts_utc'].dt.tz_convert('US/Eastern')
    df['date_str'] = df['ts_et'].dt.strftime('%Y-%m-%d')
    df['minute_of_day'] = (df['ts_et'].dt.hour * 60 + df['ts_et'].dt.minute) - (9 * 60 + 30)
    
    # Ensure numeric
    df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # 4. Filter for Market Hours Only (0 to 389 minutes)
    # We only care about 9:30 AM to 4:00 PM ET for the profile
    market_df = df[(df['minute_of_day'] >= 0) & (df['minute_of_day'] < 390)].copy()

    # 5. Build the History Matrix (M_hist)
    # Pivot: Index=Date, Columns=Minute (0-389), Values=Volume
    pivot_vol = market_df.pivot_table(
        index='date_str', 
        columns='minute_of_day', 
        values='volume', 
        aggfunc='sum' # Sum in case multiple ticks per minute
    )
    
    # Ensure all 390 minutes exist
    pivot_vol = pivot_vol.reindex(columns=range(390), fill_value=np.nan)

    # Separate "History" from "Today"
    today_str = day_dt.strftime('%Y-%m-%d')
    
    if today_str in pivot_vol.index:
        today_vol_vector = pivot_vol.loc[today_str].values # Shape (390,)
        # Use previous N days for history, excluding today
        history_dates = pivot_vol.index[pivot_vol.index < today_str][-volume_profiling_burn_in_days:]
        M_hist = pivot_vol.loc[history_dates].values # Shape (N, 390)
    else:
        # Fallback if today isn't in DB yet (or partial)
        today_vol_vector = np.full(390, np.nan)
        M_hist = pivot_vol.values


    daily_totals = np.nansum(M_hist, axis=1, keepdims=True).astype(float)
    # Avoid div by zero if a day has 0 volume
    daily_totals[daily_totals == 0] = np.nan 
    M_normalized = M_hist / daily_totals

    # B. Derive Master Shape (Median of percentages)
    # Shape: (390,)
    shape_curve = np.nanmedian(M_normalized, axis=0)
    
    # Fill gaps in shape_curve (if a specific minute never had volume in history)
    # using simple interpolation just for the curve
    shape_series = pd.Series(shape_curve).interpolate(limit_direction='both')
    shape_curve = shape_series.values

    # C. Calculate Today's Scale (Projected Total Volume)
    valid_mask = ~np.isnan(today_vol_vector) & (today_vol_vector > 0)
    
    if np.sum(valid_mask) > 5: # Require at least 5 minutes of data to project
        current_vol_sum = np.sum(today_vol_vector[valid_mask])
        expected_pct_sum = np.sum(shape_curve[valid_mask])
        
        if expected_pct_sum > 0.01: # Avoid instability if pct sum is tiny
            projected_total_vol = current_vol_sum / expected_pct_sum
        else:
            projected_total_vol = np.nanmedian(daily_totals)
    else:
        # Not enough data today? Assume average day.
        projected_total_vol = np.nanmedian(daily_totals)

    # D. Execute Fill
    filled_volume = today_vol_vector.copy()
    nan_mask = np.isnan(today_vol_vector)
    
    # Fill = (Percent Pattern) * (Projected Daily Total)
    fill_values = shape_curve * projected_total_vol
    filled_volume[nan_mask] = fill_values[nan_mask]

    # ---------------------------------------------------------
    # Final Series Reconstruction
    # ---------------------------------------------------------
    
    # Reconstruct the index for Today
    t0 = pd.Timestamp(f"{day} 13:30", tz="UTC") # 9:30 ET
    t1 = pd.Timestamp(f"{day} 20:00", tz="UTC") # 4:00 ET (Exclusive usually, but range handles it)
    full_idx = pd.date_range(start=t0, end=t1, freq='1min', inclusive='left')[:390]

    # Price: Simple Forward Fill (Martingale)
    # We fetch the specific prices for today again to ensure clean reindexing
    df_today = df[df['date_str'] == today_str].set_index('ts_utc').copy()
    
    # Reindex Price (puts NaNs where minutes are missing)
    price_series = df_today['close_price'].reindex(full_idx)
    # Apply Limit to ffill (Engineering Safety)
    price_series = price_series.ffill(limit=30) 
    
    # Volume: Assign our calculated vector
    volume_series = pd.Series(data=filled_volume, index=full_idx)

    return price_series, volume_series


def plot_market_data(price_series, volume_series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(price_series.index, price_series, color='blue', linewidth=1.5, label='Close Price')
    ax1.set_title('Underlying Price (Forward Filled)', fontsize=12)
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    ax2.fill_between(volume_series.index, volume_series, color='darkorange', alpha=0.6, label='Volume')
    ax2.set_title('Volume Profile (Relative Scaled)', fontsize=12)
    ax2.set_ylabel('Shares')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# BAW Mode
def _d1(S, K, T, r, q, sigma):
    # Calculates d1 for BS/BAW
    if sigma == 0 or T <= 0: return float('inf') if S > K else -float('inf')
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_put_price(S, K, T, r, q, sigma):
    # Standard European Put Price
    d_1 = _d1(S, K, T, r, q, sigma)
    d_2 = d_1 - sigma * math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    return K * df_r * _norm_cdf(-d_2) - S * df_q * _norm_cdf(-d_1)

# -------------------- 2. BAW Components --------------------

def _m2_put(r, q, sigma):
    # Calculates the exponent m2
    sigma2 = sigma**2
    r_minus_q_over_sigma2 = (r - q) / sigma2
    
    m2 = 0.5 - r_minus_q_over_sigma2 - math.sqrt( (r_minus_q_over_sigma2 - 0.5)**2 + 2 * r / sigma2 )
    return m2


def _put_delta(S, K, T, r, q, sigma):
    d_1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    return df_q * (_norm_cdf(d_1) - 1.0)

def bs_call_price(S, K, T, r, q, sigma):
    d_1 = _d1(S, K, T, r, q, sigma)
    d_2 = d_1 - sigma * math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    return S * df_q * _norm_cdf(d_1) - K * df_r * _norm_cdf(d_2)


def _m1_call(r, q, sigma):
    # Calculates the exponent m1 (Positive Root for Calls)
    sigma2 = sigma**2
    r_minus_q_over_sigma2 = (r - q) / sigma2
    
    m1 = 0.5 - r_minus_q_over_sigma2 + math.sqrt( (r_minus_q_over_sigma2 - 0.5)**2 + 2 * r / sigma2 )
    return m1




def _s_star_residual_put(S_star, K, T, r, q, sigma, m2):
    # Residual function for Put: K - S* - P(S*) - (1/m2)*S* * dP/dS = 0
    if S_star <= 0: return 1e18
    P = bs_put_price(S_star, K, T, r, q, sigma)
    Delta = _put_delta(S_star, K, T, r, q, sigma)
    return K - S_star - P - (1.0 / m2) * S_star * Delta

def _s_star_residual_call(S_star, K, T, r, q, sigma, m1):
    # Residual function for Call: (S* - K) - C(S*) - (S*/m1) * (1 - Delta_Call) = 0
    if S_star <= 0: return 1e18
    
    C = bs_call_price(S_star, K, T, r, q, sigma)
    
    # Delta for Call = e^(-qT) * N(d1)
    d_1 = _d1(S_star, K, T, r, q, sigma)
    Delta_Call = math.exp(-q * T) * _norm_cdf(d_1)
    
    return (S_star - K) - C - (S_star / m1) * (1.0 - Delta_Call)






def get_company_id(engine: Engine, ticker: str) -> Optional[int]:
    """Fetches the company's primary key ID from the database using its ticker."""
    sql = text("SELECT id FROM companies WHERE ticker = :ticker_symbol")
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"ticker_symbol": ticker.upper()}).fetchone()
            if result:
                return int(result[0])
            print(f"[ERROR] Ticker '{ticker}' not found.")
            return None
    except Exception as e:
        print(f"[ERROR] DB error (get_company_id for {ticker}): {e}")
        return None

def _get_sod(engine: Engine, ticker_id: int, d: date) -> Optional[float]:
    """Fetches the last known close price *before* the given date for SOD anchor."""
    try:
        start_ts_unix = int(pd.Timestamp(d, tz="UTC").timestamp())
    except Exception as e:
        print(f"[ERROR] Invalid date in _get_sod: {d}. {e}")
        return None

    sql = text("""
        SELECT close_price FROM quotes
        WHERE company_id = :cid AND time_entry_ts < :ts_start
        ORDER BY time_entry_ts DESC LIMIT 1
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"cid": ticker_id, "ts_start": start_ts_unix}).scalar_one_or_none()
            return float(result) if result is not None else None
    except Exception as e:
        print(f"[ERROR] DB error (_get_sod for CID={ticker_id} on {d}): {e}")
        return None
def _get_rate(engine: Engine, d: date) -> Optional[float]:
    """Fetches the last known risk-free rate *before* the given date for the time loop."""
    try:
        # ORIGINAL: start_ts_unix = int(pd.Timestamp(d, tz="UTC").timestamp())
        # FIX: The input 'd' is already timezone-aware (e.g., 2025-08-28 13:30:00+00:00).
        # Passing 'tz="UTC"' when 'd' already has tzinfo causes the error.
        # We can just let pd.Timestamp take the tz-aware object and extract the timestamp.
        start_ts_unix = int(pd.Timestamp(d).timestamp())
    except Exception as e:
        # The error message should probably reflect that this is for the rate, not SOD.
        print(f"[ERROR] Invalid date in _get_rate: {d}. {e}")
        return None

    sql = text("""
        SELECT risk_free_rate_r FROM federal_funds_rate
        WHERE time_entry_ts < :ts_start
        ORDER BY time_entry_ts DESC LIMIT 1
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, { "ts_start": start_ts_unix}).scalar_one_or_none()
            return float(result) if result is not None else None
    except Exception as e:
        print(f"[ERROR] DB error (_get_rate on {d}): {e}")
        return None
def _business_days(start_d: str, end_d: str) -> List[date]:
    return list(pd.date_range(start_d, end_d, freq="B").date)



def brentq_american_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:

    EuPrice = bs_call_price(S, K, T, r, q, sigma)
    if q <= 1e-9:
        return EuPrice
    if EuPrice < (S - K):
            # This usually implies deep ITM + High Div, boundary crossed
            return S - K

    # 3. Solve for S* (Critical Price)
    m1 = _m1_call(r, q, sigma)

    S_high = max(K, S) * 2.0
    MAX_BOUND = max(K, S) * 10.0
    
    def safe_residual_call(S_val):
        if S_val <= K: return -1e9 # Should be negative here (LHS < RHS usually)
        return _s_star_residual_call(S_val, K, T, r, q, sigma, m1)

    # Adaptive Bracket Expansion for Call
    while True:
        res_low = safe_residual_call(K)
        res_high = safe_residual_call(S_high)
        
        if res_low * res_high < 0:
            break 
        
        if S_high >= MAX_BOUND:
            # Failed to bracket, likely S* is infinity or European is optimal
            return EuPrice
        
        S_high *= 2.0
    S_star = brentq(safe_residual_call, K, S_high, xtol=1e-5)
    


    # 4. Final Calculation
    if S >= S_star:
        return S - K
    else:
        d1_star = _d1(S_star, K, T, r, q, sigma)
        Delta_star = math.exp(-q*T) * _norm_cdf(d1_star)
        A1 = (S_star / m1) * (1.0 - Delta_star)
        return EuPrice + A1 * (S / S_star)**m1

def _objective_func_am_call(sigma, market_price, S, K, T, r, q):

    model_price = brentq_american_call(S, K, T, r, q, sigma) 
    
    return model_price - market_price


def calculate_iv_american_call(market_price: float, S: float, K: float, T: float, r: float, q: float, time_entry_ts: int) -> float:
    intrinsic = max(S - K, 0)
    # fast check without brentq solver
    if market_price < intrinsic:
        # MARKET SENTIMENT: its not worth holding anymore, just exercise it, we will throw in a little reward for not cluttering the market manifold.
        return 1e-4
    low_vol, high_vol = 0.001, 10.0 
  
    # 2. Check Bracket Integrity (Need opposing signs)
    f_low = _objective_func_am_call(low_vol, market_price, S, K, T, r, q)
    f_high = _objective_func_am_call(high_vol, market_price, S, K, T, r, q)
    
    # If both model prices are below market price (f < 0), the IV is > 3.0.
    # If both model prices are above market price (f > 0), the IV is < 0.001 or arbitrage.
    # slow check without brentq solver
    # slow bretnq check
    if f_low * f_high > 0:
        # Scenario 1: IV too LOW (f_low and f_high are both POSITIVE)
        if f_low > 0: 
            return 0.0001 # Cap to epsilon
            
        # Scenario 2: IV too HIGH (f_low and f_high are both NEGATIVE)
        elif f_low < 0:
            print(f"[FAIL] IV too high (> {high_vol}). Cannot converge.")
            return np.nan # Flag as non-convergent
    
    # 3. Solve for the root (IV)
    iv = brentq(
        _objective_func_am_call,
        low_vol,
        high_vol,
        args=(market_price, S, K, T, r, q)
    )
    return iv

def brentq_american_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:

    EuPrice = bs_put_price(S, K, T, r, q, sigma)

    m2 = _m2_put(r, q, sigma)
    
    # 2. Solve for S* (Critical Price < K)
    try:
        S_low = K * 0.1 
        S_high = K 

        def safe_residual_put(S_val):
            if S_val <= 1e-6: return 1e18
            return _s_star_residual_put(S_val, K, T, r, q, sigma, m2)

        # Adaptive Bracket Expansion for Put
        while True:
            res_low = safe_residual_put(S_low)
            res_high = safe_residual_put(S_high)
            
            if res_low * res_high < 0:
                break 
            
            # Check convergence failure
            if S_low < 1e-6:
                # Bounds collapsed to 0
                return EuPrice

            S_low /= 2.0
            
        S_star = brentq(safe_residual_put, S_low, S_high, xtol=1e-5)
        
    except (ValueError, RuntimeError):
        return EuPrice
    Premium_at_S_star = (K - S_star) - bs_put_price(S_star, K, T, r, q, sigma) 
        

    S_star_safe = max(S_star, 1e-5) # Set S_star to a tiny positive number if it's near zero
    
    # Check if S_star was too small (and treat it as European price)
    if S_star <= 1e-5: 
        return EuPrice
    
    # Use S_star_safe for the power operation to prevent ZeroDivisionError/Overflow
    # A2_coefficient = Premium_at_S_star / (S_star**m2) # Original, dangerous line
    A2_coefficient = Premium_at_S_star / max((S_star_safe**m2), 1e-5)
    
    if S <= S_star:
        # Region 1: Exercise Region (S at or below critical price)
        return K - S # Return Intrinsic Value
    else:
        # Region 2: Continuation Region (S > S*)
        # Use the original S_star for the early exercise premium calculation if S_star_safe was used
        Early_Exercise_Premium = A2_coefficient * (S**m2) 
        return EuPrice + Early_Exercise_Premium


def _objective_func_am_put(sigma, market_price, S, K, T, r, q):

        
    # Calls the theoretical pricing function with the guessed sigma
    model_price = brentq_american_put(S, K, T, r, q, sigma) 
    
    return model_price - market_price

def calculate_iv_american_put(market_price: float, S: float, K: float, T: float, r: float, q: float, time_entry_ts: int)  -> float:
    intrinsic = max(K - S, 0)
    time_str = time_entry_ts.strftime('%Y-%m-%d %H:%M:%S')
    # quick check before brentq
    if market_price < intrinsic:
        # Market sentiment: just exercise it, not worth trading anymore.
        return 1e-4
    
    low_vol, high_vol = 0.001,  10.0 
    

    # 2. Check Bracket Integrity
    f_low = _objective_func_am_put(low_vol, market_price, S, K, T, r, q)
    f_high = _objective_func_am_put(high_vol, market_price, S, K, T, r, q)
    
    # slow bretnq check
    if f_low * f_high > 0:
        # Scenario 1: IV too LOW (f_low and f_high are both POSITIVE)
        if f_low > 0: 
            return 0.0001 # Correct: Cap to epsilon
            
        # Scenario 2: IV too HIGH (f_low and f_high are both NEGATIVE)
        elif f_low < 0:
            print(f"[FAIL] IV too high (> {high_vol}). Cannot converge.")
            return np.nan # Correct: Flag as non-convergent

    # 3. Solve for the root (IV)
    iv = brentq(
        _objective_func_am_put,
        low_vol,
        high_vol,
        args=(market_price, S, K, T, r, q)
    )
    return iv


