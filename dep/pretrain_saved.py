
import argparse
from datetime import date, datetime, timedelta
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine  
from scipy.optimize import minimize
from multiprocessing import Pool, Manager
from pretrain_helper import _business_days, _d1, _get_rate, _get_sod, _m2_put, _norm_cdf,  _put_delta, bs_put_price, get_company_id, load_minutes, load_underlying, brentq_american_price


from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
load_dotenv()

SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)

class SurfaceKalmanTracker:
    def __init__(self):
        # State: {expiry_date: {'x': vector, 'P': covariance_matrix, 'last_seen': timestamp}}
        self.sabr_states = {} 
        self.vol_states = {}
        self.txn_states = {}
        
        # Remove hardcoded 3x3 matrices here
        self.decay = 0.99        

    def update_sabr(self, expiry, raw_params, dim=3, has_data=True):
        return self._run_filter(self.sabr_states, expiry, raw_params, dim, has_data)

    def update_volume(self, expiry, raw_params, has_data=True):
        return self._run_filter(self.vol_states, expiry, raw_params, 3, has_data)
    
    def update_transactions(self, expiry, raw_params, has_data=True):
        return self._run_filter(self.txn_states, expiry, raw_params, 3, has_data)

    def _run_filter(self, state_dict, expiry, z_measurement, dim, has_data):
        # Initialize if new expiry
        if expiry not in state_dict:
            state_dict[expiry] = {
                'x': np.zeros(dim), 
                'P': np.eye(dim) * 1.0
            }
            if has_data: state_dict[expiry]['x'] = z_measurement

        state = state_dict[expiry]
        x, P = state['x'], state['P']

        # --- FIX START ---
        # Create dynamic Noise Matrices based on 'dim'
        # 12 for SABR, 3 for Volume/Txn
        Q = np.eye(dim) * 0.01 
        R = np.eye(dim) * 0.1
        # --- FIX END ---

        # 1. Predict (Time Evolution)
        x_pred = x 
        P_pred = P + Q  # Use dynamic Q, not self.Q_sabr

        if not has_data:
            state['x'] = x_pred
            state['P'] = P_pred
            return x_pred

        # 2. Update (Measurement Correction)
        y = z_measurement - x_pred 
        S = P_pred + R   # Use dynamic R, not self.R_sabr
        K = P_pred @ np.linalg.inv(S) 
        
        x_new = x_pred + K @ y
        P_new = (np.eye(dim) - K) @ P_pred

        # Store
        state['x'] = x_new
        state['P'] = P_new
        return x_new

def _hagan_vol(k, F, T, alpha, beta, rho, nu):
    """Standard Hagan 2002 SABR Log-Normal Volatility."""
    # Safety rails
    if F <= 0 or k <= 0 or T <= 0: return 0.0
    if alpha <= 0 or nu <= 0: return 0.0

    log_fk = np.log(F / k)
    fk_beta = (F * k)**((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk

    if abs(z) < 1e-5:
        x_z = 1.0
    else:
        # Term inside sqrt must be positive
        sq_term = 1 - 2 * rho * z + z**2
        if sq_term < 0: sq_term = 0
        chi = np.log((np.sqrt(sq_term) + z - rho) / (1 - rho))
        x_z = chi / z

    # Expansion terms
    term1 = alpha / (fk_beta * (1 + ((1 - beta)**2 / 24) * log_fk**2 + ((1 - beta)**4 / 1920) * log_fk**4))
    
    # Adjust for x(z)
    if abs(z) >= 1e-5:
        term1 = term1 * (z / x_z) # Note: Hagan formula uses z/chi

    term2 = 1 + (((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) +
                 (0.25 * rho * beta * nu * alpha) / (F**(1 - beta)) +
                 ((2 - 3 * rho**2) / 24) * nu**2) * T

    return term1 * term2

def get_sabr_from_poly(coeffs, T):
    """
    Decodes the 12-vector into Alpha, Rho, Nu for a specific T.
    coeffs: [a0, a1, a2, a3,  r0, r1, r2, r3,  n0, n1, n2, n3]
    Polynomial: p(T) = c0 + c1*T + c2*T^2 + c3*T^3
    """
    # Extract chunks
    a_coeffs = coeffs[0:4]
    r_coeffs = coeffs[4:8]
    n_coeffs = coeffs[8:12]
    
    # Calculate values (Horners method or dot product)
    # Using T is safer than log(T) for polynomial stability near 0, 
    # but sqrt(T) is often best for vol. Let's stick to T based on your request.
    
    T_vec = np.array([1.0, T, T**2, T**3])
    
    alpha = np.dot(a_coeffs, T_vec)
    rho   = np.dot(r_coeffs, T_vec)
    nu    = np.dot(n_coeffs, T_vec)
    
    return alpha, rho, nu

def fit_polynomial_sabr_surface(all_options_data, prev_coeffs=None):
    """
    Fits 12 coefficients to the entire cloud of options data at once.
    all_options_data: List of tuples (strike, implied_vol, F, T)
    """
    if not all_options_data:
        return np.zeros(12)

    # 1. Objective Function
    def objective(coeffs):
        total_error = 0.0
        penalty = 0.0
        
        # Iterate through every single option in this minute
        for (k, v_mkt, F, T) in all_options_data:
            
            # Calculate SABR params at this specific T using the polynomials
            alpha, rho, nu = get_sabr_from_poly(coeffs, T)
            
            # --- Constraints / Penalties ---
            # 1. Rho must be between -1 and 1
            if abs(rho) >= 0.999: 
                penalty += 1000.0 * (abs(rho) - 0.999)**2
                rho = np.clip(rho, -0.999, 0.999) # Clip for calculation safety
                
            # 2. Alpha and Nu must be positive
            if alpha <= 0.001:
                penalty += 1000.0 * (0.001 - alpha)**2
                alpha = 0.001
            if nu <= 0.001:
                penalty += 1000.0 * (0.001 - nu)**2
                nu = 0.001
                
            # Calculate Model Vol
            v_model = _hagan_vol(k, F, T, alpha, 1.0, rho, nu)
            
            # Squared Error
            total_error += (v_model - v_mkt)**2
            
        # Regularization (optional): Penalize high-order terms (a3, r3, etc) 
        # to prevent "wobbly" surfaces if data is sparse.
        # coeffs indices: 3, 7, 11 are the cubic terms
        penalty += 0.1 * (coeffs[3]**2 + coeffs[7]**2 + coeffs[11]**2)
            
        return total_error + penalty

    # 2. Initial Guess
    # If we have previous Kalman state, use it. 
    # Otherwise, initialize as a Flat Surface:
    # Alpha = 0.5 (a0=0.5, others=0), Rho=0, Nu=0.5
    if prev_coeffs is not None and len(prev_coeffs) == 12:
        x0 = prev_coeffs
    else:
        x0 = np.zeros(12)
        x0[0] = 0.5 # Alpha intercept
        x0[8] = 0.5 # Nu intercept
        
    # 3. Optimize
    # We use unconstrained fitting (BFGS) because we handled constraints via penalties
    try:
        res = minimize(objective, x0, method='L-BFGS-B', tol=1e-4)
        return res.x
    except:
        return x0

def fit_liquidity_gaussian(log_moneyness, volume):
    """Fits [Amp, Mu, Sigma] to Volume/Transactions."""
    if len(volume) < 2 or np.sum(volume) == 0:
        return np.zeros(3)
    total_vol = np.sum(volume)
    mu = np.average(log_moneyness, weights=volume)
    sigma = np.sqrt(np.average((log_moneyness - mu)**2, weights=volume))
    amp = np.max(volume)
    
    return np.array([amp, mu, sigma])

def project_parameters_to_grid(k_grid, sabr_params, vol_params, txn_params, T_expiry):
    """Expands 3 parameters back into 60 grid points using ACTUAL Time to Expiry."""
    # 1. SABR Projection
    alpha, rho, nu = sabr_params
    if alpha == 0: 
        iv_grid = np.zeros_like(k_grid)
    else:
        K_eff = np.exp(k_grid)
        iv_grid = np.array([_hagan_vol(k_val, 1.0, T_expiry, alpha, 1.0, rho, nu) for k_val in K_eff])

    amp, mu, sig = vol_params
    if amp == 0 or sig == 0:
        vol_grid = np.zeros_like(k_grid)
    else:
        vol_grid = amp * np.exp(-0.5 * ((k_grid - mu) / sig)**2)
    amp_t, mu_t, sig_t = txn_params
    if amp_t == 0 or sig_t == 0:
        txn_grid = np.zeros_like(k_grid)
    else:
        txn_grid = amp_t * np.exp(-0.5 * ((k_grid - mu_t) / sig_t)**2)
        
    return iv_grid, vol_grid, txn_grid


def _process_single_day(day_data):
    day_obj, day_dfm_map, day_minutes_map, \
    day_minute_prices_map, day_minute_vol_map, \
    q_div, ticker, db_url = day_data
    day_minutes = day_minutes_map[day_obj] 
    day_df = day_dfm_map[day_obj]
    
    tracker = SurfaceKalmanTracker()

    local_engine = create_engine(db_url, pool_pre_ping=True)

    processed_surfaces = []
    day_str = day_obj.strftime('%Y-%m-%d')
    print(f"  - Processing {day_str}...")
    r_rate = _get_rate(engine=local_engine, d=day_minutes[0])
    start_loop = int(datetime.now().timestamp())

    day_df["ts_min"] = day_df["ts_utc"].dt.floor('min')

    cols = ["ts_min", "strike", "dte", "option_close", "option_volume_weighted", "option_volume", "option_transactions", "contract_type"]
    base_day_grouped = day_df[cols].groupby("ts_min")
    C_KEY = "Call_Poly_State"
    P_KEY = "Put_Poly_State"
    # Time Loop
    for ts in day_minutes:
        S_t = day_minute_prices_map[day_obj].get(ts)
        S_vol_t = day_minute_vol_map[day_obj].get(ts)
        
        # Skip if underlying price is missing (gap > limit)
        if pd.isna(S_t) or S_t <= 0: continue

        # Get Option Slice
        sub = base_day_grouped.get_group(ts) if ts in base_day_grouped.groups else pd.DataFrame()

        joint_call_data = [] # List of (strike, iv, F, T)
        joint_put_data = []
        
        # RAW Lists for storage
        minute_market_c = []
        minute_market_p = []

        unique_dtes = sub['dte'].unique()
        for dte in unique_dtes:
            if dte < 1: continue
            T_expiry = dte / 365.0
            chain = sub[sub['dte'] == dte]
            
            # Process Calls
            calls = chain[chain['contract_type'] == 'C']
            if not calls.empty:
                # Store Raw
                raw_records = calls[['strike', 'option_volume_weighted', 'option_volume']].to_dict('records')
                for r in raw_records: r['dte'] = int(dte)
                minute_market_c.extend(raw_records)
                
                # Store for Optimization
                # (k, v, F, T)
                for row in calls.itertuples():
                    joint_call_data.append((row.strike, row.option_volume_weighted, S_t, T_expiry))

            # Process Puts
            puts = chain[chain['contract_type'] == 'P']
            if not puts.empty:
                raw_records = puts[['strike', 'option_volume_weighted', 'option_volume']].to_dict('records')
                for r in raw_records: r['dte'] = int(dte)
                minute_market_p.extend(raw_records)
                
                for row in puts.itertuples():
                    joint_put_data.append((row.strike, row.option_volume_weighted, S_t, T_expiry))


        # 2. FIT & KALMAN UPDATE (CALLS)
        # Get previous state (12-vector)
        prev_c = tracker.sabr_states.get(C_KEY, {}).get('x', None)
        
        # Fit the 12 polynomials to the cloud
        if len(joint_call_data) > 5: # Ensure we have enough data points
            raw_coeffs_c = fit_polynomial_sabr_surface(joint_call_data, prev_c)
            # Kalman Update (Dim=12)
            smooth_coeffs_c = tracker.update_sabr(C_KEY, raw_coeffs_c, dim=12, has_data=True)
        else:
            # Not enough data? Predict step only (or decay)
            smooth_coeffs_c = tracker.update_sabr(C_KEY, np.zeros(12), dim=12, has_data=False)


        # 3. FIT & KALMAN UPDATE (PUTS)
        prev_p = tracker.sabr_states.get(P_KEY, {}).get('x', None)
        if len(joint_put_data) > 5:
            raw_coeffs_p = fit_polynomial_sabr_surface(joint_put_data, prev_p)
            smooth_coeffs_p = tracker.update_sabr(P_KEY, raw_coeffs_p, dim=12, has_data=True)
        else:
            smooth_coeffs_p = tracker.update_sabr(P_KEY, np.zeros(12), dim=12, has_data=False)

        # 4. Append Snapshot
        processed_surfaces.append({
            'time_entry_ts': ts,
            'ticker': ticker,
            'price_ffill_S': S_t,
            'volume_profile_S': S_vol_t, 
            'risk_free_rate_r': r_rate,
            'market_C': minute_market_c, 
            'market_P': minute_market_p,
            'SABR_IV_Kalman_C': smooth_coeffs_c.tolist(), 
            'SABR_IV_Kalman_P': smooth_coeffs_p.tolist(),
        })

    end_loop = int(datetime.now().timestamp())
    print(f"compute loop for {day_str} took: {end_loop- start_loop} seconds")
    return processed_surfaces

def build_option_items(engine: Engine, ticker: str, start_day: str, end_day: str,  dte_max: int, k_pct: float,  q_div = 0.0, num_workers=10):
    processed_surfaces = []
    days_to_process = _business_days(start_day, end_day)
    all_filtered_dfs = []
    
    # Maps to hold data for workers
    day_sod_map = {}
    day_minutes_map = {}
    day_dfm_map = {}
    day_minute_prices_map = {} 
    day_minute_vol_map = {} 

    print(f"--- Loading data for {ticker} from {start_day} to {end_day} ---")

    ticker_id = get_company_id(engine, ticker)
    for day_obj in days_to_process:
   
        day_str = day_obj.strftime('%Y-%m-%d')
        
        S = _get_sod(engine, ticker_id, day_obj)
        if not S or S <= 0:
            print(f"[WARN] No SOD price for {day_str}, skipping day.")
            continue
            
        dfm = load_minutes(engine, ticker, day_str, dte_max)
        if dfm.empty:
            print(f"[WARN] No options data for {day_str}, skipping day.")
            continue
        
        vol_profil_days = 20 
        underlying_price_series, underlying_volume_series = load_underlying(engine, day_str, ticker_id, vol_profil_days)

        t0 = pd.Timestamp(f"{day_str} 13:30", tz="UTC") # 9:30 AM ET
        t1 = pd.Timestamp(f"{day_str} 20:00", tz="UTC") # 4:00 PM ET
        
        # Filter Market Hours
        dfm = dfm[(dfm["ts_utc"] >= t0) & (dfm["ts_utc"] <= t1)]
        dfm["strike"] = pd.to_numeric(dfm["strike"], errors="coerce")
        dfm = dfm.dropna(subset=["strike"])
        
        dfm_filtered = dfm[(np.abs(np.log(dfm["strike"].astype(float) / float(S))) <= k_pct)] 
        

        dfm_filtered = dfm_filtered[dfm_filtered["dte"].between(1, dte_max)]


        dfm_filtered = dfm_filtered[dfm_filtered['option_volume_weighted'] > 0.05]

 
        if dfm_filtered.empty:
            print(f"[WARN] No options data remaining after OTM filtering for {day_str}, skipping.")
            continue
            
        print(f"[INFO] Loaded {len(dfm_filtered)} OTM rows for {day_str} with SOD={S:.2f}")
        
        all_filtered_dfs.append(dfm_filtered)
        
        # Store Data in Maps
        day_sod_map[day_obj] = S
        day_minutes_map[day_obj] = pd.date_range(t0, t1, freq="1min", inclusive="both")
        day_dfm_map[day_obj] = dfm_filtered
        day_minute_prices_map[day_obj] = underlying_price_series
        day_minute_vol_map[day_obj] = underlying_volume_series 
    db_url = SQLALCHEMY_DATABASE_URL 
    
    day_keys = sorted(day_sod_map.keys())
    
    tasks = [
        (
            day_obj, 
            day_dfm_map,          
            day_minutes_map,      
            day_minute_prices_map,
            day_minute_vol_map,   
            q_div, 
            ticker, 
            db_url
        )
        for day_obj in day_keys
    ]

    all_day_results = []
    with Pool(processes=num_workers) as pool:
        all_day_results = pool.map(_process_single_day, tasks)
    
    for day_result in all_day_results:
        processed_surfaces.extend(day_result)

    return pd.DataFrame(processed_surfaces)



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("params_file", help="")

    args = parser.parse_args()
    params_file_path = args.params_file

    with open(params_file_path, 'r') as f:
        params = json.load(f)
    ticker = params["ticker"][0]
    dividend_rates = params["dividend_rates"][ticker]
    default_start_day = params["train_start"]
    default_end_day = params["train_end"]
    default_max_dte = params.get("max_dte", 30)
    k_pct=params.get("k_pct", .1)
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

    df = build_option_items( 
        engine, 
        ticker, 
        default_start_day, 
        default_end_day,    
        default_max_dte, 
        k_pct=k_pct, 
        q_div= dividend_rates
    )

    k_pct_str = f"{k_pct:.2f}" # e.g., '0.100'
    q_div_str = f"{dividend_rates:.4f}"

    filename = (
        f"{ticker}"
        f"_kpct{k_pct_str}"
        f"_dte{default_max_dte}"
        f"_from_{default_start_day}"
        f"_to_{default_end_day}"
        f"_qdiv{q_div_str}"
        f".parquet"
    )

    filename = filename.replace(":", "-") 
    filename = filename.replace(" ", "_") 

    df.to_parquet(filename)

if __name__ == '__main__':
    main()


import argparse
from datetime import date, datetime, timedelta
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine  
from multiprocessing import Pool, Manager
from pretrain_helper import _business_days, _get_rate, _get_sod, get_company_id, load_minutes, load_underlying, calculate_iv_american_call, calculate_iv_american_put
from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
import warnings
from scipy.optimize import minimize
import math


load_dotenv()

SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)
# BAW Mode


def _hagan_vol(k, F, T, alpha, beta, rho, nu):
    """Standard Hagan 2002 SABR Log-Normal Volatility with higher-order expansion."""
    # Safety rails
    if F <= 0 or k <= 0 or T <= 0: return 0.0
    if alpha <= 0 or nu <= 0: return 0.0

    log_fk = np.log(F / k)
    fk_beta = (F * k)**((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk

    # --- 1. Calculate z / chi(z) ---
    if abs(z) < 1e-5:
        # ATM case (K approx F): z/chi(z) approx 1
        z_over_chi = 1.0 
    elif rho == 1.0:
        # Special case: rho = 1
        z_over_chi = 1.0 / (1.0 + z)
    else:
        # General case (z != 0, rho != 1)
        sq_term = 1 - 2 * rho * z + z**2
        if sq_term < 0: sq_term = 0.0 # Safety clip for sqrt argument
        
        # FIX APPLIED HERE: Add a small epsilon to the log argument to guarantee positivity
        log_arg = (np.sqrt(sq_term) + z - rho) / (1 - rho)
        
        # Ensure log_arg is strictly positive before taking the log
        log_arg_safe = np.clip(log_arg, 1e-18, None) 
        
        chi = np.log(log_arg_safe)
        
        # Guard against zero-division for chi near zero
        if abs(chi) < 1e-10:
            z_over_chi = 1.0
        else:
            z_over_chi = z / chi

    # --- 2. Calculate Volatility at T=0 (vol_at_T0) ---
    base_vol_component = alpha / fk_beta
    
    if abs(z) < 1e-5:
        # NEAR-ATM Case: Use Taylor expansion for log-moneyness (log_fk)
        
        # *** FIX for higher-order term added here (1/1920) ***
        moneyness_adjustment = 1 / (
            1 + 
            ((1 - beta)**2 / 24) * log_fk**2 +
            ((1 - beta)**4 / 1920) * log_fk**4  # Added the quartic term
        )
        vol_at_T0 = base_vol_component * moneyness_adjustment
    else:
        # Off-ATM Case: Use z/chi adjustment
        vol_at_T0 = base_vol_component * z_over_chi
        
    # --- 3. Calculate T-Expansion Term (term2) ---
    term2 = 1 + (
        ((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) +
        (0.25 * rho * beta * nu * alpha) / (F**(1 - beta)) +
        ((2 - 3 * rho**2) / 24) * nu**2
    ) * T

    return vol_at_T0 * term2



def get_sabr_from_poly(coeffs, T):
    """
    Decodes the 12-vector into Alpha, Rho, Nu for a specific T.
    coeffs: [a0, a1, a2, a3,  r0, r1, r2, r3,  n0, n1, n2, n3]
    Polynomial: p(T) = c0 + c1*T + c2*T^2 + c3*T^3
    """
    # Extract chunks
    a_coeffs = coeffs[0:4]
    r_coeffs = coeffs[4:8]
    n_coeffs = coeffs[8:12]
    
    # Calculate values (Horners method or dot product)
    # Using T is safer than log(T) for polynomial stability near 0, 
    # but sqrt(T) is often best for vol. Let's stick to T based on your request.
    
    T_safe = np.clip(T, 1e-6, T) # Ensure T is not exactly zero
        
    T_half = np.sqrt(T_safe)
    # New Time Vector: [1, T^(1/2), T^1, T^(3/2)] (Numerically superior to [1, T, T^2, T^3])
    T_vec = np.array([1.0, T_half, T_safe, T_safe * T_half]) 
    
    alpha = np.dot(a_coeffs, T_vec)
    rho   = np.dot(r_coeffs, T_vec)
    nu    = np.dot(n_coeffs, T_vec)
    
    return alpha, rho, nu

def fit_polynomial_sabr_surface(all_options_data):
    """
    Fits 12 coefficients to the entire cloud of options data at once.
    all_options_data: List of tuples (strike, implied_vol, S, T, Price)
    """
    if not all_options_data:
        return np.zeros(12)

    # 1. Objective Function
    def objective(coeffs):
        total_error = 0.0
        penalty = 0.0
        
        # We still keep mild penalties for internal consistency (Rho bounds at specific T),
        # but the main constraints are handled by the solver bounds.
        
        for (K, v_mkt, S_t, T, market_price) in all_options_data: 
            
            F = S_t
            
            # Calculate SABR params at this specific T
            alpha, rho, nu = get_sabr_from_poly(coeffs, T)
            
            # --- Soft Constraints for Time-Specific Violations ---
            # Even if coefficients are bounded, the resulting polynomial at large T 
            # might drift. We penalize the *result* here.
            
            # 1. Rho must be between -0.999 and 0.999
            if abs(rho) >= 0.999: 
                penalty += 1000.0 * (abs(rho) - 0.999)**2 
                rho = np.clip(rho, -0.999, 0.999)
                
            # 2. Alpha and Nu must be positive
            if alpha <= 0.001:
                penalty += 1000.0 * (0.001 - alpha)**2
                alpha = 0.001
            if nu <= 0.001:
                penalty += 1000.0 * (0.001 - nu)**2
                nu = 0.001
                
            # Calculate Model Vol
            # Ensure T is clipped
            v_model = _hagan_vol(K, F, max(T, 1e-6), alpha, 1.0, rho, nu) 
            
            # Error weighting: Weight by 1/sqrt(T) to prioritize short-term structure
            weight = 1.0 / np.sqrt(max(T, 0.1))
            
            diff = (v_model - v_mkt) * 100.0 
            total_error += (diff**2) * weight
                
        # Regularization to prevent higher-order polynomial terms from exploding
        # We penalize the cubic/quadratic terms (indices 3, 7, 11 are cubic)
        reg_penalty = 0.01 * np.sum(coeffs[1:4]**2 + coeffs[5:8]**2 + coeffs[9:12]**2)
        
        return (total_error / len(all_options_data)) + penalty + reg_penalty

    # 2. Initial Guess
    x0 = np.zeros(12)
    # Intercepts (The base value at T=0)
    x0[0] = 0.3   # Alpha Intercept
    x0[4] = -0.3  # Rho Intercept
    x0[8] = 0.8   # Nu Intercept (Start high to force curvature detection)

    # 3. Define Bounds for SLSQP
    # We allow the intercepts to be strict, and the slopes to be loose but finite.
    # formatting: (min, max)
    
    # Alpha Coeffs [a0, a1, a2, a3] -> Alpha(T)
    b_alpha = [(0.01, 5.0), (-5, 5), (-5, 5), (-5, 5)] 
    
    # Rho Coeffs [r0, r1, r2, r3] -> Rho(T)
    # Intercept limited to [-0.99, 0.99], slopes loose
    b_rho   = [(-0.99, 0.99), (-2, 2), (-2, 2), (-2, 2)]
    
    # Nu Coeffs [n0, n1, n2, n3] -> Nu(T)
    b_nu    = [(0.01, 5.0), (-5, 5), (-5, 5), (-5, 5)]
    
    bounds = b_alpha + b_rho + b_nu

    # 4. Optimize
    try:
        # Pass bounds here!
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, tol=1e-5, options={'maxiter': 200})
        
        if res.success:
            return res.x
        else:
            # If it fails, often the "current" x is better than x0
            # print(f"SABR Fit Warning: {res.message}")
            return res.x 
            
    except Exception as e:
        print(f"SABR Optimization Critical Failure: {e}")
        return x0



def get_ssvi_from_poly(coeffs, T):
    """
    Decodes the 6-vector SSVI coefficients and computes the local T parameters.
    coeffs: [rho, lambda, gamma, c0, c1, c2] (where c_i define theta(T))
    """
    if len(coeffs) != 6:
        raise ValueError("SSVI coefficient vector must have exactly 6 elements.")

    # 1. Global Smile Parameters
    rho = coeffs[0]
    lam = coeffs[1]
    gamma = coeffs[2]
    
    # 2. Term Structure Coefficients
    c0, c1, c2 = coeffs[3], coeffs[4], coeffs[5]

    T_safe = np.clip(T, 1e-6, T) # Ensure T is positive
        
    # 3. Calculate ATM Total Variance (theta_T)
    # Using the standard quadratic basis: theta(T) = c0 + c1*T + c2*T^2
    
    # Ensure theta_T remains positive (Total Variance cannot be negative)
    theta_T = c0 + c1 * T_safe + c2 * T_safe**2
    theta_T = np.maximum(theta_T, 1e-6) 
    return rho, lam, gamma, theta_T


def ssvi_variance_function(k, T, theta_T, rho, lam, gamma):
    phi_T = lam / (theta_T**gamma)
    
    # Clip rho for numerical safety
    rho = np.clip(rho, -0.999, 0.999) 
    
    # Calculate w(k, T)
    term1 = phi_T * k + rho
    sqrt_term = np.sqrt(term1**2 + (1 - rho**2))
    w_kT = (theta_T / 2.0) * (1.0 + rho * phi_T * k + sqrt_term)
    
    return np.maximum(w_kT, 1e-12)

def fit_polynomial_ssvi_surface(all_options_data):
    """
    Fits 6 coefficients to the entire cloud of options data at once.
    all_options_data: List of tuples (strike, implied_vol, S, T, Price)
    """
    if not all_options_data:
        return np.zeros(6) # Fixed: Return size 6, not 12

    def objective(coeffs):
        total_error = 0.0
        penalty = 0.0
        
        # Unpack Global Params
        rho, lam, gamma, c0, c1, c2 = coeffs
        
        # --- PENALTIES ---
        # 1. Term Structure Shape: Penalize negative curvature in time (c2 < 0 is usually bad for stability)
        if c2 < 0:
            penalty += 1000.0 * (c2**2)
            
        # 2. Smile Existence: Penalize if lambda is too small (Force the smile!)
        if lam < 0.01:
             penalty += 10000.0 * (0.01 - lam)**2

        # 3. Correlation boundaries
        if abs(rho) > 0.99:
             penalty += 1000.0 * (abs(rho) - 0.99)**2

        count = 0
        for (K, v_mkt, S_t, T, market_price) in all_options_data: 
            
            F = S_t 
            T_safe = np.clip(T, 1e-4, T) # Clip slightly higher than 1e-6 to avoid instability
            
            # --- 1. Calculate LOCAL theta_T ---
            # Total Variance should generally increase with time.
            theta_T = c0 + c1 * T_safe + c2 * T_safe**2
            theta_T = np.maximum(theta_T, 1e-4)
            
            # --- 2. Calculate Log Moneyness ---
            k = np.log(K / F)
            
            # --- 3. Model Variance ---
            w_model = ssvi_variance_function(k, T_safe, theta_T, rho, lam, gamma)
            
            # --- 4. Market Variance ---
            w_market = (v_mkt ** 2) * T_safe

            # --- WEIGHTING ---
            # Vegas-style weighting: emphasize the ATM and the Wings, 
            # but scale by 1/sqrt(T) so long-dated options don't dominate error.
            weight = 1.0 / np.sqrt(T_safe)
            
            # Calculate error
            diff = (w_model - w_market) * 100.0 
            total_error += (diff**2) * weight
            count += 1
            
        if count == 0: return 1e9
        return (total_error / count) + penalty

    # 2. Initial Guess 
    # STRATEGY: Start with a "Healthy" Smile.
    # rho = -0.6 (Standard equity skew)
    # lambda = 0.3 (Meaningful curvature)
    # gamma = 0.5 (Standard Heston/Square-root decay)
    # c0 = 0.01 (approx 10% vol squared)
    x0 = np.array([-0.6, 0.3, 0.5, 0.01, 0.02, 0.0]) 

    # 3. Define Bounds
    # KEY FIX: Lower bound on Lambda is 0.01 (No flat lines allowed)
    # KEY FIX: Tight bounds on Gamma around 0.5 (Stabilizes the solver)
    bounds = [
        (-0.999, 0.999), # 0: rho
        (0.01, 5.0),     # 1: lambda (MINIMUM 0.01 CURVATURE)
        (0.2, 0.8),      # 2: gamma (Constrain between 0.2 and 0.8 to prevent degeneracy)
        (1e-5, None),    # 3: c0 (Intercept > 0)
        (0.0, None),     # 4: c1 (Slope > 0, variance increases with time)
        (None, None)     # 5: c2 (Quadratic term)
    ]
        
    # 4. Optimize
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Use 'maxiter' to ensure it tries hard enough
            res = minimize(objective, x0, method='SLSQP', tol=1e-6, bounds=bounds, options={'maxiter': 400})
        
        if res.success:
            return res.x
        else:
            return res.x # Return best effort
            
    except Exception as e:
        print(f"SSVI Optimization error: {e}")
        return x0





def _process_single_day(day_data):
    day_obj, day_dfm_map, day_minutes_map, \
    day_minute_prices_map, day_minute_vol_map, \
    q_div, ticker, db_url = day_data
    day_minutes = day_minutes_map[day_obj] 
    day_df = day_dfm_map[day_obj]
    

    local_engine = create_engine(db_url, pool_pre_ping=True)

    processed_surfaces = []
    day_str = day_obj.strftime('%Y-%m-%d')
    print(f"  - Processing {day_str}...")
    r_rate = _get_rate(engine=local_engine, d=day_minutes[0]) / 100
    start_loop = int(datetime.now().timestamp())

    day_df["ts_min"] = day_df["ts_utc"].dt.floor('min')

    cols = ["ts_min", "strike", "dte", "option_close", "option_volume_weighted", "option_volume", "option_transactions", "contract_type"]
    base_day_grouped = day_df[cols].groupby("ts_min")
    C_KEY = "Call_Poly_State"
    P_KEY = "Put_Poly_State"
    # Time Loop
    for ts in day_minutes:
        S_t = day_minute_prices_map[day_obj].get(ts)
        S_vol_t = day_minute_vol_map[day_obj].get(ts)
        
        # Skip if underlying price is missing (gap > limit)
        if pd.isna(S_t) or S_t <= 0: continue

        # Get Option Slice
        sub = base_day_grouped.get_group(ts) if ts in base_day_grouped.groups else pd.DataFrame()

        joint_call_points = [] # List of (strike, iv, F, T)
        joint_put_points = []
        
        # RAW Lists for storage
        minute_market_c = []
        minute_market_p = []

        unique_dtes = sub['dte'].unique()
        for dte in unique_dtes:
            if dte < 1: continue
            T_expiry = dte / 365.0
            chain = sub[sub['dte'] == dte]
            
            # Process Calls
            calls = chain[chain['contract_type'] == 'C']
            if not calls.empty:
                # Store Raw (Price) - UNCHANGED
                raw_records = calls[['strike', 'option_volume_weighted', 'option_volume']].to_dict('records')
                for r in raw_records: r['dte'] = int(dte)
                minute_market_c.extend(raw_records)
                
                # Store for Optimization (CALCULATING IV using market price)
                for row in calls.itertuples():
                    # 1. Define the crucial input
                    market_price = row.option_volume_weighted 
                
            
                    iv = calculate_iv_american_call(
                        market_price=market_price, 
                        S=S_t, 
                        K=row.strike, 
                        T=T_expiry, 
                        r=r_rate, 
                        q=q_div, 
                        time_entry_ts=ts 
                    )
                    
    
                    joint_call_points.append((row.strike, iv, S_t, T_expiry, market_price))
            
            
            # Process Puts
            puts = chain[chain['contract_type'] == 'P']
            if not puts.empty:
                # Store Raw (Price) - UNCHANGED
                raw_records = puts[['strike', 'option_volume_weighted', 'option_volume']].to_dict('records')
                for r in raw_records: r['dte'] = int(dte)
                minute_market_p.extend(raw_records)
                
                for row in puts.itertuples():
                    market_price = row.option_volume_weighted
        
                    iv = calculate_iv_american_put(
                                market_price=market_price, 
                                S=S_t, 
                                K=row.strike, 
                                T=T_expiry, 
                                r=r_rate, 
                                q=q_div,
                                time_entry_ts=ts 
                            )
                
                    joint_put_points.append((row.strike, iv, S_t, T_expiry, market_price))
       
        # prev_c = tracker.sabr_states.get(C_KEY, {}).get('x', None) 
        sabr_coeffs_c = fit_polynomial_sabr_surface(joint_call_points)
        # kalman_sabr_coeffs_c = tracker.update_sabr(C_KEY, raw_coeffs_c, dim=12, has_data=True)

        # prev_p = tracker.sabr_states.get(P_KEY, {}).get('x', None) 
        sabr_coeffs_p = fit_polynomial_sabr_surface(joint_put_points)
        # kalman_sabr_coeffs_c = tracker.update_sabr(P_KEY, raw_coeffs_p, dim=12, has_data=True) 





        # prev_c = tracker.sabr_states.get(C_KEY, {}).get('x', None) 
        ssvi_coeffs_c = fit_polynomial_ssvi_surface(joint_call_points)
        # kalman_sabr_coeffs_c = tracker.update_sabr(C_KEY, raw_coeffs_c, dim=12, has_data=True)

        # prev_p = tracker.sabr_states.get(P_KEY, {}).get('x', None) 
        ssvi_coeffs_p = fit_polynomial_ssvi_surface(joint_put_points)
        # kalman_sabr_coeffs_c = tracker.update_sabr(P_KEY, raw_coeffs_p, dim=12, has_data=True) 


        # 4. Append Snapshot
        processed_surfaces.append({
            'time_entry_ts': ts,
            'ticker': ticker,
            'price_ffill_S': S_t,
            'underlying_volume': S_vol_t, 
            'risk_free_rate': r_rate,
            'market_data_C': minute_market_c, 
            'market_data_P': minute_market_p,
            'iv_point_C': joint_call_points, 
            'iv_point_P': joint_put_points,
            'sabr_coeffs_C': sabr_coeffs_c.tolist(),
            'sabr_coeffs_P': sabr_coeffs_p.tolist(),
            'ssvi_coeffs_C': ssvi_coeffs_c.tolist(),
            'ssvi_coeffs_P': ssvi_coeffs_p.tolist(),
        })

    end_loop = int(datetime.now().timestamp())
    print(f"compute loop for {day_str} took: {end_loop- start_loop} seconds")
    return processed_surfaces

def build_option_items(engine: Engine, ticker: str, start_day: str, end_day: str,  dte_max: int, k_pct: float,  q_div = 0.0, num_workers=10):
    processed_surfaces = []
    days_to_process = _business_days(start_day, end_day)
    all_filtered_dfs = []
    
    # Maps to hold data for workers
    day_sod_map = {}
    day_minutes_map = {}
    day_dfm_map = {}
    day_minute_prices_map = {} 
    day_minute_vol_map = {} 

    print(f"--- Loading data for {ticker} from {start_day} to {end_day} ---")

    ticker_id = get_company_id(engine, ticker)
    for day_obj in days_to_process:
   
        day_str = day_obj.strftime('%Y-%m-%d')
        
        S = _get_sod(engine, ticker_id, day_obj)
        if not S or S <= 0:
            print(f"[WARN] No SOD price for {day_str}, skipping day.")
            continue
            
        dfm = load_minutes(engine, ticker, day_str, dte_max)
        if dfm.empty:
            print(f"[WARN] No options data for {day_str}, skipping day.")
            continue
        
        vol_profil_days = 20 
        underlying_price_series, underlying_volume_series = load_underlying(engine, day_str, ticker_id, vol_profil_days)

        t0 = pd.Timestamp(f"{day_str} 13:30", tz="UTC") # 9:30 AM ET
        t1 = pd.Timestamp(f"{day_str} 20:00", tz="UTC") # 4:00 PM ET
        
        # Filter Market Hours
        dfm = dfm[(dfm["ts_utc"] >= t0) & (dfm["ts_utc"] <= t1)]
        dfm["strike"] = pd.to_numeric(dfm["strike"], errors="coerce")
        dfm = dfm.dropna(subset=["strike"])
        
        dfm_filtered = dfm[(np.abs(np.log(dfm["strike"].astype(float) / float(S))) <= k_pct)] 
        

        dfm_filtered = dfm_filtered[dfm_filtered["dte"].between(1, dte_max)]


        dfm_filtered = dfm_filtered[dfm_filtered['option_volume_weighted'] > 0.05]

 
        if dfm_filtered.empty:
            print(f"[WARN] No options data remaining after OTM filtering for {day_str}, skipping.")
            continue
            
        print(f"[INFO] Loaded {len(dfm_filtered)} OTM rows for {day_str} with SOD={S:.2f}")
        
        all_filtered_dfs.append(dfm_filtered)
        
        # Store Data in Maps
        day_sod_map[day_obj] = S
        day_minutes_map[day_obj] = pd.date_range(t0, t1, freq="1min", inclusive="both")
        day_dfm_map[day_obj] = dfm_filtered
        day_minute_prices_map[day_obj] = underlying_price_series
        day_minute_vol_map[day_obj] = underlying_volume_series 
    db_url = SQLALCHEMY_DATABASE_URL 
    
    day_keys = sorted(day_sod_map.keys())
    
    tasks = [
        (
            day_obj, 
            day_dfm_map,          
            day_minutes_map,      
            day_minute_prices_map,
            day_minute_vol_map,   
            q_div, 
            ticker, 
            db_url
        )
        for day_obj in day_keys
    ]

    all_day_results = []
    with Pool(processes=num_workers) as pool:
        all_day_results = pool.map(_process_single_day, tasks)
    
    for day_result in all_day_results:
        processed_surfaces.extend(day_result)

    return pd.DataFrame(processed_surfaces)



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("params_file", help="")

    args = parser.parse_args()
    params_file_path = args.params_file

    with open(params_file_path, 'r') as f:
        params = json.load(f)
    ticker = params["ticker"][0]
    dividend_rates = params["dividend_rates"][ticker]
    default_start_day = params["train_start"]
    default_end_day = params["train_end"]
    default_max_dte = params.get("max_dte", 30)
    k_pct=params.get("k_pct", .1)
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

    df = build_option_items( 
        engine, 
        ticker, 
        default_start_day, 
        default_end_day,    
        default_max_dte, 
        k_pct=k_pct, 
        q_div= dividend_rates
    )

    k_pct_str = f"{k_pct:.2f}" # e.g., '0.100'
    q_div_str = f"{dividend_rates:.4f}"

    filename = (
        f"{ticker}"
        f"_kpct{k_pct_str}"
        f"_dte{default_max_dte}"
        f"_from_{default_start_day}"
        f"_to_{default_end_day}"
        f"_qdiv{q_div_str}"
        f".parquet"
    )

    filename = filename.replace(":", "-") 
    filename = filename.replace(" ", "_") 

    df.to_parquet(filename)

if __name__ == '__main__':
    main()

