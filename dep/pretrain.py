
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
from scipy.optimize import minimize, curve_fit
import math

load_dotenv()
 
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)
# BAW Mode


def _hagan_vol_vectorized(k, F, T, alpha, beta, rho, nu):
    """
    Vectorized Hagan 2002 SABR Log-Normal Volatility.
    Accepts numpy arrays for inputs.
    """
    # Ensure inputs are arrays (float type)
    k = np.asarray(k, dtype=float)
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    nu = np.asarray(nu, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # 1. Safety / Validity Mask
    # We calculate everywhere but filter results at the end
    valid_mask = (F > 0) & (k > 0) & (T > 0) & (alpha > 0) & (nu > 0)

    # Use safe values for calculation to avoid RuntimeWarnings (div/0, log(neg))
    # These dummy values won't affect the final result because of valid_mask
    F_s = np.where(valid_mask, F, 1.0)
    k_s = np.where(valid_mask, k, 1.0)
    T_s = np.where(valid_mask, T, 1.0)
    alpha_s = np.where(valid_mask, alpha, 0.1)
    nu_s = np.where(valid_mask, nu, 0.1)
    rho_s = np.clip(rho, -0.999, 0.999) 

    # 2. Helper Variables
    log_fk = np.log(F_s / k_s)
    fk_beta = (F_s * k_s)**((1 - beta) / 2)
    z = (nu_s / alpha_s) * fk_beta * log_fk

    # 3. Calculate z / chi(z)
    sq_term = 1 - 2 * rho_s * z + z**2
    sq_term = np.maximum(sq_term, 0.0) # Safety clip for sqrt
    
    # log_arg = (sqrt(...) + z - rho) / (1 - rho)
    numerator = np.sqrt(sq_term) + z - rho_s
    denominator = 1 - rho_s
    
    # Guard for division by zero if rho is exactly 1 (unlikely due to bounds)
    log_arg = numerator / np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    log_arg = np.maximum(log_arg, 1e-18) # Ensure strictly positive
    
    chi = np.log(log_arg)
    
    # Handle z close to 0 (ATM case)
    is_small_z = np.abs(z) < 1e-5
    
    # Safe division for z/chi
    chi_safe = np.where(np.abs(chi) < 1e-10, 1.0, chi) 
    z_over_chi_calc = z / chi_safe
    
    # If small z, limit is 1.0
    z_over_chi = np.where(is_small_z, 1.0, z_over_chi_calc)

    # 4. Calculate Vol at T=0
    base_vol = alpha_s / fk_beta
    
    # Taylor expansion for Near-ATM (z < 1e-5)
    # Includes the higher order term (1/1920) you added
    taylor_denom = 1 + \
                   ((1 - beta)**2 / 24) * log_fk**2 + \
                   ((1 - beta)**4 / 1920) * log_fk**4
    
    vol_atm = base_vol / taylor_denom
    vol_otm = base_vol * z_over_chi
    
    vol_T0 = np.where(is_small_z, vol_atm, vol_otm)

    # 5. Calculate T-Expansion Term
    term2 = 1 + (
        ((1 - beta)**2 / 24) * (alpha_s**2 / (F_s**(2 - 2*beta))) +
        (0.25 * rho_s * beta * nu_s * alpha_s) / (F_s**(1 - beta)) +
        ((2 - 3 * rho_s**2) / 24) * nu_s**2
    ) * T_s

    # Final Result
    vol = vol_T0 * term2
    
    # Return 0.0 for invalid inputs (matching original logic)
    return np.where(valid_mask, vol, 0.0)


def get_decay_sabr_vectorized(coeffs, T):
    alpha_s, alpha_l, k = coeffs[0], coeffs[1], coeffs[2]
    nu_s, nu_l = coeffs[3], coeffs[4]
    rho = coeffs[5]
    
    T_safe = np.maximum(T, 1e-5)
    
    # --- ENGINEERING FIX: ANCHORING ---
    t_min_approx = np.min(T_safe)
    
    # 1. ALPHA DECAY: Switch to Inverse Square Root (Shifted Hyperbola)
    # This naturally creates the "Cliff" drop from Layer 1 to Layer 2
    # without needing k=80. A k of 1.0 to 5.0 is usually sufficient.
    # Formula: 1 / (1 + k * sqrt(t))
    # We shift time so it starts at 1.0 exactly at t_min
    delta_t = np.maximum(T_safe - t_min_approx, 0.0)
    
    # This is the "Elegant Solution"
    decay_alpha = 1.0 / (1.0 + k * np.sqrt(delta_t))
    
    # 2. NU/RHO DECAY: Keep Exponential (Curvature decays slower/smoother)
    decay_nu = np.exp(-k * delta_t)
    
    # Apply
    alpha = alpha_l + (alpha_s - alpha_l) * decay_alpha
    nu = nu_l + (nu_s - nu_l) * decay_nu
    
    return alpha, rho, nu




def estimate_sabr_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr):
    """
    Intelligent Guess Function v2:
    - Uses Robust Median statistics for Long-Term parameters to ignore artifacts.
    - Uses T[1] (Second Layer) for K-decay to capture steep initial drops.
    """
    default_x0 = np.array([0.5, 0.3, 0.8, 1.0, 0.1, -0.1])

    try:
        # 1. Slice Identification
        unique_ts = np.unique(T_arr)
        unique_ts.sort()
        
        if len(unique_ts) < 2: return default_x0

        t_first = unique_ts[0]
        t_second = unique_ts[1] # Critical for decay calc
        
        # Identify the "Latter Half" for robust long-term estimation
        idx_mid = len(unique_ts) // 2
        t_latter_half = unique_ts[idx_mid:]

        # 2. Helper: Get Metrics 
        def get_slice_metrics(target_t):
            mask = (T_arr == target_t)
            if not np.any(mask): return None
            
            k_sub = K_arr[mask]
            v_sub = v_mkt_arr[mask]
            s_val = S_t_arr[mask][0]
            
            if len(k_sub) < 3: return None

            # ATM Vol
            idx_atm = np.argmin(np.abs(k_sub - s_val))
            atm_vol = v_sub[idx_atm]
            
            # Robust Slope Calculation (Inner 80%)
            log_k = np.log(k_sub / s_val)
            target_low, target_high = -0.15, 0.15
            idx_low = np.argmin(np.abs(log_k - target_low))
            idx_high = np.argmin(np.abs(log_k - target_high))
            
            if idx_low == idx_high: # Fallback for narrow slices
                idx_low, idx_high = 0, len(k_sub) - 1

            k_min = k_sub[idx_low]
            k_max = k_sub[idx_high]
            v_min = v_sub[idx_low]
            v_max = v_sub[idx_high]
            
            log_k_diff = np.log(k_max / k_min)
            slope = (v_max - v_min) / log_k_diff if abs(log_k_diff) > 1e-3 else 0.0

            return {"atm_vol": atm_vol, "slope": slope}

        # 3. Compute Short Term Metrics
        m_first = get_slice_metrics(t_first)
        m_second = get_slice_metrics(t_second)
        
        if not m_first: return default_x0

        # 4. Compute Long Term Metrics (Robust Median)
        # We gather metrics for ALL slices in the latter half and take the median
        long_alphas = []
        long_slopes = []
        
        for t_l in t_latter_half:
            m = get_slice_metrics(t_l)
            if m:
                long_alphas.append(m["atm_vol"])
                long_slopes.append(m["slope"])
        
        if not long_alphas: 
            est_alpha_l = m_first["atm_vol"] * 0.8 # Fallback

        else:
            est_alpha_l = np.median(long_alphas) # <--- Robust Statistic
            median_slope_l = np.median(long_slopes)
    
        # --- PARAMETER SETTING ---
        est_alpha_s = m_first["atm_vol"]
        
        # --- DECAY K (Using T_Second) ---
        # The user observed "dramatic decrease" from Layer 1 to 2.
        # We force k to match this specific drop.
        est_k = 1.5 # Default
        denom = est_alpha_s - est_alpha_l
        
        if abs(denom) > 0.01 and m_second is not None:
            # Formula: alpha(t) = alpha_L + (alpha_S - alpha_L) * e^(-k*t)
            
            # Solve for k using t_second
            alpha_2 = m_second["atm_vol"]
            
            # Check for monotonicity (Normal Contango or Backwardation)
            # We need ratio to be between 0 and 1
            # ratio = (alpha_2 - est_alpha_l) / denom
            
            # if 0.01 < ratio < 0.99:
            #      est_k = -np.log(ratio) / max(t_second, 0.001)
            # elif ratio <= 0.01:
            #      est_k = 10.0 # Decay happened instantly (super steep)
            # elif ratio >= 0.99:
            #      est_k = 0.1 # Almost no decay

        # --- NU SHORT ---
        est_nu_s = (2.0 * abs(m_first["slope"])) / max(est_alpha_s, 0.01)

        # --- RHO ---
        est_rho = -0.6 if m_first["slope"] < 0 else 0.1

        # Clamps
        est_k = np.clip(est_k, 0.1, 8.0) # Allow higher k for steep drops
        est_nu_s = np.clip(est_nu_s, 0.1, 3.0)
        est_nu_l = .1
        return np.array([est_alpha_s, est_alpha_l, est_k, est_nu_s, est_nu_l, est_rho])

    except Exception as e:
        print(f"SABR Guess Calc Failed: {e}")
        return default_x0
    

def fit_polynomial_sabr_surface(all_options_data, prior_guess=None):
    if not all_options_data: 
        return np.zeros(6)
    
    # --- PRE-PROCESS DATA ---
    data = list(zip(*all_options_data))
    K_arr = np.array(data[0])
    v_mkt_arr = np.array(data[1])
    S_t_arr = np.array(data[2])
    T_arr = np.array(data[3])
    n_points = len(K_arr)

    # --- 1. GENERATE WEIGHTS ---
    weights = np.ones(n_points)
    
    # Identify unique expiries
    unique_ts = np.unique(T_arr)
    unique_ts.sort()
    
    # Weight the First Layer by 2.0
    # This acts as a "Tie-Breaker" ensuring the solver prioritizes
    # the front-month anchor without ignoring the rest of the term structure.
    if len(unique_ts) >= 1:
        t_1 = unique_ts[0]
        mask_1 = (T_arr == t_1)
        weights[mask_1] = 2.0 

    # --- DEFINE OBJECTIVE ---
    def objective(coeffs):
        # NOTE: Ensure this calls the NEW Hyperbolic/Anchored decay function
        alpha, rho, nu = get_decay_sabr_vectorized(coeffs, T_arr)
        
        # assume beta = 1
        v_model = _hagan_vol_vectorized(K_arr, S_t_arr, T_arr, alpha, 1.0, rho, nu)
        
        # Weighted MSE Calculation
        diff = (v_model - v_mkt_arr) * 100.0 
        
        # Apply weights to the squared differences
        weighted_sq_error = np.sum(weights * (diff**2))
        
        # Normalize by sum of weights
        return weighted_sq_error / np.sum(weights)
    
    # --- DETERMINE INITIAL GUESS ---
    if prior_guess is not None:
        x0 = prior_guess
        x0[4] = 0.1 # Ensure fixed param consistency
    else:
        x0 = estimate_sabr_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr)
        x0[4] = 0.1 # Ensure fixed param consistency

    # --- OPTIMIZATION BOUNDS ---
    # With Hyperbolic decay, we don't need k=100. 
    # Reducing max k to 20.0 improves solver stability.
    bounds = [
        (0.01, 2.5),    # 0: Alpha S 
        (0.01, 2.5),    # 1: Alpha L 
        (0.5, 20.0),    # 2: k (Min set to 0.5 ensures decay happens within ~2 years)
        (0.01, 5.0),    # 3: Nu S (Upper bound increased slightly)
        (0.1, 0.1),     # 4: Nu L (Fixed)
        (-0.999, 0.999) # 5: Rho 
    ]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, tol=1e-5, options={'maxiter': 250})
        return res.x
    except Exception as e:
        print(f"SABR Optimization error: {e}")
        return x0 #ridge regression loss, exypry_weighted_mae, exypry_weighted_rmse, unweighted_mae, unweighted_rmse
    
def get_ssvi_from_poly(coeffs, T):
    """
    Decodes the 8-vector SSVI coefficients and computes the local T parameters.
    coeffs: [rho, gamma, lambda0, lambda1, lambda2, c0, c1, c2] 
    """
    # NOTE: The check and indexing must be updated for 8 coeffs.
    if len(coeffs) != 8:
        raise ValueError(f"SSVI coefficient vector must have exactly 8 elements (found {len(coeffs)}).")

    # 1. Global Smile & Power Parameters
    rho = coeffs[0]
    gamma = coeffs[1]
    
    # 2. Term Structure Coefficients for Lambda (Smile Intensity)
    lam0, lam1, lam2 = coeffs[2], coeffs[3], coeffs[4] # NEW
    
    # 3. Term Structure Coefficients for Theta (ATM Variance)
    c0, c1, c2 = coeffs[5], coeffs[6], coeffs[7] # SHIFTED INDICES

    T_safe = np.clip(T, 1e-6, T) # Ensure T is positive
        
    # 4. Calculate ATM Total Variance (theta_T)
    theta_T = c0 + c1 * T_safe + c2 * T_safe**2
    theta_T = np.maximum(theta_T, 1e-6) 

    # 5. Calculate Time-Dependent Lambda (lambda_T) (NEW)
    lambda_T = lam0 + lam1 * T_safe + lam2 * T_safe**2
    lambda_T = np.maximum(lambda_T, 1e-6) # Ensure lambda is positive

    # The variance function now needs to take lambda_T instead of the global lam
    return rho, lambda_T, gamma, theta_T # Return lambda_T instead of lam

def ssvi_variance_function(k, T, theta_T, rho, lam, gamma):
    phi_T = lam / (theta_T**gamma)
    
    # Clip rho for numerical safety
    rho = np.clip(rho, -0.999, 0.999) 
    
    # Calculate w(k, T)
    term1 = phi_T * k + rho
    sqrt_term = np.sqrt(term1**2 + (1 - rho**2))
    w_kT = (theta_T / 2.0) * (1.0 + rho * phi_T * k + sqrt_term)
    
    return np.maximum(w_kT, 1e-12)
def linear_quadratic_theta(T, c1, c2): 
    # This reduces the fit to 2 parameters
    return c1 * T + c2 * T**2



def estimate_ssvi_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr):
    # --- Helper: Quadratic Term Structure Model ---
    def quadratic_theta(T, c0, c1, c2):
        return c0 + c1 * T + c2 * T**2
    
    # 1. Calculate Market Total Variance (omega_mkt = v^2 * T)
    omega_mkt = (v_mkt_arr ** 2) * T_arr
    k_arr = np.log(K_arr / S_t_arr)

    # 2. Filter for ATM Points (|k| < 0.05)
    atm_mask = np.abs(k_arr) < 0.05
    T_atm = T_arr[atm_mask]
    Omega_atm = omega_mkt[atm_mask]

    # Use a robust median if standard fit fails or if data is sparse
    if len(T_atm) < 3:
        # Fallback to static guess if insufficient ATM data for polynomial
        return np.array([-0.6, 0.3, 0.5, 0.001, 0.05, 0.0]) 

    # 3. Fit the Quadratic Curve (Find c0, c1, c2)
    try:
        # Use the 2-parameter model with fixed c0=0
        # p0: initial guess for [c1, c2]. 
        popt, pcov = curve_fit(linear_quadratic_theta, T_atm, Omega_atm, p0=[0.05, 0.001],
                            bounds=([0, -np.inf], [np.inf, np.inf]), # Only 2 bounds now
                            maxfev=1000)
        c1_est, c2_est = popt
        c0_est = 1e-6 # Fixed floor for c0
        
    except Exception:
        c0_est, c1_est, c2_est = 1e-6, 0.05, 0.0

    # 4. Estimate Global Smile Parameters (Rho)
    # Focus on the most liquid layer (T1) for reliable skew estimation
    t_1 = np.min(T_arr)
    mask_1 = (T_arr == t_1)
    k_1 = k_arr[mask_1]
    v_1 = v_mkt_arr[mask_1]
    
    if len(k_1) > 5:
        # Linear fit of IV vs Log-Moneyness (k)
        slope_k = np.polyfit(k_1, v_1, 1)[0]
        est_rho = np.clip(slope_k * 5.0, -0.9, 0.5) 
    else:
        est_rho = -0.6 # Safe fallback

    # 5. Assemble x0
    est_lam0 = 1.0 # Initial guess for lambda(T=0)
    est_lam1 = 0.0 # Initial guess for linear lambda term
    est_lam2 = 0.0 # Initial guess for quadratic lambda term
    est_rho = est_rho 
    est_gamma = 0.5
    
    # We enforce c2 is close to 0 as a stable prior, overriding the fit if necessary
    c2_final = np.clip(c2_est, -0.001, 0.001)

    return np.array([
        est_rho,    # 0: rho (Global Skew)
        est_gamma,  # 1: gamma (Power Exponent)
        est_lam0,   # 2: lambda0 (T=0 Smile Intensity) <--- NEW
        est_lam1,   # 3: lambda1 (Linear Lambda Term) <--- NEW
        est_lam2,   # 4: lambda2 (Quadratic Lambda Term) <--- NEW
        c0_est,     # 5: c0 (T=0 Variance) <--- SHIFTED
        c1_est,     # 6: c1 (Linear Term) <--- SHIFTED
        c2_final    # 7: c2 (Quadratic Term) <--- SHIFTED
    ])

def fit_polynomial_ssvi_surface(all_options_data, prior_guess=None):

    if not all_options_data:
        return np.zeros(6)

    # --- PRE-PROCESS DATA ---
    data = list(zip(*all_options_data))
    K_arr = np.array(data[0])
    v_mkt_arr = np.array(data[1])
    S_t_arr = np.array(data[2]) 
    T_arr = np.array(data[3])
    n_points = len(K_arr)
        
    # --- 1. DYNAMIC WEIGHT GENERATION (WLS) ---
    weights = np.ones(n_points)
    unique_ts = np.unique(T_arr)
    unique_ts.sort()
    
    if len(unique_ts) >= 1:
        t_1 = unique_ts[0]
        mask_1 = (T_arr == t_1)
        # Weight Layer 1 heavily (5.0) to force the term structure (c1) to anchor
        weights[mask_1] = 5.0 
        
    T_safe = np.clip(T_arr, 1e-4, T_arr)
    k_arr = np.log(K_arr / S_t_arr) # Log Moneyness
    w_market = (v_mkt_arr ** 2) * T_safe # Market Total Variance

    # --- DEFINE OBJECTIVE ---
    def objective(coeffs):
        # UNPACK 8 COEFFICIENTS HERE
        rho, gamma, lam0, lam1, lam2, c0, c1, c2 = coeffs
        
        # Calculate Time-Dependent Lambda (lambda_T) (NEW)
        lambda_T = lam0 + lam1 * T_safe + lam2 * T_safe**2
        lambda_T = np.maximum(lambda_T, 1e-6)
        
        # Calculate ATM Total Variance (theta_T)
        theta_T = c0 + c1 * T_safe + c2 * T_safe**2
        theta_T = np.maximum(theta_T, 1e-4)
        
        # Model Variance (Pass lambda_T instead of the global lam)
        w_model = ssvi_variance_function(k_arr, T_safe, theta_T, rho, lambda_T, gamma)
                                
        # Weighted MSE Calculation (on Total Variance)
        diff = (w_model - w_market) * 100.0 
        weighted_sq_error = np.sum(weights * (diff**2))
        
        return weighted_sq_error / np.sum(weights)

    # --- DETERMINE INITIAL GUESS ---
    if prior_guess is not None:
        x0 = prior_guess
        # Ensure c2 remains tightly constrained even if using Kalman prior
        x0[7] = np.clip(x0[7], -0.001, 0.001) # Index 7 is now c2
    else:
        # Use the robust initialization derived from the data
        x0 = estimate_ssvi_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr)
        
    # --- OPTIMIZATION BOUNDS (8 bounds now) ---
    bounds = [
        (-0.999, 0.999),    # 0: rho (Global Skew) (Wider range for better fit)
        (0.2, 2.0),        # 1: gamma (Power Exponent)
        
        (0.01, 5.0),       # 2: lam0 (T=0 Lambda) <--- NEW: High max for short-term smile
        (-0.5, 0.5),       # 3: lam1 (Linear Lambda Term) <--- NEW
        (-0.1, 0.1),       # 4: lam2 (Quadratic Lambda Term) <--- NEW
        
        (1e-6, 0.05),      # 5: c0 (T=0 Variance) <--- SHIFTED. Increased max from 0.01 to 0.05
        (0.0, 0.5),        # 6: c1 (Linear Term) <--- SHIFTED
        (-0.001, 0.001)    # 7: c2 (Quadratic Term) <--- SHIFTED. Tightly constrained for stability
    ]
        
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(objective, x0, method='SLSQP', tol=1e-6, bounds=bounds, options={'maxiter': 400})
        return res.x
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

