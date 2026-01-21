import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import math
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def _gaussian_smooth(Z: np.ndarray, sigma_strike: float = 1.0, sigma_dte: float = 0.5) -> np.ndarray:
    """Applies Gaussian filter for general smoothing, ignoring NaNs."""
    if Z.size == 0:
        return Z
    
    # Handle NaNs: Replace with mean or 0
    Z_temp = np.nan_to_num(Z, nan=np.nanmean(Z) if np.nanmean(Z) is not np.nan else 0.0)

    # Apply Gaussian Filter
    Z_smooth = gaussian_filter(Z_temp, sigma=(sigma_strike, sigma_dte), mode="nearest")
    
    # Clip high outliers (Safety rail)
    Z_smooth = np.clip(Z_smooth, a_min=0.01, a_max=5.0) 
    
    return Z_smooth

def _robust_matrix_load(matrix_data, expected_shape):
    """Parses nested lists/arrays from Parquet into fixed numpy grids."""
    if matrix_data is None:
        return np.full(expected_shape, np.nan, dtype=np.float64)

    if isinstance(matrix_data, np.ndarray):
        if matrix_data.size == 0:
             return np.full(expected_shape, np.nan, dtype=np.float64)

        if matrix_data.ndim == 1 and matrix_data.shape[0] == expected_shape[0]:
            try:
                unpacked = np.array(matrix_data.tolist(), dtype=np.float64) 
                if unpacked.shape == expected_shape:
                    return unpacked
            except Exception:
                pass

        try:
            return matrix_data.reshape(expected_shape).astype(np.float64)
        except ValueError:
             return np.hstack(matrix_data).reshape(expected_shape).astype(np.float64)

    if isinstance(matrix_data, list):
        if not matrix_data:
             return np.full(expected_shape, np.nan, dtype=np.float64)
        if isinstance(matrix_data[0], list):
            flattened = [item for sublist in matrix_data for item in sublist]
        else:
            flattened = matrix_data
        return np.asarray(flattened, dtype=np.float64).reshape(expected_shape)

    raise TypeError(f"Unsupported matrix data type: {type(matrix_data)}")

def _propagate_last_valid_dte(Z_matrix: np.ndarray) -> np.ndarray:
    """Aggressively forward-fills NaN/zero values along the DTE axis."""
    Z_temp = Z_matrix.copy()
    Z_temp[Z_temp <= 1e-6] = np.nan
    
    # Forward fill (axis=1 is DTE/Time)
    df = pd.DataFrame(Z_temp)
    Z_filled = df.ffill(axis=1).values
    
    # Backward fill to catch leading NaNs
    Z_filled = pd.DataFrame(Z_filled).bfill(axis=1).values
    
    # Fallback floor
    Z_filled[np.isnan(Z_filled)] = 0.05 
    
    return Z_filled

# --- CORE COMPUTATION LOGIC ---
# constructs IV_Weighted_matrix
def compute_weighted_surface(processed_df: pd.DataFrame) -> pd.DataFrame:
    if processed_df.empty:
        print("[ERROR] Empty DataFrame.")
        return processed_df

    # 1. Determine Grid Shape from first row
    first_row = processed_df.iloc[0]
    full_strikes = first_row['strike_grid']
    full_dte_grid = first_row['dte_grid']
    expected_shape = (len(full_strikes), len(full_dte_grid))
    
    print(f"[INFO] Grid Shape: {expected_shape} (Strikes x DTE)")

    # 2. Check for Volume Data
    has_volume = 'Vol_C_matrix' in processed_df.columns and 'Vol_P_matrix' in processed_df.columns
    if not has_volume:
        print("[WARNING] Volume data missing. Will default to simple average.")

    # 3. Iterate and Calculate
    # We store results in a list first, then assign to DF (faster than row-by-row assignment)
    calculated_matrices = []

    total_rows = len(processed_df)
    print(f"[INFO] Starting calculation on {total_rows} rows...")

    for index, row in processed_df.iterrows():
        if index % 100 == 0:
            print(f"   Processing row {index}/{total_rows}...", end='\r')

        try:
            # Load raw IVs
            C_full = _robust_matrix_load(row['IV_C_matrix'], expected_shape)
            P_full = _robust_matrix_load(row['IV_P_matrix'], expected_shape)
            
            if has_volume:
                # Load Volumes
                C_vol = _robust_matrix_load(row['Vol_C_matrix'], expected_shape)
                P_vol = _robust_matrix_load(row['Vol_P_matrix'], expected_shape)
                
                # Sanitize Volumes (NaN -> 0)
                C_vol = np.nan_to_num(C_vol, nan=0.0)
                P_vol = np.nan_to_num(P_vol, nan=0.0)
                
                # --- THE FORMULA ---
                # IV_Final = (IV_Call * Vol_Call + IV_Put * Vol_Put) / (Vol_Call + Vol_Put)
                
                numerator = (np.nan_to_num(C_full, 0.0) * C_vol) + (np.nan_to_num(P_full, 0.0) * P_vol)
                denominator = C_vol + P_vol
                
                # Safe Divide
                Final_Calculated = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator!=0)
                
                # Fallback: Where volume is 0, use Simple Average
                simple_avg = (C_full + P_full) / 2.0
                mask_nan = np.isnan(Final_Calculated)
                Final_Calculated[mask_nan] = simple_avg[mask_nan]
            else:
                # No volume data available
                Final_Calculated = (C_full + P_full) / 2.0

            # Optional: Smoothing & Propagation 
            # (We apply this here so the saved data is "ready to use")
            Final_filled = _propagate_last_valid_dte(Final_Calculated)
            Final_smooth = _gaussian_smooth(Final_filled, sigma_strike=1.0, sigma_dte=0.5)
            
            # Flatten back to list/array format for storage in Parquet
            # (Parquet doesn't like multi-dim numpy arrays directly in cells sometimes)
            calculated_matrices.append(Final_smooth.flatten())

        except Exception as e:
            print(f"[WARN] Failed row {index}: {e}")
            calculated_matrices.append(np.full(expected_shape, np.nan).flatten())

    print(f"\n[INFO] Calculation complete.")
    
    # 4. Attach to DataFrame
    # We create a NEW column for the weighted calculation
    processed_df['IV_Weighted_matrix'] = calculated_matrices
    
    return processed_df




@dataclass
class SABRSurfaceParams:
    # 4 coeffs for alpha, 4 for rho, 4 for nu = 12 Dimensions
    a: np.ndarray # Alpha coeffs
    c: np.ndarray # Rho coeffs
    d: np.ndarray # Nu coeffs
    # Beta is fixed at 1, so no coeffs needed

def _poly4(u): return np.array([1.0, u, u*u, u*u*u])


def _hagan_vol(k, F, T, alpha, beta, rho, nu):
    """
    Standard Hagan 2002 Log-Normal Volatility Approximation.
    
    Ref: "Managing Smile Risk", Hagan et al (2002).
    """
    # 1. Safety Checks
    if F <= 0 or k <= 0 or T <= 0:
        return 0.0
    if alpha <= 0 or nu < 0: # Basic parameter sanity
        return 0.0

    # 2. Pre-compute common terms
    # If F and k are very close (ATM), handle separately to avoid div by zero
    if abs(F - k) < 1e-5:
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + (
            ((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) +
            (0.25 * rho * beta * nu * alpha) / (F**(1 - beta)) +
            ((2 - 3 * rho**2) / 24) * nu**2
        ) * T
        return term1 * term2

    # 3. OTM / ITM Case
    log_fk = math.log(F / k)
    fk_beta = (F * k)**((1 - beta) / 2)
    
    # z definition
    z = (nu / alpha) * fk_beta * log_fk
    
    # 4. Calculate z / chi(z)
    # chi(z) = log( (sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho) )
    # We need the ratio z / chi(z).
    
    if abs(z) < 1e-5:
        # Taylor expansion for small z to ensure numerical stability
        z_chi = 1.0 - 0.5 * rho * z
    else:
        # Standard formula
        # Ensure the term inside sqrt is non-negative (floating point safety)
        sq_term = 1 - 2 * rho * z + z**2
        if sq_term < 0: sq_term = 0
        
        chi = math.log((math.sqrt(sq_term) + z - rho) / (1 - rho))
        z_chi = z / chi

    # 5. Assemble the three main terms
    
    # A: The main scaling factor (Alpha / denominator expansion)
    denom_expansion = 1 + ((1 - beta)**2 / 24) * log_fk**2 + ((1 - beta)**4 / 1920) * log_fk**4
    A = alpha / (fk_beta * denom_expansion)
    
    # B: The z/chi factor calculated above
    B = z_chi
    
    # C: The time-dependent correction (small volatility of volatility expansion)
    C = 1 + T * (
        ((1 - beta)**2 / 24) * (alpha**2 / ((F * k)**(1 - beta))) +
        (0.25 * rho * beta * nu * alpha / fk_beta) +
        ((2 - 3 * rho**2) / 24) * nu**2
    )

    return A * B * C




def fit_slice_sabr(strikes, ivs, F, T):
    """Fits scalar alpha, rho, nu for a specific expiry slice."""
    def obj(params):
        a, r, n = params
        # Penalties
        if a <= 0 or n <= 0 or abs(r) >= 0.99: return 1e9
        
        err = 0.0
        for i, k in enumerate(strikes):
            model_vol = _hagan_vol(k, F, T, a, 1.0, r, n)
            err += (model_vol - ivs[i])**2
        return err

    # Guess: Alpha ~ ATM Vol, Rho=0, Nu=0.5
    guess = [np.mean(ivs), 0.0, 0.5]
    bounds = [(0.001, 5.0), (-0.99, 0.99), (0.001, 5.0)]
    
    res = minimize(obj, guess, bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else np.array([np.nan, np.nan, np.nan])

def extract_12d_parameters(iv_surface, strike_grid, dte_grid):
    """
    1. Slices the surface by DTE.
    2. Fits SABR (alpha, rho, nu) for each DTE.
    3. Fits 3 separate polynomials (degree 3, 4 coeffs) to the params vs Log(Time).
    Returns: 12-element array [alpha_coeffs(4), rho_coeffs(4), nu_coeffs(4)].
    """
    # Containers for the slice-by-slice params
    alphas, rhos, nus = [], [], []
    valid_log_t = []
    
    # We assume Forward F is approx the ATM strike (Simplification as F is not in DF)
    # In production, you'd compute F from Put-Call Parity.
    
    for j, dte in enumerate(dte_grid):
        if dte < 1.0: continue # Skip expiry < 1 day
        T = dte / 365.0
        
        # Get the slice
        iv_slice = iv_surface[:, j]
        
        # Filter valid data
        mask = (iv_slice > 0.01) & (np.isfinite(iv_slice))
        if np.sum(mask) < 3: continue
        
        ks = strike_grid[mask]
        vs = iv_slice[mask]
        
        # Estimate F (Forward) as strike with min IV (ATM proxy)
        F_est = ks[np.argmin(vs)]
        
        # Fit Slice
        a, r, n = fit_slice_sabr(ks, vs, F_est, T)
        
        if not np.isnan(a):
            alphas.append(a)
            rhos.append(r)
            nus.append(n)
            # Normalize time for polynomial fit: u = norm_u(logT)
            # For simplicity here, we fit against log(T) directly or simple scaling
            # The prompt uses _norm_u, we will standardize T later. 
            valid_log_t.append(math.log(T))

    if len(valid_log_t) < 4:
        return np.full(12, np.nan)

    # Convert to arrays
    X = np.array(valid_log_t)
    # Normalize X for stability (u)
    mu_t, sd_t = np.mean(X), np.std(X) + 1e-9
    u = (X - mu_t) / sd_t
    
    # Helper to fit poly4
    def fit_poly(y_vals):
        # Solve Least Squares: y = c0 + c1*u + c2*u^2 + c3*u^3
        # Design matrix A
        A = np.vstack([u**0, u**1, u**2, u**3]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y_vals, rcond=None)
        return coeffs

    c_alpha = fit_poly(alphas)
    c_rho   = fit_poly(rhos)
    c_nu    = fit_poly(nus)
    
    # Concatenate to 12D vector
    return np.concatenate([c_alpha, c_rho, c_nu])
def hagan_vol(K, F, T, alpha, rho, nu, beta=1.0):
    """Standard Hagan 2002 Approximation."""
    if K <= 0 or F <= 0 or T <= 0: return 0.0
    
    # Handle ATM case specifically to avoid div by zero
    if abs(K - F) < 1e-5:
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + (((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) + 
                     (rho * beta * nu * alpha) / (4 * F**(1 - beta)) + 
                     ((2 - 3 * rho**2) / 24) * nu**2) * T
        return term1 * term2

    log_fk = math.log(F / K)
    fk_beta = (F * K)**((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    
    # x(z) function
    if abs(z) < 1e-5:
        x_z = 1.0
    else:
        # Safer sqrt
        discr = 1 - 2 * rho * z + z**2
        if discr < 0: discr = 0
        chi = math.log((math.sqrt(discr) + z - rho) / (1 - rho))
        x_z = chi / z # Note: Formula often presented as z/chi, we use chi/z in denom below

    term1 = alpha / (fk_beta * (1 + ((1 - beta)**2 / 24) * log_fk**2 + ((1 - beta)**4 / 1920) * log_fk**4))
    
    # Note: We inverted x_z above, so we divide by it here (standard formula divides by chi/z, or multiplies by z/chi)
    if abs(z) >= 1e-5:
        term1 = term1 * (z / chi)

    term2 = 1 + (( (1 - beta)**2 / 24 * alpha**2 / fk_beta**2 ) + 
                 (rho * beta * nu * alpha) / (4 * fk_beta) + 
                 (2 - 3 * rho**2) / 24 * nu**2) * T
    
    return term1 * term2

def calibrate_slice(strikes, ivs, F, T):
    """Fits scalar SABR to a single maturity slice."""
    # Filter bad data
    mask = (ivs > 0.01) & (strikes > 0)
    k_clean = strikes[mask]
    v_clean = ivs[mask]
    
    if len(k_clean) < 4: return [np.nan, np.nan, np.nan]

    def objective(params):
        a, r, n = params
        error = 0.0
        for i, k in enumerate(k_clean):
            vol = hagan_vol(k, F, T, a, r, n)
            error += (vol - v_clean[i])**2
        return error

    # Bounds: alpha>0, -1<rho<1, nu>0
    bnds = ((0.01, 5.0), (-0.99, 0.99), (0.01, 5.0))
    # Guess based on ATM
    guess = [v_clean[len(v_clean)//2], 0.0, 0.5]
    
    res = minimize(objective, guess, bounds=bnds, method='L-BFGS-B')
    return res.x if res.success else [np.nan, np.nan, np.nan]

def get_12d_sabr_vector(surface_flat, strike_grid, dte_grid, n_strikes, n_dte):
    """
    Converts a flattened surface into a 12D vector:
    Fits Alpha(t), Rho(t), Nu(t). 
    Then fits 3rd degree polynomial to each parameter vs Log(Time).
    Returns 12 coeffs.
    """
    surface = surface_flat.reshape(n_strikes, n_dte)
    
    alphas, rhos, nus = [], [], []
    times = []
    
    # Forward price approx (ATM strike with min IV)
    # In production, extract F from Put/Call parity. Here we approximate.
    
    for t_idx, dte in enumerate(dte_grid):
        if dte < 2.0: continue # Skip very near term (unstable)
        
        T = dte / 365.0
        slice_iv = surface[:, t_idx]
        
        # Approximation of Forward: The strike with lowest IV (roughly ATM)
        min_idx = np.argmin(slice_iv)
        F_approx = strike_grid[min_idx]
        
        p = calibrate_slice(np.array(strike_grid), slice_iv, F_approx, T)
        
        if not np.isnan(p[0]):
            alphas.append(p[0])
            rhos.append(p[1])
            nus.append(p[2])
            times.append(math.log(T)) # Fit against Log Time
            
    if len(times) < 5:
        return np.full(12, np.nan)

    # Fit Polynomials (Degree 3 -> 4 coeffs)
    # y = c0 + c1*x + c2*x^2 + c3*x^3
    # polyfit returns [c3, c2, c1, c0] (highest power first)
    
    try:
        c_alpha = np.polyfit(times, alphas, 3)
        c_rho   = np.polyfit(times, rhos, 3)
        c_nu    = np.polyfit(times, nus, 3)
        return np.concatenate([c_alpha, c_rho, c_nu])
    except:
        return np.full(12, np.nan)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", help="Path to JSON", default="params.json") 
    args, unknown = parser.parse_known_args() 

    # 1. Setup & Loading
    parquet_filename = "AMD_kpct0.300_dte90_from_2025-10-31_01-00-00_to_2025-11-01_23-59-00_qdiv0.0000.parquet"

    if not os.path.exists(parquet_filename):
        print(f"[FATAL] File not found: {parquet_filename}")
        return

    print(f"--- 1. Loading data: {parquet_filename} ---")
    df = pd.read_parquet(parquet_filename)
    
    print("--- 2. Performing Volume-Weighted IV Computation ---")
    df_calculated = compute_weighted_surface(df)

    # Dynamic Grid Sizing
    sample_strikes = df_calculated.iloc[0]['strike_grid']
    sample_dtes = df_calculated.iloc[0]['dte_grid']
    n_strikes = len(sample_strikes)
    n_dte = len(sample_dtes)
    
    print(f"Grid Dimensions: {n_strikes} Strikes x {n_dte} DTEs")

    # 2. Fit 12D SABR to History
    print("--- 3. Fitting 12D Time-Dependent SABR to History ---")
    historical_vectors = []
    
    # Run for all rows to build full history
    for i, row_surf in enumerate(df_calculated['IV_Weighted_matrix']):
        if i % 100 == 0: print(f"Processing row {i}...")
        vec_12d = get_12d_sabr_vector(row_surf, sample_strikes, sample_dtes, n_strikes, n_dte)
        if not np.isnan(vec_12d).any():
            historical_vectors.append(vec_12d)
    
    historical_data = np.array(historical_vectors)
    print(f"Valid Historical 12D Data Shape: {historical_data.shape}")

    if len(historical_data) < 50:
        print("[ERROR] Insufficient data for PCA.")
        return

    # 3. Split & PCA Training
    # We strictly separate Train (Past) and Test (Future)
    split_idx = int(len(historical_data) * 0.8)
    train_data = historical_data[:split_idx]
    test_data  = historical_data[split_idx:]

    print(f"Training on first {split_idx} samples. Testing on last {len(test_data)} samples.")

    # PCA Training (finding the 'Loading Matrix' Vk on history)
    k = 3
    pca_model = PCA(n_components=k)
    pca_model.fit(train_data)
    
    # The Historical Loading Matrix (Vk) and Mean (mu)
    Vk_train = pca_model.components_.T  # Shape (12, k)
    mu_train = pca_model.mean_          # Shape (12,)
    
    print(f"\n[LOGIC] PCA Trained. Vk Shape: {Vk_train.shape}")
    print(f"[LOGIC] Explained Variance: {pca_model.explained_variance_ratio_}")

    # 4. N-Step Prediction Setup
    N_step = 5
    if len(test_data) < N_step:
        print("[WARN] Not enough test data for N-step logic.")
        return

    # We pick a specific target in the "Future" (Test Set)
    # Let's predict the state at T + N_step
    current_state_real = test_data[0]         # T=0 in test set
    target_state_real  = test_data[N_step]    # T=5 in test set
    
    print(f"\n--- SCENARIO 1: The 'God' Signal (Perfect PCA Information) ---")
    # Logic: Assume we know the future perfectly, but we are forced to view it 
    # through the lens of our historical PCA (k=3).
    # This measures the "Compression Loss" - the best possible result our model could ever achieve.
    
    # 1. God gives us the latent factors of the FUTURE target
    #    F_god = (P_target - mu_train) @ Vk_train
    god_factors = pca_model.transform(target_state_real.reshape(1, -1))
    
    # 2. We reconstruct using historical loadings
    #    P_reconstructed = F_god @ Vk_train.T + mu_train
    reconstructed_god = pca_model.inverse_transform(god_factors).flatten()
    
    god_error = np.linalg.norm(target_state_real - reconstructed_god)
    
    print(f"Target Real Vector (Sample): {target_state_real[:4].round(3)}...")
    print(f"God Reconstructed  (Sample): {reconstructed_god[:4].round(3)}...")
    print(f"-> God's Reconstruction Error (Limit of PCA): {god_error:.6f}")
    print("   (This is the error purely due to compressing 12D -> 3D)")

    print(f"\n--- SCENARIO 2: Actual N-Step Prediction (VAR Model) ---")
    # Logic: We don't know the future. We must predict F(t+5) based on F(t).
    
    # 1. Transform Training Data to Latent Space
    train_factors = pca_model.transform(train_data)
    
    # 2. Train Linear Propagator (VAR) on History
    X_train = train_factors[:-1]
    y_train = train_factors[1:]
    
    predictor = LinearRegression()
    predictor.fit(X_train, y_train)
    
    # 3. Step-by-Step Prediction from Test Start
    #    Start at F(Test_T0)
    current_factor = pca_model.transform(current_state_real.reshape(1, -1))
    
    print(f"Starting prediction loop for {N_step} steps...")
    for s in range(N_step):
        # Predict next step: F(t+1) = A * F(t) + b
        current_factor = predictor.predict(current_factor)
        
    # 4. Reconstruct Final Prediction
    predicted_vector = pca_model.inverse_transform(current_factor).flatten()
    
    prediction_error = np.linalg.norm(target_state_real - predicted_vector)

    print(f"Target Real Vector (Sample): {target_state_real[:4].round(3)}...")
    print(f"VAR Predicted      (Sample): {predicted_vector[:4].round(3)}...")
    print(f"-> Actual Prediction Error: {prediction_error:.6f}")
    
    # --- Summary ---
    print("\n--- FINAL ANALYSIS ---")
    print(f"1. Theoretical Minimum Error (God/PCA Limit): {god_error:.6f}")
    print(f"2. Actual Model Error (VAR + PCA):            {prediction_error:.6f}")
    
    excess_error = prediction_error - god_error
    print(f"3. Error due to bad forecasting (Excess):     {excess_error:.6f}")
    
    if excess_error < god_error:
        print("   >> CONCLUSION: Your Forecast is good. The bottleneck is PCA compression (k=3 is too low).")
    else:
        print("   >> CONCLUSION: Your Forecast is bad. The VAR model is drifting.")

    # Save debug info
    out_file = parquet_filename.replace(".parquet", "_god_vs_pred.json")
    with open(out_file, 'w') as f:
        json.dump({
            "god_error": god_error,
            "prediction_error": prediction_error,
            "vk_matrix": Vk_train.tolist(),
            "mu_vector": mu_train.tolist()
        }, f)
    print(f"Saved analysis to {out_file}")

if __name__ == "__main__":
    main()