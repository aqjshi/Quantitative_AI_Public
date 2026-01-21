
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.optimize import minimize
from collections import namedtuple
from eigen_helper import _norm_u, _poly4, _softplus, _tanh, geometric_brownian_motion, Params, sabr_black_iv, simulate_strike_axis_given_base_sabr, generate_mpl_animation

from matplotlib.animation import FuncAnimation





OUTPUT_DIR = "SABR_Animations"
T_MAX = 2.0           # Total time for the simulation (matching max expiry)
N_SAMPLES = 500       # Number of steps in the simulation
P_SIZE = 12           
K_LATENT = 5          
MARKET_NOISE = 1
INITIAL_STOCK_PRICE = 150
FORWARD = 100.0

# Time Axis (Used by all simulated paths)
TIME_LINSPACE = np.linspace(0.0, T_MAX, N_SAMPLES + 1) 

# Grid Definitions (Used to define the IV surface structure)
STRIKES = np.linspace(80, 120, 25)  
EXPIRIES = np.linspace(0.1, 2.0, 15) 
STRIKE_GRID, EXPIRY_GRID = np.meshgrid(STRIKES, EXPIRIES)


# --- 2. PRICE (STOCK) PATH PARAMETERS ---
MU_PRICE = 0.05               # Expected Annual Return (Drift for S_t)
SIGMA_PRICE = 0.20            # Annual Volatility for S_t

simulated_path = geometric_brownian_motion(
    S0=INITIAL_STOCK_PRICE, 
    mu=MU_PRICE, 
    sigma=SIGMA_PRICE, 
    T=T_MAX, 
    N=N_SAMPLES
)
simulated_path = [100] * (N_SAMPLES + 1)



# --- 3. SABR PARAMETER PATHS (ALPHA, RHO, NU) ---
MU_SABR = 0.00                # Assumed Zero Drift for Volatility Parameters
SIGMA_SABR = 0.1             # Lower Volatility for Parameter Changes

simulated_alpha_path = geometric_brownian_motion(
    S0=.2, 
    mu=.00,              # Use Mu_SABR
    sigma=SIGMA_SABR, 
    T=T_MAX, 
    N=N_SAMPLES
)

simulated_rho_path = geometric_brownian_motion(
    S0=-0.9, 
    mu=-0.00,              # Use Mu_SABR
    sigma=SIGMA_SABR, 
    T=T_MAX, 
    N=N_SAMPLES
)

simulated_nu_path = geometric_brownian_motion(
    S0=.5,
    mu=-0.0,
    sigma=SIGMA_SABR, 
    T=T_MAX, 
    N=N_SAMPLES
)

# Enforce Bounds on Rho Path
simulated_rho_path = np.clip(simulated_rho_path, -0.9999, 0.9999)

# --- Final Linespace Definitions ---
P_MIN = np.min(simulated_path)
P_MAX = np.max(simulated_path)
R_RATE = 0.02   # Example Risk-Free Rate (2%)
Q_YIELD = 0.00  # Example Dividend Yield (0%)




if __name__ == "__main__":
    market_history = []
    
    for t_idx in range(TIME_LINSPACE.size):
        
        # 1. Dynamic Spot Price and Time
        current_spot_S = simulated_path[t_idx]
        current_time_t = TIME_LINSPACE[t_idx] # Time elapsed since t=0
        
        # 2. Dynamic SABR Parameters
        current_alpha = simulated_alpha_path[t_idx]
        current_rho = simulated_rho_path[t_idx]
        current_nu = simulated_nu_path[t_idx]
        
        strike_expiry_surface = []
        
        for expiry in EXPIRIES:
            # 3. CALCULATE DYNAMIC FORWARD (F) and TIME-TO-EXPIRY (T)
            
            # T is the ABSOLUTE time (e.g., 0.1, 0.5, 1.0, 2.0 years)
            T_remaining = max(0.001, expiry - current_time_t) 
            
            # The forward price F must be calculated for the remaining T
            current_forward_F = current_spot_S * np.exp((R_RATE - Q_YIELD) * T_remaining)
            
                        # 4. Generate IV vector using the DYNAMIC F and REMAINING T
            iv_vector = simulate_strike_axis_given_base_sabr(
                price_t= simulated_path[t_idx],
                expiry=expiry, 
                alpha=current_alpha, 
                beta=0.1, # Stabilizing beta
                rho=current_rho, 
                nu=current_nu,
                F=current_forward_F, 
                strikes_vector=STRIKES, 
                noise_sd=.00
            )
            strike_expiry_surface.append(iv_vector)

        # Convert the list of strike vectors into a single 2D NumPy array
        vol_surface_at_t = np.stack(strike_expiry_surface)
        market_history.append(vol_surface_at_t)

    frames_data_market = [{'market': surface} for surface in market_history]
        
    generate_mpl_animation(
        filename="simulated_sabr_surface_history.gif", # Use .gif or .mp4 extension
        frames_data=frames_data_market,
        title="Dynamic Market Volatility Surface (Simulated SABR)",
        STRIKE_GRID=STRIKE_GRID,
        EXPIRY_GRID=EXPIRY_GRID,
        show_overlay=False
    )




# def sabr_params_at_T(T, F_T_data, P: Params, mu, sd, beta=1):
#     """
#     Calculates the SABR parameters (alpha, rho, nu) for a specific expiry T 
#     using the 3rd-degree polynomial coefficients (P).
#     """
#     logT = math.log(T)
#     u = _norm_u(logT, mu, sd)
#     u_poly = _poly4(u)
    
#     # Extract coefficients for alpha, rho, nu
#     A_coeffs = np.array([P.a0, P.a1, P.a2, P.a3])
#     R_coeffs = np.array([P.r0, P.r1, P.r2, P.r3])
#     N_coeffs = np.array([P.n0, P.n1, P.n2, P.n3])
    
#     # Calculate transformed parameters
#     alpha_raw = float(np.dot(A_coeffs, u_poly))
#     rho_raw   = float(np.dot(R_coeffs, u_poly))
#     nu_raw    = float(np.dot(N_coeffs, u_poly))
    
#     # Apply constraints/transformations to get final SABR parameters
#     alpha = _softplus(alpha_raw) + 1e-6 # alpha > 0
#     rho   = _tanh(rho_raw)             # -1 < rho < 1
#     nu    = _softplus(nu_raw) + 1e-6   # nu > 0
    
#     return alpha, beta, rho, nu


# # performs 2d SABR fit
# def naive_2dsabr_fit(strike_expiry_meshgrid, beta=1): 
#     """
#     Performs a 2D SABR fit by simultaneously optimizing 12 polynomial coefficients 
#     for alpha, rho, and nu across the expiry grid.

#     Args:
#         strike_expiry_meshgrid (pd.DataFrame): Data containing at least 
#                                                'strike', 'expiry', 'forward', 'iv'.
#         beta (float): The fixed SABR Beta parameter.
    
#     Returns:
#         Params: A named tuple containing the 12 fitted coefficients.
#     """
    
#     # 1. Pre-calculate normalization parameters for the expiry axis
#     logT = np.log(strike_expiry_meshgrid['expiry'])
#     mu = np.mean(logT)
#     sd = np.std(logT)

#     # 2. Define the objective function
#     def objective(params_array):
#         # Convert the array of 12 parameters back into the structured named tuple
#         P = Params(*params_array)
        
#         # Calculate the Mean Squared Error (MSE)
#         error_sum_sq = 0.0
        
#         # Iterate over each point in the data grid
#         for _, row in strike_expiry_meshgrid.iterrows():
#             T = row['expiry']
#             K = row['strike']
#             F = row['forward']
#             market_iv = row['iv']
            
#             # Get the T-dependent SABR parameters
#             alpha, _, rho, nu = sabr_params_at_T(T, F, P, mu, sd, beta)
            
#             # Calculate the model implied volatility (IV)
#             model_iv = sabr_black_iv(F, K, T, alpha, beta, rho, nu)
            
#             # Accumulate squared error
#             error_sum_sq += (model_iv - market_iv)**2
        
#         # Return the Mean Squared Error (MSE)
#         mse = error_sum_sq / len(strike_expiry_meshgrid)
#         return mse

#     initial_guess = np.zeros(12) 
#     bounds = [(-5.0, 5.0)] * 12 
#     print("--- Starting 2D SABR Fit (12 Parameters) ---")
    
#     res = minimize(
#         objective, 
#         initial_guess, 
#         method='Powell',
#         bounds=bounds,
#         options={'maxiter': 1000}
#     )

#     print("--- Fit Results ---")
#     print(f"Success: {res.success}")
#     print(f"Message: {res.message}")
#     print(f"Final MSE: {res.fun:.6f}")
    
#     # 5. Return the 12 fitted coefficients
#     fitted_params = Params(*res.x)
    
#     # Return array (a0,a1,a2,a3, rho0,rho1,rho2,rho3, nu0,nu1,nu2,nu3)
#     return np.array(res.x)




# def fit_and_reconstruct_pca(iv_history_flat, n_components=K_LATENT):
#     """Returns reconstructed data AND the PCA object for shape inspection."""
#     pca = PCA(n_components=n_components)
#     transformed = pca.fit_transform(iv_history_flat) # This is the Reduced Shape
#     reconstructed_flat = pca.inverse_transform(transformed)
#     return reconstructed_flat, transformed




# if __name__ == "__main__":


#     market_history = [] 
#     for t in range(N_SAMPLES):
#         strike_expiry_meshgrid = [simulate_strike_axis_given_base_sabr(expiry, alpha,beta, rho,nu) for expiry in EXPIRIES]

    # market_history_np = np.array(market_history)
    
    # # Flatten for PCA input
    # n_samples, n_str, n_exp = market_history_np.shape
    # flat_history = market_history_np.reshape(n_samples, -1)
    
    # print("\n--- 2. Calculating Models & Shapes ---")
    
    # # A. Naive Model
    # naive_history_list = [simulate_naive_fit(m) for m in market_history]
    # naive_history_np = np.array(naive_history_list)
    
    # # B. PCA Model (Get both reconstruction and latent data)
    # pca_flat_reconstructed, pca_latent_data = fit_and_reconstruct_pca(flat_history, n_components=K_LATENT)
    # pca_history = pca_flat_reconstructed.reshape(n_samples, n_str, n_exp)
    
    # # --- PRINT SHAPES BEFORE/DURING RECONSTRUCTION ---
    # print("\n[DIAGNOSTIC] Data Shapes:")
    # print(f"  1. Original Market Input (Flattened): {flat_history.shape}  <-- 500 time steps x (25 strikes * 15 expiries)")
    # print(f"  2. PCA Latent Space (Reduced):        {pca_latent_data.shape}    <-- Compressed to 3 factors!")
    
    # # --- COMPARE DATA SHAPE BETWEEN NAIVE AND PCA ---
    # print("\n[DIAGNOSTIC] Verifying Reconstruction Shapes:")
    # print(f"  Naive Output Shape: {naive_history_np.shape}")
    # print(f"  PCA Output Shape:   {pca_history.shape}")
    
    # if naive_history_np.shape == pca_history.shape:
    #     print("  >> SUCCESS: Naive and PCA Reconstructions match dimensions.")
    # else:
    #     print("  >> WARNING: Dimension mismatch.")

    # # --- PRINT EUCLIDEAN DISTANCE AFTER RECONSTRUCTION ---
    # # We calculate the Frobenius Norm (Euclidean distance of the matrix differences)
    # dist_naive = np.linalg.norm(market_history_np - naive_history_np)
    # dist_pca = np.linalg.norm(market_history_np - pca_history)

    # print("\n[METRIC] Total Euclidean Distance (L2 Norm) from Ground Truth:")
    # print(f"  1. Distance (Market <-> Naive): {dist_naive:.4f}")
    # print(f"  2. Distance (Market <-> PCA):   {dist_pca:.4f}")
    # print("-" * 60)
    # print("  INTERPRETATION:")
    # print("  * Naive Distance is usually lower because it overfits (includes noise).")
    # print("  * PCA Distance is higher because it filters out noise (smoother surface).")
    # print("-" * 60)
    
    # print("\n--- 3. Rendering HTML Animations ---")

    # # Animation 1: Market
    # data_1 = [{'market': market_history[i], 'model': None} for i in range(N_SAMPLES)]
    # generate_html_animation("01_Market_IV_Bowl.html", data_1, "Market IV (High Convexity)", False)

    # # Animation 2: Naive
    # data_2 = [{'market': market_history[i], 'model': naive_history_list[i]} for i in range(N_SAMPLES)]
    # generate_html_animation("02_Naive_SABR_Fit.html", data_2, "Market vs Naive Fit", True)

    # # Animation 3: PCA
    # data_3 = [{'market': market_history[i], 'model': pca_history[i]} for i in range(N_SAMPLES)]
    # generate_html_animation("03_PCA_Reduced_SABR.html", data_3, "Market vs PCA Fit", True)

    # print("\nDone! Check the 'SABR_Animations' folder.")


