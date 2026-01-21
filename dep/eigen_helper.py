import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings("ignore")

import math
from scipy.optimize import minimize
from collections import namedtuple
Params = namedtuple('Params', ['a0', 'a1', 'a2', 'a3', 'r0', 'r1', 'r2', 'r3', 'n0', 'n1', 'n2', 'n3'])

def _poly4(u): 
    return np.array([1.0, u, u*u, u*u*u])

def _softplus(x): 
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)

def _tanh(x):
    return math.tanh(x)

def _norm_u(logT, mu, sd): 
    return (logT - mu) / (sd + 1e-12)






def geometric_brownian_motion(S0, mu, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    dW = np.random.normal(0, np.sqrt(dt), N)
    S = np.zeros(N + 1)
    S[0] = S0
    
    for i in range(1, N + 1):
        drift = (mu - 0.5 * sigma**2) * dt
        volatility_term = sigma * dW[i-1]
        S[i] = S[i-1] * np.exp(drift + volatility_term)
        
    return S




def sabr_black_iv(F, K, T, alpha, beta, rho, nu):
    """
    Calculates the SABR Black Implied Volatility (IV) using the Hagan 
    et al. (2002) approximation with robust numerical stability checks.
    """
    # Use 1e-12 as a minimum time to prevent division by zero near expiration
    T = max(T, 1e-12) 
    
    # 1. Check for AT THE MONEY (ATM) case (K = F)
    # Use a small tolerance (epsilon) for near-ATM pricing
    if abs(K - F) < 1e-6:
        # Simplest ATM formula
        gamma_atm = beta * (beta - 2) * alpha**2 / (24 * F**(2 - 2 * beta))
        term_rho_nu = rho * beta * nu * alpha / (4 * F**(1 - beta))
        term_nu_sq = nu**2 * (2 - 3 * rho**2) / 24
        
        IV_ATM = (alpha / F**(1 - beta)) * (1 + (gamma_atm + term_rho_nu + term_nu_sq) * T)
        return IV_ATM
    
    # 2. OUT OF THE MONEY (OTM) case (K != F)
    else:
        # Pre-calculate common terms
        F_pow = F**(1 - beta)
        K_pow = K**(1 - beta)
        
        # Check for near-Beta=1: If (1 - beta) is too small, use log approximation
        # (This is a safer alternative to the IF/ELSE structure inside your previous Z calculation)
        if abs(1 - beta) < 1e-6:
            log_FK = np.log(F / K)
            # Term Z uses the log approximation for beta near 1
            z_top = log_FK
            z_bot = alpha
        else:
            # Full power calculation for beta != 1
            z_top = (F_pow - K_pow) / (1 - beta)
            z_bot = alpha
        
        # Term Z calculation
        z = nu / z_bot * z_top
        
        # Term chi(z) - The stable hyperbolic sine approximation
        # The key fix: This mathematically smooths the function around z=0,
        # preventing the "crease" caused by the simple Taylor expansion approximation.
        if abs(z) >= 1e-6:
            # Use the definition of asinh(x)/x to handle the singularity at z=0
            chi_z = z / np.sinh(z) * (1 + (z**2) / 6) # Standard stable form, or use asinh below
            
            # **Alternative stable form (often used in modern implementations):**
            # chi_z = np.log((1 - 2 * rho * z + z**2)**0.5 + rho * z) / z
            
            # For simplicity and stability, we use the standard (asinh) formulation:
            chi_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)**2 / (1 - rho)**2) / z
        else:
            chi_z = 1.0
            
        # Term 1: SABR Alpha scaled by the volatility function
        V_alpha = alpha / ( (F * K)**((1 - beta) / 2) )
        
        # Term 2: Mean correction term (ATM gamma correction)
        gamma = (1 - beta)**2 / (24 * (F * K)**(1 - beta))
        
        # Term 3 & 4: Skew and Smile correction (Terms 3 and 4 combined into smile_skew)
        smile_skew = rho * beta * nu / (4 * alpha * (F * K)**((1 - beta)/2)) + nu**2 * (2 - 3 * rho**2) / 24
        
        # Final Volatility approximation (IV)
        # Combine the main term (V_alpha) with the correction terms
        IV = (alpha / (F * K)**((1 - beta) / 2)) * chi_z * (1 + smile_skew * T)
        
        # Final safety check
        return np.maximum(1e-6, IV)

def sabr_lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    """
    Hagan's 2002 SABR log-normal volatility approximation.
    """
    # Handle the ATM case closely to avoid division by zero
    if abs(f - k) < 1e-5:
        term1 = alpha / (f ** (1 - beta))
        term2 = ((1 - beta) ** 2) / 24 * (alpha ** 2) / (f ** (2 - 2 * beta))
        term3 = (rho * beta * volvol * alpha) / (4 * (f ** (1 - beta)))
        term4 = (2 - 3 * rho ** 2) / 24 * volvol ** 2
        return term1 * (1 + (term2 + term3 + term4) * t)

    # Standard Case (ITM/OTM)
    log_fk = np.log(f / k)
    fk_beta = (f * k) ** ((1 - beta) / 2)
    z = (volvol / alpha) * fk_beta * log_fk
    
    # x(z) function
    # We use a safe check for sqrt to avoid small numerical errors
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    
    numerator = alpha * z
    denominator_1 = fk_beta * (1 + ((1 - beta) ** 2) / 24 * log_fk ** 2 + ((1 - beta) ** 4) / 1920 * log_fk ** 4)
    denominator = denominator_1 * x_z
    
    # Small time expansion term
    term_bracket = (
        ((1 - beta) ** 2) / 24 * (alpha ** 2) / ((f * k) ** (1 - beta)) +
        (rho * beta * volvol * alpha) / (4 * fk_beta) +
        (2 - 3 * rho ** 2) / 24 * volvol ** 2
    )
    
    return (numerator / denominator) * (1 + term_bracket * t)
# frozen term, frozen tte, should return a vector size of  STRIKES linespace 
# The modified simulate_strike_axis_given_base_sabr function
def simulate_strike_axis_given_base_sabr(price_t, expiry, alpha, beta, rho, nu, F, strikes_vector, noise_sd=0.002, max_expiry=2.0):
    
    iv_vector = np.zeros_like(strikes_vector, dtype=float)
    dupire_nu = nu * ( max_expiry- expiry )
    for i, K in enumerate(strikes_vector):
        iv = sabr_lognormal_vol(k=K, f=F, t=expiry, alpha=alpha, beta=beta, rho=rho, volvol=dupire_nu*10)
        iv_vector[i] = iv
    noise = np.random.normal(0, noise_sd, iv_vector.shape)
    iv_vector_noisy = iv_vector + noise



    return iv_vector_noisy
import matplotlib.pyplot as plt
from matplotlib import animation
# --- Global State Variable ---
is_paused = False

def toggle_pause(event):
    """Callback function to toggle the pause state on spacebar press."""
    global is_paused
    # Check if the pressed key is the spacebar (' ')
    if event.key == ' ':
        is_paused = not is_paused
        print(f"Animation {'PAUSED' if is_paused else 'RESUMED'}")

def generate_mpl_animation(filename, frames_data, title, STRIKE_GRID, EXPIRY_GRID, show_overlay=False):
    """
    Generates a 3D surface animation using Matplotlib with interactive controls.

    (Docstring and Args omitted for brevity)
    """
    print(f"Generating Matplotlib animation {filename}...")
    
    # 1. Setup the Figure and 3D Axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine the Z-range for stable plotting 
    z_min, z_max = 0.0, 0.6 
    
    # 2. Initial Plot
    init_market = frames_data[0]['market']
        
    # FIX: Remove the comma. Assigns the single surface object directly.
    initial_surf = ax.plot_surface( 
        STRIKE_GRID, 
        EXPIRY_GRID, 
        init_market, 
        cmap=plt.cm.viridis, 
        linewidth=0, 
        antialiased=False
    )
    
    # Store the surface in a mutable list/container so it can be updated
    # This assumes you are using the list-based container fix from the previous response.
    surf_container = [initial_surf]
    fig.colorbar(initial_surf, shrink=0.5, aspect=5, label='Implied Volatility')
    

    
    # 4. Updated Function for Animation
    def update_surface(frame_index):
        """Updates the Z-data for the next frame."""
        
        global is_paused 
        
        # --- FIX: Access the current surface object from the container ---
        if is_paused:
            # Return the existing surface object without modification, freezing the frame.
            return surf_container[0], 

        # Access the volatility surface for the current time step
        z_data = frames_data[frame_index]['market']
        
        # Clear the previous surface (necessary for Matplotlib animation)
        ax.cla() 
        
        # Redraw the axes and set limits (must be done after ax.cla())
        # ... (Redrawing axes code is unchanged) ...
        ax.set_title(f"{title}\nTime Step: {frame_index}")
        ax.set_xlabel('Strike (K)')
        ax.set_ylabel('Expiry (T)')
        ax.set_zlabel('Implied Volatility')
        ax.set_zlim(z_min, z_max)

        # Draw the new surface
        new_surf = ax.plot_surface(
            STRIKE_GRID, 
            EXPIRY_GRID, 
            z_data, 
            cmap=plt.cm.viridis, 
            linewidth=0, 
            antialiased=False
        )
        
        surf_container[0] = new_surf 
        
        # Return the new artist
        return new_surf,
        
    # Connect the Keyboard Listener
    fig.canvas.mpl_connect('key_press_event', toggle_pause) 
    
    # 5. Create the Animation object
    ani = animation.FuncAnimation(
        fig, 
        update_surface, 
        frames=len(frames_data), 
        interval=50, 
        blit=False, 
        repeat=True
    )
    
    # 6. Display the interactive animation window.
    plt.show()

    return fig