import numpy as np
import time
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
from typing import List, Tuple, Callable, Dict, Any

# --- Type Hinting for Clarity ---
# A Prediction from the ML model: (Timestamp, Price)
Prediction = Tuple[float, float]
# A prediction stored in the pool: (Timestamp, Price, RemainingCycles)
StoredPrediction = Tuple[float, float, int]
# The main data pool
PredictionPool = List[StoredPrediction]
# The KDE function h(x)
KdeFunction = Callable[[np.ndarray], np.ndarray]

def calculate_initial_lifespan(max_forecast_horizon_sec: float, 
                               cycle_duration_sec: float) -> int:
    return int(np.floor(max_forecast_horizon_sec / cycle_duration_sec))

def calculate_weight(remaining_cycles: int, init_cycles: int) -> float:
    """
    Calculates the weight w_i for a single prediction.
    
    Ref: Section 1.3, Eq. for w_i
    $$ w_i = \frac{\text{init}_{\text{cycles}} - \text{experienced}_{\text{cycles}}}{\text{init}_{\text{cycles}}} $$
    This is equivalent to:
    $$ w_i = \frac{\text{remaining}_{\text{cycles}}}{\text{init}_{\text{cycles}}} $$
    """
    if init_cycles == 0:
        return 0.0
    return max(0.0, remaining_cycles / init_cycles)

def update_prediction_pool(current_pool: PredictionPool, 
                             new_prediction: Prediction, 
                             init_cycles: int) -> PredictionPool:
    """
    A pure function that ages the pool, removes the dead, and adds the new.
    This runs once per "Cycle".
    
    Ref: Section 1.3, 1.5, 1.6
    """
    # 1. Age all existing predictions by one cycle
    aged_pool = [
        (t, p, remaining - 1) for (t, p, remaining) in current_pool
    ]
    
    # 2. Filter out predictions whose lifespan has expired
    active_pool = [
        pred for pred in aged_pool if pred[2] > 0
    ]
    
    # 3. Add the new prediction with its full initial lifespan
    new_pred_stored = (new_prediction[0], new_prediction[1], init_cycles)
    active_pool.append(new_pred_stored)
    
    return active_pool

# ==============================================================================
# SECTION 2: WEIGHTED 2D KDE (THE "QUERY")
# ==============================================================================

def get_kde_inputs(pool: PredictionPool, 
                     init_cycles: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper to extract data and weights from the pool for KDE processing.
    """
    # Unzip the pool
    data = np.array([(t, p) for (t, p, rem) in pool])  # (N, 2) array
    remaining_cycles = np.array([rem for (t, p, rem) in pool]) # (N,) array
    
    # Calculate weights w_i
    weights = np.array([
        calculate_weight(rem, init_cycles) for rem in remaining_cycles
    ])
    
    return data, weights

def calculate_scalar_bandwidth(data: np.ndarray, n: int) -> float:
    """
    Calculates the scalar bandwidth 'b' using Scott's Rule.
    
    Ref: Section 1.8
    $$ b = \sigma \cdot N^{-1/6} $$
    
    Note: Assumes sigma is the average standard deviation of the two dimensions.
    """
    if n < 2:
        return 1.0  # Avoid division by zero
        
    # Calculate standard deviation along each axis (Time, Price)
    std_devs = np.std(data, axis=0)
    
    # Use the average standard deviation for sigma
    sigma = np.mean(std_devs)
    
    return sigma * (n ** (-1.0 / 6.0))

def define_weighted_kde_plane(pool: PredictionPool, 
                              init_cycles: int) -> Tuple[KdeFunction, np.ndarray]:
    """
    This is the main "Query" function.
    It builds and returns the weighted 2D KDE function h(x).
    
    Ref: Section 1.8, 1.10
    """
    # 1. Get the raw data (x_i) and weights (w_i)
    # data is (N, 2), weights is (N,)
    data, weights = get_kde_inputs(pool, init_cycles)
    n = len(data)
    
    if n < 2:
        # Not enough data to compute covariance, return a dummy function
        return lambda x: np.zeros(x.shape[0]), data

    # 2. Normalize weights for the summation (this handles the 1 / sum(w_i) term)
    # $$ h(\mathbf{x}) = \sum_{i=1}^{N} \left( \frac{w_i}{\sum w_j} \right) K_H(\mathbf{x} - \mathbf{x}_i) $$
    total_weight = np.sum(weights)
    if total_weight == 0:
        return lambda x: np.zeros(x.shape[0]), data
        
    normalized_weights = weights / total_weight

    # 3. Calculate Bandwidth Matrix H
    # $$ H = b^2 \cdot \Sigma $$
    
    # Calculate weighted empirical covariance matrix Sigma (2x2)
    # We must use data.T (shape 2, N) for np.cov
    try:
        sigma_matrix = np.cov(data.T, aweights=weights)
    except np.linalg.LinAlgError:
        # Fallback if covariance is singular
        sigma_matrix = np.cov(data.T) # Unweighted
        
    if np.any(np.isnan(sigma_matrix)):
        # Handle case where covariance fails
        sigma_matrix = np.eye(2) 

    # Calculate scalar bandwidth b
    scalar_b = calculate_scalar_bandwidth(data, n)
    
    # Calculate H
    h_matrix = (scalar_b ** 2) * sigma_matrix
    
    # Ensure H is positive semi-definite for the kernel
    # Add small jitter to diagonal if it's singular
    if np.linalg.det(h_matrix) == 0:
        h_matrix += np.eye(2) * 1e-9

    # 4. Define and return the KDE function h(x)
    # This function "closes over" the data, weights, and H matrix.
    def h(x_eval: np.ndarray) -> np.ndarray:
        """
        The weighted KDE function h(x).
        x_eval is an (M, 2) array of (time, price) points to evaluate.
        """
        # M is the number of points we want to evaluate
        m = x_eval.shape[0]
        # Initialize output density array
        density_sum = np.zeros(m)
        
        # This is the literal implementation of the summation:
        for i in range(n):
            xi = data[i]          # x_i
            wi_norm = normalized_weights[i] # w_i / sum(w)
            
            # K_H(x - x_i) is a Gaussian kernel centered at x_i
            # with bandwidth matrix H
            try:
                kernel = multivariate_normal(mean=xi, cov=h_matrix)
                # Add this kernel's contribution, scaled by its weight
                density_sum += wi_norm * kernel.pdf(x_eval)
            except np.linalg.LinAlgError:
                # Skip this point if its kernel is invalid
                continue
                
        return density_sum

    return h, data

# ==============================================================================
# SECTION 3: PEAK IDENTIFICATION (from the Plane)
# ==============================================================================

def evaluate_plane_on_grid(kde_func: KdeFunction, 
                             data: np.ndarray, 
                             grid_size: int = 100
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the KDE function h(x) over a 2D grid to create the 
    drawable "plane" (a 2D array of density values).
    
    Ref: Section 1.9
    """
    if len(data) < 2:
        return np.array([]), np.array([]), np.array([])
        
    # Get data boundaries to build the grid
    t_min, p_min = np.min(data, axis=0)
    t_max, p_max = np.max(data, axis=0)
    
    # Create the grid axes
    t_axis = np.linspace(t_min, t_max, grid_size)
    p_axis = np.linspace(p_min, p_max, grid_size)
    
    # Create meshgrid
    T, P = np.meshgrid(t_axis, p_axis)
    
    # Ravel the grid into an (M, 2) array for the KDE function
    # M = grid_size * grid_size
    grid_to_evaluate = np.vstack([T.ravel(), P.ravel()]).T
    
    # Evaluate h(x) on all grid points
    Z = kde_func(grid_to_evaluate)
    
    # Reshape Z back into the (grid_size, grid_size) plane
    Z_plane = Z.reshape(T.shape)
    
    return T, P, Z_plane

def find_peaks_on_plane(Z_plane: np.ndarray, 
                          min_dist: int = 5
                         ) -> np.ndarray:
    """
    Identifies the local maxima (peaks) on the 2D density plane.
    
    Ref: Section 1.9
    
    Uses scikit-image's peak_local_max.
    Returns a list of (row, col) indices of the peaks.
    """
    # peak_local_max finds local peaks in an image (our Z_plane)
    peak_indices = peak_local_max(Z_plane, min_distance=min_dist)
    return peak_indices

# ==============================================================================
# SECTION 4: SKELETON SIMULATION
# ==============================================================================

def mock_ml_model(current_time: float) -> Prediction:
    """
    Placeholder for the deep learning model.
    Generates one (Time, Price) prediction.
    """
    # Simulate a noisy prediction
    pred_time = current_time + np.random.uniform(30, 300) # Predict 30-300s out
    pred_price = 100 + np.sin(current_time / 60) * 5 + np.random.randn()
    return (pred_time, pred_price)

# --- Main Simulation ---
if __name__ == "__main__":
    
    # --- System Hyperparameters ---
    MAX_FORECAST_HORIZON_SEC = 600.0  # (H) Max forecast window (10 min)
    CYCLE_DURATION_SEC = 1.0         # (tau_cycle) 1 prediction per sec
    
    # --- Initialization ---
    INIT_CYCLES = calculate_initial_lifespan(
        MAX_FORECAST_HORIZON_SEC, 
        CYCLE_DURATION_SEC
    )
    
    prediction_pool: PredictionPool = []
    
    print(f"System Initialized. init_cycles = {INIT_CYCLES}")
    print("Running simulation...")

    # --- Main Loop (emulates time) ---
    for cycle in range(200): # Run for 200 cycles
        current_time_sec = time.time()
        
        # 1. Get new prediction from "machine"
        new_prediction = mock_ml_model(current_time_sec)
        
        # 2. Update the pool (functional style: pass old, get new)
        prediction_pool = update_prediction_pool(
            prediction_pool, 
            new_prediction, 
            INIT_CYCLES
        )
        
        # 3. Perform a "Query" every 10 cycles (and if pool is big enough)
        if cycle % 10 == 0 and len(prediction_pool) > 20:
            print(f"\n--- CYCLE {cycle} | POOL SIZE: {len(prediction_pool)} ---")
            
            # 3a. Define the plane h(x)
            kde_func, data = define_weighted_kde_plane(prediction_pool, INIT_CYCLES)
            
            # 3b. Evaluate h(x) on a grid to get the Z-plane
            T, P, Z = evaluate_plane_on_grid(kde_func, data, grid_size=50)
            
            if Z.size == 0:
                print("Not enough data to build plane.")
                continue

            # 3c. Find the peaks
            peak_indices = find_peaks_on_plane(Z, min_dist=3)
            
            print(f"Found {len(peak_indices)} peaks.")
            
            # Print peak details
            for idx in peak_indices:
                row, col = idx
                peak_time = T[row, col]
                peak_price = P[row, col]
                peak_height = Z[row, col]
                
                print(f"  > Peak at T={peak_time:.0f}, P=${peak_price:.2f} (Height={peak_height:.4f})")
        
        # time.sleep(CYCLE_DURATION_SEC) # Uncomment to run in real-time
        pass # Run as fast as possible for demo

    print("\nSimulation complete.")