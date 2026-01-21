import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import argparse
import math

# --- 1. MATH HELPERS ---

def _hagan_vol(k, F, T, alpha, beta, rho, nu):
    # Vectorized Hagan Volatility
    if F <= 0 or T <= 0: return 0.0
    
    log_fk = np.log(F / k)
    fk_beta = (F * k)**((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    
    # Expansion for z close to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        sq_term = np.sqrt(1 - 2 * rho * z + z**2)
        chi = np.log((sq_term + z - rho) / (1 - rho))
        z_chi = np.where(np.abs(z) < 1e-5, 1.0, z / chi)
    
    term1 = alpha / (fk_beta * (1 + ((1 - beta)**2 / 24) * log_fk**2 + ((1 - beta)**4 / 1920) * log_fk**4))
    term2 = 1 + (((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) +
                 (0.25 * rho * beta * nu * alpha) / (F**(1 - beta)) +
                 ((2 - 3 * rho**2) / 24) * nu**2) * T
    
    return term1 * z_chi * term2

def get_surface_from_coeffs(coeffs, S, k_grid, dte_grid):
    """
    Decodes 12-vector -> Alpha(t), Rho(t), Nu(t) -> Vol Surface
    """
    if coeffs is None or len(coeffs) != 12:
        return np.zeros((len(k_grid), len(dte_grid)))

    # Extract Polynomial Coeffs: [a0..a3, r0..r3, n0..n3]
    a_c = coeffs[0:4]
    r_c = coeffs[4:8]
    n_c = coeffs[8:12]

    surface = np.zeros((len(k_grid), len(dte_grid)))

    for i, dte in enumerate(dte_grid):
        T = dte / 365.0
        if T < 1e-4: T = 1e-4

        # 1. Evaluate Polynomials at T
        T_vec = np.array([1.0, T, T**2, T**3])
        
        alpha = np.dot(a_c, T_vec)
        rho   = np.dot(r_c, T_vec)
        nu    = np.dot(n_c, T_vec)

        # Clip Physics
        alpha = max(0.001, alpha)
        nu = max(0.001, nu)
        rho = np.clip(rho, -0.999, 0.999)

        # 2. Calculate Vol for entire Strike vector
        K_vec = S * np.exp(k_grid)
        
        for j, K in enumerate(K_vec):
            vol = _hagan_vol(K, S, T, alpha, 1.0, rho, nu) 
            surface[j, i] = vol

    return surface

def extract_market_scatter(market_list, S):
    """
    Robust extraction that handles None, Numpy Arrays, and Lists.
    """
    # 1. Handle None
    if market_list is None:
        return [], [], []

    # 2. Handle Numpy Array (The cause of your previous error)
    if isinstance(market_list, np.ndarray):
        if market_list.size == 0:
            return [], [], []
        market_list = market_list.tolist()

    # 3. Handle Empty List
    if len(market_list) == 0:
        return [], [], []

    xs, ys, zs = [], [], []
    
    for item in market_list:
        # Handle dict vs Row object
        if not isinstance(item, dict):
             strike = getattr(item, 'strike', 0)
             dte = getattr(item, 'dte', 0)
             iv = getattr(item, 'option_volume_weighted', 0)
        else:
             strike = item.get('strike')
             dte = item.get('dte')
             iv = item.get('option_volume_weighted')

        if strike is None or strike <= 0 or S <= 0: continue

        log_mon = np.log(strike / S)
        
        xs.append(dte)
        ys.append(log_mon)
        zs.append(iv)

    return xs, ys, zs


# --- 2. DASH APP ---

def run_dashboard(parquet_file):
    print(f"[INFO] Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Sorting and Striding
    df = df.sort_values('time_entry_ts').reset_index(drop=True)
    df = df.iloc[::5] # Show every 5th frame to reduce lag
    
    # Define Visualization Grid
    GRID_K = np.linspace(-0.3, 0.3, 40) # Log Moneyness (y-axis)
    GRID_DTE = np.linspace(5, 90, 40)   # DTE (x-axis)

    timestamps = df['time_entry_ts'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2(f"SABR Kalman Surface (Left) vs Market Data (Right): {df['ticker'].iloc[0]}", 
                style={'color': '#ddd', 'textAlign': 'center', 'font-family': 'sans-serif'}),
        
        dcc.Graph(id='surface-graph', style={'height': '80vh'}),
        
        html.Div([
            dcc.Slider(
                id='time-slider',
                min=0,
                max=len(df) - 1,
                value=0,
                step=1,
                marks={i: {'label': t, 'style': {'color': '#aaa', 'transform': 'rotate(45deg)'}} 
                       for i, t in enumerate(timestamps) if i % max(1, len(timestamps)//10) == 0},
            )
        ], style={'padding': '40px', 'backgroundColor': '#1e1e1e'})
    ], style={'backgroundColor': '#111', 'height': '100vh', 'margin': '-8px'})

    @app.callback(
        Output('surface-graph', 'figure'),
        Input('time-slider', 'value')
    )
    def update_graph(time_idx):
        row = df.iloc[time_idx]
        S = row['price_ffill_S']
        
        # --- 1. Generate Models (Surfaces) ---
        coeffs_c = np.array(row['SABR_IV_Kalman_C'])
        coeffs_p = np.array(row['SABR_IV_Kalman_P'])
        
        Z_Call_Model = get_surface_from_coeffs(coeffs_c, S, GRID_K, GRID_DTE)
        Z_Put_Model = get_surface_from_coeffs(coeffs_p, S, GRID_K, GRID_DTE)
        
        # --- 2. Extract Data (Scatter) ---
        XC, YC, ZC = extract_market_scatter(row['market_C'], S)
        XP, YP, ZP = extract_market_scatter(row['market_P'], S)

        # --- 3. Build Subplots ---
        # rows=1, cols=2. Both are 3D scenes.
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Fitted Surfaces (Calls=Cyan, Puts=Magenta)', 'Raw Observations (Calls=Green, Puts=Red)')
        )

        # LEFT PLOT: Surfaces
        # Call Surface
        fig.add_trace(go.Surface(
            z=Z_Call_Model, x=GRID_DTE, y=GRID_K,
            colorscale='emrld', showscale=False, opacity=0.6,
            name='Call Model'
        ), row=1, col=1)

        # Put Surface
        fig.add_trace(go.Surface(
            z=Z_Put_Model, x=GRID_DTE, y=GRID_K,
            colorscale='inferno', showscale=False, opacity=0.6,
            name='Put Model'
        ), row=1, col=1)

        # RIGHT PLOT: Scatter Observations
        # Call Scatter
        fig.add_trace(go.Scatter3d(
            x=XC, y=YC, z=ZC,
            mode='markers',
            marker=dict(size=3, color='#00FF00', opacity=0.8), # Green
            name='Call Obs'
        ), row=1, col=2)

        # Put Scatter
        fig.add_trace(go.Scatter3d(
            x=XP, y=YP, z=ZP,
            mode='markers',
            marker=dict(size=3, color='#FF0000', opacity=0.8), # Red
            name='Put Obs'
        ), row=1, col=2)

        # --- Layout Styling ---
        camera = dict(eye=dict(x=1.6, y=1.6, z=1.2))
        scene_dict = dict(
            xaxis_title='DTE',
            yaxis_title='Log Moneyness',
            zaxis_title='IV',
            xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
            zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
            camera=camera
        )

        fig.update_layout(
            title=f"Time: {timestamps[time_idx]} | Spot: ${S:.2f}",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            scene=scene_dict,  # Left Plot
            scene2=scene_dict  # Right Plot
        )
        
        return fig

    app.run(debug=False, port=8050)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", help="Path to Parquet file")
    args = parser.parse_args()
    run_dashboard(args.parquet_file)