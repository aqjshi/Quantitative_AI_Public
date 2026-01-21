import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import argparse
import math
from pretrain import get_ssvi_from_poly, ssvi_variance_function, get_decay_sabr_vectorized, _hagan_vol_vectorized
# --- accessed df strucutre ---
        # processed_surfaces.append({
        #     'time_entry_ts': ts,
        #     'ticker': ticker,
        #     'price_ffill_S': S_t,
        #     'underlying_volume': S_vol_t, 
        #     'risk_free_rate': r_rate,
        #     'market_data_C': minute_market_c, 
        #     'market_data_P': minute_market_p,
        #     'iv_point_C': joint_call_points, 
        #     'iv_point_P': joint_put_points,
        #     'sabr_coeffs_C': sabr_coeffs_c.tolist(),
        #     'sabr_coeffs_P': sabr_coeffs_p.tolist(),
        #     'ssvi_coeffs_C': ssvi_coeffs_c.tolist(),
        #     'ssvi_coeffs_P': ssvi_coeffs_p.tolist(),
        # })

def get_sabr_from_coeffs(coeffs, S, k_grid, dte_grid):
    """
    Decodes 6-vector -> SABR Surface using IMPORTED Helper Logic
    Fixes the 'float is not iterable' error by broadcasting Rho.
    """
    # Safety Check
    if coeffs is None or len(coeffs) != 6:
        return np.zeros((len(k_grid), len(dte_grid)))

    # 1. Convert DTE Grid to Years (Vectorized)
    T_grid = np.array(dte_grid) / 365.0
    
    # 2. Get Time-Dependent Parameters using the Helper
    # alphas: Array [N]
    # rho_val: Scalar Float (Constant) <--- THE CULPRIT
    # nus: Array [N]
    alphas, rho_val, nus = get_decay_sabr_vectorized(coeffs, T_grid)

    # --- FIX: BROADCAST SCALAR RHO TO ARRAY ---
    # We create an array of rhos filled with the single value
    if np.isscalar(rho_val):
        rhos = np.full(len(T_grid), rho_val)
    else:
        rhos = rho_val

    # Initialize Surface
    surface = np.zeros((len(k_grid), len(dte_grid)))
    
    # 3. Calculate Volatility
    F = S # Spot/Forward assumption
    
    # Now zip works because rhos is an array of length N
    for i, (T, alpha, rho, nu) in enumerate(zip(T_grid, alphas, rhos, nus)):
        T_safe = max(T, 1e-5)
        
        # Convert Log-Moneyness Grid to Absolute Strikes
        K_slice = S * np.exp(k_grid)
        
        # Use the IMPORTED Hagan function
        vol_slice = _hagan_vol_vectorized(K_slice, F, T_safe, alpha, 1.0, rho, nu)
        
        surface[:, i] = vol_slice

    return surface

def get_ssvi_surface(coeffs, S, k_grid, dte_grid):
    """
    Decodes 8-vector (SSVI-T) -> SSVI Vol Surface
    coeffs: [rho, gamma, lambda0, lambda1, lambda2, c0, c1, c2] (8 elements)
    """
    # 1. Update coefficient length check
    if coeffs is None or len(coeffs) != 8:
        print(f"Error: SSVI-T coefficient vector must have 8 elements. Found {len(coeffs)}.")
        return np.zeros((len(k_grid), len(dte_grid)))
    
    surface = np.zeros((len(k_grid), len(dte_grid)))
    
    # NOTE: We don't unpack rho, lam, gamma here because they are calculated/retrieved inside the loop.

    for i, dte in enumerate(dte_grid):
        T = dte / 365.0
        T_safe = np.clip(T, 1e-6, T)

        # 2. Get all Time-Dependent Parameters (rho, lambda_T, gamma, theta_T)
        # We must use the 8-parameter logic here.
        rho, lam_T, gamma, theta_T = get_ssvi_from_poly(coeffs, T)
        
        for j, log_k in enumerate(k_grid):
            # SSVI formula uses log-moneyness (k)
            k = log_k 
            
            # 3. Pass the Time-Dependent lambda_T to the variance function
            w_kT = ssvi_variance_function(k, T_safe, theta_T, rho, lam_T, gamma)
            
            # Volatility = sqrt(Variance / Time)
            vol = np.sqrt(w_kT / T_safe)
            surface[j, i] = vol

    return surface



def extract_market_price_scatter(market_list, S):
    """
    Extracts Price data from the raw market list (market_C/market_P).
    Assumes Price is stored under the 'option_volume_weighted' key.
    """
    if market_list is None or len(market_list) == 0:
        return [], [], [] # Returns DTE (X), Log Moneyness (Y), Price (Z)
    
    if isinstance(market_list, np.ndarray):
        market_list = market_list.tolist()

    xs, ys, zs_price = [], [], []
    
    for item in market_list:
        strike = item.get('strike')
        dte = item.get('dte')
        # Price is stored here in the raw market data dictionary
        price = item.get('option_volume_weighted') 
        
        if strike is None or strike <= 0 or S <= 0 or price is None: continue
        
        log_mon = np.log(strike / S)
        
        xs.append(dte)
        ys.append(log_mon)
        zs_price.append(price)

    # print("len(zs_price) : ", len(zs_price))
    # print(np.mean(zs_price))
    return xs, ys, zs_price # Returns DTE, Log Moneyness, Price


def extract_market_iv_scatter(joint_data, S, contract_type):
    """
    Extracts IV data from the joint data structure, printing failed points for debugging.
    The tuple structure in joint_data is: (K [0], IV [1], S [2], T [3], Price [4])
    """
    if joint_data is None or len(joint_data) == 0:
        return [], [], []
    
    if isinstance(joint_data, np.ndarray):
        joint_data = joint_data.tolist()
        
    xs, ys, zs_iv = [], [], []
    total_count = len(joint_data)
    failed_count = 0
    
    for item in joint_data:
        try:
            # Structure is (Strike [0], IV [1], S [2], T [3], Price [4])
            strike = item[0]
            iv = item[1] # <--- IV is at index 1
            T_expiry = item[3]
            price = item[4]
            
            # Recalculate DTE for the X-axis
            dte = T_expiry * 365.0
            
            # --- DEBUGGING CHECK ---
            # Replicate the filter from the processing script (IV > 0.001 and IV < 3.0)
            is_valid = (iv is not None) and (iv > 0.001) and (iv < 3.0)
            
            if is_valid:
                log_mon = np.log(strike / S)
                xs.append(dte)
                ys.append(log_mon)
                zs_iv.append(iv)
            else:
                # 🛑 PRINT THE FAILED ITEMS 🛑
                print(f"!!! {contract_type} IV FILTERED: K={strike:.2f}, DTE={dte:.1f}, Price={price:.4f}, IV={iv:.4f}")
                failed_count += 1
                
        except Exception as e:
            # Handle list indexing errors if the data structure is wrong
            print(f"!!! {contract_type} EXTRACTION ERROR: Item={item}, Error={e}")
            failed_count += 1
    return xs, ys, zs_iv



def run_dashboard(parquet_file):
    print(f"[INFO] Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Sorting and Striding
    df = df.sort_values('time_entry_ts').reset_index(drop=True)
    df = df.iloc[::5] # Show every 5th frame to reduce lag
    
    # Define Visualization Grid (Note: not used in this IV/Price-only view)
    GRID_K = np.linspace(-0.3, 0.3, 40) # Log Moneyness (y-axis)
    GRID_DTE = np.linspace(5, 90, 40)   # DTE (x-axis)

    timestamps = df['time_entry_ts'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2(f"Raw Option Price (Left) vs. Implied Volatility (Right): {df['ticker'].iloc[0]}", 
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
        
        # --- 1. Extract Data ---
        # Prices (for Col 1)
        XC_P, YC_P, ZC_Price = extract_market_price_scatter(row['market_data_C'], S) 
        XP_P, YP_P, ZP_Price = extract_market_price_scatter(row['market_data_P'], S) 
        # IV Points (for Col 2)
        XC_IV, YC_IV, ZC_IV = extract_market_iv_scatter(row['iv_point_C'], S, "C") 
        XP_IV, YP_IV, ZP_IV = extract_market_iv_scatter(row['iv_point_P'], S, "P")


        sabr_coeffs_c = row.get('sabr_coeffs_C', None)
        sabr_surface_c = get_sabr_from_coeffs(sabr_coeffs_c, S, GRID_K, GRID_DTE)

        ssvi_coeffs_c = row.get('ssvi_coeffs_C', None) 
        ssvi_surface_c = get_ssvi_surface(ssvi_coeffs_c, S, GRID_K, GRID_DTE)

        sabr_coeffs_p = row.get('sabr_coeffs_P', None)
        sabr_surface_p= get_sabr_from_coeffs(sabr_coeffs_p, S, GRID_K, GRID_DTE)

        ssvi_coeffs_p = row.get('ssvi_coeffs_P', None) 
        ssvi_surface_p = get_ssvi_surface(ssvi_coeffs_p, S, GRID_K, GRID_DTE)
        
        # =================================================================
        # >>> START OF REQUESTED CHANGE <<<
        # =================================================================
        # Default Title if data is missing
        sabr_subplot_title = "Fitted SABR Surface" 

        # 1. Format Call Params
        call_str = "Calls: N/A"
        if sabr_coeffs_c is not None and len(sabr_coeffs_c) == 6:
            # Mapping: 3: Nu Short, 4: Nu Long, 5: Rho
            a_s_c = sabr_coeffs_c[0]
            a_l_c = sabr_coeffs_c[1]
            k_c = sabr_coeffs_c[2]
            nu_s_c = sabr_coeffs_c[3]
            nu_l_c = sabr_coeffs_c[4]
            rho_c  = sabr_coeffs_c[5]
            # Use double curly braces {{}} to escape LaTeX in f-string
            call_str = f"Calls:a_s_c{a_s_c:.2f}, a_l_c{a_l_c:.2f},k_c{k_c:.2f},  nu_l{nu_l_c:.2f}, nu_S{nu_s_c:.2f}, r{rho_c:.2f}"

        # 2. Format Put Params
        put_str = "Puts: N/A"
        if sabr_coeffs_p is not None and len(sabr_coeffs_p) == 6:
            # Mapping: 3: Nu Short, 4: Nu Long, 5: Rho
            a_s_p = sabr_coeffs_p[0]
            a_l_p = sabr_coeffs_p[1]
            k_p = sabr_coeffs_p[2]
            nu_s_p = sabr_coeffs_p[3]
            nu_l_p = sabr_coeffs_p[4]
            rho_p  = sabr_coeffs_p[5]
            # Use double curly braces {{}} to escape LaTeX in f-string
            put_str = f"Puts:a_s_c{a_s_p:.2f}, a_l_c{a_l_p:.2f},k_c{k_p:.2f},  nu_l{nu_l_p:.2f}, nu_S{nu_s_p:.2f}, r{rho_p:.2f}"

        # Combine with HTML break for stacking
        sabr_subplot_title = f"{call_str}<br>{put_str}"
        # =================================================================
        # >>> END OF REQUESTED CHANGE <<<
        # =================================================================


        # --- 3. Build Subplots (2 Rows, 2 Cols) ---
        # Note: We pass the dynamic title `sabr_subplot_title` into the 3rd slot
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}], 
                   [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Raw Option Price', 'Market IV Points', sabr_subplot_title, 'Fitted SSVI Surface')
        )

        # PLOT 1 (ROW 1, COL 1): Raw Option PRICE
        fig.add_trace(go.Scatter3d(
            x=XC_P, y=YC_P, z=ZC_Price, mode='markers', marker=dict(size=3, color='#00FF00', opacity=0.8), name='Call Price'
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=XP_P, y=YP_P, z=ZP_Price, mode='markers', marker=dict(size=3, color='#FF0000', opacity=0.8), name='Put Price'
        ), row=1, col=1)

        # PLOT 2 (ROW 1, COL 2): Market Implied VOLATILITY (Scatter)
        fig.add_trace(go.Scatter3d(
            x=XC_IV, y=YC_IV, z=ZC_IV, mode='markers', marker=dict(size=3, color='#00FF00', opacity=0.8), name='Call IV', showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=XP_IV, y=YP_IV, z=ZP_IV, mode='markers', marker=dict(size=3, color='#FF0000', opacity=0.8), name='Put IV', showlegend=False
        ), row=1, col=2)
        
        # PLOT 3 (ROW 2, COL 1): Fitted SABR Surface (Continuous)
        fig.add_trace(go.Surface(
            x=GRID_DTE, y=GRID_K, z=sabr_surface_c, # Z = [Strike, DTE] -> No Transpose
            colorscale='Viridis', opacity=0.5, showscale=False, name='SABR Surface'
        ), row=2, col=1)
        fig.add_trace(go.Surface(
            x=GRID_DTE, y=GRID_K, z=sabr_surface_p, # Z = [Strike, DTE] -> No Transpose
            colorscale='Inferno', opacity=0.5, showscale=False, name='SABR Surface'
        ), row=2, col=1)

        # PLOT 4 (ROW 2, COL 2): Fitted SSVI Surface (Continuous)
        fig.add_trace(go.Surface(
            x=GRID_DTE, y=GRID_K, z=ssvi_surface_c, # Z = [Strike, DTE] -> No Transpose
            colorscale='Viridis', opacity=0.5, showscale=False, name='SSVI Surface'
        ), row=2, col=2)
        fig.add_trace(go.Surface(
            x=GRID_DTE, y=GRID_K, z=ssvi_surface_p, # Z = [Strike, DTE] -> No Transpose
            colorscale='Inferno', opacity=0.5, showscale=False, name='SSVI Surface'
        ), row=2, col=2)

        # --- 4. Layout Styling ---
        camera = dict(eye=dict(x=1.6, y=1.6, z=1.2))
        
        def get_scene_dict(z_title, camera_settings):
            return dict(
                xaxis_title='DTE', yaxis_title='Log Moneyness', zaxis_title=z_title,
                xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True),
                camera=camera_settings
            )

        fig.update_layout(
            title=f"Surface Model Comparison | Time: {timestamps[time_idx]} | Spot: ${S:.2f}",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            scene1=get_scene_dict('PRICE ($)', camera),   # Top Left
            scene2=get_scene_dict('IV (Market)', camera), # Top Right
            scene3=get_scene_dict('IV (SABR Fit)', camera),# Bottom Left
            scene4=get_scene_dict('IV (SSVI Fit)', camera) # Bottom Right
        )
        
        return fig

    app.run(debug=False, port=8050)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", help="Path to Parquet file")
    args = parser.parse_args()
    run_dashboard(args.parquet_file)