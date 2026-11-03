import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from statsmodels.tsa.statespace.mlemodel import MLEModel
import statsmodels.api as sm
import re



 
def fit_mixed_frequency_kalman(
    df_stationary: pd.DataFrame,
    transform_metadata: Dict[str, Dict],
    train_start: pd.Timestamp,
    eval_start: pd.Timestamp,
    config: Dict
) -> Dict:
    # 1. Slice training window
    train_mask = (df_stationary.index >= train_start) & (df_stationary.index <= eval_start)
    wide_train = df_stationary.loc[train_mask].copy()
    wide_train.index = pd.to_datetime(wide_train.index)
    wide_train = wide_train.sort_index()

    cols = list(wide_train.columns)
    n = len(cols)
    K = config.get('dynamic_factor_n_factors', 1)
    p_order = config.get('dynamic_factor_lag_order', 1)
    S = p_order+ 1
    freq_list = [transform_metadata.get(str(c), {}).get("frequency_short") for c in cols]

    # 2. Standardize strictly on training sample
    mean = wide_train.mean()
    std = wide_train.std()
    std[std == 0] = 1.0
    wide_scaled_train = (wide_train - mean) / std

    # 3. Instantiate & Fit DynamicFactorMQ (EM algorithm handles mixed frequencies natively)
    model = sm.tsa.DynamicFactorMQ(
        endog=wide_scaled_train,
        factors=K,
        factor_orders=p_order
    )

    result = model.fit(
        maxiter=config.get("dynamic_factor_statespace_maxiter", 500),
        tolerance=config.get("dynamic_factor_statespace_tolerance", 1e-5),
        disp=False
    )

    print(f"\n[DynamicFactorMQ] (K={K}, p={p_order})")
    print(result.summary())
    H_fitted = result.model.ssm["design"]  # Shape: (N, k_states)
    
    # Extract contemporary factor columns from H (every S-th column, where S = lag_order + 1)
    contemp_cols = [f * S for f in range(K)]
    loadings_arr = H_fitted[:, contemp_cols].T  # Shape: (K, N)
    
    # Convert to a DataFrame matching your columns
    loadings_df = pd.DataFrame(loadings_arr.T, index=cols, columns=[f"Factor_{i}" for i in range(K)])

    # Calculate bounded signal share (R^2 per series)
    signal_share = {}
    smoothed_factors = result.factors.smoothed.values  # Shape: (T, K)
    factor_cov = np.atleast_2d(np.cov(smoothed_factors, rowvar=False))

    for i, c in enumerate(cols):
        l_i = loadings_arr[:, i]  # Vector (K,)
        sig_var = float(l_i.T @ factor_cov @ l_i)
        
        # Total variance of standardized series is ~1.0
        signal_share[c] = float(np.clip(sig_var / (sig_var + 1.0), 0.0, 1.0))

    # print("\nTRAIN SIGNAL SHARE (common-factor R^2 per series)")
    # for c in cols:
    #     print(f"  {str(c):<18} {signal_share[c]:.4f}")



    # Extract fitted state-space matrices
    H_fitted = result.model.ssm["design"]
    F_fitted = result.model.ssm["transition"]
    Q_fitted = result.model.ssm["state_cov"]
    R_fitted = result.model.ssm["selection"]


    # Extract factor transition parameters directly from result params
    factor_rhos = []
 
    for k in range(K):
        rho_k = float(F_fitted[k, k])
        factor_rhos.append(rho_k)
        # print(f"Factor {k} AR(1) Rho (Transition Coefficient): {rho_k:.4f}")


    return {
        "params": result.params,
        "loadings": loadings_df,
        "signal_share": signal_share,
        "factor_rhos": factor_rhos,  
        "obs_var_hat": np.ones(n),
        "H": H_fitted,
        "F": F_fitted,
        "Q": Q_fitted,
        "R": R_fitted, 
        "mean": mean,
        "std": std,
        "signal_share": signal_share,
        "freq_list": freq_list,
        "cols": cols,
        "train_result": result
    }


def parse_model_diagnostics(mf_fit) -> dict:
    """
    Extracts log_likelihood, aic, bic, hqic, and em_iterations from a 
    Statsmodels DynamicFactorMQ fit object or its summary() string.
    """
    summary_txt = str(mf_fit['train_result'].summary())
    m = re.search(r"Log Likelihood\s+([-\d\.]+)", summary_txt)
    log_likelihood = float(m.group(1)) if m else 0.0
    

    m = re.search(r"\bAIC\s+([-\d\.]+)", summary_txt)
    aic = float(m.group(1)) if m else 0.0

    m = re.search(r"\bBIC\s+([-\d\.]+)", summary_txt)
    bic = float(m.group(1)) if m else 0.0

    m = re.search(r"\bHQIC\s+([-\d\.]+)", summary_txt)
    hqic = float(m.group(1)) if m else 0.0
    
    m = re.search(r"EM Iterations\s+(\d+)", summary_txt)
    em_iterations = int(m.group(1)) if m else 0
            
    # Fallback guarantees
    return {
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "hqic": float(hqic),
        "em_iterations": int(em_iterations)
    }