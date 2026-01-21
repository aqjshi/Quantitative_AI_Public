import argparse
from datetime import date, datetime, timedelta
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from multiprocessing import Pool
from pretrain_helper import (
    _business_days,
    _get_rate,
    _get_sod,
    get_company_id,
    load_minutes,
    load_underlying,
    calculate_iv_american_call,
    calculate_iv_american_put,
)
from db import SQL_USER, SQL_PWD, SQL_HOST, SQL_PORT, SQL_DB_NAME
import warnings
from scipy.optimize import minimize, curve_fit

load_dotenv()

SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{SQL_USER}:{quote_plus(SQL_PWD)}@{SQL_HOST}:{SQL_PORT}/{SQL_DB_NAME}"
)


def _hagan_vol_vectorized(k, F, T, alpha, beta, rho, nu):
    """
    Vectorized Hagan 2002 SABR Log-Normal Volatility.
    Accepts numpy arrays for inputs.
    """
    k = np.asarray(k, dtype=float)
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    nu = np.asarray(nu, dtype=float)
    rho = np.asarray(rho, dtype=float)

    valid_mask = (F > 0) & (k > 0) & (T > 0) & (alpha > 0) & (nu > 0)

    F_s = np.where(valid_mask, F, 1.0)
    k_s = np.where(valid_mask, k, 1.0)
    T_s = np.where(valid_mask, T, 1.0)
    alpha_s = np.where(valid_mask, alpha, 0.1)
    nu_s = np.where(valid_mask, nu, 0.1)
    rho_s = np.clip(rho, -0.999, 0.999)

    log_fk = np.log(F_s / k_s)
    fk_beta = (F_s * k_s) ** ((1 - beta) / 2)
    z = (nu_s / alpha_s) * fk_beta * log_fk

    sq_term = 1 - 2 * rho_s * z + z**2
    sq_term = np.maximum(sq_term, 0.0)

    numerator = np.sqrt(sq_term) + z - rho_s
    denominator = 1 - rho_s
    log_arg = numerator / np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    log_arg = np.maximum(log_arg, 1e-18)

    chi = np.log(log_arg)
    is_small_z = np.abs(z) < 1e-5

    chi_safe = np.where(np.abs(chi) < 1e-10, 1.0, chi)
    z_over_chi_calc = z / chi_safe
    z_over_chi = np.where(is_small_z, 1.0, z_over_chi_calc)

    base_vol = alpha_s / fk_beta

    taylor_denom = (
        1
        + ((1 - beta) ** 2 / 24) * log_fk**2
        + ((1 - beta) ** 4 / 1920) * log_fk**4
    )

    vol_atm = base_vol / taylor_denom
    vol_otm = base_vol * z_over_chi
    vol_T0 = np.where(is_small_z, vol_atm, vol_otm)

    term2 = 1 + (
        ((1 - beta) ** 2 / 24) * (alpha_s**2 / (F_s ** (2 - 2 * beta)))
        + (0.25 * rho_s * beta * nu_s * alpha_s) / (F_s ** (1 - beta))
        + ((2 - 3 * rho_s**2) / 24) * nu_s**2
    ) * T_s

    vol = vol_T0 * term2
    return np.where(valid_mask, vol, 0.0)


def get_decay_sabr_vectorized(coeffs, T):
    alpha_s, alpha_l, k = coeffs[0], coeffs[1], coeffs[2]
    nu_s, nu_l = coeffs[3], coeffs[4]
    rho = coeffs[5]

    T_safe = np.maximum(T, 1e-5)
    t_min_approx = np.min(T_safe)
    delta_t = np.maximum(T_safe - t_min_approx, 0.0)

    decay_alpha = 1.0 / (1.0 + k * np.sqrt(delta_t))
    decay_nu = np.exp(-k * delta_t)

    alpha = alpha_l + (alpha_s - alpha_l) * decay_alpha
    nu = nu_l + (nu_s - nu_l) * decay_nu

    return alpha, rho, nu


def estimate_sabr_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr):
    default_x0 = np.array([0.5, 0.3, 0.8, 1.0, 0.4, -0.1])

    try:
        unique_ts = np.unique(T_arr)
        unique_ts.sort()
        if len(unique_ts) < 2:
            return default_x0

        t_first = unique_ts[0]
        t_second = unique_ts[1]
        idx_mid = len(unique_ts) // 2
        t_latter_half = unique_ts[idx_mid:]

        def get_slice_metrics(target_t):
            mask = T_arr == target_t
            if not np.any(mask):
                return None
            k_sub = K_arr[mask]
            v_sub = v_mkt_arr[mask]
            s_val = S_t_arr[mask][0]
            if len(k_sub) < 3:
                return None

            idx_atm = np.argmin(np.abs(k_sub - s_val))
            atm_vol = v_sub[idx_atm]

            log_k = np.log(k_sub / s_val)
            target_low, target_high = -0.15, 0.15
            idx_low = np.argmin(np.abs(log_k - target_low))
            idx_high = np.argmin(np.abs(log_k - target_high))
            if idx_low == idx_high:
                idx_low, idx_high = 0, len(k_sub) - 1

            k_min = k_sub[idx_low]
            k_max = k_sub[idx_high]
            v_min = v_sub[idx_low]
            v_max = v_sub[idx_high]

            log_k_diff = np.log(k_max / k_min)
            slope = (v_max - v_min) / log_k_diff if abs(log_k_diff) > 1e-3 else 0.0
            return {"atm_vol": atm_vol, "slope": slope}

        m_first = get_slice_metrics(t_first)
        m_second = get_slice_metrics(t_second)
        if not m_first:
            return default_x0

        long_alphas, long_slopes = [], []
        for t_l in t_latter_half:
            m = get_slice_metrics(t_l)
            if m:
                long_alphas.append(m["atm_vol"])
                long_slopes.append(m["slope"])

        if not long_alphas:
            est_alpha_l = m_first["atm_vol"] * 0.8
            est_nu_l = 0.3
        else:
            est_alpha_l = np.median(long_alphas)
            median_slope_l = np.median(long_slopes)
            est_nu_l = (2.0 * abs(median_slope_l)) / max(est_alpha_l, 0.01)

        est_alpha_s = m_first["atm_vol"]

        est_k = 1.5
        denom = est_alpha_s - est_alpha_l
        if abs(denom) > 0.01 and m_second is not None:
            alpha_2 = m_second["atm_vol"]
            ratio = (alpha_2 - est_alpha_l) / denom
            if 0.01 < ratio < 0.99:
                est_k = -np.log(ratio) / max(t_second, 0.001)
            elif ratio <= 0.01:
                est_k = 5.0
            elif ratio >= 0.99:
                est_k = 0.1

        est_nu_s = (2.0 * abs(m_first["slope"])) / max(est_alpha_s, 0.01)
        est_rho = -0.6 if m_first["slope"] < 0 else 0.1

        est_k = np.clip(est_k, 0.1, 8.0)
        est_nu_s = np.clip(est_nu_s, 0.1, 3.0)
        est_nu_l = np.clip(est_nu_l, 0.1, 1.5)

        return np.array([est_alpha_s, est_alpha_l, est_k, est_nu_s, est_nu_l, est_rho])
    except Exception as e:
        print(f"SABR Guess Calc Failed: {e}")
        return default_x0


def fit_polynomial_sabr_surface(all_options_data, prior_guess=None):
    if not all_options_data:
        return np.zeros(6, dtype=float)

    data = list(zip(*all_options_data))
    K_arr = np.array(data[0])
    v_mkt_arr = np.array(data[1])
    S_t_arr = np.array(data[2])
    T_arr = np.array(data[3])

    n_points = len(K_arr)
    weights = np.ones(n_points, dtype=float)

    def objective(coeffs):
        alpha, rho, nu = get_decay_sabr_vectorized(coeffs, T_arr)
        v_model = _hagan_vol_vectorized(K_arr, S_t_arr, T_arr, alpha, 1.0, rho, nu)
        diff = (v_model - v_mkt_arr) * 100.0
        weighted_sq_error = np.sum(weights * (diff**2))
        return weighted_sq_error / np.sum(weights)

    if prior_guess is not None:
        x0 = np.asarray(prior_guess, dtype=float)
    else:
        x0 = np.asarray(
            estimate_sabr_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr), dtype=float
        )

    if x0.shape[0] < 6:
        x0 = np.pad(x0, (0, 6 - x0.shape[0]), mode="constant")
    elif x0.shape[0] > 6:
        x0 = x0[:6]

    x0[4] = 0.1

    bounds = [
        (0.01, 2.5),
        (0.01, 2.5),
        (0.01, 20.0),
        (0.01, 3.5),
        (0.1, 0.1),
        (-0.999, 0.999),
    ]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                tol=1e-5,
                options={"maxiter": 250},
            )
        coeffs_opt = np.asarray(res.x, dtype=float)
        if coeffs_opt.shape[0] != 6:
            coeffs_opt = np.resize(coeffs_opt, 6)
        return coeffs_opt
    except Exception as e:
        print(f"SABR Optimization error: {e}")
        return x0


def get_ssvi_from_poly(coeffs, T):
    if len(coeffs) != 8:
        raise ValueError(
            f"SSVI coefficient vector must have exactly 8 elements (found {len(coeffs)})."
        )

    rho = coeffs[0]
    gamma = coeffs[1]
    lam0, lam1, lam2 = coeffs[2], coeffs[3], coeffs[4]
    c0, c1, c2 = coeffs[5], coeffs[6], coeffs[7]

    T_safe = np.clip(T, 1e-6, None)

    theta_T = c0 + c1 * T_safe + c2 * T_safe**2
    theta_T = np.maximum(theta_T, 1e-6)

    lambda_T = lam0 + lam1 * T_safe + lam2 * T_safe**2
    lambda_T = np.maximum(lambda_T, 1e-6)

    return rho, lambda_T, gamma, theta_T


def ssvi_variance_function(k, T, theta_T, rho, lam, gamma):
    phi_T = lam / (theta_T**gamma)
    rho = np.clip(rho, -0.999, 0.999)

    term1 = phi_T * k + rho
    sqrt_term = np.sqrt(term1**2 + (1 - rho**2))
    w_kT = (theta_T / 2.0) * (1.0 + rho * phi_T * k + sqrt_term)

    return np.maximum(w_kT, 1e-12)


def linear_quadratic_theta(T, c1, c2):
    return c1 * T + c2 * T**2


def estimate_ssvi_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr):
    default_x0 = np.array(
        [-0.6, 0.5, 1.0, 0.0, 0.0, 1e-6, 0.05, 0.0], dtype=float
    )

    try:
        omega_mkt = (v_mkt_arr**2) * T_arr
        k_arr = np.log(K_arr / S_t_arr)

        atm_mask = np.abs(k_arr) < 0.05
        T_atm = T_arr[atm_mask]
        Omega_atm = omega_mkt[atm_mask]
        if len(T_atm) < 3:
            return default_x0.copy()

        try:
            popt, pcov = curve_fit(
                linear_quadratic_theta,
                T_atm,
                Omega_atm,
                p0=[0.05, 0.001],
                bounds=([0, -np.inf], [np.inf, np.inf]),
                maxfev=1000,
            )
            c1_est, c2_est = popt
            c0_est = 1e-6
        except Exception:
            c0_est, c1_est, c2_est = 1e-6, 0.05, 0.0

        t_1 = np.min(T_arr)
        mask_1 = T_arr == t_1
        k_1 = k_arr[mask_1]
        v_1 = v_mkt_arr[mask_1]

        if len(k_1) > 5:
            slope_k = np.polyfit(k_1, v_1, 1)[0]
            est_rho = float(np.clip(slope_k * 5.0, -0.9, 0.5))
        else:
            est_rho = -0.6

        est_gamma = 0.5
        est_lam0 = 1.0
        est_lam1 = 0.0
        est_lam2 = 0.0

        c2_final = np.clip(c2_est, -0.001, 0.001)

        return np.array(
            [
                est_rho,
                est_gamma,
                est_lam0,
                est_lam1,
                est_lam2,
                c0_est,
                c1_est,
                c2_final,
            ],
            dtype=float,
        )
    except Exception as e:
        print(f"SSVI Guess Calc Failed: {e}")
        return default_x0.copy()


def fit_polynomial_ssvi_surface(all_options_data, prior_guess=None):
    if not all_options_data:
        return np.zeros(8, dtype=float)

    data = list(zip(*all_options_data))
    K_arr = np.array(data[0])
    v_mkt_arr = np.array(data[1])
    S_t_arr = np.array(data[2])
    T_arr = np.array(data[3])
    n_points = len(K_arr)

    weights = np.ones(n_points, dtype=float)

    unique_ts = np.unique(T_arr)
    unique_ts.sort()

    if len(unique_ts) >= 1:
        n_layers = len(unique_ts)
        layer_weight_map = {}
        for idx, t in enumerate(unique_ts):
            if n_layers == 1:
                w_layer = 2.0
            else:
                w_layer = 1.0 + 2.0 * (1.0 - idx / (n_layers - 1))
            layer_weight_map[t] = w_layer

        layer_weights = np.array([layer_weight_map[t] for t in T_arr], dtype=float)
        weights *= layer_weights

    T_safe = np.clip(T_arr, 1e-4, None)
    k_arr = np.log(K_arr / S_t_arr)
    w_market = (v_mkt_arr**2) * T_safe

    atm_mask = np.abs(k_arr) < 0.05
    weights[atm_mask] *= 1.5

    def objective(coeffs):
        rho, gamma, lam0, lam1, lam2, c0, c1, c2 = coeffs

        lambda_T = lam0 + lam1 * T_safe + lam2 * T_safe**2
        lambda_T = np.maximum(lambda_T, 1e-6)

        theta_T = c0 + c1 * T_safe + c2 * T_safe**2
        theta_T = np.maximum(theta_T, 1e-4)

        w_model = ssvi_variance_function(k_arr, T_safe, theta_T, rho, lambda_T, gamma)
        diff = (w_model - w_market) * 100.0
        weighted_sq_error = np.sum(weights * (diff**2))
        return weighted_sq_error / np.sum(weights)

    if prior_guess is not None:
        x0 = np.asarray(prior_guess, dtype=float)
    else:
        x0 = estimate_ssvi_initial_guess(K_arr, v_mkt_arr, S_t_arr, T_arr)

    if x0.shape[0] < 8:
        x0 = np.pad(x0, (0, 8 - x0.shape[0]), mode="constant")
    elif x0.shape[0] > 8:
        x0 = x0[:8]

    x0[7] = np.clip(x0[7], -0.001, 0.001)

    bounds = [
        (-0.95, 0.95),   # 0: rho   - 不放到极端 -0.999 / 0.999，避免疯狂 skew
        (0.3, 1.2),      # 1: gamma - 更接近常用范围，防止曲面过陡或过平

        (0.1, 3.0),      # 2: lam0  - 基础笑脸强度
        (-0.3, 0.3),     # 3: lam1  - 随时间线性项收紧
        (-0.05, 0.05),   # 4: lam2  - 二阶项收得更紧，防止远端爆炸

        (1e-6, 0.04),    # 5: c0    - 起始 theta
        (0.0, 0.2),      # 6: c1    - theta 随 T 的斜率不要太大
        (-5e-4, 5e-4),   # 7: c2    - 二阶项更窄，确保 term-structure 更平滑
    ]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                tol=1e-6,
                bounds=bounds,
                options={"maxiter": 400},
            )
        coeffs_opt = np.asarray(res.x, dtype=float)
        if coeffs_opt.shape[0] != 8:
            coeffs_opt = np.resize(coeffs_opt, 8)
        return coeffs_opt
    except Exception as e:
        print(f"SSVI Optimization error: {e}")
        x0_safe = np.asarray(x0, dtype=float)
        if x0_safe.shape[0] != 8:
            x0_safe = np.resize(x0_safe, 8)
        return x0_safe



def compute_iv_loss(iv_mkt, iv_model):
    iv_mkt = np.asarray(iv_mkt, dtype=float)
    iv_model = np.asarray(iv_model, dtype=float)

    mask = (
        np.isfinite(iv_mkt)
        & np.isfinite(iv_model)
        & (iv_mkt > 0)
        & (iv_model > 0)
    )
    if not np.any(mask):
        return np.nan, np.nan

    diff = iv_model[mask] - iv_mkt[mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    return mae, rmse


def sabr_iv_from_coeffs(coeffs, K_arr, F_arr, T_arr):
    K_arr = np.asarray(K_arr, dtype=float)
    F_arr = np.asarray(F_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)

    alpha, rho, nu = get_decay_sabr_vectorized(coeffs, T_arr)
    beta = 1.0
    iv_model = _hagan_vol_vectorized(K_arr, F_arr, T_arr, alpha, beta, rho, nu)
    return iv_model


def ssvi_iv_from_coeffs(coeffs, K_arr, F_arr, T_arr):
    K_arr = np.asarray(K_arr, dtype=float)
    F_arr = np.asarray(F_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)

    k_arr = np.log(K_arr / F_arr)
    rho, lambda_T, gamma, theta_T = get_ssvi_from_poly(coeffs, T_arr)
    w_model = ssvi_variance_function(k_arr, T_arr, theta_T, rho, lambda_T, gamma)

    T_safe = np.clip(T_arr, 1e-6, None)
    iv_model = np.sqrt(w_model / T_safe)
    return iv_model


def _points_to_arrays(points):
    if not points:
        return None, None, None, None
    arr = np.array(points, dtype=float)
    K_arr = arr[:, 0]
    iv_mkt_arr = arr[:, 1]
    S_arr = arr[:, 2]
    T_arr = arr[:, 3]
    return K_arr, iv_mkt_arr, S_arr, T_arr


#  Extended Kalman Filter 结构（非线性 h(x)）

class ExtendedKalmanFilter:
    """
    标准离散 EKF:
        x_k   = f(x_{k-1}) + w_{k-1}
        z_k   = h(x_k)     + v_k

    这里我们设:
        f(x) = x  （参数在时间上做随机游走）
        h(x) = model_IV_from_params(x, K, F, T)

    所以非线性在观测模型里，通过 Jacobian(H) 做一阶线性化。
    """

    def __init__(self, dim_x, q_scale=1e-4, r_scale=0.03, p0_scale=0.1):
        self.dim_x = dim_x
        self.q_scale = q_scale
        self.r_scale = r_scale
        self.p0_scale = p0_scale

        self.Q = (q_scale**2) * np.eye(dim_x)
        self.x = None
        self.P = None

    def initialize(self, x0):
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != self.dim_x:
            x0 = np.resize(x0, self.dim_x)
        self.x = x0.copy()
        self.P = (self.p0_scale**2) * np.eye(self.dim_x)

    def predict(self):
        if self.x is None:
            return
        # f(x) = x, F = I
        self.P = self.P + self.Q

    def update(self, z, h_func, jacobian_func):
        if self.x is None:
            return
        z = np.asarray(z, dtype=float)
        z_pred = h_func(self.x)
        if z_pred.shape != z.shape:
            # 保护一下：维度不一致就不更新
            return

        y = z - z_pred  # innovation
        H = jacobian_func(self.x)  # M x N
        M = z.shape[0]
        R = (self.r_scale**2) * np.eye(M)

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 稍微加一点抖动，防止奇异
            S = S + 1e-8 * np.eye(S.shape[0])
            K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ H) @ self.P


def numerical_jacobian(x, h_func, eps=1e-4):
    """
    对观测函数 h(x) 做数值 Jacobian:
        H[i, j] = d h_i / d x_j  ≈ (h_i(x + e_j * eps) - h_i(x)) / eps
    """
    x = np.asarray(x, dtype=float)
    base = h_func(x)
    base = np.asarray(base, dtype=float)
    M = base.shape[0]
    N = x.shape[0]
    J = np.zeros((M, N), dtype=float)
    for j in range(N):
        x_pert = x.copy()
        x_pert[j] += eps
        pert = h_func(x_pert)
        J[:, j] = (np.asarray(pert, dtype=float) - base) / eps
    return J


#  单日处理：加入 EKF 对 SABR + SSVI 的非线性滤波

def _process_single_day(day_data):
    (
        day_obj,
        day_dfm_map,
        day_minutes_map,
        day_minute_prices_map,
        day_minute_vol_map,
        q_div,
        ticker,
        db_url,
    ) = day_data

    day_minutes = day_minutes_map[day_obj]
    day_df = day_dfm_map[day_obj]

    local_engine = create_engine(db_url, pool_pre_ping=True)

    processed_surfaces = []
    day_str = day_obj.strftime("%Y-%m-%d")
    print(f"  - Processing {day_str}...")
    r_rate = _get_rate(engine=local_engine, d=day_minutes[0]) / 100
    start_loop = int(datetime.now().timestamp())

    day_df["ts_min"] = day_df["ts_utc"].dt.floor("min")

    cols = [
        "ts_min",
        "strike",
        "dte",
        "option_close",
        "option_volume_weighted",
        "option_volume",
        "option_transactions",
        "contract_type",
    ]
    base_day_grouped = day_df[cols].groupby("ts_min")

    # EKF for SABR / SSVI Call / Put
    ekf_sabr_c = ExtendedKalmanFilter(dim_x=6, q_scale=5e-4, r_scale=0.03, p0_scale=0.2)
    ekf_sabr_p = ExtendedKalmanFilter(dim_x=6, q_scale=5e-4, r_scale=0.03, p0_scale=0.2)
    ekf_ssvi_c = ExtendedKalmanFilter(dim_x=8, q_scale=3e-4, r_scale=0.03, p0_scale=0.2)
    ekf_ssvi_p = ExtendedKalmanFilter(dim_x=8, q_scale=3e-4, r_scale=0.03, p0_scale=0.2)

    for ts in day_minutes:
        S_t = day_minute_prices_map[day_obj].get(ts)
        S_vol_t = day_minute_vol_map[day_obj].get(ts)

        if pd.isna(S_t) or S_t <= 0:
            continue

        sub = (
            base_day_grouped.get_group(ts)
            if ts in base_day_grouped.groups
            else pd.DataFrame()
        )
        if sub.empty or ("dte" not in sub.columns):
            continue

        joint_call_points = []
        joint_put_points = []

        minute_market_c = []
        minute_market_p = []

        unique_dtes = sub["dte"].unique()
        for dte in unique_dtes:
            if dte < 1:
                continue
            T_expiry = dte / 365.0
            chain = sub[sub["dte"] == dte]

            calls = chain[chain["contract_type"] == "C"]
            if not calls.empty:
                raw_records = calls[
                    ["strike", "option_volume_weighted", "option_volume"]
                ].to_dict("records")
                for r in raw_records:
                    r["dte"] = int(dte)
                minute_market_c.extend(raw_records)

                for row in calls.itertuples():
                    market_price = row.option_volume_weighted
                    iv = calculate_iv_american_call(
                        market_price=market_price,
                        S=S_t,
                        K=row.strike,
                        T=T_expiry,
                        r=r_rate,
                        q=q_div,
                        time_entry_ts=ts,
                    )
                    joint_call_points.append(
                        (row.strike, iv, S_t, T_expiry, market_price)
                    )

            puts = chain[chain["contract_type"] == "P"]
            if not puts.empty:
                raw_records = puts[
                    ["strike", "option_volume_weighted", "option_volume"]
                ].to_dict("records")
                for r in raw_records:
                    r["dte"] = int(dte)
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
                        time_entry_ts=ts,
                    )
                    joint_put_points.append(
                        (row.strike, iv, S_t, T_expiry, market_price)
                    )

        # 静态 eSABR / SSVI 拟合 
        sabr_coeffs_c = fit_polynomial_sabr_surface(joint_call_points)
        sabr_coeffs_p = fit_polynomial_sabr_surface(joint_put_points)

        ssvi_coeffs_c = fit_polynomial_ssvi_surface(joint_call_points)
        ssvi_coeffs_p = fit_polynomial_ssvi_surface(joint_put_points)

        # 计算静态 loss (原始) 
        (
            sabr_rmse_c,
            sabr_mae_c,
            ssvi_rmse_c,
            ssvi_mae_c,
            sabr_rmse_p,
            sabr_mae_p,
            ssvi_rmse_p,
            ssvi_mae_p,
        ) = (np.nan,) * 8

        (
            ekf_sabr_rmse_c,
            ekf_sabr_mae_c,
            ekf_ssvi_rmse_c,
            ekf_ssvi_mae_c,
            ekf_sabr_rmse_p,
            ekf_sabr_mae_p,
            ekf_ssvi_rmse_p,
            ekf_ssvi_mae_p,
        ) = (np.nan,) * 8

        # -------- CALL 侧 --------
        Kc, iv_mkt_c, Sc, Tc = _points_to_arrays(joint_call_points)
        if Kc is not None:
            try:
                # 静态
                iv_sabr_c = sabr_iv_from_coeffs(sabr_coeffs_c, Kc, Sc, Tc)
                iv_ssvi_c = ssvi_iv_from_coeffs(ssvi_coeffs_c, Kc, Sc, Tc)
                ssvi_mae_c, ssvi_rmse_c = compute_iv_loss(iv_mkt_c, iv_ssvi_c)
                sabr_mae_c, sabr_rmse_c = compute_iv_loss(iv_mkt_c, iv_sabr_c)
            except Exception as e:
                pass

            # EKF 初始化（第一次有数据时，用静态拟合作为 x0）
            if ekf_sabr_c.x is None:
                ekf_sabr_c.initialize(sabr_coeffs_c)
            if ekf_ssvi_c.x is None:
                ekf_ssvi_c.initialize(ssvi_coeffs_c)

            # EKF * eSABR 
            try:
                def h_sabr_c(x):
                    return sabr_iv_from_coeffs(x, Kc, Sc, Tc)

                def J_sabr_c(x):
                    return numerical_jacobian(x, h_sabr_c, eps=1e-4)

                ekf_sabr_c.predict()
                ekf_sabr_c.update(iv_mkt_c, h_sabr_c, J_sabr_c)
                sabr_coeffs_c_ekf = ekf_sabr_c.x.copy()
                iv_sabr_c_ekf = sabr_iv_from_coeffs(sabr_coeffs_c_ekf, Kc, Sc, Tc)
                ekf_sabr_mae_c, ekf_sabr_rmse_c = compute_iv_loss(
                    iv_mkt_c, iv_sabr_c_ekf
                )
            except Exception as e:
                pass

            # EKF * SSVI 
            try:
                def h_ssvi_c(x):
                    return ssvi_iv_from_coeffs(x, Kc, Sc, Tc)

                def J_ssvi_c(x):
                    return numerical_jacobian(x, h_ssvi_c, eps=1e-4)

                ekf_ssvi_c.predict()
                ekf_ssvi_c.update(iv_mkt_c, h_ssvi_c, J_ssvi_c)
                ssvi_coeffs_c_ekf = ekf_ssvi_c.x.copy()
                iv_ssvi_c_ekf = ssvi_iv_from_coeffs(ssvi_coeffs_c_ekf, Kc, Sc, Tc)
                ekf_ssvi_mae_c, ekf_ssvi_rmse_c = compute_iv_loss(
                    iv_mkt_c, iv_ssvi_c_ekf
                )
            except Exception as e:
                pass

        # -------- PUT 侧 --------
        Kp, iv_mkt_p, Sp, Tp = _points_to_arrays(joint_put_points)
        if Kp is not None:
            try:
                iv_sabr_p = sabr_iv_from_coeffs(sabr_coeffs_p, Kp, Sp, Tp)
                iv_ssvi_p = ssvi_iv_from_coeffs(ssvi_coeffs_p, Kp, Sp, Tp)
                ssvi_mae_p, ssvi_rmse_p = compute_iv_loss(iv_mkt_p, iv_ssvi_p)
                sabr_mae_p, sabr_rmse_p = compute_iv_loss(iv_mkt_p, iv_sabr_p)
            except Exception as e:
                pass

            if ekf_sabr_p.x is None:
                ekf_sabr_p.initialize(sabr_coeffs_p)
            if ekf_ssvi_p.x is None:
                ekf_ssvi_p.initialize(ssvi_coeffs_p)

            # === EKF * eSABR (PUT) ===
            try:
                def h_sabr_p(x):
                    return sabr_iv_from_coeffs(x, Kp, Sp, Tp)

                def J_sabr_p(x):
                    return numerical_jacobian(x, h_sabr_p, eps=1e-4)

                ekf_sabr_p.predict()
                ekf_sabr_p.update(iv_mkt_p, h_sabr_p, J_sabr_p)
                sabr_coeffs_p_ekf = ekf_sabr_p.x.copy()
                iv_sabr_p_ekf = sabr_iv_from_coeffs(sabr_coeffs_p_ekf, Kp, Sp, Tp)
                ekf_sabr_mae_p, ekf_sabr_rmse_p = compute_iv_loss(
                    iv_mkt_p, iv_sabr_p_ekf
                )
            except Exception as e:
                pass

            # === EKF * SSVI (PUT) ===
            try:
                def h_ssvi_p(x):
                    return ssvi_iv_from_coeffs(x, Kp, Sp, Tp)

                def J_ssvi_p(x):
                    return numerical_jacobian(x, h_ssvi_p, eps=1e-4)

                ekf_ssvi_p.predict()
                ekf_ssvi_p.update(iv_mkt_p, h_ssvi_p, J_ssvi_p)
                ssvi_coeffs_p_ekf = ekf_ssvi_p.x.copy()
                iv_ssvi_p_ekf = ssvi_iv_from_coeffs(ssvi_coeffs_p_ekf, Kp, Sp, Tp)
                ekf_ssvi_mae_p, ekf_ssvi_rmse_p = compute_iv_loss(
                    iv_mkt_p, iv_ssvi_p_ekf
                )
            except Exception as e:
                pass

        processed_surfaces.append(
            {
                "time_entry_ts": ts,
                "ticker": ticker,
                "price_ffill_S": S_t,
                "underlying_volume": S_vol_t,
                "risk_free_rate": r_rate,
                "market_data_C": minute_market_c,
                "market_data_P": minute_market_p,
                "iv_point_C": joint_call_points,
                "iv_point_P": joint_put_points,
                "sabr_coeffs_C": sabr_coeffs_c.tolist(),
                "sabr_coeffs_P": sabr_coeffs_p.tolist(),
                "ssvi_coeffs_C": ssvi_coeffs_c.tolist(),
                "ssvi_coeffs_P": ssvi_coeffs_p.tolist(),
                # 原始 loss
                "sabr_rmse_C": sabr_rmse_c,
                "sabr_mae_C": sabr_mae_c,
                "ssvi_rmse_C": ssvi_rmse_c,
                "ssvi_mae_C": ssvi_mae_c,
                "sabr_rmse_P": sabr_rmse_p,
                "sabr_mae_P": sabr_mae_p,
                "ssvi_rmse_P": ssvi_rmse_p,
                "ssvi_mae_P": ssvi_mae_p,
                # EKF 后的 loss
                "ekf_sabr_rmse_C": ekf_sabr_rmse_c,
                "ekf_sabr_mae_C": ekf_sabr_mae_c,
                "ekf_ssvi_rmse_C": ekf_ssvi_rmse_c,
                "ekf_ssvi_mae_C": ekf_ssvi_mae_c,
                "ekf_sabr_rmse_P": ekf_sabr_rmse_p,
                "ekf_sabr_mae_P": ekf_sabr_mae_p,
                "ekf_ssvi_rmse_P": ekf_ssvi_rmse_p,
                "ekf_ssvi_mae_P": ekf_ssvi_mae_p,
            }
        )

    end_loop = int(datetime.now().timestamp())
    print(f"compute loop for {day_str} took: {end_loop - start_loop} seconds")
    return processed_surfaces



def build_option_items(
    engine: Engine,
    ticker: str,
    start_day: str,
    end_day: str,
    dte_max: int,
    k_pct: float,
    q_div=0.0,
    num_workers=10,
):
    processed_surfaces = []
    days_to_process = _business_days(start_day, end_day)
    all_filtered_dfs = []

    day_sod_map = {}
    day_minutes_map = {}
    day_dfm_map = {}
    day_minute_prices_map = {}
    day_minute_vol_map = {}

    print(f"--- Loading data for {ticker} from {start_day} to {end_day} ---")

    ticker_id = get_company_id(engine, ticker)
    for day_obj in days_to_process:
        day_str = day_obj.strftime("%Y-%m-%d")

        S = _get_sod(engine, ticker_id, day_obj)
        if not S or S <= 0:
            print(f"[WARN] No SOD price for {day_str}, skipping day.")
            continue

        dfm = load_minutes(engine, ticker, day_str, dte_max)
        if dfm.empty:
            print(f"[WARN] No options data for {day_str}, skipping day.")
            continue

        vol_profil_days = 20
        underlying_price_series, underlying_volume_series = load_underlying(
            engine, day_str, ticker_id, vol_profil_days
        )

        t0 = pd.Timestamp(f"{day_str} 13:30", tz="UTC")
        t1 = pd.Timestamp(f"{day_str} 20:00", tz="UTC")

        dfm = dfm[(dfm["ts_utc"] >= t0) & (dfm["ts_utc"] <= t1)]
        dfm["strike"] = pd.to_numeric(dfm["strike"], errors="coerce")
        dfm = dfm.dropna(subset=["strike"])

        dfm_filtered = dfm[
            (np.abs(np.log(dfm["strike"].astype(float) / float(S))) <= k_pct)
        ]

        dfm_filtered = dfm_filtered[dfm_filtered["dte"].between(1, dte_max)]
        dfm_filtered = dfm_filtered[dfm_filtered["option_volume_weighted"] > 0.05]

        if dfm_filtered.empty:
            print(
                f"[WARN] No options data remaining after OTM filtering for {day_str}, skipping."
            )
            continue

        print(
            f"[INFO] Loaded {len(dfm_filtered)} OTM rows for {day_str} with SOD={S:.2f}"
        )

        all_filtered_dfs.append(dfm_filtered)

        day_sod_map[day_obj] = S
        day_minutes_map[day_obj] = pd.date_range(
            t0, t1, freq="1min", inclusive="both"
        )
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
            db_url,
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

    with open(params_file_path, "r") as f:
        params = json.load(f)

    ticker = params["ticker"][0]
    dividend_rates = params["dividend_rates"][ticker]
    default_start_day = params["train_start"]
    default_end_day = params["train_end"]
    default_max_dte = params.get("max_dte", 30)
    k_pct = params.get("k_pct", 0.1)

    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

    df = build_option_items(
        engine,
        ticker,
        default_start_day,
        default_end_day,
        default_max_dte,
        k_pct=k_pct,
        q_div=dividend_rates,
    )

    k_pct_str = f"{k_pct:.2f}"
    q_div_str = f"{dividend_rates:.4f}"

    filename = (
        f"{ticker}"
        f"_kpct{k_pct_str}"
        f"_dte{default_max_dte}"
        f"_from_{default_start_day}"
        f"_to_{default_end_day}"
        f"_qdiv{q_div_str}"
        f"_EKF.parquet"
    )
    filename = filename.replace(":", "-").replace(" ", "_")
    df.to_parquet(filename)

    
    loss_cols = [
        "time_entry_ts",
        # 原始
        "sabr_rmse_C",
        "sabr_mae_C",
        "ssvi_rmse_C",
        "ssvi_mae_C",
        "sabr_rmse_P",
        "sabr_mae_P",
        "ssvi_rmse_P",
        "ssvi_mae_P",
        # EKF 后
        "ekf_sabr_rmse_C",
        "ekf_sabr_mae_C",
        "ekf_ssvi_rmse_C",
        "ekf_ssvi_mae_C",
        "ekf_sabr_rmse_P",
        "ekf_sabr_mae_P",
        "ekf_ssvi_rmse_P",
        "ekf_ssvi_mae_P",
    ]

    loss_cols_existing = [c for c in loss_cols if c in df.columns]
    loss_df = df[loss_cols_existing].copy()

    loss_filename = (
        f"{ticker}"
        f"_loss_kpct{k_pct_str}"
        f"_dte{default_max_dte}"
        f"_from_{default_start_day}"
        f"_to_{default_end_day}"
        f"_qdiv{q_div_str}"
        f"_EKF.txt"  
    )
    loss_filename = loss_filename.replace(":", "-").replace(" ", "_")

    loss_df.to_csv(loss_filename, index=False)
    print(f"[INFO] EKF Loss metrics saved to {loss_filename}")


if __name__ == "__main__":
    main()
