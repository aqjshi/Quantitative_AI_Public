import sys
import os
import json
import requests
from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd 
import numpy as np
from datetime import timezone

# Project Imports
# Add current directory to path to import from Quantitative_AI_Prod/core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.db import DATABASE_URL, POLY_KEY
from core.models import Base, Index, Company, Quote, IndexQuote
import core.sieve as sieve
import random
import time
from tqdm import tqdm

from core.db import DATABASE_URL

def load_market_data(engine, etf_list, ticker_list, t0=None, t1=None):
    """
    Loads Index Proxies and Equity Tickers into a dictionary of DataFrames.
    Optionally filters by [t0, t1] in unix ms.
    """
    vault = {}

    # --- 1. Load ETFs (Indices) ---
    print(f"[*] Ingesting {len(etf_list)} Indices...")
    for etf in tqdm(etf_list, desc="Processing ETFs"):
        if t0 is not None and t1 is not None:
            query = text("""
                SELECT * FROM index_quotes
                WHERE index_symbol = :symbol AND t >= :t0 AND t <= :t1
                ORDER BY t ASC
            """)
            params = {"symbol": etf, "t0": int(t0), "t1": int(t1)}
        else:
            query = text("SELECT * FROM index_quotes WHERE index_symbol = :symbol ORDER BY t ASC")
            params = {"symbol": etf}

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                vault[etf] = df

    # --- 2. Load Whitelisted Tickers (Equities) ---
    print(f"[*] Ingesting {len(ticker_list)} Equities...")
    for ticker in tqdm(ticker_list, desc="Processing Tickers"):
        if t0 is not None and t1 is not None:
            query = text("""
                SELECT q.* FROM quotes q
                JOIN companies c ON q.company_cik = c.cik
                WHERE c.ticker = :ticker AND q.t >= :t0 AND q.t <= :t1
                ORDER BY q.t ASC
            """)
            params = {"ticker": ticker, "t0": int(t0), "t1": int(t1)}
        else:
            query = text("""
                SELECT q.* FROM quotes q
                JOIN companies c ON q.company_cik = c.cik
                WHERE c.ticker = :ticker ORDER BY q.t ASC
            """)
            params = {"ticker": ticker}

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                vault[ticker] = df

    return vault

def sync_to_etf(vault, etf_symbol, ticker_list, col_to_sync='c'):
    """
    Combines the ETF and ticker DataFrames into a single Wide-Format DataFrame.
    
    :param vault: The dictionary containing DataFrames from load_market_data
    :param etf_symbol: The symbol of the ETF to use as the base
    :param ticker_list: List of tickers to merge
    :param col_to_sync: The column to keep (default 'c' for close price)
    :return: A single synchronized DataFrame
    """
    if etf_symbol not in vault:
        print(f"Error: ETF {etf_symbol} not found in vault.")
        return pd.DataFrame()

    # 1. Start with the ETF dataframe as the base
    master_df = vault[etf_symbol][['t', col_to_sync]].copy()
    master_df.columns = ['t', etf_symbol]
    master_df.set_index('t', inplace=True)

    # 2. Iteratively merge each ticker
    for ticker in ticker_list:
        if ticker in vault:
            ticker_df = vault[ticker][['t', col_to_sync]].copy()
            ticker_df.columns = ['t', ticker]
            ticker_df.set_index('t', inplace=True)

            # We use an outer join to keep all timestamps, 
            # or 'left' if you only want timestamps that exist in the ETF
            master_df = master_df.join(ticker_df, how='left')

    # 3. Clean up the data
    # Sort by time (essential after an outer join)
    master_df.sort_index(inplace=True)
    
    # Forward fill: if a stock didn't trade at time T, 
    # use its last known price from T-1
    master_df.ffill(inplace=True)
    
    # Optional: drop rows where any data is still NaN (usually at the very beginning)
    # master_df.dropna(inplace=True)

    return master_df.reset_index()



def format_ts(ts_ms):
    """Converts unix milliseconds to a readable string."""
    return pd.to_datetime(ts_ms, unit='ms').strftime('%Y-%m-%d %H:%M')

def iso_to_ms(s: str, end_of_day: bool = False) -> int:
    # Treat as UTC midnight / end-of-day
    if end_of_day:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=999000, tzinfo=timezone.utc
        )
    else:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def safe_log_returns(wide_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute log returns. Keep NaNs for tickers (do NOT fill with 0 here).
    Only require ETF return exists.
    """
    df = wide_df[["t"] + cols].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    logp = np.log(df[cols].replace(0, np.nan))
    rets = logp.diff()
    rets["t"] = df["t"].values

    # require ETF return exists
    etf_col = cols[0]
    rets = rets.dropna(subset=[etf_col])

    return rets


def solve_window_weights(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """
    Rolling window solver:
    - window内标准化（关键）
    - Lasso(positive=True) 做稀疏选择
    - 映射回原始尺度
    - 非负 + sum-to-1 归一化
    """

    # --- 0. 安全转换 ---
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_obs, n_feat = X.shape
    if n_obs < 5 or n_feat < 3:
        return np.zeros(n_feat)

    # --- 1. window 内标准化（这是你之前缺失的关键步骤） ---
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)

    # 丢掉 window 内几乎不动的特征（否则会数值爆炸）
    valid = X_std > 1e-12
    if valid.sum() < 3:
        return np.zeros(n_feat)

    Xs = (X[:, valid] - X_mean[valid]) / X_std[valid]

    y_mean = y.mean()
    y_std  = y.std()
    if y_std < 1e-12:
        return np.zeros(n_feat)

    ys = (y - y_mean) / y_std

    # --- 2. 在标准化空间跑 Lasso（positive=True） ---
    try:
        from sklearn.linear_model import Lasso
        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            positive=True, # swap to False if we want negative weights.
            max_iter=5000
        )
        model.fit(Xs, ys)
        w_std = model.coef_
    except Exception:
        return np.zeros(n_feat)

    # --- 3. 映射回原始尺度 ---
    w = np.zeros(n_feat)
    w[valid] = w_std / X_std[valid]

    # --- 4. 非负 + 归一化 ---
    w[w < 0] = 0.0
    s = w.sum()
    if s > 0:
        w /= s

    return w



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py params.json")
        return

    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    engine = create_engine(DATABASE_URL)
    
    # 1. Prepare lists from params
    etf_list = list(params.get("PROXY_MAP", {}).values())
    ticker_list = params.get("tickers", [])

    # 2. Ingest everything into the Vault
    memory_vault = load_market_data(engine, etf_list, ticker_list)

    print("\n" + "="*80)
    print(f"{'TICKER/ETF':<12} | {'START DATE':<18} | {'END DATE':<18} | {'ROW COUNT':<10}")
    print("-"*80)

    # 3. Audit ETFs First
    for etf in etf_list:
        if etf in memory_vault:
            df = memory_vault[etf]
            start = format_ts(df['t'].iloc[0])
            end = format_ts(df['t'].iloc[-1])
            count = len(df)
            print(f"{etf:<12} | {start:<18} | {end:<18} | {count:<10,}")
    
    print("-"*80)

    # 4. Audit Whitelisted Tickers
    for ticker in ticker_list:
        if ticker in memory_vault:
            df = memory_vault[ticker]
            start = format_ts(df['t'].iloc[0])
            end = format_ts(df['t'].iloc[-1])
            count = len(df)
            print(f"{ticker:<12} | {start:<18} | {end:<18} | {count:<10,}")

    print("="*80)
    print(f"[*] Total distinct series in RAM: {len(memory_vault)}")
    
        # Optional: respect start/end in params to reduce RAM
    start = params.get("start")
    end = params.get("end")
    t0 = iso_to_ms(start, end_of_day=False) if start else None
    t1 = iso_to_ms(end, end_of_day=True) if end else None

    # Re-ingest with time filter if start/end provided (recommended)
    if t0 is not None and t1 is not None:
        print(f"[*] Filtering ingest to range {start} -> {end}")
        memory_vault = load_market_data(engine, etf_list, ticker_list, t0=t0, t1=t1)

    print("="*80)
    print(f"[*] Total distinct series in RAM: {len(memory_vault)}")

    window = int(params.get("window", 60))  # allow override; default 60
    alpha = float(params.get("alpha", 1e-4))  # allow override
    min_coverage = float(params.get("min_coverage", 0.90))

    for etf in etf_list:
        target_etf = etf  # FIX: use the loop variable
        print(f"\n[*] Syncing all tickers to {target_etf}...")

        synced_df = sync_to_etf(memory_vault, target_etf, ticker_list, col_to_sync='c')
        if synced_df.empty:
            print(f"[!] No synced data for {target_etf}. Skipping.")
            continue

        # Keep only columns that exist
        cols = [target_etf] + [tk for tk in ticker_list if tk in synced_df.columns]

        # Coverage filter (by non-NaN prices post-ffill)
        usable = []
        for tk in cols[1:]:
            cov = 1.0 - synced_df[tk].isna().mean()
            if cov >= min_coverage:
                usable.append(tk)
        cols = [target_etf] + usable

        if len(cols) <= 1:
            print(f"[!] No usable tickers for {target_etf} after coverage filter.")
            continue

        # Convert to returns
        rets = safe_log_returns(synced_df, cols)
        if len(rets) < window:
            print(f"[!] Not enough rows for rolling window on {target_etf}: rows={len(rets)}")
            continue

        times = rets["t"].values.astype("int64")
        feature_tickers = cols[1:]
        y_all = rets[target_etf].to_numpy(dtype=float)

        W = np.zeros((len(rets), len(feature_tickers)), dtype=float)

        # thresholds
        min_valid_frac = 0.90     # window内至少90%非NaN
        min_std = 1e-8            # window内变化太小的列丢弃（分钟logret量级很小，阈值别设太大）
        min_features = 3          # 少于3个特征就不拟合

        print(f"[*] Rolling regression for {target_etf}: rows={len(rets):,}, features={len(feature_tickers)}, window={window}")

        for i in tqdm(range(window - 1, len(rets)), desc=f"Rolling {target_etf}"):
            sl = slice(i - window + 1, i + 1)

            # window y
            yw = y_all[sl]
            if np.isnan(yw).any():
                # ETF return 不应有NaN，但保险
                continue

            # window X as DataFrame to filter columns by NaN coverage and std
            Xw_df = rets.loc[rets.index[sl], feature_tickers]

            # keep cols with enough valid data
            valid_frac = 1.0 - Xw_df.isna().mean()
            keep = valid_frac[valid_frac >= min_valid_frac].index.tolist()
            if len(keep) < min_features:
                continue

            Xw_df = Xw_df[keep]

            # drop near-constant cols (std too small)
            stds = Xw_df.std(axis=0, skipna=True)
            keep2 = stds[stds >= min_std].index.tolist()
            if len(keep2) < min_features:
                continue

            Xw_df = Xw_df[keep2]

            # fill remaining NaNs in kept cols with 0 *within window only*
            # (now NaNs mean missing trades; treating as 0 within window is less harmful after filtering)
            Xw = Xw_df.fillna(0.0).to_numpy(dtype=float)

            w_sub = solve_window_weights(Xw, yw, alpha=alpha)

            # write back into full W matrix
            # map sub-weights back to global feature positions
            for j, tk in enumerate(keep2):
                W[i, feature_tickers.index(tk)] = w_sub[j]

        # Build output DF
        w_cols = [f"{tk}_w" for tk in cols[1:]]
        out = pd.DataFrame(W, columns=w_cols)
        out.insert(0, "t", times)

        # Probability P: rolling mean of indicator(w>0)
        ind = (out[w_cols] > 0.0).astype(float)
        p = ind.rolling(window=window, min_periods=window).mean()
        p.columns = [c.replace("_w", "_p") for c in w_cols]

        out = pd.concat([out, p], axis=1)

        # Drop early rows without full window definition
        out = out.iloc[window - 1:].reset_index(drop=True)

        # Add readable time
        out.insert(1, "timestamp_utc", pd.to_datetime(out["t"], unit="ms", utc=True))

        # Write CSV named as proxy (exactly as Anthony requested)
        out_path = f"{target_etf}.csv"
        out.to_csv(out_path, index=False)
        print(f"[*] Wrote {out_path} (rows={len(out):,}, cols={out.shape[1]})")

        # Quick sanity: top weights at last row
        last = out.iloc[-1][w_cols].sort_values(ascending=False).head(10)
        print("[*] Top 10 weights at last timestamp:")
        for k, v in last.items():
            if v > 0:
                print(f"    {k.replace('_w','')}: {v:.4f}")

        
        # Save to CSV for inspection if needed
        synced_df.to_csv("synced_market_data.csv", index=False)

if __name__ == "__main__":
    main()