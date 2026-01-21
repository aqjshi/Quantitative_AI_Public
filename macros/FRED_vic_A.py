# =========================================================
# FRED Macro Causal DAG Pipeline (Python 3.12 SAFE)
# =========================================================

# ===== SSL FIX =====
import ssl
import certifi
def _ssl_context():
    return ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = _ssl_context
# ===================

from fredapi import Fred
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import grangercausalitytests
import itertools
import networkx as nx

# =========================================================
# 1. FRED API
# =========================================================
fred = Fred(api_key="cacba652bcd29503058a140e2dbc26f7")

START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"

# =========================================================
# 2. CORE SERIES (目标 50 条)
# =========================================================
FRED_SERIES = {
    # Monetary
    "FEDFUNDS": "fed_funds",
    "GS10": "ust_10y",
    "GS2": "ust_2y",
    "TB3MS": "tbill_3m",
    "T10Y3M": "yc_10y_3m",
    "M2SL": "m2",
    "RRPONTSYD": "rrp",
    "IORB": "iorb",

    # Financial stress
    "STLFSI2": "stl_fsi",
    "NFCI": "nfci",
    "BAA10Y": "baa_spread",
    "AAA10Y": "aaa_spread",
    "TEDRATE": "ted_spread",
    "BAMLH0A0HYM2": "hy_spread",

    # Labor
    "UNRATE": "u3",
    "U6RATE": "u6",
    "CIVPART": "labor_participation",
    "LNS11300060": "prime_epop",
    "PAYEMS": "nonfarm_payroll",
    "AHETPI": "hourly_earnings",

    # Labor flows
    "ICSA": "initial_claims",
    "CCSA": "continuing_claims",
    "JTSJOL": "job_openings",
    "JTSQUR": "quits_rate",
    "JTSHIR": "hiring_rate",
    "JTSLDL": "layoffs",

    # Business cycle
    "WEI": "weekly_econ_index",
    "INDPRO": "industrial_prod",
    "GDPC1": "real_gdp",
    "NAPM": "ism_pmi",          # ✅ 替换 NAPM
    "BUSLOANS": "business_loans",
    "ISRATIO": "inventory_sales",

    # Consumer
    "DSPIC96": "real_disp_income",
    "PCE": "pce",
    "PSAVERT": "savings_rate",
    "REVOLSL": "credit_card_bal",
    "DRCCLACBS": "cc_delinquency",
    "TOTALSL": "household_debt",
    "UMCSENT": "consumer_sentiment",   # ✅ 新增

    # Housing
    "CSUSHPINSA": "case_shiller",
    "HOUST": "housing_starts",
    "MORTGAGE30US": "mortgage_30y",
    "MORTGAGE15US": "mortgage_15y",
    "HHMSDODNS": "mortgage_delinquency",

    # Markets
    "SP500": "sp500",
    "NASDAQCOM": "nasdaq",
    "DJIA": "dow_jones",           # ✅ 替换 Wilshire
    "VIXCLS": "vix",
    "CPILFESL": "core_cpi",
    "DTWEXBGS": "usd_index",
    "T10YIE": "breakeven_infl"
    
}


# =========================================================
# 3. LOAD DATA
# =========================================================
df_raw = pd.DataFrame()

for sid, name in FRED_SERIES.items():
    try:
        s = fred.get_series(sid, START_DATE, END_DATE)
        s = s.resample("ME").last()
        df_raw[name] = s
    except Exception as e:
        print(f"[SKIP] {sid}: {e}")

print(f"Loaded series: {df_raw.shape[1]}")

# ===== Add CHURN (constructed from JOLTS rates) =====
# churn = hires + quits + layoffs (all are JOLTS rates you already pulled)
if {"hiring_rate", "quits_rate", "layoffs"}.issubset(df_raw.columns):
    df_raw["churn"] = df_raw["hiring_rate"] + df_raw["quits_rate"] + df_raw["layoffs"]
    print("[OK] Added derived series: churn = hiring_rate + quits_rate + layoffs")
else:
    print("[WARN] Cannot build churn: missing one of {hiring_rate, quits_rate, layoffs}")

# ================= SAVE RAW DATA =================
df_raw.to_csv("vicki/macro_raw.csv", index=True)
print("Saved: vicki/macro_raw.csv")


# =========================================================
# 4. SMOOTHING
# =========================================================
df_smooth = df_raw.rolling(window=3, min_periods=2).mean()

df_hp = pd.DataFrame(index=df_raw.index)
for col in df_raw.columns:
    try:
        cycle, trend = hpfilter(df_raw[col].dropna(), lamb=129600)
        df_hp[col] = cycle
    except:
        pass

# =========================================================
# 5. TRANSFORM (rate -> diff, level -> log-diff)
# =========================================================
RATE_LIKE = {
    "fed_funds","ust_10y","ust_2y","tbill_3m","yc_10y_3m","iorb",
    "stl_fsi","nfci","baa_spread","aaa_spread","ted_spread","hy_spread",
    "u3","u6","labor_participation","prime_epop","hourly_earnings",
    "quits_rate","hiring_rate","layoffs",
    "churn", 
    "savings_rate","cc_delinquency","mortgage_30y","mortgage_15y","mortgage_delinquency",
    "vix","core_cpi","usd_index","breakeven_infl"
    
}

def transform_one(name: str, s: pd.Series) -> pd.Series:
    s = s.astype(float)
    # 先轻微平滑，降低噪声
    s = s.rolling(window=3, min_periods=2).mean()

    if name in RATE_LIKE:
        return s.diff()
    else:
        # level/quantity/index：能取log就log-diff，否则diff
        ss = s.dropna()
        if len(ss) == 0:
            return s * np.nan
        if (ss <= 0).any():
            return s.diff()
        return np.log(s).diff()

df_transformed = pd.DataFrame(index=df_raw.index)
for col in df_raw.columns:
    df_transformed[col] = transform_one(col, df_raw[col])

# 注意：这里不要全局 dropna！
# df_transformed = df_transformed.dropna()

df_transformed.to_csv("vicki/macro_transformed.csv", index=True)
print("Saved: vicki/macro_transformed.csv")
print("Transformed rows (non-empty any col):", df_transformed.dropna(how="all").shape[0])

# =========================================================
# 6. GRANGER CAUSALITY (pairwise dropna + diagnostics + sparsify)
# =========================================================
from statsmodels.tsa.stattools import grangercausalitytests

MAX_LAG = 6
ALPHA = 0.01
MIN_T = 80          # 月频数据，至少 ~6-7 年共同样本更稳
TOP_K_PER_CAUSE = 3 # ✅ 稀疏化：每个 cause 最多保留 3 条边（强烈建议）

variables = list(df_transformed.columns)
edge_rows = []

tested = 0
skipped_short = 0
errors = 0

for cause, effect in itertools.permutations(variables, 2):
    pair = df_transformed[[effect, cause]].dropna()

    # 样本数需要大于 lag 的一个倍数，否则 F-test 不稳
    if len(pair) < max(MIN_T, MAX_LAG * 10):
        skipped_short += 1
        continue

    tested += 1

    try:
        res = grangercausalitytests(pair, maxlag=MAX_LAG, verbose=False)
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, MAX_LAG + 1)]
        best_lag = int(np.argmin(pvals) + 1)
        best_p = float(np.min(pvals))

        if best_p < ALPHA:
            # ✅ weight：保留区分度，避免饱和
            p_clip = float(np.clip(best_p, 1e-300, 1.0))
            weight = float(-np.log(p_clip))

            edge_rows.append({
                "cause": cause,
                "effect": effect,
                "best_lag": best_lag,
                "p_value": best_p,
                "weight": weight,
                "n_obs": int(len(pair))
            })

    except Exception:
        errors += 1
        continue

print(f"Granger tested pairs: {tested}, skipped_short: {skipped_short}, errors: {errors}")
print(f"Detected causal edges (raw): {len(edge_rows)}")

# =========================
# ✅ 稀疏化：每个 cause 只留 TOP_K 条最强边
# =========================
# ✅ 统一下游使用的边：用 sparsified edges_df
edges_df = pd.DataFrame(edge_rows)
if not edges_df.empty:
    edges_df = edges_df.sort_values(["cause", "weight", "p_value"], ascending=[True, False, True])
    edges_df = edges_df.groupby("cause").head(TOP_K_PER_CAUSE).reset_index(drop=True)

edges_df.to_csv("vicki/macro_granger_edges.csv", index=False)

# ✅ 下游统一的 records / edges
edges_records = edges_df.to_dict("records")
edges = [(r["cause"], r["effect"]) for r in edges_records]


# =========================================================
# 7. DAG NODES + ADJ + WEIGHT MATRICES (variable-level)
# =========================================================
nodes_df = pd.DataFrame({
    "node_id": list(range(len(variables))),
    "node_name": variables
})
nodes_df.to_csv("vicki/macro_dag_nodes.csv", index=False)
print("Saved: vicki/macro_dag_nodes.csv")

idx = {v:i for i,v in enumerate(variables)}
adj = np.zeros((len(variables), len(variables)), dtype=int)
wmat = np.zeros((len(variables), len(variables)), dtype=float)

for r in edges_records:
    i = idx[r["cause"]]
    j = idx[r["effect"]]
    adj[i, j] = 1
    wmat[i, j] = r["weight"]


adj_df = pd.DataFrame(adj, index=variables, columns=variables)
w_df   = pd.DataFrame(wmat, index=variables, columns=variables)

adj_df.to_csv("vicki/macro_dag_adjacency_matrix.csv", index=True)
w_df.to_csv("vicki/macro_dag_weight_matrix.csv", index=True)
print("Saved: vicki/macro_dag_adjacency_matrix.csv")
print("Saved: vicki/macro_dag_weight_matrix.csv")

# =========================================================
# 8. SCC + SCC DAG (SCC-level adjacency/weight + key)
# =========================================================
G = nx.DiGraph()
G.add_nodes_from(variables)
G.add_edges_from(edges)

sccs = list(nx.strongly_connected_components(G))
print(f"Number of SCCs: {len(sccs)}")

# variable -> scc_id
var_to_scc = {}
for i, comp in enumerate(sccs):
    for v in comp:
        var_to_scc[v] = i

# SCC key (导师要“每个 SCC 里有什么”)
key_rows = []
for i, comp in enumerate(sccs):
    key_rows.append({
        "scc_id": i,
        "size": len(comp),
        "members": "|".join(sorted(list(comp)))
    })
scc_key_df = pd.DataFrame(key_rows).sort_values(["size","scc_id"], ascending=[False, True])
scc_key_df.to_csv("vicki/macro_SCC_key.csv", index=False)
print("Saved: vicki/macro_SCC_key.csv")

# Build SCC DAG with weights aggregated
SCC_G = nx.DiGraph()
SCC_G.add_nodes_from(range(len(sccs)))

scc_w = {}
for r in edges_records:
    sx = var_to_scc[r["cause"]]
    sy = var_to_scc[r["effect"]]
    if sx == sy:
        continue
    scc_w[(sx, sy)] = max(scc_w.get((sx, sy), 0.0), r["weight"])


for (sx, sy), wt in scc_w.items():
    SCC_G.add_edge(sx, sy, weight=wt)

# SCC edges file
scc_edges_df = pd.DataFrame(
    [{"scc_cause": sx, "scc_effect": sy, "weight": wt} for (sx, sy), wt in scc_w.items()],
    columns=["scc_cause", "scc_effect", "weight"]  # ✅ 强制表头
)
scc_edges_df.to_csv("vicki/macro_SCC_edges.csv", index=False)

print("Saved: vicki/macro_SCC_edges.csv")

# SCC adjacency + weight matrices
nS = len(sccs)
scc_adj = np.zeros((nS, nS), dtype=int)
scc_wmat = np.zeros((nS, nS), dtype=float)

for (sx, sy), wt in scc_w.items():
    scc_adj[sx, sy] = 1
    scc_wmat[sx, sy] = wt

pd.DataFrame(scc_adj).to_csv("vicki/macro_SCC_adjacency_matrix.csv", index=False)
pd.DataFrame(scc_wmat).to_csv("vicki/macro_SCC_weight_matrix.csv", index=False)
print("Saved: vicki/macro_SCC_adjacency_matrix.csv")
print("Saved: vicki/macro_SCC_weight_matrix.csv")
