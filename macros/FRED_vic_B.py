import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================
# INPUTS (from Deliverable A)
# ==========================
SCC_KEY_FILE   = "macro_SCC_key.csv"          # columns: scc_id,size,members (members separated by |)
SCC_EDGES_FILE = "macro_SCC_edges.csv"        # columns: scc_cause,scc_effect,weight

# ==========================
# CONFIG: liquidity proxies
# ==========================
LIQUIDITY_PROXIES = [
    # stress/conditions
    "stl_fsi", "nfci",
    # spreads
    "ted_spread", "baa_spread", "hy_spread",
    # policy/liquidity facilities
    "rrp", "iorb",
    # credit availability / household stress
    "cc_delinquency", "business_loans",
    # (optional) include churn if你想把劳动力流动也作为融资/现金流压力的leading proxy
    # "churn",
]

OUT_LIQ_NODES = "deliverableB_liquidity_nodes.csv"
OUT_RANK      = "deliverableB_upstream_scc_rank.csv"
OUT_PATHS     = "deliverableB_root_paths.csv"
OUT_PNG       = "deliverableB_focus_scc_dag.png"

# ==========================
# 1) Load SCC key & edges
# ==========================
scc_key = pd.read_csv(SCC_KEY_FILE)
# scc_key: scc_id, size, members ("a|b|c")
scc_key["member_list"] = scc_key["members"].fillna("").apply(lambda x: x.split("|") if x else [])

edges = pd.read_csv(SCC_EDGES_FILE)

# Robust rename (in case your header differs)
rename_map = {}
for c in edges.columns:
    lc = c.lower()
    if lc in ["scc_src", "src", "cause", "scc_cause"]: rename_map[c] = "scc_cause"
    if lc in ["scc_dst", "dst", "effect", "scc_effect"]: rename_map[c] = "scc_effect"
    if lc == "weight": rename_map[c] = "weight"
edges = edges.rename(columns=rename_map)

required = {"scc_cause","scc_effect"}
if not required.issubset(set(edges.columns)):
    raise ValueError(f"SCC edges file must include {required}, got columns={list(edges.columns)}")

# keep numeric SCC ids
edges["scc_cause"]  = edges["scc_cause"].astype(int)
edges["scc_effect"] = edges["scc_effect"].astype(int)
if "weight" in edges.columns:
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")

n_scc = int(scc_key["scc_id"].max()) + 1

# ==========================
# 2) Build SCC DAG (important: add all nodes first)
# ==========================
G = nx.DiGraph()
G.add_nodes_from(range(n_scc))
for _, r in edges.iterrows():
    G.add_edge(int(r["scc_cause"]), int(r["scc_effect"]), weight=float(r["weight"]) if "weight" in edges.columns and pd.notna(r["weight"]) else 1.0)

# ==========================
# 3) Identify liquidity SCCs via proxies
# ==========================
proxy_to_scc = []
liquidity_sccs = set()

members_map = dict(zip(scc_key["scc_id"], scc_key["members"]))

for proxy in LIQUIDITY_PROXIES:
    hit = scc_key[scc_key["member_list"].apply(lambda xs: proxy in xs)]
    if len(hit) == 0:
        # not fatal; some proxy might not exist in your 50 indicators
        continue
    for _, row in hit.iterrows():
        sid = int(row["scc_id"])
        liquidity_sccs.add(sid)
        proxy_to_scc.append({
            "proxy_var": proxy,
            "scc_id": sid,
            "scc_members": row["members"]
        })

liq_df = pd.DataFrame(proxy_to_scc).drop_duplicates().sort_values(["scc_id","proxy_var"])
liq_df.to_csv(OUT_LIQ_NODES, index=False)
print(f"[OK] Liquidity SCCs: {sorted(list(liquidity_sccs))}")
print(f"[OK] Saved: {OUT_LIQ_NODES}")

if len(liquidity_sccs) == 0:
    raise RuntimeError("No liquidity SCC found. Check LIQUIDITY_PROXIES names vs your macro variables.")

# ==========================
# 4) Focus subgraph = all upstream SCCs that can reach any liquidity SCC + liquidity SCCs
# ==========================
upstream = set()
for t in liquidity_sccs:
    if t in G:
        upstream |= nx.ancestors(G, t)

focus_nodes = upstream | set(liquidity_sccs)
H = G.subgraph(sorted(list(focus_nodes))).copy()

# roots in focus graph (in-degree==0)
roots = [n for n in H.nodes() if H.in_degree(n) == 0]

# ==========================
# 5) Rank upstream SCCs
# ==========================
rows = []
for n in H.nodes():
    # reach how many liquidity SCCs?
    reach = 0
    min_hops = np.inf
    for t in liquidity_sccs:
        if n == t:
            reach += 1
            min_hops = min(min_hops, 0)
            continue
        if nx.has_path(H, n, t):
            reach += 1
            try:
                d = nx.shortest_path_length(H, n, t)
                min_hops = min(min_hops, d)
            except nx.NetworkXNoPath:
                pass

    rows.append({
        "scc_id": n,
        "is_liquidity": int(n in liquidity_sccs),
        "in_degree_in_focus": int(H.in_degree(n)),
        "out_degree_in_focus": int(H.out_degree(n)),
        "can_reach_liquidity_count": int(reach),
        "min_hops_to_liquidity": int(min_hops) if np.isfinite(min_hops) else None,
        "members": members_map.get(n, "")
    })

rank_df = pd.DataFrame(rows)
# score: prefer high reach, then closer, then more outgoing influence
rank_df["score"] = (
    rank_df["can_reach_liquidity_count"] * 100
    + rank_df["out_degree_in_focus"] * 5
    - rank_df["min_hops_to_liquidity"].fillna(999)
)
rank_df = rank_df.sort_values(["score","can_reach_liquidity_count","min_hops_to_liquidity"], ascending=[False,False,True])
rank_df.to_csv(OUT_RANK, index=False)
print(f"[OK] Saved: {OUT_RANK}")

# ==========================
# 6) Root -> Liquidity shortest paths
# ==========================
path_rows = []
for r in roots:
    for t in liquidity_sccs:
        if r == t:
            path_rows.append({"root_scc": r, "target_liquidity_scc": t, "path": f"{r}->{t}", "hops": 0})
            continue
        if nx.has_path(H, r, t):
            p = nx.shortest_path(H, r, t)
            path_rows.append({
                "root_scc": r,
                "target_liquidity_scc": t,
                "path": "->".join(map(str,p)),
                "hops": len(p)-1
            })

paths_df = pd.DataFrame(path_rows).sort_values(["hops","root_scc","target_liquidity_scc"])
paths_df.to_csv(OUT_PATHS, index=False)
print(f"[OK] Saved: {OUT_PATHS}")

# ==========================
# 7) Plot: layered layout (roots -> middle -> liquidity)
# ==========================
# assign layer by min distance to ANY liquidity SCC (reverse layering)
layer = {}
for n in H.nodes():
    if n in liquidity_sccs:
        layer[n] = 2
    elif n in roots:
        layer[n] = 0
    else:
        layer[n] = 1

# position nodes vertically by score rank within each layer
pos = {}
for L in [0,1,2]:
    nodes_L = [n for n in H.nodes() if layer[n] == L]
    # sort by score descending for readability
    nodes_L = sorted(nodes_L, key=lambda x: float(rank_df.loc[rank_df["scc_id"]==x, "score"].values[0]) if (rank_df["scc_id"]==x).any() else 0.0, reverse=True)
    for i, n in enumerate(nodes_L):
        pos[n] = (L, -i)

# colors
node_colors = []
for n in H.nodes():
    if n in liquidity_sccs:
        node_colors.append("red")
    elif n in roots:
        node_colors.append("green")
    else:
        node_colors.append("steelblue")

plt.figure(figsize=(18,8))
nx.draw_networkx_edges(H, pos, arrows=True, alpha=0.35, width=1.2)
nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=800, alpha=0.9)
nx.draw_networkx_labels(H, pos, font_size=9)

plt.title("Deliverable B: Focused SCC-DAG (Upstream -> Liquidity SCCs)\n(red=liquidity, green=roots)")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"[OK] Saved: {OUT_PNG}")
