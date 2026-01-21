#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fred_pc_from_json_submit.py

Reads indicators from a JSON file,
fetches FRED data (2000-2025), preprocesses (smoothing -> log-normal -> zscore),
runs PC (causal-learn), exports:
- pc_graph.png
- nodes.csv (n nodes)
- edges.csv (m edges + m weights float)
- incidence_mxn.csv (m x n edge-node incidence matrix; matches "m*n adjacency" wording)
- adjacency_nxn.csv (extra)
- SCC condensation DAG: scc_graph.png, scc_nodes.csv, scc_edges.csv

Usage:
  pip install fredapi causallearn networkx pydot requests pandas numpy
  export FRED_API_KEY="your_key"   
  python fred_pc_from_json_submit.py --json_path fred_50_for_causal_pc.json --outdir submission_out
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fredapi import Fred

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

import networkx as nx

# -------------------------
# Ensure Graphviz is in PATH (Windows compatibility)
# -------------------------
def ensure_graphviz_in_path():
    """Add Graphviz bin directory to PATH if not already present."""
    import sys
    graphviz_path = r"C:\Program Files\Graphviz\bin"
    if sys.platform == "win32" and os.path.exists(graphviz_path):
        current_path = os.environ.get("PATH", "")
        if graphviz_path not in current_path:
            os.environ["PATH"] = current_path + os.pathsep + graphviz_path

# Call this early to ensure Graphviz is available
ensure_graphviz_in_path()

# -------------------------
# JSON loader
# -------------------------
def load_indicators_json(json_path: str) -> pd.DataFrame:
    """
    Expected JSON structure:
    {
      "meta": {...},
      "indicators": [
        {"node":"a1","fred_id":"UNRATE", ...},
        ...
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "indicators" not in cfg or not isinstance(cfg["indicators"], list):
        raise ValueError("JSON must contain 'indicators' as a list.")

    rows = []
    for item in cfg["indicators"]:
        if "node" not in item or "fred_id" not in item:
            raise ValueError("Each indicator must have 'node' and 'fred_id'.")
        rows.append({
            "node": str(item["node"]),
            "series_id": str(item["fred_id"]),
            # keep optional fields for documentation/debugging
            "block": item.get("block", ""),
            "role": item.get("role", ""),
            "transform_hint": item.get("transform", "")
        })

    df = pd.DataFrame(rows)

    # basic validation
    if df["node"].duplicated().any():
        dup = df[df["node"].duplicated()]["node"].tolist()
        raise ValueError(f"Duplicate node labels in JSON: {dup}")

    if df["series_id"].duplicated().any():
        # allowed, but generally undesirable
        pass

    # sort by numeric suffix if nodes are a1..a50
    def node_key(x: str) -> int:
        try:
            return int(x[1:])
        except Exception:
            return 10**9
    df = df.sort_values("node", key=lambda s: s.map(node_key)).reset_index(drop=True)

    return df


# -------------------------
# Data fetch + preprocessing
# -------------------------
def fetch_weekly_series(
    fred: Fred, series_id: str, start: str, end: str, weekly_rule: str = "W-FRI"
) -> pd.Series:
    """
    Fetch series and align to weekly.
    """
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = pd.Series(s)
    s.index = pd.to_datetime(s.index)

    # Weekly align
    s = s.resample(weekly_rule).mean()

    # Fill gaps (linear interpolation + forward/back fill)
    s = s.interpolate("linear").ffill().bfill()
    return s


def smooth_then_log_normal(
    df: pd.DataFrame, smooth_window: int, eps: float = 1e-6
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Required: smoothing then log normal transform.

    - smoothing: rolling mean
    - log-normal transform: log( x - min(x) + eps ) to ensure positivity
    - then z-score (helps FisherZ)
    """
    # 1) smoothing
    df_smooth = df.rolling(smooth_window, min_periods=1).mean()

    # 2) log-normal (robust shift)
    df_log = df_smooth.copy()
    for c in df_log.columns:
        x = df_log[c].astype(float)
        xmin = np.nanmin(x.values)
        x_shift = x - xmin + eps
        df_log[c] = np.log(x_shift)

    # 3) z-score
    df_z = (df_log - df_log.mean()) / (df_log.std(ddof=0) + 1e-12)
    df_z = df_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df_smooth, df_log, df_z


# -------------------------
# PC edge extraction + matrices
# -------------------------
def extract_edges_with_weights_bidirectional_for_uncertain(
    cg, labels: List[str], data_z: np.ndarray
) -> pd.DataFrame:
    """
    Export m edges + m weights (float).
    - PC output can include undirected/partially directed edges.
    - To preserve potential cycles (for SCC), treat uncertain edges as bidirectional.
    Weight: abs(corr) on transformed z data (simple + stable).
    """
    G = cg.G
    n = len(labels)
    edges = []

    def is_tail(m):  return m == -1
    def is_arrow(m): return m == 1

    for i in range(n):
        for j in range(i + 1, n):
            if G.graph[i, j] == 0 and G.graph[j, i] == 0:
                continue

            w = float(abs(np.corrcoef(data_z[:, i], data_z[:, j])[0, 1]))
            mark_ij = G.graph[i, j]
            mark_ji = G.graph[j, i]

            if is_tail(mark_ij) and is_arrow(mark_ji):
                edges.append((labels[i], labels[j], w))
            elif is_arrow(mark_ij) and is_tail(mark_ji):
                edges.append((labels[j], labels[i], w))
            else:
                edges.append((labels[i], labels[j], w))
                edges.append((labels[j], labels[i], w))

    edf = pd.DataFrame(edges, columns=["src", "dst", "weight"])
    edf = edf.groupby(["src", "dst"], as_index=False)["weight"].max()
    edf.insert(0, "edge_id", [f"e{k+1}" for k in range(len(edf))])
    return edf


def build_weighted_adjacency_nxn(labels: List[str], edges_df: pd.DataFrame) -> pd.DataFrame:
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    A = np.zeros((n, n), dtype=float)
    for _, r in edges_df.iterrows():
        A[idx[r["src"]], idx[r["dst"]]] = float(r["weight"])
    return pd.DataFrame(A, index=labels, columns=labels)


def build_incidence_mxn(labels: List[str], edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    "adjacency matrix of m * n size" -> incidence matrix (m x n).
    For directed edge src->dst with weight w:
      M[e, src] = -w
      M[e, dst] = +w
    """
    idx = {lab: i for i, lab in enumerate(labels)}
    m = len(edges_df)
    n = len(labels)
    M = np.zeros((m, n), dtype=float)

    for k, (_, r) in enumerate(edges_df.iterrows()):
        s = idx[r["src"]]
        t = idx[r["dst"]]
        w = float(r["weight"])
        M[k, s] = -w
        M[k, t] = +w

    return pd.DataFrame(M, index=edges_df["edge_id"].tolist(), columns=labels)


# -------------------------
# SCC condensation
# -------------------------
def compute_scc_and_condensation(edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph]:
    DG = nx.DiGraph()
    for _, r in edges_df.iterrows():
        DG.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    sccs = list(nx.strongly_connected_components(DG))

    node2scc: Dict[str, str] = {}
    for i, comp in enumerate(sccs, 1):
        for node in comp:
            node2scc[node] = f"SCC_{i}"

    scc_nodes = pd.DataFrame(
        [{"scc": f"SCC_{i+1}", "size": len(comp), "members": ",".join(sorted(list(comp)))}
         for i, comp in enumerate(sccs)]
    ).sort_values("size", ascending=False)

    CDG = nx.DiGraph()
    for u, v, d in DG.edges(data=True):
        su, sv = node2scc[u], node2scc[v]
        if su == sv:
            continue
        w = float(d.get("weight", 0.0))
        if CDG.has_edge(su, sv):
            CDG[su][sv]["weight"] = max(CDG[su][sv]["weight"], w)
        else:
            CDG.add_edge(su, sv, weight=w)

    scc_edges = pd.DataFrame(
        [{"src_scc": u, "dst_scc": v, "weight": float(d.get("weight", 0.0))}
         for u, v, d in CDG.edges(data=True)]
    )
    return scc_nodes, scc_edges, CDG


def draw_scc_graph_png(CDG: nx.DiGraph, out_png: str):
    try:
        import pydot  # noqa: F401
        from networkx.drawing.nx_pydot import to_pydot
        pdot = to_pydot(CDG)
        pdot.write_png(out_png)
    except Exception:
        with open(out_png.replace(".png", ".txt"), "w", encoding="utf-8") as f:
            f.write("pydot/graphviz not available; SCC edges:\n")
            for u, v, d in CDG.edges(data=True):
                f.write(f"{u} -> {v} (w={d.get('weight', 0.0)})\n")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="fred_50_for_causal_pc.json")
    parser.add_argument("--api_key", type=str, default="1c9a5e4dda9d1a15de4b6d959faf545f", help="FRED API key or env FRED_API_KEY.")
    parser.add_argument("--outdir", type=str, default="submission_out")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--weekly_rule", type=str, default="W-FRI")
    parser.add_argument("--smooth_window", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("ERROR: Please set FRED_API_KEY env var or pass --api_key.")

    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # 1) load nodes from JSON
    nodes_df = load_indicators_json(args.json_path)
    if len(nodes_df) < 50:
        raise SystemExit(f"ERROR: JSON must contain at least 50 indicators. Got {len(nodes_df)}")

    # 2) fetch data
    fred = Fred(api_key=args.api_key)
    series_data = {}
    failed = []

    # We keep node label as the column name (a1..a50), as required.
    for _, r in nodes_df.iterrows():
        node = r["node"]
        sid = r["series_id"]
        try:
            s = fetch_weekly_series(fred, sid, args.start, args.end, args.weekly_rule)
            series_data[node] = s
        except Exception as e:
            failed.append({"node": node, "series_id": sid, "err": str(e)})

    if len(series_data) < 50:
        # If a few fail, it's better to stop and fix the JSON list (research-driven).
        msg = "\n".join([f"{x['node']} {x['series_id']} -> {x['err']}" for x in failed[:10]])
        raise SystemExit(f"ERROR: Only fetched {len(series_data)} series (<50). Fix these failures first:\n{msg}")

    # align into a single weekly dataframe
    df_weekly = pd.DataFrame(series_data).dropna().ffill().bfill()

    # Ensure column ordering is node order a1..a50
    df_weekly = df_weekly[nodes_df["node"].tolist()]

    # save nodes with metadata (block/role hints)
    nodes_df.to_csv(os.path.join(args.outdir, "nodes.csv"), index=False)
    df_weekly.to_csv(os.path.join(args.outdir, "data_weekly_raw.csv"))

    # 3) preprocess: smoothing -> log-normal -> zscore
    df_smooth, df_log, df_z = smooth_then_log_normal(df_weekly, smooth_window=args.smooth_window)
    df_smooth.to_csv(os.path.join(args.outdir, "data_weekly_smooth.csv"))
    df_log.to_csv(os.path.join(args.outdir, "data_weekly_log.csv"))
    df_z.to_csv(os.path.join(args.outdir, "data_weekly_log_z.csv"))

    # 4) run PC
    data = df_z.values.astype(float)
    labels = list(df_z.columns)

    cg = pc(
        data,
        args.alpha,
        "fisherz",
        True,    # stable
        0,       # uc_rule
        2,       # uc_priority
        False,   # mvpc
        "MV_Crtn_Fisher_Z",
        None,
        False,   # verbose
        True     # show_progress
    )

    # 5) export graph png
    pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    pyd.write_png(os.path.join(args.outdir, "pc_graph.png"))

    # 6) edges + weights
    edges_df = extract_edges_with_weights_bidirectional_for_uncertain(cg, labels, data)
    edges_df.to_csv(os.path.join(args.outdir, "edges.csv"), index=False)

    # 7) matrices
    incidence = build_incidence_mxn(labels, edges_df)
    incidence.to_csv(os.path.join(args.outdir, "incidence_mxn.csv"))

    adj_nxn = build_weighted_adjacency_nxn(labels, edges_df)
    adj_nxn.to_csv(os.path.join(args.outdir, "adjacency_nxn.csv"))

    # 8) SCC condensation DAG
    scc_nodes, scc_edges, CDG = compute_scc_and_condensation(edges_df)
    scc_nodes.to_csv(os.path.join(args.outdir, "scc_nodes.csv"), index=False)
    scc_edges.to_csv(os.path.join(args.outdir, "scc_edges.csv"), index=False)
    draw_scc_graph_png(CDG, os.path.join(args.outdir, "scc_graph.png"))

    with open(os.path.join(args.outdir, "scc_description.txt"), "w", encoding="utf-8") as f:
        f.write("SCC membership (largest first):\n\n")
        for _, r in scc_nodes.iterrows():
            f.write(f"{r['scc']} (size={r['size']}): {r['members']}\n")

    # 9) run manifest for traceability
    run_manifest = {
        "json_path": args.json_path,
        "time_period": {"start": args.start, "end": args.end},
        "preprocessing": {"weekly_rule": args.weekly_rule, "smooth_window": args.smooth_window, "transform": "log_normal_shift_then_log + zscore"},
        "pc_params": {"alpha": args.alpha, "indep_test": "fisherz", "stable": True, "uc_rule": 0, "uc_priority": 2},
        "n_nodes": len(labels),
        "n_edges": int(len(edges_df)),
        "failed_series": failed[:50]
    }
    with open(os.path.join(args.outdir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("DONE. Outputs written to:", args.outdir)
    if failed:
        print(f"Warning: {len(failed)} series failed (script stops if <50). See run_manifest.json.")


if __name__ == "__main__":
    main()
