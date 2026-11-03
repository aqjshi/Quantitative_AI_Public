import sys
import os
import json
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict, deque

import pandas as pd
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database.identity import _prepare_global_identity
from accounting.config import load_configuration
from accounting.database.fetcher import _fetch_clean_universe


# ──────────────────────────────────────────────────────────────────────────────
# Static non-data-driven accounting DAG.
# Edge value is child / parent. Missing child/parent fields are treated as 0.
# ──────────────────────────────────────────────────────────────────────────────
STATIC_ACCOUNTING_DAG = {
    # Income statement branch
    "revenue": [
        "gross_profit",
        "cost_of_revenue",
        "operating_income",
        "research_development",
        "selling_general_administrative",
        "receivables",
    ],
    "gross_profit": [
        "operating_income",
        "total_operating_expenses",
        "research_development",
        "selling_general_administrative",
    ],
    "operating_income": [
        "income_before_income_taxes",
        "income_taxes",
        "interest_expense",
        "net_income_loss_attributable_common_shareholders",
        "ebitda",
    ],
    "income_before_income_taxes": [
        "income_taxes",
        "net_income_loss_attributable_common_shareholders",
    ],

    # Balance sheet branch
    "total_assets": [
        "cash_and_equivalents",
        "short_term_investments",
        "liquid_assets",
        "receivables",
        "inventories",
        "property_plant_equipment_net",
        "goodwill",
        "intangible_assets_net",
        "total_current_assets",
        "total_liabilities",
        "total_equity",
        "total_liabilities_and_equity",
        "retained_earnings_deficit",
    ],
    "total_current_assets": [
        "cash_and_equivalents",
        "short_term_investments",
        "receivables",
        "inventories",
        "other_current_assets",
    ],
    "total_liabilities": [
        "accounts_payable",
        "debt_current",
        "long_term_debt_and_capital_lease_obligations",
        "total_debt",
        "total_current_liabilities",
        "other_noncurrent_liabilities",
    ],
    "total_equity": [
        "common_stock",
        "additional_paid_in_capital",
        "retained_earnings_deficit",
        "treasury_stock",
        "total_equity_attributable_to_parent",
    ],

    # Cash-flow branch
    "net_cash_from_operating_activities": [
        "net_income",
        "depreciation_depletion_and_amortization",
        "change_in_other_operating_assets_and_liabilities_net",
        "purchase_of_property_plant_and_equipment",
        "net_cash_from_investing_activities",
        "net_cash_from_financing_activities",
        "dividends",
        "capex",
    ],
    "net_cash_from_investing_activities": [
        "purchase_of_property_plant_and_equipment",
        "sale_of_property_plant_and_equipment",
        "other_investing_activities",
    ],
    "net_cash_from_financing_activities": [
        "dividends",
        "long_term_debt_issuances_repayments",
        "short_term_debt_issuances_repayments",
        "other_financing_activities",
    ],
}

DEFAULT_SOURCES = ["revenue", "total_assets", "net_cash_from_operating_activities"]
TARGET_IMAGE_DATES = [pd.Timestamp("2020-04-01"), pd.Timestamp("2026-04-01")]
OUTPUT_DIR = Path("accounting/outputs")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def safe_ratio(child, parent, clip: float = 10.0) -> float:
    child_f = safe_float(child)
    parent_f = safe_float(parent)
    if parent_f == 0.0:
        return 0.0
    value = child_f / parent_f
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(np.clip(value, -clip, clip))


def normalized_abs_error(lhs, rhs) -> float:
    lhs_f = safe_float(lhs)
    rhs_f = safe_float(rhs)
    scale = max(abs(lhs_f), abs(rhs_f), 1.0)
    return abs(lhs_f - rhs_f) / scale


def field_value(row: pd.Series, field: str) -> float:
    if field not in row.index:
        return 0.0
    return safe_float(row[field])


def build_edges_from_sources(sources: list[str], max_depth: int) -> list[tuple[str, str]]:
    """Bounded BFS over STATIC_ACCOUNTING_DAG. Returns stable unique edge list."""
    edges: list[tuple[str, str]] = []
    seen_edges = set()
    frontier = deque([(src, 0) for src in sources])

    while frontier:
        parent, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for child in STATIC_ACCOUNTING_DAG.get(parent, []):
            edge = (parent, child)
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
            frontier.append((child, depth + 1))

    return edges


def edge_key(parent: str, child: str) -> str:
    return f"{parent}__to__{child}"


def parse_edge_key(key: str) -> tuple[str, str]:
    parent, child = key.split("__to__", 1)
    return parent, child


def accounting_dependency_dag(
    sources: list[str],
    balance_sheet_vars: list[str],
    cash_flow_vars: list[str],
    income_statement_vars: list[str],
    max_depth: int,
) -> list[str]:
    """
    Returns a deterministic linearized DAG layout.
    Each element is an edge key: parent__to__child.
    Vector length = number of accounting DAG edges at this BFS depth.
    """
    allowed = set(balance_sheet_vars) | set(cash_flow_vars) | set(income_statement_vars)
    allowed |= {"total_debt", "liquid_assets", "capex", "cfo", "dda"}
    allowed |= set(sources)

    edges = build_edges_from_sources(sources, max_depth=max_depth)
    filtered_edges = [(p, c) for p, c in edges if p in allowed and c in allowed]
    return [edge_key(p, c) for p, c in filtered_edges]


def latest_statement_before(ticker_df: pd.DataFrame, target_date: datetime, is_ttm: bool | None = None) -> pd.DataFrame:
    """Returns a one-row DataFrame for the latest point-in-time statement before target_date."""
    if ticker_df.empty:
        return pd.DataFrame()

    df = ticker_df.copy()
    df = df[df["timeframe"].notna()]
    if df.empty:
        return pd.DataFrame()

    df["filing_date_t"] = pd.to_datetime(df["filing_date_t"])
    df = df[df["filing_date_t"] <= pd.to_datetime(target_date)]
    if df.empty:
        return pd.DataFrame()

    if is_ttm is not None and "timeframe" in df.columns:
        timeframe = df["timeframe"].astype(str).str.lower()
        if is_ttm:
            preferred = df[timeframe.str.contains("ttm|trailing", regex=True, na=False)]
        else:
            preferred = df[timeframe.str.contains("q|quarter", regex=True, na=False)]
        if not preferred.empty:
            df = preferred

    return df.sort_values(by="filing_date_t").tail(1)


def vectorize_accounting_state(
    ticker_df: pd.DataFrame,
    layout: list[str],
    target_date: datetime,
    is_ttm: bool = False,
) -> np.ndarray:
    """
    Produces a scale-invariant vector of child / parent ratios.
    Missing/null accounting fields default to 0 so sparse tickers do not crash.
    """
    snapshot = latest_statement_before(ticker_df, target_date, is_ttm=is_ttm)
    if snapshot.empty:
        return np.zeros(len(layout), dtype=np.float32)

    row = snapshot.iloc[0]
    values = []
    for key in layout:
        parent, child = parse_edge_key(key)
        values.append(safe_ratio(field_value(row, child), field_value(row, parent)))
    return np.asarray(values, dtype=np.float32)


def adjacency_from_snapshot(snapshot_df: pd.DataFrame, sources: list[str], max_depth: int) -> dict:
    """Builds JSON adjacency list with edge weights = child / parent."""
    if snapshot_df.empty:
        return {}
    row = snapshot_df.iloc[0]
    adj = defaultdict(dict)
    for parent, child in build_edges_from_sources(sources, max_depth=max_depth):
        adj[parent][child] = safe_ratio(field_value(row, child), field_value(row, parent))
    return {p: dict(children) for p, children in adj.items()}


def save_json(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_graph_image(adj: dict, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    G = nx.DiGraph()
    for parent, children in adj.items():
        for child, weight in children.items():
            G.add_edge(parent, child, weight=round(float(weight), 3))

    plt.figure(figsize=(16, 11))
    if len(G.nodes) == 0:
        plt.title(title + " (empty graph)")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(G, seed=42, k=1.2)
    nx.draw_networkx_nodes(G, pos, node_size=1800)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=15, width=1.0)
    nx.draw_networkx_labels(G, pos, font_size=7)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def prefix_adjacency(adj: dict, ticker: str) -> dict:
    out = {}
    for parent, children in adj.items():
        p = f"{ticker}:{parent}"
        out[p] = {f"{ticker}:{child}": weight for child, weight in children.items()}
    return out


def merge_adjacencies(adjs: list[dict]) -> dict:
    merged = defaultdict(dict)
    for adj in adjs:
        for parent, children in adj.items():
            merged[parent].update(children)
    return {p: dict(children) for p, children in merged.items()}


def accounting_dependency_dag_error(snapshot_df: pd.DataFrame, sources: list[str], max_depth: int) -> dict:
    """Computes stratified accounting identity/reconstruction errors for one snapshot."""
    if snapshot_df.empty:
        return {
            "balance_sheet_deviation": np.nan,
            "cash_flow_deviation": np.nan,
            "income_statement_deviation": np.nan,
            "dag_reconstruction_mae": np.nan,
            "mae": np.nan,
        }

    row = snapshot_df.iloc[0]

    # Balance sheet identity: Assets ≈ Liabilities + Equity
    bs_1 = normalized_abs_error(
        field_value(row, "total_assets"),
        field_value(row, "total_liabilities_and_equity"),
    )
    bs_2 = normalized_abs_error(
        field_value(row, "total_assets"),
        field_value(row, "total_liabilities") + field_value(row, "total_equity"),
    )
    balance_sheet_deviation = float(np.nanmean([bs_1, bs_2]))

    # Income statement identities
    inc_1 = normalized_abs_error(
        field_value(row, "revenue"),
        field_value(row, "cost_of_revenue") + field_value(row, "gross_profit"),
    )
    inc_2 = normalized_abs_error(
        field_value(row, "gross_profit"),
        field_value(row, "total_operating_expenses") + field_value(row, "operating_income"),
    )
    income_statement_deviation = float(np.nanmean([inc_1, inc_2]))

    # Cash flow identity: Δcash ≈ CFO + CFI + CFF + FX
    cash_flow_deviation = normalized_abs_error(
        field_value(row, "change_in_cash_and_equivalents"),
        field_value(row, "net_cash_from_operating_activities")
        + field_value(row, "net_cash_from_investing_activities")
        + field_value(row, "net_cash_from_financing_activities")
        + field_value(row, "effect_of_currency_exchange_rate"),
    )

    # DAG reconstruction sanity: parent * ratio should reconstruct child unless missing/defaulted.
    dag_errors = []
    for parent, child in build_edges_from_sources(sources, max_depth=max_depth):
        p_val = field_value(row, parent)
        c_val = field_value(row, child)
        ratio = safe_ratio(c_val, p_val)
        pred_child = p_val * ratio
        dag_errors.append(normalized_abs_error(c_val, pred_child))
    dag_reconstruction_mae = float(np.nanmean(dag_errors)) if dag_errors else np.nan

    mae = float(np.nanmean([
        balance_sheet_deviation,
        cash_flow_deviation,
        income_statement_deviation,
        dag_reconstruction_mae,
    ]))

    return {
        "balance_sheet_deviation": balance_sheet_deviation,
        "cash_flow_deviation": cash_flow_deviation,
        "income_statement_deviation": income_statement_deviation,
        "dag_reconstruction_mae": dag_reconstruction_mae,
        "mae": mae,
    }


def iter_batches(start_date: str, end_date: str, batch_months: int):
    """Yields chunked chronological windows based on stride months configuration."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    curr = start_dt
    while curr <= end_dt:
        nxt = min(curr + pd.DateOffset(months=batch_months), end_dt)
        yield curr, nxt
        curr = nxt + pd.Timedelta(days=1)


def export_depth_sweep(config: dict, sources: list[str]):
    rows = []
    for depth in [1, 2, 3, 4]:
        layout = accounting_dependency_dag(
            sources=sources,
            balance_sheet_vars=config["balance_sheet_vars"],
            cash_flow_vars=config["cash_flow_vars"],
            income_statement_vars=config["income_statement_vars"],
            max_depth=depth,
        )
        nodes = set()
        for key in layout:
            parent, child = parse_edge_key(key)
            nodes.add(parent)
            nodes.add(child)
        rows.append({
            "depth": depth,
            "node_count": len(nodes),
            "edge_count": len(layout),
            "vector_length": len(layout),
        })
    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "depth_sweep.csv", index=False)
    tqdm.write("[*] Wrote depth sweep to accounting/outputs/depth_sweep.csv")
    tqdm.write(df.to_string(index=False))


def target_dates_inside_batch(batch_start, batch_end):
    return [d for d in TARGET_IMAGE_DATES if pd.to_datetime(batch_start) <= d <= pd.to_datetime(batch_end)]


def main():
    config = load_configuration()
    today_str = datetime.today().strftime("%Y-%m-%d")
    config["train_end"] = today_str

    # Active DAG settings for main vectorization and image generation.
    depth = 2
    sources = DEFAULT_SOURCES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_depth_sweep(config, sources)

    identity_matrix = _prepare_global_identity(
        config["case_study"],
        datetime.today(),
        config["reconstruction_heartbeat_freq_months"],
    )

    if identity_matrix.empty:
        tqdm.write("[FATAL] Identity matrix tracking returned empty. Check input tickers / database.")
        return

    global_error_registry = []
    cross_asset_registry = {d.strftime("%Y-%m-%d"): [] for d in TARGET_IMAGE_DATES}
    emitted_ticker_graphs = set()

    for batch_start, batch_end in iter_batches(config["train_start"], config["train_end"], config["batch_stride_months"]):
        tqdm.write(
            f"\n[BATCH INTERVAL] ANCHOR AT [{batch_end.strftime('%Y-%m-%d')}] "
            f"Executing window: {batch_start.strftime('%Y-%m-%d')} -> {batch_end.strftime('%Y-%m-%d')}"
        )

        full_fundamentals_df = _fetch_clean_universe(
            identity_matrix,
            batch_start,
            batch_end,
            context_depth_days=config["context_depth_days"],
            forecast_depth_days=config["forecast_depth_days"],
            fundamental_context_depth_months=config["fundamental_context_depth_months"],
            fundamental_proxy_filing_date_delta_days=config["fundamental_proxy_filing_date_delta_days"],
            reconstruction_heartbeat_freq_months=config["reconstruction_heartbeat_freq_months"],
        )

        if full_fundamentals_df.empty:
            continue

        layout_vector = accounting_dependency_dag(
            sources=sources,
            balance_sheet_vars=config["balance_sheet_vars"],
            cash_flow_vars=config["cash_flow_vars"],
            income_statement_vars=config["income_statement_vars"],
            max_depth=depth,
        )

        batch_target_dates = target_dates_inside_batch(batch_start, batch_end)

        for row in identity_matrix.itertuples(index=False):
            ticker = row.ticker
            ticker_hash = int(row.ticker_hash)

            asset_fund_df = full_fundamentals_df[full_fundamentals_df["ticker_hash"] == ticker_hash]
            if asset_fund_df.empty:
                continue

            anchor_date = batch_end
            snapshot_slice = latest_statement_before(asset_fund_df, anchor_date, is_ttm=False)
            if snapshot_slice.empty:
                continue

            actual_filing_dt = snapshot_slice["filing_date_t"].iloc[0]
            if (anchor_date - actual_filing_dt) > timedelta(days=config["fundamental_period_end_stale_delta_days"]):
                continue

            error_metrics = accounting_dependency_dag_error(snapshot_slice, sources=sources, max_depth=depth)
            meta_record = {
                "ticker": ticker,
                "ticker_hash": ticker_hash,
                "anchor_date": anchor_date.strftime("%Y-%m-%d"),
                "period_end": snapshot_slice["period_end_t"].iloc[0],
                "filing_date": actual_filing_dt,
                **error_metrics,
            }
            global_error_registry.append(meta_record)

            v_q = vectorize_accounting_state(asset_fund_df, layout_vector, anchor_date, is_ttm=False)
            v_ttm = vectorize_accounting_state(asset_fund_df, layout_vector, anchor_date, is_ttm=True)

            revenue_val = snapshot_slice["revenue"].iloc[0] if "revenue" in snapshot_slice.columns else "N/A"
            tassets_val = snapshot_slice["total_assets"].iloc[0] if "total_assets" in snapshot_slice.columns else "N/A"

            p_end_dt = snapshot_slice["period_end_t"].iloc[0] if "period_end_t" in snapshot_slice.columns else None
            f_date_dt = snapshot_slice["filing_date_t"].iloc[0] if "filing_date_t" in snapshot_slice.columns else None
            p_end_str = p_end_dt.strftime("%Y-%m-%d") if pd.notna(p_end_dt) else "N/A"
            f_date_str = f_date_dt.strftime("%Y-%m-%d") if pd.notna(f_date_dt) else "N/A"

            print(
                f"[DIAGNOSTIC] Ticker: {ticker:<5} | "
                f"Len linearized Q: {len(v_q):<3} | "
                f"Len linearized TTM: {len(v_ttm):<3} | "
                f"Period End: {p_end_str} | "
                f"Filing Date: {f_date_str} | "
                f"Revenue: {revenue_val} | "
                f"Total Asset: {tassets_val} | "
                f"MAE: {error_metrics['mae']:.6f}"
            )

            # Deliverable: images + JSON adjacency list at target dates.
            for target_date in batch_target_dates:
                date_key = target_date.strftime("%Y-%m-%d")
                graph_key = (ticker_hash, date_key)
                if graph_key in emitted_ticker_graphs:
                    continue

                target_snapshot = latest_statement_before(asset_fund_df, target_date, is_ttm=False)
                if target_snapshot.empty:
                    continue

                adj = adjacency_from_snapshot(target_snapshot, sources=sources, max_depth=depth)
                date_dir = OUTPUT_DIR / date_key
                safe_ticker = str(ticker).replace("/", "_")
                save_json(adj, date_dir / f"{safe_ticker}_adj.json")
                save_graph_image(adj, date_dir / f"{safe_ticker}_graph.png", f"{ticker} accounting DAG at {date_key}")
                cross_asset_registry[date_key].append(prefix_adjacency(adj, ticker))
                emitted_ticker_graphs.add(graph_key)

        # Write incremental validation outputs after each batch.
        if global_error_registry:
            error_df = pd.DataFrame(global_error_registry)
            error_df["anchor_date"] = pd.to_datetime(error_df["anchor_date"])
            error_df = error_df.sort_values(["ticker", "anchor_date"])
            error_df["cumsum_error"] = error_df.groupby("ticker")["mae"].cumsum()

            validation_dir = OUTPUT_DIR / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            error_df.to_csv(validation_dir / "error_by_ticker_over_time.csv", index=False)

            aggregate = (
                error_df.groupby("anchor_date")[[
                    "balance_sheet_deviation",
                    "cash_flow_deviation",
                    "income_statement_deviation",
                    "dag_reconstruction_mae",
                    "mae",
                ]]
                .mean()
                .reset_index()
            )
            aggregate.to_csv(validation_dir / "aggregate_error_over_time.csv", index=False)

    # Cross-asset graph export for target dates.
    for date_key, prefixed_adjs in cross_asset_registry.items():
        if not prefixed_adjs:
            continue
        cross_adj = merge_adjacencies(prefixed_adjs)
        date_dir = OUTPUT_DIR / date_key
        save_json(cross_adj, date_dir / "cross_asset_adj.json")
        save_graph_image(cross_adj, date_dir / "cross_asset_graph.png", f"Cross-asset accounting DAG at {date_key}")

    # Final error plot.
    validation_csv = OUTPUT_DIR / "validation" / "aggregate_error_over_time.csv"
    if validation_csv.exists():
        aggregate = pd.read_csv(validation_csv)
        aggregate["anchor_date"] = pd.to_datetime(aggregate["anchor_date"])
        plt.figure(figsize=(12, 6))
        plt.plot(aggregate["anchor_date"], aggregate["mae"], marker="o", label="aggregate_mae")
        plt.plot(aggregate["anchor_date"], aggregate["balance_sheet_deviation"], marker="o", label="balance_sheet")
        plt.plot(aggregate["anchor_date"], aggregate["cash_flow_deviation"], marker="o", label="cash_flow")
        plt.plot(aggregate["anchor_date"], aggregate["income_statement_deviation"], marker="o", label="income_statement")
        plt.legend()
        plt.title("Accounting validation error over time")
        plt.xlabel("Anchor date")
        plt.ylabel("Normalized error")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "validation" / "error_over_time.png", dpi=180)
        plt.close()

    tqdm.write("\n[DONE] Accounting DAG outputs written under accounting/outputs/")


if __name__ == "__main__":
    main()
