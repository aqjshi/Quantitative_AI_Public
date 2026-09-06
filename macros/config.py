import os
import sys
import json
from typing import Tuple, Optional
from datetime import datetime
import secrets
import pandas as pd
import csv



def load_configuration():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            params = json.load(f)
    else:
        params = {}
    
    return {
        "fetch_start":                                              params.get("fetch_start", "2004-01-01").split(' ')[0],
        "train_start":                                              params.get("train_start", "2004-01-01").split(' ')[0],
        "train_earliest_lookback":                                  params.get("train_earliest_lookback", "2000-01-01").split(' ')[0],
        "download_batch_years":                                     params.get("download_batch_years", 5), 
        "fred_seed_category_id":                                    params.get("fred_seed_category_id", 0), 
        "random_seed":                                              params.get("random_seed",42),
        "string_hash_seed":                                         params.get("string_hash_seed",42),

        "num_workers":                                              params.get("num_workers", 2),
        "rate_limit_per_sec":                                       params.get("rate_limit_per_sec",20),
        "upsert_categories_bfs_depth":                              params.get("upsert_categories_bfs_depth", 2), 
        "upsert_series_lte_depth":                                  params.get("upsert_series_lte_depth", 3), 
        "upsert_categories_depth_1_whitelist":                      params.get("upsert_categories_depth_1_whitelist", {}), 

        "exogeneity_set"                                            :params.get("exogeneity_set", "seed_2.txt"),  
        "verbose":                                                  params.get("verbose", False),
        "fred_base_query":                                          params.get("fred_base_query","base_query.sql"),

        "fred_frequency_short_max_staleness_months":  params.get("fred_frequency_short_max_staleness_months", {"D":3, "M":3, "Q":9, "A":24}),
        "req_min_coverage_ratio":  params.get("req_min_coverage_ratio", 0.8),

        "causal_algorithm_list_supported":                          params.get("causal_algorithm_list_supported", ["kalman"]), 
        "causal_algorithm":                                         params.get("causal_algorithm", "kalman"),

        "walk_stride_months":                                       params.get("walk_stride_months", 24), 
        "production_forward_months":                                params.get("production_forward_months", 3), 
        "train_start_back_months":                                  params.get("train_start_back_months", 72), 
        "eval_start_back_months":                                   params.get("eval_start_back_months", 12), 
        "eval_end_back_months":                                     params.get("eval_end_back_months", 6), 
        
        "resample_frequency":                       params.get("resample_frequency", "1W"),
        "dynamic_factor_n_factors":                 params.get("dynamic_factor_n_factors",  2),
        "dynamic_factor_lag_order":                 params.get("dynamic_factor_lag_order",  1),
        "dynamic_factor_statespace_tolerance":      params.get("dynamic_factor_statespace_tolerance", 1e-5),
        "dynamic_factor_statespace_maxiter":        params.get("dynamic_factor_statespace_maxiter", 1000)
    }



def load_blacklist() -> set:
    """
    Parses the blacklist CSV file path provided via CLI argument 2 (sys.argv[2]).
    Returns a hash-set of integer category IDs for real-time O(1) BFS filtering.
    """
    # Fallback to default file name if argument 2 isn't explicitly passed
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "upsert_categories_blacklist.csv"
    
    blacklist_ids = set()
    
    if not os.path.exists(csv_path):
        print(f"[WARNING] Blacklist target '{csv_path}' not found. Initializing empty runtime filter.")
        return blacklist_ids

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        # Use csv.DictReader to safely isolate the target column regardless of row position
        reader = csv.DictReader(f)
        
        # Verify the schema contract
        if "category_id" not in reader.fieldnames:
            raise KeyError(f"Fatal Schema Error: '{csv_path}' missing mandatory 'category_id' column header.")
            
        for row in reader:
            try:
                # Cast string fields to integers for precise data type matching in the matrix
                cat_id = int(row["category_id"].strip())
                blacklist_ids.add(cat_id)
            except (ValueError, TypeError):
                # Gracefully skip empty rows or corrupted string artifacts
                continue
                
    
    print(f"[INFO] Blacklist operational. Successfully structuralized {len(blacklist_ids)} category firewalls.")
        
    return blacklist_ids

def initialize_run_environment(config: dict) -> Tuple[str, str]:
    """
    Sets up local directories, locks the unique execution hash,
    handles OS-level binary paths, and snapshots hyperparameters.
    """
    run_hash = secrets.token_hex(4)
    config["run_hash"] = run_hash

    # Handle OS-specific Graphviz paths
    if os.name == "nt" or config.get("MACHINE_TYPE") == "win":
        graphviz_bin = r"C:\Program Files\Graphviz\bin"
        if graphviz_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + graphviz_bin

    causal_algorithm = config["causal_algorithm"]
    meta_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "out", "macros", f"{causal_algorithm}_{run_hash}"
    )
    os.makedirs(meta_dir, exist_ok=True)

    # Snapshot active configuration hyperparameter space
    serializable_config = {}
    for k, v in config.items():
        if isinstance(v, (datetime, pd.Timestamp, pd.DatetimeIndex)):
            serializable_config[k] = str(v)
        else:
            serializable_config[k] = v

    config_snapshot_path = os.path.join(meta_dir, "config_snapshot.json")
    with open(config_snapshot_path, "w") as f:
        json.dump(serializable_config, f, indent=4)
    
    print(f"[+] Active hyperparameters locked and snapshotted to: {config_snapshot_path}")
    return run_hash, meta_dir



