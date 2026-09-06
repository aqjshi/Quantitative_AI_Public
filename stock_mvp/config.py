import os
import sys
import json

from datetime import datetime

# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_configuration():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            params = json.load(f)
    else:
        params = {}

    return {
        "ticker_type": params.get("ticker_type"),
        "fetch_start": params.get("fetch_start").split(' ')[0],
    
        "quote_batch_size_days": params.get("quote_batch_size_days"),
        "case_study": params.get("case_study"),
        "reconstruction_heartbeat_freq_months": params.get("reconstruction_heartbeat_freq_months"),
        "multiplier": params.get("multiplier"), 
        'timespan': params.get("timespan"), 
        'seed': params.get("seed"), 
        'string_hash_seed': params.get("string_hash_seed"), 
        
        'num_workers': params.get("num_workers"), 
        'rate_limit_per_sec': params.get("rate_limit_per_sec"), 
        'fundamentals_batch_size': params.get("fundamentals_batch_size") , 
        "maintenance_lookback_months": params.get("maintenance_lookback_months" , 12),
        "skip_phase1":  params.get("skip_phase1", True)
    }
