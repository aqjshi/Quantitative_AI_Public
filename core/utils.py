import os 
import random
import torch 
import numpy as np
import pandas as pd 
import gc
from datetime import datetime

def format_val(v) -> str:
    if pd.isna(v) or v is None: return "N/A"
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1_000_000: return f"{sign}{abs_v/1_000_000:.3f}m"
    elif abs_v >= 1_000: return f"{sign}{abs_v/1_000:.3f}k"
    else: return f"{sign}{abs_v:.3f}"
        
def format_date(d) -> str:
    if pd.isna(d) or d is None or isinstance(d, pd._libs.tslibs.nattype.NaTType): return "N/A"
    return pd.to_datetime(d).strftime('%Y-%m-%d')


def to_str(val) -> str:
    if val is None:
        return 'None'
    # CRITICAL: Neutralize newlines and tabs so they don't break the COPY frame
    return str(val).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def to_int(val) -> str:
    if val is None or str(val).strip() == '':
        return 'None'
    try:
        # Extrapolate floating points (like "8.0") cleanly into base integers
        return str(int(float(val)))
    except (ValueError, TypeError):
        return 'None'
    

def _to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, (set, tuple)):
        return list(val)
    if val is None:
        return []
    return [str(val)]  # Wrap single string into a list



def extract_series_series_id(item):
    if isinstance(item, dict):
        return item.get("series_id")
    elif isinstance(item, str):
        return item
    return None




def cleanup_memory(*objects):
    """Deletes passed objects/arrays and flushes RAM and CUDA GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def extract_scalar(v):
    # Handle containers (lists, tuples, 1D arrays/tensors)
    if isinstance(v, (list, tuple)):
        v = np.ravel(v)[0] if len(v) > 0 else np.nan
    elif hasattr(v, 'item'):  # PyTorch tensor or NumPy scalar
        try:
            v = v.item()
        except ValueError:  # If multi-element array, squeeze or ravel first
            v = np.ravel(v)[0]

    # Preserve non-numeric strings (e.g., timestamps '2025-10-01 00:00:00')
    if isinstance(v, (str, pd.Timestamp, datetime)):
        return str(v)

    # Convert numeric values cleanly to float
    try:
        return float(v)
    except (ValueError, TypeError):
        return str(v)

            
