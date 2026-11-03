import os
from typing import Dict, List, Optional, Tuple, Any
import sys 
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import timedelta,  datetime
import numpy as np
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from statsmodels.tsa.stattools import adfuller


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macros.math.transform import reverse_transxf_fred, transxf_fred
import os, sys , json, re, csv

ADF_EDGE_PATH = os.path.join(os.path.dirname(__file__), "exogeneity", "adf_edge.json")
with open(ADF_EDGE_PATH, "r") as f:
    # print("found the units json")
    edge_cases = json.load(f)
    ADF_EDGE_MAP = edge_cases

RE_HAS_DOLLAR_DENOM = re.compile(r"(?:bil|mil|thous|billion|million|thousand).*(?:\$|dollar)|(?:\$|dollar).*(?:bil|mil|thous|billion|million|thousand)",
        re.IGNORECASE
    )



def adf_is_stationary(s: pd.Series, alpha: float = 0.05) -> bool:
    """Helper: Runs Augmented Dickey-Fuller test.

    Returns True if p-value <= alpha (reject H0 = Stationary).
    """
    clean = s.dropna()
    if len(clean) < 12 or clean.nunique() <= 1:
        return True  # Default to level if insufficient variance or observations

    try:
        res = adfuller(clean, autolag="AIC")
        return bool(res[1] <= alpha)
    except Exception:
        return True

    
def classify(
    series: pd.Series,
    freq: Optional[str] = None,
    units_short: Optional[str] = None,
    series_id: Optional[str] = None,
    title: Optional[str] = None,
    notes: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Classifies a time series and applies standard FRED-MD transformations.

    Captures initial values (init_val1, init_val2) required by
    reverse_transxf_fred.
    """
    clean = series.dropna().astype(float)
    code9_trillion_denom = 1.0
    # Calculate initial values for downstream exact inversion via reverse_transxf_fred
    first_valid_date = series.first_valid_index()
    init_val1 = (
        float(series.loc[first_valid_date])
        if first_valid_date is not None
        else None
    )

    valid_series = series.dropna()
    init_val2 = (
        float(valid_series.iloc[1]) if len(valid_series) >= 2 else init_val1
    )

    if clean.empty or clean.nunique() <= 1:
        stat_s = transxf_fred(series, 1)
        return stat_s, {
            "code": 1,
            "base_level": init_val1 or 0.0,
            "init_val1": init_val1,
            "init_val2": init_val2,
            "mean": 0.0,
            "std": 1.0,
        }

    units_str = str(units_short).lower() 
    sid_str = str(series_id)
    title_str = str(title).lower() 
    notes_str = str(notes).lower() 
            
    # GUARD 1: Rates and percentages should NEVER be log-transformed
    is_rate_or_pct = any(
        k in units_str for k in ["%", "percent", "rate", "ratio"]
    )

    # GUARD 2: Check positivity for natural logs
    can_log = (clean.min() > 1e-6) and (not is_rate_or_pct)

    # GUARD 3: Allow Code 6 (2nd Diff Log) only for explicit price indices
    is_price_index = any(
        k in units_str 
        for k in ["cpi", "ppi", "price index", "deflator"]
    )

    # ------------------------------------------------------------------
    # Step -1: Edge case
    # ------------------------------------------------------------------
    if sid_str in ADF_EDGE_MAP:
        chosen_code = ADF_EDGE_MAP[sid_str]

    elif ("net" in title_str)  and any(k in units_str for k in ["%", "percent", "percentage"]):
        # print(f"found 8 {title_str}")
        chosen_code = 8
    elif ("index" in units_str) or bool(re.search(r"\b(vix|vx)\b", title_str)) or "volatility index" in title_str:
        if can_log:
            chosen_code = 5
        else:
            # Fallback if a volatility index ever hits zero/negative (e.g., spreads)
            chosen_code = 2
    elif ("100" in units_str or "cents" in units_str or "average price" in title_str):
        chosen_code = 5

    elif ("$ per" in units_str and "price" in title_str):
            chosen_code = 5


    elif ("number of commercial paper issues" in title_str):
        if can_log:
            chosen_code = 5  # Converts 16,250 diff into ~0.05 log growth
        else:
            code9_trillion_denom = 1e3  # Scale down if linear
            chosen_code = 9
    elif ("covered employment" in title_str) or ("initial claims" in title_str) or ("covered unemployment" in title_str) or ("continued claims" in title_str) :
        chosen_code = 5  # Converts 16,250 diff into ~0.05 log growth

    elif ("number of respondents") in units_str:
        chosen_code = 2


    elif (
         bool(re.search(r"\bdeficit\b", title_str)) or  bool(re.search(r"\bsurplus\b", title_str)) or bool(re.search(r"\bchange\b", title_str)) or bool(re.search(r"\bflow\b", title_str)) or bool(re.search(r"\bnet\b", title_str)  or bool(re.search(r"subtrac", notes_str))) 
         or any(k in title_str
         for k in ["income (loss)","total value of issues", "retained earnings", "unused loan commitments", "underwriting commitments", "quarterly banking profile",  "provision"]
         )
    ) and bool(RE_HAS_DOLLAR_DENOM.search(units_str)):
        if "bil." in units_str:
            code9_trillion_denom = 1e3
        elif "mil." in units_str:
            code9_trillion_denom = 1e6
        elif "thous." in units_str:
            code9_trillion_denom = 1e9
    
        chosen_code = 9
        # print("found code 9:" , sid_str, title_str, chosen_code, code9_trillion_denom, "arg1", ("flow" in title_str or "net" in  title_str), "arg2",  any(u in units_str for u in ["bil. of $", "mil. of $", "thous. of $", "$, annual rate"]))

  
    elif not ("flow" in title_str or "flow" in units_str or "net" in title_str) and ("$, annual rate" in units_str or "bil. of $" in units_str or "mil. of $" in units_str or "thous. of $" in units_str):
        chosen_code = 5

    elif ("percent of" in title_str):
        chosen_code = 5  # Converts 16,250 diff into ~0.05 log growth
    
    elif ("u.s. dollar spot exchange rate" in title_str):
        chosen_code = 5  # Converts 16,250 diff into ~0.05 log growth
    

    elif any(
        k in units_str 
        for k in ["months", "weeks", "days", "years", "turnover"]
    ) and not "net" in title_str:
        chosen_code = 2
    # Rates, Percentages, and Spreads
    elif "rate" in units_str or "%" in units_str or "percent" in units_str:
        # 1. Net Surveys / Balances -> Code 8
        if "net" in units_str  or "minus" in units_str or "spread" in title_str:
            chosen_code = 8
        # 2. Standard Rates (FEDFUNDS, CPFFM, UNRATE) -> Code 2
        else:
            chosen_code = 2

    # ------------------------------------------------------------------
    # Step 1: Test Raw Level (Code 1)
    # ------------------------------------------------------------------
    elif adf_is_stationary(clean, alpha=alpha):
        if any(k in units_str
                for k in [
                    "dollar",
                    "bil.",
                    "mil.",
                    "thous.",
                    "credit",
                    "amount",
                    "person",
                ]
            ):
            chosen_code = 5 if can_log else 2


        elif "mortgage" in title_str:
            # print(f"found 2 {title_str}")
            chosen_code = 2
        else:
            chosen_code = 1

    # ------------------------------------------------------------------
    # Step 2: Test Log Differences (for positive aggregates/counts)
    # ------------------------------------------------------------------
    elif can_log:
        log_diff1 = np.log(clean).diff().dropna()
        if adf_is_stationary(log_diff1, alpha=alpha):
            chosen_code = 5
        elif is_price_index:
            log_diff2 = log_diff1.diff().dropna()
            chosen_code = 6 if adf_is_stationary(log_diff2, alpha=alpha) else 5
        else:
            chosen_code = 5

    # ------------------------------------------------------------------
    # Step 3: Test Linear Differences (for rates, percentages, negative values)
    # ------------------------------------------------------------------
    else:
        diff1 = clean.diff().dropna()
        if adf_is_stationary(diff1, alpha=alpha):
            chosen_code = 2
        else:
            diff2 = diff1.diff().dropna()
            chosen_code = 3 if adf_is_stationary(diff2, alpha=alpha) else 2

    # Apply forward transformation
    stat_s = transxf_fred(series, chosen_code, code9_trillion_denom)

    meta = {
        "code": chosen_code,
        "code9_trillion_denom":code9_trillion_denom,
        "base_level": init_val1 or 0.0,
        "init_val1": init_val1,
        "init_val2": init_val2,
        "mean": float(stat_s.mean()) if not stat_s.dropna().empty else 0.0,
        "std": float(stat_s.std()) if not stat_s.dropna().empty else 1.0,
    }

    return stat_s, meta



def classify_ADF(
    obs_context: List[Dict],
    train_start: pd.Timestamp,
    current_time: pd.Timestamp,
    resample_frequency: str = "1ME",
    metadata_lookup: Dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:

    df_raw = pd.DataFrame(obs_context)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["realtime_start"] = pd.to_datetime(
        df_raw.get("realtime_start", df_raw["date"])
    )
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_raw = df_raw.dropna(subset=["value"])

    train_start = pd.to_datetime(train_start)
    current_time = pd.to_datetime(current_time)

    df_raw = df_raw[
        (df_raw["date"] >= train_start) & (df_raw["date"] <= current_time)
    ]

    # BITEMPORAL DEDUP
    df_raw = df_raw.sort_values(
        by=["series_id", "date", "realtime_start"]
    ).drop_duplicates(subset=["series_id", "date"], keep="last")

    if resample_frequency in ["1MS", "MS", "1ME", "ME", "M"]:
        df_raw["resampled_date"] = (
            df_raw["date"].dt.to_period("M").dt.to_timestamp(how="end")
        )
    else:
        df_raw["resampled_date"] = df_raw["date"].dt.floor(resample_frequency)




    series_frames = []
    
    for sid, group in df_raw.groupby("series_id"):
        meta = metadata_lookup.get(sid)
        resampled = group.set_index("date")["value"].resample(resample_frequency).last()    
        resampled.name = sid
        series_frames.append(resampled)



    df_pivot_levels = pd.concat(series_frames, axis=1)

    df_pivot_stat = pd.DataFrame(index=df_pivot_levels.index)
    df_pivot_reversed = pd.DataFrame(index=df_pivot_levels.index)
    transform_metadata = {}

 
    for col in df_pivot_levels.columns:
        meta = metadata_lookup.get(col) or {}
        freq = meta.get("frequency_short")
        u_short = meta.get("units_short")
        raw_levels = df_pivot_levels[col]

        # 1. Drop off-month NaNs to isolate native observation dates
        native_levels = raw_levels.dropna()

    
        # 2. Run Forward Transform at NATIVE frequency (Q/A)
        native_stat_s, tf_meta = classify(
            series=native_levels,
            freq=freq,
            units_short=u_short,
            series_id=meta.get("series_id", col),
            title=meta.get("title"),
            notes=meta.get("notes")
        )

        # 3. Reindex back onto the full monthly grid (off-months naturally remain NaNs!)
        stat_s = native_stat_s.reindex(raw_levels.index)

        code = tf_meta.get("code", 1)
        init1 = tf_meta.get("init_val1")
        init2 = tf_meta.get("init_val2")
        code9_trillion_denom  = tf_meta.get("code9_trillion_denom", 1)
        # 2. Reverse Transform
        native_reversed_s = reverse_transxf_fred(
            y=native_stat_s, tcode=code, init_val1=init1, init_val2=init2, code9_trillion_denom=code9_trillion_denom
        )
        reversed_s = native_reversed_s.reindex(raw_levels.index)

        df_pivot_stat[col] = stat_s
        df_pivot_reversed[col] = reversed_s


        

        transform_metadata[str(col)] = {
            "frequency_short": freq,
            "code": code,
            "base_level": tf_meta.get("base_level", 0.0),
            "init_val1": init1,
            "init_val2": init2,
            "mean": tf_meta.get("mean", 0.0),
            "std": tf_meta.get("std", 1.0),
        }


    return df_pivot_stat, transform_metadata

