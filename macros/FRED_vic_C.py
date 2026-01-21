# =========================
# FRED: batch download series (monthly aligned)
# =========================
import ssl, certifi
import pandas as pd
from fredapi import Fred

def _ssl_context():
    return ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = _ssl_context

fred = Fred(api_key="cacba652bcd29503058a140e2dbc26f7")

START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"

# 你 Deliverable B/C 最常用的一组“流动性/金融压力/信用条件” proxies
FRED_LIQUIDITY_SERIES = {
    # Financial conditions / stress
    "STLFSI2": "stl_fsi",
    "NFCI": "nfci",

    # Spreads / credit stress
    "TEDRATE": "ted_spread",
    "BAA10Y": "baa_10y_spread",
    "BAMLH0A0HYM2": "hy_spread",

    # Policy / money / facilities
    "RRPONTSYD": "rrp",
    "IORB": "iorb",
    "FEDFUNDS": "fed_funds",
    "GS10": "ust_10y",
    "GS2": "ust_2y",
    "T10Y3M": "yc_10y_3m",

    # Credit availability / balance sheet proxies
    "BUSLOANS": "business_loans",
    "REVOLSL": "credit_card_bal",
    "DRCCLACBS": "cc_delinquency",

    # Growth / cycle controls
    "WEI": "wei",
    "INDPRO": "indpro",
    "UNRATE": "unrate",
}

def fred_get_monthly(series_id: str, start: str, end: str) -> pd.Series:
    s = fred.get_series(series_id, start, end)
    s.index = pd.to_datetime(s.index)
    # 统一成月频：月末取最后一个值（与你 A 的逻辑一致）
    s = s.resample("ME").last()
    return s

df_fred = pd.DataFrame()
for sid, name in FRED_LIQUIDITY_SERIES.items():
    try:
        df_fred[name] = fred_get_monthly(sid, START_DATE, END_DATE)
    except Exception as e:
        print(f"[SKIP] {sid} -> {name}: {e}")

print("[OK] FRED monthly shape:", df_fred.shape)
df_fred.to_csv("fred_liquidity_proxies_monthly.csv", index=True)
print("[OK] Saved: fred_liquidity_proxies_monthly.csv")


# =========================
# yfinance: price & returns
# =========================
import yfinance as yf
import pandas as pd

TICKERS = ["UBER", "DASH"]
START = "2019-01-01"
END   = "2025-12-31"

def get_price_panel(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=str.lower)
    df["ret_d"] = df["close"].pct_change()
    # 月频版本（月底）
    m = df.resample("ME").last()
    m["ret_m"] = m["close"].pct_change()
    m["ticker"] = ticker
    return m[["ticker","close","volume","ret_m"]]

price_panel = pd.concat([get_price_panel(t, START, END) for t in TICKERS], axis=0)
price_panel.to_csv("yfinance_prices_monthly.csv", index=True)
print("[OK] Saved: yfinance_prices_monthly.csv")


# =========================
# yfinance: financial statements (annual & quarterly)
# =========================
import yfinance as yf
import pandas as pd

TICKERS = ["UBER", "DASH"]

def get_financials(ticker: str):
    tk = yf.Ticker(ticker)

    # 注意：yfinance 的表是“列=期间”，需要转置成“行=期间”
    bs_a = tk.balance_sheet.T        # annual balance sheet
    cf_a = tk.cashflow.T             # annual cashflow
    is_a = tk.financials.T           # annual income statement

    bs_q = tk.quarterly_balance_sheet.T
    cf_q = tk.quarterly_cashflow.T
    is_q = tk.quarterly_financials.T

    # 统一列名方便 merge
    for df in [bs_a, cf_a, is_a, bs_q, cf_q, is_q]:
        df.index = pd.to_datetime(df.index)

    return (bs_a, cf_a, is_a, bs_q, cf_q, is_q)

# 你 Deliverable C 常用字段（yfinance 有时字段名会不同，所以用“候选字段列表”）
CASH_FIELDS = ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"]
DEBT_FIELDS = ["Total Debt", "Long Term Debt", "Short Long Term Debt", "LongTermDebt"]
OCF_FIELDS  = ["Total Cash From Operating Activities", "Operating Cash Flow"]
CAPEX_FIELDS= ["Capital Expenditures"]
FCF_FIELDS  = ["Free Cash Flow"]  # 有时不存在，就自己算 OCF - Capex

def pick_first_existing(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

rows = []
for t in TICKERS:
    bs_a, cf_a, is_a, bs_q, cf_q, is_q = get_financials(t)

    # annual
    cash_a = pick_first_existing(bs_a, CASH_FIELDS)
    debt_a = pick_first_existing(bs_a, DEBT_FIELDS)
    ocf_a  = pick_first_existing(cf_a, OCF_FIELDS)
    capex_a= pick_first_existing(cf_a, CAPEX_FIELDS)
    fcf_a  = pick_first_existing(cf_a, FCF_FIELDS)

    if fcf_a is None and (ocf_a is not None) and (capex_a is not None):
        fcf_a = ocf_a - capex_a

    # revenue / net income
    rev_a = pick_first_existing(is_a, ["Total Revenue", "Revenue"])
    ni_a  = pick_first_existing(is_a, ["Net Income", "NetIncome"])

    # assemble annual rows
    idx = sorted(set(bs_a.index) | set(cf_a.index) | set(is_a.index))
    for dt in idx:
        rows.append({
            "ticker": t,
            "freq": "A",
            "date": dt,
            "cash": float(cash_a.loc[dt]) if cash_a is not None and dt in cash_a.index else None,
            "total_debt": float(debt_a.loc[dt]) if debt_a is not None and dt in debt_a.index else None,
            "ocf": float(ocf_a.loc[dt]) if ocf_a is not None and dt in ocf_a.index else None,
            "capex": float(capex_a.loc[dt]) if capex_a is not None and dt in capex_a.index else None,
            "fcf": float(fcf_a.loc[dt]) if fcf_a is not None and dt in fcf_a.index else None,
            "revenue": float(rev_a.loc[dt]) if rev_a is not None and dt in rev_a.index else None,
            "net_income": float(ni_a.loc[dt]) if ni_a is not None and dt in ni_a.index else None,
        })

    # quarterly（同理）
    cash_q = pick_first_existing(bs_q, CASH_FIELDS)
    debt_q = pick_first_existing(bs_q, DEBT_FIELDS)
    ocf_q  = pick_first_existing(cf_q, OCF_FIELDS)
    capex_q= pick_first_existing(cf_q, CAPEX_FIELDS)
    fcf_q  = pick_first_existing(cf_q, FCF_FIELDS)
    if fcf_q is None and (ocf_q is not None) and (capex_q is not None):
        fcf_q = ocf_q - capex_q
    rev_q = pick_first_existing(is_q, ["Total Revenue", "Revenue"])
    ni_q  = pick_first_existing(is_q, ["Net Income", "NetIncome"])

    idxq = sorted(set(bs_q.index) | set(cf_q.index) | set(is_q.index))
    for dt in idxq:
        rows.append({
            "ticker": t,
            "freq": "Q",
            "date": dt,
            "cash": float(cash_q.loc[dt]) if cash_q is not None and dt in cash_q.index else None,
            "total_debt": float(debt_q.loc[dt]) if debt_q is not None and dt in debt_q.index else None,
            "ocf": float(ocf_q.loc[dt]) if ocf_q is not None and dt in ocf_q.index else None,
            "capex": float(capex_q.loc[dt]) if capex_q is not None and dt in capex_q.index else None,
            "fcf": float(fcf_q.loc[dt]) if fcf_q is not None and dt in fcf_q.index else None,
            "revenue": float(rev_q.loc[dt]) if rev_q is not None and dt in rev_q.index else None,
            "net_income": float(ni_q.loc[dt]) if ni_q is not None and dt in ni_q.index else None,
        })

fin_panel = pd.DataFrame(rows).sort_values(["ticker","freq","date"])
fin_panel.to_csv("yfinance_financials_panel.csv", index=False)
print("[OK] Saved: yfinance_financials_panel.csv")

print("\n[NOTE] yfinance 字段名可能因公司/版本不同而缺失。若 cash/ocf/capex 全是 None，说明 yfinance 返回缺字段，建议改用 10-K/XBRL。")

# =========================
# yfinance: company snapshot metrics
# =========================
import yfinance as yf
import pandas as pd

TICKERS = ["UBER", "DASH"]
snap = []

for t in TICKERS:
    info = yf.Ticker(t).info  # 注意：可能慢/字段不全
    snap.append({
        "ticker": t,
        "marketCap": info.get("marketCap"),
        "enterpriseValue": info.get("enterpriseValue"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
        "beta": info.get("beta"),
        "sharesOutstanding": info.get("sharesOutstanding"),
        "totalCash": info.get("totalCash"),
        "totalDebt": info.get("totalDebt"),
    })

pd.DataFrame(snap).to_csv("yfinance_snapshot.csv", index=False)
print("[OK] Saved: yfinance_snapshot.csv")

import pandas as pd

fin = pd.read_csv("yfinance_financials_panel.csv")
print(fin.head())

# 看看 UBER / DASH 的 cash、ocf、capex 是否大量缺失
chk = (fin.groupby(["ticker","freq"])[["cash","ocf","capex","fcf","total_debt","revenue"]]
         .apply(lambda x: x.isna().mean())
      )
print(chk)

# 如果 cash/ocf/capex 这些缺失率 > 0.5，说明 yfinance 对这家公司字段不稳定

import pandas as pd
import numpy as np

fin = pd.read_csv("yfinance_financials_panel.csv")
# 只用 annual 先做最稳
finA = fin[fin["freq"]=="A"].copy()

# burn 用 FCF；如果 FCF 为正，说明不烧钱，runway 设为 inf
finA["burn"] = np.where(finA["fcf"].notna(), -finA["fcf"], np.nan)  # burn>0 表示烧钱
finA["runway_years"] = np.where(finA["burn"]>0, finA["cash"]/finA["burn"], np.inf)
finA["runway_months"] = finA["runway_years"] * 12

# 一个“liquidity wall date”定义：runway_months < 6 视为危险
finA["wall_flag_6m"] = (finA["runway_months"] < 6).astype(int)
finA["wall_flag_12m"] = (finA["runway_months"] < 12).astype(int)

out = finA[["ticker","date","cash","fcf","burn","runway_months","wall_flag_6m","wall_flag_12m","revenue","net_income","total_debt"]]
out.to_csv("company_liquidity_wall_annual.csv", index=False)
print("[OK] Saved: company_liquidity_wall_annual.csv")
print(out)
