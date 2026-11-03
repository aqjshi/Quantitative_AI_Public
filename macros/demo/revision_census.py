"""
Revision census for every series in fred_series_filtered.

For each series, over observations dated in [--since, today], counts how heavily
the series gets revised after first publication:

    n_obs            distinct observation dates
    n_vintage_rows   total bitemporal rows (one per publication of each date)
    n_revisions      n_vintage_rows - n_obs, i.e. republications beyond first print
    n_revised_obs    observation dates published more than once
    pct_revised      n_revised_obs / n_obs
    rev_mae          mean |final value - first published value|
    rev_max          max  |final value - first published value|
    rev_mae_pct      rev_mae as a share of the mean |level|, so series of
                     different magnitude are comparable
    sign_flips       dates where first print and final disagree on sign

Why it matters: the nowcast is scored against the oracle at as_of 9999-12-31,
the FULLY REVISED value, which for a heavily-revised series was not published
until long after the target date. That part of the error is revision risk no
nowcast could have avoided. Read rev_mae against a series' reported MAE:

    rev_mae ~ 0        -> the score is real forecast signal
    rev_mae ~ error    -> the score is mostly revision noise
    rev_mae >  error   -> the row carries no forecastable signal at all

Usage:
    python macros/validation/revision_census.py                # since 2000-01-01
    python macros/validation/revision_census.py --since 2010-01-01
    python macros/validation/revision_census.py --top 40
"""
import os, sys, time
import numpy as np
import pandas as pd
from sqlalchemy import text

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from core.db import engine


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SINCE = arg("--since", "2000-01-01")
TOP = int(arg("--top", "25"))

SQL = text("""
WITH scoped AS (
    SELECT f.series_id, o.date, o.value, o.realtime_start
    FROM fred_observations o
    JOIN fred_series_filtered f USING (series_id_hash)
    WHERE o.value IS NOT NULL
      AND o.date >= :since
      AND o.date <= CURRENT_DATE
),
per_obs AS (
    SELECT series_id,
           date,
           count(*) AS n_vintages,
           (array_agg(value ORDER BY realtime_start ASC))[1]  AS first_print,
           (array_agg(value ORDER BY realtime_start DESC))[1] AS final_val
    FROM scoped
    GROUP BY series_id, date
)
SELECT p.series_id,
       f.title,
       f.frequency_short,
       f.units_short,
       count(*)                                              AS n_obs,
       sum(p.n_vintages)                                     AS n_vintage_rows,
       sum(p.n_vintages) - count(*)                          AS n_revisions,
       sum(CASE WHEN p.n_vintages > 1 THEN 1 ELSE 0 END)     AS n_revised_obs,
       avg(abs(p.final_val - p.first_print))                 AS rev_mae,
       max(abs(p.final_val - p.first_print))                 AS rev_max,
       avg(abs(p.final_val))                                 AS mean_level,
       sum(CASE WHEN sign(p.final_val) <> sign(p.first_print)
                THEN 1 ELSE 0 END)                           AS sign_flips,
       min(p.date)                                           AS first_obs,
       max(p.date)                                           AS last_obs
FROM per_obs p
JOIN fred_series_filtered f ON f.series_id = p.series_id
GROUP BY p.series_id, f.title, f.frequency_short, f.units_short
ORDER BY n_revisions DESC
""")

t0 = time.time()
with engine.connect() as c:
    df = pd.DataFrame(
        c.execute(SQL, {"since": SINCE}).fetchall(),
        columns=["series_id", "title", "frequency_short", "units_short",
                 "n_obs", "n_vintage_rows", "n_revisions", "n_revised_obs",
                 "rev_mae", "rev_max", "mean_level", "sign_flips",
                 "first_obs", "last_obs"])
elapsed = time.time() - t0

# Postgres returns count/sum as Decimal -> object dtype; cast everything numeric
num = ["n_obs", "n_vintage_rows", "n_revisions", "n_revised_obs",
       "rev_mae", "rev_max", "mean_level", "sign_flips"]
df[num] = df[num].apply(pd.to_numeric, errors="coerce")
df["pct_revised"] = (df.n_revised_obs / df.n_obs * 100).round(2)
df["rev_mae_pct"] = (df.rev_mae / df.mean_level.replace(0, np.nan) * 100).round(4)
df["vintages_per_obs"] = (df.n_vintage_rows / df.n_obs).round(3)

out = os.path.join(HERE, "revision_census.csv")
df.to_csv(out, index=False)

total_filtered = pd.read_sql("SELECT count(*) AS n FROM fred_series_filtered", engine)["n"][0]

print("=" * 96)
print(f"REVISION CENSUS  --  fred_series_filtered, observations dated {SINCE} .. today")
print("=" * 96)
print(f"series in fred_series_filtered : {total_filtered}")
print(f"series with observations       : {len(df)}")
print(f"query time                     : {elapsed:.1f}s")
print()
print(f"total observation dates        : {int(df.n_obs.sum()):,}")
print(f"total bitemporal rows          : {int(df.n_vintage_rows.sum()):,}")
print(f"total revisions                : {int(df.n_revisions.sum()):,}")
print()
never = int((df.n_revisions == 0).sum())
print(f"series NEVER revised           : {never}  ({never / len(df) * 100:.1f}%)")
print(f"series with any revision       : {len(df) - never}  ({(len(df) - never) / len(df) * 100:.1f}%)")
print(f"series with a sign flip        : {int((df.sign_flips > 0).sum())}")
print()
print("distribution of revisions per series:")
for q in (0.50, 0.75, 0.90, 0.95, 0.99, 1.00):
    print(f"  p{int(q * 100):<3} {df.n_revisions.quantile(q):>12,.0f}")

print()
print("=" * 96)
print(f"TOP {TOP} MOST-REVISED SERIES (by revision count)")
print("=" * 96)
print(f"{'series_id':<18}{'freq':>5}{'n_obs':>7}{'revs':>8}{'v/obs':>7}"
      f"{'%rev':>7}{'rev_mae':>13}{'rev%lvl':>9}{'flips':>6}  title")
for _, r in df.head(TOP).iterrows():
    rp = "n/a" if pd.isna(r.rev_mae_pct) else f"{r.rev_mae_pct:.2f}"
    print(f"{r.series_id[:17]:<18}{str(r.frequency_short):>5}{r.n_obs:>7}{r.n_revisions:>8}"
          f"{r.vintages_per_obs:>7}{r.pct_revised:>7}{r.rev_mae:>13.4g}{rp:>9}"
          f"{r.sign_flips:>6}  {str(r.title)[:34]}")

print()
print("=" * 96)
print(f"TOP {TOP} BY REVISION SIZE RELATIVE TO LEVEL (rev_mae as % of mean level)")
print("=" * 96)
big = df[(df.n_obs >= 12) & df.rev_mae_pct.notna()].sort_values("rev_mae_pct", ascending=False)
print(f"{'series_id':<18}{'freq':>5}{'n_obs':>7}{'revs':>8}{'rev_mae':>13}{'rev%lvl':>10}{'flips':>6}  title")
for _, r in big.head(TOP).iterrows():
    print(f"{r.series_id[:17]:<18}{str(r.frequency_short):>5}{r.n_obs:>7}{r.n_revisions:>8}"
          f"{r.rev_mae:>13.4g}{r.rev_mae_pct:>10.2f}{r.sign_flips:>6}  {str(r.title)[:34]}")

print(f"\n-> {out}")
