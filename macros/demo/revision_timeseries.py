"""
Point-in-time revision count over anchor dates.

For each anchor t, counts revisions that were KNOWABLE at t -- both gates applied:

    observation date  <= t     AND     realtime_start <= t

so the series answers "how many revisions had actually been published by t",
not "how many exist in the table today". A bitemporal row becomes visible at
max(date, realtime_start); an observation date joins the count once its first
row is visible. Revisions = visible rows - visible observation dates.

Covers the TOP N most-revised and BOTTOM N least-revised series from
revision_census.csv. Writes a tidy CSV and a small-multiples PNG.

Usage:
    python macros/validation/revision_timeseries.py
    python macros/validation/revision_timeseries.py --since 2000-01-01 --n 10 --freq MS
    python macros/validation/revision_timeseries.py --min-obs 120
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import text

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from core.db import engine


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SINCE = arg("--since", "2000-01-01")
N = int(arg("--n", "10"))
FREQ = arg("--freq", "MS")
MIN_OBS = int(arg("--min-obs", "120"))   # floor so "least revised" isn't just short series

# ---- design tokens (reference palette) --------------------------------------
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
TOP_HUE, BOT_HUE = "#2a78d6", "#eb6834"     # categorical slots 1 and 2

# ---- pick the two cohorts from the census ----------------------------------
census_path = os.path.join(HERE, "revision_census.csv")
if not os.path.exists(census_path):
    sys.exit("run revision_census.py first -- revision_census.csv not found")
census = pd.read_csv(census_path)

elig = census[census.n_obs >= MIN_OBS].copy()
top = elig.nlargest(N, "n_revisions")
bottom = elig.nsmallest(N, "n_revisions")
cohorts = [("top", top), ("bottom", bottom)]
sids = list(top.series_id) + list(bottom.series_id)

print(f"cohorts drawn from {len(elig)} series with >= {MIN_OBS} observations")
print(f"  TOP {N}    revisions {int(top.n_revisions.min()):,} .. {int(top.n_revisions.max()):,}")
print(f"  BOTTOM {N} revisions {int(bottom.n_revisions.min()):,} .. {int(bottom.n_revisions.max()):,}")

# ---- pull the bitemporal rows for those series ------------------------------
SQL = text("""
    SELECT f.series_id, o.date, o.realtime_start
    FROM fred_observations o
    JOIN fred_series_filtered f USING (series_id_hash)
    WHERE f.series_id = ANY(:ids) AND o.value IS NOT NULL AND o.date >= :since
""")
with engine.connect() as c:
    raw = pd.DataFrame(c.execute(SQL, {"ids": sids, "since": SINCE}).fetchall(),
                       columns=["series_id", "date", "realtime_start"])
raw["date"] = pd.to_datetime(raw["date"])
raw["realtime_start"] = pd.to_datetime(raw["realtime_start"])
print(f"pulled {len(raw):,} bitemporal rows for {raw.series_id.nunique()} series")

anchors = pd.date_range(SINCE, pd.Timestamp.today().normalize(), freq=FREQ)
# pandas 2 can hand back datetime64[us] from date_range while the row values
# are datetime64[ns]; pin BOTH sides to ns or every comparison below is a
# 1000x mismatch and searchsorted silently returns 0.
a_i8 = anchors.to_numpy(dtype="datetime64[ns]").astype("int64")

records = []
for sid, g in raw.groupby("series_id", sort=False):
    # a row is knowable only once BOTH its period has passed and it is published
    visible = np.maximum(g["date"].values.astype("datetime64[ns]"),
                         g["realtime_start"].values.astype("datetime64[ns]"))
    vis_i8 = np.sort(visible.astype("int64"))

    # an observation date counts once its FIRST row becomes visible
    first_vis = np.sort((pd.Series(visible, index=g["date"].values)
                           .groupby(level=0).min()
                           .to_numpy(dtype="datetime64[ns]").astype("int64")))

    n_rows = np.searchsorted(vis_i8, a_i8, side="right")
    n_dates = np.searchsorted(first_vis, a_i8, side="right")
    revs = n_rows - n_dates

    records.append(pd.DataFrame({"series_id": sid, "anchor": anchors,
                                 "visible_rows": n_rows,
                                 "visible_obs_dates": n_dates,
                                 "revisions": revs}))

ts = pd.concat(records, ignore_index=True)
ts["cohort"] = np.where(ts.series_id.isin(set(top.series_id)), "top", "bottom")
out_csv = os.path.join(HERE, "revision_timeseries.csv")
ts.to_csv(out_csv, index=False)

# ---- small multiples --------------------------------------------------------
# 20 series is far past any categorical palette, so identity comes from the
# panel title and every line uses one hue per cohort. Per-panel y scale, because
# counts differ by four orders of magnitude between cohorts.
rows_n, cols_n = 4, N // 2 if N % 2 == 0 else N
fig, axes = plt.subplots(rows_n, cols_n, figsize=(3.0 * cols_n, 2.35 * rows_n),
                         facecolor=SURFACE, sharex=True)
axes = np.atleast_2d(axes)

order = [(s, "top") for s in top.series_id] + [(s, "bottom") for s in bottom.series_id]
for k, (sid, cohort) in enumerate(order):
    ax = axes[k // cols_n, k % cols_n]
    sub = ts[ts.series_id == sid]
    hue = TOP_HUE if cohort == "top" else BOT_HUE
    ax.plot(sub.anchor, sub.revisions, lw=2, color=hue, solid_capstyle="round")
    ax.set_facecolor(SURFACE)
    ax.set_title(f"{sid}", fontsize=9.5, color=INK, loc="left", pad=4)
    final = int(sub.revisions.iloc[-1])
    ax.text(0.97, 0.06, f"{final:,}", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=INK_2)
    ax.grid(axis="y", color=GRIDLINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)

for k in range(len(order), rows_n * cols_n):
    axes[k // cols_n, k % cols_n].set_visible(False)

fig.suptitle("Published revisions over time, by anchor date",
             x=0.012, y=0.985, ha="left", fontsize=15, color=INK, fontweight="bold")
fig.text(0.012, 0.947,
         f"Counts only revisions knowable at each anchor: observation date <= t AND realtime_start <= t."
         f"  Rows 1-2 = {N} most revised, rows 3-4 = {N} least revised (>= {MIN_OBS} obs).",
         ha="left", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out_png = os.path.join(HERE, "revision_timeseries.png")
fig.savefig(out_png, dpi=200, facecolor=SURFACE, bbox_inches="tight")

print(f"\nanchors: {len(anchors)} ({anchors[0].date()} .. {anchors[-1].date()}, freq={FREQ})")
print(f"-> {out_csv}")
print(f"-> {out_png}")
