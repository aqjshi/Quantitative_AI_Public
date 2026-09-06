"""
Confusion matrix for the FRED-MD transformation-code classifier.

Colour encodes ROW SHARE (what fraction of each true code went where), not raw
count: class support runs 1..51, so a count scale is dominated by one cell and
every other row washes out. Cell labels keep the raw counts, so nothing is lost.
Sequential magnitude -> one hue, light->dark. Never a rainbow.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# ---- design tokens (reference palette) --------------------------------------
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)

from macros.math.transform import classify

# ---- data -------------------------------------------------------------------
meta = json.load(open(os.path.join(HERE, "md_meta.json")))
raw = pd.read_csv(os.path.join(REPO, "macros/math/2026-06-MD.csv"))
tcode_row = raw.iloc[0]
data = raw.iloc[1:].copy()
data["sasdate"] = pd.to_datetime(data["sasdate"], format="mixed", errors="coerce")
data = data.dropna(subset=["sasdate"]).set_index("sasdate").sort_index()
sids = [c for c in raw.columns if c != "sasdate"]
truth = {s: int(float(tcode_row[s])) for s in sids if str(tcode_row[s]).strip()}

rows = []
for sid in sids:
    m = meta.get(sid, {})
    if not m.get("ok"):
        continue
    s = pd.to_numeric(data[sid], errors="coerce").astype(float)
    s.index = data.index
    if s.dropna().empty:
        continue
    try:
        _, tf = classify(series=s, freq=m.get("frequency_short"),
                         units_short=m.get("units_short"),
                         series_id=sid, title=m.get("title"))
        rows.append((sid, truth[sid], int(tf["code"])))
    except Exception:
        continue

df = pd.DataFrame(rows, columns=["series_id", "truth", "pred"])
ALL_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = [c for c in ALL_CODES] + sorted((set(df.truth) | set(df.pred)) - set(ALL_CODES))
cm = (pd.crosstab(df.truth, df.pred)
        .reindex(index=labels, columns=labels, fill_value=0).to_numpy())
n = len(df)
acc = (df.truth == df.pred).mean() * 100

support = cm.sum(axis=1)
predicted = cm.sum(axis=0)
row_share = np.divide(cm, support[:, None], out=np.zeros_like(cm, float),
                      where=support[:, None] > 0)
recall = np.divide(np.diag(cm), support, out=np.full(len(labels), np.nan),
                   where=support > 0) * 100
precision = np.divide(np.diag(cm), predicted, out=np.full(len(labels), np.nan),
                      where=predicted > 0) * 100

# ---- render -----------------------------------------------------------------
k = len(labels)
fig, ax = plt.subplots(figsize=(9.0, 7.0), facecolor=SURFACE)
fig.subplots_adjust(left=0.13, right=0.78, top=0.72, bottom=0.19)
ax.set_facecolor(SURFACE)
norm = Normalize(0, 1)
ax.imshow(row_share, cmap=CMAP, norm=norm, aspect="equal")

# headline
fig.text(0.045, 0.945, "FRED-MD transformation-code classifier",
         fontsize=16, color=INK, fontweight="bold", ha="left")
fig.text(0.045, 0.897,
         f"{acc:.1f}% accuracy  ·  {int(round(acc/100*n))}/{n} series  ·  2026-06 vintage",
         fontsize=11.5, color=INK_2, ha="left")
fig.text(0.045, 0.855,
         " ",
         fontsize=9.5, color=MUTED, ha="left")

# cells
for i in range(k):
    for j in range(k):
        v = cm[i, j]
        if v == 0:
            ax.text(j, i, "·", ha="center", va="center", fontsize=11, color=GRIDLINE)
            continue
        dark = row_share[i, j] > 0.55
        ax.text(j, i - 0.08, str(v), ha="center", va="center", fontsize=12.5,
                color=SURFACE if dark else INK,
                fontweight="bold" if i == j else "normal")
        ax.text(j, i + 0.27, f"{row_share[i, j] * 100:.0f}%", ha="center", va="center",
                fontsize=7.5, color=SURFACE if dark else MUTED,
                alpha=0.8 if dark else 1.0)

# diagonal emphasis: a ring, not a second colour channel
for i in range(k):
    ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                           edgecolor=INK, linewidth=1.6, zorder=4))

# 2px surface gap between cells
ax.set_xticks(np.arange(-.5, k, 1), minor=True)
ax.set_yticks(np.arange(-.5, k, 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)

ax.set_xticks(range(k), labels, fontsize=11, color=INK_2)
ax.set_yticks(range(k), labels, fontsize=11, color=INK_2)
ax.set_xlabel("predicted code", fontsize=10.5, color=MUTED, labelpad=34)
ax.set_ylabel("true code", fontsize=10.5, color=MUTED, labelpad=10)
for sp in ax.spines.values():
    sp.set_color(BASELINE)
    sp.set_linewidth(1)
ax.tick_params(length=0)

# right margin: support + recall
ax.text(k + 0.15, -0.95, "n", fontsize=9.5, color=MUTED, ha="center", va="center",
        clip_on=False)
ax.text(k + 1.05, -0.95, "recall", fontsize=9.5, color=MUTED, ha="center", va="center",
        clip_on=False)
for i in range(k):
    ax.text(k + 0.15, i, str(int(support[i])), fontsize=10.5, color=INK_2,
            ha="center", va="center", clip_on=False)
    r = recall[i]
    ax.text(k + 1.05, i, "—" if np.isnan(r) else f"{r:.0f}%", fontsize=10.5,
            color=INK if (not np.isnan(r) and r >= 50) else MUTED,
            ha="center", va="center", clip_on=False)

# bottom margin: precision.
# x in DATA coords, y in AXES-fraction, so this sits cleanly below the tick
# labels instead of colliding with the last matrix row.
xtrans = ax.get_xaxis_transform()
ax.text(-0.75, -0.125, "precision", fontsize=9.5, color=MUTED, ha="right",
        va="center", transform=xtrans, clip_on=False)
for j in range(k):
    p = precision[j]
    ax.text(j, -0.125, "—" if np.isnan(p) else f"{p:.0f}%", fontsize=10.5,
            color=INK if (not np.isnan(p) and p >= 50) else MUTED,
            ha="center", va="center", transform=xtrans, clip_on=False)

# callout for the structural finding
fig.text(0.045, 0.035,
         "",
         fontsize=9, color=INK_2, ha="left", va="bottom")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                    fraction=0.030, pad=0.20)
cbar.set_ticks([0, .25, .5, .75, 1])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
cbar.set_label("share of true code", fontsize=9.5, color=MUTED)
cbar.ax.tick_params(labelsize=9, colors=MUTED, length=0)
cbar.outline.set_visible(False)

out = os.path.join(HERE, "confusion_matrix.png")
fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
print(f"accuracy: {(df.truth == df.pred).sum()}/{n} = {acc:.2f}%")
