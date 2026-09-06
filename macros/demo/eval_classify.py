"""
Validation for macros.math.transform.classify().

PART 1 - accuracy against the answer key (offline)
    Scores classify() against the transformation codes published by McCracken &
    Ng in the FRED-MD 2026-06 vintage (macros/math/2026-06-MD.csv, row 2).

      feature vector : series_id, title, units_short, (notes), frequency_short
      target         : the tcode on row 2 of the csv
      series data    : the csv's own raw level columns -- the exact vintage the
                       codes were assigned to, so the ADF branch sees what
                       McCracken & Ng saw.

PART 2 - codes 8 and 9 (needs the database)
    Codes 8 and 9 are this repo's extensions to the 1-7 scheme, so FRED-MD never
    assigns them and PART 1 cannot score them. They are checked the only way
    available: run the real cascade over the live series universe and inspect
    which series land there and whether the transform round-trips.

PART 1 needs no database and no FRED key (md_meta.json is committed). PART 2 is
skipped with a notice if the database is unreachable, or with --no-db.

    python macros/validation/eval_classify.py
    python macros/validation/eval_classify.py --no-db
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from macros.math.transform import classify, reverse_transxf_fred

RUN_DB = "--no-db" not in sys.argv

# Every code classify() can emit, not just the observed ones. 8 and 9 are this
# repo's extensions to the McCracken & Ng 1-7 scheme, so they never appear in
# the answer key -- they are shown to document the output space.
ALL_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ============================================================================ #
# PART 1 -- accuracy against the FRED-MD answer key
# ============================================================================ #
CSV = os.path.join(REPO, "macros/math/2026-06-MD.csv")
meta = json.load(open(os.path.join(HERE, "md_meta.json")))

raw = pd.read_csv(CSV)
tcode_row = raw.iloc[0]
data = raw.iloc[1:].copy()
data["sasdate"] = pd.to_datetime(data["sasdate"], format="mixed", errors="coerce")
data = data.dropna(subset=["sasdate"]).set_index("sasdate").sort_index()

series_ids = [c for c in raw.columns if c != "sasdate"]
truth = {sid: int(float(tcode_row[sid])) for sid in series_ids if str(tcode_row[sid]).strip()}

rows = []
for sid in series_ids:
    m = meta.get(sid, {})
    if not m.get("ok"):
        rows.append({"series_id": sid, "truth": truth.get(sid), "pred": None,
                     "status": "no_metadata", "title": "", "units_short": ""})
        continue

    s = pd.to_numeric(data[sid], errors="coerce").astype(float)
    s.index = data.index
    if s.dropna().empty:
        rows.append({"series_id": sid, "truth": truth.get(sid), "pred": None,
                     "status": "no_data", "title": m.get("title", ""), "units_short": ""})
        continue

    try:
        _, tf_meta = classify(series=s, freq=m.get("frequency_short"),
                              units_short=m.get("units_short"),
                              series_id=sid, title=m.get("title"))
        pred, status = int(tf_meta.get("code")), "ok"
    except Exception as e:
        pred, status = None, f"error: {type(e).__name__}: {str(e)[:60]}"

    rows.append({"series_id": sid, "truth": truth.get(sid), "pred": pred,
                 "status": status, "title": (m.get("title") or "")[:58],
                 "units_short": m.get("units_short") or ""})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "classify_eval.csv"), index=False)

scored = df[(df.pred.notna()) & (df.truth.notna())].copy()
scored["pred"] = scored["pred"].astype(int)
scored["truth"] = scored["truth"].astype(int)
scored["hit"] = scored.pred == scored.truth

print("=" * 78)
print("PART 1 -- ACCURACY vs THE FRED-MD ANSWER KEY (2026-06 vintage)")
print("=" * 78)
print(f"series in csv            : {len(df)}")
print(f"scored                   : {len(scored)}")
for st, n in df[df.pred.isna()].status.value_counts().items():
    print(f"  excluded ({st}): {n}")
acc = scored.hit.mean() * 100
print(f"\nACCURACY                 : {scored.hit.sum()}/{len(scored)} = {acc:.2f}%")
print(f"ONE-TO-ONE               : {'YES' if scored.hit.all() else 'NO'}")

labels = list(ALL_CODES) + sorted((set(scored.truth) | set(scored.pred)) - set(ALL_CODES))
cm = pd.crosstab(scored.truth, scored.pred).reindex(index=labels, columns=labels, fill_value=0)
print("\nCONFUSION MATRIX   (rows = TRUE tcode, cols = PREDICTED)")
print("      " + "".join(f"{c:>6}" for c in labels) + "   | total")
for r in labels:
    print(f"  {r:<3} " + "".join(f"{int(cm.loc[r, c]):>6}" for c in labels)
          + f"   | {int(cm.loc[r].sum()):>5}")
print("  tot " + "".join(f"{int(cm[c].sum()):>6}" for c in labels))

print("\nPER-CODE PRECISION / RECALL")
print(f"  {'code':<6}{'support':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}")
for c in labels:
    tp = int(((scored.truth == c) & (scored.pred == c)).sum())
    fp = int(((scored.truth != c) & (scored.pred == c)).sum())
    fn = int(((scored.truth == c) & (scored.pred != c)).sum())
    sup = int((scored.truth == c).sum())
    prec = tp / (tp + fp) * 100 if tp + fp else float("nan")
    rec = tp / (tp + fn) * 100 if tp + fn else float("nan")
    print(f"  {c:<6}{sup:>8}{tp:>5}{fp:>5}{fn:>5}{prec:>10.1f}%{rec:>8.1f}%")

print("\n" + "-" * 78)
print(f"MISCLASSIFIED ({int((~scored.hit).sum())})")
print("-" * 78)
print(f"{'series':<16}{'true':>5}{'pred':>6}  {'units':<22}title")
for _, r in scored[~scored.hit].sort_values(["truth", "pred"]).iterrows():
    print(f"{r.series_id:<16}{r.truth:>5}{r.pred:>6}  {str(r.units_short)[:20]:<22}{r.title}")

print("\n" + "-" * 78)
print("FALSE POSITIVES / FALSE NEGATIVES BY CODE")
print("-" * 78)
for c in labels:
    fp = scored[(scored.truth != c) & (scored.pred == c)]
    fn = scored[(scored.truth == c) & (scored.pred != c)]
    if len(fp) == 0 and len(fn) == 0:
        continue
    print(f"\ncode {c}:")
    if len(fp):
        print(f"  FP ({len(fp)}) wrongly called {c}: " +
              ", ".join(f"{r.series_id}(true {r.truth})" for _, r in fp.iterrows()))
    if len(fn):
        print(f"  FN ({len(fn)}) truly {c} but called otherwise: " +
              ", ".join(f"{r.series_id}(pred {r.pred})" for _, r in fn.iterrows()))

# ---- write the FP / FN sets out ---------------------------------------------
fp_rows, fn_rows = [], []
for c in labels:
    for _, r in scored[(scored.truth != c) & (scored.pred == c)].iterrows():
        fp_rows.append({"code": c, "kind": "false_positive", "series_id": r.series_id,
                        "true_code": r.truth, "pred_code": r.pred,
                        "units_short": r.units_short, "title": r.title})
    for _, r in scored[(scored.truth == c) & (scored.pred != c)].iterrows():
        fn_rows.append({"code": c, "kind": "false_negative", "series_id": r.series_id,
                        "true_code": r.truth, "pred_code": r.pred,
                        "units_short": r.units_short, "title": r.title})

cols = ["code", "kind", "series_id", "true_code", "pred_code", "units_short", "title"]
fp_df = pd.DataFrame(fp_rows, columns=cols)
fn_df = pd.DataFrame(fn_rows, columns=cols)
fp_df.to_csv(os.path.join(HERE, "false_positives.csv"), index=False)
fn_df.to_csv(os.path.join(HERE, "false_negatives.csv"), index=False)

# errors.csv is ONE ROW PER MISCLASSIFIED SERIES.
#
# It is deliberately not a concatenation of the two files above. In a multiclass
# problem every error is counted twice from a per-class view -- once as a false
# negative for its true code, once as a false positive for the code it was given
# -- so concatenating them reports 16 rows for 8 errors.
errors = scored[~scored.hit][["series_id", "truth", "pred", "units_short", "title"]].copy()
errors = errors.rename(columns={"truth": "true_code", "pred": "pred_code"})
errors["counts_as_FN_for"] = errors["true_code"]
errors["counts_as_FP_for"] = errors["pred_code"]
errors = errors.sort_values(["true_code", "pred_code"])
errors.to_csv(os.path.join(HERE, "errors.csv"), index=False)

print(f"\nfull per-series results  -> classify_eval.csv")
print(f"errors, 1 row per series ({len(errors)}) -> errors.csv")
print(f"false positives by code  ({len(fp_df)}) -> false_positives.csv")
print(f"false negatives by code  ({len(fn_df)}) -> false_negatives.csv")
print(f"  note: FP+FN = {len(fp_df) + len(fn_df)} rows for {len(errors)} errors — each error is an FN"
      "\n  for its true code and an FP for its predicted code. errors.csv is deduplicated.")


# ============================================================================ #
# PART 2 -- codes 8 and 9, against the live universe
# ============================================================================ #
print("\n" + "=" * 78)
print("PART 2 -- CODES 8 AND 9 (no ground truth; checked against the live universe)")
print("=" * 78)
print("  tcode 8:  x.diff() / 100.0   net survey balances -> fractional rate")
print("  tcode 9:  x.diff() / denom   dollar flows / deficits, rescaled")
print("  FRED-MD never assigns these, so PART 1 cannot score them.")

emitted = set(scored.pred)
print(f"\n  fired on the MD panel? {'yes' if ({8, 9} & emitted) else 'no'} "
      f"-- so they cost nothing against the {acc:.2f}%")

if not RUN_DB:
    print("\n  [--no-db] skipping the live-universe audit.")
    sys.exit(0)

try:
    from sqlalchemy import text
    from core.db import engine
    conn = engine.connect()
except Exception as e:
    print(f"\n  database unreachable ({type(e).__name__}) — skipping the audit.")
    print("  PART 1 above is unaffected. Re-run with the DB up, or pass --no-db.")
    sys.exit(0)

CANDIDATE_SQL = text("""
    SELECT series_id, title, units_short, frequency_short
    FROM fred_series_filtered
    WHERE title ILIKE '%net%' OR title ILIKE '%deficit%' OR title ILIKE '%surplus%'
       OR title ILIKE '%flow%'  OR title ILIKE '%change%'  OR title ILIKE '%minus%'
       OR units_short ILIKE '%net%' OR units_short ILIKE '%minus%'
       OR title ILIKE '%commercial paper issues%'
    ORDER BY series_id
""")
OBS_SQL = text("""
    SELECT o.date, o.value
    FROM fred_observations o
    JOIN fred_series_filtered f USING (series_id_hash)
    WHERE f.series_id = :sid AND o.value IS NOT NULL
      AND o.realtime_end > '9999-01-01'
    ORDER BY o.date
""")

with conn:
    cands = pd.DataFrame(conn.execute(CANDIDATE_SQL).fetchall(),
                         columns=["series_id", "title", "units_short", "frequency_short"])
    print(f"\n  candidate series matching the 8/9 trigger words: {len(cands)}")
    print("  (the full cascade still decides, so branch order is respected)")

    audit_rows = []
    for _, c in cands.iterrows():
        obs = conn.execute(OBS_SQL, {"sid": c.series_id}).fetchall()
        if len(obs) < 24:
            continue
        s = pd.Series([float(v) for _, v in obs],
                      index=pd.to_datetime([d for d, _ in obs]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        try:
            stat, tf = classify(series=s, freq=c.frequency_short,
                                units_short=c.units_short,
                                series_id=c.series_id, title=c.title)
            code = int(tf["code"])
        except Exception as e:
            audit_rows.append({**c, "code": None, "roundtrip_err": np.nan})
            continue

        rt_err = np.nan
        try:
            back = reverse_transxf_fred(
                stat, code,
                init_val1=tf.get("init_val1"), init_val2=tf.get("init_val2"),
                code9_trillion_denom=tf.get("code9_trillion_denom", 1.0),
            )
            both = pd.concat([s.rename("orig"), pd.Series(back, name="back")], axis=1).dropna()
            if len(both):
                scale = both["orig"].abs().mean() or 1.0
                rt_err = float((both["orig"] - both["back"]).abs().mean() / scale * 100)
        except Exception:
            pass

        audit_rows.append({**c, "code": code, "roundtrip_err": rt_err})

res = pd.DataFrame(audit_rows)
res.to_csv(os.path.join(HERE, "codes_8_9_audit.csv"), index=False)

print("\n  code distribution across those candidates:")
for code, n in res.code.value_counts(dropna=False).sort_index().items():
    print(f"    code {code}: {n}")

for code in (8, 9):
    hit = res[res.code == code]
    print(f"\n  {'-' * 74}")
    print(f"  CODE {code}: {len(hit)} series")
    print(f"  {'-' * 74}")
    if hit.empty:
        print("    never fires on the live universe")
        continue
    err = pd.to_numeric(hit.roundtrip_err, errors="coerce")
    print(f"    round-trip error: median {err.median():.4g}%  max {err.max():.4g}%  "
          f"({int(err.notna().sum())}/{len(hit)} evaluated)")
    print(f"\n    {'series_id':<20}{'units':<20}{'rt err %':>10}  title")
    for _, r in hit.head(12).iterrows():
        e = "n/a" if pd.isna(r.roundtrip_err) else f"{r.roundtrip_err:.3g}"
        print(f"    {str(r.series_id)[:19]:<20}{str(r.units_short)[:19]:<20}{e:>10}  {str(r.title)[:48]}")
    if len(hit) > 12:
        print(f"    ... {len(hit) - 12} more in codes_8_9_audit.csv")

print("\n  Round-trip at machine precision means transxf_fred / reverse_transxf_fred")
print("  are exactly invertible for these codes. That is a CORRECTNESS check, not")
print("  an accuracy one -- whether 8/9 are the right choice for these series is a")
print("  modelling judgement no answer key can settle.")
print(f"\n  full audit -> codes_8_9_audit.csv")
