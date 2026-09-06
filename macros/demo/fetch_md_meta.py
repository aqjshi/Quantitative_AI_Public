"""Fetch FRED metadata (title / units / notes / frequency) for every series in
the FRED-MD 2026-06 vintage, so the classifier can be scored against McCracken &
Ng's published transformation codes."""
import os, sys, json, time, csv
import urllib.request, urllib.parse, urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from dotenv import dotenv_values

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "md_meta.json")

env = dotenv_values(os.path.join(REPO, ".env"))
KEY = env.get("FRED_KEY_0") or env.get("fred_key_0")
if not KEY:
    sys.exit("no FRED_KEY_0 in .env")

with open(os.path.join(REPO, "macros/math/2026-06-MD.csv")) as f:
    r = csv.reader(f)
    hdr = next(r)
ids = [h.strip() for h in hdr[1:] if h.strip()]

cache = {}
if os.path.exists(CACHE):
    cache = json.load(open(CACHE))

def fetch(sid):
    qs = urllib.parse.urlencode({"series_id": sid, "api_key": KEY, "file_type": "json"})
    url = f"https://api.stlouisfed.org/fred/series?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            js = json.load(resp)
        s = js["seriess"][0]
        return {
            "series_id": s.get("id"),
            "title": s.get("title"),
            "units": s.get("units"),
            "units_short": s.get("units_short"),
            "frequency": s.get("frequency"),
            "frequency_short": s.get("frequency_short"),
            "seasonal_adjustment_short": s.get("seasonal_adjustment_short"),
            "notes": s.get("notes", ""),
            "ok": True,
        }
    except urllib.error.HTTPError as e:
        return {"series_id": sid, "ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"series_id": sid, "ok": False, "error": str(e)[:80]}

todo = [i for i in ids if i not in cache]
print(f"{len(ids)} series total, {len(cache)} cached, {len(todo)} to fetch")
for n, sid in enumerate(todo, 1):
    cache[sid] = fetch(sid)
    if n % 20 == 0 or n == len(todo):
        print(f"  {n}/{len(todo)}")
        json.dump(cache, open(CACHE, "w"), indent=1)
    time.sleep(0.15)

json.dump(cache, open(CACHE, "w"), indent=1)
ok = [k for k, v in cache.items() if v.get("ok")]
bad = [k for k, v in cache.items() if not v.get("ok")]
print(f"\nresolved: {len(ok)}/{len(ids)}")
print(f"unresolved ({len(bad)}): {bad}")
