import os
from datetime import date, datetime, UTC
from datetime import timedelta, datetime, date
import threading

import time
import psycopg
import threading
import time
from datetime import datetime

import threading
import time
from datetime import datetime

class TokenBucketRateLimiter:
    def __init__(self, rate_per_sec):
        self.delay = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_call = 0

    def wait(self):
        with self.lock:
            now = datetime.now().timestamp()
            if self.next_call > now:
                time.sleep(self.next_call - now)
            self.next_call = max(self.next_call, now) + self.delay



def power_db_worker(result_queue, db_url):
    """
    Generic DB Worker that accepts raw SQL Copy commands.
    Input item format: (batch_data_list, sql_copy_command, info_str)
    """
    # Standardize DSN for psycopg
    dsn = db_url.replace("postgresql+psycopg2://", "postgresql://")
    
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            print("[!] DB Worker: Connected.")
            while True:
                item = result_queue.get()
                if item is None: break 
                
                batch_data, copy_query, info = item
                
                if not batch_data:
                    continue

                try:
                    with conn.cursor() as cur:
                        with cur.copy(copy_query) as copy:
                            for row in batch_data:
                                line = "\t".join(map(str, row)) + "\n"
                                copy.write(line)
                except Exception as e:
                    print(f"[!] BATCH ERROR ({info}): {e}")
    except Exception as e:
        print(f"CRITICAL: DB Worker died: {e}")
        os._exit(1)


def parse_osi(osi: str):
    s = osi[2:] if osi.startswith("O:") else osi

    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    ul = s[:i]
    rest = s[i:]

    if len(rest) < 15:
        raise ValueError(f"Bad OSI: {osi} (rest too short: '{rest}')")

    yymmdd = rest[:6]
    right  = rest[6].upper()
    kcode  = rest[7:15]

    if right not in ("C", "P"):
        raise ValueError(f"Bad OSI right: {osi}")
    if not (yymmdd.isdigit() and kcode.isdigit()):
        raise ValueError(f"Bad OSI digits: {osi}")

    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    exp = date(2000 + yy, mm, dd)
    K = int(kcode) / 1000.0
    return ul, exp, right, K


    
# --- PHASE 1: EXPLORATION & POLICY ---
def get_iso_date(val):
    """
    Standardizes date inputs into a rigid ISO string (YYYY-MM-DD).
    Essential for creating stable dictionary keys for deduplication.
    """
    if val is None:
        return "none"
    
    # Handle already processed date/datetime objects
    if isinstance(val, (date, datetime)):
        return val.strftime('%Y-%m-%d')
    
    # Handle string inputs (e.g. "2025-10-31T00:00:00Z" or "2025-10-31")
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() == "none":
            return "none"
        # Slice first 10 chars to catch 'YYYY-MM-DD' from a full timestamp string
        return val[:10]
    
    return str(val)


