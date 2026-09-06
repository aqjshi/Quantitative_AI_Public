import os
import sys
import time
from typing import Dict, List, Optional

import mmh3
import requests
from tqdm import tqdm
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import POLY_KEY, engine, Base
from core.sieve import TokenBucketRateLimiter

from sqlalchemy import (
    Column, Integer, BigInteger, String, Boolean, Date, Index, text
)

BASE_URL = "https://api.massive.com/v3/reference/tickers"


def hash_string_to_int64(input_string, seed=42):
    """Returns a signed 64-bit integer securely bound to standard architectures."""
    return mmh3.hash64(input_string, seed=seed)[0]


class UniqueInstrument(Base):
    __tablename__ = "unique_instruments"

    id = Column(Integer, primary_key=True)
    composite_figi = Column(String(20), nullable=False)
    composite_figi_hash = Column(BigInteger, index=True, nullable=True)

    ticker = Column(String(20))
    ticker_hash = Column(BigInteger, nullable=True)
    name = Column(String(255))

    cik = Column(BigInteger, nullable=True)

    sic_code = Column(Integer, nullable=True)
    sic_description = Column(String(255), nullable=True)
    total_employees = Column(Integer, nullable=True)
    market_cap = Column(BigInteger, nullable=True)
    share_class_shares_outstanding = Column(BigInteger, nullable=True)
    weighted_shares_outstanding = Column(BigInteger, nullable=True)
    city = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    state = Column(String(50), nullable=True)

    market = Column(String(20))
    locale = Column(String(20))
    primary_exchange = Column(String(20))
    type = Column(String(20))
    active = Column(Boolean, default=True)
    currency_name = Column(String(20))
    share_class_figi = Column(String(20))
    point_in_time_date = Column(Date)
    upsert_date = Column(Date)

    __table_args__ = (
        Index('idx_unique_figi_hash_pit', 'composite_figi_hash', 'point_in_time_date'),
    )


class TickerTrie:
    """Prefix tree over ticker symbols.

    Holds the whole discovered universe so membership is a walk of the symbol's
    own characters rather than a scan. It is also what makes the two passes
    ordered rather than merely concatenated: the active sweep runs first and
    claims a symbol, so when the inactive sweep re-reports the same symbol it is
    rejected here and the live record is the one that survives.
    """

    def __init__(self):
        self.root: Dict = {}
        self.size = 0

    def insert(self, ticker: str) -> bool:
        """Adds a symbol. Returns False when it was already present."""
        node = self.root
        for char in ticker:
            node = node.setdefault(char, {})
        if node.get("$"):
            return False
        node["$"] = True
        self.size += 1
        return True

    def __contains__(self, ticker: str) -> bool:
        node = self.root
        for char in ticker:
            if char not in node:
                return False
            node = node[char]
        return bool(node.get("$"))

    def __len__(self) -> int:
        return self.size

    def starts_with(self, prefix: str) -> int:
        """How many symbols sit under a prefix. Used for the coverage report."""
        node = self.root
        for char in prefix:
            if char not in node:
                return 0
            node = node[char]

        count = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current.get("$"):
                count += 1
            for key, child in current.items():
                if key != "$":
                    stack.append(child)
        return count


def fetch_all_tickers(market:str, active: str, trie: TickerTrie, api_key: str, 
                      limiter: TokenBucketRateLimiter) -> List[dict]:
    """Walks every page of the reference endpoint for one active state.

    limit=1000 is the per-page maximum, not a cap on the sweep -- the loop keeps
    following next_url until the vendor stops issuing one, so the whole universe
    is collected. Symbols already in the trie are dropped here, which is what
    keeps the second pass from overwriting the first.
    """
    params = {
        "market": market,
        "active": active,
        "order": "asc",
        "limit": 1000,
        "sort": "ticker",
        "ticker.gte": "A",
        "apiKey": api_key,
    }

    collected: List[dict] = []
    url, current_params = BASE_URL, params
    page = 0
    pbar = tqdm(desc=f"Discovering active={active}", unit="page")

    while url:
        limiter.wait()
        try:
            resp = requests.get(url, params=current_params, timeout=30)
        except Exception as e:
            tqdm.write(f" [!] Request failed on page {page}: {e}")
            break

        if resp.status_code == 429:
            tqdm.write(" [!] Rate limit hit. Backing off 5 seconds...")
            time.sleep(5)
            continue
        if resp.status_code != 200:
            tqdm.write(f" [!] Discovery stopped at page {page}: HTTP {resp.status_code}")
            break

        data = resp.json()
        for raw in data.get("results") or []:
            ticker = raw.get("ticker")
            if not ticker or not trie.insert(ticker):
                continue
            collected.append(raw)

        page += 1
        pbar.update(1)
        pbar.set_postfix(kept=len(collected), universe=len(trie))

        next_url = data.get("next_url")
        if not next_url:
            break
        url = next_url if "apiKey" in next_url else f"{next_url}&apiKey={api_key}"
        current_params = None

    pbar.close()
    tqdm.write(f"[*] active={active}: {page} pages, {len(collected)} new symbols kept.")
    return collected


def to_row(raw: dict, is_active: bool, seed: int = 42) -> tuple:
    """Maps one API record onto the unique_instruments column order."""
    ticker = raw.get("ticker")
    figi = raw.get("composite_figi") or "UNKNOWN"
    cik = raw.get("cik")
    today = datetime.now(timezone.utc).date()

    return (
        figi,
        hash_string_to_int64(figi, seed=seed),
        ticker,
        hash_string_to_int64(ticker, seed=seed),
        (raw.get("name") or "")[:255] or None,
        int(str(cik).lstrip("0")) if cik and str(cik).lstrip("0").isdigit() else None,
        raw.get("market"),
        raw.get("locale"),
        raw.get("primary_exchange"),
        raw.get("type"),
        bool(raw.get("active", is_active)),
        raw.get("currency_name"),
        raw.get("share_class_figi"),
        today,
        today,
    )


def upsert_unique_instruments(rows: List[tuple]) -> int:
    """Writes the discovered universe, skipping symbols already stored.

    unique_instruments carries no unique constraint on ticker_hash, only the
    surrogate primary key, so ON CONFLICT has nothing to bind to and a plain
    INSERT would append a second copy of every symbol on a re-run. The NOT EXISTS
    guard makes the pass idempotent.
    """
    if not rows:
        return 0

    insert_sql = text("""
        INSERT INTO unique_instruments (
            composite_figi, composite_figi_hash, ticker, ticker_hash, name, cik,
            market, locale, primary_exchange, type, active, currency_name,
            share_class_figi, point_in_time_date, upsert_date
        )
        SELECT
            :composite_figi, :composite_figi_hash, :ticker, :ticker_hash, :name, :cik,
            :market, :locale, :primary_exchange, :type, :active, :currency_name,
            :share_class_figi, :point_in_time_date, :upsert_date
        WHERE NOT EXISTS (
            SELECT 1 FROM unique_instruments WHERE ticker_hash = :ticker_hash
        )
    """)

    columns = ("composite_figi", "composite_figi_hash", "ticker", "ticker_hash",
               "name", "cik", "market", "locale", "primary_exchange", "type",
               "active", "currency_name", "share_class_figi",
               "point_in_time_date", "upsert_date")

    written = 0
    chunk_size = 1000
    with engine.begin() as conn:
        for i in tqdm(range(0, len(rows), chunk_size), desc="Writing universe", unit="chunk"):
            payload = [dict(zip(columns, row)) for row in rows[i:i + chunk_size]]
            result = conn.execute(insert_sql, payload)
            written += result.rowcount if result.rowcount and result.rowcount > 0 else 0
    return written


# --- MAIN PIPELINE EXECUTION ---
def main():
    tables_to_drop = ["unique_instruments"]

    with engine.begin() as conn:
        for table_name in tables_to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
            tqdm.write(f" -> Dropped table: {table_name}")

    tqdm.write("[*] Rebuilding fresh database schema definitions...")
    Base.metadata.create_all(engine)


    limiter = TokenBucketRateLimiter(rate_per_sec=10)
    trie = TickerTrie()

    # Active first so a symbol that appears in both states keeps its live record.
    active_equities = fetch_all_tickers("stocks", "true", trie, POLY_KEY, limiter)
    inactive_equities = fetch_all_tickers("stocks", "false", trie, POLY_KEY, limiter)
    active_indices = fetch_all_tickers("indices", "true", trie, POLY_KEY, limiter)
    inactive_indices = fetch_all_tickers("indices", "false", trie, POLY_KEY, limiter)

    rows = ([to_row(raw, True) for raw in active_equities]
            + [to_row(raw, False) for raw in inactive_equities] 
            + [to_row(raw, False) for raw in active_indices]
            + [to_row(raw, False) for raw in inactive_indices]
            )

    tqdm.write(f"[*] Trie holds {len(trie)} unique symbols. Writing {len(rows)} rows...")
    written = upsert_unique_instruments(rows)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM unique_instruments")).scalar()
        live = conn.execute(text("SELECT count(*) FROM unique_instruments WHERE active")).scalar()

    tqdm.write(f"[*] Inserted {written} new rows. unique_instruments now holds {total} "
               f"({live} active, {total - live} inactive).")
    tqdm.write(f"[*] Trie spot check -- symbols under 'A': {trie.starts_with('A')}, "
               f"'B': {trie.starts_with('B')}, 'Z': {trie.starts_with('Z')}")

    engine.dispose()
    tqdm.write("[*] Pipeline connection pools disposed. Execution run loop finished cleanly.")


if __name__ == "__main__":
    main()
