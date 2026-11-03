import os
import sys
import json
import time
import datetime
import random
import multiprocessing
from tqdm import tqdm
from sqlalchemy import text
import uuid
# Ensure core modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import POLY_KEY, engine, DATABASE_URL, Base


def execute_db_query(query, batch_data, info=""):
    """Replaces the entire power_db_worker loop. Executes database commands directly and synchronously."""
    # Convert pool URL to raw psycopg format if necessary
    dsn = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
    
    try:
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as conn:
            if "COPY" in query and ";" in query:
                with conn.transaction():
                    with conn.cursor() as cur:
                        parts = [p.strip() for p in query.split(";") if p.strip()]
                        copy_cmd, pre_cmds, post_cmds = "", [], []
                        found_copy = False
                        for p in parts:
                            if p.upper().startswith("COPY"):
                                copy_cmd = p
                                found_copy = True
                            elif not found_copy: pre_cmds.append(p)
                            else: post_cmds.append(p)

                        for cmd in pre_cmds: cur.execute(cmd)
                        if "FROM STDIN" not in copy_cmd.upper(): copy_cmd += " FROM STDIN"
                        with cur.copy(copy_cmd) as copy:
                            for row in batch_data:
                                copy.write("\t".join(map(str, row)) + "\n")
                        for cmd in post_cmds: cur.execute(cmd)
                        
            elif query.strip().upper().startswith("COPY"):
                with conn.cursor() as cur:
                    with cur.copy(query) as copy:
                        for row in batch_data:
                            copy.write("\t".join(map(str, row)) + "\n")
            else:
                with conn.cursor() as cur:
                    cur.execute(query, batch_data)
        return True
    except Exception as e:
        print(f"\n!!!! [DATABASE ERROR] Failed during {info}: {e} !!!!\n")
        return False
    



