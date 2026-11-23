"""Anonymous response storage. No names or contact details are stored.

Online, responses go to the Postgres database set in the DATABASE_URL secret
(Supabase). Locally, with no DATABASE_URL, they go to data/responses.db (SQLite).
"""
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text

from .data import DATA_DIR

DB_PATH = DATA_DIR / "responses.db"

# Balanced Latin square: each condition appears once in each position and
# follows every other condition exactly once across the four orders.
CONDITIONS = ["black_box", "lime", "shap", "plain_language"]
ORDERS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS participants (
        pid TEXT PRIMARY KEY, order_idx INTEGER, started_at TEXT, completed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS responses (
        pid TEXT, block INTEGER, condition TEXT, case_id TEXT, item TEXT, value TEXT, recorded_at TEXT)""",
]
# Supabase exposes public tables through its REST API; row-level security with no
# policies closes that route. The app connects as the table owner, which bypasses RLS.
POSTGRES_ONLY = [
    "ALTER TABLE participants ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE responses ENABLE ROW LEVEL SECURITY",
]

_engines = {}


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:  # no secrets file locally
            url = None
    if not url:
        DATA_DIR.mkdir(exist_ok=True)
        return f"sqlite:///{DB_PATH}"
    # Supabase gives postgresql://...; use the psycopg 3 driver.
    return url.replace("postgres://", "postgresql://", 1).replace("postgresql://", "postgresql+psycopg://", 1)


def backend() -> str:
    return "Supabase (Postgres)" if database_url().startswith("postgresql") else "local SQLite"


def _engine():
    url = database_url()
    if url not in _engines:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.begin() as con:
            for stmt in SCHEMA + (POSTGRES_ONLY if engine.dialect.name == "postgresql" else []):
                con.execute(text(stmt))
        _engines[url] = engine
    return _engines[url]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_participant() -> tuple[str, list[str]]:
    """Create an anonymous participant and assign the next counterbalanced order."""
    with _engine().begin() as con:
        n = con.execute(text("SELECT COUNT(*) FROM participants")).scalar_one()
        pid, order_idx = uuid.uuid4().hex[:8], n % len(ORDERS)
        con.execute(text("INSERT INTO participants VALUES (:pid, :order_idx, :started, NULL)"),
                    {"pid": pid, "order_idx": order_idx, "started": _now()})
    return pid, [CONDITIONS[i] for i in ORDERS[order_idx]]


def save(pid: str, block: int, condition: str, case_id: str, answers: dict):
    rows = [{"pid": pid, "block": block, "condition": condition, "case_id": case_id,
             "item": k, "value": str(v), "at": _now()} for k, v in answers.items()]
    with _engine().begin() as con:
        con.execute(text("INSERT INTO responses VALUES (:pid, :block, :condition, :case_id, :item, :value, :at)"), rows)


def complete(pid: str):
    with _engine().begin() as con:
        con.execute(text("UPDATE participants SET completed_at = :at WHERE pid = :pid"), {"at": _now(), "pid": pid})


def export() -> pd.DataFrame:
    with _engine().connect() as con:
        return pd.read_sql(text("""SELECT r.*, p.order_idx, p.completed_at FROM responses r
                                   JOIN participants p USING (pid) ORDER BY r.pid, r.block"""), con)


def counts() -> tuple[int, int]:
    with _engine().connect() as con:
        total, done = con.execute(text("SELECT COUNT(*), COUNT(completed_at) FROM participants")).one()
    return total, done
