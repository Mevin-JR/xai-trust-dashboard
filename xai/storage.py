"""Anonymous response storage (SQLite). No names or contact details are stored."""
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd

from .data import DATA_DIR

DB_PATH = DATA_DIR / "responses.db"

# Balanced Latin square: each condition appears once in each position and
# follows every other condition exactly once across the four orders.
CONDITIONS = ["black_box", "lime", "shap", "plain_language"]
ORDERS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]


def _connect():
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS participants (
        pid TEXT PRIMARY KEY, order_idx INTEGER, started_at TEXT, completed_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS responses (
        pid TEXT, block INTEGER, condition TEXT, case_id TEXT, item TEXT, value TEXT, recorded_at TEXT)""")
    return con


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_participant() -> tuple[str, list[str]]:
    """Create an anonymous participant and assign the next counterbalanced order."""
    with _connect() as con:
        n = con.execute("SELECT COUNT(*) FROM participants").fetchone()[0]
        pid, order_idx = uuid.uuid4().hex[:8], n % len(ORDERS)
        con.execute("INSERT INTO participants VALUES (?, ?, ?, NULL)", (pid, order_idx, _now()))
    return pid, [CONDITIONS[i] for i in ORDERS[order_idx]]


def save(pid: str, block: int, condition: str, case_id: str, answers: dict):
    with _connect() as con:
        con.executemany("INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(pid, block, condition, case_id, k, str(v), _now()) for k, v in answers.items()])


def complete(pid: str):
    with _connect() as con:
        con.execute("UPDATE participants SET completed_at = ? WHERE pid = ?", (_now(), pid))


def export() -> pd.DataFrame:
    with _connect() as con:
        return pd.read_sql("""SELECT r.*, p.order_idx, p.completed_at FROM responses r
                              JOIN participants p USING (pid) ORDER BY r.pid, r.block""", con)


def counts() -> tuple[int, int]:
    with _connect() as con:
        return con.execute("SELECT COUNT(*), COUNT(completed_at) FROM participants").fetchone()
