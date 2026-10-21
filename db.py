"""
db.py - SQLite schema and loading helpers.

Three tables:
  problems    one row per Codeforces problem (rating, tags)
  users       the sample of users we collected
  submissions every submission we downloaded (verdict + timestamp)
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    problem_id TEXT PRIMARY KEY,      -- e.g. '1700B'
    contest_id INTEGER NOT NULL,
    idx        TEXT NOT NULL,         -- 'A', 'B', 'C1' ...
    name       TEXT,
    rating     INTEGER,               -- NULL when Codeforces has not rated it
    tags       TEXT                   -- comma separated, e.g. 'dp,greedy'
);
CREATE TABLE IF NOT EXISTS users (
    handle     TEXT PRIMARY KEY,
    rating     INTEGER,
    fetched_at INTEGER                -- NULL until we downloaded their submissions
);
CREATE TABLE IF NOT EXISTS submissions (
    id         INTEGER PRIMARY KEY,   -- Codeforces submission id
    handle     TEXT NOT NULL,
    problem_id TEXT NOT NULL,
    verdict    TEXT NOT NULL,         -- 'OK', 'WRONG_ANSWER', ...
    created_at INTEGER NOT NULL       -- unix seconds
);
CREATE INDEX IF NOT EXISTS idx_sub_handle  ON submissions(handle);
CREATE INDEX IF NOT EXISTS idx_sub_problem ON submissions(problem_id);
"""


@contextmanager
def connect(path=None):
    """Open the database, make sure tables exist, and always close it."""
    conn = sqlite3.connect(path or config.DB_PATH)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def load_problems(path=None) -> pd.DataFrame:
    """Return problems with `tags` as a Python list per row."""
    with connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT problem_id, contest_id, idx, name, rating, tags FROM problems", conn
        )
    df["tags"] = df["tags"].fillna("").apply(lambda s: [t for t in s.split(",") if t])
    return df


def load_submissions(path=None) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(
            "SELECT handle, problem_id, verdict, created_at FROM submissions", conn
        )
