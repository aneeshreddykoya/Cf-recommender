"""
data_fetcher.py - download data from the public Codeforces API into SQLite.

Typical use (takes ~20 minutes for 400 users because Codeforces allows
only one request every 2 seconds; it is RESUMABLE, so you can stop and re-run):

    python data_fetcher.py all --n 400

Or step by step:
    python data_fetcher.py problems                 # problem list (ratings + tags)
    python data_fetcher.py sample --n 400           # pick users, spread over rating levels
    python data_fetcher.py add --handles tourist Petr
    python data_fetcher.py submissions              # download submissions of picked users
"""
import argparse
import re
import threading
import time

import numpy as np
import pandas as pd
import requests

import config
import db

# Verdicts that say nothing about a user's skill - we don't store them.
IGNORED_VERDICTS = {"TESTING", "COMPILATION_ERROR", "SKIPPED"}

_throttle_lock = threading.Lock()
_last_call = [0.0]


class CFError(Exception):
    """Codeforces answered, but with an error (e.g. handle not found)."""


class CFUnavailable(Exception):
    """We could not reach Codeforces (network down, timeout, 5xx...)."""


def is_valid_handle(handle: str) -> bool:
    """Security: only letters, digits, _ - . and 3-24 chars may pass."""
    return isinstance(handle, str) and re.fullmatch(config.HANDLE_RE, handle) is not None


def cf_get(method: str, params: dict | None = None) -> dict:
    """
    Call one Codeforces API method politely:
      * waits so we never exceed 1 request / 2.1 s (shared by all threads)
      * has a timeout and a few retries
      * passes parameters via `params=` so requests URL-encodes them safely
        (we never build URLs by gluing user text into a string)
    """
    last_error = None
    for attempt in range(config.CF_RETRIES):
        with _throttle_lock:
            wait = config.CF_MIN_INTERVAL - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            resp = requests.get(config.CF_API_URL + method, params=params, timeout=config.CF_TIMEOUT)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)          # back off: 1s, 2s, 4s
            continue
        if data.get("status") == "OK":
            return data["result"]
        comment = str(data.get("comment", "unknown error"))
        if "limit exceeded" in comment.lower():   # we were too fast: wait and retry
            last_error = CFError(comment)
            time.sleep(3 * (attempt + 1))
            continue
        raise CFError(comment)
    raise CFUnavailable(f"Codeforces unreachable: {last_error}")


# --------------------------------------------------------------------------
# 1. Problems
# --------------------------------------------------------------------------
def fetch_problems(db_path=None) -> int:
    result = cf_get("problemset.problems")
    rows = []
    for p in result["problems"]:
        if "contestId" not in p or p["contestId"] >= 100000:   # skip gym problems
            continue
        rows.append((
            f"{p['contestId']}{p['index']}", p["contestId"], p["index"],
            p.get("name", ""), p.get("rating"), ",".join(p.get("tags", [])),
        ))
    with db.connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO problems VALUES (?,?,?,?,?,?)", rows)
    print(f"Saved {len(rows)} problems.")
    return len(rows)


# --------------------------------------------------------------------------
# 2. Users
# --------------------------------------------------------------------------
RATING_BINS = [0, 1200, 1400, 1600, 1900, 2200, 5000]   # roughly Newbie ... Grandmaster


def sample_users(n: int, db_path=None, seed: int = 42) -> int:
    """Pick n active rated users, the same number from each rating band."""
    print("Downloading the list of active rated users (one big request)...")
    rated = cf_get("user.ratedList", {"activeOnly": "true"})
    df = pd.DataFrame([{"handle": u["handle"], "rating": u.get("rating", 0)} for u in rated])
    df["band"] = pd.cut(df["rating"], RATING_BINS, right=False)
    per_band = max(1, n // (len(RATING_BINS) - 1))
    picked = pd.concat([
        g.sample(min(len(g), per_band), random_state=seed)
        for _, g in df.groupby("band", observed=True)
    ])
    with db.connect(db_path) as conn:
        conn.executemany("INSERT OR IGNORE INTO users (handle, rating) VALUES (?, ?)",
                         list(zip(picked["handle"], picked["rating"].astype(int))))
    print(f"Added {len(picked)} users ({per_band} per rating band).")
    return len(picked)


def add_users(handles: list[str], db_path=None) -> int:
    good = [h for h in handles if is_valid_handle(h)]
    for bad in set(handles) - set(good):
        print(f"Skipping invalid handle: {bad!r}")
    with db.connect(db_path) as conn:
        conn.executemany("INSERT OR IGNORE INTO users (handle) VALUES (?)", [(h,) for h in good])
    return len(good)


# --------------------------------------------------------------------------
# 3. Submissions
# --------------------------------------------------------------------------
def parse_submissions(result: list[dict], handle: str) -> list[tuple]:
    rows = []
    for s in result:
        prob = s.get("problem", {})
        verdict = s.get("verdict")
        if "contestId" not in prob or prob["contestId"] >= 100000:   # no gym
            continue
        if not verdict or verdict in IGNORED_VERDICTS:
            continue
        rows.append((s["id"], handle, f"{prob['contestId']}{prob['index']}",
                     verdict, int(s["creationTimeSeconds"])))
    return rows


def fetch_submissions_df(handle: str, max_subs: int | None = None) -> pd.DataFrame:
    """Download one user's submissions and return them as a DataFrame (no DB write)."""
    if not is_valid_handle(handle):
        raise CFError("invalid handle")
    params = {"handle": handle}
    if max_subs:
        params["from"], params["count"] = 1, max_subs
    rows = parse_submissions(cf_get("user.status", params), handle)
    return pd.DataFrame(rows, columns=["id", "handle", "problem_id", "verdict", "created_at"])


def fetch_all_submissions(db_path=None, limit: int | None = None, max_subs: int | None = None):
    """Download submissions for every user not fetched yet. Safe to stop and re-run."""
    with db.connect(db_path) as conn:
        todo = [r[0] for r in conn.execute("SELECT handle FROM users WHERE fetched_at IS NULL")]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} users to download (~{len(todo) * 2.5 / 60:.0f} min).")
    for i, handle in enumerate(todo, 1):
        try:
            df = fetch_submissions_df(handle, max_subs)
        except CFError as exc:
            print(f"[{i}/{len(todo)}] {handle}: skipped ({exc})")
            with db.connect(db_path) as conn:      # mark so we don't retry forever
                conn.execute("UPDATE users SET fetched_at = -1 WHERE handle = ?", (handle,))
            continue
        except CFUnavailable as exc:
            print(f"Stopping: {exc}. Re-run later to continue.")
            return
        with db.connect(db_path) as conn:
            conn.executemany("INSERT OR IGNORE INTO submissions VALUES (?,?,?,?,?)",
                             df.itertuples(index=False, name=None))
            conn.execute("UPDATE users SET fetched_at = ? WHERE handle = ?", (int(time.time()), handle))
        print(f"[{i}/{len(todo)}] {handle}: {len(df)} submissions")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["problems", "sample", "add", "submissions", "all"])
    ap.add_argument("--n", type=int, default=400, help="number of users to sample")
    ap.add_argument("--handles", nargs="*", default=[], help="handles for the 'add' command")
    ap.add_argument("--limit", type=int, help="download at most this many users now")
    ap.add_argument("--max-subs", type=int, help="keep only each user's newest N submissions")
    ap.add_argument("--db", help="database file (default: cf_data.db or $CF_DB)")
    a = ap.parse_args()

    if a.command in ("problems", "all"):
        fetch_problems(a.db)
    if a.command in ("sample", "all"):
        sample_users(a.n, a.db)
    if a.command == "add":
        print(f"Added {add_users(a.handles, a.db)} users.")
    if a.command in ("submissions", "all"):
        fetch_all_submissions(a.db, a.limit, a.max_subs)


if __name__ == "__main__":
    main()
