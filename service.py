"""
service.py - the recommendation logic behind the API, with NO web-framework code.
(Keeping it separate from main.py means we can unit-test it without a server.)
"""
import json
import os
import re
import threading
import time

import config
import db
from data_fetcher import CFError, CFUnavailable, fetch_submissions_df, is_valid_handle
from models import RecContext, build_catalog, make_models


class ServiceError(Exception):
    """An error with an HTTP-style status code and a safe, human-readable message."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class RecommenderService:
    LIVE_TTL = 600          # cache live Codeforces downloads for 10 minutes
    LIVE_MAX = 256          # ...and never keep more than 256 users in memory

    def __init__(self, db_path=None):
        self.db_path = db_path
        self.ready = False
        self._live_cache: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def load(self):
        subs, problems = db.load_submissions(self.db_path), db.load_problems(self.db_path)
        if subs.empty or problems.empty:
            self.ready = False
            return self
        catalog = build_catalog(subs, problems)
        self.ctx = RecContext(subs, catalog)
        self.models = make_models(self.ctx)
        self.lookup = {h.lower(): h for h in self.ctx.handles}     # case-insensitive handles
        self.meta = catalog.set_index("problem_id")
        self.ready = True
        return self

    # ------------------------------------------------------------------
    def _live_profile(self, handle: str):
        """For users NOT in our database: download their submissions from Codeforces."""
        now = time.time()
        with self._lock:
            hit = self._live_cache.get(handle.lower())
            if hit and now - hit[0] < self.LIVE_TTL:
                return hit[1]
        try:
            df = fetch_submissions_df(handle)
        except CFError:
            raise ServiceError(404, "Handle not found on Codeforces.")
        except CFUnavailable:
            raise ServiceError(502, "Codeforces is not reachable right now. Try again later.")
        profile = self.ctx.build_profile(df)
        with self._lock:
            if len(self._live_cache) >= self.LIVE_MAX:
                self._live_cache.clear()
            self._live_cache[handle.lower()] = (now, profile)
        return profile

    def recommend(self, handle: str, model: str = "item_cf", k: int = 10) -> dict:
        if not self.ready:
            raise ServiceError(503, "No data loaded. Run data_fetcher.py (or make_synthetic_data.py) first.")
        if not is_valid_handle(handle):
            raise ServiceError(422, "Invalid handle: use 3-24 letters, digits, '_', '-' or '.'.")
        if model not in self.models:
            raise ServiceError(422, f"Unknown model. Choose one of: {sorted(self.models)}")
        if not 1 <= k <= 50:
            raise ServiceError(422, "k must be between 1 and 50.")

        canonical = self.lookup.get(handle.lower())
        if canonical:
            profile, source, handle = self.ctx.profiles[canonical], "database", canonical
        else:
            profile, source = self._live_profile(handle), "live"

        m = self.models[model]
        items = m.recommend(profile, k)
        recs = []
        for i in items:
            pid = self.ctx.item_ids[i]
            row = self.meta.loc[pid]
            recs.append({
                "problem_id": pid,
                "name": row["name"],
                "rating": int(row["rating"]),
                "tags": list(row["tags"]),
                "url": f"https://codeforces.com/problemset/problem/{int(row['contest_id'])}/{row['idx']}",
                "reason": m.explain(profile, i),
            })
        result = {
            "handle": handle, "model": model, "source": source,
            "estimated_level": profile.level, "known_solved_problems": int(len(profile.solved_idx)),
            "recommendations": recs,
        }
        if len(profile.solved_idx) == 0:
            result["note"] = (f"No solved problems found for this user, so a default level of "
                              f"{config.DEFAULT_LEVEL} was assumed (cold start).")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def evaluation_results() -> dict:
        path = os.path.join(config.RESULTS_DIR, "results.json")
        if not os.path.exists(path):
            raise ServiceError(404, "No evaluation results yet. Run: python evaluate.py")
        with open(path) as f:
            return json.load(f)
