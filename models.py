"""
models.py - the five recommenders, all behind the same interface.

Big picture
-----------
1. RecContext   : turns raw submissions into a clean numeric form ONCE
                  (a "profile" per user + arrays describing every problem).
2. Recommenders : each one only has to answer: "give every problem a score".
                  The base class then
                     (a) throws away problems the user already solved,
                     (b) throws away problems far from the user's level,
                     (c) returns the top-k by score.
   Steps (a) and (b) are IDENTICAL for all models, so the comparison in
   evaluate.py is fair: the only thing that differs is how problems are scored.

Ties are broken by popularity (then problem index), so results are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import config


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def build_catalog(subs: pd.DataFrame, problems: pd.DataFrame) -> pd.DataFrame:
    """
    The catalog = problems we are allowed to recommend: rated, and seen in `subs`.
    (When evaluating we pass ONLY the training submissions, so problems that
    were published after the cutoff date are not in the catalog. Recommending
    a problem that did not exist yet would be cheating.)
    """
    seen = subs["problem_id"].unique()
    return problems[problems["rating"].notna() & problems["problem_id"].isin(seen)].reset_index(drop=True)


@dataclass
class Profile:
    """Everything the models need to know about ONE user."""
    solved_idx: np.ndarray      # catalog indices of problems solved (in order of first solve)
    level: int                  # estimated rating = median of recent solved problem ratings
    tag_stats: dict             # tag -> [problems solved, problems attempted]


class RecContext:
    """Numeric view of the data: problem arrays + one Profile per user."""

    def __init__(self, subs: pd.DataFrame, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        self.item_ids = self.catalog["problem_id"].tolist()
        self.item_index = {pid: i for i, pid in enumerate(self.item_ids)}
        self.ratings = self.catalog["rating"].to_numpy(dtype=float)
        self.tags = self.catalog["tags"].tolist()
        self.n_items = len(self.item_ids)

        self.profiles: dict[str, Profile] = {
            handle: self.build_profile(g) for handle, g in subs.groupby("handle", sort=True)
        }
        self.handles = list(self.profiles)

        # popularity = how many users solved each problem
        pop = np.zeros(self.n_items)
        for p in self.profiles.values():
            pop[p.solved_idx] += 1
        self.popularity = pop

    def build_profile(self, subs: pd.DataFrame) -> Profile:
        """Build a Profile from one user's submissions (columns: problem_id, verdict, created_at)."""
        subs = subs[subs["problem_id"].isin(self.item_index)]
        if subs.empty:
            return Profile(np.array([], dtype=int), config.DEFAULT_LEVEL, {})

        subs = subs.sort_values("created_at")
        idx = subs["problem_id"].map(self.item_index).to_numpy()
        ok = (subs["verdict"] == "OK").to_numpy()

        solved_idx = pd.unique(idx[ok])          # unique, keeps time order
        attempted_idx = pd.unique(idx)
        solved_set = {int(i) for i in solved_idx}

        recent = solved_idx[-config.RECENT_SOLVES:]
        level = int(np.median(self.ratings[recent])) if len(recent) else config.DEFAULT_LEVEL

        tag_stats: dict[str, list[int]] = {}
        for i in attempted_idx:
            was_solved = int(i) in solved_set
            for tag in self.tags[int(i)]:
                st = tag_stats.setdefault(tag, [0, 0])
                st[1] += 1
                st[0] += was_solved
        return Profile(solved_idx.astype(int), level, tag_stats)

    def candidate_mask(self, profile: Profile) -> np.ndarray:
        """True for problems that are unsolved AND rated within [level-200, level+300]."""
        lo, hi = profile.level + config.BAND_LOW, profile.level + config.BAND_HIGH
        mask = (self.ratings >= lo) & (self.ratings <= hi)
        mask[profile.solved_idx] = False
        return mask


def _solved_matrix(ctx: RecContext) -> sparse.csr_matrix:
    """Users x problems matrix with 1 where the user solved the problem."""
    rows, cols = [], []
    for u, handle in enumerate(ctx.handles):
        s = ctx.profiles[handle].solved_idx
        rows.append(np.full(len(s), u))
        cols.append(s)
    rows = np.concatenate(rows) if rows else np.array([], dtype=int)
    cols = np.concatenate(cols) if cols else np.array([], dtype=int)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ctx.handles), ctx.n_items))


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class BaseRecommender:
    name = "base"
    description = ""

    def fit(self, ctx: RecContext):
        self.ctx = ctx
        return self

    def score_items(self, profile: Profile) -> np.ndarray:
        """Return one score per catalog problem (higher = better). Subclasses implement this."""
        raise NotImplementedError

    def recommend(self, profile: Profile, k: int = 10) -> list[int]:
        """Top-k catalog indices. Same filtering for every model."""
        cand = np.flatnonzero(self.ctx.candidate_mask(profile))
        if len(cand) == 0:
            return []
        scores = self.score_items(profile)[cand]
        # np.lexsort: LAST key is the main one -> score, then popularity, then index
        order = np.lexsort((cand, -self.ctx.popularity[cand], -scores))
        return cand[order[:k]].tolist()

    def explain(self, profile: Profile, item: int) -> str:
        return self.description


# --------------------------------------------------------------------------
# 1. Random baseline
# --------------------------------------------------------------------------
class RandomRecommender(BaseRecommender):
    name = "random"
    description = "Random pick among unsolved problems near your level."

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score_items(self, profile):
        return self.rng.random(self.ctx.n_items)


# --------------------------------------------------------------------------
# 2. Popularity
# --------------------------------------------------------------------------
class PopularityRecommender(BaseRecommender):
    name = "popularity"
    description = "Solved by many users, and near your level."

    def score_items(self, profile):
        return self.ctx.popularity.copy()


# --------------------------------------------------------------------------
# 3. Weak-tag heuristic
# --------------------------------------------------------------------------
class WeakTagRecommender(BaseRecommender):
    """
    For each tag (dp, graphs, ...) estimate how often THIS user solves problems
    with that tag:   accept_rate = (solved + prior * global_rate) / (attempted + prior)
    The "prior" part pulls rates of tags with few attempts towards the average
    over all users (otherwise 1 solved of 1 attempted would look like 100%).
    weakness = 1 - accept_rate.  A problem's score = average weakness of its tags.
    """
    name = "weak_tag"
    description = "Targets the tags where you solve the fewest problems."

    def fit(self, ctx):
        super().fit(ctx)
        tags = sorted({t for ts in ctx.tags for t in ts})
        self.tag_index = {t: j for j, t in enumerate(tags)}
        self.tag_names = tags
        rows, cols = [], []
        for i, ts in enumerate(ctx.tags):
            for t in ts:
                rows.append(i)
                cols.append(self.tag_index[t])
        T = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(ctx.n_items, len(tags)))
        counts = np.asarray(T.sum(axis=1)).ravel()
        counts[counts == 0] = 1
        self.T = (sparse.diags(1.0 / counts) @ T).tocsr()      # rows sum to 1 -> "average over tags"

        solved = np.zeros(len(tags))
        attempted = np.zeros(len(tags))
        for p in ctx.profiles.values():
            for t, (s, a) in p.tag_stats.items():
                j = self.tag_index.get(t)
                if j is not None:
                    solved[j] += s
                    attempted[j] += a
        self.global_rate = np.where(attempted > 0, solved / np.maximum(attempted, 1), 0.5)
        return self

    def tag_weakness(self, profile) -> np.ndarray:
        solved = np.zeros(len(self.tag_names))
        attempted = np.zeros(len(self.tag_names))
        for t, (s, a) in profile.tag_stats.items():
            j = self.tag_index.get(t)
            if j is not None:
                solved[j], attempted[j] = s, a
        prior = config.WEAKTAG_PRIOR
        rate = (solved + prior * self.global_rate) / (attempted + prior)
        return 1.0 - rate

    def score_items(self, profile):
        return self.T @ self.tag_weakness(profile)

    def explain(self, profile, item):
        weak = self.tag_weakness(profile)
        tags = self.ctx.tags[item]
        if not tags:
            return "Problem has no tags."
        tag = max(tags, key=lambda t: weak[self.tag_index[t]])
        s, a = profile.tag_stats.get(tag, [0, 0])
        return f"Targets your weak tag '{tag}' (you solved {s} of {a} attempted)."


# --------------------------------------------------------------------------
# 4. Item-based collaborative filtering
# --------------------------------------------------------------------------
class ItemCFRecommender(BaseRecommender):
    """
    "People who solved the problems you solved also solved ...".

    Similarity of problems i and j = cosine of their columns in the user x problem
    matrix (how much the two groups of solvers overlap, ignoring popularity size).
    Score of candidate c = sum over the user's solved problems j of sim(j, c).

    Trick: that sum equals  X[:, c] . v   with  v = sum of X[:, j] over solved j,
    so we never build the huge problems x problems similarity matrix.
    """
    name = "item_cf"
    description = "Users who solved the same problems as you also solved this."

    def fit(self, ctx):
        super().fit(ctx)
        R = _solved_matrix(ctx)
        norms = np.sqrt(np.asarray(R.sum(axis=0)).ravel())       # column L2 norm of a 0/1 column
        X = R @ sparse.diags(1.0 / np.maximum(norms, 1.0))        # normalise each problem column
        self.X = X.tocsc()
        self.Xt = X.T.tocsr()
        return self

    def score_items(self, profile):
        if len(profile.solved_idx) == 0:
            return np.zeros(self.ctx.n_items)
        v = np.asarray(self.X[:, profile.solved_idx].sum(axis=1)).ravel()   # one weight per user
        return self.Xt @ v


# --------------------------------------------------------------------------
# 5. Matrix factorisation (TruncatedSVD)
# --------------------------------------------------------------------------
class SVDRecommender(BaseRecommender):
    """
    Compress the user x problem matrix into ~50 hidden "taste" dimensions, then
    reconstruct it: problems that the low-rank model "expects" this user to
    have solved but they haven't get a high score.
    """
    name = "svd"
    description = "Matches your hidden solving 'taste' learned by matrix factorisation."

    def fit(self, ctx):
        super().fit(ctx)
        R = _solved_matrix(ctx)
        n = min(config.SVD_COMPONENTS, R.shape[0] - 1, R.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n, random_state=0).fit(R) if n >= 1 else None
        return self

    def score_items(self, profile):
        if self.svd is None or len(profile.solved_idx) == 0:
            return np.zeros(self.ctx.n_items)
        row = sparse.csr_matrix(
            (np.ones(len(profile.solved_idx)), (np.zeros(len(profile.solved_idx), dtype=int), profile.solved_idx)),
            shape=(1, self.ctx.n_items),
        )
        return (self.svd.transform(row) @ self.svd.components_).ravel()


# --------------------------------------------------------------------------
MODEL_CLASSES = {
    "random": RandomRecommender,
    "popularity": PopularityRecommender,
    "weak_tag": WeakTagRecommender,
    "item_cf": ItemCFRecommender,
    "svd": SVDRecommender,
}


def make_models(ctx: RecContext) -> dict[str, BaseRecommender]:
    """Create and fit every model on the same context."""
    return {name: cls().fit(ctx) for name, cls in MODEL_CLASSES.items()}
