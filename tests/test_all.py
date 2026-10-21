"""
Tests. Run with:  python -m pytest -q      (or: python tests/test_all.py)
They check the maths by hand-computed examples, protect against data leakage,
and smoke-test the whole pipeline on tiny synthetic data.
"""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evaluate
import make_synthetic_data
import metrics
import service as service_module
from data_fetcher import CFError, is_valid_handle, parse_submissions
from models import ItemCFRecommender, PopularityRecommender, RecContext, build_catalog


# ---------------------------------------------------------------- metrics
def test_precision_recall_ndcg_hand_computed():
    recs, relevant = [1, 2, 3], {2, 9}
    assert math.isclose(metrics.precision_at_k(recs, relevant, 3), 1 / 3)
    assert math.isclose(metrics.recall_at_k(recs, relevant, 3), 1 / 2)
    dcg = 1 / math.log2(3)                     # one hit at rank 2
    idcg = 1 + 1 / math.log2(3)                # two relevant items could fill ranks 1-2
    assert math.isclose(metrics.ndcg_at_k(recs, relevant, 3), dcg / idcg)


def test_ndcg_perfect_and_empty():
    assert metrics.ndcg_at_k([1, 2], {1, 2}, 2) == 1.0
    assert metrics.ndcg_at_k([5, 6], {1, 2}, 2) == 0.0
    assert metrics.recall_at_k([1], set(), 1) == 0.0


def test_paired_comparison_detects_real_difference_and_ties():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    assert metrics.paired_comparison(b + 0.2, b)["p_value"] < 1e-6
    same = metrics.paired_comparison(b, b)
    assert same["p_value"] == 1.0 and same["mean_diff"] == 0.0


# ---------------------------------------------------------------- security helpers
def test_handle_validation():
    assert is_valid_handle("tourist") and is_valid_handle("a.b-c_9")
    for bad in ["", "ab", "x" * 25, "a b", "a;b", "../etc", "x' OR 1=1 --", None]:
        assert not is_valid_handle(bad)


def test_parse_submissions_skips_gym_and_noise():
    raw = [
        {"id": 1, "problem": {"contestId": 10, "index": "A"}, "verdict": "OK", "creationTimeSeconds": 5},
        {"id": 2, "problem": {"contestId": 200000, "index": "A"}, "verdict": "OK", "creationTimeSeconds": 6},
        {"id": 3, "problem": {"contestId": 10, "index": "B"}, "verdict": "COMPILATION_ERROR", "creationTimeSeconds": 7},
        {"id": 4, "problem": {"contestId": 10, "index": "B"}, "creationTimeSeconds": 8},
    ]
    assert parse_submissions(raw, "h") == [(1, "h", "10A", "OK", 5)]


# ---------------------------------------------------------------- models on a tiny world
def tiny_world():
    problems = pd.DataFrame({
        "problem_id": [f"P{i}" for i in range(1, 7)],
        "contest_id": range(1, 7), "idx": "A", "name": "x", "rating": 900,
        "tags": [["dp"]] * 6,
    })
    solved = {"u1": ["P1", "P2"], "u2": ["P1", "P2", "P3"], "u3": ["P4", "P5"],
              "u4": ["P4", "P5"], "u5": ["P4", "P5"]}
    rows, t = [], 0
    for h, ps in solved.items():
        for p in ps:
            t += 1
            rows.append({"handle": h, "problem_id": p, "verdict": "OK", "created_at": t})
    return pd.DataFrame(rows), problems


def test_item_cf_recommends_what_similar_users_solved_but_popularity_does_not():
    subs, problems = tiny_world()
    ctx = RecContext(subs, build_catalog(subs, problems))
    u1 = ctx.profiles["u1"]
    cf = ItemCFRecommender().fit(ctx).recommend(u1, 1)
    pop = PopularityRecommender().fit(ctx).recommend(u1, 1)
    assert ctx.item_ids[cf[0]] == "P3"          # u2 overlaps with u1 and also solved P3
    assert ctx.item_ids[pop[0]] in {"P4", "P5"}  # most-solved unsolved problem


def test_recommendations_never_include_solved_problems():
    subs, problems = tiny_world()
    ctx = RecContext(subs, build_catalog(subs, problems))
    solved = set(ctx.profiles["u2"].solved_idx.tolist())
    for cls in (ItemCFRecommender, PopularityRecommender):
        recs = cls().fit(ctx).recommend(ctx.profiles["u2"], 10)
        assert not solved & set(recs)


# ---------------------------------------------------------------- evaluation
def test_time_split_has_no_overlap():
    subs, _ = tiny_world()
    train, test, cutoff = evaluate.time_split(subs, cutoff_ts=3)
    assert train["created_at"].max() < cutoff <= test["created_at"].min()


def test_catalog_excludes_problems_unseen_in_training():
    subs, problems = tiny_world()
    train = subs[subs["problem_id"] != "P6"]            # P6 only exists "in the future"
    assert "P6" not in build_catalog(train, problems)["problem_id"].tolist()


def test_full_pipeline_on_synthetic_data_and_service():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        make_synthetic_data.generate(path, n_users=120, n_problems=500, seed=1)
        out = evaluate.run_evaluation(path, k=10, out_dir=os.path.join(d, "res"), verbose=False)
        by_model = {r["model"]: r for r in out["results"]}
        assert set(by_model) == {"random", "popularity", "weak_tag", "item_cf", "svd"}
        assert by_model["item_cf"]["ndcg"] > by_model["random"]["ndcg"]   # synthetic world has real structure
        for f in ("results.csv", "significance.csv", "per_user.csv", "comparison.png", "results.json"):
            assert os.path.exists(os.path.join(d, "res", f))

        svc = service_module.RecommenderService(path).load()
        known = svc.ctx.handles[0]
        r = svc.recommend(known.upper(), "weak_tag", 5)          # case-insensitive lookup
        assert r["source"] == "database" and len(r["recommendations"]) == 5
        assert "weak tag" in r["recommendations"][0]["reason"]

        # unknown handle -> goes "live"; we fake the network call
        fake = pd.DataFrame({"problem_id": [svc.ctx.item_ids[0]], "verdict": ["OK"], "created_at": [1]})
        service_module.fetch_submissions_df = lambda h: fake
        assert svc.recommend("brand_new_user", "item_cf", 3)["source"] == "live"

        def not_found(h):
            raise CFError("handle not found")
        service_module.fetch_submissions_df = not_found
        for args, status in [(("bad handle!",), 422), (("zzz_unknown", "nope"), 422), (("zzz_unknown2",), 404)]:
            try:
                svc.recommend(*args)
                assert False, "should have raised"
            except service_module.ServiceError as e:
                assert e.status == status


if __name__ == "__main__":                   # tiny runner so tests work even without pytest
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as exc:
                failed += 1; print("FAIL", name, "->", repr(exc))
    sys.exit(1 if failed else 0)
