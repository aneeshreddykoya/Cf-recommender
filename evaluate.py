"""
evaluate.py - the honest evaluation. Run:  python evaluate.py

What it does
  1. Picks a cutoff date. Everything BEFORE it is "training" data, everything
     after is "the future" we try to predict.
  2. Fits all 5 models using ONLY the training data (no peeking).
  3. For each eligible user, every model recommends k problems.
  4. A recommendation is a HIT if the user really solved that problem after
     the cutoff. We compute Precision@k, Recall@k, NDCG@k per user.
  5. Compares models with a paired t-test + bootstrap confidence interval.

Outputs go to ./results : results.csv, significance.csv, per_user.csv,
comparison.png and results.json (served by the API at /evaluation).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")                    # draw charts without a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import db
from metrics import METRICS, paired_comparison
from models import RecContext, build_catalog, make_models


def time_split(subs: pd.DataFrame, cutoff_ts: int | None = None, quantile: float = 0.8):
    """Split submissions by TIME (never randomly - random splits leak the future)."""
    if cutoff_ts is None:
        cutoff_ts = int(subs.loc[subs["verdict"] == "OK", "created_at"].quantile(quantile))
    train = subs[subs["created_at"] < cutoff_ts]
    test = subs[subs["created_at"] >= cutoff_ts]
    return train, test, cutoff_ts


def build_eval_set(test: pd.DataFrame, ctx: RecContext, max_users: int | None = None, seed: int = 0):
    """
    handle -> set of catalog indices the user solved AFTER the cutoff.
    A user is evaluated only if they have enough history AND enough new solves.
    Problems they had already solved before the cutoff are not counted.
    """
    ok = test[(test["verdict"] == "OK") & test["problem_id"].isin(ctx.item_index)]
    eval_set = {}
    for handle, g in ok.groupby("handle", sort=True):
        prof = ctx.profiles.get(handle)
        if prof is None or len(prof.solved_idx) < config.MIN_TRAIN_SOLVES:
            continue
        already = set(prof.solved_idx.tolist())
        new = {ctx.item_index[p] for p in g["problem_id"].unique()} - already
        if len(new) >= config.MIN_TEST_SOLVES:
            eval_set[handle] = new
    if max_users and len(eval_set) > max_users:
        rng = np.random.default_rng(seed)
        keep = rng.choice(sorted(eval_set), size=max_users, replace=False)
        eval_set = {h: eval_set[h] for h in sorted(keep)}
    return eval_set


def score_models(ctx, models, eval_set, k):
    """Return a long DataFrame: one row per (model, user) with the three metrics."""
    rows = []
    for handle, relevant in eval_set.items():
        profile = ctx.profiles[handle]
        for name, model in models.items():
            recs = model.recommend(profile, k)
            row = {"model": name, "handle": handle}
            for metric_name, fn in METRICS.items():
                row[metric_name] = fn(recs, relevant, k)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_user: pd.DataFrame) -> pd.DataFrame:
    g = per_user.groupby("model")
    out = pd.DataFrame({m: g[m].mean() for m in METRICS})
    for m in METRICS:
        out[f"{m}_se"] = g[m].std(ddof=1) / np.sqrt(g[m].count())      # standard error
    return out.sort_values("ndcg", ascending=False).reset_index()


def significance_table(per_user: pd.DataFrame, references=("random", "popularity")) -> pd.DataFrame:
    """Every model vs the weak baseline (random) and the strong baseline (popularity)."""
    wide = {m: per_user.pivot(index="handle", columns="model", values=m) for m in METRICS}
    rows = []
    for metric, table in wide.items():
        for ref in references:
            for model in table.columns:
                if model == ref:
                    continue
                res = paired_comparison(table[model].to_numpy(), table[ref].to_numpy())
                rows.append({"metric": metric, "model": model, "vs": ref, "n_users": len(table), **res})
    return pd.DataFrame(rows)


def save_chart(summary: pd.DataFrame, k: int, path: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / len(METRICS)
    x = np.arange(len(summary))
    for j, metric in enumerate(METRICS):
        ax.bar(x + j * width, summary[metric], width, yerr=1.96 * summary[f"{metric}_se"],
               capsize=3, label=f"{metric.upper() if metric == 'ndcg' else metric.title()}@{k}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("score (higher is better)")
    ax.set_title(f"Model comparison on held-out future solves (bars: mean, whiskers: 95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(db_path=None, k=10, quantile=0.8, cutoff_ts=None, max_users=None,
                   seed=0, out_dir=None, verbose=True):
    out_dir = out_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    subs, problems = db.load_submissions(db_path), db.load_problems(db_path)
    if subs.empty or problems.empty:
        raise SystemExit("Database is empty. Run data_fetcher.py first (or make_synthetic_data.py).")

    train, test, cutoff_ts = time_split(subs, cutoff_ts, quantile)
    catalog = build_catalog(train, problems)            # only problems that existed at the cutoff
    ctx = RecContext(train, catalog)
    models = make_models(ctx)
    eval_set = build_eval_set(test, ctx, max_users, seed)
    if not eval_set:
        raise SystemExit("No eligible users. Fetch more users/history or lower MIN_* in config.py.")

    per_user = score_models(ctx, models, eval_set, k)
    summary = summarise(per_user)
    sig = significance_table(per_user)

    per_user.to_csv(os.path.join(out_dir, "per_user.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    sig.to_csv(os.path.join(out_dir, "significance.csv"), index=False)
    save_chart(summary, k, os.path.join(out_dir, "comparison.png"))

    payload = {
        "config": {
            "k": k,
            "cutoff_utc": pd.Timestamp(cutoff_ts, unit="s", tz="UTC").isoformat(),
            "train_submissions": int(len(train)), "test_submissions": int(len(test)),
            "catalog_problems": int(ctx.n_items), "users_in_training": len(ctx.handles),
            "users_evaluated": len(eval_set),
        },
        "results": summary.round(4).to_dict(orient="records"),
        "significance": sig.round(4).replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        c = payload["config"]
        print(f"Cutoff {c['cutoff_utc'][:10]} | {c['users_in_training']} users, "
              f"{c['catalog_problems']} problems, {c['users_evaluated']} users evaluated\n")
        show = summary[["model"] + list(METRICS)].copy()
        show.columns = ["model"] + [f"{m}@{k}" for m in METRICS]
        print(show.round(4).to_string(index=False))
        print(f"\nPaired comparison on NDCG@{k} (95% bootstrap CI of the difference):")
        s = sig[sig["metric"] == "ndcg"]
        for _, r in s.iterrows():
            print(f"  {r['model']:<11} vs {r['vs']:<10} diff={r['mean_diff']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p_value']:.4g}")
        print(f"\nSaved files in {out_dir}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--quantile", type=float, default=0.8,
                    help="cutoff = this quantile of solve times (0.8 -> last 20%% of time is the test set)")
    ap.add_argument("--cutoff", help="explicit cutoff date, e.g. 2024-06-01 (overrides --quantile)")
    ap.add_argument("--max-users", type=int, help="evaluate a random subset of users (faster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output folder (default: results)")
    a = ap.parse_args()
    cutoff = int(pd.Timestamp(a.cutoff, tz="UTC").timestamp()) if a.cutoff else None
    run_evaluation(a.db, a.k, a.quantile, cutoff, a.max_users, a.seed, a.out)
