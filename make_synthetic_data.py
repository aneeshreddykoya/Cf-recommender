"""
make_synthetic_data.py - FAKE Codeforces-like data, for testing the pipeline offline.

!!! Results on this data prove NOTHING about real users. It only checks that your
!!! code runs end to end. Never put numbers from it on a resume.

    python make_synthetic_data.py            # writes cf_demo.db
    python make_synthetic_data.py --db other.db --users 300

How the fake world works (so the models have something to discover):
  * 5 "topics"; each problem belongs to one topic and has topic-related tags
  * each user has a skill level and a hidden liking for each topic
  * users mostly attempt problems near their level, in topics they like
  * harder / disliked-topic problems fail more often
"""
import argparse
import os

import numpy as np

import db

TOPIC_TAGS = [
    ["dp", "math", "number theory"],
    ["graphs", "dfs and similar", "trees"],
    ["greedy", "sortings", "constructive algorithms"],
    ["strings", "hashing", "string suffix structures"],
    ["data structures", "binary search", "two pointers"],
]
T_END = 1_735_689_600            # 2025-01-01
SPAN = 3 * 365 * 24 * 3600       # three years of history


def generate(path: str, n_users: int = 300, n_problems: int = 1500, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_topics = len(TOPIC_TAGS)

    # ---- problems -------------------------------------------------------
    levels = np.arange(800, 3300, 100)
    probs = np.exp(-(levels - 800) / 700.0)
    ratings = rng.choice(levels, size=n_problems, p=probs / probs.sum())
    topic = rng.integers(0, n_topics, n_problems)
    pop = rng.lognormal(0, 0.6, n_problems)
    release = T_END - SPAN + (np.arange(n_problems) / n_problems) * SPAN * 0.95
    prob_rows = []
    for i in range(n_problems):
        tags = list(rng.choice(TOPIC_TAGS[topic[i]], size=rng.integers(1, 3), replace=False))
        contest, letter = 1000 + i // 6, "ABCDEF"[i % 6]
        prob_rows.append((f"{contest}{letter}", contest, letter, f"Problem {contest}{letter}",
                          int(ratings[i]), ",".join(tags)))
    pids = [r[0] for r in prob_rows]

    # ---- users and submissions -----------------------------------------
    user_rows, sub_rows, sid = [], [], 1
    for u in range(n_users):
        handle = f"user_{u:04d}"
        ability = float(np.clip(rng.normal(1500, 400), 800, 2900))
        taste = rng.dirichlet(np.full(n_topics, 0.5)) * n_topics        # mean 1
        join = rng.uniform(T_END - SPAN, T_END - SPAN * 0.4)
        n_attempts = int(rng.integers(80, 450))
        times = np.sort(rng.uniform(join, T_END, n_attempts))
        done = set()
        user_rows.append((handle, int(ability), 1))
        for t in times:
            level_now = ability - 200 * (T_END - t) / SPAN * 2          # skill grows over time
            avail = release < t
            w = (pop * taste[topic] * np.exp(-((ratings - (level_now + 100)) ** 2) / (2 * 250.0 ** 2)) * avail)
            for d in done:
                w[d] = 0
            if w.sum() <= 0:
                continue
            i = int(rng.choice(n_problems, p=w / w.sum()))
            p_solve = 1 / (1 + np.exp(-((level_now - ratings[i]) / 250 + 1.2 * (taste[topic[i]] - 1))))
            for _ in range(int(rng.geometric(0.6)) - 1):                  # some wrong tries first
                sub_rows.append((sid, handle, pids[i], "WRONG_ANSWER", int(t))); sid += 1; t += 60
            if rng.random() < p_solve:
                sub_rows.append((sid, handle, pids[i], "OK", int(t))); sid += 1
                done.add(i)
            else:
                sub_rows.append((sid, handle, pids[i], "WRONG_ANSWER", int(t))); sid += 1

    if os.path.exists(path):
        os.remove(path)
    with db.connect(path) as conn:
        conn.executemany("INSERT INTO problems VALUES (?,?,?,?,?,?)", prob_rows)
        conn.executemany("INSERT INTO users VALUES (?,?,?)", user_rows)
        conn.executemany("INSERT INTO submissions VALUES (?,?,?,?,?)", sub_rows)
    print(f"Wrote {path}: {n_problems} problems, {n_users} users, {len(sub_rows)} submissions (SYNTHETIC).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cf_demo.db")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--problems", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.db, a.users, a.problems, a.seed)
