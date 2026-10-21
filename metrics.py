"""
metrics.py - ranking metrics and the statistical comparison.

For one user we have:
  recs      the ordered list of k recommended problems
  relevant  the set of problems the user REALLY solved after the cutoff date
"""
import numpy as np
from scipy import stats


def precision_at_k(recs, relevant, k):
    """Of the k things we recommended, what fraction did the user really solve?"""
    return len(set(recs[:k]) & set(relevant)) / k


def recall_at_k(recs, relevant, k):
    """Of everything the user solved, what fraction did we manage to recommend?"""
    return len(set(recs[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs, relevant, k):
    """
    Like precision, but hits near the top of the list count more
    (a hit at rank r is worth 1/log2(r+1)). Divided by the best possible
    score, so 1.0 = perfect ranking.
    """
    relevant = set(relevant)
    dcg = sum(1.0 / np.log2(rank + 2) for rank, r in enumerate(recs[:k]) if r in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


METRICS = {"precision": precision_at_k, "recall": recall_at_k, "ndcg": ndcg_at_k}


def paired_comparison(a, b, n_boot=2000, seed=0):
    """
    Is model A really better than model B?  a[i], b[i] = score of the two models
    for the SAME user i. We look at the per-user differences d = a - b.

      * paired t-test : is the mean difference different from 0?
      * bootstrap CI  : resample users with replacement 2000 times, take the
                        mean difference each time -> a 95% confidence interval
                        that does not assume the data is bell-shaped.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n < 2:
        return dict(mean_diff=float(d.mean()) if n else 0.0, ci_low=np.nan, ci_high=np.nan,
                    t_stat=np.nan, p_value=np.nan)
    rng = np.random.default_rng(seed)
    boot_means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    if d.std() == 0:                                   # identical models -> nothing to test
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_1samp(d, 0.0)
    return dict(
        mean_diff=float(d.mean()),
        ci_low=float(np.percentile(boot_means, 2.5)),
        ci_high=float(np.percentile(boot_means, 97.5)),
        t_stat=float(t_stat),
        p_value=float(p_value),
    )
