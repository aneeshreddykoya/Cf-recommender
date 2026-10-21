# Codeforces Problem Recommender with Honest Evaluation

Recommends Codeforces problems to a user and **measures, on real held-out data,
which recommendation method works best**. Built with Python, SQLite, scikit-learn,
SciPy and FastAPI. Requires Python 3.10+.

> **Rule for this project:** only quote numbers that `evaluate.py` produced on
> **real** Codeforces data. The synthetic demo data is only for checking that the code runs.

## 1. Setup (Windows 11 / VS Code terminal)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Quick demo on FAKE data (1 minute, no internet)

```bash
python make_synthetic_data.py          # creates cf_demo.db
python evaluate.py --db cf_demo.db     # prints the model comparison table
```

## 3. The real thing: download real Codeforces data

```bash
python data_fetcher.py all --n 400     # ~20-25 minutes. Safe to stop (Ctrl+C) and re-run: it resumes.
python evaluate.py                     # results in ./results
```
Codeforces allows 1 request per 2 seconds, so the downloader waits between calls.
Use `--n 120` first if you want a faster first run. More users = more trustworthy results.

## 4. Run the API

```bash
python -m uvicorn main:app --reload
```
(To use the demo data: PowerShell `$env:CF_DB="cf_demo.db"` or cmd `set CF_DB=cf_demo.db` before starting.)

| Endpoint | What it does |
|---|---|
| `/recommend/{handle}?model=item_cf&k=10` | Recommendations + a reason for each. Models: `random`, `popularity`, `weak_tag`, `item_cf`, `svd` |
| `/evaluation` | The real results table produced by `evaluate.py` |
| `/models`, `/health`, `/docs` | Model list, status, interactive docs |

If the handle isn't in your database, the API downloads that user's submissions live
from Codeforces (cached 10 minutes) and recommends anyway.

## 5. Tests
```bash
python -m pytest -q
```

## How it works (read this before any interview)

| File | Purpose |
|---|---|
| `data_fetcher.py` | Downloads problems, a rating-stratified sample of users, and their submissions into SQLite. Rate-limited, retrying, resumable. |
| `db.py`, `config.py` | Table definitions; all tunable numbers in one place. |
| `models.py` | The 5 recommenders. |
| `metrics.py` | Precision@k, Recall@k, NDCG@k, paired t-test, bootstrap confidence interval. |
| `evaluate.py` | Time-based train/test split, runs every model, writes results + chart. |
| `service.py`, `main.py` | Recommendation logic and the thin FastAPI layer. |
| `make_synthetic_data.py` | Fake data for offline testing only. |

**The 5 models.** All of them use the same rules first: never recommend a solved problem, and only
consider problems rated between *level - 200* and *level + 300* (level = median rating of the user's
last 30 solves). They differ only in how they rank what is left:
1. **random** - the floor every model must beat.
2. **popularity** - most-solved problems first (a strong, simple baseline).
3. **weak_tag** - problems whose tags the user is weakest at (smoothed accept rate per tag).
4. **item_cf** - "users who solved what you solved also solved this" (cosine similarity between problems).
5. **svd** - matrix factorisation (`TruncatedSVD`) learning ~50 hidden taste dimensions.

**How the evaluation avoids cheating**
- *Time split:* train on submissions before a cutoff date, test on what users solved after it. A random split would let models "see the future".
- *Catalog:* only problems that already appeared in the training data can be recommended.
- *Fair comparison:* identical filtering for all models; only the scoring differs.
- *Hit definition:* a recommendation is a hit if the user truly solved that problem after the cutoff.
- *Statistics:* per-user paired differences, a paired t-test and a bootstrap 95% CI. Compared against both `random` and `popularity`.

## Questions you should be able to answer
- **Why a time split instead of random?** To avoid data leakage; real recommenders predict the future.
- **What is implicit feedback?** We only know what users solved, not what they disliked; a missing solve isn't a "no".
- **What is the cold-start problem?** New users or problems have no history; we fall back to a default level and popularity tie-breaks.
- **Why NDCG as well as precision?** It rewards putting hits near the top of the list.
- **Why compare against popularity, not just random?** Beating random is easy; popularity is the honest baseline.
- **What if the weak-tag model loses?** Report it. A negative result with a good explanation is still a good result.

## Limitations (say these yourself, before an interviewer finds them)
- A user "solving a problem later" is not proof the recommendation *caused* it, and unsolved ≠ disliked.
  Absolute scores are low for this reason; compare models *relative* to each other.
- Users are a sample of ~400 active rated users, so results may not generalise to everyone.
- The band filter means problems far outside a user's level can never be hit.
- Multiple comparisons: with several models and metrics, treat borderline p-values (e.g. 0.03) cautiously.
- Offline evaluation can't replace a live A/B test.

## Security notes (relevant to an information-security team)
- Every handle is validated against a strict allow-list regex before use.
- Outbound requests use `params=` (URL-encoded), timeouts, retries with back-off, and a global rate limiter.
- All SQL is parameterised (`?` placeholders); no string-built queries.
- API inputs are bounded (`k` ≤ 50); live downloads are cached with a size cap; errors return safe messages, not stack traces.
- No secrets or API keys are used or stored.
- Not implemented (good "next steps"): per-client rate limiting, authentication, HTTPS/reverse proxy.

## Resume bullets (fill in X, Y, N from YOUR real run)
- Built a Codeforces problem recommender (FastAPI, SQLite, scikit-learn) on N submissions from M users; compared random, popularity, tag-based, item-based CF and SVD models.
- Evaluated with a time-based split and Precision/Recall/NDCG@10; best model improved NDCG@10 from X to Y over the popularity baseline (paired t-test, p = Z).
- Added input validation, rate-limited API access, parameterised SQL and unit tests.
