"""
config.py - every tunable number lives here, so you can explain each one.
"""
import os

# --- Paths -----------------------------------------------------------------
# You can point the whole project at a different database with an env var:
#   PowerShell:  $env:CF_DB="cf_demo.db"
#   cmd.exe:     set CF_DB=cf_demo.db
DB_PATH = os.environ.get("CF_DB", "cf_data.db")
RESULTS_DIR = os.environ.get("CF_RESULTS", "results")

# --- Codeforces API --------------------------------------------------------
CF_API_URL = "https://codeforces.com/api/"
CF_MIN_INTERVAL = 2.1      # Codeforces asks for at most 1 request per 2 seconds
CF_TIMEOUT = 60            # seconds before we give up on a request
CF_RETRIES = 3

# Codeforces handles: 3-24 characters of letters, digits, _ - .
# We validate every handle against this BEFORE it goes near a URL or SQL.
HANDLE_RE = r"^[A-Za-z0-9_.\-]{3,24}$"

# --- Recommender settings --------------------------------------------------
DEFAULT_LEVEL = 800        # assumed rating for a user with no solved problems
RECENT_SOLVES = 30         # a user's "level" = median rating of their last 30 solves
BAND_LOW = -200            # candidates must be rated within [level-200, level+300]
BAND_HIGH = 300            # (same band for EVERY model, so the comparison is fair)
WEAKTAG_PRIOR = 5          # smoothing strength for tag accept rates (see models.py)
SVD_COMPONENTS = 50        # number of latent factors for the SVD model

# --- Evaluation settings ---------------------------------------------------
MIN_TRAIN_SOLVES = 20      # a user needs >= 20 solves before the cutoff to be evaluated
MIN_TEST_SOLVES = 5        # ...and >= 5 NEW solves after the cutoff
