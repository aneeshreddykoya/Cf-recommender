"""
main.py - the FastAPI web server (thin wrapper around service.py).

Start:  python -m uvicorn main:app --reload
Docs :  http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from models import MODEL_CLASSES
from service import RecommenderService, ServiceError

service = RecommenderService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data and fitting models...")
    service.load()
    print("Ready." if service.ready else "WARNING: database is empty - run data_fetcher.py first.")
    yield


app = FastAPI(title="Codeforces Problem Recommender", lifespan=lifespan)


def _run(fn, *args, **kwargs):
    """Turn our ServiceError into a proper HTTP error (never leak internal details)."""
    try:
        return fn(*args, **kwargs)
    except ServiceError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": service.ready}


@app.get("/models")
def list_models():
    return {name: cls.description or cls.__doc__.strip().splitlines()[0] for name, cls in MODEL_CLASSES.items()}


@app.get("/recommend/{handle}")
def recommend(handle: str, model: str = "item_cf", k: int = Query(10, ge=1, le=50)):
    """Recommend k unsolved problems near the user's level. model: random | popularity | weak_tag | item_cf | svd"""
    return _run(service.recommend, handle, model, k)


@app.get("/evaluation")
def evaluation():
    """The REAL offline-evaluation results produced by evaluate.py."""
    return _run(service.evaluation_results)
