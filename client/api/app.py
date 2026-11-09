from copy import deepcopy
import re
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .contracts import Health, Summary, ContextPage, ContextDetail, SourceResult, ClaimRating, RatingUpdate
from .provenance import assemble_provenance
from .repository import Repository, DatabaseUnavailable, ContextNotFound
from .sources import SourceService, SourceBusy
from .ratings import RatingRepository, RatingsUnavailable


def create_app(settings=None, repository=None, provider=None):
    settings = settings or Settings.from_environment()
    repo = repository or Repository(settings)
    sources = SourceService(settings, provider)
    ratings = RatingRepository(settings)
    app = FastAPI(title="dgiLIT · Claim Explorer API", version="1.0.0")
    app.state.repository = repo
    app.state.sources = sources
    app.state.ratings = ratings
    summary_lock = threading.Lock()
    summary_cache = {"at": 0, "value": None}

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DatabaseUnavailable)
    def database_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": "database_unavailable"})

    @app.exception_handler(ContextNotFound)
    def missing_context(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc), "code": "context_not_found"})

    @app.exception_handler(SourceBusy)
    def busy_source(request, exc):
        return JSONResponse(status_code=429, content={"detail": str(exc), "code": "source_busy"})

    @app.exception_handler(RatingsUnavailable)
    def ratings_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": "ratings_unavailable"})

    def detail(task_id):
        if not re.fullmatch(r"[a-f0-9]{64}", task_id):
            raise ContextNotFound("This positive context was not found.")
        value = assemble_provenance(repo.detail(task_id), settings)
        value.update(ratings.get(task_id))
        return value

    @app.get("/api/health", response_model=Health)
    def health():
        return repo.health()

    @app.get("/api/summary", response_model=Summary)
    def summary(refresh: bool = False):
        with summary_lock:
            if refresh or summary_cache["value"] is None or time.monotonic() - summary_cache["at"] >= settings.summary_ttl:
                summary_cache.update(at=time.monotonic(), value=repo.summary())
            return deepcopy(summary_cache["value"])

    @app.get("/api/contexts", response_model=ContextPage)
    def contexts(drug: str = Query("", max_length=200), gene: str = Query("", max_length=200),
                 article: str = Query("", max_length=300), cohort: Literal["all", "pilot", "beyond_pilot"] = "all",
                 coverage: Literal["all", "full_text", "abstract_only"] = "all",
                 rating: Literal["all", "up", "hide_down"] = "all",
                 page: int = Query(1, ge=1, le=10000000), page_size: int = Query(50, ge=1, le=100)):
        value = repo.contexts(drug=drug.strip(), gene=gene.strip(), article=article.strip(), cohort=cohort,
                             coverage=coverage, rating_filter=rating, review_db=settings.review_db,
                             page=page, page_size=page_size)
        page_ratings = ratings.get_many([row["task_id"] for row in value["items"]])
        for row in value["items"]:
            row.update(page_ratings.get(row["task_id"], {"rating": None, "rated_at": None}))
        return value

    @app.get("/api/contexts/{task_id}", response_model=ContextDetail)
    def context_detail(task_id: str):
        return detail(task_id)

    @app.put("/api/contexts/{task_id}/rating", response_model=ClaimRating)
    def rate_claim(task_id: str, assessment: RatingUpdate):
        # The extraction repository validates the claim using a read-only
        # connection, which closes before the app-owned database is written.
        if not re.fullmatch(r"[a-f0-9]{64}", task_id):
            raise ContextNotFound("This positive context was not found.")
        repo.detail(task_id)
        return ratings.set(task_id, assessment.rating)

    @app.get("/api/contexts/{task_id}/source", response_model=SourceResult)
    def cached_source(task_id: str):
        try:
            return sources.cached(detail(task_id)["task"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/contexts/{task_id}/source", response_model=SourceResult)
    def retrieve_source(task_id: str):
        # Load/close the read transaction before any network request starts.
        task = detail(task_id)["task"]
        try:
            return sources.fetch(task)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
