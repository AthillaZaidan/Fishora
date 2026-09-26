import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.contracts import ImageValidationError
from apps.main_api.api.auth import router as auth_router
from apps.main_api.api.buyers import router as buyers_router
from apps.main_api.api.discover import router as discover_router
from apps.main_api.api.fish import knowledge_router, router as fish_router
from apps.main_api.api.jobs import router as jobs_router
from apps.main_api.api.lots import landing_router, router as lots_router
from apps.main_api.api.quality import router as quality_router
from apps.main_api.api.reviews import router as reviews_router
from apps.main_api.api.species import router as species_router
from apps.main_api.config import DEFAULT_CORS_ALLOW_ORIGINS, MainSettings, parse_origins
from apps.main_api.db.lot_repository import SqlLandingPointRepository, SqlLotRepository
from apps.main_api.db.preference_repository import SqlPreferenceRepository
from apps.main_api.db.repositories import TAXONOMY_STATUS_BY_LABEL, SqlKnowledgeRepository
from apps.main_api.db.session import session_factory
from apps.main_api.db.sql_repositories import SqlKnowledgeJobRepository, SqlPredictionRepository, SqlSpeciesRepository
from apps.main_api.db.user_repository import SqlUserRepository
from apps.main_api.errors import (
    RetrievalUnavailable,
    BidOutbid,
    CvUnavailable,
    Forbidden,
    InvalidGeneratedKnowledge,
    InvalidLot,
    LotAlreadyPublished,
    LotClosed,
    LotNotAllocatable,
    LotNotFound,
    OpenCodeUnavailable,
    PredictionNotFound,
    PredictionNotVerified,
    Unauthenticated,
    PhotoRejected,
    UnsupportedCvLabel,
    UnsupportedSpecies,
)
from apps.main_api.ports import AppDependencies
from apps.main_api.services.cv_client import HttpCVClient
from apps.main_api.services.embeddings import LocalE5Embedder
from apps.main_api.services.image_store import FilesystemImageStore
from apps.main_api.services.session import SessionService, seed_demo_users


def create_main_app(settings: MainSettings | None = None, deps: AppDependencies | None = None) -> FastAPI:
    """Main API factory.

    Tests inject every external port through `deps`; the production factory
    leaves them None and wires the SQLAlchemy repositories, HTTP CV client,
    and filesystem image store lazily in the lifespan, so importing this
    module never connects to Postgres or any network service. Settings and
    production ports are instantiated only when actually missing: a complete
    fake bundle (all five ports injected) never constructs MainSettings, never
    reads environment variables, and never creates a DB session factory.
    """
    deps = deps or AppDependencies()
    app = FastAPI(lifespan=_lifespan)
    app.state.settings = settings  # may be None when all ports are injected
    app.state.deps = deps
    # Built now, not in the lifespan: a complete fake bundle never gets past the
    # lifespan's first check, and still needs sign-in. The lifespan rebuilds a
    # service it made itself once settings and the users table are known.
    app.state.owns_session_service = deps.session_service is None
    if deps.session_service is None:
        deps.session_service = _session_service(settings, deps.user_repo)
    _register_cors(app, settings)
    _register_health(app)
    _register_error_handlers(app)
    app.include_router(fish_router)
    app.include_router(knowledge_router)
    app.include_router(lots_router)
    app.include_router(landing_router)
    app.include_router(buyers_router)
    app.include_router(auth_router)
    app.include_router(discover_router)
    app.include_router(reviews_router)
    app.include_router(jobs_router)
    app.include_router(species_router)
    app.include_router(quality_router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _ensure_production_deps(app)
    _warm_embedder(app)
    yield


def _warm_embedder(app: FastAPI) -> None:
    """Load the embedding model in the background at startup, so the first
    knowledge request does not wait for it (about 11 s cold, W7). Startup does
    not block, and a failure only means the first request loads it instead."""
    embedder = getattr(app.state.deps, "embedder", None)
    warmup = getattr(embedder, "warmup", None)
    if not callable(warmup):
        return

    def run():
        try:
            warmup()
        except Exception:
            logging.getLogger(__name__).warning("embedder warmup failed; the first request will load it", exc_info=True)

    threading.Thread(target=run, name="embedder-warmup", daemon=True).start()


def _ensure_production_deps(app: FastAPI) -> None:
    """Fill any un-injected port with the production implementation (idempotent).

    The five concrete ports are the completeness criterion: when cv_client,
    species_repo, prediction_repo, image_store, and embedder are all injected,
    nothing is missing. A session factory is only a means to build the SQL
    repos, not an end port, so a complete fake bundle never constructs
    MainSettings, a DB session factory, a network client, or an embedder —
    even when the lifespan runs. The production embedder is constructed
    without loading weights: LocalE5Embedder stays lazy until first embed.
    """
    deps = app.state.deps
    complete = (
        deps.cv_client is not None
        and deps.species_repo is not None
        and deps.prediction_repo is not None
        and deps.image_store is not None
        and deps.embedder is not None
    )
    if complete:
        return
    settings = app.state.settings or MainSettings()
    # Stored so routes read configured values, not the class defaults.
    app.state.settings = settings
    if deps.session_factory is None:
        deps.session_factory = session_factory(settings)
    if deps.cv_client is None:
        deps.cv_client = HttpCVClient(settings.cv_service_url, settings.cv_timeout_seconds)
    if deps.species_repo is None:
        deps.species_repo = SqlSpeciesRepository(deps.session_factory)
    if deps.prediction_repo is None:
        deps.prediction_repo = SqlPredictionRepository(deps.session_factory)
    if deps.image_store is None:
        deps.image_store = FilesystemImageStore(settings.image_storage_dir)
    if deps.embedder is None:
        deps.embedder = LocalE5Embedder(
            settings.embedding_model_name,
            device=settings.embedding_device,
        )
    if deps.knowledge_repo is None:
        deps.knowledge_repo = SqlKnowledgeRepository(deps.session_factory)
    if deps.lot_repo is None:
        deps.lot_repo = SqlLotRepository(deps.session_factory)
    if deps.review_repo is None:
        from apps.main_api.db.review_repository import SqlReviewRepository

        deps.review_repo = SqlReviewRepository(deps.session_factory)
    if deps.landing_point_repo is None:
        deps.landing_point_repo = SqlLandingPointRepository(deps.session_factory)
        from apps.main_api.services.landing_points import seed_demo_landing_points

        seed_demo_landing_points(deps.landing_point_repo)
    if deps.preference_repo is None:
        deps.preference_repo = SqlPreferenceRepository(deps.session_factory)
    if deps.user_repo is None:
        deps.user_repo = SqlUserRepository(deps.session_factory)
        try:
            seed_demo_users(deps.user_repo)
        except Exception:
            # Startup goes on so the rest of the API works; sign-in fails until
            # the users table exists.
            logging.getLogger(__name__).exception("could not seed the demo accounts; run `alembic upgrade head`")
    if getattr(app.state, "owns_session_service", False):
        deps.session_service = _session_service(settings, deps.user_repo)
    if getattr(deps, "job_repo", None) is None:
        try:
            deps.job_repo = SqlKnowledgeJobRepository(deps.session_factory)
        except Exception:
            # Without it cards are made synchronously on request; say so.
            logging.getLogger(__name__).exception("knowledge job repository unavailable")


def _session_service(settings: MainSettings | None, users) -> SessionService:
    """No settings means a fake bundle: a per-process secret and, without a
    user repository, in-memory accounts seeded with the demo pair."""
    if settings is None:
        return SessionService(users=users)
    return SessionService(
        settings.session_secret.get_secret_value() or None,
        users,
        cookie_secure=settings.session_cookie_secure,
    )


def _register_health(app: FastAPI) -> None:
    """Liveness plus whether the taxonomy is seeded.

    An unseeded database is a distinct failure mode: the API answers, but every
    identification and manual declaration fails on species resolution. Callers
    that need to tell "down" from "not ready" cannot do it from a bare 200.
    """

    @app.get("/health")
    async def health(request: Request):
        repo = request.app.state.deps.species_repo
        seeded = False
        if repo is not None:
            seeded = any(
                repo.get_by_normalized_label(label) is not None
                for label in TAXONOMY_STATUS_BY_LABEL
            )
        return {"status": "ok", "taxonomy_seeded": seeded}


def _register_cors(app: FastAPI, settings: MainSettings | None) -> None:
    """Allow the frontend's origin to reach this API from a browser."""
    origins = (
        settings.cors_origins
        if settings is not None
        else parse_origins(DEFAULT_CORS_ALLOW_ORIGINS)
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["content-type", "authorization"],
    )


def _register_error_handlers(app: FastAPI) -> None:
    # Fixed generic details only: never echo an internal CV URL, credentials, or headers.
    @app.exception_handler(ImageValidationError)
    async def _image_validation(request: Request, exc: ImageValidationError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(CvUnavailable)
    async def _cv_unavailable(request: Request, exc: CvUnavailable):
        return JSONResponse(status_code=503, content={"detail": "fish identification service is temporarily unavailable"})

    @app.exception_handler(PredictionNotFound)
    async def _prediction_not_found(request: Request, exc: PredictionNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedSpecies)
    async def _unsupported_species(request: Request, exc: UnsupportedSpecies):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PhotoRejected)
    async def _photo_rejected(request: Request, exc: PhotoRejected):
        # The detail text is what the frontend keys on (lib/api/errors.ts).
        detail = {"not_fish": "photo rejected: not a fish",
                  "unknown_species": "photo rejected: unknown species"}.get(exc.reason, "photo rejected")
        return JSONResponse(status_code=422, content={"detail": detail, "reason": exc.reason})

    @app.exception_handler(UnsupportedCvLabel)
    async def _unsupported_cv_label(request: Request, exc: UnsupportedCvLabel):
        return JSONResponse(status_code=502, content={"detail": "cv returned an unsupported species label"})

    @app.exception_handler(PredictionNotVerified)
    async def _prediction_not_verified(request: Request, exc: PredictionNotVerified):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(OpenCodeUnavailable)
    async def _opencode_unavailable(request: Request, exc: OpenCodeUnavailable):
        # Generic detail plus chunk ids only, never credentials or internal URLs.
        return JSONResponse(status_code=502, content={
            "detail": "knowledge generation is temporarily unavailable",
            "retrieved_chunk_ids": exc.retrieved_chunk_ids,
        })

    @app.exception_handler(RetrievalUnavailable)
    async def _retrieval_unavailable(request: Request, exc: RetrievalUnavailable):
        # Generic detail: the message names an internal package, which belongs
        # in the server log rather than in a client response.
        return JSONResponse(status_code=502, content={
            "detail": "knowledge retrieval is temporarily unavailable",
        })

    @app.exception_handler(InvalidGeneratedKnowledge)
    async def _invalid_generated_knowledge(request: Request, exc: InvalidGeneratedKnowledge):
        return JSONResponse(status_code=502, content={
            "detail": "generated knowledge failed validation",
            "retrieved_chunk_ids": exc.retrieved_chunk_ids,
        })

    @app.exception_handler(InvalidLot)
    async def _invalid_lot(request: Request, exc: InvalidLot):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(LotNotFound)
    async def _lot_not_found(request: Request, exc: LotNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(LotAlreadyPublished)
    async def _lot_already_published(request: Request, exc: LotAlreadyPublished):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(LotClosed)
    async def _lot_closed(request: Request, exc: LotClosed):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(BidOutbid)
    async def _bid_outbid(request: Request, exc: BidOutbid):
        return JSONResponse(status_code=409, content={
            "detail": "bid must exceed current highest",
            "current_highest_per_kg": str(exc.current_highest_per_kg),
        })

    @app.exception_handler(LotNotAllocatable)
    async def _lot_not_allocatable(request: Request, exc: LotNotAllocatable):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(Unauthenticated)
    async def _unauthenticated(request: Request, exc: Unauthenticated):
        return JSONResponse(status_code=401, content={"detail": "authentication required"})

    @app.exception_handler(Forbidden)
    async def _forbidden(request: Request, exc: Forbidden):
        return JSONResponse(status_code=403, content={"detail": str(exc)})


_app: FastAPI | None = None


def __getattr__(name: str):
    # Lazy module-level app: importing never needs env vars or a database.
    if name == "app":
        global _app
        if _app is None:
            _app = create_main_app()
        return _app
    raise AttributeError(name)
