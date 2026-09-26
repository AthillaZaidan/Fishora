import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from apps.common.image import validate_image_bytes
from apps.contracts import CVPredictionEnvelope, ImageValidationError
from apps.cv_service.config import CVSettings
from apps.cv_service.runtime import ClassifierProtocol, load_classifier

logger = logging.getLogger(__name__)


def create_cv_app(settings: CVSettings | None = None, classifier: ClassifierProtocol | None = None) -> FastAPI:
    settings = settings or CVSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _ensure_classifier(app)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.classifier = classifier

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_version": app.state.settings.model_version}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            image = validate_image_bytes(await file.read(), file.content_type, app.state.settings.max_image_bytes)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message)
        try:
            result = _ensure_classifier(app).predict(image, top_k=3)
        except RuntimeError as exc:
            # A GPU can fail mid-session ("CUDA error: unknown error" on a laptop
            # after hours of use) and every later call fails the same way. Reload
            # on the CPU (about a second per photo) rather than refusing photos.
            if "cuda" not in str(exc).lower() or app.state.settings.device == "cpu":
                raise
            logger.exception("GPU inference failed; reloading the classifier on the CPU")
            app.state.settings = app.state.settings.model_copy(update={"device": "cpu"})
            app.state.classifier = None
            result = _ensure_classifier(app).predict(image, top_k=3)
        result["model_version"] = app.state.settings.model_version
        return CVPredictionEnvelope(**result)

    return app


def _ensure_classifier(app: FastAPI) -> ClassifierProtocol:
    # Loaded at most once per process: injected in tests, otherwise in lifespan.
    if app.state.classifier is None:
        app.state.classifier = load_classifier(app.state.settings.export_dir, app.state.settings.device)
    return app.state.classifier


_app = None


def __getattr__(name: str):
    # ponytail: lazy module-level app; importing this module never needs env vars or GPU artifacts
    if name == "app":
        global _app
        if _app is None:
            _app = create_cv_app()
        return _app
    raise AttributeError(name)