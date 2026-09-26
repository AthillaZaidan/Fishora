import logging
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from apps.main_api.config import MainSettings
from apps.main_api.services.generation import KnowledgeResponse
from apps.main_api.services.identification import IdentificationService
from apps.main_api.services.manual_entry import ManualEntryService
from apps.main_api.services.knowledge import KnowledgeService
from apps.main_api.services.verification import VerificationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fish")
knowledge_router = APIRouter(prefix="/api/v1")

# ponytail: fallback limit when a complete fake bundle supplies no settings object;
# reads the configured default without constructing MainSettings (no env required).
DEFAULT_MAX_IMAGE_BYTES = MainSettings.model_fields["cv_max_image_bytes"].default


def _card_llm(request: Request, session_id: str):
    """The LLM for one card's agent graph: the injected port when present
    (tests, alternative providers; W2), else the OpenCode Go client built from
    settings with this card's session id. None when no key is configured, in
    which case the experts claim nothing and the writer returns an empty card."""
    deps = request.app.state.deps
    if getattr(deps, "llm", None) is not None:
        return deps.llm
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not settings.opencode_go_api_key.get_secret_value().strip():
        return None
    from apps.main_api.services.generation import make_opencode_go_llm

    return make_opencode_go_llm(settings, session_id=session_id)


class VerifyRequest(BaseModel):
    prediction_id: str
    verified_species_id: str


class SpeciesCandidate(BaseModel):
    species_id: str
    normalized_label: str
    confidence: float = Field(ge=0.0, le=1.0)


class IdentificationResponse(BaseModel):
    prediction_id: str
    model_version: str
    status: Literal["confident_prediction", "low_confidence_human_verification_required"]
    prediction: SpeciesCandidate
    top_candidates: list[SpeciesCandidate]
    threshold: float
    verification_status: Literal["pending"]


class ManualEntryResponse(BaseModel):
    prediction_id: str
    model_version: str
    verified_species_id: str
    normalized_label: str
    verification_status: Literal["confirmed", "corrected"]


class VerificationResponse(BaseModel):
    prediction_id: str
    predicted_species_id: str
    verified_species_id: str
    verification_status: Literal["confirmed", "corrected"]


@router.post("/identify", response_model=IdentificationResponse)
async def identify(request: Request, file: UploadFile = File(...)):
    deps = request.app.state.deps
    settings = request.app.state.settings  # None when a complete fake bundle is injected
    service = IdentificationService(
        cv_client=deps.cv_client,
        species_repo=deps.species_repo,
        prediction_repo=deps.prediction_repo,
        image_store=deps.image_store,
        max_image_bytes=settings.cv_max_image_bytes if settings is not None else DEFAULT_MAX_IMAGE_BYTES,
    )
    result = service.identify(
        await file.read(),
        filename=file.filename or "image",
        content_type=file.content_type,
    )
    return IdentificationResponse(
        prediction_id=result.prediction_id,
        model_version=result.model_version,
        status=result.status,
        prediction=SpeciesCandidate(**asdict(result.prediction)),
        top_candidates=[SpeciesCandidate(**asdict(candidate)) for candidate in result.top_candidates],
        threshold=result.threshold,
        verification_status=result.verification_status,
    )


@router.post("/verify", response_model=VerificationResponse)
async def verify(payload: VerifyRequest, request: Request, background_tasks: BackgroundTasks):
    deps = request.app.state.deps
    result = VerificationService(
        species_repo=deps.species_repo,
        prediction_repo=deps.prediction_repo,
    ).verify(payload.prediction_id, payload.verified_species_id)
    job_repo = getattr(deps, "job_repo", None)
    if job_repo is not None:
        # Verification has already succeeded; a scheduling failure must not undo
        # it, but it is logged instead of swallowed (W12).
        try:
            # Iteration 2: the writer-critic workflow won the 2x2 against the
            # 4-expert graph (evals/iteration2.py); run_graph stays for comparison.
            from apps.main_api.services.workflow import run_workflow

            job = job_repo.create(result.prediction_id, result.prediction_id, result.verified_species_id)
            llm = _card_llm(request, session_id=f"fishora-card-{job.id}")
            background_tasks.add_task(
                run_workflow,
                job.id,
                result.verified_species_id,
                result.prediction_id,
                getattr(deps, "knowledge_repo", None),
                getattr(deps, "embedder", None),
                llm,
                llm,
                getattr(deps, "species_repo", None),
                job_repo,
            )
        except Exception:
            logger.exception("could not schedule the knowledge job for prediction %s", result.prediction_id)
    return VerificationResponse(
        prediction_id=result.prediction_id,
        predicted_species_id=result.predicted_species_id,
        verified_species_id=result.verified_species_id,
        verification_status=result.verification_status,
    )


@router.post("/manual", response_model=ManualEntryResponse)
async def manual(
    request: Request,
    file: UploadFile = File(...),
    species_id: str = Form(...),
):
    """Operator names the species themselves. Used when identification is down."""
    deps = request.app.state.deps
    settings = request.app.state.settings
    result = ManualEntryService(
        species_repo=deps.species_repo,
        prediction_repo=deps.prediction_repo,
        image_store=deps.image_store,
        max_image_bytes=settings.cv_max_image_bytes if settings is not None else DEFAULT_MAX_IMAGE_BYTES,
    ).declare(
        await file.read(),
        filename=file.filename or "image",
        content_type=file.content_type,
        species_id=species_id,
    )
    return ManualEntryResponse(
        prediction_id=result.prediction_id,
        model_version=result.model_version,
        verified_species_id=result.verified_species_id,
        normalized_label=result.normalized_label,
        verification_status=result.verification_status,
    )


@knowledge_router.get("/predictions/{prediction_id}/knowledge", response_model=KnowledgeResponse)
async def knowledge_card(prediction_id: str, request: Request):
    from fastapi.responses import JSONResponse

    deps = request.app.state.deps
    # If async job exists, gate on its status (background LangGraph)
    job_repo = getattr(deps, "job_repo", None)
    if job_repo is not None:
        # Try direct job_id lookup, then fallback to latest job by prediction_id
        job = job_repo.get(prediction_id)
        if job is None and hasattr(job_repo, "list_by_prediction"):
            try:
                jobs = job_repo.list_by_prediction(prediction_id)
                if jobs:
                    # pick latest by created_at if available, else last
                    job = sorted(jobs, key=lambda j: getattr(j, "created_at", ""), reverse=True)[0] if hasattr(jobs[0], "created_at") else jobs[-1]
            except Exception:
                job = None
        if job is not None:
            if job.status == "processing":
                return JSONResponse(status_code=202, content={"detail": "Agent orchestrating...", "job_id": job.id, "status": "processing"})
            if job.status == "completed":
                if job.final_card is not None:
                    from apps.main_api.services.generation import KnowledgeCard

                    try:
                        card = KnowledgeCard.model_validate(job.final_card)
                        return KnowledgeResponse(prediction_id=job.prediction_id, species_id=job.species_id, card=card)
                    except Exception:
                        return JSONResponse(status_code=502, content={"detail": "knowledge generation failed", "job_id": job.id, "status": "failed"})
                return JSONResponse(status_code=502, content={"detail": "knowledge generation failed", "job_id": job.id, "status": "failed"})
            if job.status == "failed":
                return JSONResponse(status_code=502, content={"detail": job.error or "knowledge generation failed", "job_id": job.id, "status": "failed"})
    return KnowledgeService(
        prediction_repo=deps.prediction_repo,
        species_repo=deps.species_repo,
        retriever=deps.retriever,
        generator=deps.generator,
    ).get_for_prediction(prediction_id)
