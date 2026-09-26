from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from apps.main_api.services.session import require_role

router = APIRouter(prefix="/api/v1/jobs")


class JobResponse(BaseModel):
    job_id: str
    prediction_id: str
    species_id: str
    status: str
    final_card: dict | None = None
    error: str | None = None
    expert_outputs: dict | None = None
    critic_feedback: str | None = None


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request):
    require_role(request, "operator")
    deps = request.app.state.deps
    job_repo = getattr(deps, "job_repo", None)
    if job_repo is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return JobResponse(
        job_id=job.id,
        prediction_id=job.prediction_id,
        species_id=job.species_id,
        status=job.status,
        final_card=job.final_card,
        error=job.error,
        expert_outputs=job.expert_outputs,
        critic_feedback=job.critic_feedback,
    )
