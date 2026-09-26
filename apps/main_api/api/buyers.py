from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from apps.main_api.api.lots import LotResponse, _lot_response
from apps.main_api.contracts import BuyerPreferenceRecord
from apps.main_api.errors import Forbidden
from apps.main_api.services.geo import DEFAULT_SERVICEABILITY_RADIUS_KM
from apps.main_api.services.lots import LotService
from apps.main_api.services.matching import (
    MATCH_THRESHOLD,
    nearest_landing_point,
    normalise_business_type,
    recommend,
)
from apps.main_api.services.session import SessionUser, require_role

router = APIRouter(prefix="/api/v1/buyers")


class PreferenceRequest(BaseModel):
    business_type: str = Field(min_length=1, max_length=80)
    intended_uses: list[str] = Field(default_factory=list, max_length=30)
    characteristics: list[str] = Field(default_factory=list, max_length=30)
    # Positive when given: "no limit" is sent as null, and a zero limit would
    # silently match nothing.
    max_price_per_kg: Decimal | None = Field(default=None, gt=0)
    min_quantity_kg: Decimal | None = Field(default=None, ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @field_validator("business_type")
    @classmethod
    def _business_type(cls, value: str) -> str:
        # The form used to save Indonesian keys; store every profile on one set.
        return normalise_business_type(value)

    @field_validator("intended_uses", "characteristics")
    @classmethod
    def _choices(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if item and item.casefold() not in {existing.casefold() for existing in cleaned}:
                cleaned.append(item[:80])
        return cleaned


class PreferenceResponse(PreferenceRequest):
    buyer_id: str


class MatchReasonResponse(BaseModel):
    criterion: str
    met: bool
    detail: str
    value: str | None = None


class RecommendationItem(BaseModel):
    lot: LotResponse
    score: float
    # score >= match_threshold, computed here so every client badges the same lots.
    matched: bool = False
    reasons: list[MatchReasonResponse]


class RecommendationsResponse(BaseModel):
    items: list[RecommendationItem]
    profile_missing: bool = False
    match_threshold: float = MATCH_THRESHOLD


class NearestLandingPoint(BaseModel):
    id: str
    name: str
    distance_km: float


class PreferencePreviewResponse(BaseModel):
    """What a draft profile would reach, computed exactly as recommendations are."""

    in_range: int
    matched: int
    match_threshold: float = MATCH_THRESHOLD
    radius_km: float = DEFAULT_SERVICEABILITY_RADIUS_KM
    nearest_landing_point: NearestLandingPoint | None = None


def _own_buyer(request: Request, buyer_id: str, action: str) -> SessionUser:
    user = require_role(request, "buyer")
    if user.id != buyer_id:
        raise Forbidden(f"cannot {action} another buyer's preferences")
    return user


def _record(buyer_id: str, payload: PreferenceRequest) -> BuyerPreferenceRecord:
    return BuyerPreferenceRecord(buyer_id=buyer_id, **payload.model_dump())


def _response(record: BuyerPreferenceRecord) -> PreferenceResponse:
    return PreferenceResponse(
        buyer_id=record.buyer_id,
        business_type=record.business_type,
        intended_uses=record.intended_uses,
        characteristics=record.characteristics,
        max_price_per_kg=record.max_price_per_kg,
        min_quantity_kg=record.min_quantity_kg,
        latitude=record.latitude,
        longitude=record.longitude,
    )


def _ranked(request: Request, prefs: BuyerPreferenceRecord):
    """Active lots within reach, scored. The one path behind both the saved
    recommendations and the draft preview, so the two counts cannot disagree."""
    deps = request.app.state.deps
    lot_service = LotService(
        prediction_repo=deps.prediction_repo,
        lot_repo=deps.lot_repo,
        landing_point_repo=deps.landing_point_repo,
        # So recommended lots carry their catch photo like every other listing.
        image_store=getattr(deps, "image_store", None),
    )
    lots = lot_service.list_lots(status="active")
    points = {point.id: point for point in deps.landing_point_repo.all()} if deps.landing_point_repo else {}
    return lot_service, points, recommend(lots, prefs, points)


@router.get("/{buyer_id}/preferences", response_model=PreferenceResponse)
def get_preferences(buyer_id: str, request: Request):
    """The saved profile, so the form opens on it instead of wiping it on save."""
    _own_buyer(request, buyer_id, "read")
    record = request.app.state.deps.preference_repo.get(buyer_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no preferences saved")
    return _response(record)


@router.put("/{buyer_id}/preferences", response_model=PreferenceResponse)
def put_preferences(buyer_id: str, payload: PreferenceRequest, request: Request):
    _own_buyer(request, buyer_id, "write")
    saved = request.app.state.deps.preference_repo.upsert(_record(buyer_id, payload))
    return _response(saved)


@router.post("/{buyer_id}/preferences/preview", response_model=PreferencePreviewResponse)
def preview_preferences(buyer_id: str, payload: PreferenceRequest, request: Request):
    """Counts for an unsaved draft. Nothing is stored."""
    _own_buyer(request, buyer_id, "preview")
    _, points, ranked = _ranked(request, _record(buyer_id, payload))
    nearest = nearest_landing_point(payload.latitude, payload.longitude, list(points.values()))
    return PreferencePreviewResponse(
        in_range=len(ranked),
        matched=sum(1 for _, result in ranked if result.matched),
        nearest_landing_point=None if nearest is None else NearestLandingPoint(
            id=nearest[0].id, name=nearest[0].name, distance_km=round(nearest[1], 1)
        ),
    )


@router.get("/{buyer_id}/recommendations", response_model=RecommendationsResponse)
def get_recommendations(buyer_id: str, request: Request):
    user = require_role(request, "buyer")
    if user.id != buyer_id:
        raise Forbidden("cannot read another buyer's recommendations")
    prefs = request.app.state.deps.preference_repo.get(buyer_id)
    if prefs is None:
        return RecommendationsResponse(items=[], profile_missing=True)

    lot_service, _, ranked = _ranked(request, prefs)
    items = [
        RecommendationItem(
            lot=_lot_response(lot_service, lot),
            score=result.score,
            matched=result.matched,
            reasons=[MatchReasonResponse(
                criterion=reason.criterion, met=reason.met, detail=reason.detail, value=reason.value
            ) for reason in result.reasons],
        )
        for lot, result in ranked
    ]
    return RecommendationsResponse(items=items, profile_missing=False)
