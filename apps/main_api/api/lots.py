from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from apps.main_api.contracts import BidRecord, LotRecord
from apps.main_api.errors import Forbidden
from apps.main_api.services.geo import DEFAULT_SERVICEABILITY_RADIUS_KM
from apps.main_api.services.lots import (
    AUCTION_MINUTE_OPTIONS,
    MAX_LOT_COUNT,
    LotService,
)
from apps.main_api.services.search import search_lots
from apps.main_api.services.session import require_role

router = APIRouter(prefix="/api/v1/lots")


class PublishLotRequest(BaseModel):
    prediction_id: str
    operator_id: str | None = None
    # Per lot: the catch is published as `lot_count` lots of this many kg.
    quantity_kg: Decimal = Field(gt=0)
    lot_count: int = Field(default=1, ge=1, le=MAX_LOT_COUNT)
    starting_price_per_kg: Decimal = Field(gt=0)
    size_category: Literal["S", "M", "L"]
    landing_point_id: str
    auction_minutes: int | None = Field(
        default=None, description=f"One of {', '.join(map(str, AUCTION_MINUTE_OPTIONS))}"
    )
    seller_fisher_group: str | None = Field(default=None, max_length=160)


class LotResponse(BaseModel):
    id: str
    prediction_id: str
    operator_id: str
    species_id: str
    landing_point_id: str
    quantity_kg: Decimal
    size_category: Literal["S", "M", "L"]
    starting_price_per_kg: Decimal
    status: Literal["draft", "active", "closed", "allocated"]
    auction_starts_at: datetime
    auction_ends_at: datetime
    public_slug: str
    allocated_buyer_id: str | None = None
    seller_fisher_group: str | None = None
    batch_index: int = 1
    batch_size: int = 1
    current_highest_per_kg: Decimal | None = None
    serviceability_radius_km: float = DEFAULT_SERVICEABILITY_RADIUS_KM


class PlaceBidRequest(BaseModel):
    buyer_id: str | None = None
    amount_per_kg: Decimal = Field(gt=0)


class BidResponse(BaseModel):
    id: str
    lot_id: str
    buyer_id: str
    amount_per_kg: Decimal
    created_at: datetime


class AllocateResponse(BaseModel):
    id: str
    status: Literal["allocated"]
    allocated_buyer_id: str
    current_highest_per_kg: Decimal | None = None


def _service(request: Request, *, publishing: bool = False) -> LotService:
    deps = request.app.state.deps
    knowledge_service = None
    # Only publication can need a card made on the spot (no completed job), and
    # only then is an LLM client built. The card goes through the same graded
    # graph as the job (W19).
    if publishing and deps.knowledge_repo is not None and deps.embedder is not None:
        from apps.main_api.services.card_llm import card_llm
        from apps.main_api.services.knowledge import KnowledgeService

        knowledge_service = KnowledgeService(
            prediction_repo=deps.prediction_repo,
            species_repo=deps.species_repo,
            knowledge_repo=deps.knowledge_repo,
            embedder=deps.embedder,
            llm=card_llm(deps, getattr(request.app.state, "settings", None), session_id=None),
        )
    return LotService(
        prediction_repo=deps.prediction_repo,
        lot_repo=deps.lot_repo,
        landing_point_repo=deps.landing_point_repo,
        knowledge_service=knowledge_service,
        job_repo=getattr(deps, "job_repo", None),
    )


def _lot_response(service: LotService, lot: LotRecord) -> LotResponse:
    return LotResponse(
        id=lot.id,
        prediction_id=lot.prediction_id,
        operator_id=lot.operator_id,
        species_id=lot.species_id,
        landing_point_id=lot.landing_point_id,
        quantity_kg=lot.quantity_kg,
        size_category=lot.size_category,
        starting_price_per_kg=lot.starting_price_per_kg,
        status=lot.status,
        auction_starts_at=lot.auction_starts_at,
        auction_ends_at=lot.auction_ends_at,
        public_slug=lot.public_slug,
        allocated_buyer_id=lot.allocated_buyer_id,
        seller_fisher_group=lot.seller_fisher_group,
        batch_index=lot.batch_index,
        batch_size=lot.batch_size,
        current_highest_per_kg=service.current_highest(lot.id),
        serviceability_radius_km=DEFAULT_SERVICEABILITY_RADIUS_KM,
    )


def _bid_response(bid: BidRecord) -> BidResponse:
    return BidResponse(
        id=bid.id,
        lot_id=bid.lot_id,
        buyer_id=bid.buyer_id,
        amount_per_kg=bid.amount_per_kg,
        created_at=bid.created_at,
    )


@router.post("", response_model=list[LotResponse])
def publish_lot(payload: PublishLotRequest, request: Request):
    """Publish a verified catch as `lot_count` lots, each its own auction."""
    user = require_role(request, "operator")
    if payload.operator_id and payload.operator_id != user.id:
        raise Forbidden("operator token cannot publish as another operator")
    service = _service(request, publishing=True)
    lots = service.publish(
        prediction_id=payload.prediction_id,
        operator_id=user.id,
        quantity_kg=payload.quantity_kg,
        lot_count=payload.lot_count,
        starting_price_per_kg=payload.starting_price_per_kg,
        size_category=payload.size_category,
        landing_point_id=payload.landing_point_id,
        auction_minutes=payload.auction_minutes,
        seller_fisher_group=payload.seller_fisher_group,
    )
    return [_lot_response(service, lot) for lot in lots]


@router.get("", response_model=list[LotResponse])
def list_lots(
    request: Request,
    species_id: list[str] | None = Query(default=None),
    intended_use: list[str] | None = Query(default=None),
    characteristic: list[str] | None = Query(default=None),
    min_price: Decimal | None = None,
    max_price: Decimal | None = None,
    min_quantity: Decimal | None = None,
    max_quantity: Decimal | None = None,
    status: str | None = None,
    mine: bool = Query(default=False),
    buyer_lat: float | None = Query(default=None),
    buyer_lon: float | None = Query(default=None),
    serviceability_radius_km: float | None = Query(default=None),
):
    """HANDOFF Slice C filters.

    `species_id`, `intended_use` and `characteristic` repeat, and repeats are
    OR: `?intended_use=digoreng&intended_use=fillet` returns lots suited to
    either. The lists AND with each other and with the range filters. OR
    mirrors the matching engine, which scores each criterion on set
    intersection, so the buyer profile preview and the saved recommendation
    count cannot disagree.
    """
    service = _service(request)
    operator_id = require_role(request, "operator").id if mine else None
    lots = service.list_lots(
        species_ids=species_id,
        intended_uses=intended_use,
        characteristics=characteristic,
        operator_id=operator_id,
        min_price=min_price,
        max_price=max_price,
        min_quantity=min_quantity,
        max_quantity=max_quantity,
        status=status,
        buyer_lat=buyer_lat,
        buyer_lon=buyer_lon,
        serviceability_radius_km=serviceability_radius_km,
    )
    return [_lot_response(service, lot) for lot in lots]


class SearchResponse(BaseModel):
    matches: list[LotResponse]
    # Other open lots the matched fish's knowledge card names as similar.
    similar: list[LotResponse]
    similar_to: list[str]


@router.get("/search", response_model=SearchResponse)
def search_lots_endpoint(
    request: Request,
    q: str = Query(default="", max_length=120),
    min_price: Decimal | None = None,
    max_price: Decimal | None = None,
):
    """Open lots matching `q` by name or knowledge-card characteristics, plus similar fish."""
    service = _service(request)
    lots = service.list_lots(status="active", min_price=min_price, max_price=max_price)
    result = search_lots(lots, q)
    return SearchResponse(
        matches=[_lot_response(service, lot) for lot in result.matches],
        similar=[_lot_response(service, lot) for lot in result.similar],
        similar_to=result.similar_to,
    )


@router.get("/{lot_id}", response_model=LotResponse)
def get_lot(lot_id: str, request: Request):
    service = _service(request)
    return _lot_response(service, service.get(lot_id))


@router.post("/{lot_id}/bids", response_model=BidResponse)
def place_bid(lot_id: str, payload: PlaceBidRequest, request: Request):
    user = require_role(request, "buyer")
    if payload.buyer_id and payload.buyer_id != user.id:
        raise Forbidden("buyer token cannot bid as another buyer")
    service = _service(request)
    return _bid_response(service.place_bid(lot_id, user.id, payload.amount_per_kg))


@router.get("/{lot_id}/bids", response_model=list[BidResponse])
def list_bids(lot_id: str, request: Request):
    service = _service(request)
    return [_bid_response(bid) for bid in service.list_bids(lot_id)]


@router.post("/{lot_id}/close", response_model=LotResponse)
def close_lot(lot_id: str, request: Request):
    user = require_role(request, "operator")
    service = _service(request)
    lot = service.get(lot_id)
    if lot.operator_id != user.id:
        raise Forbidden("only the listing operator can close")
    return _lot_response(service, service.close(lot_id))


@router.post("/{lot_id}/allocate", response_model=AllocateResponse)
def allocate_lot(lot_id: str, request: Request):
    user = require_role(request, "operator")
    service = _service(request)
    lot = service.get(lot_id)
    if lot.operator_id != user.id:
        raise Forbidden("only the listing operator can allocate")
    lot = service.allocate(lot_id)
    return AllocateResponse(
        id=lot.id,
        status="allocated",
        allocated_buyer_id=lot.allocated_buyer_id or "",
        current_highest_per_kg=service.current_highest(lot.id),
    )
