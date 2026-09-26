from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from apps.main_api.contracts import BidRecord, LotRecord
from apps.main_api.errors import Forbidden, LotNotFound, Unauthenticated
from apps.main_api.services.geo import DEFAULT_SERVICEABILITY_RADIUS_KM
from apps.main_api.services.landing_points import list_landing_points
from apps.main_api.services.lots import (
    AUCTION_MINUTE_OPTIONS,
    MAX_LOT_COUNT,
    LotService,
    bidder_labels,
    winning_bid,
)
from apps.main_api.services.search import search_lots
from apps.main_api.services.session import SessionUser, current_user, require_role

router = APIRouter(prefix="/api/v1/lots")
# Registered on its own in main.py: landing points are not lots.
landing_router = APIRouter(prefix="/api/v1/landing-points")


class PublishLotRequest(BaseModel):
    prediction_id: str
    operator_id: str | None = None
    # Per lot: the catch is published as `lot_count` lots of this many kg.
    # Digit limits mirror the NUMERIC columns, so an oversized value is a 422
    # rather than a database overflow, and never rounded on save.
    quantity_kg: Decimal = Field(gt=0, max_digits=12, decimal_places=3)
    lot_count: int = Field(default=1, ge=1, le=MAX_LOT_COUNT)
    starting_price_per_kg: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
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
    # An auction past auction_ends_at always reads as closed, never active.
    status: Literal["draft", "active", "closed", "allocated"]
    auction_starts_at: datetime
    auction_ends_at: datetime
    public_slug: str
    # Only the lot's operator and the winning buyer see who won; everyone else
    # gets null. `won_by_you` answers the one question a buyer has.
    allocated_buyer_id: str | None = None
    won_by_you: bool = False
    seller_fisher_group: str | None = None
    batch_index: int = 1
    batch_size: int = 1
    current_highest_per_kg: Decimal | None = None
    serviceability_radius_km: float = DEFAULT_SERVICEABILITY_RADIUS_KM
    # Path on this API of the catch photo, when one was stored.
    photo_url: str | None = None


class PlaceBidRequest(BaseModel):
    buyer_id: str | None = None
    # NUMERIC(12, 2): a third decimal would be rounded on save, onto a tie with
    # the bid it had to beat.
    amount_per_kg: Decimal = Field(gt=0, max_digits=12, decimal_places=2)


class BidResponse(BaseModel):
    id: str
    lot_id: str
    # "Bidder 1", "Bidder 2"... by first-bid order, or "You" for the viewer's
    # own bids. Buyer ids are account identifiers and are not public.
    bidder: str
    is_you: bool = False
    # Only the lot's operator, and a buyer for their own bids, get the id.
    buyer_id: str | None = None
    amount_per_kg: Decimal
    created_at: datetime


class AllocateResponse(BaseModel):
    id: str
    status: Literal["allocated"]
    allocated_buyer_id: str
    current_highest_per_kg: Decimal | None = None
    # The outcome the operator is shown: the winner's bid and public label.
    winning_amount_per_kg: Decimal | None = None
    winner_label: str | None = None


class LandingPointResponse(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float


def _viewer(request: Request) -> SessionUser | None:
    """The signed-in user, or None. Lot reads are public, but what they show
    (who won, whose bid is whose) depends on who is asking."""
    try:
        return current_user(request)
    except Unauthenticated:
        return None


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
        image_store=deps.image_store,
    )


def _lot_response(service: LotService, lot: LotRecord, viewer: SessionUser | None = None) -> LotResponse:
    viewer_id = viewer.id if viewer is not None else None
    sees_winner = viewer_id is not None and viewer_id in (lot.operator_id, lot.allocated_buyer_id)
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
        allocated_buyer_id=lot.allocated_buyer_id if sees_winner else None,
        won_by_you=viewer_id is not None and lot.allocated_buyer_id == viewer_id,
        seller_fisher_group=lot.seller_fisher_group,
        batch_index=lot.batch_index,
        batch_size=lot.batch_size,
        current_highest_per_kg=service.current_highest(lot.id),
        serviceability_radius_km=DEFAULT_SERVICEABILITY_RADIUS_KM,
        photo_url=f"/api/v1/lots/{lot.id}/photo" if service.photo(lot) else None,
    )


def _bid_response(
    bid: BidRecord, labels: dict[str, str], lot: LotRecord, viewer: SessionUser | None
) -> BidResponse:
    viewer_id = viewer.id if viewer is not None else None
    mine = viewer_id is not None and bid.buyer_id == viewer_id
    return BidResponse(
        id=bid.id,
        lot_id=bid.lot_id,
        bidder="You" if mine else labels.get(bid.buyer_id, "Bidder"),
        is_you=mine,
        buyer_id=bid.buyer_id if mine or viewer_id == lot.operator_id else None,
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
    return [_lot_response(service, lot, user) for lot in lots]


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
    won: bool = Query(default=False, description="Only the signed-in buyer's allocated lots"),
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
    viewer = _viewer(request)
    operator_id = require_role(request, "operator").id if mine else None
    # Scoped by the session, never by a query parameter: which lots a buyer won
    # is theirs to read, and allocated_buyer_id is hidden from everyone else.
    winner_id = require_role(request, "buyer").id if won else None
    lots = service.list_lots(
        species_ids=species_id,
        intended_uses=intended_use,
        characteristics=characteristic,
        operator_id=operator_id,
        allocated_buyer_id=winner_id,
        min_price=min_price,
        max_price=max_price,
        min_quantity=min_quantity,
        max_quantity=max_quantity,
        status="allocated" if won else status,
        buyer_lat=buyer_lat,
        buyer_lon=buyer_lon,
        serviceability_radius_km=serviceability_radius_km,
    )
    return [_lot_response(service, lot, viewer) for lot in lots]


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
    viewer = _viewer(request)
    lots = service.list_lots(status="active", min_price=min_price, max_price=max_price)
    result = search_lots(lots, q)
    return SearchResponse(
        matches=[_lot_response(service, lot, viewer) for lot in result.matches],
        similar=[_lot_response(service, lot, viewer) for lot in result.similar],
        similar_to=result.similar_to,
    )


@router.get("/{lot_id}", response_model=LotResponse)
def get_lot(lot_id: str, request: Request):
    service = _service(request)
    return _lot_response(service, service.get(lot_id), _viewer(request))


@router.get("/{lot_id}/photo", response_class=FileResponse)
def get_lot_photo(lot_id: str, request: Request):
    """The catch photo the lot was identified from. Public, like the lot."""
    service = _service(request)
    photo = service.photo(service.get(lot_id))
    if photo is None:
        raise LotNotFound(f"{lot_id} has no photo")
    path, content_type = photo
    # A prediction's photo never changes once stored.
    return FileResponse(path, media_type=content_type, headers={"cache-control": "public, max-age=86400"})


@router.post("/{lot_id}/bids", response_model=BidResponse)
def place_bid(lot_id: str, payload: PlaceBidRequest, request: Request):
    user = require_role(request, "buyer")
    if payload.buyer_id and payload.buyer_id != user.id:
        raise Forbidden("buyer token cannot bid as another buyer")
    service = _service(request)
    bid = service.place_bid(lot_id, user.id, payload.amount_per_kg)
    # The bidder's own bid: labelled "You", so no other label is needed.
    return _bid_response(bid, {}, service.get(lot_id), user)


@router.get("/{lot_id}/bids", response_model=list[BidResponse])
def list_bids(lot_id: str, request: Request):
    service = _service(request)
    lot = service.get(lot_id)
    bids = service.list_bids(lot_id)
    labels = bidder_labels(bids)
    viewer = _viewer(request)
    return [_bid_response(bid, labels, lot, viewer) for bid in bids]


@router.post("/{lot_id}/close", response_model=LotResponse)
def close_lot(lot_id: str, request: Request):
    user = require_role(request, "operator")
    service = _service(request)
    lot = service.get(lot_id)
    if lot.operator_id != user.id:
        raise Forbidden("only the listing operator can close")
    return _lot_response(service, service.close(lot_id), user)


@router.post("/{lot_id}/allocate", response_model=AllocateResponse)
def allocate_lot(lot_id: str, request: Request):
    user = require_role(request, "operator")
    service = _service(request)
    lot = service.get(lot_id)
    if lot.operator_id != user.id:
        raise Forbidden("only the listing operator can allocate")
    lot = service.allocate(lot_id)
    bids = service.list_bids(lot_id)
    # The repository picked the winner; report that buyer's best bid rather
    # than recomputing a winner that could disagree with it.
    winning = winning_bid([bid for bid in bids if bid.buyer_id == lot.allocated_buyer_id])
    return AllocateResponse(
        id=lot.id,
        status="allocated",
        allocated_buyer_id=lot.allocated_buyer_id or "",
        current_highest_per_kg=service.current_highest(lot.id),
        winning_amount_per_kg=winning.amount_per_kg if winning else None,
        winner_label=bidder_labels(bids).get(lot.allocated_buyer_id or ""),
    )


@landing_router.get("", response_model=list[LandingPointResponse])
def list_landing_points_endpoint(request: Request):
    """Where an operator can publish from: the ids `POST /api/v1/lots` accepts."""
    repo = request.app.state.deps.landing_point_repo
    return [
        LandingPointResponse(id=point.id, name=point.name, latitude=point.latitude, longitude=point.longitude)
        for point in list_landing_points(repo)
    ]
