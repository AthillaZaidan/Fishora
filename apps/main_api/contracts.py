from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class SpeciesRecord:
    id: str
    normalized_label: str
    common_name_id: str
    scientific_name: str | None
    taxonomic_rank: str
    taxonomy_status: str
    notes: str | None


@dataclass(frozen=True)
class KnowledgeSourceWrite:
    """Verified source row handed to the knowledge repository for insertion."""

    id: str
    title: str
    source_type: str
    url: str | None
    publisher: str | None
    reviewed_at: datetime | None
    verification_status: str


@dataclass(frozen=True)
class KnowledgeChunkWrite:
    """Verified chunk row with its local embedding, for transactional insert."""

    id: str
    species_id: str
    source_id: str
    category: str
    content: str
    embedding: list[float]
    embedding_model: str
    verification_status: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Verified retrieval hit with citation metadata for generation (Task 7).

    ``source_id`` plus the source fields let generation validate citations
    against the store instead of trusting generated text; both verification
    statuses are carried so downstream citation checks never re-trust rows.
    """

    chunk_id: str
    species_id: str
    source_id: str
    source_type: str
    category: str
    content: str
    distance: float
    chunk_verification_status: str
    source_verification_status: str
    source_title: str
    source_publisher: str | None
    source_url: str | None
    source_reviewed_at: datetime | None


@dataclass
class PredictionRecord:
    id: str
    image_reference: str
    predicted_species_id: str
    confidence: float
    top_candidates: list[dict[str, object]]
    model_version: str
    verification_status: Literal["pending", "confirmed", "corrected"]
    verified_species_id: str | None = None


@dataclass
class KnowledgeJobRecord:
    id: str
    prediction_id: str
    species_id: str
    status: Literal["processing", "completed", "failed"]
    expert_outputs: dict | None = None
    critic_feedback: str | None = None
    final_card: dict | None = None
    error: str | None = None
    # Stage trace of the run that produced the card (W12): evidence, expert
    # subsets, prompt hashes, verdicts, timings, tokens.
    trace: dict | None = None


@dataclass(frozen=True)
class TaxonomySeed:
    raw_folder: str
    raw_label: str
    normalized_label: str
    scientific_name: str | None
    common_name_id: str
    taxonomic_rank: str
    confidence: str
    source: str
    notes: str | None
    taxonomy_status: str


@dataclass
class LotRecord:
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
    knowledge_snapshot: dict | None = None
    allocated_buyer_id: str | None = None
    # PRD 8.3.1 requires Seller / Fisher Group on every lot. The operator
    # publishes, so this is how the fisher stays visible in the record.
    seller_fisher_group: str | None = None
    # Lot n of the N a catch was split into at publication.
    batch_index: int = 1
    batch_size: int = 1


@dataclass
class BidRecord:
    id: str
    lot_id: str
    buyer_id: str
    amount_per_kg: Decimal
    created_at: datetime


@dataclass
class BuyerPreferenceRecord:
    buyer_id: str
    business_type: str
    intended_uses: list[str]
    characteristics: list[str]
    max_price_per_kg: Decimal | None
    min_quantity_kg: Decimal | None
    latitude: float
    longitude: float


@dataclass
class LandingPointRecord:
    id: str
    name: str
    latitude: float
    longitude: float


@dataclass
class ReviewRecord:
    """Commercial-buyer review (PRD 8.5). A market signal, never verified
    knowledge, so it is stored and served apart from the knowledge card."""

    id: str
    lot_id: str
    species_id: str
    buyer_id: str
    actual_use: str
    processing_suitability: int
    substitute_acceptance: bool | None = None
    comment: str | None = None
    created_at: datetime | None = None
