from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from apps.main_api.db.base import Base


class FishSpecies(Base):
    __tablename__ = "fish_species"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    normalized_label: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    common_name_id: Mapped[str] = mapped_column(String(120), nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String(160))
    taxonomic_rank: Mapped[str] = mapped_column(String(40), nullable=False)
    taxonomy_status: Mapped[str] = mapped_column(String(80), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    __table_args__ = (
        CheckConstraint("verification_status IN ('candidate', 'verified')", name="ck_knowledge_sources_verification_status"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(String(160))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        CheckConstraint(
            "category IN ('identity', 'physical_characteristics', 'taste_texture', "
            "'processing_methods', 'commercial_uses', 'substitutes')",
            name="ck_knowledge_chunks_category",
        ),
        CheckConstraint("verification_status IN ('candidate', 'verified')", name="ck_knowledge_chunks_verification_status"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    species_id: Mapped[str] = mapped_column(ForeignKey("fish_species.id"), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("knowledge_sources.id"), index=True)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(160), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("verification_status IN ('pending', 'confirmed', 'corrected')", name="ck_predictions_verification_status"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    image_reference: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_species_id: Mapped[str] = mapped_column(ForeignKey("fish_species.id"), index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_candidates: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)
    verified_species_id: Mapped[str | None] = mapped_column(ForeignKey("fish_species.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LandingPoint(Base):
    __tablename__ = "landing_points"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)


class Lot(Base):
    __tablename__ = "lots"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'active', 'closed', 'allocated')",
            name="ck_lots_status",
        ),
        CheckConstraint("quantity_kg > 0", name="ck_lots_quantity_positive"),
        CheckConstraint("starting_price_per_kg > 0", name="ck_lots_price_positive"),
        CheckConstraint("auction_ends_at > auction_starts_at", name="ck_lots_auction_window"),
        CheckConstraint("size_category IN ('S', 'M', 'L')", name="ck_lots_size_category"),
        UniqueConstraint("public_slug", name="uq_lots_public_slug"),
        # One catch publishes once, as a batch of lots numbered 1..batch_size,
        # each its own auction. A second publish of the catch collides on lot 1.
        UniqueConstraint("prediction_id", "batch_index", name="uq_lots_prediction_batch"),
        CheckConstraint("batch_index BETWEEN 1 AND batch_size", name="ck_lots_batch_index"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    # No index=True: uq_lots_prediction_batch leads with this column.
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    operator_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    species_id: Mapped[str] = mapped_column(ForeignKey("fish_species.id"), nullable=False, index=True)
    landing_point_id: Mapped[str] = mapped_column(ForeignKey("landing_points.id"), nullable=False)
    quantity_kg: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    size_category: Mapped[str] = mapped_column(String(1), nullable=False)
    starting_price_per_kg: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    auction_starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    auction_ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    knowledge_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    public_slug: Mapped[str] = mapped_column(String(160), nullable=False)
    allocated_buyer_id: Mapped[str | None] = mapped_column(String(120))
    seller_fisher_group: Mapped[str | None] = mapped_column(String(160))
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Bid(Base):
    __tablename__ = "bids"
    __table_args__ = (
        CheckConstraint("amount_per_kg > 0", name="ck_bids_amount_positive"),
        Index("ix_bids_lot_id_created_at", "lot_id", text("created_at DESC")),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    lot_id: Mapped[str] = mapped_column(ForeignKey("lots.id"), nullable=False)
    buyer_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    amount_per_kg: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BuyerPreference(Base):
    __tablename__ = "buyer_preferences"

    buyer_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    business_type: Mapped[str] = mapped_column(String(80), nullable=False)
    intended_uses: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    characteristics: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    max_price_per_kg: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    min_quantity_kg: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)


class KnowledgeJob(Base):
    __tablename__ = "knowledge_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('processing', 'completed', 'failed')", name="ck_knowledge_jobs_status"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    species_id: Mapped[str] = mapped_column(ForeignKey("fish_species.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    expert_outputs: Mapped[dict | None] = mapped_column(JSONB)
    critic_feedback: Mapped[str | None] = mapped_column(Text)
    final_card: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    trace: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CommercialBuyerReview(Base):
    __tablename__ = "commercial_buyer_reviews"
    __table_args__ = (
        CheckConstraint(
            "processing_suitability BETWEEN 1 AND 5",
            name="ck_reviews_suitability_range",
        ),
        # One review per buyer per lot: post-use feedback, not a comment thread.
        UniqueConstraint("lot_id", "buyer_id", name="uq_reviews_lot_buyer"),
        # Reviews are read by species so one buyer's experience reaches every
        # auction for that fish, including lots from a different fisher group.
        Index("ix_reviews_species_created_at", "species_id", text("created_at DESC")),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    lot_id: Mapped[str] = mapped_column(ForeignKey("lots.id"), nullable=False, index=True)
    species_id: Mapped[str] = mapped_column(ForeignKey("fish_species.id"), nullable=False)
    buyer_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    actual_use: Mapped[str] = mapped_column(String(120), nullable=False)
    processing_suitability: Mapped[int] = mapped_column(Integer, nullable=False)
    substitute_acceptance: Mapped[bool | None] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('operator', 'buyer')", name="ck_users_role"),
        UniqueConstraint("username", name="uq_users_username"),
    )

    # Lots, bids and preferences store this id as a plain string, not a foreign
    # key: the seeded demo ids (op_rian, buyer_dewi) predate this table.
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    username: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
