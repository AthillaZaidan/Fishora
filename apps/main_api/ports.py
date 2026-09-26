from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

from sqlalchemy.orm import Session

from apps.contracts import CVPredictionEnvelope
from apps.main_api.contracts import (
    BidRecord,
    BuyerPreferenceRecord,
    KnowledgeChunkWrite,
    KnowledgeJobRecord,
    KnowledgeSourceWrite,
    LandingPointRecord,
    LotRecord,
    PredictionRecord,
    RetrievedChunk,
    SpeciesRecord,
)


class CVClient(Protocol):
    def predict(self, image_bytes: bytes, *, filename: str, content_type: str) -> CVPredictionEnvelope: ...


class ImageStore(Protocol):
    def save(self, prediction_id: str, image_bytes: bytes, content_type: str) -> str: ...
    def delete(self, image_reference: str) -> None: ...


class SpeciesRepository(Protocol):
    def get_by_normalized_label(self, label: str) -> SpeciesRecord | None: ...
    def get_by_id(self, species_id: str) -> SpeciesRecord | None: ...
    def list_all(self) -> list[SpeciesRecord]: ...


class Tokenizer(Protocol):
    """A real tokenizer (encode/decode), not a bare token counter.

    Defaults mirror HF tokenizers (encode wraps in special tokens, decode
    renders them); chunking always opts out so chunk text is clean.
    """

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...


class Embedder(Protocol):
    model_name: str
    tokenizer: Tokenizer

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class KnowledgeRepository(Protocol):
    """Transactional store for approved sources/chunks (one commit, all or nothing)."""

    def embedding_models_in_store(self) -> set[str]: ...
    def insert_verified(
        self,
        sources: Sequence[KnowledgeSourceWrite],
        chunks: Sequence[KnowledgeChunkWrite],
    ) -> int: ...

    def search_verified(
        self,
        species_id: str,
        query_vector: list[float],
        embedding_model: str,
        limit: int,
    ) -> list[RetrievedChunk]: ...


class PredictionRepository(Protocol):
    def create(
        self,
        prediction_id: str,
        image_reference: str,
        predicted_species_id: str,
        confidence: float,
        top_candidates: list[dict[str, object]],
        model_version: str,
    ) -> PredictionRecord: ...

    def get(self, prediction_id: str) -> PredictionRecord | None: ...

    def verify(
        self,
        prediction_id: str,
        verified_species_id: str,
        verification_status: Literal["confirmed", "corrected"],
    ) -> PredictionRecord: ...


class KnowledgeJobRepository(Protocol):
    def create(self, job_id: str, prediction_id: str, species_id: str) -> KnowledgeJobRecord: ...
    def get(self, job_id: str) -> KnowledgeJobRecord | None: ...
    def update(self, job_id: str, status: str | None = None, **fields) -> KnowledgeJobRecord | None: ...
    def list_by_prediction(self, prediction_id: str) -> list[KnowledgeJobRecord]: ...


@dataclass
class AppDependencies:
    """Small concrete bundle of ports for the main API.

    Production wiring happens lazily in the app lifespan; tests inject fakes
    through the bundle instead of a nine-argument factory signature. The five
    original concrete ports stay the completeness criterion; knowledge_repo,
    retriever, and generator are additionally built in the production path
    and injected by tests that exercise the knowledge endpoint.
    """

    session_factory: Callable[[], Session] | None = None
    cv_client: CVClient | None = None
    species_repo: SpeciesRepository | None = None
    prediction_repo: PredictionRepository | None = None
    image_store: ImageStore | None = None
    embedder: Embedder | None = None
    knowledge_repo: KnowledgeRepository | None = None
    retriever: object | None = None  # VerifiedRetriever
    generator: object | None = None  # KnowledgeGenerator
    lot_repo: object | None = None  # LotRepository
    preference_repo: object | None = None  # PreferenceRepository
    landing_point_repo: object | None = None
    session_service: object | None = None
    review_repo: object | None = None  # ReviewRepository
    job_repo: KnowledgeJobRepository | None = None
    # Chat model for the agent graph (LangChain-style ``invoke``). None builds
    # the OpenCode Go client from settings per card (api/fish.py::_card_llm).
    llm: object | None = None
