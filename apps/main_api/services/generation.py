"""LangChain-grounded generation via the OpenCode Go Responses API.

The OpenCodeGoClient uses LangChain's ``ChatOpenAI`` Responses API adapter
with strict structured output. Timeout/connection failures become
``OpenCodeUnavailable`` carrying only the retrieved chunk ids; credentials,
internal URLs, and headers never appear in messages. The production client
rejects a blank API key, and is constructed lazily (see KnowledgeGenerator)
so empty-evidence requests never touch it.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.errors import InvalidGeneratedKnowledge, OpenCodeUnavailable

SYSTEM_PROMPT = (
    Path(__file__).resolve().parents[1] / "prompts" / "knowledge_card_system.txt"
).read_text(encoding="utf-8")
# First limitation of a card built with no evidence at all.
NO_INFORMATION = "No information available yet"


class GeneratedCitation(BaseModel):
    """The LLM may emit only source ids; all title/url/publisher metadata is
    enriched server-side from retrieved evidence, never trusted from text."""

    source_id: str


class GeneratedKnowledgeCard(BaseModel):
    """Strict generated payload: extra fields are rejected outright."""

    model_config = ConfigDict(extra="forbid")

    common_name: str
    scientific_name: str | None
    taxonomy_status: str
    physical_characteristics: str | None
    taste: str | None
    texture: str | None
    processing_methods: list[str]
    commercial_uses: list[str]
    similar_or_substitute_species: list[str]
    potential_buyer_segments: list[str]
    limitations: list[str]
    sources: list[GeneratedCitation]


class OpenCodeGoClient:
    """LangChain client making one OpenCode Go Responses API call per generation."""

    def __init__(self, settings, session_id: str | None = None):
        llm = make_opencode_go_llm(settings, session_id=session_id)
        self._structured_llm = llm.with_structured_output(
            GeneratedKnowledgeCard,
            method="json_schema",
            strict=True,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", "{system_prompt}"), ("human", "{payload}")]
        )

    def generate(
        self,
        system_prompt: str,
        evidence: list[RetrievedChunk],
        species: SpeciesRecord,
    ) -> GeneratedKnowledgeCard:
        messages = self._prompt.invoke(
            {"system_prompt": system_prompt, "payload": _user_payload(species, evidence)}
        ).to_messages()
        try:
            result = self._structured_llm.invoke(messages)
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise OpenCodeUnavailable(
                "opencode go generation unavailable",
                [chunk.chunk_id for chunk in evidence],
            ) from exc
        try:
            return GeneratedKnowledgeCard.model_validate(result)
        except ValidationError as exc:
            raise InvalidGeneratedKnowledge(
                "generated knowledge failed schema validation",
                [chunk.chunk_id for chunk in evidence],
            ) from exc


USER_AGENT = "fishora-knowledge-cards/0.1"


def make_opencode_go_llm(settings, timeout: float | None = None, session_id: str | None = None):
    """Create the shared Luna client used by every orchestration agent.

    OpenCode Go rejects requests without a stable ``x-opencode-session`` header
    (400 MissingSessionID) and asks clients to identify themselves with their
    own user agent (finding W20). One session id per card keeps the calls of
    that card together; callers that do not pass one get a fresh id.
    """
    api_key = settings.opencode_go_api_key.get_secret_value()
    if not api_key.strip():
        raise ValueError("OPENCODE_GO_API_KEY must be set to construct the production client")
    return ChatOpenAI(
        model=settings.opencode_go_model,
        base_url=settings.opencode_go_base_url,
        api_key=api_key,
        timeout=timeout if timeout is not None else settings.opencode_go_timeout_seconds,
        use_responses_api=True,
        default_headers={
            "x-opencode-session": session_id or f"fishora-{uuid.uuid4().hex}",
            "User-Agent": USER_AGENT,
        },
    )


def _user_payload(species: SpeciesRecord, evidence: list[RetrievedChunk]) -> str:
    """User payload: relational species fields plus the supplied evidence
    passages, each delimited by source_id/species/category/content."""
    lines = [
        "Relational species data (must be followed, never overridden):",
        f"- common name: {species.common_name_id}",
        f"- scientific name: {species.scientific_name}",
        f"- taxonomic rank: {species.taxonomic_rank}",
        f"- taxonomy status: {species.taxonomy_status}",
        "",
        "Supplied evidence (use only this):",
    ]
    for chunk in evidence:
        lines.append(
            f"[source_id: {chunk.source_id}] [species: {chunk.species_id}] "
            f"[category: {chunk.category}]"
        )
        lines.append(chunk.content)
    return "\n".join(lines)


class SourceMetadata(BaseModel):
    """Server-side enrichment of a generated citation. Never trusted from
    generated text: title/url/publisher/type/review all come from the
    retrieved verified source row."""

    source_id: str
    title: str
    source_type: str
    url: str
    publisher: str
    reviewed_at: datetime | None
    verification_status: Literal["verified"]


class KnowledgeCard(BaseModel):
    common_name: str
    scientific_name: str | None
    taxonomy_status: str
    physical_characteristics: str | None
    taste: str | None
    texture: str | None
    processing_methods: list[str]
    commercial_uses: list[str]
    similar_or_substitute_species: list[str]
    potential_buyer_segments: list[str]
    limitations: list[str]
    sources: list[SourceMetadata]


class KnowledgeResponse(BaseModel):
    prediction_id: str
    species_id: str
    card: KnowledgeCard


# Relational taxonomy overrides: per species label, the scientific name the
# card must carry and the limitation appended when the LLM would narrow or
# rename it. Keyed by normalized label; the values mirror the taxonomy CSV.
TAXONOMY_GUARDRAILS: dict[str, tuple[str | None, str]] = {
    "tuna": (
        "Thunnus spp.",
        "The common name 'tuna' covers several species (possibly also skipjack or "
        "bonito, Katsuwonus/Euthynnus); taxonomy is locked at the genus Thunnus spp. "
        "until expert verification.",
    ),
    "gembolo": (
        None,
        "The market name 'gembolo' is ambiguous: it refers to different species by "
        "region (Rastrelliger spp., Selaroides leptolepis, Caranx spp.); the scientific "
        "name cannot be confirmed without expert identification.",
    ),
    "tenggiri": (
        "Scomberomorus commerson",
        "The market name 'tenggiri' is also used for Scomberomorus guttatus "
        "(Indo-Pacific king mackerel); this card follows its main vernacular species, "
        "Scomberomorus commerson.",
    ),
}


def _relational_identity(species: SpeciesRecord) -> tuple[str, str | None, str, list[str]]:
    """Relational taxonomy always wins over the LLM: common name, scientific
    name, and status come from the species record, with per-label guardrails
    applied and their limitations appended."""
    scientific_name = species.scientific_name
    limitations: list[str] = []
    if species.normalized_label in TAXONOMY_GUARDRAILS:
        forced_name, limitation = TAXONOMY_GUARDRAILS[species.normalized_label]
        scientific_name = forced_name
        limitations.append(limitation)
    return species.common_name_id, scientific_name, species.taxonomy_status, limitations


class KnowledgeGenerator:
    """Grounded card builder: relational taxonomy wins, citations are
    validated against the retrieved evidence, and source metadata is enriched
    server-side. With no evidence the card is built without ever constructing
    or calling OpenCode.

    ``generator`` is an OpenCode client-like object, or a zero-arg factory
    returning one (production laziness: the factory is only invoked when
    evidence exists, so a blank-key client is never built for empty evidence).

    ``empty_card`` and ``build_card`` are public so the multi-agent
    orchestrator reuses these exact fail-closed invariants instead of
    reimplementing them (HANDOFF section 9, P0).
    """

    def __init__(self, generator=None):
        self._generator = generator

    def generate(self, species: SpeciesRecord, evidence: list[RetrievedChunk]) -> KnowledgeCard:
        if not evidence:
            return self.empty_card(species)
        try:
            client = self._generator() if callable(self._generator) else self._generator
        except ValueError as exc:
            # A blank key used to escape as an unhandled 500 (finding W3); it is
            # the same condition as an unreachable provider.
            raise OpenCodeUnavailable(
                "opencode go generation unavailable", [chunk.chunk_id for chunk in evidence]
            ) from exc
        generated = client.generate(SYSTEM_PROMPT, evidence, species)
        return self.build_card(species, evidence, generated)

    def empty_card(self, species: SpeciesRecord) -> KnowledgeCard:
        common_name, scientific_name, taxonomy_status, limitations = _relational_identity(species)
        return KnowledgeCard(
            common_name=common_name,
            scientific_name=scientific_name,
            taxonomy_status=taxonomy_status,
            physical_characteristics=None,
            taste=None,
            texture=None,
            processing_methods=[],
            commercial_uses=[],
            similar_or_substitute_species=[],
            potential_buyer_segments=[],
            limitations=[NO_INFORMATION] + limitations,
            sources=[],
        )

    def build_card(
        self,
        species: SpeciesRecord,
        evidence: list[RetrievedChunk],
        raw: str | GeneratedKnowledgeCard,
    ) -> KnowledgeCard:
        chunk_ids = [chunk.chunk_id for chunk in evidence]
        if isinstance(raw, GeneratedKnowledgeCard):
            generated = raw
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InvalidGeneratedKnowledge(
                    f"generated knowledge is not valid JSON: {exc}", chunk_ids
                ) from exc
            try:
                generated = GeneratedKnowledgeCard.model_validate(payload)
            except ValidationError as exc:
                raise InvalidGeneratedKnowledge(
                    f"generated knowledge failed schema validation: {exc}", chunk_ids
                ) from exc
        if not generated.sources:
            raise InvalidGeneratedKnowledge(
                "generated knowledge must cite at least one supplied source "
                "when evidence exists",
                chunk_ids,
            )
        by_source = {chunk.source_id: chunk for chunk in evidence}
        for citation in generated.sources:
            if citation.source_id not in by_source:
                raise InvalidGeneratedKnowledge(
                    f"generated knowledge cites unretrieved source {citation.source_id!r}",
                    chunk_ids,
                )

        common_name, scientific_name, taxonomy_status, guardrail_limitations = _relational_identity(species)
        return KnowledgeCard(
            common_name=common_name,
            scientific_name=scientific_name,
            taxonomy_status=taxonomy_status,
            physical_characteristics=generated.physical_characteristics,
            taste=generated.taste,
            texture=generated.texture,
            processing_methods=generated.processing_methods,
            commercial_uses=generated.commercial_uses,
            similar_or_substitute_species=generated.similar_or_substitute_species,
            potential_buyer_segments=generated.potential_buyer_segments,
            limitations=generated.limitations + guardrail_limitations,
            sources=[
                SourceMetadata(
                    source_id=citation.source_id,
                    title=by_source[citation.source_id].source_title,
                    source_type=by_source[citation.source_id].source_type,
                    url=by_source[citation.source_id].source_url or "",
                    publisher=by_source[citation.source_id].source_publisher or "",
                    reviewed_at=by_source[citation.source_id].source_reviewed_at,
                    verification_status="verified",
                )
                for citation in generated.sources
            ],
        )
