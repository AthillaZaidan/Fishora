"""Card orchestration for Fishora: researcher, four experts, critic, writer.

The graph is plain Python with an asyncio fan-out, so the four experts really
overlap (asyncio.gather over worker threads, since the LLM calls block on I/O).
``grade_card`` is the only way a card is made: the background job and the
synchronous path (manual entry, publication fallback) both run it (W19).
Every run records a stage trace (evidence, what each expert saw, prompt
hashes, verdicts, timings, tokens) that the job row keeps (W12).

Grounding is fail-closed and centralised: the critic grades every claim
against the specific chunk it cites, the writer keeps only ``supported``
claims, and the card itself is built by ``KnowledgeGenerator`` so this path
inherits the same citation and empty-evidence invariants as the sync one.

Iteration 1 (checkpoint-2) changes, each tied to a baseline finding:
cross-lingual grounding instead of lexical overlap (W1); replies read through
``llm_output`` so content blocks and fenced JSON parse (W21, W4); no LLM
sub-query and no post-grading polish, two LLM rounds instead of four (W5);
a card cache keyed by the evidence (W6); experts with no evidence are not
called (W11); whole chunks, not 300 characters (W14); Indonesian output is
required (W22); a card with nothing groundable is an honest empty card rather
than a failed job (W24); and failures are logged, not swallowed (W12).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import math
import re
import threading
import time
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ValidationError

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.errors import InvalidGeneratedKnowledge
from apps.main_api.services.generation import (
    GeneratedKnowledgeCard,
    KnowledgeCard,
    KnowledgeGenerator,
)
from apps.main_api.services.llm_output import reply_json
from apps.main_api.services.retrieval import CATEGORY_ORDER, VerifiedRetriever

logger = logging.getLogger(__name__)

EXPERT_NAMES = ("physical", "taste", "commercial", "substitute")

# Bump when a prompt or grading rule changes: it is part of the card cache key.
PIPELINE_VERSION = "iteration-1"

# Cross-lingual grounding threshold on E5 symmetric cosine (query:/query:).
# Chosen on the dev split of evals/datasets/grounding_claims.json together
# with the number check below; see evals/calibrate_grounding.py.
GROUNDING_TAU = 0.805
# The critic's LLM judge pass is off. Measured on the grounding pairs and the
# negation/scope traps (evals/experiment_nli_grounding.py, R2), adding
# gpt-5.6-luna as a downgrade-only judge cut held-out recall from 1.0 to 0.5
# and still accepted 3 of 14 traps; the verifier alone scores higher on the
# dev split. It also costs one sequential LLM round per card (W5).
USE_LLM_JUDGE = False
NO_GROUNDED_CLAIM = "Belum ada klaim yang dapat diverifikasi dari bukti yang tersedia."
# Indonesian field names for the missing-evidence limitation (R4).
_FIELD_LABELS = {
    "physical_characteristics": "ciri fisik",
    "taste": "rasa",
    "texture": "tekstur",
    "processing_methods": "cara pengolahan",
    "commercial_uses": "penggunaan komersial",
    "similar_or_substitute_species": "spesies pengganti",
    "potential_buyer_segments": "segmen pembeli",
}


def missing_evidence_limitation(evidence) -> str | None:
    """One limitation naming the card fields no verified chunk can support,
    so an empty field reads as missing evidence rather than as nothing to say."""
    present = {chunk.category for chunk in evidence}
    missing = [label for field, label in _FIELD_LABELS.items() if not (FIELD_EVIDENCE[field] & present)]
    return f"Belum ada bukti terverifikasi untuk: {', '.join(missing)}." if missing else None


class ClaimStatus(BaseModel):
    """Per-claim grounding verdict. ``chunk_ids`` are the specific verified
    chunks that carry the claim; the writer accepts only ``supported``."""

    field: str
    status: Literal["supported", "unsupported", "no_evidence"]
    chunk_ids: list[str]
    reason: str


def merge_expert_outputs(left: dict | None, right: dict | None) -> dict:
    """Fan-out reducer: four experts write the same state key concurrently, and
    without a reducer langgraph rejects that and a plain dict loses updates."""
    return {**(left or {}), **(right or {})}


class FishoraState(TypedDict, total=False):
    job_id: str
    prediction_id: str
    species_id: str
    species: SpeciesRecord
    broad_evidence: list[RetrievedChunk]
    refined_evidence: list[RetrievedChunk]
    expert_outputs: Annotated[dict, merge_expert_outputs]
    claim_statuses: list[ClaimStatus]
    critic_feedback: str | None
    final_card: KnowledgeCard | dict | None
    error: str | None
    cache_hit: bool
    known_binomials: tuple
    trace: dict


# ---- Researcher -----------------------------------------------------------

CARD_QUERY = (
    "Buat kartu pengetahuan bahasa Indonesia untuk {common_name}: identitas, "
    "ciri fisik, rasa dan tekstur, cara pengolahan, penggunaan komersial, dan "
    "spesies pengganti."
)


def hybrid_researcher(state: FishoraState, knowledge_repo, embedder, llm_medium=None) -> dict:
    """The species' verified evidence: the whole slice while it is small (R1),
    ranked category-first above FULL_CONTEXT_MAX. The former LLM sub-query is
    gone (W5): the species filter already returns every row of the species,
    so a second query could only reorder the same evidence."""
    species_id = state["species_id"]
    species = state.get("species")
    common = species.common_name_id if species else species_id
    evidence = VerifiedRetriever(knowledge_repo, embedder).card_evidence(
        species_id, CARD_QUERY.format(common_name=common)
    )
    return {"broad_evidence": evidence, "refined_evidence": list(evidence)}


# ---- Expert nodes (luna) -----------------------------------------------

_INDONESIAN = " Tulis semua nilai dalam bahasa Indonesia, walaupun buktinya berbahasa Inggris."

_EXPERT_PROMPTS = {
    "physical": "Tulis physical_characteristics dari bukti kategori physical_characteristics dan identity. Jika tidak ada, null. Jawab JSON {\"physical_characteristics\": str|null, \"sources\": [{\"source_id\": str, \"chunk_id\": str}]}" + _INDONESIAN,
    "taste": "Tulis taste dan texture dari bukti taste_texture. Jika tidak ada, null. JSON {\"taste\": str|null, \"texture\": str|null, \"sources\": [{\"source_id\": str, \"chunk_id\": str}]}" + _INDONESIAN,
    "commercial": "Tulis processing_methods dan commercial_uses dari bukti processing_methods dan commercial_uses. JSON {\"processing_methods\": [], \"commercial_uses\": [], \"sources\": [{\"source_id\": str, \"chunk_id\": str}]}" + _INDONESIAN,
    "substitute": "Tulis similar_or_substitute_species dan potential_buyer_segments dari bukti substitutes dan commercial_uses. JSON {\"similar_or_substitute_species\": [], \"potential_buyer_segments\": [], \"sources\": [{\"source_id\": str, \"chunk_id\": str}]}" + _INDONESIAN,
}

_EXPERT_CATEGORIES = {
    "physical": {"physical_characteristics", "identity"},
    "taste": {"taste_texture"},
    "commercial": {"processing_methods", "commercial_uses"},
    "substitute": {"substitutes", "commercial_uses"},
}

# Which card fields each expert is allowed to claim; drives per-claim grading.
_EXPERT_CLAIM_FIELDS = {
    "physical": ("physical_characteristics",),
    "taste": ("taste", "texture"),
    "commercial": ("processing_methods", "commercial_uses"),
    "substitute": ("similar_or_substitute_species", "potential_buyer_segments"),
}
_CLAIM_OWNER = {
    field: expert for expert, fields in _EXPERT_CLAIM_FIELDS.items() for field in fields
}
# Evidence categories that can support each card field (R4, W11). Substitutes
# need substitute evidence: a commercial-use chunk cannot ground a claim that
# one species replaces another, which is how the substitute field used to be
# filled while the corpus has no substitute chunks at all.
FIELD_EVIDENCE = {
    "physical_characteristics": {"physical_characteristics", "identity"},
    "taste": {"taste_texture"},
    "texture": {"taste_texture"},
    "processing_methods": {"processing_methods"},
    "commercial_uses": {"commercial_uses"},
    "similar_or_substitute_species": {"substitutes"},
    "potential_buyer_segments": {"commercial_uses"},
}

_EMPTY_EXPERT_CLAIMS = {
    "physical": {"physical_characteristics": None},
    "taste": {"taste": None, "texture": None},
    "commercial": {"processing_methods": [], "commercial_uses": []},
    "substitute": {"similar_or_substitute_species": [], "potential_buyer_segments": []},
}


def _bind_citations(raw_sources, subset: list[RetrievedChunk]) -> list[dict]:
    """Tie every citation to one specific retrieved chunk.

    A bare source_id is resolved only when the source contributes exactly one
    chunk to this expert's evidence, so the chunk is determined, not guessed.
    """
    by_chunk = {chunk.chunk_id: chunk for chunk in subset}
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in subset:
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk.chunk_id)
    bound: list[dict] = []
    seen: set[str] = set()
    for entry in raw_sources or []:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        if chunk_id not in by_chunk:
            candidates = chunks_by_source.get(entry.get("source_id"), [])
            chunk_id = candidates[0] if len(candidates) == 1 else None
        if chunk_id is None or chunk_id in seen:
            continue
        bound.append({"source_id": by_chunk[chunk_id].source_id, "chunk_id": chunk_id})
        seen.add(chunk_id)
    return bound[:3]


def _expert_node(category: str, state: FishoraState, llm_luna) -> dict:
    evidence = state.get("refined_evidence", [])
    if llm_luna is None:
        # No LLM available (tests / empty OpenCode Go key): claim nothing rather
        # than error, and let the writer decide whether that is fatal.
        return {**_EMPTY_EXPERT_CLAIMS[category], "sources": [], "skipped": True}
    # No cross-category fallback: giving an expert evidence outside its own
    # categories is exactly how off-topic claims get a plausible citation.
    subset = [c for c in evidence if c.category in _EXPERT_CATEGORIES[category]]
    if not subset:
        # Nothing to ground on: calling the model could only produce a borrowed
        # fact (W11), and it costs a call.
        return {**_EMPTY_EXPERT_CLAIMS[category], "sources": [], "skipped": True}
    payload = "\n".join(
        f"[chunk_id: {c.chunk_id}] [source_id: {c.source_id}] [{c.category}] {c.content}"
        for c in subset
    )
    prompt = _EXPERT_PROMPTS[category] + f"\nBukti:\n{payload}"
    trace = {"chunk_ids": [c.chunk_id for c in subset],
             "prompt_sha256": hashlib.sha256(_EXPERT_PROMPTS[category].encode("utf-8")).hexdigest()[:16]}
    started = time.perf_counter()
    try:
        reply = llm_luna.invoke(prompt)
        trace["tokens"] = _usage(reply)
        data = reply_json(reply)
        data["sources"] = _bind_citations(data.get("sources"), subset)
    except Exception as exc:
        logger.warning("job %s: expert %s failed: %s: %s", state.get("job_id"), category, type(exc).__name__, exc)
        data = {**_EMPTY_EXPERT_CLAIMS[category], "error": "expert generation failed", "sources": []}
    trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    # Carried out under a private key; the executor moves it into the stage
    # trace, so it never reaches the stored expert outputs.
    data[_TRACE_KEY] = trace
    return data


_TRACE_KEY = "_trace"


def _usage(reply) -> dict:
    """Token counts from a LangChain reply's usage_metadata, when the provider sends them."""
    usage = getattr(reply, "usage_metadata", None) or {}
    out = {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens") if isinstance(usage.get(key), int)}
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    if isinstance(reasoning, int):
        out["reasoning_tokens"] = reasoning
    return out


def physical_expert(state: FishoraState, llm_luna) -> dict:
    return {"expert_outputs": {"physical": _expert_node("physical", state, llm_luna)}}


def taste_expert(state: FishoraState, llm_luna) -> dict:
    return {"expert_outputs": {"taste": _expert_node("taste", state, llm_luna)}}


def commercial_expert(state: FishoraState, llm_luna) -> dict:
    return {"expert_outputs": {"commercial": _expert_node("commercial", state, llm_luna)}}


def substitute_expert(state: FishoraState, llm_luna) -> dict:
    return {"expert_outputs": {"substitute": _expert_node("substitute", state, llm_luna)}}


# ---- Critic -----------------------------------------------------------

# Function words carry no grounding signal, so overlap on them would let any
# citation pass the content check.
_STOPWORDS = frozenset({
    "adalah", "akan", "atau", "bagi", "banyak", "bisa", "dalam", "dapat",
    "dari", "dengan", "hingga", "ikan", "itu", "juga", "karena", "kemudian",
    "lain", "lebih", "namun", "oleh", "pada", "paling", "sangat", "sebagai",
    "serta", "setelah", "sudah", "telah", "terhadap", "tetapi", "tidak",
    "untuk", "yang",
})
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Latin family and order names; Indonesian words never end like this.
_TAXON = re.compile(r"\b[A-Z][a-z]+(?:idae|inae|iformes)\b")


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[0-9a-z]{4,}", text.lower())} - _STOPWORDS


def _numbers(text: str) -> set[str]:
    """Numbers normalised across the Indonesian decimal comma (43,5 == 43.5)."""
    return {m.group().replace(",", ".") for m in _NUMBER.finditer(text)}


def _taxa(text: str, known_binomials: frozenset = frozenset()) -> set[str]:
    """Family/order names, plus any supported species' binomial, named in text."""
    lowered = text.lower()
    return {m.group().lower() for m in _TAXON.finditer(text)} | {b for b in known_binomials if b in lowered}


def _claim_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(str(item) for item in value)


def _is_verified(chunk: RetrievedChunk | None) -> bool:
    return (
        chunk is not None
        and chunk.chunk_verification_status == "verified"
        and chunk.source_verification_status == "verified"
    )


class _Verifier:
    """Cross-lingual grounding g*: E5 symmetric cosine >= tau, and every number
    in the claim appears in the cited chunk, and so does every taxon it names
    (family/order names, supported species' binomials): translation keeps
    numbers and Latin names, so exact checks are safe where a judge is not. Replaces lexical overlap, which scored an
    Indonesian translation as fabrication (W1). Chunk vectors are cached per
    critic call."""

    def __init__(self, embedder, tau: float = GROUNDING_TAU, known_binomials=()):
        self._embedder = embedder
        self._tau = tau
        self._known = frozenset(b.lower() for b in known_binomials if b and " " in b and "spp" not in b)
        self._vectors: dict[str, list[float]] = {}

    def _vector(self, key: str, text: str) -> list[float]:
        if key not in self._vectors:
            self._vectors[key] = self._embedder.embed_query(text)
        return self._vectors[key]

    def supports(self, claim: str, chunk: RetrievedChunk) -> bool:
        if not _numbers(claim) <= _numbers(chunk.content):
            return False
        # A borrowed identity names another taxon: "famili Chanidae" cited
        # against a croaker chunk is close in embedding space but wrong.
        if not _taxa(claim, self._known) <= _taxa(chunk.content, self._known):
            return False
        a = self._vector(f"claim:{claim}", claim)
        b = self._vector(f"chunk:{chunk.chunk_id}", chunk.content)
        cosine = sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)
        return cosine >= self._tau


def _grade_claim(field: str, value, citations, by_chunk: dict, verifier: _Verifier | None = None) -> ClaimStatus:
    """Grade one claim against the verified chunks it cites."""
    text = _claim_text(value).strip()
    if not text:
        return ClaimStatus(field=field, status="no_evidence", chunk_ids=[], reason="tidak ada klaim")
    allowed = FIELD_EVIDENCE.get(field)
    cited = [c["chunk_id"] for c in citations
             if _is_verified(by_chunk.get(c.get("chunk_id")))
             and (allowed is None or by_chunk[c["chunk_id"]].category in allowed)]
    if not cited:
        return ClaimStatus(
            field=field, status="unsupported", chunk_ids=[],
            reason="klaim tidak terikat pada chunk terverifikasi",
        )
    if verifier is not None:
        grounded = [cid for cid in cited if verifier.supports(text, by_chunk[cid])]
    else:  # no embedder (unit tests): the former lexical rule
        claim_tokens = _tokens(text)
        grounded = [cid for cid in cited if claim_tokens & _tokens(by_chunk[cid].content)]
    if not grounded:
        return ClaimStatus(
            field=field, status="unsupported", chunk_ids=[],
            reason="isi chunk yang disitasi tidak mendukung klaim",
        )
    return ClaimStatus(
        field=field, status="supported", chunk_ids=grounded,
        reason="didukung isi chunk terverifikasi",
    )


def _llm_downgrade(statuses: list[ClaimStatus], by_chunk: dict, llm_medium) -> list[ClaimStatus]:
    """One bounded adjudication pass. It may only downgrade: an LLM outage or a
    malformed answer must never turn an ungrounded claim into a supported one."""
    supported = [s for s in statuses if s.status == "supported"]
    if not supported:
        return statuses
    claims = {
        s.field: [by_chunk[cid].content for cid in s.chunk_ids] for s in supported
    }
    import json

    prompt = (
        "Untuk setiap field, tentukan apakah kutipan bukti mendukung klaim. "
        "Jawab JSON {field: \"supported\"|\"unsupported\"}.\n"
        f"Klaim dan bukti: {json.dumps(claims, ensure_ascii=False)}"
    )
    try:
        verdicts = reply_json(llm_medium.invoke(prompt))
    except Exception as exc:
        logger.warning("critic downgrade pass skipped: %s", type(exc).__name__)
        return statuses
    return [
        s.model_copy(update={"chunk_ids": [], "status": "unsupported", "reason": "ditolak critic LLM"})
        if s.status == "supported" and verdicts.get(s.field) == "unsupported"
        else s
        for s in statuses
    ]


def critic_node(state: FishoraState, llm_medium=None, embedder=None) -> dict:
    evidence = state.get("refined_evidence", [])
    by_chunk = {chunk.chunk_id: chunk for chunk in evidence}
    outputs = state.get("expert_outputs", {})
    verifier = (_Verifier(embedder, known_binomials=state.get("known_binomials", ()))
                if embedder is not None else None)
    statuses: list[ClaimStatus] = []
    for expert, fields in _EXPERT_CLAIM_FIELDS.items():
        data = outputs.get(expert) or {}
        citations = [c for c in data.get("sources", []) if isinstance(c, dict)]
        for field in fields:
            statuses.append(_grade_claim(field, data.get(field), citations, by_chunk, verifier))
    if USE_LLM_JUDGE and llm_medium is not None:
        statuses = _llm_downgrade(statuses, by_chunk, llm_medium)
    feedback = "; ".join(f"{s.field}={s.status}" for s in statuses)
    return {"claim_statuses": statuses, "critic_feedback": feedback}


# ---- Writer ----------------------------------------------------------------

def _supported_claims(outputs: dict, statuses: list[ClaimStatus]) -> dict:
    """Only ``supported`` claims survive into the card."""
    claims: dict = {}
    for status in statuses:
        if status.status != "supported":
            continue
        owner = _CLAIM_OWNER.get(status.field)
        value = (outputs.get(owner) or {}).get(status.field)
        if value not in (None, [], ""):
            claims[status.field] = value
    return claims


def writer_node(state: FishoraState, llm_luna=None, species: SpeciesRecord | None = None) -> dict:
    """Build the card from supported claims only. The former LLM "polish" pass
    is gone (W5): it rewrote text after the critic had verified it."""
    sp = species or state.get("species")
    if sp is None:
        return {"error": "knowledge generation failed: no species record", "final_card": None}
    # Never re-trust a row's verification status from earlier in the graph.
    evidence = [c for c in state.get("refined_evidence", []) if _is_verified(c)]
    generator = KnowledgeGenerator()
    if not evidence:
        return {"final_card": generator.empty_card(sp)}

    outputs = state.get("expert_outputs", {})
    statuses = state.get("claim_statuses") or []
    by_chunk = {chunk.chunk_id: chunk for chunk in evidence}
    claims = _supported_claims(outputs, statuses)

    if not claims:
        called = [o for o in outputs.values() if isinstance(o, dict) and not o.get("skipped")]
        if called and all(o.get("error") for o in called):
            # Every expert that ran failed: that is an outage, not an empty card.
            return {"error": "knowledge generation failed: every expert failed", "final_card": None}
        # Evidence exists but nothing in it could be grounded (e.g. a species
        # whose only chunk is a limitation note): say so instead of failing (W24).
        card = generator.empty_card(sp)
        gap = missing_evidence_limitation(evidence)
        card.limitations = [NO_GROUNDED_CLAIM] + ([gap] if gap else []) + card.limitations[1:]
        return {"final_card": card}

    # Cite only the chunks that actually carried a surviving claim.
    cited: list[str] = []
    for status in statuses:
        if status.status != "supported" or status.field not in claims:
            continue
        for chunk_id in status.chunk_ids:
            source_id = by_chunk[chunk_id].source_id
            if source_id not in cited:
                cited.append(source_id)

    try:
        generated = GeneratedKnowledgeCard(
            common_name=sp.common_name_id,
            scientific_name=sp.scientific_name,
            taxonomy_status=sp.taxonomy_status,
            physical_characteristics=claims.get("physical_characteristics"),
            taste=claims.get("taste"),
            texture=claims.get("texture"),
            processing_methods=claims.get("processing_methods", []),
            commercial_uses=claims.get("commercial_uses", []),
            similar_or_substitute_species=claims.get("similar_or_substitute_species", []),
            potential_buyer_segments=claims.get("potential_buyer_segments", []),
            limitations=[text for text in (missing_evidence_limitation(evidence),) if text],
            sources=[{"source_id": source_id} for source_id in cited],
        )
    except ValidationError:
        return {"error": "knowledge validation failed", "final_card": None}
    # build_card is the fail-closed gate: evidence with no citation raises here
    # rather than completing the job with an empty, uncited card.
    try:
        card = generator.build_card(sp, evidence, generated)
    except InvalidGeneratedKnowledge:
        return {"error": "knowledge generation failed: no grounded claim", "final_card": None}
    return {"final_card": card}


# ---- Card cache ------------------------------------------------------------

_CARD_CACHE: dict[str, KnowledgeCard] = {}
_CARD_CACHE_LOCK = threading.Lock()


def card_cache_key(species_id: str, evidence: list[RetrievedChunk], llm) -> str:
    """Species, evidence ids and contents, model and pipeline version (W6).
    Any corpus change changes the key, so the cache invalidates itself."""
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__
    digest = hashlib.sha256()
    for part in (species_id, str(model), PIPELINE_VERSION):
        digest.update(part.encode("utf-8") + b"\0")
    for chunk in sorted(evidence, key=lambda c: c.chunk_id):
        digest.update(f"{chunk.chunk_id}\0{chunk.content}\0".encode("utf-8"))
    return digest.hexdigest()


def clear_card_cache() -> None:
    with _CARD_CACHE_LOCK:
        _CARD_CACHE.clear()


# ---- Graph factory -----------------------------------------------------

def make_graph(knowledge_repo=None, embedder=None, llm_luna=None, llm_medium=None):
    """Return the card graph: an object with .invoke(state) and .ainvoke(state).

    Researcher, then the four experts in parallel, then the critic and the
    writer. The result carries ``trace``, the stage trace of this run (W12).
    The former optional LangGraph build is gone: langgraph was never installed,
    so that branch never ran and was never tested.
    """

    class SimpleGraph:
        async def ainvoke(self, state: FishoraState) -> FishoraState:
            s = dict(state)
            timings: dict[str, float] = {}
            started = time.perf_counter()

            def lap(stage: str, since: float) -> float:
                now = time.perf_counter()
                timings[stage] = round((now - since) * 1000, 1)
                return now

            s.update(hybrid_researcher(s, knowledge_repo, embedder, llm_medium))
            mark = lap("research", started)
            evidence = s.get("refined_evidence", [])
            trace: dict = {
                "pipeline_version": PIPELINE_VERSION,
                "model": _model_name(llm_luna),
                "evidence": [{"chunk_id": c.chunk_id, "category": c.category,
                              "distance": None if c.distance is None else round(float(c.distance), 4)}
                             for c in evidence],
            }
            key = card_cache_key(s["species_id"], evidence, llm_luna)
            with _CARD_CACHE_LOCK:
                cached = _CARD_CACHE.get(key)
            if cached is not None:
                lap("total", started)
                s.update({"final_card": copy.deepcopy(cached), "cache_hit": True,
                          "critic_feedback": "cache hit",
                          "trace": {**trace, "cache_hit": True, "timings_ms": timings}})
                return s
            # Expert LLM calls block on network I/O, so real overlap needs
            # threads; gather also keeps the single merge point for the reducer.
            results = await asyncio.gather(
                *(asyncio.to_thread(_expert_node, name, s, llm_luna) for name in EXPERT_NAMES)
            )
            experts = {}
            for name, result in zip(EXPERT_NAMES, results):
                expert_trace = result.pop(_TRACE_KEY, {})
                experts[name] = {**expert_trace, "skipped": bool(result.get("skipped")),
                                 "error": bool(result.get("error"))}
            s["expert_outputs"] = merge_expert_outputs(
                s.get("expert_outputs"), dict(zip(EXPERT_NAMES, results))
            )
            mark = lap("experts", mark)
            s.update(critic_node(s, llm_medium, embedder))
            mark = lap("critic", mark)
            s.update(writer_node(s, llm_luna, s.get("species")))
            lap("writer", mark)
            lap("total", started)
            s["trace"] = {
                **trace,
                "cache_hit": False,
                "experts": experts,
                "claims": [status.model_dump() for status in s.get("claim_statuses") or []],
                "error": s.get("error"),
                "timings_ms": timings,
            }
            final = s.get("final_card")
            if final is not None and not s.get("error") and llm_luna is not None:
                with _CARD_CACHE_LOCK:
                    _CARD_CACHE[key] = copy.deepcopy(final)
            return s

        def invoke(self, state: FishoraState) -> FishoraState:
            # Callers are sync (a BackgroundTasks worker thread, or a threadpool
            # request handler), so there is no loop to reuse here.
            return asyncio.run(self.ainvoke(state))

    return SimpleGraph()


def _model_name(llm) -> str | None:
    if llm is None:
        return None
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__


def known_binomials(species_repo) -> tuple[str, ...]:
    """Scientific names the critic's taxon check accepts in a claim."""
    if species_repo is None or not hasattr(species_repo, "list_all"):
        return ()
    return tuple(s.scientific_name for s in species_repo.list_all() if s.scientific_name)


def grade_card(species_id: str, *, prediction_id: str, knowledge_repo, embedder, llm,
               species_repo, job_id: str | None = None) -> FishoraState:
    """One card through researcher, experts, critic and writer.

    The only way a card is made (W19): the background job and the synchronous
    path both call this, so every published claim has passed the per-claim
    critic. Returns the final state; ``final_card`` and ``error`` say how it went
    and ``trace`` records the run.
    """
    species = species_repo.get_by_id(species_id) if species_repo else None
    graph = make_graph(knowledge_repo, embedder, llm, llm)
    state: FishoraState = {"job_id": job_id or f"sync-{prediction_id}", "prediction_id": prediction_id,
                           "species_id": species_id, "species": species, "expert_outputs": {},
                           "known_binomials": known_binomials(species_repo)}
    result = graph.invoke(state)
    trace = result.get("trace") or {}
    logger.info("card %s species=%s cache_hit=%s error=%s timings_ms=%s", state["job_id"], species_id,
                trace.get("cache_hit"), result.get("error"), trace.get("timings_ms"))
    return result


def run_graph(job_id: str, species_id: str, prediction_id: str, knowledge_repo, embedder, llm_luna, llm_medium, species_repo, job_repo):
    """Background entry: grades the card and persists it with its trace. Sync for BackgroundTasks.

    ``llm_medium`` is accepted for the existing call signature; the critic's LLM
    judge pass is off (USE_LLM_JUDGE), so one LLM serves the whole graph.
    """
    try:
        result = grade_card(species_id, prediction_id=prediction_id, knowledge_repo=knowledge_repo,
                            embedder=embedder, llm=llm_luna, species_repo=species_repo, job_id=job_id)
        final = result.get("final_card")
        err = result.get("error")
        trace = result.get("trace")
        if final is not None and err is None:
            data = final.model_dump(mode="json") if hasattr(final, "model_dump") else final
            job_repo.update(job_id, status="completed", final_card=data, expert_outputs=result.get("expert_outputs"),
                            critic_feedback=result.get("critic_feedback"), trace=trace)
        else:
            logger.warning("knowledge job %s failed: %s", job_id, err)
            # generic error in the response; expert outputs, verdicts and trace kept for debugging
            job_repo.update(job_id, status="failed", error="knowledge generation failed",
                            expert_outputs=result.get("expert_outputs"), critic_feedback=result.get("critic_feedback"),
                            trace=trace)
    except Exception:
        logger.exception("knowledge job %s crashed", job_id)
        try:
            job_repo.update(job_id, status="failed", error="knowledge generation failed")
        except Exception:
            logger.exception("could not mark knowledge job %s failed", job_id)
