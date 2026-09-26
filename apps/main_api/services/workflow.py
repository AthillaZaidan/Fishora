"""Writer + Critic card workflow (iteration 2, the alternative to the 4-expert graph).

    evidence loader (no LLM) -> writer (1 LLM call, every field, one claim per item)
      -> claim_verifier (deterministic checks + 1 LLM call) -> assembler (no LLM)

If more than half of the writer's claims are rejected, the writer gets one
revision with the rejections and the verifier grades the revision once more
(an evaluator-optimizer loop capped at one retry). Nothing rewrites a claim
after it was verified. ``run_workflow`` has the same contract as
``orchestrator.run_graph`` so either can back the knowledge job; both it and
the synchronous path go through ``workflow_card`` (W19), which records a stage
trace (W12).
"""

from __future__ import annotations

import copy
import json
import logging
import time

from pydantic import ValidationError

from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.errors import InvalidGeneratedKnowledge
from apps.main_api.services import claim_verifier
from apps.main_api.services.generation import GeneratedKnowledgeCard, KnowledgeGenerator
from apps.main_api.services.llm_output import reply_json
from apps.main_api.services.orchestrator import (
    _CARD_CACHE,
    _CARD_CACHE_LOCK,
    CARD_QUERY,
    NO_GROUNDED_CLAIM,
    _model_name,
    card_cache_key,
    known_binomials,
    missing_evidence_limitation,
)
from apps.main_api.services.retrieval import VerifiedRetriever

logger = logging.getLogger(__name__)

WORKFLOW_VERSION = "writer-critic-2-en"
MAX_REVISIONS = 1

WRITER_PROMPT = """Write claims for a fish knowledge card in plain English, ONLY from the evidence below.

Rules:
- One claim = one fact or one list item. Every claim cites the chunk_id that carries its fact.
- Fields and the evidence categories each may use:
  physical_characteristics <- identity, physical_characteristics; taste, texture <- taste_texture;
  processing_methods <- processing_methods; commercial_uses, potential_buyer_segments <- commercial_uses;
  similar_or_substitute_species <- substitutes.
- Keep the scope of the evidence: if the evidence is about a processed product, a specific population, a projection, or one species of a multi-species label, the claim must say so too.
- Do not infer anything that is not written. An empty field is better than a guess.
- Name fish by their English common name when the evidence gives one.
Answer only JSON: {"claims": [{"field": "<field>", "chunk_ids": ["<chunk_id>"], "text": "<claim>"}]}
"""

REVISION_PROMPT = """The fact checker rejected some of your claims. Rewrite the claim list: fix or drop the rejected claims and keep the accepted ones. Same rules and format as before.
Rejected claims and the reasons:
"""


def _evidence_block(species: SpeciesRecord | None, evidence: list[RetrievedChunk]) -> str:
    head = []
    if species is not None:
        head = [f"Species: {species.common_name_id} ({species.scientific_name or 'scientific name not fixed'}), "
                f"taxonomy status {species.taxonomy_status}"]
    lines = [f"[chunk_id: {c.chunk_id}] [source_id: {c.source_id}] [{c.category}] {c.content}" for c in evidence]
    return "\n".join(head + ["Evidence:"] + lines)


def _parse_claims(reply) -> list[claim_verifier.Claim]:
    data = reply_json(reply)
    claims = []
    for item in data.get("claims", []):
        if not isinstance(item, dict):
            continue
        # "teks" is the key of the Indonesian-era prompt; still read so replayed replies parse.
        fld, text = item.get("field"), str(item.get("text") or item.get("teks") or "").strip()
        if fld not in claim_verifier.FIELD_EVIDENCE or not text:
            continue
        ids = [str(i) for i in item.get("chunk_ids") or [] if i]
        claims.append(claim_verifier.Claim(id=len(claims), field=fld, text=text, chunk_ids=ids))
    return claims


def _write(llm, block: str, extra: str = "") -> list[claim_verifier.Claim]:
    return _parse_claims(llm.invoke(WRITER_PROMPT + extra + "\n" + block))


def generate(species: SpeciesRecord | None, evidence: list[RetrievedChunk], llm, known_binomials=(), embedder=None) -> dict:
    """Writer -> verifier (-> one revision). Returns claims, verdicts and call counts."""
    block = _evidence_block(species, evidence)
    calls, revised = 1, False
    claims = _write(llm, block)
    result = claim_verifier.verify(claims, evidence, llm, known_binomials, embedder=embedder)
    calls += result.llm_calls
    rejected = [c for c in result.claims if c.label != "supported"]
    if result.claims and len(rejected) * 2 > len(result.claims) and MAX_REVISIONS > 0 and result.llm_error is None:
        notes = "\n".join(f"- ({c.field}) {c.text} -> {c.label}: {c.reason}" for c in rejected)
        try:
            revision = _write(llm, block, "\n" + REVISION_PROMPT + notes)
            calls += 1
            second = claim_verifier.verify(revision, evidence, llm, known_binomials, embedder=embedder)
            calls += second.llm_calls
            if len(second.supported()) >= len(result.supported()):
                result, revised = second, True
        except Exception as exc:
            logger.warning("writer revision failed: %s: %s", type(exc).__name__, exc)
    return {"result": result, "llm_calls": calls, "revised": revised}


def assemble(species: SpeciesRecord, evidence: list[RetrievedChunk], result: claim_verifier.VerificationResult):
    """Supported claims only -> the fail-closed citation gate in build_card."""
    generator = KnowledgeGenerator()
    kept = result.supported()
    if not kept:
        card = generator.empty_card(species)
        gap = missing_evidence_limitation(evidence)
        card.limitations = [NO_GROUNDED_CLAIM] + ([gap] if gap else []) + card.limitations[1:]
        return card
    by_chunk = {c.chunk_id: c for c in evidence}
    values: dict = {f: [] for f in claim_verifier.LIST_FIELDS}
    for fld in claim_verifier.TEXT_FIELDS:
        texts = [c.text for c in kept if c.field == fld]
        values[fld] = " ".join(texts) or None
    for c in kept:
        if c.field in claim_verifier.LIST_FIELDS and c.text not in values[c.field]:
            values[c.field].append(c.text)
    cited: list[str] = []
    for c in kept:
        for cid in c.chunk_ids:
            sid = by_chunk[cid].source_id
            if sid not in cited:
                cited.append(sid)
    generated = GeneratedKnowledgeCard(
        common_name=species.common_name_id, scientific_name=species.scientific_name,
        taxonomy_status=species.taxonomy_status, **values,
        limitations=[t for t in (missing_evidence_limitation(evidence),) if t],
        sources=[{"source_id": s} for s in cited],
    )
    return generator.build_card(species, evidence, generated)


def workflow_card(species_id: str, *, prediction_id: str, knowledge_repo, embedder, llm, species_repo,
                  job_id: str | None = None) -> dict:
    """One card through the writer-critic workflow.

    The only way a card is made (W19): the background job (``run_workflow``) and
    the synchronous path (``KnowledgeService``: a manually declared catch, the
    publication fallback) both call it, so every card a buyer sees has passed the
    claim verifier. Returns ``final_card`` (or None), ``error``, ``critic_feedback``
    and ``trace``, the stage trace of this run (W12).
    """
    run_id = job_id or f"sync-{prediction_id}"
    started = time.perf_counter()
    timings: dict[str, float] = {}
    species = species_repo.get_by_id(species_id) if species_repo else None
    known = known_binomials(species_repo)
    common = species.common_name_id if species else species_id
    evidence = VerifiedRetriever(knowledge_repo, embedder).card_evidence(species_id, CARD_QUERY.format(common_name=common))
    evidence = [c for c in evidence if claim_verifier._verified(c)]
    timings["research"] = round((time.perf_counter() - started) * 1000, 1)
    trace: dict = {
        "pipeline": WORKFLOW_VERSION,
        "model": _model_name(llm),
        "evidence": [{"chunk_id": c.chunk_id, "category": c.category,
                      "distance": None if c.distance is None else round(float(c.distance), 4)} for c in evidence],
        "cache_hit": False,
    }

    def done(**out) -> dict:
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        trace["timings_ms"] = timings
        trace["error"] = out.get("error")
        logger.info("card %s species=%s cache_hit=%s error=%s timings_ms=%s", run_id, species_id,
                    trace["cache_hit"], out.get("error"), timings)
        return {"final_card": None, "error": None, "critic_feedback": None, **out, "trace": trace}

    if species is None:
        return done(error="knowledge generation failed: no species record")
    if not evidence:
        return done(final_card=KnowledgeGenerator().empty_card(species))
    key = card_cache_key(species_id, evidence, llm) + WORKFLOW_VERSION
    with _CARD_CACHE_LOCK:
        cached = _CARD_CACHE.get(key)
    if cached is not None:
        trace["cache_hit"] = True
        return done(final_card=copy.deepcopy(cached), critic_feedback="cache hit")
    mark = time.perf_counter()
    out = generate(species, evidence, llm, known, embedder)
    timings["write_and_verify"] = round((time.perf_counter() - mark) * 1000, 1)
    result = out["result"]
    trace.update(llm_calls=out["llm_calls"], revised=out["revised"],
                 claims=[dict(c.__dict__) for c in result.claims], verifier_error=result.llm_error)
    if result.llm_error and not result.supported():
        return done(error=f"knowledge generation failed: verifier unavailable ({result.llm_error})")
    mark = time.perf_counter()
    try:
        card = assemble(species, evidence, result)
    except (InvalidGeneratedKnowledge, ValidationError) as exc:
        logger.warning("card %s produced an invalid card: %s", run_id, type(exc).__name__)
        return done(error="knowledge generation failed: invalid card")
    timings["assemble"] = round((time.perf_counter() - mark) * 1000, 1)
    with _CARD_CACHE_LOCK:
        _CARD_CACHE[key] = copy.deepcopy(card)
    feedback = json.dumps({"revised": out["revised"], "llm_calls": out["llm_calls"],
                           "verdicts": trace["claims"]}, ensure_ascii=False)
    return done(final_card=card, critic_feedback=feedback)


def run_workflow(job_id: str, species_id: str, prediction_id: str, knowledge_repo, embedder, llm, _unused,
                 species_repo, job_repo):
    """Background entry with the same signature as ``orchestrator.run_graph``;
    persists the card with its stage trace."""
    try:
        out = workflow_card(species_id, prediction_id=prediction_id, knowledge_repo=knowledge_repo,
                            embedder=embedder, llm=llm, species_repo=species_repo, job_id=job_id)
        card = out["final_card"]
        if card is not None and out["error"] is None:
            data = card.model_dump(mode="json") if hasattr(card, "model_dump") else card
            job_repo.update(job_id, status="completed", final_card=data, critic_feedback=out["critic_feedback"],
                            trace=out["trace"])
        else:
            logger.warning("knowledge job %s failed: %s", job_id, out["error"])
            job_repo.update(job_id, status="failed", error="knowledge generation failed", trace=out["trace"])
    except Exception:
        logger.exception("knowledge job %s crashed", job_id)
        try:
            job_repo.update(job_id, status="failed", error="knowledge generation failed")
        except Exception:
            logger.exception("could not mark knowledge job %s failed", job_id)
