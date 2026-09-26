"""Writer + Critic card workflow (iteration 2, the alternative to the 4-expert graph).

    evidence loader (no LLM) -> writer (1 LLM call, every field, one claim per item)
      -> claim_verifier (deterministic checks + 1 LLM call) -> assembler (no LLM)

If more than half of the writer's claims are rejected, the writer gets one
revision with the rejections and the verifier grades the revision once more
(an evaluator-optimizer loop capped at one retry). Nothing rewrites a claim
after it was verified. ``run_workflow`` has the same contract as
``orchestrator.run_graph`` so either can back the knowledge job.
"""

from __future__ import annotations

import copy
import json
import logging

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
    card_cache_key,
    missing_evidence_limitation,
)
from apps.main_api.services.retrieval import VerifiedRetriever

logger = logging.getLogger(__name__)

WORKFLOW_VERSION = "writer-critic-1"
MAX_REVISIONS = 1

WRITER_PROMPT = """Tulis klaim untuk kartu pengetahuan ikan dalam bahasa Indonesia, HANYA dari bukti di bawah (bukti berbahasa Inggris).

Aturan:
- Satu klaim = satu fakta atau satu butir daftar. Setiap klaim menyitasi chunk_id yang memuat faktanya.
- Field dan kategori bukti yang boleh dipakai:
  physical_characteristics <- identity, physical_characteristics; taste, texture <- taste_texture;
  processing_methods <- processing_methods; commercial_uses, potential_buyer_segments <- commercial_uses;
  similar_or_substitute_species <- substitutes.
- Pertahankan cakupan bukti: jika bukti menyebut produk olahan, populasi tertentu, proyeksi, atau satu spesies dari label multi-spesies, klaim harus menyebutnya juga.
- Jangan menyimpulkan hal yang tidak tertulis. Lebih baik field kosong daripada menebak.
Jawab hanya JSON: {"claims": [{"field": "<field>", "chunk_ids": ["<chunk_id>"], "teks": "<klaim>"}]}
"""

REVISION_PROMPT = """Beberapa klaimmu ditolak pemeriksa fakta. Tulis ulang daftar klaim: perbaiki atau buang klaim yang ditolak, pertahankan yang diterima. Aturan dan format sama seperti sebelumnya.
Klaim yang ditolak dan alasannya:
"""


def _evidence_block(species: SpeciesRecord | None, evidence: list[RetrievedChunk]) -> str:
    head = []
    if species is not None:
        head = [f"Spesies: {species.common_name_id} ({species.scientific_name or 'nama ilmiah tidak tetap'}), "
                f"status taksonomi {species.taxonomy_status}"]
    lines = [f"[chunk_id: {c.chunk_id}] [source_id: {c.source_id}] [{c.category}] {c.content}" for c in evidence]
    return "\n".join(head + ["Bukti:"] + lines)


def _parse_claims(reply) -> list[claim_verifier.Claim]:
    data = reply_json(reply)
    claims = []
    for item in data.get("claims", []):
        if not isinstance(item, dict):
            continue
        fld, text = item.get("field"), str(item.get("teks") or item.get("text") or "").strip()
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


def run_workflow(job_id: str, species_id: str, prediction_id: str, knowledge_repo, embedder, llm, _unused,
                 species_repo, job_repo):
    """Background entry with the same signature as ``orchestrator.run_graph``."""
    try:
        species = species_repo.get_by_id(species_id) if species_repo else None
        known = (tuple(s.scientific_name for s in species_repo.list_all() if s.scientific_name)
                 if species_repo is not None and hasattr(species_repo, "list_all") else ())
        common = species.common_name_id if species else species_id
        evidence = VerifiedRetriever(knowledge_repo, embedder).card_evidence(species_id, CARD_QUERY.format(common_name=common))
        evidence = [c for c in evidence if claim_verifier._verified(c)]
        if species is None:
            raise ValueError("no species record")
        if not evidence:
            job_repo.update(job_id, status="completed", final_card=KnowledgeGenerator().empty_card(species).model_dump(mode="json"))
            return
        key = card_cache_key(species_id, evidence, llm) + WORKFLOW_VERSION
        with _CARD_CACHE_LOCK:
            cached = _CARD_CACHE.get(key)
        if cached is not None:
            job_repo.update(job_id, status="completed", final_card=cached.model_dump(mode="json"), critic_feedback="cache hit")
            return
        out = generate(species, evidence, llm, known, embedder)
        result = out["result"]
        if result.llm_error and not result.supported():
            raise RuntimeError(f"verifier failed: {result.llm_error}")
        card = assemble(species, evidence, result)
        with _CARD_CACHE_LOCK:
            _CARD_CACHE[key] = copy.deepcopy(card)
        feedback = json.dumps({"revised": out["revised"], "llm_calls": out["llm_calls"],
                               "verdicts": [c.__dict__ for c in result.claims]}, ensure_ascii=False)
        job_repo.update(job_id, status="completed", final_card=card.model_dump(mode="json"), critic_feedback=feedback)
    except (InvalidGeneratedKnowledge, ValidationError) as exc:
        logger.warning("knowledge job %s produced an invalid card: %s", job_id, type(exc).__name__)
        job_repo.update(job_id, status="failed", error="knowledge generation failed")
    except Exception:
        logger.exception("knowledge job %s crashed", job_id)
        try:
            job_repo.update(job_id, status="failed", error="knowledge generation failed")
        except Exception:
            logger.exception("could not mark knowledge job %s failed", job_id)
