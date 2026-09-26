"""Claim-level verifier shared by both card architectures (iteration 2).

Every claim is one list item, or one sentence of a prose field. It is checked
in two stages:

1. Deterministic, no LLM: it must cite a verified chunk whose category may
   ground its field, and every number and taxon it names must appear in the
   cited chunks. These checks are exact where a judge is not (translation keeps
   numbers and Latin names).
2. Embedding filter (optional, reject-only): E5 symmetric cosine to the best
   cited chunk must reach ``TAU``, the iteration-1 calibrated threshold.
3. One LLM call per card grades every surviving claim as ``supported``,
   ``inferred`` or ``unsupported`` against the full text of the chunks it cites,
   with the reason written before the label.

Only ``supported`` claims reach a card. The stage is fail-closed: a failed or
malformed LLM reply leaves every claim of that card unverified, and the failure
is logged rather than swallowed.

The iteration-1 critic (E5 cosine >= tau) scored a claim by embedding
similarity. On the locked gold set (evals/protocol/claims_gold.csv) it accepted
claims that restate a scope-less or inferred fact, and cosine cannot see a
negation (R2); both motivate an entailment judgement.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from apps.main_api.contracts import RetrievedChunk
from apps.main_api.services.llm_output import reply_json

logger = logging.getLogger(__name__)

Label = Literal["supported", "inferred", "unsupported"]

# Evidence categories that can ground each card field (same rule as the
# iteration-1 critic, R4/W11).
FIELD_EVIDENCE = {
    "physical_characteristics": {"physical_characteristics", "identity"},
    "taste": {"taste_texture"},
    "texture": {"taste_texture"},
    "processing_methods": {"processing_methods"},
    "commercial_uses": {"commercial_uses"},
    "similar_or_substitute_species": {"substitutes"},
    "potential_buyer_segments": {"commercial_uses"},
}
LIST_FIELDS = ("processing_methods", "commercial_uses", "similar_or_substitute_species", "potential_buyer_segments")
TEXT_FIELDS = ("physical_characteristics", "taste", "texture")

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_TAXON = re.compile(r"\b[A-Z][a-z]+(?:idae|inae|iformes)\b")
TAU = 0.805  # iteration-1 dev-split threshold (evals/calibrate_grounding.py)
# Stage 2 on/off. Measured trade-off on the locked gold claims (evals/iteration2_gold.py):
# it lowers leakage and recall together; evals/iteration2.py runs both variants.
USE_EMBEDDING_FILTER = True
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

VERIFIER_PROMPT = """You are a fact checker. Compare each claim with the evidence it cites, and only that evidence.

Labels:
- "supported": every fact in the claim (numbers, traits, uses, scope) is stated or directly implied by the evidence.
- "inferred": a reasonable derivation that is not written in the evidence, or a fact about one species generalised to a multi-species label without saying so.
- "unsupported": a fact is missing from the evidence or contradicts it (including a negation), the scope is wrong (e.g. a property of a processed product stated as a property of the fish, a projection stated as fact), or it misreads the evidence.

A claim may paraphrase the evidence or be written in another language; judge the meaning, not the wording.
Answer only JSON: {"verdicts": [{"id": <id>, "reason": "<short>", "label": "supported"|"inferred"|"unsupported"}]}
"""


@dataclass
class Claim:
    id: int
    field: str
    text: str
    chunk_ids: list[str]
    label: Label | None = None
    reason: str = ""
    stage: str = ""  # "deterministic", "embedding" or "llm"


@dataclass
class VerificationResult:
    claims: list[Claim]
    llm_calls: int = 0
    llm_error: str | None = None
    notes: list[str] = field(default_factory=list)

    def supported(self) -> list[Claim]:
        return [c for c in self.claims if c.label == "supported"]


def split_field(field_name: str, value) -> list[str]:
    """One claim per list item, or per sentence of a prose field."""
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in _SENTENCE.split(str(value)) if s.strip()]


def _numbers(text: str) -> set[str]:
    return {m.group().replace(",", ".") for m in _NUMBER.finditer(text)}


def _taxa(text: str, known: frozenset[str]) -> set[str]:
    """Family/order names, plus any known species binomial, named in text."""
    lowered = text.lower()
    return {m.group().lower() for m in _TAXON.finditer(text)} | {b for b in known if b in lowered}


def _named_in(taxon: str, text: str) -> bool:
    """A binomial abbreviated in the evidence ("S. commerson"), or a family
    named by its adjective ("scombrid" for Scombridae), still counts."""
    lowered = text.lower()
    if " " in taxon:
        genus, species = taxon.split(" ", 1)
        return re.search(rf"\b{re.escape(genus[0])}\.\s*{re.escape(species)}\b", lowered) is not None
    return taxon.endswith("idae") and re.search(rf"\b{re.escape(taxon[:-2])}s?\b", lowered) is not None


def _verified(chunk: RetrievedChunk | None) -> bool:
    return (chunk is not None and chunk.chunk_verification_status == "verified"
            and chunk.source_verification_status == "verified")


def _deterministic(claim: Claim, by_chunk: dict[str, RetrievedChunk], known: frozenset[str]) -> None:
    allowed = FIELD_EVIDENCE.get(claim.field, set())
    cited = [by_chunk[c] for c in claim.chunk_ids if _verified(by_chunk.get(c)) and by_chunk[c].category in allowed]
    claim.chunk_ids = [c.chunk_id for c in cited]
    if not cited:
        claim.label, claim.reason, claim.stage = "unsupported", "cites no verified chunk allowed for this field", "deterministic"
        return
    text = " ".join(c.content for c in cited)
    missing_numbers = _numbers(claim.text) - _numbers(text)
    if missing_numbers:
        claim.label, claim.stage = "unsupported", "deterministic"
        claim.reason = f"numbers not in the evidence: {', '.join(sorted(missing_numbers))}"
        return
    missing_taxa = {t for t in _taxa(claim.text, known) - _taxa(text, known) if not _named_in(t, text)}
    if missing_taxa:
        claim.label, claim.stage = "unsupported", "deterministic"
        claim.reason = f"taxon names not in the evidence: {', '.join(sorted(missing_taxa))}"


def _cosine(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _embedding_filter(pending: list[Claim], by_chunk: dict, embedder, tau: float) -> None:
    vectors: dict[str, list[float]] = {}

    def vec(key: str, text: str):
        if key not in vectors:
            vectors[key] = embedder.embed_query(text)
        return vectors[key]

    for c in pending:
        best = max(_cosine(vec("q:" + c.text, c.text), vec("c:" + cid, by_chunk[cid].content)) for cid in c.chunk_ids)
        if best < tau:
            c.label, c.stage = "unsupported", "embedding"
            c.reason = f"similarity to the evidence {best:.3f} < {tau}"


def verify(claims: list[Claim], evidence: list[RetrievedChunk], llm, known_binomials=(),
           embedder=None, tau: float = TAU, use_llm: bool = True, attempts: int = 2) -> VerificationResult:
    """Grade every claim; only ``supported`` ones may reach a card."""
    by_chunk = {c.chunk_id: c for c in evidence}
    known = frozenset(b.lower() for b in known_binomials if b and " " in b and "spp" not in b)
    for claim in claims:
        _deterministic(claim, by_chunk, known)
    if embedder is not None and USE_EMBEDDING_FILTER:
        _embedding_filter([c for c in claims if c.label is None], by_chunk, embedder, tau)
    pending = [c for c in claims if c.label is None]
    if not use_llm:
        for c in pending:
            c.label, c.reason, c.stage = "supported", "passed the deterministic and embedding checks", "embedding"
        return VerificationResult(claims=claims)
    result = VerificationResult(claims=claims)
    if not pending:
        return result
    if llm is None:
        for c in pending:
            c.label, c.reason, c.stage = "unsupported", "no LLM verifier", "llm"
        result.llm_error = "no llm"
        return result
    used = sorted({cid for c in pending for cid in c.chunk_ids})
    payload = {
        "evidence": {cid: f"[{by_chunk[cid].category}] {by_chunk[cid].content}" for cid in used},
        "claims": [{"id": c.id, "field": c.field, "text": c.text, "citations": c.chunk_ids} for c in pending],
    }
    verdicts: dict = {}
    for attempt in range(attempts):
        result.llm_calls += 1
        try:
            data = reply_json(llm.invoke(VERIFIER_PROMPT + "\n" + json.dumps(payload, ensure_ascii=False)))
            verdicts = {int(v["id"]): v for v in data.get("verdicts", []) if isinstance(v, dict) and "id" in v}
            result.llm_error = None
            break
        except Exception as exc:  # fail closed, but say so
            logger.warning("claim verifier attempt %d failed: %s: %s", attempt + 1, type(exc).__name__, exc)
            result.llm_error = type(exc).__name__
    for c in pending:
        v = verdicts.get(c.id)
        label = v.get("label") if v else None
        if label in ("supported", "inferred", "unsupported"):
            # "alasan" is the Indonesian-era key; still read so replayed replies parse.
            c.label, c.reason = label, str(v.get("reason") or v.get("alasan") or "")[:300]
        else:
            c.label, c.reason = "unsupported", "the verifier gave no label"
        c.stage = "llm"
    return result
