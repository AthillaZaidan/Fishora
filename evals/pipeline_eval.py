"""End-to-end card pipeline: the production ``run_graph`` with a scripted LLM.

Measures what the guardrails and orchestration do, independent of LLM prose:
job success, true-claim retention, hallucination leakage, citation validity,
LLM calls per card, critical-path latency, and repeat (cache) latency. The
scripted LLM sleeps ``delay`` seconds per call, so ``sequential_rounds`` is
wall time divided by that delay.
"""

from __future__ import annotations

import statistics
import time

from apps.main_api.services import orchestrator
from evals.corpus import load_corpus, load_dataset, species_records
from evals.fakes import InMemoryJobRepository, InMemorySpeciesRepository, ScriptedLLM, new_id


def _claim_in_card(card: dict, field: str, text: str) -> bool:
    value = card.get(field)
    if isinstance(value, list):
        return text in value
    return value == text


def _scripted_llm(delay: float, fenced: bool) -> ScriptedLLM:
    corpus = load_corpus()
    return ScriptedLLM(
        claims=load_dataset("grounding_claims.json")["claims"],
        chunk_species={c.id: c.species_label for c in corpus},
        chunk_category={c.id: c.category for c in corpus},
        delay=delay,
        fenced=fenced,
    )


def run_scenario(store, embedder, *, delay: float, fenced: bool, repeat: bool = True) -> dict:
    species = species_records()
    species_repo = InMemorySpeciesRepository(species)
    job_repo = InMemoryJobRepository()
    own_sources: dict[str, set[str]] = {}
    for chunk in load_corpus():
        own_sources.setdefault(chunk.species_label, set()).add(chunk.source["id"])
    clear = getattr(orchestrator, "clear_card_cache", None)
    if clear:
        clear()

    per_card, repeat_ms = [], []
    llm = _scripted_llm(delay, fenced)
    for record in species:
        calls_before, emitted_before = llm.calls, len(llm.emitted)
        job_id = new_id()
        job_repo.create(job_id, job_id, record.id)
        started = time.perf_counter()
        orchestrator.run_graph(job_id, record.id, job_id, store, embedder, llm, llm, species_repo, job_repo)
        elapsed = time.perf_counter() - started
        job = job_repo.get(job_id)
        card = job.final_card or {}
        emitted = llm.emitted[emitted_before:]
        cited = [s["source_id"] for s in card.get("sources", [])]
        per_card.append({
            "species_label": record.normalized_label,
            "status": job.status,
            "latency_ms": round(elapsed * 1000, 1),
            "llm_calls": llm.calls - calls_before,
            "true_emitted": sum(c.kind == "true" for c in emitted),
            "true_kept": sum(c.kind == "true" and _claim_in_card(card, c.field, c.text) for c in emitted),
            "halluc_emitted": sum(c.kind == "hallucination" for c in emitted),
            "halluc_kept": sum(c.kind == "hallucination" and _claim_in_card(card, c.field, c.text) for c in emitted),
            "citations": len(cited),
            "citations_valid": sum(s in own_sources[record.normalized_label] for s in cited),
        })
        if repeat:
            job_id = new_id()
            job_repo.create(job_id, job_id, record.id)
            started = time.perf_counter()
            orchestrator.run_graph(job_id, record.id, job_id, store, embedder, llm, llm, species_repo, job_repo)
            repeat_ms.append((time.perf_counter() - started) * 1000)

    def total(key):
        return sum(row[key] for row in per_card)

    latencies = [row["latency_ms"] for row in per_card]
    return {
        "delay_s": delay,
        "fenced_json": fenced,
        "cards": len(per_card),
        "job_success_rate": round(sum(r["status"] == "completed" for r in per_card) / len(per_card), 4),
        "claim_retention": round(total("true_kept") / max(1, total("true_emitted")), 4),
        "hallucination_leakage": round(total("halluc_kept") / max(1, total("halluc_emitted")), 4),
        "citation_validity": round(total("citations_valid") / max(1, total("citations")), 4) if total("citations") else None,
        "llm_calls_per_card": round(total("llm_calls") / len(per_card), 2),
        "latency_ms_mean": round(statistics.fmean(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
        "sequential_rounds": round(statistics.fmean(latencies) / 1000 / delay, 2) if delay else None,
        "repeat_latency_ms_mean": round(statistics.fmean(repeat_ms), 1) if repeat_ms else None,
        "per_card": per_card,
    }


def evaluate(store, embedder, delay: float = 0.2) -> dict:
    return {
        "plain": run_scenario(store, embedder, delay=delay, fenced=False),
        "fenced": run_scenario(store, embedder, delay=0.0, fenced=True, repeat=False),
    }
