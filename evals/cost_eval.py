"""Real-LLM cost and latency of both card generation paths.

    python -m evals.cost_eval --label baseline [--species nila,tuna] [--repeat 1]

Needs OPENCODE_GO_API_KEY (from .env). Application code is not modified:

* agent path (``run_graph``, the operator card): the real LLM is wrapped in
  ``RecordingLLM``. It reads ``usage_metadata`` off every response and labels
  the call by pipeline stage (sub-query, expert, critic, polish).
* published path (``KnowledgeGenerator`` + ``OpenCodeGoClient``, the lot/QR
  card): LangChain's usage callback captures its one structured-output call.

Retrieval runs on the real E5 model over the in-memory candidate store, as in
evals/run.py. Cost uses the OpenCode Go list price. On the subscription that
price is drawn from a per-model allowance ($15/month, $3 per 5 h), so the
report also gives cards per allowance window.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from evals.corpus import build_store, species_records
from evals.fakes import InMemoryJobRepository, InMemorySpeciesRepository, new_id
from evals.run import REPORTS_DIR, _jsonable, git_revision

# OpenCode Go list prices, USD per 1M tokens (opencode.ai/docs/go, read 2026-09-26),
# <=272K-token context tier; (input, output, cached input).
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20, 0.02),
    "gpt-6-luna": (0.10, 0.50, 0.01),
    "glm-5.3-flash": (0.15, 0.50, 0.03),
    "glm-5.2": (1.40, 4.40, 0.26),
    "kimi-k2.6": (0.95, 4.00, 0.16),
    "deepseek-v4.1-flash": (0.30, 1.20, 0.006),  # peak tier (off-peak is half)
}
# Endpoint family per model on OpenCode Go (docs table): responses vs chat/completions.
RESPONSES_API = {"gpt-5.6-luna", "gpt-6-luna"}
# Per-model allowance on the $10/month Go subscription: 5-hour, weekly, monthly (USD).
ALLOWANCE = {
    "gpt-5.6-luna": {"5h": 3.0, "week": 7.5, "month": 15.0},
    "gpt-6-luna": {"5h": 3.0, "week": 7.5, "month": 15.0},
    "glm-5.3-flash": {"5h": 12.0, "week": 30.0, "month": 60.0},
    "glm-5.2": {"5h": 12.0, "week": 30.0, "month": 60.0},
    "kimi-k2.6": {"5h": 12.0, "week": 30.0, "month": 60.0},
    "deepseek-v4.1-flash": {"5h": 12.0, "week": 30.0, "month": 60.0},
}

CARD_QUERY = (
    "Buat kartu pengetahuan bahasa Indonesia untuk {common_name}: identitas, "
    "ciri fisik, rasa dan tekstur, cara pengolahan, penggunaan komersial, dan "
    "spesies pengganti."
)
_STAGE_PREFIXES = (
    ("Untuk spesies", "researcher_subquery"),
    ("Tulis physical_characteristics", "expert_physical"),
    ("Tulis taste dan texture", "expert_taste"),
    ("Tulis processing_methods", "expert_commercial"),
    ("Tulis similar_or_substitute_species", "expert_substitute"),
    ("Untuk setiap field", "critic_llm"),
    ("Perbaiki bahasa", "writer_polish"),
)


EVAL_USER_AGENT = "fishora-rag-eval/0.1"


def make_llm(settings, session_id: str, model: str | None = None,
             timeout: float | None = None, max_retries: int | None = None, reasoning_effort: str | None = None):
    """Eval client for the model comparison (any model, bounded timeouts).
    Baseline runs also used it for the production model, because the shipped
    factory sent no session header then (W20); it now does, so the cost run
    uses the production factory for the configured model."""
    from langchain_openai import ChatOpenAI

    model = model or settings.opencode_go_model
    return ChatOpenAI(
        model=model,
        base_url=settings.opencode_go_base_url,
        api_key=settings.opencode_go_api_key.get_secret_value(),
        timeout=timeout or settings.opencode_go_timeout_seconds,
        **({"max_retries": max_retries} if max_retries is not None else {}),
        use_responses_api=model in RESPONSES_API,
        **({"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}),
        default_headers={"x-opencode-session": session_id, "User-Agent": EVAL_USER_AGENT},
    )


@dataclass
class CallRecord:
    path: str
    species: str
    stage: str
    started_s: float
    latency_s: float
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    cost_usd: float = 0.0


def _usage_fields(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int((usage.get("input_token_details") or {}).get("cache_read") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int((usage.get("output_token_details") or {}).get("reasoning") or 0),
    }


def price(model: str, input_tokens: int, cached: int, output: int) -> float:
    p_in, p_out, p_cache = PRICES[model]
    return ((input_tokens - cached) * p_in + cached * p_cache + output * p_out) / 1_000_000


@dataclass
class RecordingLLM:
    """Pass-through LLM proxy recording usage and latency per call."""

    llm: object
    model: str
    species: str = ""
    # True: hand the orchestrator ``response.text`` instead of the Responses-API
    # content-block list it cannot parse (W21). Eval-only, to price a *working*
    # agent card; False measures the pipeline exactly as shipped.
    normalize_content: bool = False
    path: str = "agent"
    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _t0: float = field(default_factory=time.perf_counter)

    def invoke(self, prompt, *args, **kwargs):
        text = prompt if isinstance(prompt, str) else str(prompt)
        stage = next((name for prefix, name in _STAGE_PREFIXES if text.startswith(prefix)), "other")
        started = time.perf_counter()
        error, usage = None, None
        try:
            response = self.llm.invoke(prompt, *args, **kwargs)
            usage = getattr(response, "usage_metadata", None)
            if self.normalize_content and not isinstance(response.content, str):
                response = response.model_copy(update={"content": response.text})
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            fields = _usage_fields(usage)
            record = CallRecord(
                path=self.path, species=self.species, stage=stage,
                started_s=round(started - self._t0, 3),
                latency_s=round(time.perf_counter() - started, 3), error=error, **fields,
            )
            record.cost_usd = price(self.model, record.input_tokens, record.cached_tokens, record.output_tokens)
            with self._lock:
                self.records.append(record)


_EN = {"the", "and", "with", "of", "is", "are", "for", "from", "its", "fish"}
_ID = {"dan", "yang", "dengan", "untuk", "dari", "adalah", "ikan", "atau", "pada", "sebagai"}


def english_leak(card: dict | None) -> bool | None:
    """True when a card's prose has more English than Indonesian function
    words. The prompt requires Indonesian; the evidence is English."""
    if not card:
        return None
    text = " ".join(
        v if isinstance(v, str) else " ".join(map(str, v))
        for k, v in card.items()
        if k in {"physical_characteristics", "taste", "texture", "processing_methods", "commercial_uses",
                 "similar_or_substitute_species", "potential_buyer_segments"} and v
    ).lower().split()
    if not text:
        return None
    return sum(w in _EN for w in text) > sum(w in _ID for w in text)


def _card_fill(card: dict | None) -> dict:
    if not card:
        return {}
    fields = ("physical_characteristics", "taste", "texture", "processing_methods",
              "commercial_uses", "similar_or_substitute_species", "potential_buyer_segments")
    return {f: bool(card.get(f)) for f in fields}


def run_agent(species, store, embedder, settings, repeat: int, normalize: bool = False) -> tuple[list[dict], list[CallRecord]]:
    from apps.main_api.services import orchestrator

    model = settings.opencode_go_model
    path = "agent_normalized" if normalize else "agent"
    llm = RecordingLLM(None, model, normalize_content=normalize, path=path)
    species_repo = InMemorySpeciesRepository(species_records())
    jobs = InMemoryJobRepository()
    cards = []
    for _ in range(repeat):
        for record in species:
            llm.species = record.normalized_label
            before = len(llm.records)
            job_id = new_id()
            # The production client (it sends the session header itself since W20).
            from apps.main_api.services.generation import make_opencode_go_llm
            llm.llm = make_opencode_go_llm(settings, session_id=f"fishora-card-{job_id}")
            jobs.create(job_id, job_id, record.id)
            started = time.perf_counter()
            orchestrator.run_graph(job_id, record.id, job_id, store, embedder, llm, llm, species_repo, jobs)
            wall = time.perf_counter() - started
            job = jobs.get(job_id)
            calls = llm.records[before:]
            cards.append({
                "path": path, "species": record.normalized_label, "status": job.status,
                "wall_s": round(wall, 2), "llm_calls": len(calls),
                "llm_errors": sum(c.error is not None for c in calls),
                "llm_time_s": round(sum(c.latency_s for c in calls), 2),
                "input_tokens": sum(c.input_tokens for c in calls),
                "cached_tokens": sum(c.cached_tokens for c in calls),
                "output_tokens": sum(c.output_tokens for c in calls),
                "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
                "cost_usd": round(sum(c.cost_usd for c in calls), 6),
                "critic_feedback": job.critic_feedback,
                "fields_filled": _card_fill(job.final_card),
                "english_leak": english_leak(job.final_card),
                "card": job.final_card,
                "sources": len((job.final_card or {}).get("sources", [])),
                "error": job.error,
            })
            print(f"[cost] {path} {record.normalized_label}: {job.status} {wall:.1f}s "
                  f"{cards[-1]['llm_calls']} calls ${cards[-1]['cost_usd']:.5f}", flush=True)
    return cards, llm.records


def run_published(species, store, embedder, settings, repeat: int, model: str | None = None,
                  timeout: float | None = None, max_retries: int | None = None) -> tuple[list[dict], list[CallRecord]]:
    from langchain_core.callbacks import get_usage_metadata_callback

    from apps.main_api.services import generation
    from apps.main_api.services.generation import KnowledgeGenerator, OpenCodeGoClient
    from apps.main_api.services.retrieval import VerifiedRetriever

    model = model or settings.opencode_go_model
    retriever = VerifiedRetriever(store, embedder)
    generator = KnowledgeGenerator(lambda: OpenCodeGoClient(settings))
    cards, records = [], []
    t0 = time.perf_counter()
    production_factory = generation.make_opencode_go_llm
    for _ in range(repeat):
        for record in species:
            session = f"fishora-card-{new_id()}"
            # OpenCodeGoClient looks the factory up at call time; swap in the
            # header-carrying one for this card only (eval process, not app code).
            if model != settings.opencode_go_model or timeout or max_retries is not None:
                # Model comparison: another model or bounded calls, via the eval client.
                generation.make_opencode_go_llm = (
                    lambda s, timeout_=None, session_id=None, _sid=session, _m=model: make_llm(s, _sid, _m, timeout, max_retries))
            started = time.perf_counter()
            evidence = retriever.retrieve(record.id, CARD_QUERY.format(common_name=record.common_name_id))
            retrieval_s = time.perf_counter() - started
            status, error, card = "completed", None, None
            gen_started = time.perf_counter()
            with get_usage_metadata_callback() as cb:
                try:
                    card = generator.generate(record, evidence).model_dump(mode="json")
                except Exception as exc:
                    cause = exc.__cause__ or exc
                    # Class plus the provider's short reason (API errors carry no credentials).
                    status, error = "failed", f"{type(exc).__name__}: {str(cause)[:160]}"
            gen_s = time.perf_counter() - gen_started
            usage = {}
            for model_usage in cb.usage_metadata.values():
                for key, value in _usage_fields(model_usage).items():
                    usage[key] = usage.get(key, 0) + value
            usage = {**_usage_fields(None), **usage}
            call = CallRecord(path="published", species=record.normalized_label, stage="sync_generate",
                              started_s=round(gen_started - t0, 3), latency_s=round(gen_s, 3),
                              error=error, **usage)
            call.cost_usd = price(model, call.input_tokens, call.cached_tokens, call.output_tokens)
            records.append(call)
            cards.append({
                "path": "published", "species": record.normalized_label, "status": status,
                "wall_s": round(time.perf_counter() - started, 2), "retrieval_ms": round(retrieval_s * 1000, 1),
                "llm_calls": 1, "llm_errors": int(error is not None), "llm_time_s": round(gen_s, 2),
                **usage, "cost_usd": round(call.cost_usd, 6),
                "fields_filled": _card_fill(card), "english_leak": english_leak(card),
                "model": model, "card": card, "sources": len((card or {}).get("sources", [])),
                "error": error,
            })
            print(f"[cost] published {record.normalized_label}: {status} {gen_s:.1f}s ${call.cost_usd:.5f}", flush=True)
    generation.make_opencode_go_llm = production_factory
    return cards, records


def _dist(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]
    return {"mean": round(statistics.fmean(values), 4), "p50": round(pick(0.5), 4),
            "p95": round(pick(0.95), 4), "max": round(max(values), 4), "n": len(values)}


def summarize(cards: list[dict], records: list[CallRecord], model: str) -> dict:
    by_path = {}
    for path in ("published", "agent", "agent_normalized"):
        every = [c for c in cards if c["path"] == path]
        # A completed agent card with no LLM call came from the card cache (W6).
        # Generation statistics exclude those so they compare with uncached runs;
        # cache hits are reported on their own.
        hits = [c for c in every if path != "published" and c["status"] == "completed" and c["llm_calls"] == 0]
        pc = [c for c in every if c not in hits]
        if not pc:
            continue
        mean_cost = statistics.fmean(c["cost_usd"] for c in pc)
        out = sum(c["output_tokens"] for c in pc)
        by_path[path] = {
            "cards": len(pc),
            "success_rate": round(sum(c["status"] == "completed" for c in pc) / len(pc), 4),
            "cost_usd_per_card": _dist([c["cost_usd"] for c in pc]),
            "wall_s_per_card": _dist([c["wall_s"] for c in pc]),
            "llm_calls_per_card": _dist([c["llm_calls"] for c in pc]),
            "input_tokens_per_card": _dist([c["input_tokens"] for c in pc]),
            "output_tokens_per_card": _dist([c["output_tokens"] for c in pc]),
            "reasoning_share_of_output": round(sum(c["reasoning_tokens"] for c in pc) / out, 4) if out else None,
            "cache_hit_share_of_input": round(sum(c["cached_tokens"] for c in pc) / max(1, sum(c["input_tokens"] for c in pc)), 4),
            "output_share_of_cost": round(
                sum(c["output_tokens"] for c in pc) * PRICES[model][1] / 1e6 / max(1e-12, sum(c["cost_usd"] for c in pc)), 4),
            "english_leak_rate": round(sum(bool(c.get("english_leak")) for c in pc) / len(pc), 4),
            "cards_per_allowance": {window: int(limit // mean_cost) if mean_cost else None
                                    for window, limit in ALLOWANCE[model].items()},
            "cost_usd_per_1000_cards": round(mean_cost * 1000, 2),
            "cache_hits": len(hits),
            "cache_hit_wall_s": _dist([c["wall_s"] for c in hits]),
            "fields_filled_rate": {
                f: round(sum(c["fields_filled"].get(f, False) for c in pc) / len(pc), 3)
                for f in (pc[0]["fields_filled"] or {"physical_characteristics": 0})
            } if any(c["fields_filled"] for c in pc) else {},
        }
    stages = {}
    for record in records:
        stages.setdefault((record.path, record.stage), []).append(record)
    by_stage = {
        f"{path}:{stage}": {
            "calls": len(rs),
            "errors": sum(r.error is not None for r in rs),
            "latency_s": _dist([r.latency_s for r in rs]),
            "input_tokens": _dist([r.input_tokens for r in rs]),
            "output_tokens": _dist([r.output_tokens for r in rs]),
            "reasoning_tokens": _dist([r.reasoning_tokens for r in rs]),
            "cost_usd_total": round(sum(r.cost_usd for r in rs), 6),
        }
        for (path, stage), rs in sorted(stages.items())
    }
    total_cost = sum(r.cost_usd for r in records)
    return {
        "by_path": by_path,
        "by_stage": by_stage,
        "stage_cost_share": {k: round(v["cost_usd_total"] / total_cost, 4) if total_cost else 0.0
                             for k, v in by_stage.items()},
        "run_total_cost_usd": round(total_cost, 5),
    }


def main(argv: list[str] | None = None):
    from apps.main_api.config import MainSettings
    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.cost_eval")
    parser.add_argument("--label", default="current")
    parser.add_argument("--species", default="", help="comma-separated labels; default all 11")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--paths", default="published,agent")
    args = parser.parse_args(argv)

    settings = MainSettings()
    if not settings.opencode_go_api_key.get_secret_value().strip():
        raise SystemExit("OPENCODE_GO_API_KEY is not set (put it in .env)")
    model = settings.opencode_go_model
    if model not in PRICES:
        raise SystemExit(f"no list price recorded for {model!r}; add it to PRICES")
    wanted = {s.strip() for s in args.species.split(",") if s.strip()}
    species = [s for s in species_records() if not wanted or s.normalized_label in wanted]

    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    store = build_store(embedder)
    started = time.perf_counter()
    cards, records = [], []
    if "published" in args.paths:
        c, r = run_published(species, store, embedder, settings, args.repeat)
        cards += c
        records += r
    for normalize, name in ((False, "agent"), (True, "agent_normalized")):
        if name in args.paths.split(","):
            c, r = run_agent(species, store, embedder, settings, args.repeat, normalize)
            cards += c
            records += r

    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "model": model,
        "prices_usd_per_1m": dict(zip(("input", "output", "cached_input"), PRICES[model])),
        "allowance_usd": ALLOWANCE[model],
        "wall_time_s": round(time.perf_counter() - started, 1),
        "summary": summarize(cards, records, model),
        "cards": cards,
        "calls": [asdict(r) for r in records],
    }
    out = REPORTS_DIR / args.label / "cost_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    print(f"[cost] wrote {out}; run cost ${result['summary']['run_total_cost_usd']}")
    return out


if __name__ == "__main__":
    main()
