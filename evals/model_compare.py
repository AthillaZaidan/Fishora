"""Cost, latency and card validity of the published (sync) card path across
OpenCode Go models reachable with the configured key.

    python -m evals.model_compare --label baseline [--models gpt-6-luna,glm-5.3-flash]

Each model generates the same 11 species cards through the production
``KnowledgeGenerator`` (strict JSON schema + citation gate). "Valid" means the
card passed that gate; it says nothing about faithfulness, which needs the
judge (evals/research/judge-methodology.md).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

from evals.corpus import build_store, species_records
from evals.cost_eval import ALLOWANCE, PRICES, _dist, run_published
from evals.run import REPORTS_DIR, _jsonable, git_revision

DEFAULT_MODELS = ("gpt-5.6-luna", "gpt-6-luna", "deepseek-v4.1-flash", "glm-5.3-flash", "kimi-k2.6", "glm-5.2")


def main(argv: list[str] | None = None):
    from apps.main_api.config import MainSettings
    from apps.main_api.services.embeddings import LocalE5Embedder

    parser = argparse.ArgumentParser(prog="python -m evals.model_compare")
    parser.add_argument("--label", default="current")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--timeout", type=float, default=45.0, help="per-call timeout, seconds")
    parser.add_argument("--retries", type=int, default=0, help="client retries per call")
    args = parser.parse_args(argv)
    out = REPORTS_DIR / args.label / "model_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = MainSettings()
    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    store = build_store(embedder)
    species = species_records()

    per_model, all_cards = {}, []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        if model not in PRICES:
            print(f"[compare] skip {model}: no list price recorded")
            continue
        started = time.perf_counter()
        cards, calls = run_published(species, store, embedder, settings, repeat=1, model=model,
                                     timeout=args.timeout, max_retries=args.retries)
        ok = [c for c in cards if c["status"] == "completed"]
        errors: dict[str, int] = {}
        for c in cards:
            if c["error"]:
                errors[c["error"]] = errors.get(c["error"], 0) + 1
        mean_cost = statistics.fmean(c["cost_usd"] for c in cards) if cards else 0.0
        per_model[model] = {
            "cards": len(cards),
            "valid_card_rate": round(len(ok) / len(cards), 4),
            "errors": errors,
            "english_leak_rate": round(sum(bool(c.get("english_leak")) for c in ok) / len(ok), 4) if ok else None,
            "mean_fields_filled": round(statistics.fmean(sum(c["fields_filled"].values()) for c in ok), 2) if ok else None,
            "mean_sources": round(statistics.fmean(c["sources"] for c in ok), 2) if ok else None,
            "latency_s": _dist([c["llm_time_s"] for c in cards]),
            "input_tokens": _dist([c["input_tokens"] for c in cards]),
            "output_tokens": _dist([c["output_tokens"] for c in cards]),
            "reasoning_tokens": _dist([c["reasoning_tokens"] for c in cards]),
            "cost_usd_per_card": _dist([c["cost_usd"] for c in cards]),
            "cost_usd_per_valid_card": round(sum(c["cost_usd"] for c in cards) / len(ok), 6) if ok else None,
            "cost_usd_per_1000_cards": round(mean_cost * 1000, 3),
            "cards_per_5h_allowance": int(ALLOWANCE[model]["5h"] // mean_cost) if mean_cost else None,
            "prices_usd_per_1m": dict(zip(("input", "output", "cached_input"), PRICES[model])),
            "wall_s": round(time.perf_counter() - started, 1),
        }
        all_cards += cards
        write(out, args, per_model, all_cards)  # after every model, so a stalled model loses nothing
        print(f"[compare] {model}: valid {per_model[model]['valid_card_rate']:.0%} "
              f"${per_model[model]['cost_usd_per_card'].get('mean', 0):.5f}/card "
              f"p50 {per_model[model]['latency_s'].get('p50')}s errors {errors}", flush=True)

    write(out, args, per_model, all_cards)
    print(f"[compare] wrote {out}")
    return out


def write(out, args, per_model, all_cards) -> None:
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "path": "published (KnowledgeService sync path)",
        "call_timeout_s": args.timeout,
        "client_retries": args.retries,
        "models": per_model,
        "cards": all_cards,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")


if __name__ == "__main__":
    main()
