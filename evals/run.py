"""Run every RAG evaluation and write ``reports/<label>/rag_eval.json``.

    python -m evals.run --label current
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.corpus import ROOT, build_store

REPORTS_DIR = ROOT / "reports"


def git_revision() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def run_all(label: str, delay: float = 0.2) -> dict:
    from apps.main_api.services.embeddings import LocalE5Embedder
    from evals import grounding_eval, pipeline_eval, retrieval_eval

    started = time.perf_counter()
    embedder = LocalE5Embedder()
    embedder.embed_query("warmup")
    cold_load_s = time.perf_counter() - started
    store = build_store(embedder)

    sections = {}
    timings = {}
    for name, fn in (
        ("corpus", lambda: retrieval_eval.corpus_health(embedder)),
        ("retrieval", lambda: retrieval_eval.evaluate(store, embedder)),
        ("grounding", lambda: grounding_eval.evaluate(store, embedder)),
        ("pipeline", lambda: pipeline_eval.evaluate(store, embedder, delay=delay)),
    ):
        t = time.perf_counter()
        sections[name] = fn()
        timings[name] = round(time.perf_counter() - t, 2)
        print(f"[evals] {name} done in {timings[name]}s", flush=True)

    return {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "embedding_model": embedder.model_name,
            "embedder_cold_load_s": round(cold_load_s, 2),
            "llm": "scripted (no OPENCODE_GO_API_KEY): measures guardrails, not model prose",
            "store": "in-memory exact cosine, same filters as SqlKnowledgeRepository.search_verified",
        },
        "timings_s": timings,
        **sections,
    }


def _jsonable(value):
    """numpy scalars from the metric code serialize as plain numbers."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("--label", default="current")
    parser.add_argument("--delay", type=float, default=0.2, help="simulated seconds per LLM call")
    args = parser.parse_args(argv)
    result = run_all(args.label, args.delay)
    out = REPORTS_DIR / args.label / "rag_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    print(f"[evals] wrote {out}")
    return out


if __name__ == "__main__":
    main()
