"""Deterministic probes of the shipped code, written to reports/<label>/probes.json.

    python -m evals.probes --label baseline

These are the checks behind findings that no test or eval run measures:
static facts about the source (silent excepts, the missing OpenCode session
header, which knowledge path lot publication uses), the taxonomy seed vs the
corpus identity chunks, and one runtime probe of the sync knowledge path with
a blank key. No network, no LLM key, no database.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from evals.corpus import ROOT, load_corpus
from evals.run import REPORTS_DIR, _jsonable, git_revision

RAG_SOURCES = (
    "apps/main_api/api/fish.py",
    "apps/main_api/services/orchestrator.py",
    "apps/main_api/services/generation.py",
    "apps/main_api/services/knowledge.py",
    "apps/main_api/services/lots.py",
)
TAXONOMY_CSV = ROOT / "artifacts/Dataset/fishora_dataset/metadata/taxonomy.csv"
_BINOMIAL = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,})\b")
# Capitalised English phrases the binomial pattern also matches in chunk prose.
_NOT_EPITHETS = {"and", "corresponds", "is", "the", "of", "label", "reaching", "bunga"}


def silent_excepts() -> dict:
    """Handlers that catch everything and neither re-raise nor log."""
    found = []
    for rel in RAG_SOURCES:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"})
            if not broad:
                continue
            body_src = ast.unparse(node)
            logs = any(tok in body_src for tok in ("logger.", "logging.", "log.", "raise"))
            if not logs:
                found.append({"file": rel, "line": node.lineno, "body": ast.unparse(node.body[0])[:60] if node.body else ""})
    return {"count": len(found), "handlers": found}


def opencode_headers() -> dict:
    src = (ROOT / "apps/main_api/services/generation.py").read_text(encoding="utf-8")
    return {
        "sends_session_header": "x-opencode-session" in src,
        "sets_user_agent": "User-Agent" in src or "default_headers" in src,
    }


def publication_path() -> dict:
    lots = (ROOT / "apps/main_api/services/lots.py").read_text(encoding="utf-8")
    knowledge = (ROOT / "apps/main_api/services/knowledge.py").read_text(encoding="utf-8")
    matching = (ROOT / "apps/main_api/services/matching.py").read_text(encoding="utf-8")
    return {
        "publish_uses_sync_knowledge_service": "get_for_prediction" in lots,
        "sync_path_has_claim_critic": any(tok in knowledge for tok in ("critic", "_grade_claim", "ClaimStatus")),
        "matching_reads_snapshot_fields": sorted(
            f for f in ("processing_methods", "commercial_uses", "taste", "texture", "physical_characteristics")
            if f'"{f}"' in matching
        ),
    }


def taxonomy_vs_corpus() -> dict:
    if not TAXONOMY_CSV.exists():
        return {"available": False}
    seeded = {}
    with TAXONOMY_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seeded[row["normalized_label"]] = (row["scientific_name"] or "").strip()
    corpus_names: dict[str, set[str]] = {}
    for chunk in load_corpus():
        if chunk.category == "identity":
            corpus_names.setdefault(chunk.species_label, set()).update(
                name for name in _BINOMIAL.findall(chunk.content) if name.split()[1] not in _NOT_EPITHETS)
    rows = []
    for label, seeded_name in sorted(seeded.items()):
        names = corpus_names.get(label, set())
        genus_level = seeded_name.endswith(" spp.")
        agrees = (not seeded_name) or seeded_name in names or (genus_level and any(n.startswith(seeded_name.split()[0]) for n in names))
        rows.append({"species": label, "seeded": seeded_name or None, "corpus": sorted(names), "agrees": bool(agrees)})
    return {
        "available": True,
        "source": str(TAXONOMY_CSV.relative_to(ROOT)).replace("\\", "/"),
        "mismatches": [r for r in rows if not r["agrees"]],
        "rows": rows,
    }


def sync_path_blank_key() -> dict:
    """GET knowledge for a verified prediction when no job exists and the
    OpenCode key is blank (the fallback a failed job schedule drops into)."""
    from fastapi.testclient import TestClient

    from apps.main_api.config import MainSettings
    from apps.main_api.main import create_main_app
    from apps.main_api.ports import AppDependencies
    from apps.main_api.services.generation import KnowledgeGenerator, OpenCodeGoClient
    from evals.corpus import species_records
    from evals.fakes import FixedCVClient, InMemoryImageStore, InMemoryPredictionRepository, InMemorySpeciesRepository

    class OneChunkRetriever:
        def retrieve(self, species_id, query, max_chunks=6):
            from apps.main_api.contracts import RetrievedChunk

            return [RetrievedChunk(
                chunk_id="chunk_nila_taste_001", species_id=species_id, source_id="fao_en_niletilapia",
                source_type="species_fact_sheet", category="taste_texture", content="mild flavour",
                distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
                source_title="FAO", source_publisher="FAO", source_url="https://fao.org", source_reviewed_at=None,
            )]

    settings = MainSettings(_env_file=None, database_url="postgresql+psycopg://probe@localhost/probe",
                            opencode_go_api_key="")
    predictions = InMemoryPredictionRepository()
    deps = AppDependencies(
        cv_client=FixedCVClient("nila"), species_repo=InMemorySpeciesRepository(species_records()),
        prediction_repo=predictions, image_store=InMemoryImageStore(),
        embedder=type("E5Stub", (), {"model_name": "intfloat/multilingual-e5-base"})(),
        retriever=OneChunkRetriever(), generator=KnowledgeGenerator(lambda: OpenCodeGoClient(settings)),
        job_repo=None,
    )
    predictions.create("probe1", "memory://probe1", "species_nila", 0.9, [], "probe")
    predictions.verify("probe1", "species_nila", "confirmed")
    app = create_main_app(settings=settings, deps=deps)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/predictions/probe1/knowledge")
    return {"status_code": response.status_code, "body": response.text[:120]}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.probes")
    parser.add_argument("--label", default="current")
    args = parser.parse_args(argv)
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "silent_excepts": silent_excepts(),
        "opencode_headers": opencode_headers(),
        "publication_path": publication_path(),
        "taxonomy_vs_corpus": taxonomy_vs_corpus(),
        "sync_path_blank_key": sync_path_blank_key(),
    }
    out = REPORTS_DIR / args.label / "probes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    print(f"[probes] wrote {out}")
    return out


if __name__ == "__main__":
    main()
