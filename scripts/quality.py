"""Run the RAG test suite and evaluations; write the dashboard's artifacts.

    python -m scripts.quality --label baseline   # once, on the code before a change
    python -m scripts.quality --label current    # after it; also writes comparison.json

Writes, under reports/<label>/:
    junit.xml, coverage.json   raw pytest outputs
    tests.json                 parsed test results (per layer, per test)
    rag_eval.json              evals.run output
and reports/comparison.json whenever both baseline and current exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# The modules the RAG path runs through; coverage is reported for these apart
# from the rest of main_api (commerce, auth), which this suite does not target.
RAG_MODULES = (
    "apps/main_api/services/chunking.py",
    "apps/main_api/services/retrieval.py",
    "apps/main_api/services/embeddings.py",
    "apps/main_api/services/generation.py",
    "apps/main_api/services/orchestrator.py",
    "apps/main_api/services/knowledge.py",
    "apps/main_api/services/verification.py",
    "apps/main_api/services/identification.py",
    "apps/main_api/services/card_cache.py",
    "apps/main_api/api/fish.py",
    "apps/main_api/api/jobs.py",
    "apps/main_api/api/quality.py",
    "apps/main_api/main.py",
)
LAYERS = ("unit", "retrieval", "generation", "integration", "e2e")


def run_pytest(out_dir: Path) -> dict:
    junit = out_dir / "junit.xml"
    coverage = out_dir / "coverage.json"
    command = [
        sys.executable, "-m", "pytest", "evals/tests", "-q", "-p", "no:cacheprovider",
        f"--junitxml={junit}", "--cov=apps.main_api", f"--cov-report=json:{coverage}",
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "HF_HUB_OFFLINE": "1"})
    duration = time.perf_counter() - started
    (out_dir / "pytest.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return {"exit_code": proc.returncode, "duration_s": round(duration, 1),
            **parse_junit(junit), "coverage": parse_coverage(coverage)}


def parse_junit(path: Path) -> dict:
    if not path.exists():
        return {"totals": {}, "layers": {}, "tests": []}
    tests = []
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        layer = next((part for part in classname.split(".") if part in LAYERS), "other")
        status, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (node.get("message") or node.text or "").strip().splitlines()[0][:240] if (node.get("message") or node.text) else ""
                break
        tests.append({
            "layer": layer,
            "module": classname.split(".")[-1],
            "name": case.get("name"),
            "status": status,
            "time_s": round(float(case.get("time") or 0), 3),
            "message": message,
        })
    def count(rows):
        return {s: sum(r["status"] == s for r in rows) for s in ("passed", "failed", "error", "skipped")} | {"total": len(rows)}
    return {
        "totals": count(tests),
        "layers": {layer: count([t for t in tests if t["layer"] == layer]) for layer in LAYERS},
        "tests": tests,
    }


def parse_coverage(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = {name.replace("\\", "/"): info["summary"] for name, info in data["files"].items()}
    rag = {name: files[name] for name in RAG_MODULES if name in files}
    covered = sum(s["covered_lines"] for s in rag.values())
    statements = sum(s["num_statements"] for s in rag.values())
    return {
        "all_main_api_percent": round(data["totals"]["percent_covered"], 1),
        "rag_percent": round(100 * covered / statements, 1) if statements else 0.0,
        "rag_modules": {
            name.split("/")[-1]: {"percent": round(s["percent_covered"], 1),
                                  "missing_lines": s["missing_lines"], "statements": s["num_statements"]}
            for name, s in rag.items()
        },
    }


# (section, path, label, higher_is_better)
HEADLINE = [
    ("retrieval", "scoped.recall@6", "Retrieval recall@6 (species-scoped)", True),
    ("retrieval", "scoped.mrr", "Retrieval MRR (species-scoped)", True),
    ("retrieval", "global.species_hit@1", "Species hit@1 without filter", True),
    ("retrieval", "card_query_coverage.mean", "Card query category coverage", True),
    ("retrieval", "latency_ms.p95", "Retrieval p95 latency (ms)", False),
    ("grounding", "test.f1", "Grounding F1 (test split)", True),
    ("grounding", "test.recall", "Grounding recall (test split)", True),
    ("grounding", "test.false_support_rate", "Grounding false-support rate", False),
    ("pipeline", "plain.job_success_rate", "Card job success rate", True),
    ("pipeline", "fenced.job_success_rate", "Job success, fenced-JSON LLM", True),
    ("pipeline", "plain.claim_retention", "True-claim retention", True),
    ("pipeline", "plain.hallucination_leakage", "Hallucination leakage", False),
    ("pipeline", "plain.citation_validity", "Citation validity", True),
    ("pipeline", "plain.llm_calls_per_card", "LLM calls per card", False),
    ("pipeline", "plain.sequential_rounds", "Sequential LLM rounds per card", False),
    ("pipeline", "plain.repeat_latency_ms_mean", "Repeat card latency (ms)", False),
    ("environment", "embedder_cold_load_s", "Embedder cold load (s)", False),
    ("corpus", "chunker_fits_e5", "Chunker limit fits E5 window", True),
]


def _get(data: dict, section: str, path: str):
    node = data.get(section, {})
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def compare(baseline: dict, current: dict) -> dict:
    rows = []
    for section, path, label, higher in HEADLINE:
        before, after = _get(baseline["eval"], section, path), _get(current["eval"], section, path)
        if isinstance(before, bool) or isinstance(after, bool):
            delta, improved = None, (after is True and before is not True)
        elif isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = round(after - before, 4)
            improved = delta > 0 if higher else delta < 0
        else:
            delta, improved = None, None
        rows.append({"metric": label, "section": section, "path": path, "before": before,
                     "after": after, "delta": delta, "higher_is_better": higher,
                     "improved": improved if delta not in (0, 0.0) else None})
    tb, tc = baseline["tests"]["totals"], current["tests"]["totals"]
    before_status = {(t["module"], t["name"]): t["status"] for t in baseline["tests"]["tests"]}
    fixed = [f"{t['module']}::{t['name']}" for t in current["tests"]["tests"]
             if t["status"] == "passed" and before_status.get((t["module"], t["name"])) in ("failed", "error")]
    regressed = [f"{t['module']}::{t['name']}" for t in current["tests"]["tests"]
                 if t["status"] in ("failed", "error") and before_status.get((t["module"], t["name"])) == "passed"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_commit": baseline["eval"].get("git", {}).get("commit"),
        "current_commit": current["eval"].get("git", {}).get("commit"),
        "metrics": rows,
        "tests": {
            "baseline": tb, "current": tc,
            "baseline_pass_rate": round(tb.get("passed", 0) / max(1, tb.get("total", 0)), 4),
            "current_pass_rate": round(tc.get("passed", 0) / max(1, tc.get("total", 0)), 4),
            "fixed": fixed, "regressed": regressed,
        },
        "coverage": {
            "baseline_rag_percent": baseline["tests"].get("coverage", {}).get("rag_percent"),
            "current_rag_percent": current["tests"].get("coverage", {}).get("rag_percent"),
        },
    }


def load_run(label: str) -> dict | None:
    folder = REPORTS / label
    if not (folder / "tests.json").exists() or not (folder / "rag_eval.json").exists():
        return None
    return {"tests": json.loads((folder / "tests.json").read_text(encoding="utf-8")),
            "eval": json.loads((folder / "rag_eval.json").read_text(encoding="utf-8"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.quality")
    parser.add_argument("--label", default="current")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-evals", action="store_true")
    args = parser.parse_args(argv)
    out_dir = REPORTS / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        print(f"[quality] running test suite -> {out_dir}", flush=True)
        tests = run_pytest(out_dir)
        tests["label"] = args.label
        tests["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out_dir / "tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
        t = tests["totals"]
        print(f"[quality] tests: {t.get('passed')} passed, {t.get('failed')} failed, "
              f"{t.get('error')} errors, {t.get('skipped')} skipped; "
              f"RAG coverage {tests['coverage'].get('rag_percent')}%", flush=True)
    if not args.skip_evals:
        from evals.run import main as run_evals

        run_evals(["--label", args.label])

    baseline, current = load_run("baseline"), load_run("current")
    if baseline and current:
        (REPORTS / "comparison.json").write_text(json.dumps(compare(baseline, current), indent=2), encoding="utf-8")
        print(f"[quality] wrote {REPORTS / 'comparison.json'}")

    # Free, keyless artifacts, then the page. The paid real-LLM runs
    # (evals.cost_eval, evals.model_compare) stay separate commands; the
    # dashboard shows them when their JSON exists for the label.
    from evals import cv_eval, dashboard, findings, probes

    probes.main(["--label", args.label])
    cv_eval.main(["--label", args.label])  # species-ID results (working tree, else origin/feat/baseline-cv)
    findings.main(["--label", args.label])
    if args.label == "baseline" or (REPORTS / "baseline" / "tests.json").exists():
        dashboard.main(["--baseline", "baseline", "--current", "current"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
