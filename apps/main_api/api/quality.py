"""Evaluation dashboard and summary, served from generated artifacts.

GET /quality                 the dashboard HTML (reports/dashboard.html, else the committed baseline copy)
GET /api/v1/quality/summary  which runs exist and the baseline-vs-current comparison

Read-only: nothing here runs an evaluation. The pages contain evaluation
results and weakness lists, no secrets; put them behind operator auth before a
public deployment.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
COMMITTED = ROOT / "evals" / "results"

router = APIRouter()


def _dashboard_path() -> Path | None:
    for candidate in (REPORTS / "dashboard.html", COMMITTED / "iteration-1" / "dashboard.html",
                      COMMITTED / "baseline" / "dashboard.html"):
        if candidate.exists():
            return candidate
    return None


@router.get("/quality", response_class=HTMLResponse)
def quality_dashboard():
    path = _dashboard_path()
    if path is None:
        raise HTTPException(status_code=404, detail="no dashboard yet; run python -m scripts.quality")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/api/v1/quality/summary")
def quality_summary():
    runs = {}
    for base in (REPORTS, COMMITTED):
        if not base.exists():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(f.stem for f in folder.glob("*.json"))
            if files:
                runs.setdefault(folder.name, {"source": str(base.relative_to(ROOT)).replace("\\", "/"), "artifacts": files})
    comparison = None
    for candidate in (REPORTS / "comparison.json", COMMITTED / "iteration-1" / "comparison.json"):
        if candidate.exists():
            comparison = json.loads(candidate.read_text(encoding="utf-8"))
            break
    return {"runs": runs, "comparison": comparison}
