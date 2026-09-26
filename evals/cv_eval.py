"""Import the CV (species identification) evaluation into the report pipeline.

    python -m evals.cv_eval --label baseline [--source auto|<dir>|git:<ref>]

The CV suite (evaluation/cv/, branch feat/baseline-cv) writes per-model
results/<run>/{run,metrics}.json and results/latency.json. This adapter reads
them either from the working tree (after that branch is merged) or straight
from a git ref (before), and writes one normalised reports/<label>/cv_eval.json
that evals.findings and evals.dashboard consume. Nothing is re-run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evals.corpus import ROOT
from evals.run import REPORTS_DIR, _jsonable

RESULTS = "evaluation/cv/results"
DEFAULT_REF = "origin/feat/baseline-cv"
# The CV service loads the frozen DINOv3 ViT-L export (FISHORA_CV_EXPORT_DIR),
# i.e. the linear-probe run below; it is the production reference point.
PRODUCTION_RUN = "lp_vit_l"
CORRUPTIONS = ("dark", "bright", "blur", "jpeg", "lowres", "rotate", "occlusion")


class Source:
    """Reads result files from a directory or from a git ref, same interface."""

    def __init__(self, spec: str):
        local = ROOT / RESULTS
        if spec == "auto":
            spec = str(local) if local.exists() else f"git:{DEFAULT_REF}"
        self.spec = spec
        self.ref = spec[4:] if spec.startswith("git:") else None
        self.dir = None if self.ref else Path(spec)

    def _git(self, *args) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                              encoding="utf-8").stdout

    def runs(self) -> list[str]:
        if self.dir:
            return sorted(p.name for p in self.dir.iterdir() if (p / "metrics.json").exists())
        names = self._git("ls-tree", "--name-only", f"{self.ref}:{RESULTS}").split()
        return sorted(n for n in names if self.exists(f"{n}/metrics.json"))

    def exists(self, rel: str) -> bool:
        if self.dir:
            return (self.dir / rel).exists()
        return subprocess.run(["git", "cat-file", "-e", f"{self.ref}:{RESULTS}/{rel}"], cwd=ROOT,
                              capture_output=True).returncode == 0

    def json(self, rel: str) -> dict | None:
        if not self.exists(rel):
            return None
        text = (self.dir / rel).read_text(encoding="utf-8") if self.dir else self._git("show", f"{self.ref}:{RESULTS}/{rel}")
        return json.loads(text)

    def commit(self) -> str | None:
        if not self.ref:
            return None
        return self._git("rev-parse", "--short", self.ref).strip()


def normalise(run: str, metrics: dict, meta: dict | None, latency: dict | None) -> dict:
    s = metrics["slices"]
    acc = {name: v["accuracy"] for name, v in s.items()}
    corr = {k: acc[k] for k in CORRUPTIONS if k in acc}
    worst = min(corr.items(), key=lambda kv: kv[1]) if corr else (None, None)
    ood = {
        name: {k: v.get(k) for k in ("n", "auroc", "fpr_at_95_tpr", "accepted_at_production_threshold", "mean_conf", "confident_share")}
        for name, v in metrics.get("ood", {}).items()
    }
    lat = (latency or {}).get("models", {}).get(run, {})
    field = s.get("field", {})
    return {
        "run": run,
        "model_name": metrics.get("model_name"),
        "production": run == PRODUCTION_RUN,
        "clean_accuracy": s.get("clean", {}).get("accuracy"),
        "clean_accuracy_ci": s.get("clean", {}).get("accuracy_ci"),
        "field_accuracy": field.get("accuracy"),
        "field_accuracy_ci": field.get("accuracy_ci"),
        "field_macro_f1": field.get("macro_f1"),
        "field_ece": field.get("ece"),
        "field_wrong_confident": field.get("wrong_confident"),
        "field_n": field.get("n"),
        "corruption_mean_accuracy": statistics.fmean(corr.values()) if corr else None,
        "worst_corruption": {"slice": worst[0], "accuracy": worst[1]},
        "slice_accuracy": acc,
        "bg_swap_accuracy": s.get("bg_swap", {}).get("accuracy"),
        "shortcut_rate": metrics.get("background", {}).get("fish_erased", {}).get("shortcut_rate"),
        "shortcut_chance": metrics.get("background", {}).get("fish_erased", {}).get("chance"),
        "ood": ood,
        "ood_mean_auroc": statistics.fmean(v["auroc"] for v in ood.values() if v.get("auroc") is not None) if ood else None,
        "ood_accepted_at_threshold": max((v.get("accepted_at_production_threshold") or 0) for v in ood.values()) if ood else None,
        "production_threshold": metrics.get("production_threshold"),
        "cpu_p50_ms": lat.get("p50_ms"),
        "cpu_p95_ms": lat.get("p95_ms"),
        "params_m": lat.get("params_m"),
        "weights_mb": lat.get("weights_mb"),
        "peak_rss_mb": lat.get("peak_rss_mb"),
        "items": (meta or {}).get("items"),
        "screening_sha256": (meta or {}).get("screening_sha256"),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.cv_eval")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--source", default="auto", help="auto, a results directory, or git:<ref>")
    args = parser.parse_args(argv)
    try:
        src = Source(args.source)
        runs = src.runs()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[cv] no CV results available ({type(exc).__name__}); skipping")
        return None
    latency = src.json("latency.json")
    models = {run: normalise(run, src.json(f"{run}/metrics.json"), src.json(f"{run}/run.json"), latency) for run in runs}
    result = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": src.spec,
        "source_commit": src.commit(),
        "production_run": PRODUCTION_RUN,
        "latency_machine": {k: (latency or {}).get(k) for k in ("device", "threads", "runs", "machine")},
        "models": models,
    }
    out = REPORTS_DIR / args.label / "cv_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    print(f"[cv] {len(models)} models from {src.spec} -> {out}")
    return out


if __name__ == "__main__":
    main()
