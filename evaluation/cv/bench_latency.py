"""CPU latency and memory of each exported classifier, as an operator would see it.

Each model is measured in its own subprocess so load time and peak memory are
not polluted by a previous model. The timed call is the production wrapper's
`predict()` on a real photo (decode + resize-pad + forward + softmax), batch 1,
with the torch thread count pinned so every model gets the same CPU budget.

    python -m evaluation.cv.bench_latency baseline_vit_l_prod=ai/fishora_dinov3_large_frozen/export lp_vit_s=...
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from evaluation.cv.common import RESULTS_DIR, ROOT, load_split

THREADS = 4
WARMUP = 10
RUNS = 100


def measure(export: str, image: str) -> dict:
    import resource
    import time

    import numpy as np
    import torch

    torch.set_num_threads(THREADS)
    from apps.cv_service.runtime import load_classifier

    t0 = time.perf_counter()
    clf = load_classifier(Path(export), device="cpu")
    load_s = time.perf_counter() - t0
    for _ in range(WARMUP):
        clf.predict(image)
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        clf.predict(image)
        times.append((time.perf_counter() - t) * 1000)
    params = sum(p.numel() for p in clf.model.parameters())
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    return {
        "load_seconds": round(load_s, 2),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "mean_ms": round(float(np.mean(times)), 1),
        "params_m": round(params / 1e6, 2),
        "weights_mb": round((Path(export) / "model_state_dict.pt").stat().st_size / 2**20, 1),
        "peak_rss_mb": round(rss_mb, 0),
    }


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--child":
        print(json.dumps(measure(argv[1], argv[2])))
        return
    pairs = [a.split("=", 1) for a in argv]
    image = str(load_split("test")[0].path)
    rows = {}
    for run, export in pairs:
        export_path = Path(export) if Path(export).is_absolute() else ROOT / export
        proc = subprocess.run(
            [sys.executable, "-m", "evaluation.cv.bench_latency", "--child", str(export_path), image],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        rows[run] = json.loads(proc.stdout.strip().splitlines()[-1])
        print(run, rows[run], flush=True)
    out = {
        "machine": f"{platform.machine()} {platform.processor() or ''} {platform.platform()}".strip(),
        "device": "cpu",
        "threads": THREADS,
        "warmup": WARMUP,
        "runs": RUNS,
        "image": str(Path(image).relative_to(ROOT)) if Path(image).is_relative_to(ROOT) else image,
        "models": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RESULTS_DIR / 'latency.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
