"""Pack everything the Kaggle evaluation notebook needs into one zip.

The zip mirrors the repository layout, so evaluation/cv/cv_suite.py runs unchanged once
the notebook unpacks it and links the Fishora dataset images in:

    fishora_eval_kit/
      evaluation/cv/             suite code + protocol/ (locked screening and folds)
      evaluation/cv/data/        field + OOD photos, fish masks, gate training pool,
                                 attribution,
                                 dataset/metadata/manifest.csv
      apps/cv_service/runtime.py export loader shared with the CV service
      exports/prod_vit_l/        only with --with-prod (1.2 GB)

The photos are third-party (CC licensed, see attribution.csv), so upload the
zip as a *private* Kaggle dataset.

    python evaluation/cv/kaggle/pack_kit.py                # ~70 MB
    python evaluation/cv/kaggle/pack_kit.py --with-prod    # + production ViT-L export
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "fishora_eval_kit"

CODE = [
    "evaluation/__init__.py", "evaluation/cv/__init__.py", "evaluation/cv/common.py", "evaluation/cv/cv_suite.py", "evaluation/cv/report.py",
    "evaluation/cv/bench_latency.py", "evaluation/cv/merge_xfit.py", "evaluation/cv/protocol/screening.csv", "evaluation/cv/protocol/screening.lock",
    "evaluation/cv/protocol/field_folds.csv", "evaluation/cv/protocol/field_folds.lock",
    "apps/__init__.py", "apps/cv_service/__init__.py", "apps/cv_service/runtime.py",
]
DATA_DIRS = ["evaluation/cv/data/field", "evaluation/cv/data/ood", "evaluation/cv/data/masks",
             "evaluation/cv/data/gate_train"]
DATA_FILES = ["evaluation/cv/data/attribution.csv", "evaluation/cv/data/mask_stats.csv",
              "evaluation/cv/data/gate_attribution.csv", "evaluation/cv/data/gate_leak_report.csv",
              "evaluation/cv/data/gate_screening.csv"]
MANIFEST = "evaluation/cv/data/dataset/metadata/manifest.csv"
PROD_EXPORT = "ai/fishora_dinov3_large_frozen/export"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-prod", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "cv" / "kaggle_outputs" / "fishora_eval_kit.zip")
    args = parser.parse_args()

    entries: list[tuple[Path, str]] = []
    for rel in CODE + DATA_FILES + [MANIFEST]:
        entries.append((ROOT / rel, rel))
    for rel in DATA_DIRS:
        for p in sorted((ROOT / rel).rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                entries.append((p, str(p.relative_to(ROOT))))
    if args.with_prod:
        for name in ("inference.py", "inference_config.json", "model_state_dict.pt"):
            entries.append((ROOT / PROD_EXPORT / name, f"exports/prod_vit_l/{name}"))

    missing = [str(src) for src, _ in entries if not src.exists()]
    if missing:
        raise SystemExit(f"missing: {missing[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, rel in entries:
            # Weights and JPEGs are already compressed; storing them is faster.
            method = zipfile.ZIP_STORED if src.suffix in {".pt", ".jpg", ".png"} else zipfile.ZIP_DEFLATED
            zf.write(src, f"{PREFIX}/{rel}", compress_type=method)
    size = args.out.stat().st_size / 2**20
    print(f"{len(entries)} files -> {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
