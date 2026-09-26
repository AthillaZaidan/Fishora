"""Shared paths and model loading for the Fishora evaluation suite.

The classifier is loaded through the same export wrapper the CV service uses
(`apps.cv_service.runtime.load_classifier`), so the suite measures the exact
preprocessing and temperature that production serves.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

CV_DIR = Path(__file__).resolve().parent
ROOT = CV_DIR.parents[1]
PROTOCOL_DIR = CV_DIR / "protocol"
DATA_DIR = Path(os.environ.get("FISHORA_EVAL_DATA", CV_DIR / "data"))
DATASET_DIR = DATA_DIR / "dataset"
RESULTS_DIR = Path(os.environ.get("FISHORA_EVAL_RESULTS", CV_DIR / "results"))
EXPORT_DIR = Path(
    os.environ.get("FISHORA_CV_EXPORT_DIR", ROOT / "ai" / "fishora_dinov3_large_frozen" / "export")
)


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    label: str
    source_group: str
    background: str
    quality: str


def load_split(split: str = "test") -> list[Sample]:
    """Rows of the held-out split with the metadata used for slicing."""
    manifest = DATASET_DIR / "metadata" / "manifest.csv"
    with manifest.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return [
        Sample(
            image_id=r["image_id"],
            path=DATASET_DIR / r["clean_path"],
            label=r["normalized_label"],
            source_group=r["source_group"],
            background=r["background_category"],
            quality=r["quality_status"],
        )
        for r in rows
    ]


def pick_device() -> str:
    import torch

    if os.environ.get("FISHORA_CV_DEVICE"):
        return os.environ["FISHORA_CV_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_classifier():
    from apps.cv_service.runtime import load_classifier as _load

    return _load(EXPORT_DIR, device=pick_device())


def run_dir(run_name: str) -> Path:
    path = RESULTS_DIR / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
