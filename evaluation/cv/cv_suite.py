"""Run one exported classifier over every evaluation slice and store raw outputs.

The suite only collects predictions; `evaluation/cv/report.py` turns them into metrics.
Each run writes evaluation/cv/results/<run>/predictions.csv with one row per
(slice, severity, item) holding all eleven calibrated probabilities, so any
metric can be recomputed later without touching the model again.

Slices
------
in-distribution (held-out test split, 849 images)
    clean                       unchanged
    dark / bright / blur / jpeg / lowres / rotate / occlusion   3 severities each
background (needs evaluation/cv/data/masks, see evaluation/cv/prep/make_masks.py)
    bg_removed                  fish kept, background replaced by the pad colour
    fish_erased                 fish replaced by the pad colour, background kept
    bg_swap                     fish pasted onto a background plate from the
                                other source family (Roboflow red cloth <-> Fish-gres tray)
external (evaluation/cv/protocol/screening.csv decides what is used)
    field                       iNaturalist photos of in-class species, decision=keep
    field_live                  EXCLUDED, see OUT_OF_SCOPE
    ood_unknown_fish            fish species outside the 11 classes
    ood_nonfish                 non-fish objects
    ood_synthetic               solid colours, noise and gradients (generated)

All perturbations are applied after scaling the photo to the model's working
resolution (longer side = img_size), so a severity means the same thing for a
3000 px Roboflow photo and a 520 px Fish-gres photo.

    python -m evaluation.cv.cv_suite --run baseline_vit_l_prod --export ai/fishora_dinov3_large_frozen/export
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

from evaluation.cv.common import DATA_DIR, PROTOCOL_DIR, ROOT, load_split, pick_device, run_dir

MASK_DIR = DATA_DIR / "masks"
SCREENING = PROTOCOL_DIR / "screening.csv"
SCREENING_LOCK = PROTOCOL_DIR / "screening.lock"
FOLDS = PROTOCOL_DIR / "field_folds.csv"
FOLDS_LOCK = PROTOCOL_DIR / "field_folds.lock"
SEED = 1234
MIN_FG, MAX_FG = 0.02, 0.90
PLATES_PER_FAMILY = 24

# Slices the team ruled out of the product's operating domain. Fishora sees a
# fish after it is landed, so live fish under water are not a valid input.
# Decided after an early run had already scored them (31.7% accuracy on 63
# photos), so the exclusion is disclosed in run.json and the report rather than
# made by editing the locked screening file.
OUT_OF_SCOPE = {
    "field_live": "out of operational scope: Fishora classifies fish after landing, not live fish under water",
}

PERTURBATIONS = {
    "dark":      [0.6, 0.4, 0.25],
    "bright":    [1.4, 1.8, 2.2],
    "blur":      [1.0, 2.0, 4.0],
    "jpeg":      [50, 20, 8],
    "lowres":    [128, 96, 64],
    "rotate":    [90, 180, 270],
    "occlusion": [0.15, 0.30, 0.50],
}


@dataclass
class Item:
    slice: str
    severity: str
    item_id: str
    label: str
    group: str
    image: Image.Image


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Runner:
    """Batched inference that reuses the export wrapper's own model and
    transform, so preprocessing and temperature match production exactly."""

    def __init__(self, export_dir: Path, device: str):
        from apps.cv_service.runtime import load_classifier

        self.clf = load_classifier(export_dir, device=device)
        self.cfg = self.clf.cfg
        self.classes = self.cfg["classes"]
        self.img_size = int(self.cfg["img_size"])
        self.fill = tuple(self.cfg["fill_rgb"])
        self.temperature = float(self.cfg["temperature"])
        # A gated export (evaluation/cv/kaggle/gated_inference.py) scores both
        # gates in the same pass through its own score_batch, so the suite
        # measures the serving code rather than a re-implementation.
        self.gated = bool(self.cfg.get("gate")) and hasattr(self.clf, "score_batch")

    @torch.inference_mode()
    def score(self, images: list[Image.Image], batch: int = 32):
        probs, fish, knn = [], [], []
        for i in range(0, len(images), batch):
            x = torch.stack([self.clf.transform(im.convert("RGB")) for im in images[i:i + batch]]).to(self.clf.device)
            if self.gated:
                logits, f, k = self.clf.score_batch(x)
                fish.append(f.cpu().numpy())
                knn.append(k.cpu().numpy())
            else:
                logits = self.clf.model(x).float()
            probs.append(torch.softmax(logits.float() / self.temperature, dim=1).cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, len(self.classes)))
        if not self.gated:
            return probs, None, None
        return probs, np.concatenate(fish), np.concatenate(knn)


# ---------------------------------------------------------------------------
# Image operations (all at working resolution)
# ---------------------------------------------------------------------------

def to_working(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = size / max(img.size)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.BICUBIC)


def perturb(img: Image.Image, kind: str, level, fill, mask: Image.Image | None, rng: random.Random) -> Image.Image:
    if kind == "dark" or kind == "bright":
        arr = np.asarray(img, dtype=np.float32) * float(level)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(level)))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "lowres":
        scale = int(level) / max(img.size)
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.BILINEAR)
        return small.resize(img.size, Image.Resampling.BILINEAR)
    if kind == "rotate":
        return img.rotate(int(level), expand=True, fillcolor=fill)
    if kind == "occlusion":
        # A flat box covering `level` of the fish bounding box (whole frame if
        # no usable mask), placed at a seeded position inside that box.
        x0, y0, x1, y1 = (mask.getbbox() if mask is not None and mask.getbbox() else (0, 0, *img.size))
        bw, bh = x1 - x0, y1 - y0
        side = np.sqrt(float(level))
        ow, oh = max(1, int(bw * side)), max(1, int(bh * side))
        ox = x0 + rng.randint(0, max(0, bw - ow))
        oy = y0 + rng.randint(0, max(0, bh - oh))
        out = img.copy()
        out.paste((128, 128, 128), (ox, oy, ox + ow, oy + oh))
        return out
    raise ValueError(kind)


def load_mask(image_id: str, size: tuple[int, int]) -> Image.Image | None:
    path = MASK_DIR / f"{image_id}.png"
    if not path.exists():
        return None
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    frac = (np.asarray(mask) > 0).mean()
    return mask if MIN_FG <= frac <= MAX_FG else None


def dilate(mask: Image.Image, px: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(2 * px + 1))


def background_plate(img: Image.Image, mask: Image.Image) -> Image.Image:
    """The photo with its fish removed: every masked pixel takes the colour of
    the nearest unmasked pixel, which keeps the cloth / tray texture around it."""
    arr = np.asarray(img)
    hole = np.asarray(dilate(mask, 6)) > 0
    _, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
    return Image.fromarray(arr[iy, ix])


def paste_on(fish: Image.Image, mask: Image.Image, plate: Image.Image) -> Image.Image:
    plate = ImageOps.fit(plate, fish.size, Image.Resampling.BICUBIC)
    return Image.composite(fish, plate, mask)


# ---------------------------------------------------------------------------
# Slice builders
# ---------------------------------------------------------------------------

def collect_plates(runner: Runner, samples) -> dict[str, list[Image.Image]]:
    """Background plates for bg_swap: Roboflow red-cloth and Fish-gres gray-tray
    photos with the fish filled in. Plates come from the test split itself and
    each image is swapped onto a plate from the *other* family only."""
    plates: dict[str, list[Image.Image]] = {"ROBOFLOW": [], "FISH_GRES": []}
    for s in samples:
        family_ok = (s.source_group == "ROBOFLOW" and s.background == "red") or (
            s.source_group == "FISH_GRES" and s.background == "gray"
        )
        if not family_ok or len(plates[s.source_group]) >= PLATES_PER_FAMILY:
            continue
        img = to_working(Image.open(s.path), runner.img_size)
        mask = load_mask(s.image_id, img.size)
        if mask is not None:
            plates[s.source_group].append(background_plate(img, mask))
        if all(len(v) >= PLATES_PER_FAMILY for v in plates.values()):
            break
    return plates


def iter_test_items(runner: Runner, samples, with_masks: bool, plates, corruptions: bool = True):
    size, fill = runner.img_size, runner.fill
    for idx, s in enumerate(samples):
        img = to_working(Image.open(s.path), size)
        mask = load_mask(s.image_id, img.size) if with_masks else None
        rng = random.Random(f"{SEED}-{s.image_id}")
        group = f"{s.source_group}|{s.background}"
        yield Item("clean", "0", s.image_id, s.label, group, img)
        for kind, levels in (PERTURBATIONS.items() if corruptions else ()):
            for level in levels:
                yield Item(kind, str(level), s.image_id, s.label, group,
                           perturb(img, kind, level, fill, mask, rng))
        if mask is None:
            continue
        pad = Image.new("RGB", img.size, fill)
        yield Item("bg_removed", "0", s.image_id, s.label, group, Image.composite(img, pad, mask))
        yield Item("fish_erased", "0", s.image_id, s.label, group, Image.composite(pad, img, dilate(mask, 4)))
        other = "FISH_GRES" if s.source_group == "ROBOFLOW" else "ROBOFLOW"
        if plates.get(other):
            plate = plates[other][idx % len(plates[other])]
            yield Item("bg_swap", other, s.image_id, s.label, group, paste_on(img, mask, plate))


def iter_screened_items(runner: Runner, heldout: str | None = None, skip_field: bool = False):
    """External photos. A cross-fitted model (``heldout_fold`` in its export
    config) is scored only on the field photos of its held-out fold; a model
    trained on every field photo (``field_training: all``) on none of them."""
    size = runner.img_size
    allowed = None
    if heldout:
        with FOLDS.open(encoding="utf-8") as f:
            allowed = {r["file"] for r in csv.DictReader(f) if r["fold"] == heldout}
    slice_of = {
        ("field", "keep"): "field",
        ("field", "slice"): "field_live",
        ("ood_unknown_fish", "keep"): "ood_unknown_fish",
        ("ood_nonfish", "keep"): "ood_nonfish",
    }
    with SCREENING.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        name = slice_of.get((row["set"], row["decision"]))
        if name is None or name in OUT_OF_SCOPE:
            continue
        if name == "field" and (skip_field or (allowed is not None and row["file"] not in allowed)):
            continue
        label = row["label"] if name.startswith("field") else ""
        group = "" if name.startswith("field") else row["label"]
        img = to_working(Image.open(DATA_DIR / row["file"]), size)
        yield Item(name, "0", Path(row["file"]).stem, label, group, img)


def synthetic_items(runner: Runner, n_each: int = 8) -> list[Item]:
    size = runner.img_size
    rng = np.random.default_rng(SEED)
    colours = {
        "red_cloth": (165, 35, 40), "gray_tray": (128, 128, 128), "black": (10, 10, 10),
        "white": (245, 245, 245), "blue": (40, 80, 170), "green": (60, 140, 70),
    }
    items = []
    for name, rgb in colours.items():
        base = np.full((size, size, 3), rgb, dtype=np.float32)
        noisy = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
        items.append(Item("ood_synthetic", "0", f"solid_{name}", "", "solid", Image.fromarray(noisy)))
    for i in range(n_each):
        items.append(Item("ood_synthetic", "0", f"noise_{i}", "", "noise",
                          Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))))
        a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, size)[None, :, None]
        grad = (a * (1 - t) + b * t).repeat(size, axis=0).astype(np.uint8)
        items.append(Item("ood_synthetic", "0", f"gradient_{i}", "", "gradient", Image.fromarray(grad)))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_lock(path: Path, lock: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    locked = dict(line.split("=", 1) for line in lock.read_text().splitlines() if "=" in line)
    if locked.get(f"{path.name} sha256") != digest:
        raise SystemExit(f"{path.relative_to(ROOT)} changed after it was locked; refusing to run")
    return digest


def verify_screening_lock() -> str:
    return verify_lock(SCREENING, SCREENING_LOCK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="results folder name, e.g. baseline_vit_l_prod")
    parser.add_argument("--export", required=True, type=Path, help="export dir with inference_config.json")
    parser.add_argument("--no-masks", action="store_true", help="skip background slices")
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke test)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--skip-corruptions", action="store_true",
                        help="clean, background, field and OOD slices only")
    args = parser.parse_args()

    screening_sha = verify_screening_lock()
    device = pick_device()
    export = args.export if args.export.is_absolute() else ROOT / args.export
    t0 = time.time()
    runner = Runner(export, device)
    load_s = time.time() - t0
    heldout = runner.cfg.get("heldout_fold")
    folds_sha = verify_lock(FOLDS, FOLDS_LOCK) if heldout else None
    skip_field = runner.cfg.get("field_training") == "all"

    samples = load_split("test")
    if args.limit:
        samples = samples[: args.limit]
    plates = {} if args.no_masks else collect_plates(runner, samples)

    def all_items():
        yield from iter_test_items(runner, samples, not args.no_masks, plates, not args.skip_corruptions)
        yield from iter_screened_items(runner, heldout, skip_field)
        yield from synthetic_items(runner)

    out = run_dir(args.run)
    classes = runner.classes
    counts: dict[str, int] = {}
    t1 = time.time()
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        gate_cols = ["fish_prob", "knn_dist", "gate"] if runner.gated else []
        gate_cfg = runner.cfg.get("gate") or {}
        writer.writerow(["slice", "severity", "item_id", "label", "group", "pred", "conf",
                         *[f"p_{c}" for c in classes], *gate_cols])
        chunk: list[Item] = []

        def flush():
            probs, fish, knn = runner.score([i.image for i in chunk], batch=args.batch)
            for j, (item, p) in enumerate(zip(chunk, probs)):
                gates = []
                if runner.gated:
                    # Same order as the wrapper's predict(): Gate 1 first.
                    decision = ("not_fish" if fish[j] < gate_cfg["fish_threshold"]
                                else "unknown_species" if knn[j] > gate_cfg["knn_threshold"] else "pass")
                    gates = [f"{fish[j]:.6f}", f"{knn[j]:.6f}", decision]
                writer.writerow([item.slice, item.severity, item.item_id, item.label, item.group,
                                 classes[int(p.argmax())], f"{p.max():.6f}", *[f"{v:.6f}" for v in p], *gates])
                counts[item.slice] = counts.get(item.slice, 0) + 1
            chunk.clear()
            done = sum(counts.values())
            print(f"  {done} items, {done / (time.time() - t1):.1f} img/s", flush=True)

        for item in all_items():
            chunk.append(item)
            if len(chunk) >= args.batch * 16:
                flush()
        if chunk:
            flush()
    infer_s = time.time() - t1
    items = sum(counts.values())
    print(f"{items} items across {len(counts)} slices on {device}: {counts}")

    meta = {
        "run": args.run,
        "export": str(export.relative_to(ROOT) if export.is_relative_to(ROOT) else export),
        "model_name": runner.cfg["model_name"],
        "temperature": runner.temperature,
        "abstain_threshold": runner.cfg.get("abstain_threshold"),
        "device": device,
        "items": items,
        "slice_counts": counts,
        "plates": {k: len(v) for k, v in plates.items()},
        "load_seconds": round(load_s, 2),
        "inference_seconds": round(infer_s, 2),
        "screening_sha256": screening_sha,
        "heldout_fold": heldout,
        "field_training": runner.cfg.get("field_training"),
        "gate": runner.cfg.get("gate"),
        "field_folds_sha256": folds_sha,
        "masks_used": not args.no_masks,
        "corruptions": not args.skip_corruptions,
        "out_of_scope": OUT_OF_SCOPE,
        "limit": args.limit,
        "seed": SEED,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'predictions.csv'} ({infer_s:.0f}s inference)")


if __name__ == "__main__":
    main()
