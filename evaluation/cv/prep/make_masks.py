"""Fish masks for the held-out test split, used by the background slices.

Masks come from rembg (ISNet general-use), a class-agnostic salient-object
segmenter, so they carry no knowledge of the Fishora labels. Each mask is a
single-channel PNG (255 = fish) in evaluation/cv/data/masks/<image_id>.png, and
mask_stats.csv records the foreground fraction used to flag obvious failures.

    python -m evaluation.cv.prep.make_masks --sample 20   # stratified preview + review sheet
    python -m evaluation.cv.prep.make_masks               # every test image
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from evaluation.cv.common import DATA_DIR, load_split

MASK_DIR = DATA_DIR / "masks"
MODEL = "isnet-general-use"
# A mask covering less than 2% or more than 90% of the frame is treated as a
# segmentation failure and excluded from the background slices.
MIN_FG, MAX_FG = 0.02, 0.90


def stratified_sample(samples, n: int, seed: int = 0):
    """Round-robin over (label, background) groups so the preview covers
    red cloth, gray tray and the minority backgrounds of every class."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list] = {}
    for s in samples:
        groups.setdefault((s.label, s.background), []).append(s)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    picked = []
    while len(picked) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def review_sheet(rows, out, tile: int = 220):
    """Original | overlay pairs, two pairs per row."""
    cols = 4
    n_rows = (len(rows) + 1) // 2
    sheet = Image.new("RGB", (cols * tile, n_rows * (tile + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (sample, mask, frac) in enumerate(rows):
        img = ImageOps.exif_transpose(Image.open(sample.path)).convert("RGB")
        img = ImageOps.pad(img, (tile, tile), color=(255, 255, 255))
        m = ImageOps.pad(mask, (tile, tile), color=0)
        tint = Image.new("RGB", img.size, (255, 0, 255))
        overlay = Image.composite(img, Image.blend(img, tint, 0.6), m)
        x = (i % 2) * 2 * tile
        y = (i // 2) * (tile + 16)
        sheet.paste(img, (x, y + 16))
        sheet.paste(overlay, (x + tile, y + 16))
        flag = "" if MIN_FG <= frac <= MAX_FG else " FLAG"
        draw.text((x + 3, y + 2), f"{i:02d} {sample.label}/{sample.background} fg={frac:.2f}{flag}", fill="black")
    sheet.save(out, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="stratified preview of N images")
    args = parser.parse_args()

    from rembg import new_session, remove

    session = new_session(MODEL)
    samples = load_split("test")
    if args.sample:
        samples = stratified_sample(samples, args.sample)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    rows, stats = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        dest = MASK_DIR / f"{s.image_id}.png"
        if dest.exists():
            mask = Image.open(dest).convert("L")
        else:
            img = ImageOps.exif_transpose(Image.open(s.path)).convert("RGB")
            mask = remove(img, session=session, only_mask=True).convert("L")
            mask = mask.point(lambda v: 255 if v >= 128 else 0)
            mask.save(dest)
        frac = float((np.asarray(mask) > 0).mean())
        rows.append((s, mask, frac))
        stats.append({
            "image_id": s.image_id, "label": s.label, "background": s.background,
            "fg_fraction": round(frac, 4), "flag": not (MIN_FG <= frac <= MAX_FG),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)} ({(time.time() - t0) / (i + 1):.2f}s/img)")

    suffix = f"_sample{args.sample}" if args.sample else ""
    with (DATA_DIR / f"mask_stats{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    if args.sample:
        out = DATA_DIR / "screening" / f"masks{suffix}.jpg"
        review_sheet(rows, out)
        print(f"review sheet -> {out}")
    flagged = sum(r["flag"] for r in stats)
    print(f"{len(stats)} masks, {flagged} flagged, {(time.time() - t0) / len(stats):.2f}s/img")


if __name__ == "__main__":
    main()
