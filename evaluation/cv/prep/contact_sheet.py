"""Numbered contact sheets for manually screening external test images.

One sheet per folder under evaluation/cv/data/{field,ood}. Each tile is labelled with
its index and file stem so rejected images can be listed by number.

    python -m evaluation.cv.prep.contact_sheet            # all folders
    python -m evaluation.cv.prep.contact_sheet field/nila # one folder
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from evaluation.cv.common import DATA_DIR

TILE = 200
COLS = 10
LABEL_H = 18


def build_sheet(folder: Path, out: Path) -> Path:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = (len(files) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, rows * (TILE + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(files):
        x, y = (i % COLS) * TILE, (i // COLS) * (TILE + LABEL_H)
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img = ImageOps.fit(img, (TILE, TILE))
        except Exception:
            img = Image.new("RGB", (TILE, TILE), "black")
        sheet.paste(img, (x, y + LABEL_H))
        draw.text((x + 3, y + 2), f"{i:02d} {path.stem[:22]}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out


def main(argv: list[str]) -> None:
    targets = [DATA_DIR / a for a in argv] or [
        *sorted((DATA_DIR / "field").iterdir()),
        *sorted((DATA_DIR / "ood").iterdir()),
    ]
    # Kept under data/ (gitignored): the sheets contain third-party photos.
    out_dir = DATA_DIR / "screening"
    for folder in targets:
        if folder.is_dir():
            name = str(folder.relative_to(DATA_DIR)).replace("/", "__")
            print(build_sheet(folder, out_dir / f"{name}.jpg"))


if __name__ == "__main__":
    main(sys.argv[1:])
