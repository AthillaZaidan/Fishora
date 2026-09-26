"""Write a synthetic taxonomy CSV so the stack can be exercised without the real dataset.

The real artifact lives under `artifacts/`, which is gitignored, so a fresh
clone cannot seed species and every commerce flow fails on species resolution.
This regenerates a stand-in with the correct shape and the eleven supported
labels.

The values are NOT authoritative. Statuses come from the in-code
TAXONOMY_STATUS_BY_LABEL, the three scientific names the tests pin are correct,
and the rest follow the identity chunks of the knowledge corpus
(artifacts/knowledge_sources), so a card never pairs one species' name with
another species' evidence (finding W18). They still need expert verification. Every row is stamped `synthetic-dev-fixture` so a
generated row can never be mistaken for dataset provenance.

Usage: python -m scripts.make_synthetic_taxonomy [--force]
"""

import argparse
import csv
import sys
from pathlib import Path

TAXONOMY_PATH = Path("artifacts/Dataset/fishora_dataset/metadata/taxonomy.csv")
SOURCE = "synthetic-dev-fixture"
UNVERIFIED = "Synthetic development fixture. Verify against the real dataset before any published claim."

FIELDS = [
    "raw_folder",
    "raw_label",
    "normalized_label",
    "scientific_name",
    "common_name_id",
    "taxonomic_rank",
    "confidence",
    "source",
    "notes",
]

# tuna first: tests/unit/test_taxonomy_seed.py assumes rows[0] is tuna, so the
# real dataset is not ordered alphabetically.
ROWS = [
    ("tuna", "Thunnus spp.", "GENUS", "medium",
     "Common name covers several species, so taxonomy is locked at genus until expert verification."),
    ("bandeng", "Chanos chanos", "SPECIES", "high", UNVERIFIED),
    ("gelama_bunga", "Nibea albiflora", "SPECIES", "medium", UNVERIFIED),
    ("gembolo", "", "VERNACULAR_AMBIGUOUS", "low",
     "Vernacular name maps to several species by region (Rastrelliger spp., Selaroides leptolepis, "
     "Caranx spp.), so no scientific name is asserted."),
    ("gulamah", "Johnius trachycephalus", "SPECIES", "medium", UNVERIFIED),
    ("kembung", "Rastrelliger faughni", "SPECIES", "high", UNVERIFIED),
    ("kuniran", "Upeneus moluccensis", "SPECIES", "medium", UNVERIFIED),
    ("mujair", "Oreochromis mossambicus", "SPECIES", "high", UNVERIFIED),
    ("nila", "Oreochromis niloticus", "SPECIES", "high", UNVERIFIED),
    ("senangin", "Eleutheronema tetradactylum", "SPECIES", "medium", UNVERIFIED),
    ("tenggiri", "Scomberomorus commerson", "SPECIES", "medium",
     "Label is also used for Scomberomorus guttatus, so this follows the primary vernacular species."),
]


def _looks_synthetic(path: Path) -> bool:
    return SOURCE in path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite even a file that does not look synthetic",
    )
    args = parser.parse_args()

    if TAXONOMY_PATH.exists() and not _looks_synthetic(TAXONOMY_PATH) and not args.force:
        # Never clobber the real dataset once someone has restored it.
        print(f"refusing to overwrite {TAXONOMY_PATH}: it does not look synthetic. Use --force.")
        return 1

    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TAXONOMY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for label, scientific, rank, confidence, notes in ROWS:
            common = " ".join(part.capitalize() for part in label.split("_"))
            writer.writerow(
                [label, common, label, scientific, common, rank, confidence, SOURCE, notes]
            )

    print(f"wrote {len(ROWS)} synthetic rows to {TAXONOMY_PATH}")
    print("Values are not authoritative. Replace with the real dataset before publishing anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
