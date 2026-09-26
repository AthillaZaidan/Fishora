"""Fetch training data for the fish / non-fish gate into evaluation/cv/data/gate_train/.

The gate is trained, so none of this may overlap the locked test sets:

- nonfish/  Openverse photos for queries disjoint from the test non-fish
            queries (landing-site scenes, gear, non-fish seafood, people).
- fish/     iNaturalist fish of taxa outside both the 11 classes and the
            unknown-species test set, so the gate learns "fish" rather than
            "one of our 11 species". Dead-annotated observations first.

The dataset's own train split is added as fish positives in the notebook.

Leak checks against everything under data/field and data/ood:
same source id, same iNaturalist observer (fish), and a near-identical image
(64-bit difference hash, Hamming distance <= 6). Dropped files are deleted
and listed in gate_leak_report.csv.

    python -m evaluation.cv.prep.fetch_gate_data
"""

from __future__ import annotations

import csv
import time

import httpx
import numpy as np
import pandas as pd
from PIL import Image

from evaluation.cv.common import DATA_DIR
from evaluation.cv.prep.fetch_external_data import (
    ATTRIBUTION_FIELDS, NONFISH_QUERIES, OPENVERSE_API, UNKNOWN_FISH_TAXA, FIELD_TAXA, _get, fetch_inat,
)

OUT_DIR = DATA_DIR / "gate_train"

# None of these is a test query; several are near neighbours of test queries on
# purpose (prawn vs shrimp, octopus vs squid) so the gate is trained on the
# hard cases, while the dHash check guards against the same photo reappearing.
GATE_NONFISH_QUERIES = [
    "prawn", "lobster", "octopus", "cuttlefish", "mussels", "clams", "oysters", "sea urchin",
    "jellyfish", "seaweed", "sea cucumber", "starfish",
    "fishing boat deck", "fishing net", "harbor dock", "styrofoam box", "cooler box", "plastic crate",
    "rope", "rubber boots", "weighing scale", "plastic bag", "sack", "concrete floor", "sand beach",
    "vegetables market", "fruit stall", "cat", "chicken", "person portrait", "smartphone", "motorcycle",
]
NONFISH_PER_QUERY = 20

# Fish landed or sold in Indonesia, outside FIELD_TAXA and UNKNOWN_FISH_TAXA.
GATE_FISH_TAXA = {
    "baronang": "Siganus", "kuwe": "Caranx", "selar": "Selaroides leptolepis", "layang": "Decapterus",
    "tembang": "Sardinella", "teri": "Stolephorus", "kakap_putih": "Lates calcarifer",
    "layur": "Trichiurus lepturus", "belanak": "Mugil cephalus", "manyung": "Arius",
    "gabus": "Channa striata", "ikan_mas": "Cyprinus carpio", "betok": "Anabas testudineus",
    "sunu": "Plectropomus leopardus", "kurisi": "Nemipterus", "swanggi": "Priacanthus",
    "lemuru": "Sardinella lemuru", "tongkol": "Euthynnus affinis", "cucut": "Carcharhinus",
    "pari": "Neotrygon",
}
FISH_PER_TAXON = 20

DHASH_MAX_DISTANCE = 6


def observer(author: str) -> str:
    name = str(author).removeprefix("(c)").strip().split(",")[0].strip()
    return "" if name in ("", "no rights reserved") else name


def dhash(path) -> int:
    img = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (img[:, 1:] > img[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fetch_openverse_pages(client, query: str, n: int, skip_ids: set[str]) -> list[dict]:
    out_dir = OUT_DIR / "nonfish"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = query.replace(" ", "_")
    rows: list[dict] = []
    page = 1
    while len(rows) < n and page <= 3:
        resp = _get(client, OPENVERSE_API, params={
            "q": query, "page_size": 20, "page": page, "license_type": "all-cc", "mature": "false",
        })
        results = resp.json()["results"]
        if not results:
            break
        for item in results:
            if len(rows) >= n or item["id"] in skip_ids:
                continue
            dest = out_dir / f"ov_{slug}_{item['id'][:8]}.jpg"
            if not dest.exists():
                try:
                    dest.write_bytes(_get(client, f"{OPENVERSE_API}{item['id']}/thumb/").content)
                except httpx.HTTPError as exc:
                    print(f"  skip {item['id']}: {exc}")
                    continue
                time.sleep(0.3)
            skip_ids.add(item["id"])
            rows.append({
                "set": "gate_nonfish", "label": slug, "file": str(dest.relative_to(DATA_DIR)),
                "source": "openverse", "source_id": item["id"],
                "source_url": item.get("foreign_landing_url") or item["url"],
                "license": f"{item['license']} {item.get('license_version') or ''}".strip(),
                "author": item.get("creator") or "", "taxon": "", "annotated_dead": "",
            })
        page += 1
        time.sleep(1)
    print(f"  gate_nonfish/{slug}: {len(rows)} images")
    return rows


def main() -> None:
    assert not set(GATE_NONFISH_QUERIES) & set(NONFISH_QUERIES), "gate query reused from the test set"
    assert not set(GATE_FISH_TAXA.values()) & (set(FIELD_TAXA.values()) | set(UNKNOWN_FISH_TAXA.values()))

    test = pd.read_csv(DATA_DIR / "attribution.csv", dtype=str)
    test_ids = set(test.source_id)
    test_observers = {observer(a) for a in test[test.source == "inaturalist"].author} - {""}

    rows: list[dict] = []
    headers = {"User-Agent": "fishora-eval/0.1 (research evaluation)"}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        print("gate fish (iNaturalist, taxa outside every test set)")
        for label, taxon in GATE_FISH_TAXA.items():
            try:
                rows += fetch_inat(client, taxon, FISH_PER_TAXON, OUT_DIR / "fish" / label, "gate_fish", label)
            except (LookupError, httpx.HTTPError) as exc:
                print(f"  skip taxon {taxon}: {exc}")
        print("gate non-fish (Openverse, queries disjoint from the test set)")
        for query in GATE_NONFISH_QUERIES:
            try:
                rows += fetch_openverse_pages(client, query, NONFISH_PER_QUERY, set(test_ids))
            except httpx.HTTPError as exc:
                print(f"  skip query {query}: {exc}")

    print("leak check against data/field and data/ood")
    test_hashes = []
    for p in list((DATA_DIR / "field").rglob("*.jpg")) + list((DATA_DIR / "ood").rglob("*.jpg")):
        try:
            test_hashes.append((dhash(p), str(p.relative_to(DATA_DIR))))
        except OSError:
            pass
    test_hash_arr = np.array([h for h, _ in test_hashes], dtype=np.uint64)

    kept, dropped = [], []
    for row in rows:
        path = DATA_DIR / row["file"]
        reason = ""
        if str(row["source_id"]) in test_ids:
            reason = "same_source_id"
        elif row["source"] == "inaturalist" and observer(row["author"]) in test_observers:
            reason = "same_observer"
        else:
            try:
                h = np.uint64(dhash(path))
            except OSError:
                reason = "unreadable"
            else:
                dist = np.array([bin(int(x)).count("1") for x in (test_hash_arr ^ h)])
                if dist.size and dist.min() <= DHASH_MAX_DISTANCE:
                    reason = f"near_duplicate:{test_hashes[int(dist.argmin())][1]}"
        if reason:
            path.unlink(missing_ok=True)
            dropped.append({"file": row["file"], "reason": reason})
        else:
            kept.append(row)

    with (DATA_DIR / "gate_attribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTRIBUTION_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    pd.DataFrame(dropped, columns=["file", "reason"]).to_csv(DATA_DIR / "gate_leak_report.csv", index=False)
    by_set = pd.DataFrame(kept).groupby("set").size().to_dict() if kept else {}
    print(f"kept {len(kept)} {by_set}, dropped {len(dropped)} -> gate_attribution.csv, gate_leak_report.csv")


if __name__ == "__main__":
    main()
