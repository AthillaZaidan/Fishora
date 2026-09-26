"""Fetch openly licensed external test images into evaluation/cv/data/.

Three sets, all held out from training by construction:

- field/<label>/        in-class fish photographed outside the training sources
                        (iNaturalist, research grade). Observations annotated
                        "Dead" (catch / market photos) are taken first.
- ood/unknown_fish/     fish species outside the 11 classes (iNaturalist)
- ood/nonfish/          objects likely to be photographed by mistake at a
                        landing site (Openverse, Creative Commons)

Every image is recorded in evaluation/cv/data/attribution.csv with its source URL,
license and author, as the CC-BY licenses require.

    python -m evaluation.cv.prep.fetch_external_data
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import httpx

from evaluation.cv.common import DATA_DIR

INAT_API = "https://api.inaturalist.org/v1/observations"
INAT_TAXA_API = "https://api.inaturalist.org/v1/taxa"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
INAT_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa"
# iNaturalist annotation "Alive or Dead" (term 17) = "Dead" (value 19).
DEAD_TERM, DEAD_VALUE = 17, 19

FIELD_TAXA = {
    "bandeng": "Chanos chanos",
    "tenggiri": "Scomberomorus commerson",
    "senangin": "Eleutheronema tetradactylum",
    "gelama_bunga": "Nibea albiflora",
    "mujair": "Oreochromis mossambicus",
    "nila": "Oreochromis niloticus",
    # R. faughni has 7 observations; the genus mixes in R. kanagurta and
    # R. brachysoma, which are also sold as kembung. Reported as a caveat.
    "kembung": "Rastrelliger",
    "kuniran": "Upeneus moluccensis",
    "tuna": "Thunnus",
}
UNKNOWN_FISH_TAXA = {
    "lele": "Clarias",
    "kakap": "Lutjanus",
    "bawal": "Pampus argenteus",
    "cakalang": "Katsuwonus pelamis",
    "kerapu": "Epinephelus",
    "patin": "Pangasianodon hypophthalmus",
    "gurame": "Osphronemus goramy",
}
NONFISH_QUERIES = [
    "plastic bucket", "woven basket", "ice cubes", "market floor", "human hand",
    "wooden table", "tarpaulin", "shrimp", "squid", "crab",
]

FIELD_PER_CLASS = 50
UNKNOWN_PER_TAXON = 15
NONFISH_PER_QUERY = 10

ATTRIBUTION_FIELDS = [
    "set", "label", "file", "source", "source_id", "source_url",
    "license", "author", "taxon", "annotated_dead",
]


def _get(client: httpx.Client, url: str, retries: int = 4, **kwargs) -> httpx.Response:
    """GET with backoff; iNaturalist drops connections under sustained load."""
    for attempt in range(retries):
        try:
            resp = client.get(url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def resolve_taxon(client: httpx.Client, name: str) -> int:
    """Exact-name taxon id. The observations endpoint silently ignores
    ``taxon_name``, so filtering must go through ``taxon_id``."""
    resp = _get(client, INAT_TAXA_API, params={"q": name, "is_active": "true", "per_page": 10})
    resp.raise_for_status()
    for taxon in resp.json()["results"]:
        if taxon["name"].lower() == name.lower():
            return taxon["id"]
    raise LookupError(f"no exact iNaturalist taxon for {name!r}")


def _inat_page(client: httpx.Client, taxon_id: int, per_page: int, dead_only: bool, exclude: set[int]):
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photo_license": INAT_LICENSES,
        "photos": "true",
        "per_page": min(200, per_page + len(exclude)),
        "order_by": "votes",
    }
    if dead_only:
        params.update({"term_id": DEAD_TERM, "term_value_id": DEAD_VALUE})
    resp = _get(client, INAT_API, params=params)
    resp.raise_for_status()
    # Defence in depth: keep only observations whose taxon is the requested
    # one or a descendant of it.
    return [
        obs for obs in resp.json()["results"]
        if obs["id"] not in exclude
        and obs.get("taxon")
        and (obs["taxon"]["id"] == taxon_id or taxon_id in obs["taxon"].get("ancestor_ids", []))
    ]


def fetch_inat(client, taxon: str, n: int, out_dir: Path, set_name: str, label: str) -> list[dict]:
    """Up to n observations of taxon, dead-annotated first, one photo each."""
    taxon_id = resolve_taxon(client, taxon)
    picked: list[tuple[dict, bool]] = []
    seen: set[int] = set()
    for dead_only in (True, False):
        if len(picked) >= n:
            break
        for obs in _inat_page(client, taxon_id, n - len(picked), dead_only, seen):
            if len(picked) >= n:
                break
            picked.append((obs, dead_only))
            seen.add(obs["id"])
        time.sleep(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for obs, dead in picked:
        photo = next((p for p in obs["photos"] if p.get("license_code")), None)
        if photo is None:
            continue
        url = photo["url"].replace("/square.", "/medium.")
        dest = out_dir / f"inat_{obs['id']}.jpg"
        if not dest.exists():
            try:
                img = _get(client, url)
                img.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  skip {obs['id']}: {exc}")
                continue
            dest.write_bytes(img.content)
            time.sleep(0.3)
        rows.append({
            "set": set_name,
            "label": label,
            "file": str(dest.relative_to(DATA_DIR)),
            "source": "inaturalist",
            "source_id": obs["id"],
            "source_url": obs["uri"],
            "license": photo["license_code"],
            "author": photo.get("attribution", ""),
            "taxon": obs["taxon"]["name"],
            "annotated_dead": dead,
        })
    print(f"  {set_name}/{label}: {len(rows)} images ({sum(r['annotated_dead'] for r in rows)} dead-annotated)")
    return rows


def fetch_openverse(client, query: str, n: int, out_dir: Path) -> list[dict]:
    resp = _get(client, OPENVERSE_API, params={
        "q": query, "page_size": n, "license_type": "all-cc", "mature": "false",
    })
    resp.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = query.replace(" ", "_")
    rows = []
    for item in resp.json()["results"][:n]:
        dest = out_dir / f"ov_{slug}_{item['id'][:8]}.jpg"
        if not dest.exists():
            try:
                # Openverse serves a resized thumbnail, so no multi-MB originals.
                img = _get(client, f"{OPENVERSE_API}{item['id']}/thumb/")
                img.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  skip {item['id']}: {exc}")
                continue
            dest.write_bytes(img.content)
            time.sleep(0.3)
        rows.append({
            "set": "ood_nonfish",
            "label": slug,
            "file": str(dest.relative_to(DATA_DIR)),
            "source": "openverse",
            "source_id": item["id"],
            "source_url": item.get("foreign_landing_url") or item["url"],
            "license": f"{item['license']} {item.get('license_version') or ''}".strip(),
            "author": item.get("creator") or "",
            "taxon": "",
            "annotated_dead": "",
        })
    print(f"  ood_nonfish/{slug}: {len(rows)} images")
    time.sleep(1)
    return rows


def main() -> None:
    rows: list[dict] = []
    headers = {"User-Agent": "fishora-eval/0.1 (research evaluation)"}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        print("field (in-class, external)")
        for label, taxon in FIELD_TAXA.items():
            rows += fetch_inat(client, taxon, FIELD_PER_CLASS, DATA_DIR / "field" / label, "field", label)
        print("ood unknown fish")
        for label, taxon in UNKNOWN_FISH_TAXA.items():
            rows += fetch_inat(client, taxon, UNKNOWN_PER_TAXON, DATA_DIR / "ood" / "unknown_fish", "ood_unknown_fish", label)
        print("ood non-fish")
        for query in NONFISH_QUERIES:
            rows += fetch_openverse(client, query, NONFISH_PER_QUERY, DATA_DIR / "ood" / "nonfish")

    out = DATA_DIR / "attribution.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTRIBUTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
