"""Seed a demonstrable marketplace: a handful of live lots across species and landing points.

A fresh database has taxonomy and landing points but no lots, so /marketplace
renders its empty state and nothing downstream of it can be exercised by hand.
This creates enough lots to show filtering, sizes, price spread, closing-soon
countdowns, and an already-allocated lot that a buyer can review.

Also clears rows left behind when the pytest suite is pointed at a development
database instead of a throwaway one. Those tests use synthetic labels like
`race_tenggiri`, which otherwise show up in the UI as real commodities.

Every id is prefixed `demo_` so this is repeatable and so a demo row can never
be mistaken for a real listing.

Usage: FISHORA_DATABASE_URL=... python -m scripts.seed_demo_lots [--keep-test-rows]
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import delete, select

from apps.main_api.config import MainSettings
from apps.main_api.db.models import (
    Bid,
    CommercialBuyerReview,
    LandingPoint,
    Lot,
    Prediction,
)
from apps.main_api.db.session import session_factory
from apps.main_api.services.reference_cards import REFERENCE

DEMO_PREFIX = "demo_"
# The seeded demo logins from apps/main_api/services/session.py. Inventing ids
# here would leave these lots invisible to the operator's own listings page and
# would keep the review form from ever appearing for the buyer who won a lot.
DEMO_OPERATOR = "op_rian"
DEMO_BUYER = "buyer_dewi"

# Test fixtures leak into a development database when pytest is pointed at it.
# These are the synthetic labels the suite creates, matched on id.
TEST_ROW_MARKER = "_race_"

# (label, size, kg, price/kg, landing point, fisher group, hours until close)
#
# All eleven supported species now have catch photography, so every one can be
# seeded and the marketplace shows the full commodity range.
# The spread is deliberate: two species repeat at different sizes and prices so
# the filters and the sort have something to actually separate, and the closing
# windows are staggered so the countdowns do not all read the same.
DEMO_LOTS = [
    ("tenggiri", "L", "120.000", "68000.00", "lp_muara_angke", "KUB Bahari Jaya", 6),
    ("tuna", "L", "240.000", "92000.00", "lp_cilacap", "KUB Samudra Biru", 11),
    ("kembung", "M", "310.000", "27000.00", "lp_muara_angke", "KUB Bahari Jaya", 3),
    ("kembung", "S", "180.000", "21000.00", "lp_karangsong", None, 30),
    ("bandeng", "M", "150.000", "32000.00", "lp_karangsong", "KUB Tambak Makmur", 44),
    ("gelama_bunga", "S", "85.000", "17500.00", "lp_cilacap", None, 22),
    ("gembolo", "M", "140.000", "23000.00", "lp_cilacap", "KUB Mina Sejahtera", 38),
    ("gulamah", "M", "165.000", "24500.00", "lp_muara_angke", "KUB Mina Sejahtera", 17),
    ("kuniran", "S", "260.000", "18500.00", "lp_muara_angke", None, 26),
    ("mujair", "M", "130.000", "26000.00", "lp_karangsong", "KUB Tambak Makmur", 34),
    ("senangin", "L", "95.000", "61000.00", "lp_cilacap", "KUB Samudra Biru", 15),
    # Same species as the allocated lot below, different fisherman and landing
    # point: reviews are keyed to the species, so a review earned on that lot
    # has to surface here too. Without this pair there is nothing to show it on.
    ("nila", "L", "175.000", "34000.00", "lp_muara_angke", "KUB Bahari Jaya", 9),
]

# One lot already won, so the review flow has something to attach to.
ALLOCATED_LOT = ("nila", "M", "200.000", "29000.00", "lp_karangsong", "KUB Tambak Makmur")

# A snapshot per species, so the knowledge panel and the buyer-preference
# matching have something to work on without the generation service running.
# Every card carries an explicit limitation naming itself as a fixture, because
# an unlabelled synthetic card is indistinguishable from a retrieved one and the
# whole point of the panel is that its claims are traceable.
FIXTURE_LIMITATION = (
    "This card is sample data for development, not the result of a verified source search."
)

# The reference notes every seeded card is written from.
KNOWLEDGE = REFERENCE


def _snapshot(label, common_name, scientific_name, taxonomy_status):
    """A synthetic knowledge card, or None for a species with nothing written."""
    entry = KNOWLEDGE.get(label)
    if entry is None:
        return None
    physical, taste, texture, processing, uses, similar, segments = entry
    return {
        "common_name": common_name,
        "scientific_name": scientific_name,
        "taxonomy_status": taxonomy_status,
        "physical_characteristics": physical,
        "taste": taste,
        "texture": texture,
        "processing_methods": processing,
        "commercial_uses": uses,
        "similar_or_substitute_species": similar,
        "potential_buyer_segments": segments,
        "limitations": [FIXTURE_LIMITATION],
        # Empty on purpose: a fixture has no retrieved sources, and inventing
        # citations would be worse than showing none.
        "sources": [],
    }


def _purge_test_rows(session) -> int:
    """Drop the synthetic rows pytest leaves behind. Only ever touches ids containing `_race_`."""
    from apps.main_api.db.models import FishSpecies

    lot_ids = [
        row[0]
        for row in session.execute(select(Lot.id).where(Lot.id.contains(TEST_ROW_MARKER)))
    ]
    removed = 0
    if lot_ids:
        removed += session.execute(delete(Bid).where(Bid.lot_id.in_(lot_ids))).rowcount or 0
        removed += session.execute(delete(Lot).where(Lot.id.in_(lot_ids))).rowcount or 0

    # Order matters: lots reference predictions, species, and landing points.
    for model in (Prediction, LandingPoint, FishSpecies):
        stmt = delete(model).where(model.id.contains(TEST_ROW_MARKER))
        removed += session.execute(stmt).rowcount or 0
    return removed


def _clear_demo_rows(session) -> int:
    """Drop the previous seed so a species dropped from the set cannot linger."""
    lot_ids = [
        row[0]
        for row in session.execute(select(Lot.id).where(Lot.id.startswith(DEMO_PREFIX)))
    ]
    removed = 0
    if lot_ids:
        # Reviews and bids reference the lot, so they go first.
        removed += (
            session.execute(
                delete(CommercialBuyerReview).where(CommercialBuyerReview.lot_id.in_(lot_ids))
            ).rowcount
            or 0
        )
        removed += session.execute(delete(Bid).where(Bid.lot_id.in_(lot_ids))).rowcount or 0
        removed += session.execute(delete(Lot).where(Lot.id.in_(lot_ids))).rowcount or 0
    # Predictions are deliberately left alone. They are upserted on every run,
    # and a lot published by hand through the API can reference one, so deleting
    # them by prefix trips the lots foreign key.
    return removed


def _clear_all_lots(session) -> int:
    """Clear every lot, not just the seeded ones.

    The PRD walkthrough publishes a real lot on each run, so a development
    database collects one more allocated Tenggiri every time the e2e suite is
    run and the operator's list stops being readable.
    """
    removed = session.execute(delete(CommercialBuyerReview)).rowcount or 0
    removed += session.execute(delete(Bid)).rowcount or 0
    removed += session.execute(delete(Lot)).rowcount or 0
    return removed


def _species(session, label: str):
    from apps.main_api.db.models import FishSpecies

    found = session.execute(
        select(FishSpecies).where(FishSpecies.normalized_label == label)
    ).scalar_one_or_none()
    if found is None:
        raise SystemExit(
            f"species '{label}' is not seeded. Run `python -m scripts.seed_taxonomy` first."
        )
    return found


def _write_lot(session, now, index, spec, *, allocated=False):
    label, size, kg, price, landing_point, group = spec[:6]
    hours = 0 if allocated else spec[6]
    slug = f"{DEMO_PREFIX}{label}-{size.lower()}-{index}"
    prediction_id = f"{DEMO_PREFIX}pred_{index}"
    species = _species(session, label)
    species_id = species.id

    session.merge(
        Prediction(
            id=prediction_id,
            image_reference=f"demo/{label}.jpg",
            predicted_species_id=species_id,
            confidence=0.0,
            top_candidates=[],
            # Not a model output: nothing was inferred for a seeded row.
            model_version="demo-fixture",
            verification_status="confirmed",
            verified_species_id=species_id,
        )
    )
    session.merge(
        Lot(
            id=f"{DEMO_PREFIX}lot_{index}",
            prediction_id=prediction_id,
            operator_id=DEMO_OPERATOR,
            species_id=species_id,
            landing_point_id=landing_point,
            quantity_kg=Decimal(kg),
            size_category=size,
            starting_price_per_kg=Decimal(price),
            status="allocated" if allocated else "active",
            auction_starts_at=now - timedelta(hours=2),
            auction_ends_at=now - timedelta(hours=1) if allocated else now + timedelta(hours=hours),
            knowledge_snapshot=_snapshot(
                label, species.common_name_id, species.scientific_name, species.taxonomy_status
            ),
            public_slug=slug,
            allocated_buyer_id=DEMO_BUYER if allocated else None,
            seller_fisher_group=group,
        )
    )
    return slug


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-test-rows",
        action="store_true",
        help="leave pytest's synthetic rows in place",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear every lot first, including ones published by hand or by the e2e walkthrough",
    )
    args = parser.parse_args()

    # Passed in, not called at import: the auction windows must be relative to
    # the moment of seeding or every countdown seeds already-expired.
    now = datetime.now(timezone.utc)

    settings = MainSettings()
    with session_factory(settings)() as session:
        if not args.keep_test_rows:
            removed = _purge_test_rows(session)
            print(f"removed {removed} row(s) left by the test suite")

        if args.reset:
            wiped = _clear_all_lots(session)
            print(f"reset: cleared {wiped} row(s) across every lot")
        else:
            cleared = _clear_demo_rows(session)
            print(f"cleared {cleared} row(s) from the previous demo seed")

        slugs = [_write_lot(session, now, i, spec) for i, spec in enumerate(DEMO_LOTS, start=1)]
        allocated_index = len(DEMO_LOTS) + 1
        slugs.append(_write_lot(session, now, allocated_index, ALLOCATED_LOT, allocated=True))

        # A review on the allocated lot. Reviews are keyed to the species, so
        # this also appears on the other fisherman's lot of the same fish, which
        # is the whole point of the feature and is invisible with no rows.
        session.merge(
            CommercialBuyerReview(
                id=f"{DEMO_PREFIX}review_1",
                lot_id=f"{DEMO_PREFIX}lot_{allocated_index}",
                species_id=_species(session, ALLOCATED_LOT[0]).id,
                buyer_id=DEMO_BUYER,
                actual_use="Fillet for catering",
                processing_suitability=4,
                substitute_acceptance=True,
                comment="Even sizes, and the bones come away easily.",
            )
        )
        session.commit()

    print(f"seeded {len(slugs)} demo lots ({len(DEMO_LOTS)} live, 1 allocated to {DEMO_BUYER})")
    print("Values are illustrative. Not real catch data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
