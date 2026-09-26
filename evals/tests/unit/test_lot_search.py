"""Marketplace search: typed characters, card characteristics, and similar fish."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from apps.main_api.contracts import LotRecord
from apps.main_api.services.search import search_lots

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def _lot(label, *, taste=None, uses=(), similar=()):
    return LotRecord(
        id=f"lot_{label}", prediction_id=f"p_{label}", operator_id="op", species_id=f"species_{label}",
        landing_point_id="lp", quantity_kg=Decimal("10"), size_category="M",
        starting_price_per_kg=Decimal("20000"), status="active", auction_starts_at=NOW,
        auction_ends_at=NOW, public_slug=label,
        knowledge_snapshot={"common_name": label.replace("_", " ").title(), "taste": taste,
                            "processing_methods": list(uses),
                            "similar_or_substitute_species": list(similar)},
    )


LOTS = [
    _lot("bandeng", taste="Manis dan lembut.", uses=["Presto"], similar=["Nila"]),
    _lot("nila", taste="Ringan dan bersih.", uses=["Bakar"], similar=["Mujair"]),
    _lot("mujair", taste="Ringan.", uses=["Goreng"], similar=["Nila"]),
    _lot("tenggiri", taste="Gurih.", uses=["Fillet"], similar=["Kembung"]),
]


def ids(lots):
    return [lot.id for lot in lots]


def test_partial_name_matches_as_the_buyer_types():
    assert ids(search_lots(LOTS, "band").matches) == ["lot_bandeng"]


def test_a_name_match_recommends_the_similar_fish_its_card_names():
    result = search_lots(LOTS, "bandeng")
    assert ids(result.similar) == ["lot_nila"] and result.similar_to == ["Bandeng"]


def test_similarity_runs_both_ways():
    # Nila's card names Mujair, and Mujair's card names Nila.
    assert ids(search_lots(LOTS, "mujair").similar) == ["lot_nila"]


def test_characteristics_match_without_recommending():
    result = search_lots(LOTS, "gurih")
    assert ids(result.matches) == ["lot_tenggiri"] and result.similar == []


def test_every_term_must_match_and_the_generic_word_is_ignored():
    assert ids(search_lots(LOTS, "ikan ringan goreng").matches) == ["lot_mujair"]


def test_blank_query_returns_everything():
    result = search_lots(LOTS, " ")
    assert ids(result.matches) == ids(LOTS) and result.similar == []
