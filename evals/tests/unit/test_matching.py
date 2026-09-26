"""Buyer preference matching: vocabulary, negation, business type, distance."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from apps.main_api.contracts import BuyerPreferenceRecord, LandingPointRecord, LotRecord
from apps.main_api.services.matching import (
    CRITERIA,
    MATCH_THRESHOLD,
    WEIGHTS,
    match_lot,
    match_terms,
    matching_choices,
    phrase_terms,
    recommend,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
MUARA_ANGKE = LandingPointRecord(id="lp_muara_angke", name="PPI Muara Angke", latitude=-6.104, longitude=106.792)
CILACAP = LandingPointRecord(id="lp_cilacap", name="TPI Cilacap", latitude=-7.732, longitude=109.015)
POINTS = {point.id: point for point in (MUARA_ANGKE, CILACAP)}

# Shaped like the English cards the generator now writes.
TENGGIRI_CARD = {
    "taste": "Savory and not too oily.",
    "texture": "Firm with fine fibers and white flesh.",
    "physical_characteristics": "Elongated body with faint vertical bars.",
    "processing_methods": ["Fillet", "Smoked", "Fish balls"],
    "commercial_uses": ["Restaurants", "Fish ball processing", "Catering"],
    "potential_buyer_segments": ["Fish ball processors", "Seafood restaurants"],
}


def _lot(card=None, *, lot_id="lot_1", landing="lp_muara_angke", price="68000", kg="120"):
    return LotRecord(
        id=lot_id, prediction_id=f"p_{lot_id}", operator_id="op", species_id="species_tenggiri",
        landing_point_id=landing, quantity_kg=Decimal(kg), size_category="L",
        starting_price_per_kg=Decimal(price), status="active", auction_starts_at=NOW,
        auction_ends_at=NOW, public_slug=lot_id, knowledge_snapshot=card,
    )


def _prefs(*, uses=(), chars=(), business="restaurant", max_price=None, min_kg=None,
           lat=-6.2, lon=106.85):
    return BuyerPreferenceRecord(
        buyer_id="buyer", business_type=business, intended_uses=list(uses),
        characteristics=list(chars),
        max_price_per_kg=None if max_price is None else Decimal(max_price),
        min_quantity_kg=None if min_kg is None else Decimal(min_kg),
        latitude=lat, longitude=lon,
    )


def _reason(result, criterion):
    return next(reason for reason in result.reasons if reason.criterion == criterion)


def test_weights_sum_to_one_and_cover_every_criterion():
    assert set(WEIGHTS) == set(CRITERIA)
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("form", ["fried", "fry", "frying", "Fries", "Deep-fried", "goreng"])
def test_frying_inflections_meet(form):
    assert phrase_terms(form) & phrase_terms("fried")


@pytest.mark.parametrize(("chip", "card"), [
    ("grilled", "Grill"), ("grilled", "suitable for grilling"), ("grilled", "Bakar"),
    ("smoked", "Smoking"), ("smoked", "Pengasapan"), ("canned", "Canneries"),
    ("fish balls", "Fish ball processing"), ("boiled", "Pindang (salt-boiled)"),
])
def test_chip_vocabulary_meets_card_wording(chip, card):
    assert matching_choices([chip], match_terms([card])) == [chip]


def test_modal_can_is_not_canned():
    assert "canned" not in match_terms(["Can be grilled or fried."])


@pytest.mark.parametrize("text", [
    "not oily", "Savory and not too oily.", "without any oiliness", "tidak terlalu berminyak",
    "Mild, non-oily flesh", "low fat",
])
def test_negated_terms_are_not_asserted(text):
    assert "oily" not in match_terms([text])


def test_negation_ends_at_the_clause():
    terms = match_terms(["Not too oily, firm and savory."])
    assert {"firm", "savory"} <= terms and "oily" not in terms
    assert "firm" in match_terms(["Not oily but firm."])


def test_a_multi_word_choice_needs_every_word():
    assert matching_choices(["white flesh"], match_terms(["Soft pink flesh."])) == []
    assert matching_choices(["white flesh"], match_terms(["White, firm meat."])) == ["white flesh"]


def test_negated_characteristic_does_not_match_the_lot():
    result = match_lot(_lot(TENGGIRI_CARD), _prefs(chars=["oily"]), MUARA_ANGKE)
    assert not _reason(result, "characteristics").met


def test_card_wording_matches_english_chips():
    result = match_lot(
        _lot(TENGGIRI_CARD), _prefs(uses=["fried", "fish balls"], chars=["savory", "white flesh"]), MUARA_ANGKE
    )
    assert _reason(result, "intended_use").met
    assert "fish balls" in _reason(result, "intended_use").detail
    assert "fried" not in _reason(result, "intended_use").detail
    assert _reason(result, "characteristics").met


@pytest.mark.parametrize(("business", "met"), [
    ("restaurant", True), ("processor", True), ("catering", True), ("hotel", True),
    ("exporter", False), ("supermarket", False),
    # Values the Indonesian form saved.
    ("rumah_makan", True), ("pengolah", True),
])
def test_business_type_reads_buyer_segments(business, met):
    result = match_lot(_lot(TENGGIRI_CARD), _prefs(business=business), MUARA_ANGKE)
    assert _reason(result, "business_type").met is met


def test_business_type_is_a_light_signal():
    base = _prefs(uses=["fillet"])
    with_segment = match_lot(_lot(TENGGIRI_CARD), base, MUARA_ANGKE).score
    without = match_lot(_lot(TENGGIRI_CARD), _prefs(uses=["fillet"], business="exporter"), MUARA_ANGKE).score
    assert with_segment - without == pytest.approx(WEIGHTS["business_type"])


def test_threshold_needs_the_fish_itself_to_fit():
    # Logistics and business type alone stay under the badge.
    logistics = match_lot(_lot(TENGGIRI_CARD), _prefs(), MUARA_ANGKE)
    assert logistics.score < MATCH_THRESHOLD and not logistics.matched
    fits = match_lot(_lot(TENGGIRI_CARD), _prefs(chars=["firm"], business="exporter"), MUARA_ANGKE)
    assert fits.matched
    # use + price + volume is exactly the threshold; float addition must not drop it below.
    edge = match_lot(_lot(TENGGIRI_CARD), _prefs(uses=["smoked"], business="exporter"), None)
    assert edge.score == MATCH_THRESHOLD and edge.matched


def test_price_and_volume_limits():
    result = match_lot(_lot(TENGGIRI_CARD), _prefs(max_price="60000", min_kg="200"), MUARA_ANGKE)
    assert not _reason(result, "price").met and "above your Rp 60.000/kg limit" in _reason(result, "price").detail
    assert not _reason(result, "volume").met and "below your 200 kg minimum" in _reason(result, "volume").detail


def test_reason_details_are_english():
    result = match_lot(_lot(TENGGIRI_CARD), _prefs(uses=["fried"]), MUARA_ANGKE)
    text = " ".join(reason.detail for reason in result.reasons).casefold()
    for word in ("cocok", "ciri", "harga", "lokasi", "tidak"):
        assert word not in text


def test_distance_limits_recommendations_to_the_radius():
    lots = [_lot(TENGGIRI_CARD, lot_id="jakarta"), _lot(TENGGIRI_CARD, lot_id="cilacap", landing="lp_cilacap")]
    in_jakarta = recommend(lots, _prefs(), POINTS)
    assert [lot.id for lot, _ in in_jakarta] == ["jakarta"]
    # A buyer who says they are in Cilacap now reaches the Cilacap lot.
    in_cilacap = recommend(lots, _prefs(lat=-7.72, lon=109.0), POINTS)
    assert [lot.id for lot, _ in in_cilacap] == ["cilacap"]
    assert _reason(in_cilacap[0][1], "distance").met


def test_unknown_landing_point_is_not_a_distance_match():
    result = match_lot(_lot(TENGGIRI_CARD), _prefs(), None)
    assert not _reason(result, "distance").met


def test_lot_without_a_card_scores_only_logistics():
    result = match_lot(_lot(None), _prefs(uses=["fried"], chars=["firm"]), MUARA_ANGKE)
    assert result.score == pytest.approx(WEIGHTS["price"] + WEIGHTS["volume"] + WEIGHTS["distance"])


def test_seeded_cards_overlap_the_form_chips():
    """Every chip the preference form offers matches at least one seeded card,
    so a demo buyer never picks an option that can match nothing."""
    from scripts.seed_demo_lots import KNOWLEDGE, _snapshot

    uses = ["fried", "grilled", "fillet", "smoked", "steamed", "boiled", "fish balls", "curry", "canned"]
    chars = ["savory", "firm", "soft", "mild", "sweet", "oily", "white flesh"]
    cards = [_snapshot(label, label, None, "species") for label in KNOWLEDGE]
    lots = [_lot(card, lot_id=f"lot_{index}") for index, card in enumerate(cards)]
    for chip in uses:
        assert any(matching_choices([chip], match_terms(_uses(lot))) for lot in lots), chip
    for chip in chars:
        assert any(matching_choices([chip], match_terms(_chars(lot))) for lot in lots), chip


def _uses(lot):
    from apps.main_api.services.matching import lot_uses

    return lot_uses(lot)


def _chars(lot):
    from apps.main_api.services.matching import lot_characteristics

    return lot_characteristics(lot)
