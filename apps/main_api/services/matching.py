import re
from dataclasses import dataclass
from decimal import Decimal

from apps.main_api.contracts import BuyerPreferenceRecord, LandingPointRecord, LotRecord
from apps.main_api.services.geo import (
    DEFAULT_SERVICEABILITY_RADIUS_KM,
    haversine_km,
    within_serviceability,
)

# Sums to 1, so a score reads as the share of the buyer's profile a lot meets.
# Business type is a light signal on purpose: a card's buyer segments are a
# short generated list, and missing one should not sink an otherwise good lot.
WEIGHTS = {
    "intended_use": 0.25,
    "characteristics": 0.20,
    "business_type": 0.10,
    "price": 0.20,
    "volume": 0.15,
    "distance": 0.10,
}

CRITERIA = ("intended_use", "characteristics", "business_type", "price", "volume", "distance")

# A lot is "Matched for you" from this score up. With the weights above, price,
# volume and distance alone reach 0.45; one of use or characteristics on top is
# needed, so a badge always means the fish itself fits, not just the logistics.
MATCH_THRESHOLD = 0.6


@dataclass(frozen=True)
class MatchReason:
    criterion: str
    met: bool
    detail: str
    value: str | None


@dataclass(frozen=True)
class MatchResult:
    score: float
    reasons: list[MatchReason]

    @property
    def matched(self) -> bool:
        return self.score >= MATCH_THRESHOLD


def fold_terms(values: list[str]) -> set[str]:
    return {item.casefold().strip() for item in values if item and item.strip()}


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold_words(values: list[str]) -> set[str]:
    """Every word across the values, casefolded.

    Used for names (search compares fish names by whole word). Preference
    matching needs more than this and uses `match_terms`.
    """
    words: set[str] = set()
    for value in values:
        if value:
            words.update(match.group().casefold() for match in _WORD.finditer(value))
    return words


# --- Controlled vocabulary -------------------------------------------------
#
# A buyer picks "fried"; a card says "Fried", "Deep-fried" or "suitable for
# frying"; an older card written in Indonesian says "Goreng". Exact word
# matching misses all but the first. Each group maps every form we expect to
# one canonical term. Forms are listed explicitly rather than derived, because
# a derived stem collides ("canned" -> "can", as in "can be grilled").
# Indonesian forms stay so snapshots published before the switch to English
# cards keep matching.
_CONCEPTS: dict[str, tuple[str, ...]] = {
    # How the fish is used or cooked.
    "fried": ("fry", "fried", "frying", "fries", "goreng", "digoreng", "gorengan"),
    "grilled": ("grill", "grilled", "grilling", "grills", "barbecue", "barbecued", "bbq",
                "broil", "broiled", "roast", "roasted", "roasting", "bakar", "dibakar"),
    "fillet": ("fillet", "fillets", "filleted", "filleting", "filet", "filets", "loin", "loins"),
    "smoked": ("smoke", "smoked", "smoking", "asap", "diasap", "pengasapan", "asapan"),
    "steamed": ("steam", "steamed", "steaming", "pepes", "kukus", "dikukus"),
    "boiled": ("boil", "boiled", "boiling", "poach", "poached", "poaching", "braise", "braised",
               "braising", "stew", "stewed", "pindang"),
    "ball": ("ball", "balls", "meatball", "meatballs", "bakso"),
    "surimi": ("surimi", "paste", "otak"),
    "curry": ("curry", "curries", "curried", "gulai"),
    "soup": ("soup", "soups", "broth", "sup"),
    "canned": ("canned", "canning", "cannery", "canneries", "tinned", "pengalengan", "kaleng"),
    "frozen": ("frozen", "freeze", "freezing", "beku"),
    "salted": ("salted", "salting", "cured", "asin"),
    "raw": ("raw", "sashimi", "sushi"),
    "cracker": ("cracker", "crackers", "kerupuk", "krupuk"),
    # Taste and texture.
    "savory": ("savory", "savoury", "umami", "gurih"),
    "firm": ("firm", "firmer", "dense", "compact", "padat"),
    "soft": ("soft", "softer", "tender", "lembut", "empuk"),
    "oily": ("oil", "oily", "oiliness", "fatty", "fat", "greasy", "berminyak", "minyak",
             "lemak", "berlemak"),
    "flesh": ("flesh", "meat", "daging"),
    "white": ("white", "whitish", "pale", "putih"),
    "mild": ("mild", "light", "neutral", "delicate", "ringan", "netral"),
    "sweet": ("sweet", "sweetish", "manis"),
    "flaky": ("flaky", "flake", "flakes"),
    "rich": ("rich", "kaya", "pekat"),
    # Who buys it (potential_buyer_segments, commercial_uses).
    "restaurant": ("restaurant", "restaurants", "restoran", "eatery", "eateries", "diner", "diners",
                   "stall", "stalls", "warung"),
    "hotel": ("hotel", "hotels", "hospitality", "horeca"),
    "catering": ("catering", "caterer", "caterers", "cater", "katering"),
    "retail": ("retail", "retailer", "retailers", "supermarket", "supermarkets", "grocery",
               "groceries", "hypermarket", "minimarket", "peritel", "swalayan"),
    "market": ("market", "markets", "trader", "traders", "fishmonger", "fishmongers", "vendor",
               "vendors", "pedagang", "pasar"),
    "processor": ("processor", "processors", "processing", "processed", "pengolah", "pengolahan",
                  "factory", "factories", "industry", "industrial", "manufacturer", "manufacturers"),
    "wholesale": ("wholesale", "wholesaler", "wholesalers", "distributor", "distributors",
                  "distribution", "supplier", "suppliers", "grosir"),
    "export": ("export", "exports", "exported", "exporting", "exporter", "exporters", "ekspor",
               "eksportir"),
}
_CANONICAL = {form: concept for concept, forms in _CONCEPTS.items() for form in (concept, *forms)}

# Dropped before matching. They carry no preference, and "too" or "slightly"
# sit between a negator and the word it negates.
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "from", "into", "as", "than",
    "is", "are", "be", "it", "its", "has", "have", "with", "or", "if", "when", "can", "this",
    "that", "which", "very", "too", "slightly", "quite", "somewhat", "rather", "fairly",
    "mildly", "bit", "little", "most", "more", "also", "well", "good", "suitable", "use",
    "used", "fish", "ikan", "yang", "agak", "sedikit", "terlalu", "sangat", "untuk",
})
# A negator switches the following words off until the clause ends:
# "not too oily" is not a claim that the fish is oily.
_NEGATORS = frozenset({
    "not", "no", "without", "never", "neither", "nor", "isn", "aren", "doesn", "don", "tidak",
    "bukan", "tanpa", "kurang",
})
# These modify one word only: "non-oily flesh" still asserts the flesh.
_WORD_NEGATORS = frozenset({"non", "less", "low"})
_CLAUSE_WORDS = frozenset({
    "but", "and", "yet", "though", "although", "while", "however", "tetapi", "namun", "dan",
})
_TOKEN = re.compile(r"[^\W\d_]+|[.,;:!?()/]", re.UNICODE)


def _stem(word: str) -> str:
    """A light English stemmer for words outside the vocabulary.

    Only strips regular inflection so "restaurants" and "restaurant", or
    "fibers" and "fiber", meet. Both sides go through it, so an odd stem
    ("whit") is harmless: it is compared only with itself.
    """
    stem = word
    if len(stem) > 4 and stem.endswith(("ies", "ied")):
        stem = stem[:-3] + "y"
    elif len(stem) > 5 and stem.endswith("ing"):
        stem = stem[:-3]
    elif len(stem) > 4 and stem.endswith("ed"):
        stem = stem[:-2]
    elif len(stem) > 4 and stem.endswith(("ches", "shes", "sses", "xes")):
        stem = stem[:-2]
    elif len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss"):
        stem = stem[:-1]
    if len(stem) > 3 and stem.endswith("e"):
        stem = stem[:-1]
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        stem = stem[:-1]
    return stem


def normalise_word(word: str) -> str:
    folded = word.casefold()
    return _CANONICAL.get(folded) or _CANONICAL.get(_stem(folded)) or _stem(folded)


def phrase_terms(text: str) -> frozenset[str]:
    """The asserted terms of one phrase, with negated words left out."""
    terms: set[str] = set()
    negated = False
    negate_next = False
    for token in _TOKEN.findall(text or ""):
        word = token.casefold()
        if not word[0].isalpha() or word in _CLAUSE_WORDS:
            negated = negate_next = False
            continue
        if word in _NEGATORS:
            negated = True
            continue
        if word in _WORD_NEGATORS:
            negate_next = True
            continue
        if word in _STOPWORDS:
            continue
        if not (negated or negate_next):
            terms.add(normalise_word(word))
        negate_next = False
    return frozenset(terms)


def match_terms(values: list[str]) -> set[str]:
    """Every asserted term across the values, normalised."""
    terms: set[str] = set()
    for value in values:
        if isinstance(value, str):
            terms |= phrase_terms(value)
    return terms


def matching_choices(wanted: list[str], available: set[str]) -> list[str]:
    """The buyer's choices the lot satisfies.

    A choice is met when every term in it is on the card: "white flesh" needs
    both words, where any one word would match every card that says "flesh".
    """
    return [
        choice
        for choice in wanted
        if (terms := phrase_terms(choice)) and terms <= available
    ]


# --- What a lot offers -----------------------------------------------------

def _snapshot_list(lot: LotRecord, key: str) -> list[str]:
    snapshot = lot.knowledge_snapshot or {}
    return [item for item in snapshot.get(key) or [] if isinstance(item, str)]


def lot_uses(lot: LotRecord) -> list[str]:
    """What the fish can be used for.

    Both fields, because a buyer's answer to "what do you cook or sell" can be
    either. PRD 8.3.5 reads "Suitable for grilling", a processing method, while
    `commercial_uses` holds segments like Restaurants, so reading only the
    latter scored a cooking method against a list of customer types.
    """
    return [*_snapshot_list(lot, "processing_methods"), *_snapshot_list(lot, "commercial_uses")]


def lot_characteristics(lot: LotRecord) -> list[str]:
    snapshot = lot.knowledge_snapshot or {}
    # `characteristics` is not a KnowledgeCard field; kept only for snapshots
    # written before the card contract settled.
    values = list(snapshot.get("characteristics") or [])
    for key in ("taste", "texture", "physical_characteristics"):
        raw = snapshot.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw)
    return values


def lot_segments(lot: LotRecord) -> list[str]:
    """Who the card says buys this fish."""
    return [*_snapshot_list(lot, "potential_buyer_segments"), *_snapshot_list(lot, "commercial_uses")]


# --- Business type ---------------------------------------------------------

# The form's business types, as the segment terms that count as "businesses
# like yours" on a card. A hotel kitchen buys what restaurants buy; a
# distributor sells on to markets and retailers.
BUSINESS_SEGMENTS: dict[str, frozenset[str]] = {
    "restaurant": frozenset({"restaurant"}),
    "hotel": frozenset({"hotel", "restaurant"}),
    "catering": frozenset({"catering"}),
    "supermarket": frozenset({"retail"}),
    "seafood_retailer": frozenset({"retail", "market"}),
    "processor": frozenset({"processor"}),
    "distributor": frozenset({"wholesale", "market", "retail"}),
    "exporter": frozenset({"export"}),
}
# Values the Indonesian form saved before the English one.
_LEGACY_BUSINESS = {
    "rumah_makan": "restaurant",
    "peritel_seafood": "seafood_retailer",
    "katering": "catering",
    "pengolah": "processor",
}
BUSINESS_LABELS = {
    "restaurant": "restaurants",
    "hotel": "hotels",
    "catering": "caterers",
    "supermarket": "supermarkets",
    "seafood_retailer": "seafood retailers",
    "processor": "processors",
    "distributor": "distributors",
    "exporter": "exporters",
}


def normalise_business_type(value: str) -> str:
    key = (value or "").strip().casefold().replace(" ", "_").replace("-", "_")
    return _LEGACY_BUSINESS.get(key, key)


def business_segment_terms(value: str) -> frozenset[str]:
    key = normalise_business_type(value)
    # A free-text type the form does not offer still gets a fair reading.
    return BUSINESS_SEGMENTS.get(key) or phrase_terms(value.replace("_", " "))


# --- Scoring ---------------------------------------------------------------

def _rupiah(value: Decimal) -> str:
    return f"Rp {value:,.0f}/kg".replace(",", ".")


def _kg(value: Decimal) -> str:
    number = f"{Decimal(value).normalize():f}"
    return f"{number} kg"


def match_lot(
    lot: LotRecord,
    prefs: BuyerPreferenceRecord,
    landing: LandingPointRecord | None,
    radius_km: float = DEFAULT_SERVICEABILITY_RADIUS_KM,
) -> MatchResult:
    uses_hit = matching_choices(prefs.intended_uses, match_terms(lot_uses(lot)))
    chars_hit = matching_choices(prefs.characteristics, match_terms(lot_characteristics(lot)))
    segments = business_segment_terms(prefs.business_type)
    business_met = bool(segments) and bool(segments & match_terms(lot_segments(lot)))
    price_met = prefs.max_price_per_kg is None or lot.starting_price_per_kg <= prefs.max_price_per_kg
    volume_met = prefs.min_quantity_kg is None or lot.quantity_kg >= prefs.min_quantity_kg
    if landing is None:
        distance_km = None
        distance_met = False
    else:
        distance_km = haversine_km(prefs.latitude, prefs.longitude, landing.latitude, landing.longitude)
        distance_met = within_serviceability(
            prefs.latitude, prefs.longitude, landing.latitude, landing.longitude, radius_km
        )

    flags = {
        "intended_use": bool(uses_hit),
        "characteristics": bool(chars_hit),
        "business_type": business_met,
        "price": price_met,
        "volume": volume_met,
        "distance": distance_met,
    }
    business_key = normalise_business_type(prefs.business_type)
    business_label = BUSINESS_LABELS.get(business_key, prefs.business_type.replace("_", " "))

    if not prefs.intended_uses:
        use_detail = "You did not choose an intended use"
    elif uses_hit:
        use_detail = "Suits how you use fish: " + ", ".join(uses_hit)
    else:
        use_detail = "Its card lists none of your uses"

    if not prefs.characteristics:
        char_detail = "You did not choose any characteristics"
    elif chars_hit:
        char_detail = "Has the qualities you want: " + ", ".join(chars_hit)
    else:
        char_detail = "Its taste and texture differ from what you asked for"

    if prefs.max_price_per_kg is None:
        price_detail = f"Starts at {_rupiah(lot.starting_price_per_kg)}, no price limit set"
    elif price_met:
        price_detail = f"Starts at {_rupiah(lot.starting_price_per_kg)}, within your limit"
    else:
        price_detail = (
            f"Starts at {_rupiah(lot.starting_price_per_kg)}, above your "
            f"{_rupiah(prefs.max_price_per_kg)} limit"
        )

    if prefs.min_quantity_kg is None:
        volume_detail = f"{_kg(lot.quantity_kg)} available, no minimum set"
    elif volume_met:
        volume_detail = f"{_kg(lot.quantity_kg)} available, meets your minimum"
    else:
        volume_detail = f"{_kg(lot.quantity_kg)} available, below your {_kg(prefs.min_quantity_kg)} minimum"

    if distance_km is None:
        distance_detail = "Landing point location unknown"
    elif distance_met:
        distance_detail = f"Landed {round(distance_km)} km from you"
    else:
        distance_detail = f"Landed {round(distance_km)} km from you, beyond the {round(radius_km)} km radius"

    details = {
        "intended_use": use_detail,
        "characteristics": char_detail,
        "business_type": (
            f"Its card names {business_label} among its buyers"
            if business_met
            else f"Its card does not name {business_label} among its buyers"
        ),
        "price": price_detail,
        "volume": volume_detail,
        "distance": distance_detail,
    }
    values = {
        "intended_use": ", ".join(prefs.intended_uses) or None,
        "characteristics": ", ".join(prefs.characteristics) or None,
        "business_type": business_label or None,
        "price": str(lot.starting_price_per_kg),
        "volume": _kg(lot.quantity_kg),
        "distance": None if distance_km is None else f"{round(distance_km)} km",
    }
    reasons = [
        MatchReason(criterion=name, met=flags[name], detail=details[name], value=values[name])
        for name in CRITERIA
    ]
    # Rounded so 0.25 + 0.20 + 0.15 compares equal to the 0.6 threshold
    # instead of falling a float epsilon either side of it.
    score = round(sum(WEIGHTS[name] for name in CRITERIA if flags[name]), 4)
    return MatchResult(score=score, reasons=reasons)


def recommend(
    lots: list[LotRecord],
    prefs: BuyerPreferenceRecord | None,
    landing_points: dict[str, LandingPointRecord],
    radius_km: float = DEFAULT_SERVICEABILITY_RADIUS_KM,
) -> list[tuple[LotRecord, MatchResult]]:
    """Every lot within reach of the buyer, best match first.

    Lots beyond the radius are left out rather than scored low: a fish that
    cannot be delivered is not a recommendation however well it fits.
    """
    if prefs is None:
        return []
    ranked = []
    for lot in lots:
        landing = landing_points.get(lot.landing_point_id)
        if landing is None or not within_serviceability(
            prefs.latitude, prefs.longitude, landing.latitude, landing.longitude, radius_km
        ):
            continue
        ranked.append((lot, match_lot(lot, prefs, landing, radius_km)))
    ranked.sort(key=lambda item: item[1].score, reverse=True)
    return ranked


def nearest_landing_point(
    latitude: float, longitude: float, landing_points: list[LandingPointRecord]
) -> tuple[LandingPointRecord, float] | None:
    best = None
    for point in landing_points:
        distance = haversine_km(latitude, longitude, point.latitude, point.longitude)
        if best is None or distance < best[1]:
            best = (point, distance)
    return best
