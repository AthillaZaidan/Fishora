"""Fishora's reference notes per species, and filling a card's gaps from them.

A generated card leaves a field empty when no verified chunk supports it. The
reference notes are the short, hand-written description of each trained
species that the demo lots were seeded with. When a catch is uploaded, the
card keeps every field the sources support and takes only the empty ones from
these notes, and it says so in its limitations, because a filled-in field has
no citation behind it.
"""

# English, like the cards the generator writes, and worded so that every chip
# on the buyer preference form matches at least one card
# (evals/tests/unit/test_matching.py checks this). Dish names with no English
# equivalent keep the local name with a gloss. Similar species keep the
# taxonomy name first: search links similar fish by that name.
REFERENCE: dict[str, tuple] = {
    "tenggiri": ("Elongated body with faint vertical bars along its sides.", "Savory and not too oily.", "Firm with fine fibers and white flesh.",
                 ["Fillet", "Smoked", "Fish balls"], ["Restaurants", "Fish ball processing", "Catering"], ["Kembung (Indian mackerel)"], ["Fish ball processors", "Seafood restaurants"]),
    "tuna": ("Large cigar-shaped body with a prominent dorsal fin.", "Rich and full-flavoured.", "Firm and dense.",
             ["Fresh loins", "Frozen", "Canned"], ["Export", "Japanese restaurants", "Canneries"], ["Tenggiri (Spanish mackerel)"], ["Exporters", "Premium restaurants"]),
    "kembung": ("Small fish with a greenish back and silvery sides.", "Savory with a strong sea flavour.", "Soft and slightly oily.",
                ["Pindang (salt-boiled)", "Smoked", "Fried"], ["Wet markets", "Pindang processing", "Food stalls"], ["Gembolo"], ["Pindang processors", "Market traders"]),
    "bandeng": ("Silvery body with a deeply forked tail fin.", "Sweet and mild.", "Fine-textured with many small bones.",
                ["Presto (pressure-cooked)", "Otak-otak (fish cake)", "Smoked"], ["Presto processing", "Catering", "Souvenir food"], ["Nila (Nile tilapia)"], ["Presto processors", "Souvenir food producers"]),
    "gelama_bunga": ("Small silvery fish with a fairly large head.", "Mild and slightly sweet.", "Soft and falls apart easily.",
                     ["Fried", "Pindang (salt-boiled)", "Surimi"], ["Wet markets", "Surimi processing"], ["Gulamah (croaker)"], ["Surimi processors", "Market traders"]),
    "gembolo": ("Small silvery fish; the name covers several species depending on the region.", "Mild and savory.", "Soft.",
                ["Fried", "Pindang (salt-boiled)", "Fish crackers"], ["Wet markets", "Food stalls", "Cracker processing"], ["Kembung (Indian mackerel)"], ["Market traders", "Cracker processors"]),
    "gulamah": ("Silvery body with a slightly downturned mouth.", "Neutral and mild.", "Soft and moist.",
                ["Surimi", "Fish balls", "Pindang (salt-boiled)"], ["Surimi processing", "Fish ball makers"], ["Kuniran (goatfish)"], ["Surimi processors"]),
    "kuniran": ("Small reddish fish with a long yellow stripe.", "Mild and slightly sweet.", "Soft.",
                ["Deep-fried", "Fish crackers", "Surimi"], ["Cracker processing", "Wet markets"], ["Gulamah (croaker)"], ["Cracker processors", "Market traders"]),
    "mujair": ("Flat body with a long spiny dorsal fin.", "Mild with a slightly earthy taste.", "Firm and fibrous.",
               ["Fried", "Grilled", "Steamed in banana leaf (pepes)"], ["Wet markets", "Food stalls", "Catering"], ["Nila (Nile tilapia)"], ["Market traders", "Caterers"]),
    "senangin": ("Four long filaments below the head.", "Delicate and clean.", "Firm, and lifts easily off the bone.",
                 ["Fillet", "Curry (gulai)", "Grilled"], ["Restaurants", "Hotels"], ["Tenggiri (Spanish mackerel)"], ["Restaurants", "Hotels"]),
    "nila": ("Deep, flat body with dark vertical bars.", "Mild and clean.", "Firm and fibrous with white flesh.",
             ["Fillet", "Grilled", "Fried"], ["Restaurants", "Catering", "Wet markets"], ["Mujair (Mozambique tilapia)"], ["Restaurants", "Caterers"]),
}


# Card field -> position in a REFERENCE tuple.
_FIELDS = {
    "physical_characteristics": 0,
    "taste": 1,
    "texture": 2,
    "processing_methods": 3,
    "commercial_uses": 4,
    "similar_or_substitute_species": 5,
    "potential_buyer_segments": 6,
}

_FIELD_NAMES = {
    "physical_characteristics": "physical characteristics",
    "taste": "taste",
    "texture": "texture",
    "processing_methods": "processing methods",
    "commercial_uses": "commercial uses",
    "similar_or_substitute_species": "similar species",
    "potential_buyer_segments": "buyer segments",
}


def fill_from_reference(card: dict | None, species_id: str) -> dict | None:
    """The card with each empty field taken from the species' reference notes.

    Fields the sources support are never replaced. Returns the card unchanged
    when nothing was empty or the species has no notes.
    """
    if not card:
        return card
    entry = REFERENCE.get(species_id.removeprefix("species_"))
    if entry is None:
        return card
    filled = dict(card)
    names = []
    for field, index in _FIELDS.items():
        if not filled.get(field):
            value = entry[index]
            filled[field] = list(value) if isinstance(value, list) else value
            names.append(_FIELD_NAMES[field])
    if not names:
        return card
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    note = f"The {listed} come from Fishora's reference notes for this species, not from a cited source."
    filled["limitations"] = [*(filled.get("limitations") or []), note]
    return filled
