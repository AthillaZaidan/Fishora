"""Marketplace search: what a buyer types, against a lot's name and knowledge card.

Matching is on characters, so a buyer halfway through "bandeng" already sees
Bandeng, and "gurih" or "fillet" finds lots by what their card says about taste,
texture and use. When the query names a fish, the result also carries the other
lots the matched fish's card lists as similar or substitute species (and lots
whose own card lists the matched fish): "besides Bandeng, this similar fish is
on sale too". Those names come from the generated knowledge card, not from here.
"""

from dataclasses import dataclass

from apps.main_api.contracts import LotRecord
from apps.main_api.services.matching import fold_words, lot_characteristics, lot_uses

MIN_TERM_LENGTH = 2
# Every name and most card sentences contain it, so it would match everything.
_STOPWORDS = {"ikan"}


@dataclass(frozen=True)
class SearchResult:
    matches: list[LotRecord]
    similar: list[LotRecord]
    # The matched fish the similar lots are recommended for, as displayed names.
    similar_to: list[str]


def species_names(lot: LotRecord) -> list[str]:
    snapshot = lot.knowledge_snapshot or {}
    label = lot.species_id.removeprefix("species_").replace("_", " ")
    names = [label, snapshot.get("common_name"), snapshot.get("scientific_name")]
    return [name for name in names if isinstance(name, str) and name.strip()]


def _substitutes(lot: LotRecord) -> list[str]:
    snapshot = lot.knowledge_snapshot or {}
    return [item for item in snapshot.get("similar_or_substitute_species") or [] if isinstance(item, str)]


def _terms(query: str) -> list[str]:
    return [
        term
        for term in query.casefold().split()
        if len(term) >= MIN_TERM_LENGTH and term not in _STOPWORDS
    ]


def _contains_all(terms: list[str], texts: list[str]) -> bool:
    haystack = " ".join(texts).casefold()
    return all(term in haystack for term in terms)


def _names_any(names: list[str], mentions: list[str]) -> bool:
    """True when one of `mentions` names one of `names`, compared by whole words."""
    mentioned = [fold_words([mention]) for mention in mentions]
    return any(
        (words := fold_words([name])) and any(words <= other for other in mentioned)
        for name in names
    )


def search_lots(lots: list[LotRecord], query: str) -> SearchResult:
    terms = _terms(query)
    if not terms:
        return SearchResult(matches=list(lots), similar=[], similar_to=[])

    matches: list[LotRecord] = []
    by_name: list[LotRecord] = []
    for lot in lots:
        names = species_names(lot)
        if _contains_all(terms, names):
            matches.append(lot)
            by_name.append(lot)
        elif _contains_all(terms, [*names, *lot_characteristics(lot), *lot_uses(lot)]):
            matches.append(lot)

    matched_ids = {lot.id for lot in matches}
    matched_names = [name for lot in by_name for name in species_names(lot)]
    matched_substitutes = [item for lot in by_name for item in _substitutes(lot)]
    similar = [
        lot
        for lot in lots
        if lot.id not in matched_ids
        and by_name
        and (
            _names_any(species_names(lot), matched_substitutes)
            or _names_any(matched_names, _substitutes(lot))
        )
    ]

    displayed = []
    for lot in by_name:
        name = (lot.knowledge_snapshot or {}).get("common_name") or species_names(lot)[0].title()
        if name not in displayed:
            displayed.append(name)
    return SearchResult(matches=matches, similar=similar, similar_to=displayed if similar else [])
