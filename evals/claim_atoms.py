"""The unit of the real-claims grounding set: one atom of an expert claim.

An atom is a list item, or a sentence / semicolon clause of a text field. It is
an evaluation unit only. Iteration 2 measured grading cards atom by atom with the
shipped verifier and rejected it: recall fell from 0.87 to 0.57 at the same
precision, because a short item ("Dibekukan") scores a low cosine against a whole
chunk. The critic grades whole claims.
"""

import re

# A decimal ("4.3 kg") has no space after its point, so it never splits.
_ATOM_BREAK = re.compile(r";\s+|(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def claim_atoms(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in _ATOM_BREAK.split(str(value).strip()) if part.strip()]
