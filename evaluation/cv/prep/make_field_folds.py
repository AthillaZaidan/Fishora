"""Split the kept field photos into two folds for 2-fold cross-fitting.

Each field photo must be scored by a model that never trained on it, and no
photographer may appear on both sides (one person's photos share phone,
background and framing). Observers are therefore assigned whole to fold A or
B, greedily, largest first, to whichever fold keeps the per-class counts
closest to half. The observer is the name in the CC attribution; CC0 photos
without a name get a group of their own.

The result is written and locked before any model is trained on it:
evaluation/cv/protocol/field_folds.csv + evaluation/cv/protocol/field_folds.lock.

    python -m evaluation.cv.prep.make_field_folds
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from evaluation.cv.common import DATA_DIR, PROTOCOL_DIR

OUT = PROTOCOL_DIR / "field_folds.csv"
LOCK = PROTOCOL_DIR / "field_folds.lock"


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"{LOCK} exists: folds are locked, delete both files deliberately to redo")
    screening = pd.read_csv(PROTOCOL_DIR / "screening.csv")
    attribution = pd.read_csv(DATA_DIR / "attribution.csv")
    field = screening[(screening.set == "field") & (screening.decision == "keep")].merge(
        attribution[["file", "author"]], on="file"
    )
    name = field.author.str.replace(r"^\(c\)\s*", "", regex=True).str.split(",").str[0].str.strip()
    anonymous = name.eq("no rights reserved") | name.eq("")
    field["observer"] = name.where(~anonymous, "anon:" + field.file)

    classes = sorted(field.label.unique())
    counts = {"A": pd.Series(0.0, index=classes), "B": pd.Series(0.0, index=classes)}
    groups = field.groupby("observer").label.value_counts().unstack(fill_value=0).reindex(columns=classes, fill_value=0)
    order = groups.sum(axis=1).sort_values(ascending=False, kind="stable").index
    assign = {}
    for obs in order:
        g = groups.loc[obs]
        # Squared per-class gap between the two folds after the assignment.
        cost = {
            "A": float(((counts["A"] + g - counts["B"]) ** 2).sum()),
            "B": float(((counts["B"] + g - counts["A"]) ** 2).sum()),
        }
        fold = min(cost, key=lambda f: (cost[f], counts[f].sum()))
        counts[fold] += g
        assign[obs] = fold
    field["fold"] = field.observer.map(assign)

    assert not set(field[field.fold == "A"].observer) & set(field[field.fold == "B"].observer)
    field[["file", "label", "observer", "fold"]].sort_values("file").to_csv(OUT, index=False)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    LOCK.write_text(
        f"field_folds.csv sha256={digest}\n"
        f"locked_at={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        "note=locked before any model was trained on field photos; do not edit after results exist\n"
    )
    table = field.pivot_table(index="label", columns="fold", values="file", aggfunc="count", fill_value=0)
    print(table.assign(total=table.sum(axis=1)).to_string())
    print(f"observers A/B: {field[field.fold == 'A'].observer.nunique()}/{field[field.fold == 'B'].observer.nunique()}")
    print(f"-> {OUT} (sha256 {digest[:12]}...)")


if __name__ == "__main__":
    main()
