"""Combine the two cross-fitted fold models of one backbone into one run.

`xf_<model>_A` was trained without fold A and scored on fold A's field photos;
`xf_<model>_B` the other way round. The merged run `xf_<model>` holds:

- field: the union of both held-out folds, i.e. all 258 photos, each scored by
  the model that never trained on it, directly comparable with the baseline;
- every other slice: the rows of both fold models together, so a metric there
  is the average over the two models.

    python -m evaluation.cv.merge_xfit xf_vit_l xf_vit_b ...
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from evaluation.cv.common import RESULTS_DIR


def merge(run: str) -> None:
    parts = {fold: RESULTS_DIR / f"{run}_{fold}" for fold in ("A", "B")}
    frames, metas = [], {}
    for fold, folder in parts.items():
        meta = json.loads((folder / "run.json").read_text())
        if meta.get("heldout_fold") != fold:
            raise SystemExit(f"{folder.name}: heldout_fold is {meta.get('heldout_fold')!r}, expected {fold!r}")
        metas[fold] = meta
        frames.append(pd.read_csv(folder / "predictions.csv", dtype=str).assign(fold_model=fold))
    df = pd.concat(frames, ignore_index=True)
    field = df[df.slice == "field"]
    if field.item_id.duplicated().any():
        raise SystemExit("a field photo was scored by both fold models: folds overlap")

    out = RESULTS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns="fold_model").to_csv(out / "predictions.csv", index=False)
    a, b = metas["A"], metas["B"]
    meta = {
        **{k: a[k] for k in ("model_name", "abstain_threshold", "device", "screening_sha256",
                             "field_folds_sha256", "out_of_scope", "seed")},
        "run": run,
        "export": [a["export"], b["export"]],
        "temperature": (a["temperature"] + b["temperature"]) / 2,
        "temperatures": {"A": a["temperature"], "B": b["temperature"]},
        "cross_fit": "field = union of held-out folds (each photo scored by the model that did not train on it); "
                     "other slices = both fold models pooled",
        "field_items": int(len(field)),
        "corruptions": a.get("corruptions", True) and b.get("corruptions", True),
        "gate": {"A": a.get("gate"), "B": b.get("gate")} if a.get("gate") else None,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"{run}: {len(field)} field photos, {len(df)} rows -> {out}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        merge(name)
