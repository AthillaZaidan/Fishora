"""Figures for the Evaluation Artifact, drawn from evaluation/cv/results only.

    python -m evaluation.cv.artifact.figures      # writes evaluation/cv/artifact/figures/*.png
"""

from __future__ import annotations

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import confusion_matrix

from evaluation.cv.common import CV_DIR, DATA_DIR, RESULTS_DIR, ROOT, load_split

OUT = CV_DIR / "artifact" / "figures"
TEAL, TEAL_D, CORAL, PEACH, GOLD, GRAY = "#2A9D8F", "#1D5C63", "#E76F51", "#F4A261", "#E9C46A", "#8D99AE"
INK = "#1F2A30"
MODELS = ["vit_l", "vit_b", "vit_s", "convnext_t", "efficientnet_b0", "mobilenetv3_l"]
LABEL = {"vit_l": "ViT-L", "vit_b": "ViT-B", "vit_s": "ViT-S", "convnext_t": "ConvNeXt-T",
         "efficientnet_b0": "EffNet-B0", "mobilenetv3_l": "MobileNetV3"}
FIELD_CLASSES = ["bandeng", "gelama_bunga", "kembung", "kuniran", "mujair", "nila", "senangin", "tenggiri", "tuna"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#B8C0C8", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def metrics(run: str) -> dict:
    return json.loads((RESULTS_DIR / run / "metrics.json").read_text())


def preds(run: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / run / "predictions.csv", dtype={"severity": str, "label": str, "group": str})
    df[["label", "group"]] = df[["label", "group"]].fillna("")
    return df


def latency() -> dict:
    return json.loads((RESULTS_DIR / "latency.json").read_text())["models"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------

def fig_dataset():
    m = pd.read_csv(DATA_DIR / "dataset" / "metadata" / "manifest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    counts = m.groupby(["normalized_label", "source_group"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    axes[0].barh(counts.index, counts.get("FISH_GRES", 0), color=TEAL, label="Fish-gres")
    axes[0].barh(counts.index, counts.get("ROBOFLOW", 0), left=counts.get("FISH_GRES", 0), color=CORAL, label="Roboflow")
    axes[0].set_title("Images per class and source")
    axes[0].set_xlabel("images")
    axes[0].legend(frameon=False, fontsize=8)
    bg = m.assign(red=m.background_category.str.startswith("red")).groupby("normalized_label").red.mean()
    bg = bg.loc[counts.index]
    colors = [CORAL if c in {"gembolo", "tenggiri", "tuna"} else TEAL for c in bg.index]
    axes[1].barh(bg.index, bg.values * 100, color=colors)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Share photographed on a red background")
    axes[1].set_xlabel("% of images")
    fig.tight_layout()
    save(fig, "dataset")


def fig_workflow():
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")
    boxes = [
        (1, 18, "Held-out test\n849 images", TEAL_D),
        (1, 3, "External photos\n258 field, 194 OOD", TEAL_D),
        (24, 10.5, "Slices\nclean, 7 corruptions x 3,\nbackground x 3,\nfield, OOD", TEAL),
        (49, 10.5, "cv_suite.py\nproduction predict path,\n11 probabilities stored", TEAL),
        (74, 18, "report.py\nmetrics + 95% CI\nfigures, findings", PEACH),
        (74, 3, "compare + latency\nPareto, McNemar", PEACH),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 21, 11, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color, ec="none"))
        ax.text(x + 10.5, y + 5.5, text, ha="center", va="center", color="white" if color != PEACH else INK,
                fontsize=8, weight="bold")
    for (x0, y0), (x1, y1) in [((22.5, 23.5), (24, 17)), ((22.5, 8.5), (24, 14)), ((45.5, 16), (49, 16)),
                               ((70.5, 17), (74, 23)), ((70.5, 15), (74, 9))]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1))
    ax.text(50, 30.5, "locked inputs: screening.csv + field_folds.csv (sha256 checked before every run)",
            ha="center", fontsize=8, color=TEAL_D, style="italic")
    save(fig, "workflow")


def fig_slices():
    """One Roboflow and one Fish-gres test image through the suite's own slice builders."""
    from evaluation.cv.cv_suite import background_plate, dilate, load_mask, paste_on, perturb, to_working

    samples = load_split("test")
    fill = (124, 116, 104)
    picks = [next(s for s in samples if s.source_group == "ROBOFLOW" and s.label == "tuna" and s.background == "red"),
             next(s for s in samples if s.source_group == "FISH_GRES" and s.label == "senangin")]
    plate_src = {s.source_group: s for s in samples if (s.source_group, s.background) in
                 {("ROBOFLOW", "red"), ("FISH_GRES", "gray")} and load_mask(s.image_id, (256, 256)) is not None}
    cols = ["clean", "blur r=4", "occlusion 50%", "background removed", "fish erased", "background swapped"]
    fig, axes = plt.subplots(2, 6, figsize=(10, 3.7))
    for r, s in enumerate(picks):
        img = to_working(Image.open(s.path), 256)
        mask = load_mask(s.image_id, img.size)
        other = "FISH_GRES" if s.source_group == "ROBOFLOW" else "ROBOFLOW"
        o = plate_src[other]
        oimg = to_working(Image.open(o.path), 256)
        plate = background_plate(oimg, load_mask(o.image_id, oimg.size))
        pad = Image.new("RGB", img.size, fill)
        tiles = [img, perturb(img, "blur", 4.0, fill, mask, random.Random(0)),
                 perturb(img, "occlusion", 0.5, fill, mask, random.Random(0)),
                 Image.composite(img, pad, mask), Image.composite(pad, img, dilate(mask, 4)),
                 paste_on(img, mask, plate)]
        for c, t in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(t)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
        axes[r, 0].set_ylabel(f"{s.label}\n({s.source_group.title().replace('_', '-')})", fontsize=8)
    fig.tight_layout()
    save(fig, "slices")


def fig_baseline_slices():
    m = metrics("lp_vit_l")
    items = [("clean", m["slices"]["clean"]["accuracy"]),
             ("dark x0.25", m["corruption_curves"]["dark"]["0.25"]["accuracy"]),
             ("rotate 270", m["corruption_curves"]["rotate"]["270"]["accuracy"]),
             ("occlusion 50%", m["corruption_curves"]["occlusion"]["0.5"]["accuracy"]),
             ("low-res 64px", m["corruption_curves"]["lowres"]["64"]["accuracy"]),
             ("JPEG q=8", m["corruption_curves"]["jpeg"]["8"]["accuracy"]),
             ("blur r=4", m["corruption_curves"]["blur"]["4.0"]["accuracy"]),
             ("bg swapped", m["slices"]["bg_swap"]["accuracy"]),
             ("field photos", m["slices"]["field"]["accuracy"])]
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    names, vals = zip(*items)
    colors = [CORAL if v < 0.9 else (PEACH if v < 0.97 else TEAL) for v in vals]
    ax.bar(names, np.array(vals) * 100, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1, f"{v * 100:.1f}", ha="center", fontsize=8, weight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Baseline (ViT-L, MVP recipe): accuracy per slice, worst severity shown")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, "baseline_slices")


def fig_corruption_curves():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharey=True)
    for ax, (kind, xs) in zip(axes, [("blur", ["1.0", "2.0", "4.0"]), ("jpeg", ["50", "20", "8"]),
                                     ("lowres", ["128", "96", "64"])]):
        for model, color in [("lp_vit_l", TEAL_D), ("lp_vit_b", CORAL)]:
            curve = metrics(model)["corruption_curves"][kind]
            ax.plot(["clean", *xs], [100, *[curve[x]["accuracy"] * 100 for x in xs]], marker="o", color=color,
                    label=LABEL[model[3:]])
        ax.set_title({"blur": "Gaussian blur radius", "jpeg": "JPEG quality", "lowres": "longest side (px)"}[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "corruption_curves")


def fig_confusions():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (run, title) in zip(axes, [("lp_vit_l", "Baseline ViT-L"), ("lp_vit_b", "ViT-B before"),
                                       ("xf_vit_b", "ViT-B after field data")]):
        d = preds(run)
        d = d[d.slice == "field"]
        labels = FIELD_CLASSES + ["gembolo", "gulamah"]
        cm = confusion_matrix(d.label, d.pred, labels=labels, normalize="true")[: len(FIELD_CLASSES)]
        ax.imshow(cm, cmap="BuGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=70, fontsize=6.5)
        ax.set_yticks(range(len(FIELD_CLASSES)), FIELD_CLASSES if ax is axes[0] else [], fontsize=6.5)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.1:
                    ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                            color="white" if cm[i, j] > 0.55 else INK)
        acc = (d.pred == d.label).mean()
        ax.set_title(f"{title}\nfield accuracy {acc * 100:.1f}%", fontsize=9)
        ax.set_xlabel("predicted", fontsize=7)
    axes[0].set_ylabel("true", fontsize=7)
    fig.tight_layout()
    save(fig, "field_confusions")


def fig_before_after():
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    x = np.arange(len(MODELS))
    for k, (prefix, color, name) in enumerate([("lp", GRAY, "before (dataset only)"), ("xf", TEAL, "after (+ field photos, cross-fit)")]):
        accs, lo, hi = [], [], []
        for m in MODELS:
            f = metrics(f"{prefix}_{m}")["slices"]["field"]
            accs.append(f["accuracy"] * 100)
            lo.append((f["accuracy"] - f["accuracy_ci"][0]) * 100)
            hi.append((f["accuracy_ci"][1] - f["accuracy"]) * 100)
        colors = [CORAL if (m == "vit_b" and prefix == "xf") else color for m in MODELS]
        ax.bar(x + (k - 0.5) * 0.38, accs, 0.38, color=colors, yerr=[lo, hi], capsize=2,
               error_kw={"lw": 0.8, "ecolor": INK}, label=name)
        for i, a in enumerate(accs):
            ax.text(x[i] + (k - 0.5) * 0.38, a + hi[i] + 1.5, f"{a:.0f}", ha="center", fontsize=7.5, weight="bold")
    ax.axhline(metrics("lp_vit_l")["slices"]["field"]["accuracy"] * 100, color=TEAL_D, ls="--", lw=0.9)
    ax.text(2.55, metrics("lp_vit_l")["slices"]["field"]["accuracy"] * 100 + 1.5, "MVP baseline 81.8%",
            ha="left", fontsize=7.5, color=TEAL_D)
    ax.set_xticks(x, [LABEL[m] for m in MODELS])
    ax.set_ylim(0, 118)
    ax.set_ylabel("field accuracy (%)")
    ax.set_title("Field photos (258, same photos before and after), 95% bootstrap CI")
    ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)
    save(fig, "before_after")


FAMILY = {"vit_l": "DINOv3 ViT", "vit_b": "DINOv3 ViT", "vit_s": "DINOv3 ViT", "convnext_t": "DINOv3 ConvNeXt",
          "efficientnet_b0": "ImageNet CNN", "mobilenetv3_l": "ImageNet CNN"}
FAMILY_COLOR = {"DINOv3 ViT": TEAL, "DINOv3 ConvNeXt": GOLD, "ImageNet CNN": GRAY}
LATENCY_BUDGET_MS = 300


def pareto_front(points):
    """Points not beaten on both axes (lower latency and higher accuracy)."""
    front = []
    for x, y, name in sorted(points):
        if not front or y > front[-1][1]:
            front.append((x, y, name))
    return front


def fig_pareto():
    """Field accuracy against CPU latency, styled after cost-vs-intelligence charts."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    lat = latency()
    baseline = metrics("lp_vit_l")["slices"]["field"]["accuracy"] * 100
    after = [(lat[f"lp_{m}"]["p50_ms"], metrics(f"xf_{m}")["slices"]["field"]["accuracy"] * 100, m) for m in MODELS]
    before = [(lat[f"lp_{m}"]["p50_ms"], metrics(f"lp_{m}")["slices"]["field"]["accuracy"] * 100, m) for m in MODELS]
    xmin, xmax, ymin, ymax = 25, 1500, 0, 100

    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    # Sweet spot: at least as accurate as the MVP baseline, fast enough to feel instant.
    ax.add_patch(plt.Rectangle((xmin, baseline), LATENCY_BUDGET_MS - xmin, ymax - baseline,
                               color="#DDF5E3", zorder=0, lw=0))
    ax.add_patch(plt.Rectangle((LATENCY_BUDGET_MS, ymin), xmax - LATENCY_BUDGET_MS, ymax - ymin,
                               color="#F2F3F5", zorder=0, lw=0))
    ax.text(xmin * 1.08, baseline + 1.5, "sweet spot: beats the MVP baseline\nwithin a 300 ms budget",
            fontsize=7.5, color="#2E7D4F", va="bottom", style="italic")
    ax.text(xmax / 1.05, ymin + 2, "over the latency budget", fontsize=7.5, color=GRAY, ha="right", style="italic")

    front = pareto_front(after)
    fx = [xmin] + [p[0] for p in front] + [xmax]
    fy = [front[0][1] - 4] + [p[1] for p in front] + [front[-1][1]]
    ax.plot(fx, fy, ls=(0, (1, 2.2)), color=INK, lw=1.6, zorder=2)

    for (xb, yb, m), (xa, ya, _) in zip(before, after):
        ax.scatter([xb], [yb], s=46, facecolor="white", edgecolor=FAMILY_COLOR[FAMILY[m]], lw=1.2, zorder=3, alpha=0.8)
        ax.plot([xb, xa], [yb, ya], color=FAMILY_COLOR[FAMILY[m]], lw=0.8, alpha=0.45, zorder=2)
    for xa, ya, m in after:
        star = m == "vit_b"
        ax.scatter([xa], [ya], s=260 if star else 90, marker="*" if star else "o",
                   color=CORAL if star else FAMILY_COLOR[FAMILY[m]], edgecolor="white", lw=0.8, zorder=4)
        offset = {"vit_l": (-10, -13), "vit_b": (-14, 14), "vit_s": (-9, 5), "convnext_t": (9, -11),
                  "efficientnet_b0": (9, 3), "mobilenetv3_l": (-6, -20)}[m]
        ax.annotate(f"{LABEL[m]}" + ("  (main model)" if star else "") + f"\n{ya:.1f}%",
                    (xa, ya), xytext=offset, textcoords="offset points", fontsize=8.5 if star else 8,
                    weight="bold" if star else "normal", color=CORAL if star else INK,
                    ha="center" if m == "mobilenetv3_l" else ("right" if offset[0] < 0 else "left"),
                    va="center", zorder=5,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.axhline(baseline, color=TEAL_D, lw=0.8, ls="--", zorder=1)
    ax.text(xmin * 1.08, baseline - 1.2, f"MVP baseline {baseline:.1f}%", fontsize=7.5, color=TEAL_D, va="top")

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ticks = [30, 50, 100, 200, 300, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
    ax.set_xlabel("CPU latency per photo (p50, batch 1, log scale)")
    ax.set_ylabel("Field accuracy (%)")
    ax.grid(axis="y", alpha=0.25)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=f) for f, c in FAMILY_COLOR.items()]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=7,
                       label="before field data"),
                Line2D([], [], ls=(0, (1, 2.2)), color=INK, lw=1.6, label="Pareto frontier (after)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8,
              handletextpad=0.3, columnspacing=1.2)
    save(fig, "pareto")


def fig_ood():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, (run, title) in zip(axes, [("lp_vit_l", "Baseline ViT-L"), ("xf_vit_b", "ViT-B after field data")]):
        d = preds(run)
        for s, color, name in [("field", TEAL, "in-scope fish (field)"), ("ood_unknown_fish", CORAL, "unknown species"),
                               ("ood_nonfish", GOLD, "non-fish")]:
            vals = d[d.slice == s].conf
            ax.hist(vals, bins=bins, weights=np.full(len(vals), 100 / len(vals)), histtype="stepfilled",
                    alpha=0.45, color=color, label=name, log=True)
        m = metrics(run)["ood"]
        ax.set_title(f"{title}\nAUROC unknown {m['ood_unknown_fish']['auroc']:.2f} | non-fish {m['ood_nonfish']['auroc']:.2f}",
                     fontsize=9)
        ax.set_xlabel("top-1 confidence")
    axes[0].set_ylabel("% of the set (log scale)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "ood")


def fig_per_class():
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    x = np.arange(len(FIELD_CLASSES))
    for k, (run, color, name) in enumerate([("lp_vit_l", GRAY, "baseline ViT-L"), ("lp_vit_b", PEACH, "ViT-B before"),
                                            ("xf_vit_b", CORAL, "ViT-B after")]):
        pc = metrics(run)["per_class"]["field"]
        vals = [pc[c]["recall"] * 100 for c in FIELD_CLASSES]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=color, label=name)
    ax.set_xticks(x, FIELD_CLASSES, rotation=25, ha="right")
    ax.set_ylabel("field recall (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    ax.set_title("Field recall per species")
    save(fig, "per_class")


def fig_background():
    fig, ax = plt.subplots(figsize=(9.5, 2.7))
    x = np.arange(len(MODELS))
    acc = [metrics(f"lp_{m}")["background"]["bg_swap"]["accuracy"] * 100 for m in MODELS]
    flip = [metrics(f"lp_{m}")["background"]["bg_swap"]["flip_rate"] * 100 for m in MODELS]
    ax.bar(x - 0.2, acc, 0.4, color=TEAL, label="accuracy after swapping background")
    ax.bar(x + 0.2, flip, 0.4, color=CORAL, label="prediction follows the new background")
    for i in range(len(MODELS)):
        ax.text(x[i] - 0.2, acc[i] + 1.5, f"{acc[i]:.1f}", ha="center", fontsize=7.5)
        ax.text(x[i] + 0.2, flip[i] + 1.5, f"{flip[i]:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [LABEL[m] for m in MODELS])
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Background swap test: fish pasted onto the other source's background")
    save(fig, "background")


UNKNOWN_ORDER = ["lele", "kerapu", "gurame", "patin", "kakap", "bawal", "cakalang"]


def fig_gates():
    df = preds("gated_vit_b")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    inputs = [("field", "in-scope\nfield fish"), ("ood_unknown_fish", "unknown\nspecies"),
              ("ood_nonfish", "non-fish"), ("ood_synthetic", "synthetic")]
    x = np.arange(len(inputs))
    after = [df[df.slice == s].gate.eq("pass").mean() * 100 for s, _ in inputs]
    a.bar(x - 0.2, [100] * len(inputs), 0.4, color=GRAY, label="before: threshold 0.0 accepts all")
    a.bar(x + 0.2, after, 0.4, color=[TEAL] + [CORAL] * 3, label="after: two gates")
    for i, v in enumerate(after):
        a.text(x[i] + 0.2, v + 2, f"{v:.1f}", ha="center", fontsize=8)
    a.set_xticks(x, [n for _, n in inputs], fontsize=8)
    a.set_ylim(0, 132)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_ylabel("share accepted (%)")
    a.set_title("Accepted by the pipeline")
    a.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=1)

    u = df[df.slice == "ood_unknown_fish"]
    shares = u.groupby("group").gate.value_counts(normalize=True).unstack(fill_value=0).reindex(UNKNOWN_ORDER) * 100
    bottom = np.zeros(len(UNKNOWN_ORDER))
    for col, color, name in [("not_fish", GOLD, "rejected: not a fish"),
                             ("unknown_species", TEAL, "rejected: unknown species"),
                             ("pass", CORAL, "accepted (leak)")]:
        vals = shares.get(col, pd.Series(0, index=UNKNOWN_ORDER)).to_numpy()
        b.bar(UNKNOWN_ORDER, vals, bottom=bottom, color=color, label=name)
        bottom += vals
    b.set_ylim(0, 100)
    b.set_ylabel("% of photos")
    b.set_title("Unknown species by taxon (cross-fitted, n=178)")
    b.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "gates")


def fig_final_corruptions():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharey=True)
    for ax, (kind, xs) in zip(axes, [("blur", ["1.0", "2.0", "4.0"]), ("jpeg", ["50", "20", "8"]),
                                     ("lowres", ["128", "96", "64"])]):
        for model, color, name in [("lp_vit_l", GRAY, "MVP baseline (ViT-L)"), ("lp_vit_b", PEACH, "ViT-B, dataset only"),
                                   ("final_vit_b", TEAL_D, "final ViT-B (+ field)")]:
            curve = metrics(model)["corruption_curves"][kind]
            ax.plot(["clean", *xs], [100, *[curve[x]["accuracy"] * 100 for x in xs]], marker="o", color=color,
                    label=name)
        ax.set_title({"blur": "Gaussian blur radius", "jpeg": "JPEG quality", "lowres": "longest side (px)"}[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "final_corruptions")


def main():
    for fn in [fig_dataset, fig_workflow, fig_slices, fig_baseline_slices, fig_corruption_curves, fig_confusions,
               fig_before_after, fig_pareto, fig_ood, fig_per_class, fig_background, fig_gates,
               fig_final_corruptions]:
        fn()
        print("ok", fn.__name__)


if __name__ == "__main__":
    main()
