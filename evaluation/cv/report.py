"""Metrics, figures and a findings draft from evaluation/cv/results/<run>/predictions.csv.

    python -m evaluation.cv.report baseline_vit_l_prod
    python -m evaluation.cv.report --compare baseline_vit_l_prod lp_vit_b lp_convnext_t

Single-run output (in the run folder): metrics.json, summary.md, figures/*.png.
Compare output: evaluation/cv/results/compare/<runs>.md and a key-metric table.

Metric choices
--------------
accuracy / macro-F1   labelled slices; macro-F1 because classes are imbalanced
wrong_confident       share of predictions that are wrong with confidence >= 0.9:
                      the failure the human-verification gate cannot see
ECE (15 bins)         calibration of the confidence the gate relies on
shortcut_rate         fish_erased only: share still predicted as the true class
                      with the fish removed (chance = 1/11)
flip_rate             bg_swap only: share moved to a class of the *new*
                      background's source family
OOD AUROC / FPR@95    max-softmax confidence separating in-scope fish (clean +
                      field) from each OOD set; FPR@95 = OOD share accepted when
                      the threshold keeps 95% of in-scope fish
95% CI                percentile bootstrap, 1000 resamples
gates (gated exports) share of each slice passed / rejected as not a fish /
                      rejected as an unknown species, and for labelled slices
                      the errors that still reach the operator after the gates
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from evaluation.cv.common import RESULTS_DIR

CONFIDENT = 0.9
ECE_BINS = 15
BOOT = 1000
ROBOFLOW_CLASSES = {"gembolo", "tenggiri", "tuna"}
LABELLED = ["clean", "dark", "bright", "blur", "jpeg", "lowres", "rotate", "occlusion",
            "bg_removed", "bg_swap", "field"]
OOD = ["ood_unknown_fish", "ood_nonfish", "ood_synthetic"]
CORRUPTIONS = ["dark", "bright", "blur", "jpeg", "lowres", "rotate", "occlusion"]


def load(run: str) -> tuple[pd.DataFrame, list[str], dict]:
    folder = RESULTS_DIR / run
    df = pd.read_csv(folder / "predictions.csv", dtype={"severity": str, "label": str, "group": str})
    df[["label", "group"]] = df[["label", "group"]].fillna("")
    if "gate" in df.columns:
        df["gate"] = df["gate"].fillna("pass")
    classes = [c[2:] for c in df.columns if c.startswith("p_")]
    meta = json.loads((folder / "run.json").read_text())
    return df, classes, meta


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    edges = np.linspace(0, 1, ECE_BINS + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def boot_ci(y: np.ndarray, p: np.ndarray, fn, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [fn(y[idx], p[idx]) for idx in (rng.integers(0, n, n) for _ in range(BOOT))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def acc_fn(y, p):
    return float((y == p).mean())


def f1_fn(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def labelled_metrics(d: pd.DataFrame, ci: bool) -> dict:
    y, p, conf = d.label.to_numpy(), d.pred.to_numpy(), d.conf.to_numpy()
    correct = y == p
    out = {
        "n": int(len(d)),
        "accuracy": acc_fn(y, p),
        "macro_f1": f1_fn(y, p),
        "mean_conf": float(conf.mean()),
        "wrong_confident": float(((~correct) & (conf >= CONFIDENT)).mean()),
        "ece": ece(conf, correct),
    }
    if ci:
        out["accuracy_ci"] = boot_ci(y, p, acc_fn)
        out["macro_f1_ci"] = boot_ci(y, p, f1_fn)
    return out


def fpr_at_tpr(id_conf: np.ndarray, ood_conf: np.ndarray, tpr: float = 0.95) -> tuple[float, float]:
    threshold = float(np.quantile(id_conf, 1 - tpr))
    return float((ood_conf >= threshold).mean()), threshold


def analyse(df: pd.DataFrame, classes: list[str], meta: dict) -> dict:
    m: dict = {"run": meta["run"], "model_name": meta["model_name"], "slices": {}, "per_class": {},
               "corruption_curves": {}, "background": {}, "ood": {}}

    for s in LABELLED:
        d = df[df.slice == s]
        if len(d):
            m["slices"][s] = labelled_metrics(d, ci=s in {"clean", "field", "bg_swap", "bg_removed"})

    for s in ["clean", "field"]:
        d = df[df.slice == s]
        if len(d):
            m["per_class"][s] = {
                c: {
                    "n": int((d.label == c).sum()),
                    "recall": float((d[d.label == c].pred == c).mean()) if (d.label == c).any() else None,
                    "f1": float(f1_score(d.label, d.pred, labels=[c], average="macro", zero_division=0)),
                    "top_confusion": d[(d.label == c) & (d.pred != c)].pred.value_counts().head(2).to_dict(),
                }
                for c in classes if (d.label == c).any()
            }

    for kind in CORRUPTIONS:
        d = df[df.slice == kind]
        m["corruption_curves"][kind] = {
            sev: {"accuracy": float((g.pred == g.label).mean()), "mean_conf": float(g.conf.mean())}
            for sev, g in d.groupby("severity", sort=False)
        }

    fe = df[df.slice == "fish_erased"]
    if len(fe):
        fe = fe.assign(family=fe.group.str.split("|").str[0])
        m["background"]["fish_erased"] = {
            "n": int(len(fe)),
            "shortcut_rate": float((fe.pred == fe.label).mean()),
            "chance": 1 / len(classes),
            "mean_conf": float(fe.conf.mean()),
            "confident_share": float((fe.conf >= CONFIDENT).mean()),
            "by_family": {
                fam: {"n": int(len(g)), "shortcut_rate": float((g.pred == g.label).mean()),
                      "mean_conf": float(g.conf.mean())}
                for fam, g in fe.groupby("family")
            },
            "by_class": {c: float((g.pred == g.label).mean()) for c, g in fe.groupby("label")},
        }
    bs = df[df.slice == "bg_swap"]
    if len(bs):
        target = np.where(bs.severity == "ROBOFLOW", 1, 0)
        pred_rf = bs.pred.isin(ROBOFLOW_CLASSES).to_numpy()
        flipped = (pred_rf == target.astype(bool)) & (bs.pred != bs.label).to_numpy()
        bs = bs.assign(flipped=flipped, source=np.where(bs.severity == "ROBOFLOW", "FISH_GRES", "ROBOFLOW"))
        m["background"]["bg_swap"] = {
            "n": int(len(bs)),
            "accuracy": float((bs.pred == bs.label).mean()),
            "flip_rate": float(flipped.mean()),
            "by_source_family": {
                fam: {"n": int(len(g)), "accuracy": float((g.pred == g.label).mean()),
                      "flip_rate": float(g.flipped.mean()),
                      "top_predictions": g.pred.value_counts().head(3).to_dict()}
                for fam, g in bs.groupby("source")
            },
            "by_class": {c: float((g.pred == g.label).mean()) for c, g in bs.groupby("label")},
        }

    id_conf = df[df.slice.isin(["clean", "field"])].conf.to_numpy()
    threshold = meta.get("abstain_threshold") or 0.0
    for s in OOD:
        d = df[df.slice == s]
        if not len(d):
            continue
        o = d.conf.to_numpy()
        y = np.r_[np.ones(len(id_conf)), np.zeros(len(o))]
        fpr, thr95 = fpr_at_tpr(id_conf, o)
        m["ood"][s] = {
            "n": int(len(d)),
            "mean_conf": float(o.mean()),
            "confident_share": float((o >= CONFIDENT).mean()),
            "accepted_at_production_threshold": float((o >= threshold).mean()),
            "auroc": float(roc_auc_score(y, np.r_[id_conf, o])),
            "fpr_at_95_tpr": fpr,
            "threshold_at_95_tpr": thr95,
            "top_predictions": d.pred.value_counts().head(3).to_dict(),
            "by_group": {g: float(x.conf.mean()) for g, x in d.groupby("group")},
        }
    m["production_threshold"] = threshold
    if "gate" in df.columns:
        m["gates"] = gate_metrics(df)
    return m


def gate_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for s, d in df.groupby("slice", sort=False):
        passed = d.gate.eq("pass")
        row = {
            "n": int(len(d)),
            "pass": float(passed.mean()),
            "rejected_not_fish": float(d.gate.eq("not_fish").mean()),
            "rejected_unknown_species": float(d.gate.eq("unknown_species").mean()),
        }
        if s in LABELLED:
            wrong = (d.pred != d.label).to_numpy()
            conf = d.conf.to_numpy() >= CONFIDENT
            p = passed.to_numpy()
            row.update({
                "accuracy_on_passed": float((~wrong[p]).mean()) if p.any() else None,
                "wrong_reaching_operator": float((wrong & p).mean()),
                "wrong_confident_reaching_operator": float((wrong & conf & p).mean()),
                "wrong_rejected_share": float((wrong & ~p).sum() / wrong.sum()) if wrong.any() else None,
            })
        out[s] = row
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_confusions(df, classes, out):
    slices = [s for s in ["clean", "field", "bg_swap"] if (df.slice == s).any()]
    fig, axes = plt.subplots(1, len(slices), figsize=(6.2 * len(slices), 5.6))
    for ax, s in zip(np.atleast_1d(axes), slices):
        d = df[df.slice == s]
        cm = confusion_matrix(d.label, d.pred, labels=classes, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(f"{s} (n={len(d)})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_corruptions(m, out):
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(2.6 * len(CORRUPTIONS), 2.8), sharey=True)
    clean = m["slices"].get("clean", {}).get("accuracy", np.nan)
    for ax, kind in zip(axes, CORRUPTIONS):
        curve = m["corruption_curves"].get(kind, {})
        xs = ["clean", *curve.keys()]
        ax.plot(range(len(xs)), [clean, *[v["accuracy"] for v in curve.values()]], marker="o", label="accuracy")
        ax.plot(range(len(xs)), [np.nan, *[v["mean_conf"] for v in curve.values()]], marker="x",
                linestyle="--", label="mean conf")
        ax.set_xticks(range(len(xs)), xs, fontsize=7)
        ax.set_title(kind, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_confidence(df, out):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, 1, 26)
    for s, colour in [("clean", "tab:blue"), ("field", "tab:green"), ("ood_unknown_fish", "tab:orange"),
                      ("ood_nonfish", "tab:red"), ("ood_synthetic", "tab:purple")]:
        d = df[df.slice == s]
        if len(d):
            ax.hist(d.conf, bins=bins, histtype="step", linewidth=1.6, density=True, label=f"{s} (n={len(d)})", color=colour)
    ax.set_xlabel("max softmax confidence")
    ax.set_ylabel("density")
    ax.set_title("Confidence: in-scope fish vs out-of-distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_reliability(df, out):
    fig, ax = plt.subplots(figsize=(4, 4))
    edges = np.linspace(0, 1, ECE_BINS + 1)
    for s in ["clean", "field", "blur", "dark"]:
        d = df[df.slice == s]
        if not len(d):
            continue
        conf, correct = d.conf.to_numpy(), (d.pred == d.label).to_numpy()
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", label=s)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def summary_md(m: dict, meta: dict) -> str:
    lines = [f"# Evaluation summary: `{m['run']}`", "",
             f"Model `{m['model_name']}`, temperature {meta['temperature']:.4f}, "
             f"production abstain threshold {m['production_threshold']}.", ""]
    if meta.get("out_of_scope"):
        lines += ["Out of scope (not scored): " + "; ".join(f"`{k}`: {v}" for k, v in meta["out_of_scope"].items()), ""]
    lines += ["## Labelled slices", "",
              "| slice | n | accuracy | macro-F1 | mean conf | wrong & conf>=0.9 | ECE |",
              "|---|---|---|---|---|---|---|"]
    for s, v in m["slices"].items():
        acc = pct(v["accuracy"]) + (f" [{pct(v['accuracy_ci'][0])}, {pct(v['accuracy_ci'][1])}]" if "accuracy_ci" in v else "")
        lines.append(f"| {s} | {v['n']} | {acc} | {v['macro_f1']:.3f} | {v['mean_conf']:.3f} | "
                     f"{pct(v['wrong_confident'])} | {v['ece']:.3f} |")
    bg = m["background"]
    if bg:
        lines += ["", "## Background tests", ""]
        if "fish_erased" in bg:
            fe = bg["fish_erased"]
            lines.append(f"- **fish_erased** (n={fe['n']}): still predicted as the true class "
                         f"{pct(fe['shortcut_rate'])} of the time (chance {pct(fe['chance'])}), "
                         f"mean confidence {fe['mean_conf']:.2f}, confident {pct(fe['confident_share'])}.")
            for fam, v in fe["by_family"].items():
                lines.append(f"  - {fam}: shortcut rate {pct(v['shortcut_rate'])} (n={v['n']}), mean conf {v['mean_conf']:.2f}")
        if "bg_swap" in bg:
            bs = bg["bg_swap"]
            lines.append(f"- **bg_swap** (n={bs['n']}): accuracy {pct(bs['accuracy'])}, "
                         f"flip to the new background's family {pct(bs['flip_rate'])}.")
            for fam, v in bs["by_source_family"].items():
                lines.append(f"  - fish from {fam}: accuracy {pct(v['accuracy'])}, flip {pct(v['flip_rate'])}, "
                             f"top predictions {v['top_predictions']}")
    if m["ood"]:
        lines += ["", "## Out-of-distribution", "",
                  "| set | n | mean conf | conf>=0.9 | accepted at production threshold | AUROC | FPR@95TPR |",
                  "|---|---|---|---|---|---|---|"]
        for s, v in m["ood"].items():
            lines.append(f"| {s} | {v['n']} | {v['mean_conf']:.3f} | {pct(v['confident_share'])} | "
                         f"{pct(v['accepted_at_production_threshold'])} | {v['auroc']:.3f} | {pct(v['fpr_at_95_tpr'])} |")
    if "field" in m["per_class"]:
        lines += ["", "## Field photos per class", "", "| class | n | recall | top confusions |", "|---|---|---|---|"]
        for c, v in sorted(m["per_class"]["field"].items(), key=lambda kv: kv[1]["recall"]):
            lines.append(f"| {c} | {v['n']} | {pct(v['recall'])} | {v['top_confusion']} |")
    if "gates" in m:
        lines += ["", "## Gates", "",
                  "| slice | n | pass | not fish | unknown species | accuracy on passed | wrong reaching operator | "
                  "wrong & conf>=0.9 reaching operator |",
                  "|---|---|---|---|---|---|---|---|"]
        for s, v in m["gates"].items():
            lines.append(f"| {s} | {v['n']} | {pct(v['pass'])} | {pct(v['rejected_not_fish'])} | "
                         f"{pct(v['rejected_unknown_species'])} | {pct(v.get('accuracy_on_passed'))} | "
                         f"{pct(v.get('wrong_reaching_operator'))} | {pct(v.get('wrong_confident_reaching_operator'))} |")
    lines += ["", "Figures: `figures/confusion.png`, `figures/corruptions.png`, `figures/confidence.png`, "
              "`figures/reliability.png`."]
    return "\n".join(lines) + "\n"


def single(run: str) -> None:
    df, classes, meta = load(run)
    m = analyse(df, classes, meta)
    folder = RESULTS_DIR / run
    figs = folder / "figures"
    figs.mkdir(exist_ok=True)
    fig_confusions(df, classes, figs / "confusion.png")
    fig_corruptions(m, figs / "corruptions.png")
    fig_confidence(df, figs / "confidence.png")
    fig_reliability(df, figs / "reliability.png")
    (folder / "metrics.json").write_text(json.dumps(m, indent=2))
    text = summary_md(m, meta)
    (folder / "summary.md").write_text(text)
    print(text)


KEY_METRICS = [
    ("clean acc", lambda m: m["slices"].get("clean", {}).get("accuracy")),
    ("field acc", lambda m: m["slices"].get("field", {}).get("accuracy")),
    ("field macro-F1", lambda m: m["slices"].get("field", {}).get("macro_f1")),
    ("field wrong&conf", lambda m: m["slices"].get("field", {}).get("wrong_confident")),
    ("bg_swap acc", lambda m: m["background"].get("bg_swap", {}).get("accuracy")),
    ("bg_swap flip", lambda m: m["background"].get("bg_swap", {}).get("flip_rate")),
    ("fish_erased shortcut", lambda m: m["background"].get("fish_erased", {}).get("shortcut_rate")),
    ("blur@4 acc", lambda m: m["corruption_curves"].get("blur", {}).get("4.0", {}).get("accuracy")),
    ("dark@0.25 acc", lambda m: m["corruption_curves"].get("dark", {}).get("0.25", {}).get("accuracy")),
    ("unknown-fish AUROC", lambda m: m["ood"].get("ood_unknown_fish", {}).get("auroc")),
    ("nonfish AUROC", lambda m: m["ood"].get("ood_nonfish", {}).get("auroc")),
    ("field ECE", lambda m: m["slices"].get("field", {}).get("ece")),
    ("field gate pass", lambda m: m.get("gates", {}).get("field", {}).get("pass")),
    ("field wrong&conf after gates", lambda m: m.get("gates", {}).get("field", {}).get("wrong_confident_reaching_operator")),
    ("unknown-fish rejected", lambda m: 1 - m["gates"]["ood_unknown_fish"]["pass"] if "ood_unknown_fish" in m.get("gates", {}) else None),
    ("nonfish rejected", lambda m: 1 - m["gates"]["ood_nonfish"]["pass"] if "ood_nonfish" in m.get("gates", {}) else None),
]


def fig_pareto(table: pd.DataFrame, out) -> None:
    """Quality on the slices that separate models vs CPU latency (log x)."""
    panels = [("field macro-F1", "field photos (macro-F1)"), ("bg_swap acc", "background swap (accuracy)"),
              ("clean acc", "in-distribution test (accuracy)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), sharex=True)
    for ax, (col, title) in zip(axes, panels):
        t = table[["cpu_p50_ms", col]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(t.cpu_p50_ms, t[col], s=40)
        for run, row in t.iterrows():
            ax.annotate(run, (row.cpu_p50_ms, row[col]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency p50 (ms, batch 1, log)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare(runs: list[str]) -> None:
    rows = []
    for run in runs:
        df, classes, meta = load(run)
        m = analyse(df, classes, meta)
        rows.append({"run": run, **{k: f(m) for k, f in KEY_METRICS}})
    table = pd.DataFrame(rows).set_index("run")
    latency_path = RESULTS_DIR / "latency.json"
    latency = json.loads(latency_path.read_text())["models"] if latency_path.exists() else {}
    # A cross-fitted run has the architecture of its linear-probe twin; only the
    # head weights differ, so it reuses that latency.
    # A merged gated run is timed through its final export, which has the same gates.
    lat = {r: latency.get(r) or latency.get(r.replace("xf_", "lp_", 1)) or latency.get(f"{r}_final", {})
           for r in table.index}
    table["cpu_p50_ms"] = [lat[r].get("p50_ms") for r in table.index]
    table["weights_mb"] = [lat[r].get("weights_mb") for r in table.index]
    out = RESULTS_DIR / "compare"
    out.mkdir(exist_ok=True)
    name = "__".join(runs) if len(runs) <= 3 else f"{len(runs)}_runs"
    if table["cpu_p50_ms"].notna().any():
        fig_pareto(table, out / f"{name}_pareto.png")
    table.to_csv(out / f"{name}.csv")
    cells = table.map(lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.3f}")
    md = "\n".join([
        "| run | " + " | ".join(cells.columns) + " |",
        "|---" * (len(cells.columns) + 1) + "|",
        *[f"| {run} | " + " | ".join(row) + " |" for run, row in cells.iterrows()],
    ])
    (out / f"{name}.md").write_text(md + "\n")
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--compare", nargs="+")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.run:
        single(args.run)
    else:
        parser.error("give a run name or --compare")


if __name__ == "__main__":
    main()
