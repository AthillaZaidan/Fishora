import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "fishora_eval_suite.ipynb"
HR  = "---"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }


def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )


cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Fishora: CV Evaluation Suite\n\n"
    "*Every exported Fishora classifier through the same robustness, field and out-of-distribution slices.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## Fishora Team\n\n"
    "- Member 1\n"
    "- Member 2\n"
    "- Member 3"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Workspace**](#3)\n"
    "4. [**Suite Runs**](#4)\n"
    "5. [**CPU Latency**](#5)\n"
    "6. [**Reports**](#6)\n"
    "7. [**Export**](#7)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*The Fishora classifier scores 100% on its held-out test split, but that split shares "
    "photographing conditions with training: the three Roboflow classes (gembolo, tenggiri, tuna) "
    "are always on red cloth and the eight Fish-gres classes mostly on a gray tray. A perfect "
    "in-distribution score therefore says little about a landing site.*"
))

cells.append(md(
    "## Aim\n\n"
    "*Run the repository's evaluation suite (`evaluation/cv/cv_suite.py`) on a GPU for every exported model "
    "and produce the per-run reports and the cross-model comparison used in the Evaluation "
    "Artifact. The suite code is the same file that runs locally; this notebook only assembles "
    "the workspace and calls it.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Accuracy and macro F1-score** per slice, with bootstrap 95% confidence intervals on the "
    "headline slices. Macro F1 is used because the classes are imbalanced.*\n\n"
    "***Wrong-and-confident rate** is the share of predictions that are wrong with confidence of "
    "at least 0.9. It measures the failures the human-verification gate cannot catch.*\n\n"
    "***OOD AUROC and FPR at 95% TPR** measure whether the max-softmax confidence separates "
    "in-scope fish from out-of-distribution images.*\n\n"
    "$$\\text{FPR@95} = P\\left(\\text{conf}_{\\text{OOD}} \\geq t_{95}\\right), \\quad "
    "P\\left(\\text{conf}_{\\text{ID}} \\geq t_{95}\\right) = 0.95$$"
))

cells.append(md(
    "## Dataset\n\n"
    "*Two inputs are combined. The Fishora dataset on Kaggle supplies the 849 held-out test "
    "images. The private `fishora_eval_kit` dataset, packed by `evaluation/cv/kaggle/pack_kit.py`, supplies "
    "the suite code, the locked screening file, 258 field photos, 172 out-of-distribution photos "
    "(iNaturalist and Openverse, Creative Commons, attribution in `attribution.csv`), and the fish "
    "masks generated locally with rembg.*\n\n"
    "*The datasets are used only for testing. No model is trained or tuned in this notebook.*\n\n"
    "```\n"
    "fishora_eval_kit/\n"
    "├── evaluation/cv/   suite code, protocol/ (locked screening and folds)\n"
    "├── apps/cv_service/ export loader shared with the CV service\n"
    "└── evaluation/cv/data/\n"
    "     ├── field/      in-class photos from outside the training sources\n"
    "     ├── ood/        unknown fish species and non-fish objects\n"
    "     ├── masks/      one fish mask per test image\n"
    "     └── dataset/metadata/manifest.csv\n"
    "```"
))

cells.append(md(
    "## Approach: Slice-Based Evaluation\n\n"
    "*Each model is run once over every slice and all eleven calibrated probabilities are stored. "
    "Metrics are computed afterwards from those stored outputs, so a new metric never requires "
    "another pass over the images.*\n\n"
    "```\n"
    "test image (849) --+-- clean\n"
    "                   +-- 7 corruptions x 3 severities\n"
    "                   +-- bg_removed / fish_erased / bg_swap   (masks)\n"
    "field photos ------+-- field\n"
    "OOD photos --------+-- ood_unknown_fish / ood_nonfish / ood_synthetic\n"
    "                          |\n"
    "                   predictions.csv -> report.py -> metrics + figures\n"
    "```\n\n"
    "*Live fish under water are out of the product's operating domain and are not scored; the "
    "exclusion is recorded in every `run.json`.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "Enable a GPU (T4 or P100) in the session settings and add three inputs: the Fishora dataset, "
    "the private `fishora_eval_kit` dataset, and the output of the linear-probe notebook. Runs whose "
    "`predictions.csv` already exists are skipped, so a re-run after a disconnect resumes. Expected "
    "runtime on a T4 is about 30 minutes for six models plus 10 minutes of CPU latency."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code("%pip install -q -U \"timm>=1.0.20\""))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import json\n"
    "import os\n"
    "import random\n"
    "import shutil\n"
    "import subprocess\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import torch\n"
    "from IPython.display import Image as ShowImage, Markdown, display\n"
    "\n"
    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "Paths and run options are centralised here. `RUN_ORDER` fixes the order of the comparison "
    "table; models that are not found are skipped with a message. The retrained ViT-L "
    "(`lp_vit_l`) stands in for the production export as the baseline: it reproduces the "
    "production recipe (temperature 0.0771 against 0.0773, 100% test for both), and it keeps the "
    "comparison inside one training run. `LIMIT` above zero restricts "
    "the test split to its first N images and is meant only for a quick dry run."
))

cells.append(code(
    "class Settings:\n"
    "    SEED      = 42\n"
    "    INPUT     = Path('/kaggle/input')\n"
    "    WORKSPACE = Path('/kaggle/working/fishora')\n"
    "    BATCH     = 64\n"
    "    LIMIT     = 0\n"
    "    RUN_LATENCY = True\n"
    "    RUN_ORDER = [\n"
    "        'lp_vit_l', 'lp_vit_b', 'lp_vit_s',\n"
    "        'lp_convnext_t', 'lp_efficientnet_b0', 'lp_mobilenetv3_l',\n"
    "    ]\n"
    "    XF_SKIP_CORRUPTIONS = True\n"
    "    LP_NAMES = {\n"
    "        'vit_l_dinov3': 'lp_vit_l', 'vit_b_dinov3': 'lp_vit_b', 'vit_s_dinov3': 'lp_vit_s',\n"
    "        'convnext_t_dinov3': 'lp_convnext_t', 'efficientnet_b0': 'lp_efficientnet_b0',\n"
    "        'mobilenetv3_l': 'lp_mobilenetv3_l',\n"
    "    }\n"
    "\n"
    "CFG = Settings()\n"
    "print(sorted(p.name for p in CFG.INPUT.iterdir()))"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Three things are located under `/kaggle/input`: the kit (a directory holding "
    "`evaluation/cv/protocol/field_folds.csv` and the masks, so an older workspace copied in with reused results is never picked), the Fishora dataset (a directory holding `cleaned/` and `splits/`), and "
    "every model export (a directory holding `inference_config.json`)."
))

cells.append(code(
    "def find_dirs(predicate):\n"
    "    return sorted({p for p in CFG.INPUT.rglob('*') if p.is_dir() and predicate(p)}, key=lambda p: len(str(p)))\n"
    "\n"
    "KIT = find_dirs(lambda p: (p / 'evaluation' / 'cv' / 'protocol' / 'field_folds.csv').exists() and (p / 'evaluation' / 'cv' / 'data' / 'masks').is_dir())[0]\n"
    "DATASET = find_dirs(lambda p: (p / 'cleaned').is_dir() and (p / 'splits').is_dir())[0]\n"
    "\n"
    "exports = {}\n"
    "for cfg_path in CFG.INPUT.rglob('inference_config.json'):\n"
    "    cfg = json.loads(cfg_path.read_text())\n"
    "    folder = cfg_path.parent.parent.name\n"
    "    fold = cfg.get('heldout_fold')\n"
    "    if cfg.get('gate'):\n"
    "        run = f'gated_vit_b_{fold}' if fold else 'gated_vit_b_final'\n"
    "    elif cfg.get('field_training') == 'all':\n"
    "        run = 'final_vit_b'\n"
    "    elif fold:\n"
    "        base = CFG.LP_NAMES.get(folder.removesuffix(f'_xf{fold}'))\n"
    "        run = base and base.replace('lp_', 'xf_', 1) + f'_{fold}'\n"
    "    elif 'recipe' in cfg:\n"
    "        run = CFG.LP_NAMES.get(folder)\n"
    "    else:\n"
    "        run = 'baseline_vit_l_prod'\n"
    "    if run and run not in exports:\n"
    "        exports[run] = cfg_path.parent\n"
    "\n"
    "XF_BASES = sorted({r.rsplit('_', 1)[0] for r in exports if r.startswith('xf_')},\n"
    "                  key=lambda r: CFG.RUN_ORDER.index(r.replace('xf_', 'lp_', 1)))\n"
    "FOLD_BASES = XF_BASES + (['gated_vit_b'] if any(r.startswith('gated_vit_b_') and r[-2:] in ('_A', '_B') for r in exports) else [])\n"
    "FINALS = [r for r in ('final_vit_b', 'gated_vit_b_final') if r in exports]\n"
    "RUNS = CFG.RUN_ORDER + [f'{b}_{f}' for b in FOLD_BASES for f in ('A', 'B')] + FINALS\n"
    "\n"
    "print('kit     :', KIT)\n"
    "print('dataset :', DATASET)\n"
    "for run in RUNS:\n"
    "    print(f'{run:22s}', exports.get(run, 'NOT FOUND'))"
))

# ---------------------------------------------------------------------------
# Section 3: Workspace
# ---------------------------------------------------------------------------
cells.append(section("Workspace", "3"))

cells.append(md(
    "The kit is copied into a writable workspace that mirrors the repository, and the Fishora "
    "test images are linked in at the path the suite expects (`evaluation/cv/data/dataset/cleaned`). The "
    "suite refuses to run if `evaluation/cv/protocol/screening.csv` no longer matches the hash in "
    "`evaluation/cv/protocol/screening.lock`, which is checked here once before any model runs."
))

cells.append(code(
    "if not (CFG.WORKSPACE / 'evaluation' / 'cv' / 'cv_suite.py').exists():\n"
    "    shutil.copytree(KIT, CFG.WORKSPACE, dirs_exist_ok=True)\n"
    "link = CFG.WORKSPACE / 'evaluation' / 'cv' / 'data' / 'dataset' / 'cleaned'\n"
    "if not link.exists():\n"
    "    link.symlink_to(DATASET / 'cleaned', target_is_directory=True)\n"
    "\n"
    "results_dir = CFG.WORKSPACE / 'evaluation' / 'cv' / 'results'\n"
    "results_dir.mkdir(parents=True, exist_ok=True)\n"
    "old_layout = list(CFG.INPUT.rglob('eval/results/*/predictions.csv'))\n"
    "for pred in [*CFG.INPUT.rglob('evaluation/cv/results/*/predictions.csv'), *old_layout]:\n"
    "    target = results_dir / pred.parent.name\n"
    "    if not target.exists():\n"
    "        shutil.copytree(pred.parent, target)\n"
    "        print('reused previous results:', pred.parent.name)\n"
    "for lat in [*CFG.INPUT.rglob('evaluation/cv/results/latency.json'), *CFG.INPUT.rglob('eval/results/latency.json')]:\n"
    "    if not (results_dir / 'latency.json').exists():\n"
    "        shutil.copy(lat, results_dir / 'latency.json')\n"
    "\n"
    "sys.path.insert(0, str(CFG.WORKSPACE))\n"
    "os.chdir(CFG.WORKSPACE)\n"
    "from evaluation.cv.cv_suite import verify_screening_lock\n"
    "from evaluation.cv.common import load_split\n"
    "\n"
    "print('screening sha256:', verify_screening_lock())\n"
    "test = load_split('test')\n"
    "missing = [s.path for s in test if not s.path.exists()]\n"
    "print(f'test images: {len(test)}  missing: {len(missing)}')\n"
    "print('masks:', len(list((CFG.WORKSPACE / 'evaluation/cv/data/masks').glob('*.png'))))"
))

# ---------------------------------------------------------------------------
# Section 4: Suite Runs
# ---------------------------------------------------------------------------
cells.append(section("Suite Runs", "4"))

cells.append(md(
    "Each model runs in a fresh Python process through the same command a local run uses, so the "
    "results are byte-for-byte comparable with `bash evaluation/cv/run_all.sh`. Progress lines are "
    "streamed; the final line of each run lists item counts per slice."
))

cells.append(code(
    "def run(cmd):\n"
    "    env = {**os.environ, 'FISHORA_CV_DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'}\n"
    "    proc = subprocess.Popen(cmd, cwd=CFG.WORKSPACE, env=env, stdout=subprocess.PIPE,\n"
    "                            stderr=subprocess.STDOUT, text=True)\n"
    "    for line in proc.stdout:\n"
    "        print(line, end='')\n"
    "    if proc.wait() != 0:\n"
    "        raise RuntimeError(f'failed: {cmd}')\n"
    "\n"
    "done = []\n"
    "for name in RUNS:\n"
    "    already = (CFG.WORKSPACE / 'evaluation/cv/results' / name / 'predictions.csv').exists()\n"
    "    if name not in exports and not already:\n"
    "        print(f'== {name}: export not found, skipped')\n"
    "        continue\n"
    "    if (CFG.WORKSPACE / 'evaluation/cv/results' / name / 'predictions.csv').exists():\n"
    "        print(f'== {name}: already done')\n"
    "    else:\n"
    "        print(f'== {name}')\n"
    "        fold_run = name.startswith(('xf_', 'gated_vit_b_')) and name[-2:] in ('_A', '_B')\n"
    "        extra = ['--skip-corruptions'] if fold_run and CFG.XF_SKIP_CORRUPTIONS else []\n"
    "        run([sys.executable, '-u', '-m', 'evaluation.cv.cv_suite', '--run', name, '--export', str(exports[name]),\n"
    "             '--batch', str(CFG.BATCH), '--limit', str(CFG.LIMIT), *extra])\n"
    "    run([sys.executable, '-m', 'evaluation.cv.report', name])\n"
    "    done.append(name)"
))

# ---------------------------------------------------------------------------
# Section 5: CPU Latency
# ---------------------------------------------------------------------------
cells.append(section("CPU Latency", "5"))

cells.append(md(
    "Latency is measured on the session CPU with four threads, batch 1, 10 warm-up and 100 timed "
    "calls of the production `predict()`, one subprocess per model. Models already in a reused "
    "`latency.json` are not measured again, and cross-fitted models reuse the latency of the same "
    "architecture, since only the head weights differ. A gated export is timed through its own "
    "`predict()`, so its latency includes both gates. The absolute numbers belong "
    "to this Kaggle CPU, not to an operator's device, but every model is measured on the same "
    "machine, so the ratios between models are what the comparison uses."
))

cells.append(code(
    "lat_path = CFG.WORKSPACE / 'evaluation/cv/results/latency.json'\n"
    "measured = json.loads(lat_path.read_text())['models'] if lat_path.exists() else {}\n"
    "todo = [n for n in done if n in exports and n[-2:] not in ('_A', '_B') and n not in measured]\n"
    "if CFG.RUN_LATENCY and todo:\n"
    "    run([sys.executable, '-u', '-m', 'evaluation.cv.bench_latency', *[f'{n}={exports[n]}' for n in todo]])\n"
    "    new = json.loads(lat_path.read_text())\n"
    "    new['models'] = {**measured, **new['models']}\n"
    "    lat_path.write_text(json.dumps(new, indent=2))\n"
    "if lat_path.exists():\n"
    "    display(pd.DataFrame(json.loads(lat_path.read_text())['models']).T)"
))

# ---------------------------------------------------------------------------
# Section 6: Reports
# ---------------------------------------------------------------------------
cells.append(section("Reports", "6"))

cells.append(md(
    "The baseline report comes first because it defines the failures the Evaluation Artifact "
    "starts from. The comparison then places every model on the slices that separate them."
))

cells.append(md("## Baseline Summary"))

cells.append(md(
    "The summary lists every labelled slice with its confidence interval, the background tests, "
    "the out-of-distribution table, and field accuracy per class."
))

cells.append(code(
    "base = CFG.WORKSPACE / 'evaluation/cv/results/lp_vit_l'\n"
    "display(Markdown((base / 'summary.md').read_text()))\n"
    "for fig in ['confusion', 'corruptions', 'confidence', 'reliability']:\n"
    "    display(ShowImage(str(base / 'figures' / f'{fig}.png')))"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: the gap between clean and field accuracy, the shortcut rate on "
    "fish_erased against the 9.1% chance level, and the out-of-distribution acceptance at the "
    "production threshold of 0.0.\n\n"
    "> Each finding should name the condition, the evidence, and the metric that hides it."
))

cells.append(md("## Model Comparison"))

cells.append(md(
    "The comparison table and the Pareto figure put quality on the separating slices against CPU "
    "latency. The in-distribution panel is included to show that it cannot rank the models."
))

cells.append(code(
    "merged = [b for b in FOLD_BASES if all(f'{b}_{f}' in done for f in ('A', 'B'))]\n"
    "if merged:\n"
    "    run([sys.executable, '-m', 'evaluation.cv.merge_xfit', *merged])\n"
    "    for b in merged:\n"
    "        run([sys.executable, '-m', 'evaluation.cv.report', b])\n"
    "lp_runs = [n for n in CFG.RUN_ORDER if n in done]\n"
    "run([sys.executable, '-m', 'evaluation.cv.report', '--compare', *lp_runs, *merged, *[f for f in FINALS if f in done]])\n"
    "for p in sorted((CFG.WORKSPACE / 'evaluation/cv/results/compare').glob('*_pareto.png')):\n"
    "    display(ShowImage(str(p)))"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: which backbones sit on the Pareto front for field macro-F1 and "
    "background-swap accuracy, and how much quality each step down in latency costs."
))

# ---------------------------------------------------------------------------
# Section 7: Export
# ---------------------------------------------------------------------------
cells.append(section("Export", "7"))

cells.append(md(
    "Only the results folder is archived: predictions, metrics, summaries, figures, and latency. "
    "It contains no photographs, so it can be committed to the repository under `evaluation/cv/results/`."
))

cells.append(code(
    "archive = shutil.make_archive('/kaggle/working/fishora_eval_results', 'zip',\n"
    "                              root_dir=CFG.WORKSPACE / 'evaluation' / 'cv', base_dir='results')\n"
    "print(archive, f'{Path(archive).stat().st_size / 2**20:.1f} MB')"
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
