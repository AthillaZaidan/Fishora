import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUT = HERE / "fishora_gated_vit_b.ipynb"
HR  = "---"

# The gated wrapper is embedded verbatim so every export loads through the same
# code the CV service runs (apps/cv_service/runtime.py imports it unchanged).
GATED_PY = (HERE / "gated_inference.py").read_text(encoding="utf-8")
assert "'''" not in GATED_PY


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
    "# Fishora: Gated ViT-B Pipeline\n\n"
    "*Two rejection gates on the frozen ViT-B embedding, so the species classifier only answers for fish it can know.*"
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
    "3. [**Species Exports**](#3)\n"
    "4. [**Embeddings**](#4)\n"
    "5. [**Gate 1: Fish Gate**](#5)\n"
    "6. [**Gate 2: Known-Species Gate**](#6)\n"
    "7. [**Gated Exports**](#7)\n"
    "8. [**Validation Results**](#8)\n"
    "9. [**Export**](#9)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Fishora's species classifier is a closed-set model: it always answers with one of eleven "
    "species. The evaluation suite showed what that costs. With the production threshold of 0.0, "
    "100% of non-fish photos and of fish from other species are accepted, and for ViT-B 37.6% of "
    "unknown species are still accepted at the operating point that keeps 95% of real fish.*\n\n"
    "*A human confirms every species, but a confident wrong label is exactly what a busy operator "
    "confirms without looking. The pipeline therefore needs to refuse inputs it cannot know before "
    "it offers a species.*"
))

cells.append(md(
    "## Aim\n\n"
    "*Add two gates in front of the ViT-B species head, both computed from the same backbone pass: "
    "Gate 1 decides whether the photo shows a fish at all, and Gate 2 decides whether the fish "
    "resembles any fish the model was trained on. A rejected photo goes to manual species entry.*\n\n"
    "*The gates are built three times, once per species export: for the two cross-fitted models "
    "(held-out folds A and B), which the evaluation suite scores on the locked test sets, and for "
    "the final model trained on all field photos, which is deployed. Every threshold is chosen on "
    "validation data in this notebook; no locked test photo is loaded here.*"
))

cells.append(md(
    "## Metric\n\n"
    "***True positive rate (TPR)** is the share of real fish that a gate lets through. It sets the "
    "operating point, because rejecting a real fish costs the operator a manual entry.*\n\n"
    "$$\\text{TPR} = \\frac{\\text{fish accepted}}{\\text{all fish}}$$\n\n"
    "***Rejection rate** is the share of out-of-scope inputs a gate stops, reported per input type "
    "on validation groups (non-fish queries and fish taxa) that were never used for training.*\n\n"
    "***AUROC** measures how well a score separates in-scope fish from out-of-scope inputs over all "
    "thresholds. It is used to choose `k` for Gate 2.*"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from Kaggle, specifically the Fishora dataset at: "
    "https://www.kaggle.com/datasets/athillazaidan/fishora, and from the private "
    "`fishora_eval_kit` dataset packed by `evaluation/cv/kaggle/pack_kit.py`.*\n\n"
    "*Three sources are used. The Fishora train and validation splits (3,932 and 847 images) are "
    "the in-scope fish of the dataset. The 258 screened field photos, split into locked folds by "
    "photographer, are in-scope fish from real conditions. The gate training pool, fetched by "
    "`evaluation/cv/prep/fetch_gate_data.py`, holds fish of 20 taxa outside both the 11 classes "
    "and the unknown-species test set, and non-fish photos for queries disjoint from the test "
    "non-fish queries, all leak-checked against the test photos.*\n\n"
    "*The dataset test split and the kit's `ood/` test photos are not used in this notebook.*\n\n"
    "```\n"
    "fishora_eval_kit/evaluation/cv/\n"
    "├── protocol/\n"
    "|    ├── field_folds.csv       fold A / B per field photo, locked\n"
    "|    └── field_folds.lock\n"
    "└── data/\n"
    "     ├── field/<species>/      258 in-scope field photos\n"
    "     ├── gate_train/\n"
    "     |    ├── fish/<taxon>/    fish outside every test set\n"
    "     |    └── nonfish/         non-fish, queries outside the test set\n"
    "     └── gate_attribution.csv\n"
    "```"
))

cells.append(md(
    "## Approach: Two Gates on One Embedding\n\n"
    "*The backbone is frozen, so every photo is embedded once and all three components read the "
    "same 768-dimensional CLS embedding, L2-normalised for the gates. Serving cost stays one "
    "backbone pass (about 267 ms on CPU) plus a dot product.*\n\n"
    "```\n"
    "Photo -> ViT-B/16 DINOv3 (frozen) -> CLS embedding [768]\n"
    "    |\n"
    "    +-- Gate 1: logistic head -> P(fish)          reject if < fish_threshold\n"
    "    +-- Gate 2: k-th nearest training embedding   reject if distance > knn_threshold\n"
    "    +-- Species head (unchanged) -> 11 classes\n"
    "```\n\n"
    "*Gate 1 is trained with fish as positives (dataset, field photos of the allowed folds, fish of "
    "other taxa) and non-fish as negatives. Fish of other taxa are positives on purpose: Gate 1 "
    "answers only whether a fish is present, and leaves unknown species to Gate 2.*\n\n"
    "*Gate 2 has no trained weights. It stores the embeddings of the dataset train split and the "
    "allowed field photos, and scores a photo by its cosine distance to the k-th nearest one "
    "(the deep nearest-neighbour detector of Sun et al., 2022). Its threshold is set on field "
    "photos scored leave-one-photographer-out, so a photo never finds itself or its photographer's "
    "other photos in the bank.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "Enable a GPU (T4 or P100) in the session settings and add three inputs: the Fishora dataset, "
    "the `fishora_eval_kit` dataset (rebuilt with the gate pool), and the species exports: the "
    "cross-fitted ViT-B folds from the field-probe notebook and the final ViT-B from the "
    "final-model notebook. Embedding extraction takes about five minutes on a T4; everything after "
    "it runs on cached embeddings in seconds."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code("%pip install -q -U \"timm>=1.0.20\""))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import hashlib\n"
    "import json\n"
    "import os\n"
    "import random\n"
    "import shutil\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import timm\n"
    "import torch\n"
    "from PIL import Image\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "print('timm', timm.__version__, '| torch', torch.__version__)"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "`FIELD_VAL_SHARE` and `SEED` must match the species notebooks, so the field photos a gate "
    "validates on are the ones the species head also held out for validation. `FISH_TARGET_TPR` "
    "and `KNN_TARGET_TPR` are the operating points: the share of real fish each gate must let "
    "through on validation."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "    INPUT      = Path('/kaggle/input') if _ON_KAGGLE else Path('../../..').resolve()\n"
    "    OUTPUT_DIR = Path('/kaggle/working/fishora_gated') if _ON_KAGGLE else Path('../kaggle_outputs/gated/fishora_gated')\n"
    "\n"
    "    VARIANTS          = ['A', 'B', 'final']\n"
    "    FIELD_VAL_SHARE   = 0.2\n"
    "    GATE_VAL_SHARE    = 0.2\n"
    "    BATCH_SIZE        = 64\n"
    "    NUM_WORKERS       = 4\n"
    "\n"
    "    FISH_C            = 1.0\n"
    "    FISH_TARGET_TPR   = 0.99\n"
    "    K_GRID            = [1, 3, 5, 10, 20]\n"
    "    KNN_TARGET_TPR    = 0.95\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('device:', DEVICE, '| output:', CFG.OUTPUT_DIR)"
))

cells.append(md(
    "## Load Dataset\n\n"
    "The dataset root is the directory holding `cleaned/` and `splits/`. The kit is the directory "
    "holding `evaluation/cv/protocol/field_folds.csv` together with the gate pool, and the fold "
    "file is checked against its lock before use. Gate photos marked `drop` in the visual screening "
    "(`gate_screening.csv`: non-fish photos that show a fish, fish photos that are fragments or tiny) are removed. Only the train and validation splits are read."
))

cells.append(code(
    "def find_dirs(pred):\n"
    "    hits = [p for p in CFG.INPUT.rglob('*') if p.is_dir() and 'kaggle_outputs' not in p.parts and pred(p)]\n"
    "    return sorted(hits, key=lambda p: len(str(p)))\n"
    "\n"
    "DATA_ROOT = find_dirs(lambda p: (p / 'cleaned').is_dir() and (p / 'splits').is_dir())[0]\n"
    "KIT = find_dirs(lambda p: (p / 'evaluation/cv/protocol/field_folds.csv').exists()\n"
    "                and (p / 'evaluation/cv/data/gate_attribution.csv').exists())[0]\n"
    "KIT_DATA = KIT / 'evaluation' / 'cv' / 'data'\n"
    "\n"
    "def load_split(name):\n"
    "    df = pd.read_csv(DATA_ROOT / 'splits' / f'{name}.csv')\n"
    "    return pd.DataFrame({'path': [str(DATA_ROOT / p) for p in df['clean_path']],\n"
    "                         'label': df['normalized_label'], 'source': f'dataset_{name}', 'group': name})\n"
    "\n"
    "folds_path = KIT / 'evaluation/cv/protocol/field_folds.csv'\n"
    "lock = dict(l.split('=', 1) for l in (KIT / 'evaluation/cv/protocol/field_folds.lock').read_text().splitlines() if '=' in l)\n"
    "assert hashlib.sha256(folds_path.read_bytes()).hexdigest() == lock['field_folds.csv sha256'], 'folds changed after lock'\n"
    "folds = pd.read_csv(folds_path)\n"
    "field_df = pd.DataFrame({'path': [str(KIT_DATA / f) for f in folds['file']], 'label': folds['label'],\n"
    "                         'source': 'field', 'group': folds['observer'], 'fold': folds['fold']})\n"
    "\n"
    "gate = pd.read_csv(KIT_DATA / 'gate_attribution.csv')\n"
    "screen = pd.read_csv(KIT_DATA / 'gate_screening.csv')\n"
    "gate = gate[gate['file'].isin(screen.loc[screen['decision'] == 'keep', 'file'])]\n"
    "gate_df = pd.DataFrame({'path': [str(KIT_DATA / f) for f in gate['file']], 'label': gate['label'],\n"
    "                        'source': gate['set'], 'group': gate['label']})\n"
    "\n"
    "items = pd.concat([load_split('train'), load_split('val'), field_df, gate_df], ignore_index=True)\n"
    "missing = [p for p in items['path'] if not Path(p).exists()]\n"
    "assert not missing, f'{len(missing)} images missing, first: {missing[0]}'\n"
    "print('dataset:', DATA_ROOT, '| kit:', KIT)\n"
    "items.groupby('source').size().rename('images').to_frame()"
))

# ---------------------------------------------------------------------------
# Section 3: Species Exports
# ---------------------------------------------------------------------------
cells.append(section("Species Exports", "3"))

cells.append(md(
    "The gates do not change the species heads. Each variant starts from an existing ViT-B export: "
    "the cross-fitted fold models identify themselves with `heldout_fold` in their config, and "
    "the final model with `field_training: all`. Because the backbone was frozen in every species "
    "run, all three exports must hold the identical backbone, which is checked here so that one "
    "embedding pass serves all three variants."
))

cells.append(code(
    "def discover_exports():\n"
    "    found = {}\n"
    "    for cfg_path in CFG.INPUT.rglob('inference_config.json'):\n"
    "        if 'kaggle_outputs' in cfg_path.parts and CFG._ON_KAGGLE:\n"
    "            continue\n"
    "        cfg = json.loads(cfg_path.read_text())\n"
    "        if not cfg.get('model_name', '').startswith('vit_base_patch16_dinov3') or cfg.get('gate'):\n"
    "            continue\n"
    "        if cfg.get('field_training') == 'all':\n"
    "            found.setdefault('final', cfg_path.parent)\n"
    "        elif cfg.get('heldout_fold') in ('A', 'B'):\n"
    "            found.setdefault(cfg['heldout_fold'], cfg_path.parent)\n"
    "    missing = [v for v in CFG.VARIANTS if v not in found]\n"
    "    assert not missing, f'species exports missing for {missing}'\n"
    "    return {v: found[v] for v in CFG.VARIANTS}\n"
    "\n"
    "EXPORTS = discover_exports()\n"
    "states = {v: torch.load(d / 'model_state_dict.pt', map_location='cpu', weights_only=True) for v, d in EXPORTS.items()}\n"
    "backbone_keys = [k for k in states['A'] if not k.startswith('head.')]\n"
    "for v in CFG.VARIANTS[1:]:\n"
    "    same = all(torch.equal(states['A'][k], states[v][k]) for k in backbone_keys)\n"
    "    assert same, f'backbone of {v} differs from A'\n"
    "del states\n"
    "for v, d in EXPORTS.items():\n"
    "    print(f'{v:6s} {d}')\n"
    "print(f'backbone identical across variants ({len(backbone_keys)} tensors)')"
))

# ---------------------------------------------------------------------------
# Section 4: Embeddings
# ---------------------------------------------------------------------------
cells.append(section("Embeddings", "4"))

cells.append(md(
    "Every image is embedded once through the gated wrapper itself, loaded from the first export, "
    "so the notebook uses the exact preprocessing and pooling that the service will run. "
    "Extraction runs in fp32 to keep the thresholds valid for the fp32 CPU service."
))

cells.append(md("## Gated Wrapper"))

cells.append(md(
    "`GATED_PY` is `evaluation/cv/kaggle/gated_inference.py` from the repository, embedded verbatim. "
    "It is written into every export as `inference.py`, and `score_batch` returns the backbone "
    "embedding path used below."
))

cells.append(code(
    "GATED_PY = r'''" + GATED_PY + "'''\n"
    "\n"
    "def load_wrapper(export_dir):\n"
    "    ns = {}\n"
    "    exec(compile(GATED_PY, str(export_dir / 'inference.py'), 'exec'), ns)\n"
    "    return ns['FishoraClassifier'](export_dir, device=str(DEVICE))\n"
    "\n"
    "clf = load_wrapper(EXPORTS['A'])\n"
    "print('embedding dim:', clf.model.num_features)"
))

cells.append(md("## Extraction"))

cells.append(md(
    "Embeddings are cached to `embeddings.npy` next to `items.csv`, so a re-run after a disconnect "
    "skips extraction. Unreadable images would raise here rather than being skipped silently."
))

cells.append(code(
    "class ImageList(Dataset):\n"
    "    def __init__(self, paths, tf):\n"
    "        self.paths, self.tf = list(paths), tf\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.paths)\n"
    "\n"
    "    def __getitem__(self, i):\n"
    "        with Image.open(self.paths[i]) as img:\n"
    "            return self.tf(img.convert('RGB'))\n"
    "\n"
    "@torch.inference_mode()\n"
    "def embed(paths):\n"
    "    loader = DataLoader(ImageList(paths, clf.transform), batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)\n"
    "    out = []\n"
    "    for x in tqdm(loader, desc='embed'):\n"
    "        x = x.to(DEVICE)\n"
    "        feats = clf.model.forward_head(clf.model.forward_features(x), pre_logits=True).float()\n"
    "        out.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())\n"
    "    return np.concatenate(out)\n"
    "\n"
    "emb_path = CFG.OUTPUT_DIR / 'embeddings.npy'\n"
    "items_path = CFG.OUTPUT_DIR / 'items.csv'\n"
    "if emb_path.exists() and items_path.exists() and pd.read_csv(items_path)['path'].tolist() == items['path'].tolist():\n"
    "    Z = np.load(emb_path)\n"
    "    print('loaded cached embeddings')\n"
    "else:\n"
    "    Z = embed(items['path'])\n"
    "    np.save(emb_path, Z)\n"
    "    items.to_csv(items_path, index=False)\n"
    "print('embeddings:', Z.shape)"
))

cells.append(md("## Validation Groups"))

cells.append(md(
    "Validation is split by group, never by image. Field photos use the same photographer split "
    "as the species notebooks (same seed, same share), and the gate pool holds out whole taxa and "
    "whole non-fish queries, so the validation numbers show how the gates handle kinds of input "
    "they were not trained on."
))

cells.append(code(
    "def split_groups(groups, share, seed=CFG.SEED):\n"
    "    groups = sorted(set(groups))\n"
    "    random.Random(seed).shuffle(groups)\n"
    "    return set(groups[: int(round(len(groups) * share))])\n"
    "\n"
    "def field_pool(variant):\n"
    "    return items['source'].eq('field') & (items['fold'].ne(variant) if variant in ('A', 'B') else True)\n"
    "\n"
    "def field_val_mask(variant):\n"
    "    pool = field_pool(variant)\n"
    "    val_obs = split_groups(items.loc[pool, 'group'], CFG.FIELD_VAL_SHARE)\n"
    "    return pool & items['group'].isin(val_obs)\n"
    "\n"
    "gate_val_groups = {s: split_groups(items.loc[items['source'].eq(s), 'group'], CFG.GATE_VAL_SHARE)\n"
    "                   for s in ('gate_fish', 'gate_nonfish')}\n"
    "is_gate_val = pd.Series(False, index=items.index)\n"
    "for s, g in gate_val_groups.items():\n"
    "    is_gate_val |= items['source'].eq(s) & items['group'].isin(g)\n"
    "for s, g in gate_val_groups.items():\n"
    "    print(f'{s} validation groups: {sorted(g)}')"
))

# ---------------------------------------------------------------------------
# Section 5: Gate 1
# ---------------------------------------------------------------------------
cells.append(section("Gate 1: Fish Gate", "5"))

cells.append(md(
    "Gate 1 is a logistic regression on the normalised embedding with balanced class weights, "
    "since fish outnumber non-fish about eight to one. For each variant, the field photos of the "
    "held-out fold are excluded entirely, so the fold models never see the photos they are later "
    "scored on."
))

cells.append(md("## Training"))

cells.append(md(
    "Positives are the dataset train split, the training field photos of the variant, and the "
    "training taxa of the other-fish pool. Negatives are the training non-fish queries. The "
    "validation set mirrors this with the dataset validation split, the validation photographers, "
    "and the held-out taxa and queries."
))

cells.append(code(
    "def gate1_sets(variant):\n"
    "    fval = field_val_mask(variant)\n"
    "    ftrain = field_pool(variant) & ~fval\n"
    "    src = items['source']\n"
    "    train = src.eq('dataset_train') | ftrain | (src.isin(['gate_fish', 'gate_nonfish']) & ~is_gate_val)\n"
    "    val = src.eq('dataset_val') | fval | (src.isin(['gate_fish', 'gate_nonfish']) & is_gate_val)\n"
    "    return train, val\n"
    "\n"
    "def fit_gate1(variant):\n"
    "    train, _ = gate1_sets(variant)\n"
    "    y = (items.loc[train, 'source'] != 'gate_nonfish').astype(int).to_numpy()\n"
    "    model = LogisticRegression(C=CFG.FISH_C, class_weight='balanced', max_iter=5000)\n"
    "    model.fit(Z[train.to_numpy()], y)\n"
    "    return model\n"
    "\n"
    "gate1 = {v: fit_gate1(v) for v in CFG.VARIANTS}\n"
    "for v, m in gate1.items():\n"
    "    train, val = gate1_sets(v)\n"
    "    print(f'{v:6s} train {int(train.sum())} / val {int(val.sum())} images')"
))

cells.append(md("## Threshold"))

cells.append(md(
    "The threshold is the lowest value that still lets `FISH_TARGET_TPR` of the fish through in "
    "every validation source separately (dataset, field, other taxa). Taking the minimum over "
    "sources keeps the easy dataset photos from setting a threshold that rejects real field photos."
))

cells.append(code(
    "def fish_prob(model, mask):\n"
    "    return model.predict_proba(Z[mask.to_numpy()])[:, 1]\n"
    "\n"
    "def gate1_threshold(variant):\n"
    "    _, val = gate1_sets(variant)\n"
    "    per_source = {}\n"
    "    for s in ('dataset_val', 'field', 'gate_fish'):\n"
    "        p = fish_prob(gate1[variant], val & items['source'].eq(s))\n"
    "        per_source[s] = float(np.quantile(p, 1 - CFG.FISH_TARGET_TPR))\n"
    "    return min(per_source.values()), per_source\n"
    "\n"
    "fish_thr = {}\n"
    "rows = []\n"
    "for v in CFG.VARIANTS:\n"
    "    thr, per_source = gate1_threshold(v)\n"
    "    fish_thr[v] = thr\n"
    "    _, val = gate1_sets(v)\n"
    "    row = {'variant': v, 'fish_threshold': thr}\n"
    "    for s in ('dataset_val', 'field', 'gate_fish', 'gate_nonfish'):\n"
    "        p = fish_prob(gate1[v], val & items['source'].eq(s))\n"
    "        row[f'{s}_pass'] = float((p >= thr).mean())\n"
    "    rows.append(row)\n"
    "gate1_val = pd.DataFrame(rows).set_index('variant')\n"
    "gate1_val.style.format('{:.3f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: the non-fish pass rate on held-out queries (`gate_nonfish_pass`, "
    "lower is better) against the field pass rate (`field_pass`, which should stay near "
    "`FISH_TARGET_TPR`).\n\n"
    "> Other-fish taxa should pass Gate 1 at a high rate. If they do not, Gate 1 has learned "
    "\"one of our species\" instead of \"a fish\", and would overlap with Gate 2."
))

# ---------------------------------------------------------------------------
# Section 6: Gate 2
# ---------------------------------------------------------------------------
cells.append(section("Gate 2: Known-Species Gate", "6"))

cells.append(md(
    "Gate 2 stores a bank of training embeddings and scores a photo by its cosine distance to the "
    "k-th nearest bank entry. A fish of a species the model knows lands close to many training "
    "photos of that species; a fish of another species does not, even when the species head "
    "confidently picks its nearest class."
))

cells.append(md("## Bank and Leave-One-Photographer-Out Scores"))

cells.append(md(
    "The bank holds the dataset train split and every field photo the variant may see. Scoring "
    "those same field photos against the bank would be circular, so each field photo is scored "
    "with every bank entry from its own photographer masked out. Other-fish photos are never in "
    "the bank and act as unknown species."
))

cells.append(code(
    "def bank_mask(variant):\n"
    "    return items['source'].eq('dataset_train') | field_pool(variant)\n"
    "\n"
    "def kth_distance(query, bank, k, exclude=None):\n"
    "    sims = torch.from_numpy(query) @ torch.from_numpy(bank).T\n"
    "    if exclude is not None:\n"
    "        sims[torch.from_numpy(exclude)] = -2.0\n"
    "    return (1.0 - sims.topk(k, dim=1).values[:, -1]).numpy()\n"
    "\n"
    "def gate2_scores(variant, k):\n"
    "    bm = bank_mask(variant).to_numpy()\n"
    "    bank = Z[bm].astype(np.float16).astype(np.float32)\n"
    "    bank_group = items.loc[bm, 'group'].to_numpy()\n"
    "    bank_is_field = items.loc[bm, 'source'].eq('field').to_numpy()\n"
    "    fp = field_pool(variant).to_numpy()\n"
    "    field_groups = items.loc[fp, 'group'].to_numpy()\n"
    "    exclude = (field_groups[:, None] == bank_group[None, :]) & bank_is_field[None, :]\n"
    "    scores = {'field_loo': kth_distance(Z[fp], bank, k, exclude)}\n"
    "    for s in ('dataset_val', 'gate_fish', 'gate_nonfish'):\n"
    "        scores[s] = kth_distance(Z[items['source'].eq(s).to_numpy()], bank, k)\n"
    "    return scores, bank\n"
    "\n"
    "def auroc(in_scores, out_scores):\n"
    "    y = np.r_[np.zeros(len(in_scores)), np.ones(len(out_scores))]\n"
    "    return float(roc_auc_score(y, np.r_[in_scores, out_scores]))"
))

cells.append(md("## Choosing k"))

cells.append(md(
    "`k` is chosen per variant by AUROC between the field photos (leave-one-photographer-out) and "
    "the other-fish taxa, which is Gate 2's actual task. Only validation data enters this choice."
))

cells.append(code(
    "k_rows = []\n"
    "for v in CFG.VARIANTS:\n"
    "    for k in CFG.K_GRID:\n"
    "        s, _ = gate2_scores(v, k)\n"
    "        k_rows.append({'variant': v, 'k': k, 'auroc_other_fish': auroc(s['field_loo'], s['gate_fish']),\n"
    "                       'auroc_nonfish': auroc(s['field_loo'], s['gate_nonfish'])})\n"
    "k_table = pd.DataFrame(k_rows)\n"
    "best_k = k_table.loc[k_table.groupby('variant')['auroc_other_fish'].idxmax()].set_index('variant')['k'].to_dict()\n"
    "print('chosen k:', best_k)\n"
    "k_table.pivot(index='k', columns='variant', values='auroc_other_fish').style.format('{:.3f}')"
))

cells.append(md("## Threshold"))

cells.append(md(
    "The distance threshold is the `KNN_TARGET_TPR` quantile of the leave-one-photographer-out "
    "field distances, so that share of real field fish passes Gate 2 by construction. The dataset "
    "validation split is reported as a sanity check and should pass almost entirely."
))

cells.append(code(
    "knn_thr, banks, rows = {}, {}, []\n"
    "for v in CFG.VARIANTS:\n"
    "    s, banks[v] = gate2_scores(v, best_k[v])\n"
    "    knn_thr[v] = float(np.quantile(s['field_loo'], CFG.KNN_TARGET_TPR))\n"
    "    rows.append({'variant': v, 'k': best_k[v], 'knn_threshold': knn_thr[v],\n"
    "                 **{f'{name}_pass': float((d <= knn_thr[v]).mean()) for name, d in s.items()}})\n"
    "gate2_val = pd.DataFrame(rows).set_index('variant')\n"
    "gate2_val.style.format('{:.3f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: the other-fish pass rate (`gate_fish_pass`, lower is better) at the "
    "fixed field pass rate, compared with the 37.6% unknown-species acceptance of max-softmax at "
    "the same 95% operating point in the evaluation suite.\n\n"
    "> These are validation numbers on taxa chosen for the gate pool. The locked unknown-species "
    "test set (lele, kakap, bawal, cakalang, kerapu, patin, gurame) is scored only by the "
    "evaluation suite."
))

# ---------------------------------------------------------------------------
# Section 7: Gated Exports
# ---------------------------------------------------------------------------
cells.append(section("Gated Exports", "7"))

cells.append(md(
    "Each gated export is the species export plus three changes: `inference.py` becomes the gated "
    "wrapper, `fish_head.pt` holds the Gate 1 weights, and `knn_bank.npy` holds the Gate 2 bank in "
    "float16. The config keeps `heldout_fold` or `field_training` from the species export, so the "
    "evaluation suite still scores each fold model only on its held-out field photos."
))

cells.append(md("## Writing the Exports"))

cells.append(md(
    "The config records the thresholds, `k`, the operating points they were set for, and the "
    "validation pass rates, so every number in the export can be traced back to this notebook."
))

cells.append(code(
    "RUN_NAME = {'A': 'vit_b_gated_xfA', 'B': 'vit_b_gated_xfB', 'final': 'vit_b_gated_final'}\n"
    "\n"
    "def write_gated_export(v):\n"
    "    src, dst = EXPORTS[v], CFG.OUTPUT_DIR / RUN_NAME[v] / 'export'\n"
    "    dst.mkdir(parents=True, exist_ok=True)\n"
    "    shutil.copy(src / 'model_state_dict.pt', dst / 'model_state_dict.pt')\n"
    "    (dst / 'inference.py').write_text(GATED_PY, encoding='utf-8')\n"
    "    m = gate1[v]\n"
    "    torch.save({'weight': torch.tensor(m.coef_, dtype=torch.float32),\n"
    "                'bias': torch.tensor(m.intercept_, dtype=torch.float32)}, dst / 'fish_head.pt')\n"
    "    np.save(dst / 'knn_bank.npy', banks[v].astype(np.float16))\n"
    "    cfg = json.loads((src / 'inference_config.json').read_text())\n"
    "    cfg['recipe'] = cfg.get('recipe', '') + '+gates_v1'\n"
    "    cfg['gate'] = {\n"
    "        'fish_head_file': 'fish_head.pt', 'fish_threshold': fish_thr[v], 'fish_target_tpr': CFG.FISH_TARGET_TPR,\n"
    "        'knn_bank_file': 'knn_bank.npy', 'knn_k': int(best_k[v]), 'knn_threshold': knn_thr[v],\n"
    "        'knn_target_tpr': CFG.KNN_TARGET_TPR, 'bank_size': int(len(banks[v])),\n"
    "        'embedding': 'L2-normalised CLS pre-logits of the frozen backbone',\n"
    "    }\n"
    "    cfg['gate_validation'] = {'gate1': gate1_val.loc[v].to_dict(), 'gate2': gate2_val.loc[v].to_dict()}\n"
    "    (dst / 'inference_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')\n"
    "    return dst\n"
    "\n"
    "GATED = {v: write_gated_export(v) for v in CFG.VARIANTS}\n"
    "for v, d in GATED.items():\n"
    "    size = sum(f.stat().st_size for f in d.iterdir()) / 2**20\n"
    "    print(f'{v:6s} {d}  ({size:.0f} MB)')"
))

cells.append(md("## Parity Check"))

cells.append(md(
    "Each export is reloaded through the wrapper exactly as the CV service loads it, and its gate "
    "scores on sample images are compared with the scores computed in this notebook. The sample "
    "covers every source, and the returned statuses show the gates acting on single photos."
))

cells.append(code(
    "sample = items.groupby('source', group_keys=False).apply(lambda d: d.sample(3, random_state=CFG.SEED))\n"
    "parity_rows = []\n"
    "for v, d in GATED.items():\n"
    "    served = load_wrapper(d)\n"
    "    for idx, row in sample.iterrows():\n"
    "        out = served.predict(row['path'])\n"
    "        z = Z[[idx]]\n"
    "        expected_fish = float(gate1[v].predict_proba(z)[0, 1])\n"
    "        expected_knn = float(kth_distance(z, banks[v], best_k[v])[0])\n"
    "        parity_rows.append({'variant': v, 'source': row['source'], 'status': out['status'],\n"
    "                            'fish_diff': abs(out['gate']['fish_prob'] - expected_fish),\n"
    "                            'knn_diff': abs(out['gate']['knn_distance'] - expected_knn)})\n"
    "    del served\n"
    "parity = pd.DataFrame(parity_rows)\n"
    "worst = parity[['fish_diff', 'knn_diff']].max()\n"
    "print(f\"max |fish_prob diff| = {worst['fish_diff']:.2e}, max |knn diff| = {worst['knn_diff']:.2e}\")\n"
    "assert worst.max() < 1e-3, 'export does not reproduce notebook gate scores'\n"
    "parity.pivot_table(index='source', columns='variant', values='status', aggfunc=lambda s: ', '.join(s))"
))

# ---------------------------------------------------------------------------
# Section 8: Validation Results
# ---------------------------------------------------------------------------
cells.append(section("Validation Results", "8"))

cells.append(md(
    "This section combines both gates on validation data only. A photo passes the pipeline when "
    "it passes Gate 1 and Gate 2. The locked test results (field, unknown species, non-fish, "
    "synthetic) come from the evaluation suite, which scores the fold exports; the numbers here "
    "only show that the operating points behave as intended."
))

cells.append(md("## Combined Pass Rates"))

cells.append(md(
    "Field photos are scored leave-one-photographer-out for Gate 2 and with the variant's Gate 1, "
    "restricted to validation photographers, which is the closest this notebook can get to unseen "
    "field conditions without touching the held-out fold."
))

cells.append(code(
    "def pipeline_pass(v, mask, knn_scores):\n"
    "    p1 = fish_prob(gate1[v], mask) >= fish_thr[v]\n"
    "    p2 = knn_scores <= knn_thr[v]\n"
    "    return p1, p2\n"
    "\n"
    "rows = []\n"
    "for v in CFG.VARIANTS:\n"
    "    s, _ = gate2_scores(v, best_k[v])\n"
    "    fp = field_pool(v)\n"
    "    fval_in_pool = field_val_mask(v)[fp].to_numpy()\n"
    "    cases = {\n"
    "        'field (val photographers)': (field_val_mask(v), s['field_loo'][fval_in_pool]),\n"
    "        'dataset val': (items['source'].eq('dataset_val'), s['dataset_val']),\n"
    "        'other fish (held-out taxa)': (items['source'].eq('gate_fish') & is_gate_val,\n"
    "                                       s['gate_fish'][is_gate_val[items['source'].eq('gate_fish')].to_numpy()]),\n"
    "        'non-fish (held-out queries)': (items['source'].eq('gate_nonfish') & is_gate_val,\n"
    "                                        s['gate_nonfish'][is_gate_val[items['source'].eq('gate_nonfish')].to_numpy()]),\n"
    "    }\n"
    "    for name, (mask, knn) in cases.items():\n"
    "        p1, p2 = pipeline_pass(v, mask, knn)\n"
    "        rows.append({'variant': v, 'input': name, 'n': int(mask.sum()), 'gate1_pass': p1.mean(),\n"
    "                     'gate2_pass': p2.mean(), 'pipeline_pass': (p1 & p2).mean()})\n"
    "combined = pd.DataFrame(rows)\n"
    "combined.to_csv(CFG.OUTPUT_DIR / 'gate_validation.csv', index=False)\n"
    "combined.pivot(index='input', columns='variant', values='pipeline_pass').style.format('{:.1%}')"
))

cells.append(md("## Score Distributions"))

cells.append(md(
    "The histograms show where each input type falls relative to the thresholds for the final "
    "variant. Overlap near a threshold is where the gates will make mistakes."
))

cells.append(code(
    "v = 'final'\n"
    "s, _ = gate2_scores(v, best_k[v])\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))\n"
    "for name, src in [('dataset val', 'dataset_val'), ('field', 'field'), ('other fish', 'gate_fish'), ('non-fish', 'gate_nonfish')]:\n"
    "    mask = field_pool(v) if src == 'field' else items['source'].eq(src)\n"
    "    axes[0].hist(fish_prob(gate1[v], mask), bins=40, range=(0, 1), alpha=0.55, density=True, label=name)\n"
    "axes[0].axvline(fish_thr[v], color='k', ls='--', lw=1)\n"
    "axes[0].set(title='Gate 1: P(fish), final variant', xlabel='P(fish)', ylabel='density')\n"
    "for name, key in [('dataset val', 'dataset_val'), ('field (LOO)', 'field_loo'), ('other fish', 'gate_fish'), ('non-fish', 'gate_nonfish')]:\n"
    "    axes[1].hist(s[key], bins=40, alpha=0.55, density=True, label=name)\n"
    "axes[1].axvline(knn_thr[v], color='k', ls='--', lw=1)\n"
    "axes[1].set(title=f'Gate 2: k-th NN distance (k={best_k[v]}), final variant', xlabel='cosine distance')\n"
    "axes[1].legend(fontsize=8)\n"
    "fig.tight_layout()\n"
    "fig.savefig(CFG.OUTPUT_DIR / 'gate_distributions.png', dpi=150)\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: the share of validation field photos the full pipeline sends to "
    "manual entry (the cost), against the share of held-out non-fish and other-fish taxa it stops "
    "(the benefit).\n\n"
    "> The fold variants and the final variant should agree closely. A large gap would mean the "
    "thresholds depend on which photographers happen to be in the bank."
))

# ---------------------------------------------------------------------------
# Section 9: Export
# ---------------------------------------------------------------------------
cells.append(section("Export", "9"))

cells.append(md(
    "Each gated export is zipped separately. The two fold exports go to the evaluation suite "
    "notebook as an input; the final export is what `FISHORA_CV_EXPORT_DIR` points at in "
    "deployment. Embeddings are not archived because they can be rebuilt in minutes."
))

cells.append(code(
    "archives = []\n"
    "for v, run in RUN_NAME.items():\n"
    "    archives.append(shutil.make_archive(str(CFG.OUTPUT_DIR.parent / f'fishora_{run}'), 'zip',\n"
    "                                        root_dir=CFG.OUTPUT_DIR, base_dir=run))\n"
    "for f in ('gate_validation.csv', 'gate_distributions.png'):\n"
    "    shutil.copy(CFG.OUTPUT_DIR / f, CFG.OUTPUT_DIR.parent / f)\n"
    "for a in archives:\n"
    "    print(f'{Path(a).name:40s} {Path(a).stat().st_size / 2**20:8.1f} MB')"
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
