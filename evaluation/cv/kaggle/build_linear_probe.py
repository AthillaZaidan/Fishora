import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
# --field builds the second notebook: the same recipe, 2-fold cross-fitted on the
# screened field photos (folds from evaluation/cv/prep/make_field_folds.py).
# --final builds on --field: ViT-B only, trained once on all 258 field photos.
FINAL = "--final" in sys.argv
FIELD = "--field" in sys.argv or FINAL
OUT = HERE / ("fishora_final_vit_b.ipynb" if FINAL else "fishora_field_probe.ipynb" if FIELD else "fishora_linear_probe.ipynb")
HR  = "---"

# The production wrapper is embedded verbatim so every export loads through
# exactly the code the CV service runs.
# Copy of the production export wrapper (ai/results/fishora_dinov3_large_frozen/export/inference.py).
INFERENCE_PY = (HERE / "production_inference.py").read_text(encoding="utf-8")
assert "'''" not in INFERENCE_PY


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
    + ("# Fishora: Linear Probe with Field Photos\n\n"
       "*The same linear-probe recipe, retrained with an iNaturalist field-photo pool added to train and validation.*"
       if FIELD else
       "# Fishora: Backbone Linear Probe Comparison\n\n"
       "*Same-recipe linear probes over six pretrained backbones, exported for the Fishora evaluation suite.*")
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
    "3. [**Data Pipeline**](#3)\n"
    "4. [**Training Recipe**](#4)\n"
    "5. [**Linear Probe Runs**](#5)\n"
    "6. [**Results**](#6)\n"
    "7. [**Export**](#7)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Fishora identifies landed fish from a photograph taken at a landing site. The production "
    "classifier is a frozen DINOv3 ViT-L/16 backbone with a trained linear head, about 300M "
    "parameters and a 1.2 GB checkpoint, which takes roughly one second per image on a CPU.*\n\n"
    "*Operators at a landing site usually have no GPU, so the size of the backbone is a product "
    "question as much as an accuracy question.*"
))

cells.append(md(
    "## Aim\n\n"
    "*Train a linear head on each of six pretrained backbones with one identical recipe, then "
    "export every model in the production format so the Fishora evaluation suite can compare "
    "them on the same robustness slices and on CPU latency.*\n\n"
    "*This notebook only trains and exports. The comparison on hard slices (background removed, "
    "background swapped, corruptions, field photos, out-of-distribution images) runs locally "
    "through `evaluation/cv/cv_suite.py`, because those slices depend on locally generated fish masks.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Macro F1-score** averages the per-class F1 over the eleven classes with equal weight. "
    "It is appropriate here because the classes are imbalanced (gembolo has 148 test images, "
    "senangin has 36), so plain accuracy would let the large classes hide the small ones.*\n\n"
    "$$\\text{Macro-F1} = \\frac{1}{K} \\sum_{k=1}^{K} \\frac{2 \\cdot P_k \\cdot R_k}{P_k + R_k}$$\n\n"
    "***Expected Calibration Error (ECE)** is reported alongside, because the production "
    "pipeline uses the softmax confidence to decide when a human must verify a prediction.*\n\n"
    "$$\\text{ECE} = \\sum_{b=1}^{B} \\frac{|S_b|}{N} \\left| \\text{acc}(S_b) - \\text{conf}(S_b) \\right|$$"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from Kaggle, specifically the Fishora dataset at: "
    "https://www.kaggle.com/datasets/athillazaidan/fishora*\n\n"
    "*The dataset contains 5,628 photographs of eleven Indonesian fish species from two source "
    "families (Roboflow and Fish-gres), split by specimen group into 3,932 train, 847 validation, "
    "and 849 test images so that no specimen appears in more than one split.*\n\n"
    "*The dataset is used to train and select each linear head (train and validation) and to "
    "report in-distribution test metrics.*\n\n"
    "```\n"
    "fishora/\n"
    "├── cleaned/\n"
    "|    ├── bandeng/\n"
    "|    ├── ...\n"
    "|    └── tuna/\n"
    "└── splits/\n"
    "     ├── train.csv\n"
    "     ├── val.csv\n"
    "     └── test.csv\n"
    "```"
))

cells.append(md(
    "## Approach: Same-Recipe Linear Probe\n\n"
    "*A linear probe freezes a pretrained backbone and trains only its final linear layer. "
    "Because every backbone gets the same head, data, augmentation, optimiser, and schedule, a "
    "difference in results measures the quality of the backbone features, not the training "
    "recipe.*\n\n"
    "*The recipe is copied from the notebook that produced the production model (AdamW, "
    "learning rate 5e-4, weight decay 0.05, label smoothing 0.1, square-root inverse class "
    "weights, cosine schedule over 10 epochs, early stopping on validation macro-F1, then "
    "temperature scaling on validation). ViT-L is retrained with it too, so the production model "
    "can be separated from the effect of retraining.*\n\n"
    "```\n"
    "Image 256x256 (resize + pad)\n"
    "    |\n"
    "    +-- Frozen backbone (eval mode)\n"
    "         |\n"
    "    Pre-logit features [384 .. 1280]\n"
    "         |\n"
    "    Trainable linear head -> 11 logits -> softmax(logits / T)\n"
    "```\n\n"
    "*One deliberate change from the original recipe: the backbone stays in eval mode during "
    "training. For the ViT backbones this is identical to the original, since they have no batch "
    "normalisation and no active drop path. For the CNN backbones it keeps BatchNorm running "
    "statistics frozen, which is what a linear probe requires.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "Enable a GPU (T4 or P100) and internet access in the Kaggle session settings before running. "
    "Each finished backbone is written to disk immediately, so after a disconnect a re-run of all "
    "cells skips the backbones that are already exported. Expected runtime on a single T4 is "
    "about 45 minutes, most of it spent on ViT-L."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code("%pip install -q -U \"timm>=1.0.20\""))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import gc\n"
    "import json\n"
    "import os\n"
    "import random\n"
    "import shutil\n"
    "import time\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import timm\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from PIL import Image, ImageOps\n"
    "from sklearn.metrics import accuracy_score, f1_score\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "from torchvision import transforms\n"
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
    "All paths, backbones, and hyperparameters are centralised here. The recipe values match the "
    "production training notebook exactly. `BACKBONES` maps a short name to the timm model name "
    "that is also written into each export."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "\n"
    "    DATA_ROOT  = None\n"
    "    OUTPUT_DIR = Path('/kaggle/working/fishora_linear_probe') if _ON_KAGGLE else Path('../kaggle_outputs/linear_probe/fishora_linear_probe')\n"
    "\n"
    "    BACKBONES = {\n"
    "        'vit_l_dinov3':     'vit_large_patch16_dinov3.lvd1689m',\n"
    "        'vit_b_dinov3':     'vit_base_patch16_dinov3.lvd1689m',\n"
    "        'vit_s_dinov3':     'vit_small_patch16_dinov3.lvd1689m',\n"
    "        'convnext_t_dinov3': 'convnext_tiny.dinov3_lvd1689m',\n"
    "        'efficientnet_b0':  'efficientnet_b0.ra_in1k',\n"
    "        'mobilenetv3_l':    'mobilenetv3_large_100.ra_in1k',\n"
    "    }\n"
    "\n"
    "    IMG_SIZE          = 256\n"
    "    BATCH_SIZE        = 16\n"
    "    NUM_WORKERS       = 4\n"
    "    EPOCHS            = 10\n"
    "    EARLY_STOP        = 3\n"
    "    HEAD_LR           = 5e-4\n"
    "    WEIGHT_DECAY      = 0.05\n"
    "    LABEL_SMOOTHING   = 0.10\n"
    "    GRAD_CLIP         = 1.0\n"
    "    AMP               = True\n"
    "    ABSTAIN_THRESHOLD = 0.0\n"
    "    ECE_BINS          = 15\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "AMP_ENABLED = CFG.AMP and DEVICE.type == 'cuda'\n"
    "print('device:', DEVICE, '| output:', CFG.OUTPUT_DIR)"
))

cells.append(md(
    "## Load Dataset\n\n"
    "The dataset root is the first directory under `/kaggle/input` that contains both `cleaned/` "
    "and `splits/`. Paths in the split files are relative to that root."
))

cells.append(code(
    "def discover_root():\n"
    "    if CFG.DATA_ROOT:\n"
    "        return Path(CFG.DATA_ROOT)\n"
    "    search = Path('/kaggle/input') if CFG._ON_KAGGLE else Path('../data/dataset').resolve().parent\n"
    "    candidates = [p for p in search.rglob('*') if p.is_dir() and (p / 'cleaned').is_dir() and (p / 'splits').is_dir()]\n"
    "    if not candidates:\n"
    "        raise FileNotFoundError('No directory with cleaned/ and splits/ found. Set Settings.DATA_ROOT.')\n"
    "    return sorted(candidates, key=lambda p: len(str(p)))[0]\n"
    "\n"
    "ROOT = discover_root()\n"
    "\n"
    "def load_split(name):\n"
    "    df = pd.read_csv(ROOT / 'splits' / f'{name}.csv')\n"
    "    df['resolved_path'] = [str(ROOT / p) for p in df['clean_path']]\n"
    "    missing = [p for p in df['resolved_path'] if not Path(p).exists()]\n"
    "    if missing:\n"
    "        raise FileNotFoundError(f'{len(missing)} {name} images missing, first: {missing[0]}')\n"
    "    return df\n"
    "\n"
    "train_df, val_df, test_df = (load_split(s) for s in ('train', 'val', 'test'))\n"
    "CLASSES = sorted(train_df['normalized_label'].unique())\n"
    "C2I = {c: i for i, c in enumerate(CLASSES)}\n"
    "NUM_CLASSES = len(CLASSES)\n"
    "\n"
    "print('root:', ROOT)\n"
    "print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  Classes: {NUM_CLASSES}')"
))

# ---------------------------------------------------------------------------
# Section 3: Data Pipeline
# ---------------------------------------------------------------------------
cells.append(section("Data Pipeline", "3"))

cells.append(md(
    "Preprocessing follows the production wrapper: the image is scaled to fit 256x256 and padded "
    "with the dataset mean colour, so the fish is never cropped. Training adds the same light "
    "augmentation as the production notebook. All six backbones use ImageNet mean and standard "
    "deviation, which was verified from their timm pretrained configs."
))

cells.append(md("## Transforms"))

cells.append(md(
    "`ResizePad` is copied from the production `inference.py`. The fill colour is the ImageNet "
    "mean in 0-255 units, matching `fill_rgb` in the production export."
))

cells.append(code(
    "class ResizePad:\n"
    "    def __init__(self, size, fill):\n"
    "        self.size = int(size)\n"
    "        self.fill = tuple(fill)\n"
    "\n"
    "    def __call__(self, image):\n"
    "        image = ImageOps.exif_transpose(image).convert('RGB')\n"
    "        w, h = image.size\n"
    "        scale = min(self.size / w, self.size / h)\n"
    "        nw = max(1, int(round(w * scale)))\n"
    "        nh = max(1, int(round(h * scale)))\n"
    "        image = image.resize((nw, nh), Image.Resampling.BICUBIC)\n"
    "        canvas = Image.new('RGB', (self.size, self.size), self.fill)\n"
    "        canvas.paste(image, ((self.size - nw) // 2, (self.size - nh) // 2))\n"
    "        return canvas\n"
    "\n"
    "MEAN = (0.485, 0.456, 0.406)\n"
    "STD  = (0.229, 0.224, 0.225)\n"
    "FILL_RGB = tuple(int(round(x * 255)) for x in MEAN)\n"
    "\n"
    "train_tf = transforms.Compose([\n"
    "    ResizePad(CFG.IMG_SIZE, FILL_RGB),\n"
    "    transforms.RandomHorizontalFlip(p=0.5),\n"
    "    transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.94, 1.04), fill=FILL_RGB),\n"
    "    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(MEAN, STD),\n"
    "])\n"
    "\n"
    "eval_tf = transforms.Compose([\n"
    "    ResizePad(CFG.IMG_SIZE, FILL_RGB),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(MEAN, STD),\n"
    "])\n"
    "print('fill_rgb:', FILL_RGB)"
))

cells.append(md("## Datasets and Class Weights"))

cells.append(md(
    "Class weights are the square root of inverse class frequency, normalised to mean one, as in "
    "the production recipe. They soften the imbalance without letting the smallest classes "
    "dominate the loss."
))

cells.append(code(
    "class FishDataset(Dataset):\n"
    "    def __init__(self, df, tf):\n"
    "        self.paths = df['resolved_path'].tolist()\n"
    "        self.labels = [C2I[c] for c in df['normalized_label']]\n"
    "        self.tf = tf\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.paths)\n"
    "\n"
    "    def __getitem__(self, i):\n"
    "        with Image.open(self.paths[i]) as img:\n"
    "            x = self.tf(img.convert('RGB'))\n"
    "        return x, self.labels[i]\n"
    "\n"
    "def make_loader(df, tf, shuffle):\n"
    "    return DataLoader(\n"
    "        FishDataset(df, tf), batch_size=CFG.BATCH_SIZE, shuffle=shuffle,\n"
    "        num_workers=CFG.NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',\n"
    "        generator=torch.Generator().manual_seed(CFG.SEED),\n"
    "    )\n"
    "\n"
    "train_loader = make_loader(train_df, train_tf, shuffle=True)\n"
    "val_loader   = make_loader(val_df, eval_tf, shuffle=False)\n"
    "test_loader  = make_loader(test_df, eval_tf, shuffle=False)\n"
    "\n"
    "counts = train_df['normalized_label'].value_counts().reindex(CLASSES).to_numpy(dtype=np.float64)\n"
    "weights = 1.0 / np.sqrt(counts)\n"
    "weights = weights / weights.mean()\n"
    "CLASS_WEIGHTS = torch.tensor(weights, dtype=torch.float32)\n"
    "pd.DataFrame({'train_count': counts.astype(int), 'weight': weights.round(3)}, index=CLASSES)"
))

# ---------------------------------------------------------------------------
# Section 4: Training Recipe
# ---------------------------------------------------------------------------
cells.append(section("Training Recipe", "4"))

cells.append(md(
    "The functions in this section are shared by every backbone, which is what makes the "
    "comparison a same-recipe comparison. Only the backbone name changes between runs."
))

cells.append(md("## Model Construction"))

cells.append(md(
    "`build_model` mirrors the production wrapper line for line: create the timm model with an "
    "11-class head, then request CLS-token pooling. The CNN backbones reject `global_pool='token'` "
    "and keep average pooling, exactly as they would inside `inference.py`. Every parameter except "
    "the final linear layer is frozen."
))

cells.append(code(
    "def build_model(model_name, pretrained):\n"
    "    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)\n"
    "    try:\n"
    "        model.reset_classifier(num_classes=NUM_CLASSES, global_pool='token')\n"
    "    except Exception:\n"
    "        pass\n"
    "    for p in model.parameters():\n"
    "        p.requires_grad = False\n"
    "    for p in model.get_classifier().parameters():\n"
    "        p.requires_grad = True\n"
    "    return model\n"
    "\n"
    "def head_logits(model, x):\n"
    "    with torch.no_grad():\n"
    "        feats = model.forward_head(model.forward_features(x), pre_logits=True)\n"
    "    return model.get_classifier()(feats.float())"
))

cells.append(md("## Train and Evaluate"))

cells.append(md(
    "The backbone forward pass runs under `no_grad` and in eval mode, and only the head receives "
    "gradients. Evaluation returns raw logits so that temperature scaling and calibration can be "
    "computed afterwards without another pass over the images."
))

cells.append(code(
    "def train_one_epoch(model, optimizer, scaler, criterion):\n"
    "    model.eval()\n"
    "    head = model.get_classifier()\n"
    "    head.train()\n"
    "    total, n = 0.0, 0\n"
    "    for x, y in tqdm(train_loader, leave=False, desc='train'):\n"
    "        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)\n"
    "        optimizer.zero_grad(set_to_none=True)\n"
    "        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP_ENABLED):\n"
    "            logits = head_logits(model, x)\n"
    "            loss = criterion(logits, y)\n"
    "        scaler.scale(loss).backward()\n"
    "        scaler.unscale_(optimizer)\n"
    "        torch.nn.utils.clip_grad_norm_(head.parameters(), CFG.GRAD_CLIP)\n"
    "        scaler.step(optimizer)\n"
    "        scaler.update()\n"
    "        total += loss.item() * len(y)\n"
    "        n += len(y)\n"
    "    return total / n\n"
    "\n"
    "@torch.no_grad()\n"
    "def collect_logits(model, loader):\n"
    "    model.eval()\n"
    "    logits, labels = [], []\n"
    "    for x, y in tqdm(loader, leave=False, desc='eval'):\n"
    "        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP_ENABLED):\n"
    "            out = model(x.to(DEVICE, non_blocking=True))\n"
    "        logits.append(out.float().cpu())\n"
    "        labels.append(y)\n"
    "    return torch.cat(logits).numpy(), torch.cat(labels).numpy()"
))

cells.append(md("## Metrics and Temperature Scaling"))

cells.append(md(
    "Temperature is fitted on validation logits by minimising negative log-likelihood, as in the "
    "production notebook. ECE uses 15 equal-width confidence bins."
))

cells.append(code(
    "def softmax_t(logits, temperature):\n"
    "    z = torch.tensor(logits, dtype=torch.float32) / temperature\n"
    "    return torch.softmax(z, dim=1).numpy()\n"
    "\n"
    "def fit_temperature(logits, labels):\n"
    "    z = torch.tensor(logits, dtype=torch.float32)\n"
    "    y = torch.tensor(labels, dtype=torch.long)\n"
    "    log_t = nn.Parameter(torch.zeros(1))\n"
    "    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100, line_search_fn='strong_wolfe')\n"
    "\n"
    "    def closure():\n"
    "        opt.zero_grad()\n"
    "        loss = nn.functional.cross_entropy(z / log_t.exp().clamp(0.05, 20.0), y)\n"
    "        loss.backward()\n"
    "        return loss\n"
    "\n"
    "    opt.step(closure)\n"
    "    return float(log_t.detach().exp().clamp(0.05, 20.0))\n"
    "\n"
    "def ece_score(probs, labels, bins=CFG.ECE_BINS):\n"
    "    conf = probs.max(axis=1)\n"
    "    correct = probs.argmax(axis=1) == labels\n"
    "    edges = np.linspace(0.0, 1.0, bins + 1)\n"
    "    ece = 0.0\n"
    "    for lo, hi in zip(edges[:-1], edges[1:]):\n"
    "        mask = (conf > lo) & (conf <= hi)\n"
    "        if mask.any():\n"
    "            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())\n"
    "    return float(ece)\n"
    "\n"
    "def summarize(probs, labels):\n"
    "    pred = probs.argmax(axis=1)\n"
    "    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()\n"
    "    return {\n"
    "        'top1': float(accuracy_score(labels, pred)),\n"
    "        'macro_f1': float(f1_score(labels, pred, average='macro')),\n"
    "        'nll': float(nll),\n"
    "        'ece': ece_score(probs, labels),\n"
    "        'mean_conf': float(probs.max(axis=1).mean()),\n"
    "    }"
))

cells.append(md("## Export in Production Format"))

cells.append(md(
    "Each export holds the full model state dict, an `inference_config.json` with the same keys "
    "as the production config, and the production `inference.py` unchanged. A parity check then "
    "reloads the export through that wrapper and confirms it reproduces the notebook's test "
    "probabilities, so the evaluation suite measures the same model that was trained here."
))

cells.append(code(
    "INFERENCE_PY = r'''" + INFERENCE_PY + "'''\n"
    "\n"
    "def write_export(model, model_name, export_dir, temperature, val_metrics, test_metrics):\n"
    "    export_dir.mkdir(parents=True, exist_ok=True)\n"
    "    torch.save(model.state_dict(), export_dir / 'model_state_dict.pt')\n"
    "    (export_dir / 'inference.py').write_text(INFERENCE_PY, encoding='utf-8')\n"
    "    config = {\n"
    "        'library': 'timm',\n"
    "        'model_name': model_name,\n"
    "        'num_classes': NUM_CLASSES,\n"
    "        'classes': CLASSES,\n"
    "        'class_to_idx': C2I,\n"
    "        'img_size': CFG.IMG_SIZE,\n"
    "        'mean': list(MEAN),\n"
    "        'std': list(STD),\n"
    "        'fill_rgb': list(FILL_RGB),\n"
    "        'temperature': temperature,\n"
    "        'abstain_threshold': CFG.ABSTAIN_THRESHOLD,\n"
    "        'training_strategy': 'frozen_backbone_head_only',\n"
    "        'recipe': 'fishora_linear_probe_v1',\n"
    "        'validation_metrics': val_metrics,\n"
    "        'test_metrics': test_metrics,\n"
    "    }\n"
    "    (export_dir / 'inference_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')\n"
    "\n"
    "def parity_check(export_dir, expected_probs, n=8):\n"
    "    ns = {}\n"
    "    exec(compile(INFERENCE_PY, str(export_dir / 'inference.py'), 'exec'), ns)\n"
    "    clf = ns['FishoraClassifier'](export_dir, device=str(DEVICE))\n"
    "    worst = 0.0\n"
    "    for i in range(n):\n"
    "        out = clf.predict(test_df['resolved_path'].iloc[i], top_k=NUM_CLASSES)\n"
    "        got = np.zeros(NUM_CLASSES)\n"
    "        for c in out['top_candidates']:\n"
    "            got[C2I[c['label']]] = c['confidence']\n"
    "        worst = max(worst, float(np.abs(got - expected_probs[i]).max()))\n"
    "    del clf\n"
    "    return worst"
))

# ---------------------------------------------------------------------------
# Section 5: Linear Probe Runs
# ---------------------------------------------------------------------------
cells.append(section("Linear Probe Runs", "5"))

cells.append(md(
    "Each backbone is trained, calibrated, evaluated, exported, and parity-checked in turn, then "
    "released from GPU memory. A backbone whose `result.json` already exists is loaded from disk "
    "instead of retrained, which makes the loop safe to resume."
))

cells.append(code(
    "def run_backbone(short, model_name):\n"
    "    run_dir = CFG.OUTPUT_DIR / short\n"
    "    result_path = run_dir / 'result.json'\n"
    "    if result_path.exists():\n"
    "        print(f'[{short}] already done, loading result')\n"
    "        return json.loads(result_path.read_text())\n"
    "\n"
    "    seed_everything(CFG.SEED)\n"
    "    run_dir.mkdir(parents=True, exist_ok=True)\n"
    "    model = build_model(model_name, pretrained=True).to(DEVICE)\n"
    "    head = model.get_classifier()\n"
    "    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE), label_smoothing=CFG.LABEL_SMOOTHING)\n"
    "    optimizer = torch.optim.AdamW(head.parameters(), lr=CFG.HEAD_LR, weight_decay=CFG.WEIGHT_DECAY)\n"
    "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)\n"
    "    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)\n"
    "\n"
    "    history, best_f1, best_state, stale = [], -1.0, None, 0\n"
    "    t0 = time.time()\n"
    "    for epoch in range(1, CFG.EPOCHS + 1):\n"
    "        train_loss = train_one_epoch(model, optimizer, scaler, criterion)\n"
    "        val_logits, val_y = collect_logits(model, val_loader)\n"
    "        val_f1 = f1_score(val_y, val_logits.argmax(axis=1), average='macro')\n"
    "        scheduler.step()\n"
    "        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_macro_f1': val_f1})\n"
    "        print(f'[{short}] epoch {epoch:02d} loss={train_loss:.4f} val_f1={val_f1:.4f}')\n"
    "        if val_f1 > best_f1 + 1e-4:\n"
    "            best_f1, stale = val_f1, 0\n"
    "            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}\n"
    "        else:\n"
    "            stale += 1\n"
    "            if stale >= CFG.EARLY_STOP:\n"
    "                print(f'[{short}] early stop')\n"
    "                break\n"
    "    train_minutes = (time.time() - t0) / 60\n"
    "    head.load_state_dict(best_state)\n"
    "\n"
    "    val_logits, val_y = collect_logits(model, val_loader)\n"
    "    test_logits, test_y = collect_logits(model, test_loader)\n"
    "    temperature = fit_temperature(val_logits, val_y)\n"
    "    val_metrics = summarize(softmax_t(val_logits, temperature), val_y)\n"
    "    test_probs = softmax_t(test_logits, temperature)\n"
    "    test_metrics = summarize(test_probs, test_y)\n"
    "    per_class_f1 = f1_score(test_y, test_probs.argmax(axis=1), average=None, labels=range(NUM_CLASSES))\n"
    "\n"
    "    export_dir = run_dir / 'export'\n"
    "    write_export(model, model_name, export_dir, temperature, val_metrics, test_metrics)\n"
    "    np.save(run_dir / 'test_logits.npy', test_logits)\n"
    "    pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)\n"
    "    parity = parity_check(export_dir, test_probs)\n"
    "\n"
    "    result = {\n"
    "        'backbone': short,\n"
    "        'model_name': model_name,\n"
    "        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,\n"
    "        'feature_dim': head.in_features,\n"
    "        'export_mb': (export_dir / 'model_state_dict.pt').stat().st_size / 2**20,\n"
    "        'epochs_run': len(history),\n"
    "        'train_minutes': train_minutes,\n"
    "        'temperature': temperature,\n"
    "        'parity_max_abs_diff': parity,\n"
    "        **{f'val_{k}': v for k, v in val_metrics.items()},\n"
    "        **{f'test_{k}': v for k, v in test_metrics.items()},\n"
    "        'test_per_class_f1': dict(zip(CLASSES, map(float, per_class_f1))),\n"
    "    }\n"
    "    result_path.write_text(json.dumps(result, indent=2))\n"
    "    print(f\"[{short}] test macro-F1={test_metrics['macro_f1']:.4f} ECE={test_metrics['ece']:.4f} \"\n"
    "          f'T={temperature:.4f} parity={parity:.2e} ({train_minutes:.1f} min)')\n"
    "\n"
    "    del model, optimizer, scaler, head\n"
    "    gc.collect()\n"
    "    torch.cuda.empty_cache()\n"
    "    return result"
))

cells.append(md(
    "The loop below runs every backbone in `Settings.BACKBONES`. Smaller backbones go last in the "
    "dictionary only for readability; the order does not affect results because every run reseeds."
))

cells.append(code(
    "results = [run_backbone(short, name) for short, name in CFG.BACKBONES.items()]\n"
    "results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'test_per_class_f1'} for r in results])\n"
    "results_df.to_csv(CFG.OUTPUT_DIR / 'results.csv', index=False)\n"
    "results_df"
))

# ---------------------------------------------------------------------------
# Section 6: Results
# ---------------------------------------------------------------------------
cells.append(section("Results", "6"))

cells.append(md(
    "These are in-distribution numbers on the held-out test split. They are expected to be close "
    "to saturated for every backbone, because the test split shares photographing conditions with "
    "training. The decisive comparison is on the hard slices in the local evaluation suite, so "
    "this section is a sanity check, not a ranking."
))

cells.append(md("## Summary Table"))

cells.append(md(
    "Parity below `1e-2` confirms that each export, loaded through the production wrapper in "
    "fp32, reproduces the notebook's fp16 probabilities. The tolerance is loose because a "
    "temperature below one amplifies small fp16 logit differences."
))

cells.append(code(
    "cols = ['backbone', 'params_m', 'export_mb', 'feature_dim', 'epochs_run', 'train_minutes',\n"
    "        'temperature', 'test_top1', 'test_macro_f1', 'test_ece', 'test_nll', 'parity_max_abs_diff']\n"
    "summary = results_df[cols].sort_values('params_m', ascending=False).reset_index(drop=True)\n"
    "summary.style.format({c: '{:.4f}' for c in cols[6:]} | {'params_m': '{:.1f}', 'export_mb': '{:.0f}', 'train_minutes': '{:.1f}'})"
))

cells.append(md("## Per-Class F1"))

cells.append(md(
    "Per-class F1 shows whether any backbone already struggles on a specific species pair in "
    "distribution, for example the two tilapia (mujair and nila) or the two croakers "
    "(gulamah and gelama bunga)."
))

cells.append(code(
    "per_class = pd.DataFrame({r['backbone']: r['test_per_class_f1'] for r in results}).T\n"
    "fig, ax = plt.subplots(figsize=(11, 3.8))\n"
    "im = ax.imshow(per_class.to_numpy(), vmin=per_class.to_numpy().min(), vmax=1.0, cmap='viridis')\n"
    "ax.set_xticks(range(NUM_CLASSES), per_class.columns, rotation=45, ha='right')\n"
    "ax.set_yticks(range(len(per_class)), per_class.index)\n"
    "for i in range(per_class.shape[0]):\n"
    "    for j in range(per_class.shape[1]):\n"
    "        ax.text(j, i, f'{per_class.iat[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white')\n"
    "fig.colorbar(im, ax=ax, label='F1')\n"
    "ax.set_title('Test per-class F1 by backbone (in-distribution)')\n"
    "fig.tight_layout()\n"
    "fig.savefig(CFG.OUTPUT_DIR / 'per_class_f1.png', dpi=150)\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fill in after the run: which backbones saturate the in-distribution test split, and "
    "whether any class pair already separates the backbones before the hard slices are applied.\n\n"
    "> A near-identical table across backbones is the expected outcome and is itself evidence "
    "that the in-distribution test split cannot rank them, which motivates the slice-based "
    "comparison."
))

# ---------------------------------------------------------------------------
# Section 7: Export
# ---------------------------------------------------------------------------
cells.append(section("Export", "7"))

cells.append(md(
    "Each backbone is zipped separately so that a single model can be downloaded without the "
    "1.2 GB ViT-L archive. Unzip an archive into `data/linear_probe/` in the repository and point "
    "the evaluation suite at its `export/` directory."
))

cells.append(code(
    "archives = []\n"
    "for short in CFG.BACKBONES:\n"
    "    base = CFG.OUTPUT_DIR.parent / f'fishora_lp_{short}'\n"
    "    archives.append(shutil.make_archive(str(base), 'zip', root_dir=CFG.OUTPUT_DIR, base_dir=short))\n"
    "shutil.copy(CFG.OUTPUT_DIR / 'results.csv', CFG.OUTPUT_DIR.parent / 'fishora_lp_results.csv')\n"
    "for a in archives:\n"
    "    print(f'{Path(a).name:40s} {Path(a).stat().st_size / 2**20:8.1f} MB')"
))

# ---------------------------------------------------------------------------
# --field variant: 2-fold cross-fitting on the kept field photos
# ---------------------------------------------------------------------------
if FIELD:
    def patch(old: str, new: str) -> None:
        hits = [c for c in cells if old in c["source"]]
        assert len(hits) == 1, (old[:60], len(hits))
        hits[0]["source"] = hits[0]["source"].replace(old, new)

    patch(
        "*Train a linear head on each of six pretrained backbones with one identical recipe, then "
        "export every model in the production format",
        "*Retrain the linear head of all six backbones with the identical recipe, adding field "
        "photos to training. The baseline evaluation showed field photos as the main failure (81.8% "
        "for ViT-L against 100% in distribution), so the only change from the first linear-probe run "
        "is the training data.*\n\n"
        "*The field photos are the evaluation photos themselves, so they are used through 2-fold "
        "cross-fitting: every backbone is trained twice, once with fold B and scored on fold A "
        "(`_xfA`), once with fold A and scored on fold B (`_xfB`). Every field photo is then scored by "
        "a model that never saw it, and the pooled field accuracy is directly comparable with the "
        "baseline on the same 258 photos. Export every model in the production format",
    )
    patch(
        "*The dataset is used to train and select each linear head (train and validation) and to "
        "report in-distribution test metrics.*",
        "*The dataset is used to train and select each linear head (train and validation) and to "
        "report in-distribution test metrics.*\n\n"
        "*The field photos come from the `fishora_eval_kit` input: the 258 photos kept by the locked "
        "screening file, assigned to folds A and B by `evaluation/cv/prep/make_field_folds.py` (129 each). The folds "
        "are grouped by photographer, balanced per class, and locked before training. Gulamah and "
        "gembolo have no field photos.*",
    )
    patch(
        "    OUTPUT_DIR = Path('/kaggle/working/fishora_linear_probe') if _ON_KAGGLE else Path('../kaggle_outputs/linear_probe/fishora_linear_probe')\n",
        "    OUTPUT_DIR = Path('/kaggle/working/fishora_field_xfit') if _ON_KAGGLE else Path('../kaggle_outputs/field_xfit/fishora_field_xfit')\n"
        "    FOLDS      = ['A', 'B']\n"
        "    FIELD_VAL_SHARE = 0.2\n"
        "    RUN_SUFFIX = ''\n"
        "    HELDOUT    = None\n",
    )
    patch("        'recipe': 'fishora_linear_probe_v1',\n",
          "        'recipe': 'fishora_linear_probe_v1+field_xfit',\n"
          "        'heldout_fold': CFG.HELDOUT,\n")
    patch("    run_dir = CFG.OUTPUT_DIR / short\n", "    run_dir = CFG.OUTPUT_DIR / f'{short}{CFG.RUN_SUFFIX}'\n")
    patch("        'backbone': short,\n", "        'backbone': f'{short}{CFG.RUN_SUFFIX}',\n        'heldout_fold': CFG.HELDOUT,\n")
    patch(
        "results = [run_backbone(short, name) for short, name in CFG.BACKBONES.items()]\n",
        "results = []\n"
        "for heldout in CFG.FOLDS:\n"
        "    set_fold(heldout)\n"
        "    results += [run_backbone(short, name) for short, name in CFG.BACKBONES.items()]\n",
    )
    patch(
        "for short in CFG.BACKBONES:\n"
        "    base = CFG.OUTPUT_DIR.parent / f'fishora_lp_{short}'\n"
        "    archives.append(shutil.make_archive(str(base), 'zip', root_dir=CFG.OUTPUT_DIR, base_dir=short))\n"
        "shutil.copy(CFG.OUTPUT_DIR / 'results.csv', CFG.OUTPUT_DIR.parent / 'fishora_lp_results.csv')\n",
        "for run in sorted(p.name for p in CFG.OUTPUT_DIR.iterdir() if p.is_dir()):\n"
        "    archives.append(shutil.make_archive(str(CFG.OUTPUT_DIR.parent / f'fishora_xf_{run}'), 'zip',\n"
        "                                        root_dir=CFG.OUTPUT_DIR, base_dir=run))\n"
        "shutil.copy(CFG.OUTPUT_DIR / 'results.csv', CFG.OUTPUT_DIR.parent / 'fishora_xf_results.csv')\n",
    )

    # Loaders are rebuilt per fold here. Without persistent workers every epoch
    # forks fresh workers, and a forked worker that inherits a stale iterator
    # floods the log with "can only test a child process" on cleanup. Sampling
    # order is unchanged: the seeded generator is still created per loader.
    patch(
        "        num_workers=CFG.NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',\n",
        "        num_workers=CFG.NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',\n"
        "        persistent_workers=CFG.NUM_WORKERS > 0,\n",
    )

    loaders_idx = next(i for i, c in enumerate(cells) if "CLASS_WEIGHTS = torch.tensor(weights" in c["source"])
    cells[loaders_idx + 1:loaders_idx + 1] = [
        md("## Field Folds"),
        md(
            "The kit is located as the first input directory holding `evaluation/cv/protocol/field_folds.csv`, and the "
            "fold file is checked against its lock before use. `set_fold(heldout)` rebuilds train, "
            "validation, loaders and class weights for one fold: the other fold's photos are split by "
            "photographer, 80% into train and 20% into validation, and the held-out fold is never "
            "loaded. The in-distribution test split is unchanged."
        ),
        code(
            "def discover_kit():\n"
            "    if not CFG._ON_KAGGLE:\n        return Path('../../..').resolve()\n    search = Path('/kaggle/input')\n"
            "    hits = [p.parents[3] for p in search.rglob('field_folds.csv') if p.parent.name == 'protocol' and p.parents[2].name == 'evaluation']\n"
            "    if not hits:\n"
            "        raise FileNotFoundError('No kit with evaluation/cv/protocol/field_folds.csv found.')\n"
            "    return sorted(hits, key=lambda p: len(str(p)))[0]\n"
            "\n"
            "KIT = discover_kit()\n"
            "folds_path = KIT / 'evaluation' / 'cv' / 'protocol' / 'field_folds.csv'\n"
            "lock = dict(l.split('=', 1) for l in (KIT / 'evaluation' / 'cv' / 'protocol' / 'field_folds.lock').read_text().splitlines() if '=' in l)\n"
            "assert hashlib.sha256(folds_path.read_bytes()).hexdigest() == lock['field_folds.csv sha256'], 'folds changed after lock'\n"
            "folds = pd.read_csv(folds_path)\n"
            "folds['resolved_path'] = [str(KIT / 'evaluation' / 'cv' / 'data' / f) for f in folds['file']]\n"
            "folds['normalized_label'] = folds['label']\n"
            "base_train, base_val = train_df.copy(), val_df.copy()\n"
            "cols = ['resolved_path', 'normalized_label']\n"
            "\n"
            "def set_fold(heldout):\n"
            "    global train_df, val_df, train_loader, val_loader, CLASS_WEIGHTS\n"
            "    pool = folds[folds['fold'] != heldout]\n"
            "    observers = sorted(pool['observer'].unique())\n"
            "    random.Random(CFG.SEED).shuffle(observers)\n"
            "    val_obs = set(observers[: int(round(len(observers) * CFG.FIELD_VAL_SHARE))])\n"
            "    f_train, f_val = pool[~pool['observer'].isin(val_obs)], pool[pool['observer'].isin(val_obs)]\n"
            "    train_loader = val_loader = None\n"
            "    gc.collect()\n"
            "    train_df = pd.concat([base_train[cols].assign(source='dataset'), f_train[cols].assign(source='field')], ignore_index=True)\n"
            "    val_df = pd.concat([base_val[cols].assign(source='dataset'), f_val[cols].assign(source='field')], ignore_index=True)\n"
            "    train_loader = make_loader(train_df, train_tf, shuffle=True)\n"
            "    val_loader = make_loader(val_df, eval_tf, shuffle=False)\n"
            "    counts = train_df['normalized_label'].value_counts().reindex(CLASSES).fillna(0).to_numpy(dtype=np.float64)\n"
            "    w = 1.0 / np.sqrt(counts)\n"
            "    CLASS_WEIGHTS = torch.tensor(w / w.mean(), dtype=torch.float32)\n"
            "    CFG.HELDOUT, CFG.RUN_SUFFIX = heldout, f'_xf{heldout}'\n"
            "    print(f'held out {heldout}: field train {len(f_train)} / val {len(f_val)} / held-out {int((folds.fold == heldout).sum())}')\n"
            "\n"
            "print('kit:', KIT, '| folds sha256 ok')\n"
            "folds.pivot_table(index='label', columns='fold', values='file', aggfunc='count')"
        ),
    ]
    patch("import gc\n", "import gc\nimport hashlib\n")

# ---------------------------------------------------------------------------
# --final variant: the deployable ViT-B, trained on all 258 field photos
# ---------------------------------------------------------------------------
if FINAL:
    patch("# Fishora: Linear Probe with Field Photos\n\n"
          "*The same linear-probe recipe, retrained with an iNaturalist field-photo pool added to train and validation.*",
          "# Fishora: Final ViT-B Species Head\n\n"
          "*The cross-fitted recipe, trained once on all field photos to produce the model that is deployed.*")
    patch("*Retrain the linear head of all six backbones with the identical recipe, adding field "
          "photos to training.",
          "*Train the deployable ViT-B head with the recipe that was evaluated by 2-fold cross-fitting, now "
          "on all 258 field photos (80% of photographers to train, 20% to validation). The cross-fitted "
          "models measured this recipe; this model is not scored on field photos, because it has seen "
          "them all.*\n\n"
          "*For reference, the cross-fitting runs retrained the linear head of all six backbones with the "
          "identical recipe, adding field photos to training.")
    patch("        'vit_l_dinov3':     'vit_large_patch16_dinov3.lvd1689m',\n", "")
    for short in ("vit_s_dinov3", "convnext_t_dinov3", "efficientnet_b0", "mobilenetv3_l"):
        hit = next(c for c in cells if f"        '{short}':" in c["source"])
        hit["source"] = "".join(l for l in hit["source"].splitlines(keepends=True) if not l.startswith(f"        '{short}':"))
    patch("Path('/kaggle/working/fishora_field_xfit') if _ON_KAGGLE else Path('../kaggle_outputs/field_xfit/fishora_field_xfit')",
          "Path('/kaggle/working/fishora_final') if _ON_KAGGLE else Path('../kaggle_outputs/final/fishora_final')")
    patch("    FOLDS      = ['A', 'B']\n", "    FOLDS      = ['ALL']\n")
    patch("        'recipe': 'fishora_linear_probe_v1+field_xfit',\n",
          "        'recipe': 'fishora_linear_probe_v1+field_all',\n        'field_training': 'all',\n")
    patch("    pool = folds[folds['fold'] != heldout]\n",
          "    pool = folds if heldout == 'ALL' else folds[folds['fold'] != heldout]\n")
    patch("    CFG.HELDOUT, CFG.RUN_SUFFIX = heldout, f'_xf{heldout}'\n",
          "    CFG.HELDOUT, CFG.RUN_SUFFIX = (None, '_final') if heldout == 'ALL' else (heldout, f'_xf{heldout}')\n")
    patch("f'fishora_xf_{run}'", "f'fishora_final_{run}'")
    patch("CFG.OUTPUT_DIR.parent / 'fishora_xf_results.csv'", "CFG.OUTPUT_DIR.parent / 'fishora_final_results.csv'")

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
