"""Architecture figure of the shipped vision model (gated DINOv3 ViT-B/16).

Hyperparameters that the serving code uses (temperature, gate thresholds, k, bank size) are read
from the export's inference_config.json, so the figure cannot drift from the model.

    python -m evaluation.cv.artifact.architecture
    -> evaluation/cv/artifact/figures/architecture.{pdf,png}

Needs: playwright (+ `playwright install chromium`), pillow.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

from PIL import Image

from evaluation.cv.common import CV_DIR, ROOT

EXPORT = Path(os.environ.get("FISHORA_CV_EXPORT_DIR", ROOT / "ai" / "results" / "fishora_vit_b_gated" / "export"))
OUT_DIR = CV_DIR / "artifact" / "figures"
W, H = 1800, 968

INK, MUTED, FAINT, LINE = "#0f1a21", "#56737f", "#8ea5b1", "#cfd9df"
FROZEN, FROZEN_EDGE = "#e7eef3", "#7d98a7"
FITTED, FITTED_EDGE = "#fdf0dc", "#d98613"
MEMORY, MEMORY_EDGE = "#dcefec", "#2b7a73"
NAVY, RED, RED_BG, GREEN, GREEN_BG = "#142530", "#c8553d", "#f8e3de", "#2b7a73", "#dcefec"


def photo_b64() -> str:
    """The clean tuna tile from the Evaluation Artifact's slice figure (a Fishora test image)."""
    src = Image.open(CV_DIR / "artifact" / "figures" / "slices.png").convert("RGB")
    tile = src.crop((87, 56, 374, 341)).resize((256, 256), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    tile.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- primitives


def t(x, y, s, size=14, fill=INK, weight=400, anchor="start", cls="", italic=False):
    style = "font-style:italic;" if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}" class="{cls}" style="{style}">{s}</text>')


def m(s):
    """Inline maths: serif italic, like a paper."""
    return f'<tspan class="m">{s}</tspan>'


def box(x, y, w, h, fill="#fff", stroke=LINE, r=10, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def arrow(points, color=INK, sw=1.8, dash=None, head=True):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#ah)"' if head else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{mk} stroke-linejoin="round"/>'


def tag(x, y, label, fill, stroke, color):
    w = 12 + 7.6 * len(label)
    return (f'<rect x="{x}" y="{y}" width="{w:.0f}" height="20" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            + t(x + w / 2, y + 14.5, label, 11.5, color, 700, "middle"))


def token(x, y, label, fill, color="#fff", stroke=None):
    s = stroke or fill
    return (f'<rect x="{x}" y="{y}" width="50" height="15" rx="3" fill="{fill}" stroke="{s}" stroke-width="1"/>'
            + t(x + 25, y + 11.5, label, 10.5, color, 600, "middle"))


# --------------------------------------------------------------------------- figure


def figure(cfg: dict, img: str) -> str:
    g = cfg["gate"]
    T, t1, t2 = cfg["temperature"], g["fish_threshold"], g["knn_threshold"]
    k, bank, n_cls = g["knn_k"], g["bank_size"], cfg["num_classes"]
    size = cfg["img_size"]
    patches = (size // 16) ** 2
    s = []

    # ---------------- panel (a): inference ----------------
    s.append(box(16, 16, W - 32, 548, "#fbfcfd", "#e3e8ec", 14))
    s.append(t(40, 52, "(a)", 20, INK, 800) + t(76, 52, "Inference: one frozen backbone pass feeds the species head and both rejection gates", 19, INK, 700))

    # input
    s.append(t(44, 98, "Input photo", 14, MUTED, 700))
    s.append(f'<image href="data:image/jpeg;base64,{img}" x="44" y="108" width="150" height="150" preserveAspectRatio="xMidYMid slice"/>')
    s.append(box(44, 108, 150, 150, "none", "#9fb3bd", 4, 1))
    s.append(t(44, 284, m(f"x ∈ ℝ<tspan baseline-shift='super' font-size='10'>3×{size}×{size}</tspan>"), 15))
    s.append(t(44, 306, "EXIF-fix, aspect-kept resize,", 12.5, MUTED))
    s.append(t(44, 322, f"pad to {size}², normalise", 12.5, MUTED))
    s.append(arrow([(204, 183), (228, 183)]))

    # patchify
    s.append(t(236, 98, "Patchify", 14, MUTED, 700))
    s.append(f'<image href="data:image/jpeg;base64,{img}" x="236" y="108" width="150" height="150" preserveAspectRatio="xMidYMid slice"/>')
    grid = "".join(f'<line x1="{236 + i * 150 / 16:.2f}" y1="108" x2="{236 + i * 150 / 16:.2f}" y2="258" stroke="#fff" stroke-width="0.8" opacity=".8"/>'
                   f'<line x1="236" y1="{108 + i * 150 / 16:.2f}" x2="386" y2="{108 + i * 150 / 16:.2f}" stroke="#fff" stroke-width="0.8" opacity=".8"/>'
                   for i in range(1, 16))
    s.append(grid + box(236, 108, 150, 150, "none", "#9fb3bd", 4, 1))
    s.append(t(236, 284, f"16×16 px patches", 13.5))
    s.append(t(236, 306, f"{size // 16}×{size // 16} = {patches} patches,", 12.5, MUTED))
    s.append(t(236, 322, "linear embed to 768-d", 12.5, MUTED))
    s.append(arrow([(396, 183), (420, 183)]))

    # token column
    s.append(t(428, 98, "Tokens", 14, MUTED, 700))
    y = 106
    s.append(token(428, y, "[CLS]", NAVY)); y += 17
    for _ in range(4):
        s.append(token(428, y, "reg", "#b7c6ce", INK)); y += 17
    for lab in ("p₁", "p₂", "p₃"):
        s.append(token(428, y, lab, "#fff", INK, "#9fb3bd")); y += 17
    s.append(t(453, y + 9, "⋮", 14, MUTED, 700, "middle")); y += 15
    s.append(token(428, y, f"p{''.join('₀₁₂₃₄₅₆₇₈₉'[int(c)] for c in str(patches))}", "#fff", INK, "#9fb3bd"))
    s.append(t(428, 306, f"1 + 4 + {patches}", 12.5, MUTED))
    s.append(t(428, 322, f"= {1 + 4 + patches} tokens", 12.5, MUTED))
    s.append(arrow([(486, 183), (512, 183)]))

    # encoder (stacked cards = 12 blocks)
    ex, ey, ew, eh = 518, 92, 440, 200
    s.append(box(ex + 14, ey + 14, ew, eh, "#dfe7ec", FROZEN_EDGE, 12, 1.2))
    s.append(box(ex + 7, ey + 7, ew, eh, "#e3eaef", FROZEN_EDGE, 12, 1.2))
    s.append(box(ex, ey, ew, eh, FROZEN, FROZEN_EDGE, 12, 1.6))
    s.append(t(ex + 18, ey + 30, "DINOv3 ViT-B/16 encoder", 17, INK, 700))
    s.append(tag(ex + ew - 74, ey + 14, "frozen", "#fff", FROZEN_EDGE, MUTED))
    s.append(t(ex + 18, ey + 52, "self-supervised on LVD-1689M · 86 M params · d = 768 · 12 heads", 12.5, MUTED))
    # one transformer block
    by, bh = ey + 104, 40
    bx = ex + 20
    parts = [("LN", 36), ("MHSA + RoPE", 118), ("⊕", 24), ("LN", 36), ("MLP 768→3072→768", 150), ("⊕", 24)]
    xs = []
    cx = bx
    for label, w in parts:
        xs.append((cx, w))
        if label == "⊕":
            s.append(f'<circle cx="{cx + w / 2}" cy="{by + bh / 2}" r="11" fill="#fff" stroke="{INK}" stroke-width="1.4"/>'
                     + t(cx + w / 2, by + bh / 2 + 5, "+", 16, INK, 600, "middle"))
        else:
            s.append(box(cx, by, w, bh, "#fff", "#9fb3bd", 6, 1.2) + t(cx + w / 2, by + 25, label, 12.5, INK, 600, "middle"))
        cx += w + 10
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        s.append(arrow([(a[0] + a[1] + 1, by + bh / 2), (b[0] - 2, by + bh / 2)], INK, 1.3))
    # residual paths
    r1s, r1e = xs[0][0] - 8, xs[2][0] + 12
    r2s, r2e = xs[2][0] + 24 + 4, xs[5][0] + 12
    s.append(arrow([(r1s, by + bh / 2), (r1s, by - 16), (r1e, by - 16), (r1e, by + bh / 2 - 12)], MUTED, 1.2))
    s.append(arrow([(r2s, by + bh / 2 - 8), (r2s, by - 22), (r2e, by - 22), (r2e, by + bh / 2 - 12)], MUTED, 1.2))
    s.append(arrow([(ex + 4, by + bh / 2), (bx - 2, by + bh / 2)], INK, 1.3))
    s.append(t(ex + 20, ey + eh - 18, "LayerScale on both residual branches", 12, MUTED))
    s.append(t(ex + ew - 18, ey + eh - 14, "× 12", 22, INK, 800, "end"))

    # CLS output
    s.append(arrow([(ex + ew + 16, 183), (1006, 183)]))
    s.append(token(1010, 176, "[CLS]", NAVY))
    s.append(t(1035, 164, m("h ∈ ℝ<tspan baseline-shift='super' font-size='10'>768</tspan>"), 15, INK, 400, "middle"))
    s.append(t(1035, 212, "final LN,", 11.5, MUTED, 400, "middle"))
    s.append(t(1035, 226, "CLS pooling", 11.5, MUTED, 400, "middle"))

    # species head
    sx, sy, sw_, sh = 1124, 84, 300, 104
    s.append(arrow([(1060, 183), (1080, 183), (1080, 136), (sx - 2, 136)]))
    s.append(box(sx, sy, sw_, sh, FITTED, FITTED_EDGE, 10, 1.6))
    s.append(t(sx + 16, sy + 26, "Species head", 16, INK, 700) + tag(sx + sw_ - 66, sy + 11, "fitted", "#fff", FITTED_EDGE, "#a3620d"))
    s.append(t(sx + 16, sy + 50, f"linear probe on {m('h')}, " + m(f"W ∈ ℝ<tspan baseline-shift='super' font-size='10'>{n_cls}×768</tspan>"), 13.5))
    s.append(t(sx + 16, sy + 74, m("p = softmax((Wh + b) / T)"), 15))
    s.append(t(sx + 16, sy + 94, f"T = {T:.3f}, fitted on validation", 12.5, MUTED))

    # l2 normalisation
    s.append(arrow([(1035, 234), (1035, 340), (1054, 340)]))
    s.append(f'<circle cx="1076" cy="340" r="21" fill="#fff" stroke="{INK}" stroke-width="1.5"/>' + t(1076, 346, m("ℓ₂"), 17, INK, 400, "middle"))
    s.append(t(1068, 390, m("ẑ = h/<tspan style='font-style:normal'>‖</tspan>h<tspan style='font-style:normal'>‖</tspan>"), 14, INK, 400, "middle"))

    # gate 1
    g1x, g1y, g1w, g1h = 1124, 214, 300, 96
    s.append(arrow([(1097, 340), (1106, 340), (1106, 262), (g1x - 2, 262)]))
    s.append(box(g1x, g1y, g1w, g1h, FITTED, FITTED_EDGE, 10, 1.6))
    s.append(t(g1x + 16, g1y + 26, "Gate 1 · is it a fish?", 16, INK, 700) + tag(g1x + g1w - 66, g1y + 11, "fitted", "#fff", FITTED_EDGE, "#a3620d"))
    s.append(t(g1x + 16, g1y + 50, "logistic regression on " + m("ẑ"), 13.5))
    s.append(t(g1x + 16, g1y + 76, m("P(fish) = σ(w<tspan baseline-shift='super' font-size='10'>⊤</tspan>ẑ + b)"), 15))

    # gate 2
    g2x, g2y, g2w, g2h = 1124, 328, 300, 112
    s.append(arrow([(1106, 340), (1106, 384), (g2x - 2, 384)]))
    s.append(box(g2x, g2y, g2w, g2h, MEMORY, MEMORY_EDGE, 10, 1.6))
    s.append(t(g2x + 16, g2y + 26, "Gate 2 · known species?", 16, INK, 700) + tag(g2x + g2w - 76, g2y + 11, "memory", "#fff", MEMORY_EDGE, GREEN))
    s.append(t(g2x + 16, g2y + 50, f"k-NN bank {m('B')}: {bank:,} × 768 (fp16)", 13.5))
    s.append(t(g2x + 16, g2y + 76, m("d = 1 − max<tspan baseline-shift='sub' font-size='10'>j</tspan> ẑ<tspan baseline-shift='super' font-size='10'>⊤</tspan>B<tspan baseline-shift='sub' font-size='10'>j</tspan>"), 15))
    s.append(t(g2x + 16, g2y + 98, f"nearest-neighbour cosine distance, k = {k}", 12.5, MUTED))

    # decision
    dx, dw = 1462, 130
    d1y, d2y = 222, 344
    s.append(arrow([(g1x + g1w, 262), (dx - 2, 262)]))
    s.append(box(dx, d1y, dw, 80, "#fff", INK, 10, 1.6))
    s.append(t(dx + dw / 2, d1y + 32, m("P(fish) ≥ τ₁ ?"), 16, INK, 400, "middle"))
    s.append(t(dx + dw / 2, d1y + 56, f"τ₁ = {t1:.3f}", 13, MUTED, 400, "middle"))
    s.append(arrow([(g2x + g2w, 384), (dx - 2, 384)]))
    s.append(box(dx, d2y, dw, 80, "#fff", INK, 10, 1.6))
    s.append(t(dx + dw / 2, d2y + 32, m("d ≤ τ₂ ?"), 16, INK, 400, "middle"))
    s.append(t(dx + dw / 2, d2y + 56, f"τ₂ = {t2:.3f}", 13, MUTED, 400, "middle"))
    s.append(arrow([(dx + dw / 2, d1y + 80), (dx + dw / 2, d2y - 2)]) + t(dx + dw / 2 + 8, d1y + 102, "yes", 12.5, GREEN, 700))
    # rejections
    px, pw = 1626, 152
    s.append(arrow([(dx + dw, 262), (px - 2, 262)], RED) + t(dx + dw + 5, 254, "no", 12.5, RED, 700))
    s.append(box(px, 244, pw, 36, RED_BG, RED, 18, 1.4) + t(px + pw / 2, 267, "rejected_not_fish", 12.5, RED, 700, "middle", "mono"))
    s.append(arrow([(dx + dw, 384), (px - 2, 384)], RED) + t(dx + dw + 6, 376, "no", 12.5, RED, 700))
    s.append(box(px, 360, pw, 48, RED_BG, RED, 18, 1.4)
             + t(px + pw / 2, 380, "rejected_", 12.5, RED, 700, "middle", "mono")
             + t(px + pw / 2, 397, "unknown_species", 12.5, RED, 700, "middle", "mono"))
    s.append(t(px + pw / 2, 434, "retake or manual species", 12, MUTED, 400, "middle"))
    # accept
    ay = 470
    s.append(arrow([(dx + dw / 2, d2y + 80), (dx + dw / 2, ay - 2)]) + t(dx + dw / 2 + 8, d2y + 100, "yes", 12.5, GREEN, 700))
    s.append(arrow([(sx + sw_, 136), (1444, 136), (1444, ay + 26), (dx - 14, ay + 26)], INK))
    s.append(box(dx - 10, ay, W - 22 - dx, 54, GREEN_BG, GREEN, 12, 1.6))
    s.append(t(dx + 6, ay + 22, "accepted: top-3 species + confidence", 13.5, INK, 700))
    s.append(t(dx + 6, ay + 42, "→ the operator confirms or corrects", 13, MUTED))

    # legend
    ly = 516
    lg = [(FROZEN, FROZEN_EDGE, "frozen, pre-trained"), (FITTED, FITTED_EDGE, "fitted on Fishora data"),
          (MEMORY, MEMORY_EDGE, "non-parametric memory"), (RED_BG, RED, "status returned by the API")]
    lx = 44
    for fill, edge, label in lg:
        s.append(box(lx, ly, 22, 16, fill, edge, 4, 1.4) + t(lx + 32, ly + 13, label, 13, MUTED))
        lx += 60 + 8 * len(label)
    s.append(t(940, ly + 13, "Both gates reuse the backbone's embedding: no second forward pass.", 13, MUTED))

    # ---------------- panel (b): offline fitting ----------------
    oy = 580
    s.append(box(16, oy, W - 32, 372, "#fbfcfd", "#e3e8ec", 14))
    s.append(t(40, oy + 36, "(b)", 20, INK, 800) + t(76, oy + 36, "Offline fitting: the backbone is never updated; every head is fitted on its cached embeddings", 19, INK, 700))

    def src(y, title, l1, l2):
        return (box(40, y, 380, 86, "#fff", LINE, 10, 1.4) + t(58, y + 26, title, 15, INK, 700)
                + t(58, y + 50, l1, 12.8, INK) + t(58, y + 70, l2, 12.5, MUTED))
    s.append(src(oy + 58, "Fishora dataset", "5,628 images · 11 species · 2 source booths", "split by specimen: train 3,932 · val 847 · test 849"))
    s.append(src(oy + 156, "Field photos", "258 screened iNaturalist photos · 9 species", "all 258 in the shipped model (cross-fitted for eval.)"))
    s.append(src(oy + 254, "Gate pool, leak-checked", "262 fish of 20 other taxa · 627 non-fish", "disjoint from every locked test set"))
    enc_y = oy + 150
    for yy in (oy + 101, oy + 199, oy + 297):
        s.append(arrow([(420, yy), (446, yy), (446, enc_y + 50), (470, enc_y + 50)], MUTED, 1.4, head=(yy == oy + 199)))
    s.append(box(474, enc_y, 170, 100, FROZEN, FROZEN_EDGE, 10, 1.6))
    s.append(t(559, enc_y + 30, "DINOv3 ViT-B/16", 14.5, INK, 700, "middle"))
    s.append(tag(529, enc_y + 42, "frozen", "#fff", FROZEN_EDGE, MUTED))
    s.append(t(559, enc_y + 86, m("h,  ẑ  (cached)"), 14, INK, 400, "middle"))

    cols = [
        ("Species head", FITTED, FITTED_EDGE, [
            "data: dataset train + field photos",
            "AdamW, lr 5·10⁻⁴, weight decay 0.05,",
            "   cosine schedule, ≤ 10 epochs, AMP",
            "label smoothing 0.1, √-inverse class weights",
            "early stop (patience 3) on val. macro-F1",
            f"temperature scaling on val. (LBFGS): T = {T:.3f}",
        ]),
        ("Gate 1 · fish vs. not fish", FITTED, FITTED_EDGE, [
            "logistic regression on ẑ",
            "positives: dataset train, field photos,",
            "   262 fish of other taxa (\"is a fish\", not \"ours\")",
            "negatives: 627 non-fish photos",
            "validation split by photographer / taxon / query",
            f"τ₁ = {t1:.3f}: keeps 99% of fish in every val. source",
        ]),
        ("Gate 2 · known vs. unknown species", MEMORY, MEMORY_EDGE, [
            f"bank: ẑ of the {bank:,} training photos",
            "   (dataset train + field photos), stored fp16",
            f"k = {k}, chosen by val. AUROC over k ∈ {{1, 3, 5, 10, 20}}",
            f"τ₂ = {t2:.3f}: 95th percentile of",
            "   leave-one-photographer-out field distances",
            "no gradient step: memory only",
        ]),
    ]
    cx0, cw, gap = 700, 346, 16
    for i, (title, fill, edge, lines) in enumerate(cols):
        x = cx0 + i * (cw + gap)
        s.append(box(x, oy + 58, cw, 282, "#fff", edge, 10, 1.6))
        s.append(f'<path d="M {x} {oy + 102} L {x} {oy + 68} Q {x} {oy + 58} {x + 10} {oy + 58} L {x + cw - 10} {oy + 58} '
                 f'Q {x + cw} {oy + 58} {x + cw} {oy + 68} L {x + cw} {oy + 102} Z" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
        s.append(t(x + 16, oy + 87, title, 15.5, INK, 700))
        yy = oy + 128
        for line in lines:
            cont = line.startswith("   ")
            if cont:
                yy -= 14
            else:
                s.append(f'<circle cx="{x + 20}" cy="{yy - 4}" r="2.6" fill="{edge}"/>')
            s.append(t(x + 30, yy, line.strip(), 13, INK))
            yy += 36
    trunk = 672
    s.append(arrow([(644, enc_y + 50), (trunk, enc_y + 50), (trunk, oy + 48)], MUTED, 1.4, head=False))
    for i in range(3):
        x = cx0 + i * (cw + gap)
        s.append(arrow([(trunk, oy + 48), (x + cw / 2, oy + 48), (x + cw / 2, oy + 56)], MUTED, 1.4))

    body = "".join(s)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0 L10 5 L0 10 z" fill="context-stroke"/></marker></defs>
<rect width="{W}" height="{H}" fill="#fff"/>{body}</svg>'''


def page(svg: str) -> str:
    return f'''<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;600;700;800&family=Geist+Mono:wght@600;700&family=STIX+Two+Text:ital@1&display=block" rel="stylesheet">
<style>@page {{ size: {W}px {H}px; margin: 0; }} html, body {{ margin: 0; background: #fff; }}
text {{ font-family: 'Geist', sans-serif; }} .m {{ font-family: 'STIX Two Text', serif; font-style: italic; }}
.mono {{ font-family: 'Geist Mono', monospace; }}</style></head><body>{svg}</body></html>'''


def main() -> None:
    cfg = json.loads((EXPORT / "inference_config.json").read_text())
    assert cfg.get("gate"), f"{EXPORT} is not a gated export"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = OUT_DIR / "_architecture.html"
    html.write_text(page(figure(cfg, photo_b64())))

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=4)
        pg.goto(html.as_uri(), wait_until="networkidle")
        pg.evaluate("document.fonts.ready")
        pg.screenshot(path=str(OUT_DIR / "architecture.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
        pg.pdf(path=str(OUT_DIR / "architecture.pdf"), width=f"{W}px", height=f"{H}px", print_background=True)
        b.close()
    html.unlink()
    print(f"wrote {OUT_DIR / 'architecture.pdf'} and architecture.png")


if __name__ == "__main__":
    main()
