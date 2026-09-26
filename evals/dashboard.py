"""Render the RAG quality dashboard from the Python-generated run artifacts.

    python -m evals.dashboard [--baseline baseline] [--current current]

Reads reports/<label>/{tests,rag_eval,cost_eval,model_compare,probes,cv_eval}.json,
evaluates the weakness registry (evals/findings.py) and writes:

    reports/dashboard.html            standalone page (open in a browser)
    reports/dashboard.artifact.html   the same content without the document
                                      skeleton, for publishing as a page

Every number on the page comes from those files. Nothing is read from the
markdown under evals/research/. Visual system: dark canvas, white display
type with tight tracking, charcoal surfaces, one blue signal colour, and two
gradient spotlight cards.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from html import escape

from evals.findings import evaluate, get, improvements, load_artifacts
from evals.run import REPORTS_DIR

TITLE = "Fishora RAG Quality"
FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@500;600&"
         "family=Inter:opsz,wght@14..32,400;14..32,500;14..32,700&display=swap")

CSS = """
:root{
  color-scheme:dark;
  --canvas:#090909; --surface-1:#141414; --surface-2:#1F1F1F;
  --hairline:#2A2A2A; --hairline-soft:#1C1C1C;
  --ink:#FFFFFF; --ink-muted:#999999; --on-primary:#090909;
  --accent:#0099FF; --ring:rgba(0,153,255,.35);
  --success:#34D17C; --magenta:#FF4FC3; --high:#FF6B7A; --medium:#FF9A3D; --low:#A58BFF;
  --grad-violet:radial-gradient(120% 140% at 12% 0%,#8B5CFF 0%,#4B21B8 38%,#1A0B3D 72%,#0E0820 100%);
  --grad-orange:radial-gradient(120% 140% at 88% 0%,#FF8A2A 0%,#C2410C 40%,#3A1206 75%,#1A0904 100%);
  --r-sm:6px; --r-md:10px; --r-lg:15px; --r-xl:20px; --r-xxl:30px; --r-pill:100px;
  --display:"Geist","Mona Sans","Inter",system-ui,sans-serif;
  --body:"Inter","Inter Variable",system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
body{background:var(--canvas);color:var(--ink);font:400 15px/1.3 var(--body);letter-spacing:-.15px;
  font-feature-settings:"cv01","cv05","cv09","cv11","ss03","ss07","tnum";margin:0}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
:focus-visible{outline:none;box-shadow:0 0 0 2px var(--ring);border-radius:var(--r-pill)}
.wrap{max-width:1199px;margin:0 auto;padding-inline:max(16px,min(30px,4vw));padding-block:0 96px;display:flex;flex-direction:column;gap:96px}
nav.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:rgba(9,9,9,.86);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--hairline-soft)}
nav.top .in{max-width:1199px;margin:0 auto;padding-inline:max(16px,min(30px,4vw));min-height:56px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding-block:8px}
.brand{font:600 18px/1 var(--display);letter-spacing:-.6px;margin-right:auto}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;min-height:36px;padding:10px 15px;border-radius:var(--r-pill);
  background:var(--surface-1);color:var(--ink);font:500 14px/1 var(--body);letter-spacing:-.14px;white-space:nowrap}
.pill:hover{background:var(--surface-2);text-decoration:none}
.pill.primary{background:var(--ink);color:var(--on-primary)}
h1,h2,h3{font-family:var(--display);font-weight:500;margin:0;text-wrap:balance}
h1{font-size:clamp(40px,7.2vw,85px);line-height:.95;letter-spacing:-.05em}
h2{font-size:clamp(30px,4.6vw,62px);line-height:1;letter-spacing:-.05em}
h3{font-size:22px;line-height:1.2;letter-spacing:-.8px;font-family:var(--body);font-weight:700}
.lead{font-size:18px;line-height:1.3;letter-spacing:-.18px;color:var(--ink-muted);max-width:62ch;margin:0}
.muted{color:var(--ink-muted)} .cap{font:500 13px/1.2 var(--body);letter-spacing:-.13px;color:var(--ink-muted)}
header.hero{display:flex;flex-direction:column;gap:22px;padding-top:72px}
.meta{display:flex;flex-wrap:wrap;gap:8px}
.tag{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:var(--r-pill);background:var(--surface-1);
  font:500 13px/1 var(--body);color:var(--ink-muted)} .tag b{color:var(--ink);font-weight:500}
section{display:flex;flex-direction:column;gap:30px;scroll-margin-top:72px}
.head{display:flex;flex-direction:column;gap:10px}
.grid{display:grid;gap:15px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))} .g3{grid-template-columns:repeat(3,minmax(0,1fr))} .g4{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:810px){.g2,.g3,.g4{grid-template-columns:1fr}}
.card{background:var(--surface-1);border-radius:var(--r-xl);padding:20px;display:flex;flex-direction:column;gap:15px;min-width:0}
.card.lift{background:var(--surface-2)}
.spot{border-radius:var(--r-xxl);padding:30px;min-height:240px;display:flex;flex-direction:column;justify-content:space-between;gap:30px;overflow:hidden}
.spot.violet{background:var(--grad-violet)} .spot.orange{background:var(--grad-orange)}
.spot .big{font:500 clamp(56px,8vw,110px)/.85 var(--display);letter-spacing:-.05em}
.spot .sub{font-size:24px;line-height:1.3;letter-spacing:-.01px}
.spot .row{display:flex;gap:10px;flex-wrap:wrap}
.spot .chip{background:rgba(255,255,255,.14);color:var(--ink)}
.kpi{display:flex;flex-direction:column;gap:6px}
.kpi .v{font:500 32px/1.13 var(--display);letter-spacing:-1px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:var(--r-pill);font:500 13px/1 var(--body);
  background:var(--surface-2);color:var(--ink-muted);white-space:nowrap}
.chip::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.chip.high{color:var(--high)} .chip.medium{color:var(--medium)} .chip.low{color:var(--low)} .chip.info{color:var(--ink-muted)}
.chip.open{color:var(--ink)} .chip.open::before{background:var(--high)}
.chip.partial{color:var(--ink)} .chip.partial::before{background:var(--medium)}
.chip.not_measured,.chip.not_observed{color:var(--ink-muted)}
.chip.resolved,.chip.met{color:var(--success)} .chip.miss{color:var(--ink)} .chip.miss::before{background:var(--medium)}
.chip.nochip::before{display:none}
.list{display:flex;flex-direction:column}
.row-f{display:grid;grid-template-columns:52px minmax(0,1fr) auto auto;gap:15px;align-items:start;padding:15px 0;border-bottom:1px solid var(--hairline-soft)}
.row-f:last-child{border-bottom:0}
.row-f .id{font:500 14px/1.4 var(--body);color:var(--ink-muted)}
.row-f .t{font:500 15px/1.3 var(--body)} .row-f .e{font-size:13px;color:var(--ink-muted);margin-top:4px}
details{margin-top:6px} summary{cursor:pointer;color:var(--ink-muted);font-size:13px;list-style:none} summary::-webkit-details-marker{display:none}
summary::after{content:" +"} details[open] summary::after{content:" −"}
details ul{margin:6px 0 0;padding-left:16px;color:var(--ink-muted);font-size:13px;display:flex;flex-direction:column;gap:3px}
@media (max-width:640px){.row-f{grid-template-columns:40px minmax(0,1fr)}.row-f .chips{grid-column:2}}
.chips{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.chart{width:100%;height:auto;display:block}
.chart text{fill:var(--ink-muted);font:500 12px var(--body)}
.chart .lbl{fill:var(--ink);font-size:13px}
.chart .grid{stroke:var(--hairline-soft);stroke-width:1}
.chart .axis{stroke:var(--hairline);stroke-width:1}
.chart .quad{fill:rgba(52,209,124,.10)} .chart .quad2{fill:rgba(255,255,255,.025)}
.chart .tick{stroke:var(--hairline);stroke-width:1} .chart .axl{fill:var(--ink);font-size:13px}
.chart .pline{fill:none;stroke:var(--ink);stroke-width:2.2;stroke-dasharray:0.5 6;stroke-linecap:round}
.chart .fam{fill:none;stroke:#BDBDBD;stroke-width:2}
.chart .lbl.dim{fill:var(--ink-muted);opacity:.6}
.sw.sq{border-radius:3px;width:14px;height:10px}
.ln{display:inline-block;width:22px;margin-right:6px;vertical-align:3px;border-top:2px solid #BDBDBD}
.ln.dot{border-top:2px dotted var(--ink)}
.chart .dom{fill:var(--surface-2);stroke:var(--ink-muted);stroke-width:1.2}
.chart .opt{fill:var(--ink)}
.chart .star{fill:var(--ink);stroke:var(--accent);stroke-width:3;paint-order:stroke}
.chart .track{fill:var(--surface-2)} .chart .base{fill:var(--ink)} .chart .tgt{stroke:var(--medium);stroke-width:2}
.chart .cur{fill:var(--success)} .chart .bar{fill:var(--ink)} .chart .bar2{fill:var(--ink-muted)}
.chart .pass{fill:var(--success)} .chart .fail{fill:var(--high)} .chart .skip{fill:var(--hairline)}
.chart .pos{fill:var(--ink)} .chart .neg{fill:var(--high);opacity:.8}
.legend{display:flex;gap:15px;flex-wrap:wrap;font:500 13px/1.2 var(--body);color:var(--ink-muted)}
.sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0}
.rank{display:flex;flex-direction:column}
.rank .r{display:grid;grid-template-columns:28px minmax(0,1fr) repeat(3,minmax(60px,auto));gap:12px;align-items:center;
  padding:12px 0;border-bottom:1px solid var(--hairline-soft);font-size:14px}
.rank .r:last-child{border-bottom:0}
.rank .r.top .n{color:var(--ink);font-weight:500} .rank .r .n{color:var(--ink-muted)}
.rank .num{text-align:right;color:var(--ink-muted)} .rank .r.top .num{color:var(--ink)}
.rank .hd{font:500 12px/1 var(--body);color:var(--ink-muted);padding-bottom:10px}
.conf{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;align-items:stretch}
.conf .c{border-radius:var(--r-md);padding:15px;display:flex;flex-direction:column;gap:4px}
.conf .c .v{font:500 32px/1 var(--display);letter-spacing:-1px}
.conf .ok{background:var(--surface-2)} .conf .bad{background:var(--surface-2);box-shadow:inset 0 0 0 1px rgba(255,107,122,.45)}
.conf .ax{font:500 12px/1.2 var(--body);color:var(--ink-muted);display:flex;align-items:center}
.scroll{overflow-x:auto}
footer{display:flex;flex-direction:column;gap:8px;padding-top:30px;border-top:1px solid var(--hairline-soft);font:500 13px/1.4 var(--body);color:var(--ink-muted)}
code{font:500 12px var(--body);background:var(--surface-1);padding:2px 6px;border-radius:var(--r-sm);color:var(--ink)}
@media (prefers-reduced-motion:no-preference){.pill,.chip{transition:background .15s}}
"""


# ---------------------------------------------------------------- formatting

def fmt(value, kind: str = "auto") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "usd":
        return f"${value:.5f}".rstrip("0").rstrip(".") if value else "$0"
    if kind == "s":
        return f"{value:.1f} s"
    if kind == "ms":
        return f"{value:.0f} ms"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return escape(str(value))


def usd_plain(t: float | None) -> str:
    """Plain dollars with two significant digits: $0.00033, $0.0051, $0.025."""
    if not t:
        return "$0"
    from decimal import ROUND_HALF_UP, Decimal
    decimals = max(2, -math.floor(math.log10(abs(t))) + 1)
    text = str(Decimal(repr(t)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))
    return "$" + (text.rstrip("0").rstrip(".") if "." in text else text)


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls or text}">{escape(text.replace("_", " "))}</span>'


# ---------------------------------------------------------------- charts

def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for exp in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            t = m * 10 ** exp
            if lo <= t <= hi:
                ticks.append(t)
    return ticks


def _lin_ticks(hi: float, n: int = 4) -> list[float]:
    step = hi / n
    mag = 10 ** math.floor(math.log10(step))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= step), default=step)
    return [i * step for i in range(int(hi / step) + 1)]


PROVIDERS = (  # (name prefix, provider, colour token)
    ("gpt-", "OpenAI (Luna)", "var(--ink)"),
    ("deepseek", "DeepSeek", "var(--low)"),
    ("glm", "Z AI", "var(--medium)"),
    ("kimi", "Moonshot", "var(--magenta)"),
    ("qwen", "Alibaba", "var(--high)"),
)


def provider(name: str) -> tuple[str, str]:
    for prefix, label, colour in PROVIDERS:
        if name.startswith(prefix):
            return label, colour
    return "Other", "var(--ink-muted)"


def pareto(points: list[dict], *, x_label: str, y_label: str, x_fmt, y_fmt,
           y_higher: bool = True, y_rate: bool = False, x_log: bool = False,
           quadrant: tuple[float, float] | None = None, wide: bool = False,
           colour_of=None) -> str:
    """Cost/quality scatter: shaded most-attractive quadrant, dotted Pareto
    line through the non-dominated models, a solid line joining the Luna
    family, provider-coloured points, and everything off the frontier faded.
    x: lower is better."""
    pts = [p for p in points if p["x"] is not None and p["y"] is not None and (p["x"] > 0 or not x_log)]
    W, H = (1100, 440) if wide else (400, 320)
    L, R, T, B = (64, 28, 22, 52) if wide else (62, 14, 22, 50)
    if not pts:
        return '<p class="cap">No data.</p>'
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    if x_log:
        lo, hi = min(xs) / 2.2, max(xs) * 2.2
        tx = lambda v: L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)
        xticks = _log_ticks(lo, hi)
    else:
        lo, hi = 0, (max(xs) * 1.25 or 1)
        tx = lambda v: L + v / hi * (W - L - R)
        xticks = _lin_ticks(hi)
    y_hi = 1.05 if y_rate else (max(ys) * 1.25 or 1)
    ty = lambda v: H - B - v / y_hi * (H - T - B)
    yticks = [0, .25, .5, .75, 1.0] if y_rate else _lin_ticks(y_hi)

    better = (lambda a, b: a > b) if y_higher else (lambda a, b: a < b)
    front, best = [], None
    for p in sorted(pts, key=lambda p: (p["x"], -p["y"] if y_higher else p["y"])):
        if best is None or better(p["y"], best):
            front.append(p)
            best = p["y"]
    on_front = {id(p) for p in front}

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(y_label)} against {escape(x_label)}">']
    if quadrant and quadrant[0]:
        qx, qy = tx(min(max(quadrant[0], lo if x_log else 0.0), hi)), ty(min(quadrant[1], y_hi))
        if y_higher:
            parts.append(f'<rect class="quad" x="{L}" y="{T}" width="{qx - L:.1f}" height="{qy - T:.1f}"/>'
                         f'<rect class="quad2" x="{qx:.1f}" y="{qy:.1f}" width="{W - R - qx:.1f}" height="{H - B - qy:.1f}"/>')
        else:
            parts.append(f'<rect class="quad" x="{L}" y="{qy:.1f}" width="{qx - L:.1f}" height="{H - B - qy:.1f}"/>'
                         f'<rect class="quad2" x="{qx:.1f}" y="{T}" width="{W - R - qx:.1f}" height="{qy - T:.1f}"/>')
    for t in yticks:
        parts.append(f'<line class="tick" x1="{L - 6}" x2="{L}" y1="{ty(t):.1f}" y2="{ty(t):.1f}"/>'
                     f'<text x="{L - 10}" y="{ty(t) + 4:.1f}" text-anchor="end">{y_fmt(t)}</text>')
    for t in xticks:
        parts.append(f'<line class="tick" y1="{H - B}" y2="{H - B + 6}" x1="{tx(t):.1f}" x2="{tx(t):.1f}"/>'
                     f'<text x="{tx(t):.1f}" y="{H - B + 22}" text-anchor="middle">{x_fmt(t)}</text>')
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle" class="axl">{escape(x_label)}</text>')
    parts.append(f'<text transform="translate(16 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" class="axl">{escape(y_label)}</text>')

    # Dotted Pareto line; a lone frontier point extends flat to the right edge.
    line = [(tx(p["x"]), ty(p["y"])) for p in front]
    if len(line) == 1:
        line.append((W - R, line[0][1]))
    parts.append('<polyline class="pline" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in line) + '"/>')
    family = sorted((p for p in pts if p.get("star")), key=lambda p: p["x"])
    if len(family) > 1:
        parts.append('<polyline class="fam" points="' + " ".join(f"{tx(p['x']):.1f},{ty(p['y']):.1f}" for p in family) + '"/>')

    placed: list[tuple[float, float, float, float]] = []
    order = sorted(pts, key=lambda p: (not p.get("star"), id(p) not in on_front))
    for p in order:
        cx, cy = tx(p["x"]), ty(p["y"])
        colour = (colour_of or (lambda n: provider(n)[1]))(p["name"])
        strong = bool(p.get("star")) or id(p) in on_front
        op = 1 if strong else .38
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{8 if p.get("star") else 7}" style="fill:{colour};opacity:{op}"/>')
        w = 7.4 * len(p["name"])
        for dx, dy, anchor in ((10, -14, "start"), (10, 24, "start"), (-10, -14, "end"), (-10, 24, "end"), (14, 5, "start"), (-14, 5, "end")):
            x0 = cx + dx if anchor == "start" else cx + dx - w
            box = (x0, cy + dy - 12, x0 + w, cy + dy + 3)
            inside = box[0] >= L + 2 and box[2] <= W - 4 and box[1] >= 2 and box[3] <= H - B - 2
            clash = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            if inside and not clash:
                placed.append(box)
                cls = "lbl" if strong else "lbl dim"
                parts.append(f'<text class="{cls}" x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}">{escape(p["name"])}</text>')
                break
    parts.append("</svg>")
    return "".join(parts)


def pareto_legend(names: list[str], family: str = "Luna family", group=None) -> str:
    seen, dots = set(), []
    for n in names:
        label, colour = (group or provider)(n)
        if label not in seen:
            seen.add(label)
            dots.append(f'<span><span class="sw" style="background:{colour}"></span>{escape(label)}</span>')
    return ('<div class="legend"><span><span class="sw sq" style="background:rgba(52,209,124,.22)"></span>Most attractive quadrant</span>'
            f'<span><span class="ln solid"></span>{escape(family)}</span><span><span class="ln dot"></span>Pareto line</span></div>'
            f'<div class="legend">{"".join(dots)}</div>')


def hbars(rows: list[tuple[str, float | None, str]], hi: float = 1.0, cls: str = "bar") -> str:
    W, row_h, L, R = 560, 34, 190, 70
    H = row_h * len(rows) + 6
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img">']
    for i, (label, value, shown) in enumerate(rows):
        y = i * row_h + 8
        width = 0 if not value or not hi else max(0.0, min(1.0, value / hi)) * (W - L - R)
        out.append(f'<text x="0" y="{y + 12}">{escape(label)}</text>'
                   f'<rect class="track" x="{L}" y="{y + 2}" width="{W - L - R}" height="12" rx="6"/>'
                   f'<rect class="{cls}" x="{L}" y="{y + 2}" width="{width:.1f}" height="12" rx="6"/>'
                   f'<text class="lbl" x="{W}" y="{y + 12}" text-anchor="end">{escape(shown)}</text>')
    out.append("</svg>")
    return "".join(out)


def pairbars(rows: list[tuple[str, float, float]], a: str, b: str) -> str:
    W, row_h, L, R = 560, 40, 110, 60
    H = row_h * len(rows) + 6
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img">']
    for i, (label, va, vb) in enumerate(rows):
        y = i * row_h + 6
        span = W - L - R
        out.append(f'<text x="0" y="{y + 15}">{escape(label)}</text>'
                   f'<rect class="track" x="{L}" y="{y}" width="{span}" height="10" rx="5"/>'
                   f'<rect class="bar" x="{L}" y="{y}" width="{span * (va or 0):.1f}" height="10" rx="5"/>'
                   f'<rect class="track" x="{L}" y="{y + 14}" width="{span}" height="10" rx="5"/>'
                   f'<rect class="bar2" x="{L}" y="{y + 14}" width="{span * (vb or 0):.1f}" height="10" rx="5"/>'
                   f'<text class="lbl" x="{W}" y="{y + 10}" text-anchor="end">{fmt(va)}</text>'
                   f'<text x="{W}" y="{y + 24}" text-anchor="end">{fmt(vb)}</text>')
    out.append("</svg>")
    return (''.join(out) + f'<div class="legend"><span><span class="sw" style="background:var(--ink)"></span>{escape(a)}</span>'
            f'<span><span class="sw" style="background:var(--ink-muted)"></span>{escape(b)}</span></div>')


def bullets(rows: list[dict], has_current: bool) -> str:
    """Baseline dot, target tick, current dot, each metric on its own axis."""
    W, row_h, L, R = 1100, 46, 300, 150
    H = row_h * len(rows) + 8
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Baseline against target per metric">']
    for i, r in enumerate(rows):
        y = i * row_h + 18
        vals = [v for v in (r["baseline"], r["target"], r["current"]) if isinstance(v, (int, float))]
        top = 1.0 if r["rate"] else (max(vals) * 1.15 if vals else 1)
        sx = lambda v: L + min(1.0, max(0.0, v / top)) * (W - L - R)
        out.append(f'<text x="0" y="{y + 5}" class="lbl">{escape(r["metric"])}</text>'
                   f'<text x="0" y="{y + 21}">{escape(" ".join(r["findings"]))}</text>'
                   f'<rect class="track" x="{L}" y="{y - 3}" width="{W - L - R}" height="8" rx="4"/>')
        if isinstance(r["target"], (int, float)):
            out.append(f'<line class="tgt" x1="{sx(r["target"]):.1f}" x2="{sx(r["target"]):.1f}" y1="{y - 9}" y2="{y + 11}"/>')
        if isinstance(r["baseline"], (int, float)):
            out.append(f'<circle class="base" cx="{sx(r["baseline"]):.1f}" cy="{y + 1}" r="6"/>')
        if has_current and isinstance(r["current"], (int, float)):
            out.append(f'<circle class="cur" cx="{sx(r["current"]):.1f}" cy="{y + 1}" r="6"/>')
        text = f'{r["b_txt"]} → {r["t_txt"]}' + (f' · now {r["c_txt"]}' if has_current and r["current"] is not None else "")
        out.append(f'<text x="{W}" y="{y + 5}" text-anchor="end" class="lbl">{escape(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def histogram(sim: dict) -> str:
    h = sim["histogram"]
    pos, neg, edges = h["positive"], h["negative"], h["bins"]
    W, H, L, B = 560, 220, 34, 30
    top = max(max(pos), max(neg), 1)
    bw = (W - L - 8) / len(pos)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Claim to chunk similarity">']
    for t in (0, top // 2, top):
        yy = H - B - (H - B - 12) * t / top
        out.append(f'<line class="grid" x1="{L}" x2="{W - 8}" y1="{yy:.1f}" y2="{yy:.1f}"/><text x="{L - 6}" y="{yy + 4:.1f}" text-anchor="end">{t}</text>')
    for i, (p, n) in enumerate(zip(pos, neg)):
        x = L + i * bw
        for j, (count, cls) in enumerate(((p, "pos"), (n, "neg"))):
            bh = (H - B - 12) * count / top
            out.append(f'<rect class="{cls}" x="{x + j * bw / 2 + 1:.1f}" y="{H - B - bh:.1f}" width="{bw / 2 - 2:.1f}" height="{bh:.1f}" rx="2"/>')
    for i in range(0, len(edges), 5):
        out.append(f'<text x="{L + i * bw:.1f}" y="{H - 10}" text-anchor="middle">{edges[i]:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_layers(layers: dict) -> str:
    W, row_h, L, R = 560, 34, 110, 70
    H = row_h * len(layers) + 6
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Tests by layer">']
    for i, (name, v) in enumerate(layers.items()):
        y = i * row_h + 8
        total = max(1, v.get("total", 0))
        span = W - L - R
        x = L
        for key, cls in (("passed", "pass"), ("failed", "fail"), ("error", "fail"), ("skipped", "skip")):
            w = span * v.get(key, 0) / total
            if w:
                out.append(f'<rect class="{cls}" x="{x:.1f}" y="{y + 2}" width="{w:.1f}" height="12"/>')
                x += w
        out.append(f'<text x="0" y="{y + 12}">{escape(name)}</text>'
                   f'<text class="lbl" x="{W}" y="{y + 12}" text-anchor="end">{v.get("passed", 0)}/{v.get("total", 0)}</text>')
    out.append("</svg>")
    return "".join(out)


def confusion(title: str, rows: tuple[str, str], cols: tuple[str, str], cells: list[list[tuple[int, str, bool]]]) -> str:
    """2x2 matrix; each cell is (count, caption, is_error). Error cells get a coral edge."""
    total = sum(c[0] for r in cells for c in r) or 1
    out = [f'<div class="conf"><span class="ax">{escape(title)}</span><span class="ax">{escape(cols[0])}</span><span class="ax">{escape(cols[1])}</span>']
    for label, row in zip(rows, cells):
        out.append(f'<span class="ax">{escape(label)}</span>')
        for count, caption, bad in row:
            out.append(f'<div class="c {"bad" if bad else "ok"}"><span class="v">{count}</span>'
                       f'<span class="cap">{escape(caption)} · {count / total:.0%}</span></div>')
    out.append("</div>")
    return "".join(out)


# Card field -> evidence categories that can support it (as routed to the experts).
FIELD_EVIDENCE = {
    "physical_characteristics": {"physical_characteristics", "identity"},
    "taste": {"taste_texture"}, "texture": {"taste_texture"},
    "processing_methods": {"processing_methods"}, "commercial_uses": {"commercial_uses"},
    "similar_or_substitute_species": {"substitutes"}, "potential_buyer_segments": {"commercial_uses"},
}


def abstention_cells(cards: list[dict]) -> list[list[tuple[int, str, bool]]] | None:
    """Real-LLM cards: is the field filled, and does the species have evidence for it?"""
    from evals.corpus import load_corpus
    cats: dict[str, set[str]] = {}
    for c in load_corpus():
        cats.setdefault(c.species_label, set()).add(c.category)
    n = {"fe": 0, "fn": 0, "ee": 0, "en": 0}
    seen = False
    for card in cards:
        body = card.get("card")
        if not body:
            continue
        seen = True
        for field, needed in FIELD_EVIDENCE.items():
            has_ev = bool(needed & cats.get(card["species"], set()))
            filled = bool(body.get(field))
            n[("f" if filled else "e") + ("e" if has_ev else "n")] += 1
    if not seen:
        return None
    return [[(n["fe"], "answered", False), (n["ee"], "omitted", True)],
            [(n["fn"], "filled anyway", True), (n["en"], "abstained", False)]]


ARCH = (  # CV run prefix -> (architecture, colour)
    ("lp_vit", "ViT (DINOv3)", "var(--ink)"),
    ("lp_convnext", "ConvNeXt (DINOv3)", "var(--low)"),
    ("lp_efficientnet", "EfficientNet", "var(--medium)"),
    ("lp_mobilenet", "MobileNet", "var(--magenta)"),
)


def arch(run: str) -> tuple[str, str]:
    for prefix, label, colour in ARCH:
        if run.startswith(prefix):
            return label, colour
    return "Other", "var(--ink-muted)"


def heatmap(models: dict, slices: list[str]) -> str:
    """Models x slices accuracy grid; cell brightness = accuracy, red edge below 90%."""
    runs = sorted(models, key=lambda r: -models[r]["field_accuracy"])
    cw, ch, L, T = 64, 30, 150, 78
    W, H = L + cw * len(slices) + 4, T + ch * len(runs) + 4
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Accuracy per model and image slice">']
    for j, name in enumerate(slices):
        x = L + j * cw + cw / 2
        out.append(f'<text x="{x - 8:.0f}" y="{T - 10}" text-anchor="start" transform="rotate(-35 {x - 8:.0f} {T - 10})">{escape(name.replace("_", " "))}</text>')
    for i, run in enumerate(runs):
        y = T + i * ch
        weight = "lbl" if models[run]["production"] else ""
        out.append(f'<text class="{weight}" x="{L - 10}" y="{y + ch / 2 + 4:.0f}" text-anchor="end">{escape(run.removeprefix("lp_"))}</text>')
        for j, name in enumerate(slices):
            v = models[run]["slice_accuracy"].get(name)
            if v is None:
                continue
            x = L + j * cw
            edge = ' style="stroke:var(--high);stroke-width:1.5"' if v < 0.9 else ""
            out.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="5" '
                       f'fill="rgba(255,255,255,{0.04 + 0.5 * v ** 3:.3f})"{edge}/>'
                       f'<text x="{x + cw / 2:.0f}" y="{y + ch / 2 + 4:.0f}" text-anchor="middle" '
                       f'style="fill:{"var(--canvas)" if v >= 0.9 else "var(--ink)"};font-weight:500">{v:.0%}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- page

def _kind(path: str) -> str:
    if "cost_usd" in path:
        return "usd"
    if path.endswith("_ms_mean"):
        return "ms"
    if any(k in path for k in ("success_rate", "retention", "totals", "false_support")):
        return "pct"
    return "auto"


def render(baseline: str, current: str) -> tuple[str, str]:
    has_current = (REPORTS_DIR / current / "rag_eval.json").exists()
    base_a = load_artifacts(baseline)
    cur_a = load_artifacts(current) if has_current else {}
    # Latest data per artifact: the current run where it exists, else the baseline.
    a = {k: (cur_a.get(k) if cur_a.get(k) is not None else base_a[k]) for k in base_a}
    origin = {k: (current if cur_a.get(k) is not None else baseline) for k in base_a if a[k] is not None}
    base_findings = {f["id"]: f for f in evaluate(baseline)}
    findings = evaluate(current) if has_current else list(base_findings.values())
    tests, ev, cost, comp = a["tests"] or {}, a["rag_eval"] or {}, a["cost_eval"] or {}, a["model_compare"] or {}
    totals = tests.get("totals", {})
    by_path = get(cost, "summary.by_path") or {}
    # The card the product publishes: a separate one-call path until iteration 1,
    # the graded agent card since (W19).
    pub = "published" if "published" in by_path else "agent"
    n_high = sum(f["status"] in ("open", "partial") and f["severity"] == "high" for f in findings)
    n_med = sum(f["status"] in ("open", "partial") and f["severity"] == "medium" for f in findings)
    commit = get(ev, "git.commit") or "?"

    nav = ('<nav class="top"><div class="in"><span class="brand">Fishora</span><div class="pills">'
           + "".join(f'<a class="pill" href="#{i}">{t}</a>' for i, t in (
               ("weaknesses", "Weaknesses"), ("targets", "Targets"), ("models", "Models"),
               ("cost", "Cost"), ("quality", "Quality"), ("species", "Species ID"), ("tests", "Tests")))
           + f'<span class="pill primary">Baseline · {escape(commit)}</span></div></div></nav>')

    hero = f"""
<header class="hero">
  <h1>Fishora,<br>measured.</h1>
  <p class="lead">11 species · 49 evidence chunks · E5 retrieval · {escape(cost.get("model", "gpt-5.6-luna"))} generation · DINOv3 species identification.</p>
  <div class="meta"><span class="tag">Evals <b>{escape(get(ev, "generated_at", "?")[:16].replace("T", " "))}</b></span>
  <span class="tag">Real-LLM run <b>{escape(cost.get("generated_at", "not run")[:16].replace("T", " "))}</b></span>
  <span class="tag">Built <b>{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC</b></span></div>
</header>
<div class="grid g2">
  <div class="spot violet"><span class="cap" style="color:#fff">Test suite</span>
    <span class="big">{totals.get("passed", 0)}<span style="opacity:.55">/{totals.get("total", 0)}</span></span>
    <div class="row"><span class="chip nochip">{n_high} high open</span><span class="chip nochip">{n_med} medium open</span>
    <span class="chip nochip">grounding F1 {fmt(get(ev, "grounding.test.f1"))}</span></div></div>
  <div class="spot orange"><span class="cap" style="color:#fff">Per published card</span>
    <span class="big">{fmt(get(by_path, f"{pub}.cost_usd_per_card.mean"), "usd")}</span>
    <div class="row"><span class="chip nochip">{fmt(get(by_path, f"{pub}.wall_s_per_card.p50"), "s")} p50</span>
    <span class="chip nochip">operator cards {fmt(get(by_path, "agent.success_rate"), "pct")} complete</span></div></div>
</div>"""

    # Weaknesses
    rows = []
    for f in findings:
        first, rest = (f["evidence"][0] if f["evidence"] else ""), f["evidence"][1:]
        more = (f'<details><summary>evidence</summary><ul>{"".join(f"<li>{escape(e)}</li>" for e in rest)}</ul></details>'
                if rest else "")
        rows.append(f'<div class="row-f"><span class="id">{f["id"]}</span><div><div class="t">{escape(f["title"])}</div>'
                    f'<div class="e">{escape(first)}</div>{more}</div>'
                    f'<div class="chips">{chip(f["severity"])}</div><div class="chips">'
                    + (f'{chip(base_findings[f["id"]]["status"])}<span class="cap">&rarr;</span>'
                       if has_current and f["id"] in base_findings and base_findings[f["id"]]["status"] != f["status"] else "")
                    + f'{chip(f["status"])}</div></div>')
    weaknesses = f"""
<section id="weaknesses"><div class="head"><h2>Weaknesses</h2>
<p class="cap">{len(findings)} checks computed from the {"current run, with the baseline status where it changed" if has_current else "baseline run"} · open first</p></div>
<div class="card"><div class="list">{"".join(rows)}</div></div></section>"""

    # Targets
    trows = []
    for r in improvements(baseline, current):
        kind = _kind(r["path"])
        rate = kind == "pct" or any(k in r["path"] for k in ("mrr", "recall", "f1"))
        trows.append({**r, "rate": bool(rate), "b_txt": fmt(r["baseline"], kind), "t_txt": fmt(r["target"], kind),
                      "c_txt": fmt(r["current"], kind)})
    trows = [r for r in trows if r["target"] is not None]
    targets = f"""
<section id="targets"><div class="head"><h2>Baseline → target</h2>
<div class="legend"><span><span class="sw" style="background:var(--ink)"></span>baseline</span>
<span><span class="sw" style="background:var(--medium);border-radius:1px;width:3px"></span>target</span>
<span><span class="sw" style="background:var(--success)"></span>current{"" if has_current else " (after the fix run)"}</span></div></div>
<div class="card scroll">{bullets(trows, has_current)}</div></section>"""

    # Models: one Pareto chart per comparison
    models = comp.get("models") or {}
    def valid_p50(model):
        # Latency of cards that passed; failed calls end early (400s) or at the timeout and would distort it.
        times = sorted(c["llm_time_s"] for c in comp.get("cards", []) if c.get("model") == model and c["status"] == "completed")
        return times[len(times) // 2] if times else None
    mp = [{"name": m, "star": "luna" in m, "valid": v["valid_card_rate"], "cpv": v["cost_usd_per_valid_card"],
           "cpc": v["cost_usd_per_card"].get("mean") if v["valid_card_rate"] else None, "lat": valid_p50(m)}
          for m, v in models.items()]
    usd = usd_plain
    pct = lambda t: f"{t:.0%}"
    secs = lambda t: f"{t:g}s"

    def geo(vals):
        vals = [v for v in vals if v]
        return math.sqrt(min(vals) * max(vals)) if vals else None

    main_chart = pareto([{**p, "x": p["cpv"], "y": p["valid"]} for p in mp],
                        x_label="USD per valid card (log scale)", y_label="Valid cards",
                        x_fmt=usd, y_fmt=pct, y_rate=True, x_log=True, wide=True,
                        quadrant=(geo([p["cpv"] for p in mp]) or 0, 0.9))
    small = [
        ("Valid cards vs latency", pareto([{**p, "x": p["lat"], "y": p["valid"]} for p in mp],
            x_label="p50 latency of valid cards (s, log)", y_label="Valid cards", x_fmt=secs, y_fmt=pct, y_rate=True, x_log=True,
            quadrant=(10, 0.9))),
        ("Cost vs latency", pareto([{**p, "x": p["lat"], "y": p["cpc"]} for p in mp],
            x_label="p50 latency of valid cards (s, log)", y_label="USD per card", x_fmt=secs, y_fmt=usd, y_higher=False, x_log=True,
            quadrant=(10, geo([p["cpc"] for p in mp]) or 0))),
    ]
    path_names = {"published": "published", "one_call": "one-call (reference)", "agent": "graded (shipped)",
                  "agent_normalized": "operator (fixed)"}
    paths = [{"name": path_names[k], "x": v["cost_usd_per_card"].get("mean"), "y": v["success_rate"], "star": False}
             for k, v in by_path.items()]
    small.append(("Card paths: completed vs cost", pareto(paths, x_label="USD per card", y_label="Cards completed",
                  x_fmt=usd, y_fmt=pct, y_rate=True, quadrant=(geo([p["x"] for p in paths]) or 0, 0.9))))
    unplotted = [p["name"] for p in mp if not p["cpv"]]  # no valid card: absent from every model chart
    ranked = sorted(mp, key=lambda p: (not p["star"], -(p["valid"] or 0), p["cpv"] if p["cpv"] is not None else 9e9))
    rank_rows = "".join(
        f'<div class="r{" top" if p["star"] else ""}"><span class="n">{i}</span>'
        f'<span class="n"><span class="sw" style="background:{provider(p["name"])[1]}"></span>{escape(p["name"])}</span>'
        f'<span class="num">{fmt(p["valid"], "pct")}</span><span class="num">{usd(p["cpv"]) if p["cpv"] else "n/a"}</span>'
        f'<span class="num">{fmt(p["lat"], "s")}</span></div>' for i, p in enumerate(ranked, 1))
    unplotted_note = (f'<span class="cap">Not plotted (no valid card): {escape(", ".join(unplotted))}</span>'
                      if unplotted else "")
    small_cards = "".join(f'<div class="card"><span class="cap">{escape(t)}</span>{c}</div>' for t, c in small)
    model_sec = f"""
<section id="models"><div class="head"><h2>Models</h2>{pareto_legend([p["name"] for p in mp])}</div>
<div class="card"><span class="cap">Valid cards vs cost per valid card</span>{main_chart}{unplotted_note}</div>
<div class="grid g3">{small_cards}</div>
<div class="card lift"><h3>Ranking</h3><div class="rank"><div class="r hd"><span></span><span>model</span><span class="num">valid</span>
<span class="num">per valid card</span><span class="num">p50 (valid)</span></div>{rank_rows}</div></div>
</section>"""

    # Cost
    stages = get(cost, "summary.by_stage") or {}
    st_rows = [(k.split(":", 1)[1].replace("_", " "), v["cost_usd_total"] / max(1, v["calls"]),
                f'{usd(v["cost_usd_total"] / max(1, v["calls"]))} · {v["latency_s"].get("p50", 0):.1f}s')
               for k, v in stages.items() if k.startswith(("agent_normalized", "published", "one_call", "agent:"))]
    top_stage = max((r[1] for r in st_rows), default=1)
    share = [(k, v["output_share_of_cost"], v["reasoning_share_of_output"]) for k, v in by_path.items()]
    kpis = "".join(f'<div class="card"><span class="cap">{escape(l)}</span><span class="kpi"><span class="v">{v}</span></span></div>' for l, v in (
        ("Published card p50", fmt(get(by_path, f"{pub}.wall_s_per_card.p50"), "s")),
        ("Operator card p50 (fixed)", fmt(get(by_path, "agent_normalized.wall_s_per_card.p50"), "s")),
        ("Cards per $3 / 5 h window", fmt(get(by_path, f"{pub}.cards_per_allowance.5h"))),
        ("Run total", fmt(get(cost, "summary.run_total_cost_usd"), "usd")),
    ))
    cost_sec = f"""
<section id="cost"><div class="head"><h2>Cost and latency</h2>
<p class="cap">{len(cost.get("cards", []))} real cards · list price ${get(cost, "prices_usd_per_1m.input")} / ${get(cost, "prices_usd_per_1m.output")} per 1M tokens in / out</p></div>
<div class="grid g4">{kpis}</div>
<div class="grid g2">
  <div class="card"><span class="cap">Cost per LLM call, by stage</span>{hbars(st_rows, top_stage)}</div>
  <div class="card"><span class="cap">Output share of cost · reasoning share of output</span>
  {pairbars([(n.replace("agent_normalized", "operator fixed").replace("agent", "operator"), o, r) for n, o, r in share], "output tokens / cost", "reasoning / output")}</div>
</div></section>"""

    # Quality: retrieval + grounding + guardrails
    sc, gl = get(ev, "retrieval.scoped") or {}, get(ev, "retrieval.global") or {}
    ret = pairbars([(m, sc.get(k), gl.get(k)) for m, k in (("recall@1", "recall@1"), ("recall@3", "recall@3"),
                    ("recall@6", "recall@6"), ("MRR", "mrr"), ("nDCG@6", "ndcg@6"))], "species-scoped", "unfiltered")
    cats = get(ev, "retrieval.by_category") or {}
    cat = hbars([(k.replace("_", " "), v["scoped_mrr"], f'{v["scoped_mrr"]:.2f}') for k, v in cats.items()])
    g = get(ev, "grounding") or {}
    t = g.get("test", {})
    conf = confusion("critic", ("really supported", "really borrowed"), ("accepted", "rejected"),
                     [[(t.get("tp", 0), "kept", False), (t.get("fn", 0), "wrongly dropped", True)],
                      [(t.get("fp", 0), "wrongly kept", True), (t.get("tn", 0), "caught", False)]])
    pc = get(ev, "pipeline.plain.per_card") or []
    tk, te = sum(c["true_kept"] for c in pc), sum(c["true_emitted"] for c in pc)
    hk, he = sum(c["halluc_kept"] for c in pc), sum(c["halluc_emitted"] for c in pc)
    pipe_conf = confusion("claims", ("true claim", "borrowed claim"), ("in the card", "not in the card"),
                          [[(tk, "kept", False), (te - tk, "dropped", True)],
                           [(hk, "leaked", True), (he - hk, "blocked", False)]]) if pc else ""
    real_cards = cost.get("cards", [])
    abst = [(name, abstention_cells([c for c in real_cards if c["path"] == key]))
            for key, name in (("published", "Published card"), ("one_call", "One-call card (reference)"),
                              ("agent", "Graded card"), ("agent_normalized", "Operator card (parser fixed)"))]
    abst_html = "".join(
        f'<div class="card"><span class="cap">{escape(name)} · real LLM, field × evidence</span>'
        f'{confusion("field", ("filled", "empty"), ("evidence exists", "no evidence"), [[cells[0][0], cells[1][0]], [cells[0][1], cells[1][1]]])}</div>'
        for name, cells in abst if cells)
    gb = hbars([(k, t.get(k), fmt(t.get(k))) for k in ("precision", "recall", "f1")])
    pp, pf = get(ev, "pipeline.plain") or {}, get(ev, "pipeline.fenced") or {}
    guard = hbars([("jobs completed", pp.get("job_success_rate"), fmt(pp.get("job_success_rate"), "pct")),
                   ("fenced JSON completed", pf.get("job_success_rate"), fmt(pf.get("job_success_rate"), "pct")),
                   ("true claims kept", pp.get("claim_retention"), fmt(pp.get("claim_retention"), "pct")),
                   ("borrowed claims leaked", pp.get("hallucination_leakage"), fmt(pp.get("hallucination_leakage"), "pct")),
                   ("citations valid", pp.get("citation_validity"), fmt(pp.get("citation_validity"), "pct"))])
    quality = f"""
<section id="quality"><div class="head"><h2>Quality</h2>
<p class="cap">{get(ev, "retrieval.queries")} gold queries · {g.get("pairs")} claim ↔ chunk pairs · scripted-LLM guardrail run over 11 species</p></div>
<div class="grid g2">
  <div class="card"><span class="cap">Retrieval, k = 6</span>{ret}</div>
  <div class="card"><span class="cap">MRR by requested category</span>{cat}</div>
  <div class="card"><span class="cap">Critic on the held-out split (n = {t.get("n")})</span>{conf}{gb}</div>
  <div class="card"><span class="cap">Claim ↔ chunk similarity (E5)</span>{histogram(g["similarity"]) if g.get("similarity") else ""}
  <div class="legend"><span><span class="sw" style="background:var(--ink)"></span>own chunk</span><span><span class="sw" style="background:var(--high)"></span>borrowed</span></div></div>
</div>
<div class="grid g2">
  <div class="card"><span class="cap">Card pipeline guardrails · scripted LLM, 11 species</span>{guard}</div>
  <div class="card"><span class="cap">Claim fate in the card · scripted LLM</span>{pipe_conf}</div>
  {abst_html}
</div>
</section>"""


    # Species identification (CV branch results)
    cvd = a.get("cv_eval") or {}
    cvm = cvd.get("models") or {}
    if cvm:
        pts = [{"name": r.removeprefix("lp_"), "star": r.startswith("lp_vit"), "x": m["cpu_p50_ms"], **m} for r, m in cvm.items()]
        colour_cv = lambda n: arch("lp_" + n)[1]
        ms_fmt = lambda t: f"{t:g} ms"
        cv_main = pareto([{**p, "y": p["field_accuracy"]} for p in pts], x_label="CPU p50 latency per image (ms, log)",
                         y_label="Field-photo accuracy", x_fmt=ms_fmt, y_fmt=lambda t: f"{t:.0%}", y_rate=True, x_log=True,
                         wide=True, quadrant=(300, 0.9), colour_of=colour_cv)
        cv_small = [
            ("OOD detection (mean AUROC) vs latency", pareto([{**p, "y": p["ood_mean_auroc"]} for p in pts],
                x_label="CPU p50 (ms, log)", y_label="OOD AUROC", x_fmt=ms_fmt, y_fmt=lambda t: f"{t:.2f}", y_rate=True,
                x_log=True, quadrant=(300, 0.9), colour_of=colour_cv)),
            ("Wrong and confident on field photos", pareto([{**p, "y": p["field_wrong_confident"]} for p in pts],
                x_label="CPU p50 (ms, log)", y_label="Wrong at conf >= 0.9", x_fmt=ms_fmt, y_fmt=lambda t: f"{t:.0%}",
                y_higher=False, x_log=True, quadrant=(300, 0.05), colour_of=colour_cv)),
            ("Background shortcut (fish erased)", pareto([{**p, "y": p["shortcut_rate"]} for p in pts],
                x_label="CPU p50 (ms, log)", y_label="Still predicted correctly", x_fmt=ms_fmt, y_fmt=lambda t: f"{t:.0%}",
                y_higher=False, x_log=True, quadrant=(300, 0.18), colour_of=colour_cv)),
        ]
        prod = cvm.get(cvd.get("production_run")) or {}
        cv_kpis = "".join(f'<div class="card"><span class="cap">{escape(l)}</span><span class="kpi"><span class="v">{v}</span></span></div>' for l, v in (
            ("Clean photos (production)", fmt(prod.get("clean_accuracy"), "pct")),
            ("Field photos (production)", fmt(prod.get("field_accuracy"), "pct")),
            ("OOD images accepted", fmt(prod.get("ood_accepted_at_threshold"), "pct")),
            ("CPU p50 per image", f'{prod.get("cpu_p50_ms", 0):.0f} ms'),
        ))
        slice_order = ["clean", "field", "dark", "bright", "blur", "jpeg", "lowres", "rotate", "occlusion", "bg_removed", "bg_swap"]
        cv_sec = f"""
<section id="species"><div class="head"><h2>Species ID</h2>
<p class="cap">{len(cvm)} linear-probe backbones · {fmt(prod.get("items"))} test images per model · source {escape(cvd.get("source", ""))} @ {escape(cvd.get("source_commit") or "working tree")}</p>
{pareto_legend(["lp_" + p["name"] for p in pts], family="ViT family (DINOv3); production is vit_l", group=arch)}</div>
<div class="grid g4">{cv_kpis}</div>
<div class="card"><span class="cap">Field-photo accuracy vs CPU latency</span>{cv_main}</div>
<div class="grid g3">{"".join(f'<div class="card"><span class="cap">{escape(t)}</span>{c}</div>' for t, c in cv_small)}</div>
<div class="card scroll"><span class="cap">Accuracy per image slice</span>{heatmap(cvm, slice_order)}</div>
</section>"""
    else:
        cv_sec = ""
    cov = get(tests, "coverage.rag_modules") or {}
    covb = hbars([(k, v["percent"] / 100, f'{v["percent"]:.0f}%') for k, v in sorted(cov.items(), key=lambda kv: kv[1]["percent"])])
    failing = "".join(f'<li>{escape(x["layer"])} · {escape(x["name"])}</li>' for x in tests.get("tests", []) if x["status"] in ("failed", "error"))
    tests_sec = f"""
<section id="tests"><div class="head"><h2>Tests</h2>
<p class="cap">{totals.get("passed", 0)} passed · {totals.get("failed", 0)} failed · {totals.get("skipped", 0)} skipped · RAG coverage {get(tests, "coverage.rag_percent")}%</p></div>
<div class="grid g2">
  <div class="card"><span class="cap">By layer</span>{stacked_layers(tests.get("layers") or {})}
  <details><summary>failing tests</summary><ul>{failing}</ul></details></div>
  <div class="card"><span class="cap">Coverage of RAG modules</span>{covb}</div>
</div></section>"""

    prov = " · ".join(f'{n}.json ({origin[n]})' for n in a if n in origin)
    footer = (f'<footer><span>{prov}</span><span>Regenerate: <code>python -m scripts.quality --label {escape(baseline)}</code> · '
              f'<code>python -m evals.cost_eval</code> · <code>python -m evals.model_compare</code> · <code>python -m evals.dashboard</code></span></footer>')

    body = f'{nav}<div class="wrap">{hero}{weaknesses}{targets}{model_sec}{cost_sec}{quality}{cv_sec}{tests_sec}{footer}</div>'
    head = (f'<title>{TITLE}</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            f'<link rel="stylesheet" href="{FONTS}">\n<style>{CSS}</style>\n')
    fragment = head + body
    standalone = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  f'{head}</head>\n<body>{body}</body></html>\n')
    return standalone, fragment


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m evals.dashboard")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--current", default="current")
    args = parser.parse_args(argv)
    standalone, fragment = render(args.baseline, args.current)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dashboard.html").write_text(standalone, encoding="utf-8")
    (REPORTS_DIR / "dashboard.artifact.html").write_text(fragment, encoding="utf-8")
    print(f"[dashboard] wrote {REPORTS_DIR / 'dashboard.html'} and dashboard.artifact.html")


if __name__ == "__main__":
    main()
