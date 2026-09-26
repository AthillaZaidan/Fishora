"""System architecture diagram (input -> gated CV -> human verification -> agent layer).

Keeps the look of the team's original system slide (dashed layer boxes, black header tabs,
icons) and updates the computer-vision part to the shipped gated ViT-B.

    python -m evaluation.cv.artifact.system_diagram
    -> evaluation/cv/artifact/figures/system_architecture.{png,pdf}

Needs: playwright (+ `playwright install chromium`), pillow. Icons load from cdn.jsdelivr.net.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from evaluation.cv.common import CV_DIR

OUT_DIR = CV_DIR / "artifact" / "figures"
ICONS = CV_DIR / "artifact" / "icons"  # dino and elephant cut from the team's original slide; fisherman: team-supplied 3D render
W, H = 2000, 1040
ICON = "https://cdn.jsdelivr.net/npm/simple-icons@13/icons/{}.svg"


def fish_b64() -> str:
    src = Image.open(CV_DIR / "artifact" / "figures" / "slices.png").convert("RGB")
    tile = src.crop((87, 100, 374, 300))
    buf = io.BytesIO()
    tile.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def png(name: str, h: int) -> str:
    data = base64.b64encode((ICONS / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="height:{h}px;width:auto">'


def layer(x, y, w, h, title, body="", cls=""):
    return (f'<div class="layer {cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
            f'<div class="tab">{title}</div>{body}</div>')


def card(x, y, w, h, title, body, cls=""):
    return (f'<div class="card {cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
            f'<div class="ctab">{title}</div><div class="cbody">{body}</div></div>')


def icon(name, size, color="#000"):
    return (f'<span class="ic" style="width:{size}px;height:{size}px;background:{color};'
            f'-webkit-mask:url({ICON.format(name)}) center/contain no-repeat;mask:url({ICON.format(name)}) center/contain no-repeat"></span>')


DOC = '''<svg width="86" height="104" viewBox="0 0 86 104"><path d="M18 8h40l20 20v68H18z" fill="#1f4fa8"/><path d="M8 16h40l20 20v62H8z" fill="#fff" stroke="#1f4fa8" stroke-width="4"/>
<path d="M48 16v20h20" fill="#dbe6f7" stroke="#1f4fa8" stroke-width="4"/><g stroke="#1f4fa8" stroke-width="4" stroke-linecap="round">
<path d="M18 50h40M18 62h40M18 74h40M18 86h26"/></g></svg>'''


DOC_SMALL = DOC.replace('width="86" height="104"', 'width="54" height="66"')


def build() -> str:
    fish = fish_b64()
    s = []

    # ---------------- top row ----------------
    s.append(layer(40, 60, 230, 250, "Input Layer",
                   f'<div class="center"><img class="fish" src="data:image/jpeg;base64,{fish}"><b>Fish Image</b>'
                   f'<span class="small">photo of a landed fish</span></div>'))

    cv = (
        card(24, 50, 196, 170, "Backbone",
             f'{png("dino", 78)}<b>DINOv3 ViT-B/16</b><span class="small">frozen · one pass</span>')
        + card(254, 50, 180, 170, "Gate 1",
               '<div class="q">Is it a fish?</div><span class="small">logistic head<br>on the embedding</span>')
        + card(468, 50, 180, 170, "Gate 2",
               '<div class="q">A species<br>we know?</div><span class="small">k-NN distance to<br>training photos</span>')
        + card(684, 50, 196, 170, "Species Head",
               '<div class="q">Top-3 species</div><span class="small">1 of 11 classes<br>+ confidence</span>')
    )
    s.append(layer(320, 60, 904, 250, "Computer Vision", cv))
    s.append('<div class="reject" style="left:574px;top:332px">✕ Rejected: not a fish / unknown species'
             '<span>→ retake photo or pick species manually</span></div>')

    s.append(layer(1274, 60, 230, 250, "Human Verification",
                   f'<div class="center">{png("fisherman", 128)}<b>Fisherman</b>'
                   '<span class="small">confirms or corrects the species</span></div>'))
    s.append(layer(1554, 60, 190, 250, "Context",
                   f'<div class="center">{DOC}<b>Fish Data</b><span class="small">verified species</span></div>'))
    s.append(layer(1794, 60, 170, 250, "Database",
                   f'<div class="center">{png("postgres", 100)}<b>PostgreSQL</b></div>', "db"))

    # ---------------- agent layer (RAG iteration 2: writer-critic workflow) ----------------
    s.append(layer(360, 460, 1604, 560, "Agent Layer", "", "big"))
    s.append(card(430, 500, 270, 180, "Workflow Orchestrator",
                  '<svg width="70" height="50" viewBox="0 0 70 50"><g stroke="#000" stroke-width="3">'
                  '<path d="M10 25h18M28 25 46 10M28 25l18 15"/></g><g fill="#000"><circle cx="10" cy="25" r="7"/><circle cx="28" cy="25" r="6"/>'
                  '<circle cx="46" cy="10" r="6"/><circle cx="46" cy="40" r="6"/></g></svg>'
                  '<b>Writer → Critic</b><span class="small">at most 1 revision</span>', "black"))
    s.append(card(430, 770, 270, 200, "Evidence Loader",
                  f'{DOC_SMALL}<b>Loads the whole verified<br>species slice</b><span class="small">RAG retrieval · E5 + pgvector</span>'))
    s.append(layer(40, 700, 250, 250, "Vector Database",
                   f'<div class="center">{png("postgres", 110)}<b>PG Vector</b></div>'))

    critic = ('<div class="checks">'
              '<div><i>1</i><span><b>Citation, numbers, taxa</b><br>rule-based, exact</span></div>'
              '<div><i>2</i><span><b>Evidence similarity</b><br>E5 embedding filter</span></div>'
              '<div><i>3</i><span><b>Entailment check</b><br>supported · inferred · unsupported</span></div>'
              '</div><span class="small keep">only <b>supported</b> claims pass</span>')
    ex = (f'<div class="gpt">{icon("openai", 44)}<span>ChatGPT</span><em>gpt-6-luna</em></div>'
          + card(28, 130, 300, 170, "Writer Agent",
                 '<b>Writes every field;<br>each claim cites a chunk</b><span class="small">1 LLM call</span>')
          + card(410, 110, 360, 260, "Critic Agent", critic)
          + card(852, 130, 272, 170, "Card Assembler",
                 '<b>Builds the card from<br>verified claims only</b>'))
    s.append(layer(790, 500, 1152, 490, "Agents", ex))
    s.append('<div class="revise" style="left:990px;top:932px">≤ 1 revision when more than half the claims are rejected</div>')

    # ---------------- arrows ----------------
    A = []

    def ar(pts, dash=True, head=True, color="#111", both=False):
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        A.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"'
                 + (' stroke-dasharray="9 7"' if dash else "")
                 + (' marker-end="url(#h)"' if head else "") + (' marker-start="url(#h)"' if both else "") + "/>")

    ar([(270, 185), (318, 185)])                           # input -> CV
    for x0, x1 in ((540, 572), (754, 786), (968, 1002)):    # inside CV
        ar([(x0, 195), (x1, 195)], dash=False)
    ar([(1224, 185), (1272, 185)])                         # CV -> human
    ar([(1504, 185), (1552, 185)])                         # human -> context
    ar([(664, 282), (664, 296), (878, 296), (878, 282)], dash=False, head=False, color="#c8553d")  # both gates
    ar([(771, 296), (771, 328)], color="#c8553d")          # gates -> rejected
    ar([(1649, 312), (1649, 436), (660, 436), (660, 482)])  # context -> orchestrator
    ar([(565, 688), (565, 750)], both=True)                # orchestrator <-> researcher
    ar([(428, 850), (292, 850)], dash=False)               # researcher -> vector db
    ar([(292, 900), (428, 900)])                           # vector db -> researcher
    ar([(702, 590), (788, 590)])                           # orchestrator -> agents
    ar([(1120, 715), (1198, 715)], dash=False)             # writer -> critic
    ar([(1562, 715), (1640, 715)], dash=False)             # critic -> assembler
    ar([(1380, 872), (1380, 922), (968, 922), (968, 804)], color="#b45309")  # revision loop
    ar([(1914, 715), (1978, 715), (1978, 330), (1879, 330), (1879, 312)], dash=False)  # assembler -> database

    arrows = (f'<svg class="arrows" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>'
              '<marker id="h" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
              '<path d="M0 0 L10 5 L0 10 z" fill="context-stroke"/></marker></defs>' + "".join(A) + "</svg>")
    return "".join(s) + arrows


CSS = f"""
@page {{ size: {W}px {H}px; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; position: relative; background: #fff; font-family: 'Inter', sans-serif; color: #111; }}
.layer {{ position: absolute; border: 2px dashed #555; background: #fff; }}
.layer.big {{ background: transparent; }}
.tab {{ position: absolute; left: 14px; top: -20px; background: #111; color: #fff; font-weight: 700; font-size: 20px;
        padding: 6px 18px; box-shadow: 3px 3px 0 #bbb; white-space: nowrap; }}
.card {{ position: absolute; border: 1.5px solid #333; background: #fff; box-shadow: 4px 4px 0 #ddd;
         display: flex; flex-direction: column; }}
.card.black .cbody {{ gap: 6px; }}
.ctab {{ position: absolute; left: 10px; top: -15px; background: #111; color: #fff; font-weight: 700; font-size: 15px;
         padding: 4px 12px; white-space: nowrap; }}
.cbody {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
          gap: 8px; padding: 22px 10px 12px; font-size: 17px; line-height: 1.3; }}
.center {{ position: absolute; inset: 30px 10px 14px; display: flex; flex-direction: column; align-items: center;
           justify-content: center; gap: 10px; text-align: center; font-size: 18px; }}
.small {{ font-size: 14px; color: #555; line-height: 1.3; }}
.emoji {{ font-size: 72px; line-height: 1; }}
.q {{ font-size: 19px; font-weight: 700; line-height: 1.25; }}
.fish {{ width: 190px; height: 132px; object-fit: cover; border-radius: 6px; }}
.ic {{ display: inline-block; flex: none; }}
.lg {{ display: flex; align-items: center; gap: 14px; }}
.gpt {{ position: absolute; left: 0; right: 0; top: 34px; display: flex; justify-content: center; align-items: center; gap: 14px;
        font-size: 40px; font-weight: 700; letter-spacing: -.01em; }}
.reject {{ position: absolute; border: 2px dashed #c8553d; background: #fbeeea; color: #a33b25; font-weight: 700; font-size: 17px;
           padding: 10px 18px; display: flex; flex-direction: column; gap: 3px; }}
.reject span {{ font-weight: 500; font-size: 15px; color: #7a4a3f; }}
.checks {{ display: flex; flex-direction: column; gap: 10px; text-align: left; font-size: 15px; line-height: 1.25; }}
.checks div {{ display: flex; gap: 10px; align-items: center; }}
.checks i {{ font-style: normal; width: 24px; height: 24px; border-radius: 12px; background: #111; color: #fff;
            font-size: 13px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex: none; }}
.checks b {{ font-size: 16px; }}
.keep {{ margin-top: 4px; color: #2b7a73; }}
.gpt em {{ font-style: normal; font-size: 16px; font-weight: 600; color: #555; align-self: flex-end; margin-bottom: 8px; }}
.revise {{ position: absolute; font-size: 14px; color: #b45309; font-weight: 600; background: #fff; padding: 0 6px; }}
.arrows {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = OUT_DIR / "_system.html"
    html.write_text('<!doctype html><html><head><meta charset="utf-8">'
                    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=block" rel="stylesheet">'
                    f'<style>{CSS}</style></head><body>{build()}</body></html>')
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=4)
        pg.goto(html.as_uri(), wait_until="networkidle")
        pg.evaluate("document.fonts.ready")
        pg.screenshot(path=str(OUT_DIR / "system_architecture.png"))
        pg.pdf(path=str(OUT_DIR / "system_architecture.pdf"), width=f"{W}px", height=f"{H}px", print_background=True)
        b.close()
    html.unlink()
    print(f"wrote {OUT_DIR / 'system_architecture.png'} and .pdf")


if __name__ == "__main__":
    main()
