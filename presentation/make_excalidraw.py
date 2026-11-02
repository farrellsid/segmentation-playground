"""Generate Excalidraw diagrams for the talk (per the slidev excalidraw-generation skill).

Hand-drawn aesthetic (roughness 1), colourblind-safe, dark text, few words. Colours match the
deck's Okabe-Ito palette (blue #0072B2, orange #E69F00) rather than the skill's default blue/orange
so the diagrams sit consistently beside the data figures. Each build_* function writes one
diagrams/<slug>.excalidraw; render with scripts/render-excalidraw.sh.

Run: py -3 make_excalidraw.py <slug>     (or with no arg: build all)
"""
import json
import random
import string
import sys
from pathlib import Path

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#6b7280"
TEXT = "#1f2937"
LIGHT_BG = "#f3f4f6"


def _id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=16))


def _rand():
    return random.randint(0, 1_000_000)


def text(s, x, y, w=200, h=25, size=20, color=TEXT, container=None, align="center", family=2):
    return {
        "type": "text", "version": 1, "versionNonce": _rand(), "isDeleted": False,
        "id": _id(), "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
        "roughness": 0, "opacity": 100, "angle": 0, "x": x, "y": y,
        "strokeColor": color, "backgroundColor": "transparent", "width": w, "height": h,
        "seed": _rand(), "groupIds": [], "frameId": None, "roundness": None,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
        "fontSize": size, "fontFamily": family, "text": s, "textAlign": align,
        "verticalAlign": "middle", "containerId": container, "originalText": s,
        "lineHeight": 1.25, "baseline": 18,
    }


def box(x, y, w, h, label, stroke=BLUE, size=20, fill="transparent"):
    bid = _id()
    rect = {
        "type": "rectangle", "version": 1, "versionNonce": _rand(), "isDeleted": False,
        "id": bid, "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 0, "opacity": 100, "angle": 0, "x": x, "y": y,
        "strokeColor": stroke, "backgroundColor": fill, "width": w, "height": h,
        "seed": _rand(), "groupIds": [], "frameId": None, "roundness": {"type": 3},
        "boundElements": [], "updated": 1, "link": None, "locked": False,
    }
    lines = label.count("\n") + 1
    th = int(round(lines * size * 1.25))          # natural text height, not the box height
    ty = y + (h - th) // 2                          # vertically centre inside the box
    t = text(label, x + 8, ty, w=w - 16, h=th, size=size, container=bid)
    rect["boundElements"].append({"type": "text", "id": t["id"]})
    return rect, t


def _edge(shape, side):
    x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
    return {
        "right": (x + w, y + h / 2), "left": (x, y + h / 2),
        "top": (x + w / 2, y), "bottom": (x + w / 2, y + h),
    }[side]


def _focus(side):
    return {"right": (1, 0), "left": (-1, 0), "top": (0, -1), "bottom": (0, 1)}[side]


def arrow(a, a_side, b, b_side, color=GRAY):
    sx, sy = _edge(a, a_side)
    ex, ey = _edge(b, b_side)
    fa, fb = _focus(a_side), _focus(b_side)
    return {
        "type": "arrow", "version": 1, "versionNonce": _rand(), "isDeleted": False,
        "id": _id(), "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 0, "opacity": 100, "angle": 0, "x": sx, "y": sy,
        "strokeColor": color, "backgroundColor": "transparent",
        "width": abs(ex - sx), "height": abs(ey - sy), "seed": _rand(),
        "groupIds": [], "frameId": None, "roundness": {"type": 2}, "boundElements": [],
        "updated": 1, "link": None, "locked": False,
        "startBinding": {"elementId": a["id"], "focus": fa[1], "gap": 8},
        "endBinding": {"elementId": b["id"], "focus": fb[1], "gap": 8},
        "lastCommittedPoint": None, "startArrowhead": None, "endArrowhead": "arrow",
        "points": [[0, 0], [ex - sx, ey - sy]],
    }


def shape(kind, x, y, w, h, stroke=BLUE, fill="transparent", opacity=100, rounded=True):
    """A plain shape (rectangle / ellipse / diamond) with no bound text."""
    return {
        "type": kind, "version": 1, "versionNonce": _rand(), "isDeleted": False,
        "id": _id(), "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 0, "opacity": opacity, "angle": 0, "x": x, "y": y,
        "strokeColor": stroke, "backgroundColor": fill, "width": w, "height": h,
        "seed": _rand(), "groupIds": [], "frameId": None,
        "roundness": {"type": 3} if (rounded and kind == "rectangle") else None,
        "boundElements": [], "updated": 1, "link": None, "locked": False,
    }


def assemble(elements):
    return {
        "type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None, "theme": "light"},
        "files": {},
    }


def write(slug, elements):
    out = Path("diagrams") / f"{slug}.excalidraw"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assemble(elements), indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(elements)} elements)")


# --- diagrams ---------------------------------------------------------------

def build_data_flow():
    els = []
    em = box(40, 90, 230, 90, "Raw EM stack\n2,354 slices", stroke=BLUE)
    sk = box(40, 240, 230, 90, "Hand-traced skeletons\n(CATMAID)", stroke=BLUE)
    pipe = box(380, 165, 190, 90, "Pipeline\nSAM2 / SAM3", stroke=ORANGE, fill="#fdf1dd")
    out = box(680, 165, 240, 90, "Per-neuron 3D masks\nfor Blender", stroke=BLUE)
    for rect, t in (em, sk, pipe, out):
        els.append(rect)
        els.append(t)
    els.append(arrow(em[0], "right", pipe[0], "left"))
    els.append(arrow(sk[0], "right", pipe[0], "left"))
    els.append(arrow(pipe[0], "right", out[0], "left"))
    els.append(text("we work on ~300 slices\n(the densest nerve ring)", 40, 190, w=230, h=40,
                    size=14, color=GREEN, align="left"))
    write("data-flow", els)


def build_three_methods():
    els = []
    y, w, h = 90, 240, 90
    xs = [40, 340, 640]
    names = ["Video propagation", "Per-slice", "Per-frame / dense"]
    caps = ["seed once, track through", "re-anchor every slice", "segment all at once"]
    for x, name, cap in zip(xs, names, caps):
        r, t = box(x, y, w, h, name, stroke=BLUE)
        els += [r, t]
        els.append(text(cap, x, y + h + 14, w=w, h=24, size=16, color=TEXT))
    els.append(text("SAM2 and SAM3 are a model swap used with all three methods",
                    40, y + h + 74, w=840, h=24, size=16, color=GREEN))
    write("three-methods", els)


def build_sam_modes():
    els = []
    y, w, h = 90, 330, 110
    b1 = box(40, y, w, h, "Image mode", stroke=BLUE)
    b2 = box(450, y, w, h, "Video mode", stroke=BLUE)
    for r, t in (b1, b2):
        els += [r, t]
    els.append(text("one point, one mask (single slice)", 40, y + h + 16, w=w, h=24, size=16))
    els.append(text("carry the mask across slices", 450, y + h + 16, w=w, h=24, size=16))
    els.append(text("two ways to use SAM", 40, y - 46, w=740, h=24, size=16, color=GREEN))
    write("sam-modes", els)


def build_adaptations():
    els = []
    y, w, h, gap = 90, 200, 120, 30
    xs = [40 + i * (w + gap) for i in range(4)]
    items = [
        "Branch chains\nsplit at branch points",
        "Two-res crops\nneurite is ~3 px wide",
        "Negative prompts\nneighbours touch and bleed",
        "Automated QC\nflag likely errors",
    ]
    for x, label in zip(xs, items):
        r, t = box(x, y, w, h, label, stroke=BLUE, size=17)
        els += [r, t]
    write("adaptations", els)


def build_merge_metric():
    els = []
    # two adjacent cells sharing a membrane
    els.append(shape("rectangle", 120, 70, 220, 180, stroke=GRAY, fill="#f3f4f6"))
    els.append(shape("rectangle", 340, 70, 220, 180, stroke=GRAY, fill="#f3f4f6"))
    # the shared membrane (thick dark divider)
    els.append(shape("rectangle", 334, 70, 10, 180, stroke=TEXT, fill=TEXT, rounded=False))
    # one mask (blue, semi-transparent) that covers its own cell AND bleeds across the membrane
    els.append(shape("rectangle", 150, 95, 300, 130, stroke=BLUE, fill=BLUE, opacity=30))
    # own node: green circle, inside the mask in the left cell
    els.append(shape("ellipse", 205, 146, 28, 28, stroke=GREEN, fill=GREEN))
    # foreign node: orange diamond, engulfed by the bleed in the right cell
    els.append(shape("diamond", 404, 144, 32, 32, stroke=ORANGE, fill=ORANGE))
    # labels
    els.append(text("own node covered\n= good coverage", 120, 262, w=220, h=44, size=15, color=GREEN))
    els.append(text("foreign node inside\n= merge (bleed)", 340, 262, w=220, h=44, size=15, color=ORANGE))
    write("merge-metric", els)


def build_motivation():
    els = []
    a = box(40, 95, 280, 100, "mEMbrain attempt\n2 people, 5 months", stroke=ORANGE)
    b = box(600, 95, 300, 100, "Mei's target\n~30 min per neuron", stroke=BLUE)
    for r, t in (a, b):
        els += [r, t]
    els.append(arrow(a[0], "right", b[0], "left"))
    els.append(text("minimize manual correction", 340, 66, w=260, h=24, size=16, color=GREEN))
    l1 = box(210, 250, 270, 72, "Better segmentation", stroke=BLUE, size=17)
    l2 = box(510, 250, 320, 72, "Smoother correction (GUI)", stroke=BLUE, size=17)
    for r, t in (l1, l2):
        els += [r, t]
    els.append(text("two levers", 40, 272, w=150, h=24, size=16, color=TEXT))
    write("motivation", els)


def build_per_slice():
    """A neurite running down through the slices with a seed on every one."""
    els = []
    els.append(shape("rectangle", 150, 20, 20, 380, stroke=GRAY, fill="#e5e9ee", rounded=False))
    for i in range(6):
        y = 34 + i * 68
        els.append(shape("ellipse", 144, y, 32, 32, stroke=GREEN, fill=GREEN))
    write("per-slice", els)


BUILDERS = {
    "per-slice": build_per_slice,
    "motivation": build_motivation,
    "data-flow": build_data_flow,
    "three-methods": build_three_methods,
    "sam-modes": build_sam_modes,
    "adaptations": build_adaptations,
    "merge-metric": build_merge_metric,
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(BUILDERS)
    for name in which:
        BUILDERS[name]()
