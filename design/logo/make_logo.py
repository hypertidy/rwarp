"""The rwarp hex logo, drawn from rwarp's own output.

    python3 make_logo.py DATA_DIR OUT_DIR

DATA_DIR holds mur.json and mur-tilings.json from make_logo_data.R. Writes
rwarp-hex.svg (the logo) and favicon.svg (a simplified mark for small sizes)
to OUT_DIR.

On navy: the target (EPSG:3031, 10 km) cut into a regular 6 x 6 grid and
filled with MUR SST warped by rwarp; only the bottom-left half shows, cut
square to the hex's 30 degree diagonal. "rwarp" runs down the cut. At top
right, one row per visible tile, in plan order and the tile's colour,
two-toned by the source pixels the tile reads (warp_plan per tile).
"""
import json, math, os, sys
DATA, OUT = sys.argv[1], sys.argv[2]
D = json.load(open(os.path.join(DATA, "mur.json"))); TL = json.load(open(os.path.join(DATA, "mur-tilings.json")))
W = 300.0; H = W * 2 / math.sqrt(3); CX, CY = W / 2, H / 2
HEX = [(W/2, 0), (W, H/4), (W, 3*H/4), (W/2, H), (0, 3*H/4), (0, H/4)]
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
pts = lambda p: " ".join(f"{x:.2f},{y:.2f}" for x, y in p)
def ramp(anchors, n):
    hx = lambda c: [int(c[i:i+2], 16) for i in (1, 3, 5)]
    a = [hx(c) for c in anchors]; out = []
    for k in range(n):
        t = k / (n - 1) * (len(a) - 1); i = min(int(t), len(a) - 2); f = t - i
        out.append("#" + "".join(f"{round(a[i][j] + f * (a[i+1][j] - a[i][j])):02x}" for j in range(3)))
    return out
def mix(c1, c2, t):
    a = [int(c1[i:i+2], 16) for i in (1, 3, 5)]; b = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[j] + t * (b[j] - a[j])):02x}" for j in range(3))
F = D["field"]; FN, FR = F["ncol"], F["nrow"]
MUTED = ramp(["#3e5c76", "#7d98ad", "#c3cfd6", "#e7e1d4", "#d8b98e", "#b9855a"], F["hi"] - F["lo"])
tn, tr = TL["target"]["ncol"], TL["target"]["nrow"]
# distinct hues, ordered so that neighbours in the list differ strongly
PAL = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
       "#9c755f", "#86bcb6", "#f1ce63", "#d4a6c8", "#a0cbe8", "#ffbe7d", "#8cd17d", "#fabfd2"]
d = (math.cos(math.radians(30)), -math.sin(math.radians(30)))
s_of = lambda x, y: d[0] * (x - CX) + d[1] * (y - CY)
def zone(s0, side):
    nx, ny = -d[1], d[0]; c = (CX + d[0] * side * s0, CY + d[1] * side * s0); L = 1000
    a = (c[0] - nx * L, c[1] - ny * L); b = (c[0] + nx * L, c[1] + ny * L)
    return [a, b, (b[0] + d[0] * side * L, b[1] + d[1] * side * L), (a[0] + d[0] * side * L, a[1] + d[1] * side * L)]
def in_hex(x, y):
    if not (0 <= x <= W and 0 <= y <= H): return False
    dx = abs(x - CX); return y >= H/4 - dx * (H/4) / (W/2) and y <= 3*H/4 + dx * (H/4) / (W/2)
def hex_right(y, margin):
    if y < H / 4: x = CX + (W / 2) * y / (H / 4)
    elif y > 3 * H / 4: x = CX + (W / 2) * (H - y) / (H / 4)
    else: x = W
    return x - margin

def field(bx, land):
    ox, oy, w, h = bx
    out = [f'<rect x="{ox:.2f}" y="{oy:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{land}"/>']
    for b in F["bands"]:
        if not b["x"]: continue
        rings = {}
        for x, y, i in zip(b["x"], b["y"], b["id"]):
            rings.setdefault(i, []).append((ox + x / FN * w, oy + y / FR * h))
        dd = " ".join("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in r) + "Z" for r in rings.values())
        out.append(f'<path d="{dd}" fill="{MUTED[int(b["lo"]) - F["lo"]]}" fill-rule="evenodd"/>')
    return "".join(out)

def rect_of(bx, t):
    ox, oy, w, h = bx
    return (ox + t["dst_xoff"] / tn * w, oy + t["dst_yoff"] / tr * h, t["dst_xsize"] / tn * w, t["dst_ysize"] / tr * h)

def visible(bx, tiles, s0, min_area):
    vis = []
    for t in tiles:
        x, y, w, h = rect_of(bx, t); k = 14; hit = 0
        for a in range(k):
            for b in range(k):
                px, py = x + (a + 0.5) * w / k, y + (b + 0.5) * h / k
                if in_hex(px, py) and s_of(px, py) < -s0: hit += 1
        if hit / (k * k) * w * h >= min_area: vis.append(t)
    return vis

def tiles_svg(bx, tiles, vis, colour, sep, tint):
    out = []
    for t in tiles:
        x, y, w, h = rect_of(bx, t)
        if t["tile"] in colour:
            c = colour[t["tile"]]
            out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{c}" fill-opacity="{tint}"/>')
            out.append(f'<rect x="{x + 2:.2f}" y="{y + 2:.2f}" width="{w - 4:.2f}" height="{h - 4:.2f}" fill="none" stroke="{c}" stroke-width="2"/>')
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="none" stroke="{sep}" stroke-width="2"/>')
    return "".join(out)

def rows(vis, colour, s0, bg, top, bottom, margin=16):
    n = len(vis); pitch = (bottom - top) / n; rh = pitch * 0.62
    mx = max(t["src_pixels"] for t in vis); out = []
    for k, t in enumerate(vis):
        y = top + k * pitch; ym = y + rh / 2
        x0 = CX + (s0 - d[1] * (ym - CY)) / d[0]
        x1 = hex_right(y, margin) if y < H / 4 else hex_right(ym, margin)
        w = x1 - x0
        c = colour[t["tile"]]; f = t["src_pixels"] / mx
        out.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{rh:.2f}" rx="{rh/2:.2f}" fill="{mix(c, bg, 0.6)}"/>')
        out.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{max(rh, w * f):.2f}" height="{rh:.2f}" rx="{rh/2:.2f}" fill="{c}"/>')
    return "".join(out)

def sticker(name, n, bg, land, sep, border, title_fill, tint=0.42, mw=400, at=(84, 226), s0=27, size=42, min_area=420,
            with_field=True, with_title=True):
    tiling = TL["tilings"][str(n)]
    rowsdf = tiling["tiles"]
    tiles = [dict(zip(rowsdf.keys(), v)) for v in zip(*rowsdf.values())] if isinstance(rowsdf, dict) else rowsdf
    mh = mw * tr / tn; bx = (at[0] - mw / 3, at[1] - mh * 2 / 3, mw, mh)
    vis = visible(bx, tiles, s0, min_area)
    colour = {t["tile"]: PAL[k % len(PAL)] for k, t in enumerate(vis)}
    top = 34 if len(vis) > 9 else 40
    bottom = min(232, top + len(vis) * 19)
    base = field(bx, land) if with_field else f'<rect x="-10" y="-10" width="{W + 20}" height="{H + 20}" fill="{land}"/>'
    title = (f'<text x="{CX:.1f}" y="{CY:.1f}" transform="rotate(60 {CX:.1f} {CY:.1f})" dy="0.35em" text-anchor="middle" '
             f'font-family="{FONT}" font-weight="700" font-size="{size}" letter-spacing="2" fill="{title_fill}">rwarp</text>') if with_title else ""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-4 -4 {W+8:.2f} {H+8:.2f}" width="{W+8:.0f}" height="{H+8:.0f}">'
           f'<defs><clipPath id="hx"><polygon points="{pts(HEX)}"/></clipPath>'
           f'<clipPath id="bl"><polygon points="{pts(zone(s0, -1))}"/></clipPath></defs>'
           f'<polygon points="{pts(HEX)}" fill="{bg}"/>'
           f'<g clip-path="url(#hx)"><g clip-path="url(#bl)">{base}{tiles_svg(bx, tiles, vis, colour, sep, tint)}</g>'
           f'{rows(vis, colour, s0 + 8, bg, top, bottom)}</g>'
           f'{title}'
           f'<polygon points="{pts(HEX)}" fill="none" stroke="{border}" stroke-width="6" stroke-linejoin="round"/></svg>')
    open(os.path.join(OUT, name + ".svg"), "w").write(svg)
    print(name, f"{n} x {n}:", len(vis), "visible tiles, rows", [t["tile"] for t in vis])

NAVY = "#13283a"
sticker("rwarp-hex", 6, NAVY, "#2a4459", NAVY, "#2f5f80", "#eef3f6")
sticker("favicon", 6, NAVY, "#2a4459", NAVY, "#2f5f80", "#eef3f6", tint=0.9, with_field=False, with_title=False)
