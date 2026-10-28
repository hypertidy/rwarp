"""Regenerate the gdalwarp -cutline fixtures for tests/cutline.rs.

    python3 tests/fixtures/cutline/gen.py

Needs the GDAL Python bindings (and GEOS, for -cblend). Writes src.txt and
src_nodata.txt (64x64 Float32, the same patterns as tests/fixtures/kernels)
and one <case>.txt per case with gdalwarp's output.

Cutlines are given here in source pixel/line coordinates and passed to
gdalwarp as WKT in the source's georeferenced coordinates (x, 64 - line for
the affine cases). Each fixture header records the cutline as nested lists
in those georeferenced coordinates ("cut"), the rule, the blend distance and
everything else the Rust test needs to repeat the warp.

GDALWARP_DENSIFY_CUTLINE=NO for the affine cases: with no CRS on either
side gdalwarp would otherwise segmentize the cutline (it only skips that
when the raster and cutline share a CRS), and the interpolated points can
move a vertex placed exactly through pixel centres by an ulp, which is
exactly what the tie cases are about. The map is affine, so densifying
changes nothing else.
"""

import array
import json
import math
import os

from osgeo import gdal

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
N = 64
NODATA = -9999.0

LAEA = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs"
LONLAT = "+proj=longlat +datum=WGS84 +no_defs"
LAEA_GT = [-64000.0, 2000.0, 0.0, 64000.0, 0.0, -2000.0]


def source(holes):
    vals = array.array("f")
    for r in range(N):
        for c in range(N):
            v = 100 * math.sin(c * 0.9) * math.cos(r * 0.7) + 3 * c + r
            if holes and (r * 7 + c * 3) % 23 == 0:
                v = NODATA
            vals.append(v)
    return vals


def mem_source(holes, srs=None, gt=None):
    ds = gdal.GetDriverByName("MEM").Create("", N, N, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt or [0, 1, 0, N, 0, -1])
    if srs:
        ds.SetProjection(srs)
    band = ds.GetRasterBand(1)
    band.WriteRaster(0, 0, N, N, source(holes).tobytes())
    if holes:
        band.SetNoDataValue(NODATA)
    return ds


# ---- cutlines, in source pixel/line ----------------------------------------

def star(cx, cy, r0, r1, n, rot):
    pts = []
    for k in range(2 * n):
        r = r0 if k % 2 == 0 else r1
        a = rot + math.pi * k / n
        pts.append((round(cx + r * math.cos(a), 6), round(cy + r * math.sin(a), 6)))
    return pts


POLYS = {
    # Edges and vertices exactly through pixel centres: GDAL's tie rules.
    "rect_tie": [[[(10.5, 12.5), (30.5, 12.5), (30.5, 40.5), (10.5, 40.5)]]],
    "diamond_tie": [[[(32.5, 4.5), (60.5, 32.5), (32.5, 60.5), (4.5, 32.5)]]],
    # A hole whose top edge lies on a centre line, sides through centres.
    "hole_tie": [[[(2, 2), (62, 2), (62, 62), (2, 62)],
                  [(20.5, 20.5), (40.5, 20.5), (40.5, 44.5), (20.5, 44.5)]]],
    # The same, with each ring the other way round.
    "hole_tie_rev": [[[(2, 2), (2, 62), (62, 62), (62, 2)],
                      [(20.5, 20.5), (20.5, 44.5), (40.5, 44.5), (40.5, 20.5)]]],
    # Horizontal edges on centre lines in a concave exterior (a U).
    "u_tie": [[[(8.5, 8.5), (20.5, 8.5), (20.5, 40.5), (40.5, 40.5), (40.5, 8.5),
                (52.5, 8.5), (52.5, 54.5), (8.5, 54.5)]]],
    "tri": [[[(4.3, 4.7), (60.2, 20.9), (10.6, 59.1)]]],
    "star": [[star(32.2, 31.7, 29.3, 12.1, 7, 0.21)]],
    "star_hole": [[star(32.2, 31.7, 29.3, 12.1, 7, 0.21),
                   [(27.3, 26.9), (37.8, 28.4), (35.1, 38.6), (26.2, 36.3)]]],
    "multi": [[[(5.2, 5.7), (28.4, 6.1), (25.9, 29.3), (6.6, 24.8)],
               [(10.3, 10.2), (20.1, 11.4), (15.7, 20.6)]],
              [[(35.1, 33.3), (60.7, 38.2), (52.4, 61.5), (36.8, 55.9)]]],
    # All-touched edge cases: edges on pixel walls, a diagonal through
    # pixel corners, edges within GDAL's 1e-4 of a wall, and an edge that
    # is "vertical" to GDAL (|dx| < 0.01) while crossing a wall.
    "aligned": [[[(10, 12), (30, 12), (30, 40), (10, 40)]]],
    "corner_diag": [[[(10, 10), (40, 40), (10, 40)]]],
    "near_wall": [[[(10.00005, 12.00005), (29.99995, 12), (30.00002, 40), (10, 39.99997)]]],
    "near_vertical": [[[(20.995, 5.3), (21.004, 50.2), (5.5, 50.2)]]],
}


def affine_world(polys):
    return [[[(x, N - y) for (x, y) in ring] for ring in poly] for poly in polys]


def laea_world(polys):
    g = LAEA_GT
    return [[[(g[0] + x * g[1], g[3] + y * g[5]) for (x, y) in ring] for ring in poly] for poly in polys]


def wkt(polys):
    def ring(r):
        pts = list(r) + [r[0]]
        return "(" + ",".join(f"{x!r} {y!r}" for x, y in pts) + ")"
    parts = ["(" + ",".join(ring(r) for r in poly) + ")" for poly in polys]
    return "MULTIPOLYGON(" + ",".join(parts) + ")"


def rot30_gt():
    """8x8 pixels of size 4 rotated 30 degrees about the source centre (as
    tests/fixtures/kernels)."""
    t, size = math.radians(30), 4.0
    g1, g2, g4, g5 = size * math.cos(t), size * math.sin(t), size * math.sin(t), -size * math.cos(t)
    return [32 - 4 * (g1 + g2), g1, g2, 32 - 4 * (g4 + g5), g4, g5]


def rot30_big_gt():
    """32x32 pixels of size 1.6, rotated 30 degrees about the source centre."""
    t, size, n = math.radians(30), 1.6, 32
    g1, g2, g4, g5 = size * math.cos(t), size * math.sin(t), size * math.sin(t), -size * math.cos(t)
    return [32 - n / 2 * (g1 + g2), g1, g2, 32 - n / 2 * (g4 + g5), g4, g5]


def write_grid(path, header, w, h, vals):
    with open(path, "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write(f"{w} {h}\n")
        for r in range(h):
            f.write(" ".join(repr(float(v)) for v in vals[r * w:(r + 1) * w]) + "\n")


# name, alg, dst, cutline, all touched, blend, src nodata holes, dst nodata,
# extra warp options. dst is (w, h) over the source extent, "id" for the
# source grid, or "rot30"/"rot30big".
POINT = {"XSCALE": "1", "YSCALE": "1"}
CASES = []
# Masks through an identity nearest warp: exactly which pixels are in.
for p in ["rect_tie", "diamond_tie", "hole_tie", "hole_tie_rev", "u_tie", "tri", "star_hole", "multi"]:
    CASES.append((f"mask_{p}", "near", "id", p, False, 0, False, True, {}))
for p in ["rect_tie", "diamond_tie", "hole_tie", "u_tie", "tri", "star_hole", "multi",
          "aligned", "corner_diag", "near_wall", "near_vertical"]:
    CASES.append((f"mask_at_{p}", "near", "id", p, True, 0, False, True, {}))
# Downsampled warps.
# Mode is left out: on continuous Float32 data every value is distinct and
# GDAL's float mode breaks those ties differently from its 8/16-bit path,
# a known difference (see the GDAL parity chapter).
for alg in ["near", "bilinear", "cubic", "lanczos", "cubicspline", "average", "max", "med", "rms"]:
    CASES.append((f"{alg}_star", alg, (23, 23), "star", False, 0, False, True, {}))
for alg in ["bilinear", "cubic", "lanczos", "cubicspline"]:
    CASES.append((f"{alg}_star_point", alg, (23, 23), "star", False, 0, False, True, POINT))
CASES.append(("cubic_star_up", "cubic", (96, 96), "star", False, 0, False, True, {}))
CASES.append(("bilinear_star_up", "bilinear", (96, 96), "star", False, 0, False, True, {}))
# Holes, multipolygons, nodata.
for alg in ["near", "bilinear", "cubic", "lanczos", "average"]:
    CASES.append((f"{alg}_multi_nodata", alg, (23, 23), "multi", False, 0, True, True, {}))
CASES.append(("bilinear_star_hole", "bilinear", (23, 23), "star_hole", False, 0, False, True, {}))
CASES.append(("average_star_hole", "average", (23, 23), "star_hole", False, 0, False, True, {}))
CASES.append(("average_multi_nodata_pct",
              "average", (23, 23), "multi", False, 0, True, True, {"NODATA_VALUES_PCT_THRESHOLD": "50"}))
# All touched.
for alg in ["near", "bilinear", "cubic", "average"]:
    CASES.append((f"{alg}_star_at", alg, (23, 23), "star", True, 0, False, True, {}))
# Blend distance, into a destination with nodata (blending only thresholds
# the result) and without (the result is scaled towards the 0 fill).
for alg in ["near", "bilinear", "cubic", "lanczos", "average", "max"]:
    CASES.append((f"{alg}_star_blend", alg, (23, 23), "star", False, 3.0, False, True, {}))
for alg in ["near", "bilinear", "cubic", "lanczos", "cubicspline"]:
    CASES.append((f"{alg}_star_blend_nodst", alg, (23, 23), "star", False, 3.0, False, False, {}))
CASES.append(("near_id_blend_nodst", "near", "id", "multi", False, 2.5, False, False, {}))
CASES.append(("bilinear_id_blend_nodst", "bilinear", "id", "star_hole", False, 4.0, False, False, {}))
CASES.append(("bilinear_star_blend_at_nodst", "bilinear", (23, 23), "star", True, 3.0, False, False, {}))
CASES.append(("bilinear_multi_blend_nodata", "bilinear", (23, 23), "multi", False, 2.0, True, True, {}))
# Rotated destination.
for alg in ["near", "bilinear", "cubic", "average"]:
    CASES.append((f"{alg}_star_rot30", alg, "rot30big", "star", False, 0, False, True, {}))
CASES.append(("bilinear_star_rot30_blend_nodst", "bilinear", "rot30big", "star", False, 3.0, False, False, {}))


def run_affine(name, alg, dst, pname, at, blend, holes, dstnd, wo):
    polys = POLYS[pname]
    world = affine_world(polys)
    wo = dict(wo)
    if at:
        wo["CUTLINE_ALL_TOUCHED"] = "YES"
    src = mem_source(holes)
    kw = dict(resampleAlg=alg, errorThreshold=0, warpOptions=wo, cutlineWKT=wkt(world))
    if blend:
        kw["cutlineBlend"] = blend
    if dst in ("rot30", "rot30big"):
        n = 8 if dst == "rot30" else 32
        out = gdal.GetDriverByName("MEM").Create("", n, n, 1, gdal.GDT_Float32)
        out.SetGeoTransform(rot30_gt() if dst == "rot30" else rot30_big_gt())
        if dstnd:
            out.GetRasterBand(1).SetNoDataValue(NODATA)
            out.GetRasterBand(1).Fill(NODATA)
            kw["dstNodata"] = NODATA
        gdal.Warp(out, src, **kw)
        w = h = n
    else:
        w, h = (N, N) if dst == "id" else dst
        out = gdal.Warp("", src, format="MEM", outputBounds=[0, 0, N, N], width=w, height=h,
                        outputType=gdal.GDT_Float32, dstNodata=NODATA if dstnd else None, **kw)
    vals = array.array("f", out.GetRasterBand(1).ReadRaster(buf_type=gdal.GDT_Float32))
    opts = " ".join(f"{k}={v}" for k, v in wo.items() if k != "CUTLINE_ALL_TOUCHED") or "-"
    write_grid(os.path.join(HERE, f"{name}.txt"),
               [f"gdal {gdal.__version__}", f"alg {alg}", f"wo {opts}",
                f"rule {'all_touched' if at else 'centre'}", f"blend {blend}",
                f"nodata_src {holes}", f"dst_nodata {dstnd}", "crs -",
                "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform()),
                "cut " + json.dumps(world)],
               w, h, vals)


def run_reproj(name, alg, pname, cut_in_dst, dstnd):
    """LAEA source (2 km pixels) to lon/lat. The cutline is in the source
    CRS (no densification), or in lon/lat (gdalwarp densifies it to about
    a pixel per segment)."""
    if cut_in_dst:
        world = [[[(146.55, -41.62), (147.48, -41.78), (147.35, -42.45), (146.62, -42.31)]]]
        srs = LONLAT
    else:
        world = laea_world(POLYS[pname])
        srs = LAEA
    src = mem_source(False, LAEA, LAEA_GT)
    w = h = 40
    out = gdal.Warp("", src, format="MEM", dstSRS=LONLAT, outputBounds=[146.2, -42.6, 147.8, -41.4],
                    width=w, height=h, resampleAlg=alg, errorThreshold=0, outputType=gdal.GDT_Float32,
                    dstNodata=NODATA if dstnd else None, cutlineWKT=wkt(world), cutlineSRS=srs)
    vals = array.array("f", out.GetRasterBand(1).ReadRaster(buf_type=gdal.GDT_Float32))
    write_grid(os.path.join(HERE, f"{name}.txt"),
               [f"gdal {gdal.__version__}", f"alg {alg}", "wo -", "rule centre", "blend 0",
                "nodata_src False", f"dst_nodata {dstnd}",
                f"crs {'dst' if cut_in_dst else 'src'}",
                "src_crs " + LAEA, "dst_crs " + LONLAT,
                "src_gt " + " ".join(repr(v) for v in LAEA_GT),
                "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform()),
                "cut " + json.dumps(world)],
               w, h, vals)


def main():
    write_grid(os.path.join(HERE, "src.txt"), [f"gdal {gdal.__version__}"], N, N, source(False))
    write_grid(os.path.join(HERE, "src_nodata.txt"), [f"gdal {gdal.__version__}"], N, N, source(True))
    gdal.SetConfigOption("GDALWARP_DENSIFY_CUTLINE", "NO")
    for case in CASES:
        run_affine(*case)
    gdal.SetConfigOption("GDALWARP_DENSIFY_CUTLINE", None)
    for alg in ["near", "bilinear", "average"]:
        run_reproj(f"{alg}_reproj", alg, "star", False, True)
    run_reproj("near_reproj_dstcut", "near", None, True, True)
    run_reproj("bilinear_reproj_dstcut", "bilinear", None, True, True)
    print(len(CASES) + 5, "cases")


if __name__ == "__main__":
    main()
