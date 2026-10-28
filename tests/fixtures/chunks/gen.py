"""Regenerate the gdalwarp oracle fixtures for tests/chunks.rs.

    python3 tests/fixtures/chunks/gen.py

Needs the GDAL Python bindings (the fixtures were made with the GDAL named
in each file's header). Each case warps a whole lon/lat source to a
target with a warp memory limit (gdalwarp's -wm, in bytes; small, so GDAL
splits the target into many chunks, or GDAL's default for one chunk).
Each <case>.txt holds gdalwarp's output and, from GDAL's CPL_DEBUG log,
the chunks it warped (source and target windows, kernel scale), plus the
special points GDAL computes, the centre longitude GDAL wraps the source
around (CENTER_LONG) and target-to-source and source-to-target points
through GDAL's exact transformer (GDALCreateGenImgProjTransformer2 with
the source and the output dataset, as gdalwarp builds it). GDAL logs only
chunks that read something. The source is the arithmetic field in
`value()` below, which tests/chunks.rs computes the same way.
"""

import os
import re

from osgeo import gdal, osr

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
NODATA = -9999.0


def value(c, r):
    # Exact in f64 but for the /10, which IEEE rounds the same way in Rust
    # and Python. Stored as Float64: GDAL refuses a warp memory limit under
    # 100000 bytes, and 64-bit pixels split small targets into many chunks.
    return ((c * 7919 + r * 104729) % 1000) / 10.0 + c * 0.5 - r * 0.25


def source(gt, w, h, crs, nodata):
    import array

    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.SetFromUserInput(crs)
    ds.SetProjection(srs.ExportToWkt())
    vals = array.array("d", [value(c, r) for r in range(h) for c in range(w)])
    band = ds.GetRasterBand(1)
    band.WriteRaster(0, 0, w, h, vals.tobytes())
    if nodata:
        band.SetNoDataValue(NODATA)
    return ds


GLOBAL1 = ([-180, 1, 0, 90, 0, -1], 360, 180)
# The LAEA target overhangs the regional source on every side and crosses
# 180 degrees: GDAL wraps the source's longitudes around its centre
# (CENTER_LONG, 170 here), so target longitudes past 180 find the source's
# 180..200 columns.
REGION = ([140, 0.25, 0, -10, 0, -0.25], 240, 160)
LAEA = "+proj=laea +lat_0=-30 +lon_0=170 +datum=WGS84 +units=m +no_defs"
MERC_X = 20037508.342789244
MERC_Y = 15538711.096309224  # about 80 degrees

# Regional sources across 180 degrees, stored either side of it, and
# global sources in 0..360 and wider than 360 degrees.
E170 = ([170, 0.1, 0, -5, 0, -0.1], 200, 300)  # 170..190 E
W190 = ([-190, 0.1, 0, -5, 0, -0.1], 200, 300)  # the same, as -190..-170
SOUTH = ([160, 0.25, 0, -55, 0, -0.25], 160, 100)  # 160..200 E, 55..80 S
GLOBAL360 = ([0, 1, 0, 90, 0, -1], 360, 180)
GLOBAL361 = ([-180.5, 1, 0, 90, 0, -1], 361, 180)
LAEA180 = "+proj=laea +lat_0=-20 +lon_0=180 +datum=WGS84 +units=m +no_defs"
LAEA180_TE = (-1400000, -2000000, 1400000, 2000000)

# name, alg, source, dst crs, dst extent (xmin, ymin, xmax, ymax), dst size,
# -wm bytes (None: GDAL's default, one chunk), source nodata, target block
# size (None: MEM)
CASES = [
    ("merc_lanczos", "lanczos", GLOBAL1, "EPSG:3857",
     (-MERC_X, -MERC_Y, MERC_X, MERC_Y), (100, 84), 100000, False, None),
    ("merc_bilinear", "bilinear", GLOBAL1, "EPSG:3857",
     (-MERC_X, -MERC_Y, MERC_X, MERC_Y), (100, 84), 100000, False, None),
    ("merc_bilinear_tiled", "bilinear", GLOBAL1, "EPSG:3857",
     (-MERC_X, -MERC_Y, MERC_X, MERC_Y), (128, 128), 100000, False, (32, 32)),
    ("stere_near", "near", GLOBAL1, "EPSG:3031",
     (-4000000, -4000000, 4000000, 4000000), (100, 100), 100000, False, None),
    ("stere_cubic", "cubic", GLOBAL1, "EPSG:3031",
     (-4000000, -4000000, 4000000, 4000000), (100, 100), 100000, False, None),
    ("laea_near", "near", REGION, LAEA,
     (-3500000, -3500000, 3500000, 3500000), (100, 100), 100000, True, None),
    ("laea_bilinear", "bilinear", REGION, LAEA,
     (-3500000, -3500000, 3500000, 3500000), (100, 100), 100000, True, None),
    # CENTER_LONG
    ("am_e170_laea_bilinear", "bilinear", E170, LAEA180, LAEA180_TE, (70, 100), 100000, True, None),
    ("am_w190_laea_bilinear", "bilinear", W190, LAEA180, LAEA180_TE, (70, 100), 100000, True, None),
    ("am_e170_laea_cubic_whole", "cubic", E170, LAEA180, LAEA180_TE, (70, 100), None, True, None),
    ("am_e170_merc_near", "near", E170, "EPSG:3857",
     (18500000, -4500000, 21500000, -300000), (100, 140), 100000, True, None),
    ("am_south_stere_bilinear", "bilinear", SOUTH, "EPSG:3031",
     (-1400000, -3600000, 1400000, -800000), (100, 100), 100000, True, None),
    ("am_g360_merc_bilinear", "bilinear", GLOBAL360, "EPSG:3857",
     (-MERC_X, -MERC_Y, MERC_X, MERC_Y), (100, 84), 100000, False, None),
    ("am_g360_stere_near", "near", GLOBAL360, "EPSG:3031",
     (-4000000, -4000000, 4000000, 4000000), (100, 100), 100000, False, None),
    ("am_g360_geog_near_whole", "near", GLOBAL360, "EPSG:4326",
     (-180, -90, 180, 90), (180, 90), None, False, None),
    ("am_g361_geog_te_near", "near", GLOBAL361, "EPSG:4326",
     (90, -90, 270, 90), (180, 90), 100000, False, None),
]


def special_points(dst_gt):
    # GDALTransformLonLatToDestGenImgProjTransformer (gdaltransformer.cpp)
    # for a source CRS geographic in degrees only swaps axes if needed and
    # applies the target's inverse geotransform: longitude 0 and latitude
    # +-89.9999 are taken as target CRS coordinates, not reprojected (the
    # reprojection is in the else branch, since GDAL 3.6, 37b5f82b05).
    inv = gdal.InvGeoTransform(dst_gt)
    return [gdal.ApplyGeoTransform(inv, 0.0, lat) for lat in (-89.9999, 89.9999)]


def center_long(gt, w, h, dst_crs, te):
    # InsertCenterLong (gdaltransformer.cpp), formatted with %g as GDAL does
    xs = [gt[0] + p * gt[1] + l * gt[2] for p in (0, w) for l in (0, h)]
    lo, hi = min(xs), max(xs)
    if hi - lo > 360:
        d = osr.SpatialReference()
        d.SetFromUserInput(dst_crs)
        if not (d.IsGeographic() and ((te[0] >= -179 and te[0] < 180 and te[2] > 180)
                                      or (te[0] < -180 and te[2] > -180 and te[2] <= 179))):
            return None
    return float("%g" % ((hi + lo) / 2.0))


def probe(src, out, te, sw, sh, w, h):
    # target pixel -> source pixel, and source pixel -> target pixel, through
    # GDAL's exact transformer built as gdalwarp builds it
    opts = ["TARGET_EXTENT=%.17g,%.17g,%.17g,%.17g" % tuple(te)]
    t = gdal.Transformer(src, out, opts)
    rows = []
    for direction, (nx, ny) in ((1, (w, h)), (0, (sw, sh))):
        for j in range(9):
            for i in range(9):
                x, y = i * nx / 8.0, j * ny / 8.0
                ok, (ox, oy, _) = t.TransformPoint(direction, x, y)
                rows.append((direction, x, y, ok, ox, oy))
    return rows


def run(case):
    name, alg, (gt, sw, sh), dst_crs, te, (w, h), wm, nodata, block = case
    src = source(gt, sw, sh, "EPSG:4326", nodata)
    log = []

    def handler(cls, num, msg):
        if cls == gdal.CE_Debug:
            log.append(msg)

    gdal.PushErrorHandler(handler)
    gdal.SetConfigOption("CPL_DEBUG", "ON")
    try:
        if block:
            dst_path = "/vsimem/chunks_%s.tif" % name
            fmt = "GTiff"
            co = ["TILED=YES", "BLOCKXSIZE=%d" % block[0], "BLOCKYSIZE=%d" % block[1]]
        else:
            dst_path, fmt, co = "", "MEM", []
        kw = {} if wm is None else {"warpMemoryLimit": wm}
        out = gdal.Warp(dst_path, src, format=fmt, creationOptions=co, dstSRS=dst_crs,
                        outputBounds=te, width=w, height=h, resampleAlg=alg,
                        multithread=False, **kw)
    finally:
        gdal.SetConfigOption("CPL_DEBUG", None)
        gdal.PopErrorHandler()
    dst_gt = out.GetGeoTransform()
    data = out.GetRasterBand(1).ReadAsArray()
    chunks, scale = [], None
    for m in log:
        s = re.search(r"dfXScale = ([0-9.]+), dfYScale = ([0-9.]+)", m)
        if s:
            scale = (s.group(1), s.group(2))
        k = re.search(r"Src=(\d+),(\d+),(\d+)x(\d+) Dst=(\d+),(\d+),(\d+)x(\d+)", m)
        if k:
            chunks.append(k.groups() + scale)
    centre = center_long(gt, sw, sh, dst_crs, te)
    with open(os.path.join(HERE, name + ".txt"), "w") as f:
        f.write("# gdal %s\n" % gdal.__version__)
        f.write("# alg %s\n" % alg)
        f.write("# src_gt %s\n" % " ".join(repr(float(v)) for v in gt))
        f.write("# src_size %d %d\n" % (sw, sh))
        f.write("# dst_crs %s\n" % dst_crs)
        f.write("# dst_gt %s\n" % " ".join(repr(float(v)) for v in dst_gt))
        f.write("# te %s\n" % " ".join(repr(float(v)) for v in te))
        f.write("# wm %d\n" % (wm if wm is not None else 64 * 1024 * 1024))
        f.write("# nodata %s\n" % nodata)
        f.write("# block %s\n" % ("%d %d" % block if block else "-"))
        f.write("# center_long %s\n" % ("-" if centre is None else repr(centre)))
        for x, y in special_points(dst_gt):
            f.write("# special %r %r\n" % (x, y))
        # dst_to_src(1/0) x y ok out_x out_y
        for d, x, y, ok, ox, oy in probe(src, out, te, sw, sh, w, h):
            f.write("# point %d %r %r %d %r %r\n" % (d, x, y, ok, ox, oy))
        # src x y w h, dst x y w h, GDAL's kernel scale x y (as logged, %f)
        for c in chunks:
            f.write("# chunk %s\n" % " ".join(c))
        f.write("%d %d\n" % (w, h))
        for row in data:
            f.write(" ".join("%.12g" % v for v in row) + "\n")
    print(name, len(chunks), "chunks warped, center_long", centre)


for case in CASES:
    run(case)
