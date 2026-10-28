"""Regenerate the gdalwarp oracle fixtures for tests/chunks.rs.

    python3 tests/fixtures/chunks/gen.py

Needs the GDAL Python bindings (the fixtures were made with the GDAL named
in each file's header). Each case warps a whole lon/lat source to a
projected target with a small warp memory limit (gdalwarp's -wm, in
bytes), so GDAL splits the target into many chunks. Each <case>.txt holds
gdalwarp's output and, from GDAL's CPL_DEBUG log, the chunks it warped
(source and target windows, kernel scale), plus the special points
(longitude 0 at latitude +-89.9999 in target pixels) GDAL computes from
the CRSs. GDAL logs only chunks that read something. The source is the arithmetic field in `value()` below, which
tests/chunks.rs computes the same way.
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
# The LAEA target overhangs the regional source on every side but stays
# clear of 180 degrees: GDAL wraps a geographic source's longitudes around
# its centre (CENTER_LONG), which rwarp's transformer does not.
REGION = ([85, 0.25, 0, -10, 0, -0.25], 240, 160)
LAEA = "+proj=laea +lat_0=-30 +lon_0=115 +datum=WGS84 +units=m +no_defs"
MERC_X = 20037508.342789244
MERC_Y = 15538711.096309224  # about 80 degrees

# name, alg, source, dst crs, dst extent (xmin, ymin, xmax, ymax), dst size,
# -wm bytes, source nodata, target block size (None: MEM)
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
]


def special_points(src_crs, dst_crs, dst_gt):
    # GDALTransformLonLatToDestGenImgProjTransformer for a geographic
    # source: lon/lat as they are, forward to the target, then its pixels.
    s = osr.SpatialReference()
    s.SetFromUserInput(src_crs)
    d = osr.SpatialReference()
    d.SetFromUserInput(dst_crs)
    for x in (s, d):
        x.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(s, d)
    inv = gdal.InvGeoTransform(dst_gt)
    out = []
    for lat in (-89.9999, 89.9999):
        try:
            x, y, _ = ct.TransformPoint(0.0, lat)
        except RuntimeError:
            continue
        if x == float("inf") or y == float("inf"):
            continue
        out.append(gdal.ApplyGeoTransform(inv, x, y))
    return out


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
        out = gdal.Warp(dst_path, src, format=fmt, creationOptions=co, dstSRS=dst_crs,
                        outputBounds=te, width=w, height=h, resampleAlg=alg,
                        warpMemoryLimit=wm, multithread=False)
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
    with open(os.path.join(HERE, name + ".txt"), "w") as f:
        f.write("# gdal %s\n" % gdal.__version__)
        f.write("# alg %s\n" % alg)
        f.write("# src_gt %s\n" % " ".join(repr(float(v)) for v in gt))
        f.write("# src_size %d %d\n" % (sw, sh))
        f.write("# dst_crs %s\n" % dst_crs)
        f.write("# dst_gt %s\n" % " ".join(repr(float(v)) for v in dst_gt))
        f.write("# wm %d\n" % wm)
        f.write("# nodata %s\n" % nodata)
        f.write("# block %s\n" % ("%d %d" % block if block else "-"))
        for x, y in special_points("EPSG:4326", dst_crs, dst_gt):
            f.write("# special %r %r\n" % (x, y))
        # src x y w h, dst x y w h, GDAL's kernel scale x y (as logged, %f)
        for c in chunks:
            f.write("# chunk %s\n" % " ".join(c))
        f.write("%d %d\n" % (w, h))
        for row in data:
            f.write(" ".join("%.12g" % v for v in row) + "\n")
    print(name, len(chunks), "chunks warped")


for case in CASES:
    run(case)
