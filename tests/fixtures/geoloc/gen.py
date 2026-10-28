"""Regenerate the gdalwarp oracle fixtures for tests/geoloc_backmap.rs.

    python3 tests/fixtures/geoloc/gen.py

Needs the GDAL Python bindings and numpy (each <case>.txt records the GDAL
version that wrote it). The geolocation arrays are computed here with
plain Python floats, in the order tests/geoloc_backmap.rs computes them, so
both sides hold bitwise the same arrays (each file records a hash as a
check). Each case is a curvilinear lon/lat source (EPSG:4326 geolocation
arrays) warped by gdalwarp with the geolocation transformer onto an affine
target, and records:

- `coords`: for every target pixel centre, the source pixel/line from
  GDAL's transformer (destination to source), `nan nan` where it fails;
- `approx` (some cases): the same with GDAL_GEOLOC_USE_MAX_ACCURACY=NO,
  which is GDAL's backmap estimate alone;
- one block of warped values per resampling method, with no approximation
  (-et 0), and for some cases `<method>_et` blocks with gdalwarp's default
  approximation (-et 0.125). Nearest neighbour warps the source pixel id
  (col + 1000 * row); the other methods warp the arithmetic field
  `value()`. Source nodata -9999 where `src_nodata()` says, if the case
  has source nodata.

The grids, from wxylib/fixtures' curvilinear set (scripts/
make_curvilinear_fixtures.py, which wxy-warp's geoloc.rs was checked
against) with sin replaced by polynomials so that Rust can rebuild them
exactly: a curved swath (cell centres, and cell corners with
TOP_LEFT_CORNER), a ring of cells around the south pole crossing the
antimeridian, plus a swath across the antimeridian, a swath with holes in
its geolocation arrays and source nodata, and a "bow-tie" swath whose
scans overlap (folds), where GDAL's backmap inverse and rwarp's quad index
choose differently.
"""

import os

from osgeo import gdal, osr

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
NODATA = -9999.0
GEOLOC_NODATA = -999.0


def value(c, r):
    # As tests/fixtures/chunks/gen.py: exact in f64 but for the /10.
    return ((c * 7919 + r * 104729) % 1000) / 10.0 + c * 0.5 - r * 0.25


def src_nodata(c, r):
    return (c * 3 + r * 5) % 17 == 0


def geoloc_missing(c, r):
    """Holes in the 'holes' grid: a block (a dropped scan segment), a
    scatter of single samples, and the start of the first column."""
    return (20 <= r < 24 and 30 <= c < 42) or (r * 7 + c * 13) % 37 == 0 or (c == 0 and r < 10)


def wrap(lon):
    # Python's float % is floored, as Rust's rem_euclid for a positive divisor.
    return (lon + 180.0) % 360.0 - 180.0


def grid_point(kind, nx, ny, c, r, shift):
    """Longitude and latitude of array sample (c, r) of an nx by ny array."""
    i = c + shift
    j = r + shift
    if kind in ("swath", "holes"):
        h = ny / 2.0
        t = i / nx
        lon = 140.0 + 0.275 * i + 0.075 * j + 0.0025 * ((j - h) * (j - h))
        lat = -40.0 - 0.225 * j + 0.05 * i + 6.0 * (t * (1.0 - t))
    elif kind == "polar":
        lon = wrap(-180.0 + 360.0 * (i + 0.5) / nx + 0.2 * j)
        lat = -60.0 - 25.0 * (j + 0.5) / ny
    elif kind == "dateline":
        h = ny / 2.0
        lon = wrap(168.0 + 0.35 * i + 0.06 * j + 0.0015 * ((j - h) * (j - h)))
        lat = -45.0 - 0.25 * j + 0.04 * i
    elif kind == "bowtie":
        # Scans of 6 rows. Each scan starts o rows back from where the last
        # ended: no overlap at the swath centre (o = 0.5), overlap towards
        # its edges (o up to 1.5), as MODIS bow-ties.
        e = (i - nx / 2.0) / (nx / 2.0)
        o = 0.5 + 1.0 * (e * e)
        k = float(r // 6)
        lon = 150.0 + 0.3 * i + 0.02 * j
        lat = -50.0 - 0.3 * j + 0.3 * (o * k)
    else:
        raise ValueError(kind)
    return lon, lat


def arrays(kind, nx, ny, shift, nodata):
    lon, lat = [], []
    for r in range(ny):
        for c in range(nx):
            x, y = grid_point(kind, nx, ny, c, r, shift)
            if nodata and geoloc_missing(c, r):
                x, y = GEOLOC_NODATA, GEOLOC_NODATA
            lon.append(x)
            lat.append(y)
    return lon, lat


def fnv1a(vals):
    """64-bit FNV-1a over the little-endian bytes of float64 values."""
    import struct

    h = 0xCBF29CE484222325
    for b in struct.pack("<%dd" % len(vals), *vals):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def srs_wkt(crs):
    srs = osr.SpatialReference()
    srs.SetFromUserInput(crs)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs.ExportToWkt()


def mem_band(name, vals, nx, ny, nodata=None):
    import array

    ds = gdal.GetDriverByName("GTiff").Create(name, nx, ny, 1, gdal.GDT_Float64)
    band = ds.GetRasterBand(1)
    band.WriteRaster(0, 0, nx, ny, array.array("d", vals).tobytes())
    if nodata is not None:
        band.SetNoDataValue(nodata)
    ds.FlushCache()
    return ds


def source(case, field):
    """The source raster (Float64) with its GEOLOCATION metadata."""
    import array

    nx, ny = case["raster"]
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Float64)
    vals = []
    for r in range(ny):
        for c in range(nx):
            if case["src_nodata"] and src_nodata(c, r):
                vals.append(NODATA)
            elif field == "id":
                vals.append(float(c + 1000 * r))
            else:
                vals.append(value(c, r))
    band = ds.GetRasterBand(1)
    band.WriteRaster(0, 0, nx, ny, array.array("d", vals).tobytes())
    band.SetNoDataValue(NODATA)
    ds.SetMetadata(
        {
            "SRS": srs_wkt("EPSG:4326"),
            "X_DATASET": "/vsimem/geoloc_lon.tif",
            "X_BAND": "1",
            "Y_DATASET": "/vsimem/geoloc_lat.tif",
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
            "GEOREFERENCING_CONVENTION": case["convention"],
        },
        "GEOLOCATION",
    )
    return ds


def target(case):
    w, h = case["dst_size"]
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(case["dst_gt"])
    ds.SetProjection(srs_wkt(case["dst_crs"]))
    ds.GetRasterBand(1).SetNoDataValue(NODATA)
    ds.GetRasterBand(1).Fill(NODATA)
    return ds


def coords(src, dst, approx):
    w, h = dst.RasterXSize, dst.RasterYSize
    if approx:
        gdal.SetConfigOption("GDAL_GEOLOC_USE_MAX_ACCURACY", "NO")
    try:
        tr = gdal.Transformer(src, dst, ["METHOD=GEOLOC_ARRAY"])
        pts = [(c + 0.5, r + 0.5) for r in range(h) for c in range(w)]
        out, ok = tr.TransformPoints(1, pts)
    finally:
        gdal.SetConfigOption("GDAL_GEOLOC_USE_MAX_ACCURACY", None)
    rows = []
    for p, s in zip(out, ok):
        rows.append("%.12g %.12g" % (p[0], p[1]) if s else "nan nan")
    return rows


def gt_for(bounds, size):
    xmin, ymin, xmax, ymax = bounds
    w, h = size
    return [xmin, (xmax - xmin) / w, 0.0, ymax, 0.0, -(ymax - ymin) / h]


SWATH_4326 = (140.0, -52.0, 164.0, -35.0)
SWATH_3031 = (1.3e6, -6.0e6, 3.8e6, -3.4e6)
DATELINE_LAEA = "+proj=laea +lat_0=-50 +lon_0=180 +datum=WGS84 +units=m +no_defs"

# name, grid, array size (nx, ny), shift, convention, raster size,
# geoloc nodata, source nodata, dst crs, dst bounds, dst size, methods,
# approx coords
CASES = [
    ("swath_pc_4326", "swath", (64, 48), 0.0, "PIXEL_CENTER", (64, 48), False, False,
     "EPSG:4326", SWATH_4326, (80, 57), ("near", "bilinear", "cubic", "near_et", "bilinear_et"),
     True),
    ("swath_pc_4326_coarse", "swath", (64, 48), 0.0, "PIXEL_CENTER", (64, 48), False, False,
     "EPSG:4326", SWATH_4326, (30, 20), ("near", "bilinear", "cubic"), True),
    ("swath_pc_3031", "swath", (64, 48), 0.0, "PIXEL_CENTER", (64, 48), False, False,
     "EPSG:3031", SWATH_3031, (50, 52), ("near", "bilinear"), False),
    # cell corners: arrays one larger than the raster, TOP_LEFT_CORNER
    ("swath_tlc_4326", "swath", (65, 49), -0.5, "TOP_LEFT_CORNER", (64, 48), False, False,
     "EPSG:4326", SWATH_4326, (80, 57), ("near", "bilinear"), False),
    ("swath_tlc_3031", "swath", (65, 49), -0.5, "TOP_LEFT_CORNER", (64, 48), False, False,
     "EPSG:3031", SWATH_3031, (50, 52), ("near", "bilinear"), False),
    ("polar_pc_3031", "polar", (180, 50), 0.0, "PIXEL_CENTER", (180, 50), False, False,
     "EPSG:3031", (-3.5e6, -3.5e6, 3.5e6, 3.5e6), (70, 70),
     ("near", "bilinear", "near_et", "bilinear_et"), True),
    ("dateline_pc_laea", "dateline", (48, 40), 0.0, "PIXEL_CENTER", (48, 40), False, False,
     DATELINE_LAEA, (-1.0e6, -0.7e6, 0.6e6, 0.9e6), (64, 64), ("near", "bilinear"), False),
    ("holes_pc_4326", "holes", (64, 48), 0.0, "PIXEL_CENTER", (64, 48), True, True,
     "EPSG:4326", SWATH_4326, (80, 57), ("near", "bilinear"), True),
    ("bowtie_tlc_4326", "bowtie", (40, 36), 0.0, "TOP_LEFT_CORNER", (40, 36), False, False,
     "EPSG:4326", (149.0, -61.0, 164.0, -49.0), (60, 48), ("near", "bilinear"), True),
]


def run(case):
    nx, ny = case["size"]
    lon, lat = arrays(case["grid"], nx, ny, case["shift"], case["geoloc_nodata"])
    nd = GEOLOC_NODATA if case["geoloc_nodata"] else None
    mem_band("/vsimem/geoloc_lon.tif", lon, nx, ny, nd)
    mem_band("/vsimem/geoloc_lat.tif", lat, nx, ny, nd)

    w, h = case["dst_size"]
    gt = gt_for(case["bounds"], case["dst_size"])
    case["dst_gt"] = gt
    lines = [
        "# gdal " + gdal.__version__,
        "# grid %s" % case["grid"],
        "# size %d %d" % (nx, ny),
        "# shift %r" % case["shift"],
        "# convention %s" % case["convention"],
        "# raster %d %d" % case["raster"],
        "# geoloc_nodata %s" % (repr(GEOLOC_NODATA) if case["geoloc_nodata"] else "-"),
        "# src_nodata %s" % case["src_nodata"],
        # To check that Rust rebuilds bitwise the same arrays.
        "# fnv1a %d" % fnv1a(lon + lat),
        "# dst_crs %s" % case["dst_crs"],
        "# dst_gt %s" % " ".join(repr(float(v)) for v in gt),
        "# dst_size %d %d" % (w, h),
    ]
    src = source(case, "id")
    lines.append("@ coords")
    lines += coords(src, target(case), False)
    if case["approx"]:
        lines.append("@ approx")
        lines += coords(src, target(case), True)
    for section in case["methods"]:
        m, et = (section[:-3], 0.125) if section.endswith("_et") else (section, 0)
        src = source(case, "id" if m == "near" else "value")
        dst = target(case)
        gdal.Warp(
            dst,
            src,
            options=gdal.WarpOptions(
                resampleAlg=m,
                errorThreshold=et,
                srcNodata=NODATA,
                dstNodata=NODATA,
                multithread=False,
                transformerOptions=["METHOD=GEOLOC_ARRAY"],
            ),
        )
        vals = dst.GetRasterBand(1).ReadAsArray()
        lines.append("@ " + section)
        fmt = "%d" if m == "near" else "%.12g"
        for row in vals:
            lines.append(" ".join(fmt % v for v in row))
    gdal.Unlink("/vsimem/geoloc_lon.tif")
    gdal.Unlink("/vsimem/geoloc_lat.tif")
    with open(os.path.join(HERE, case["name"] + ".txt"), "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def main():
    for (name, grid, size, shift, conv, raster, gnd, snd, crs, bounds, dst_size, methods,
         approx) in CASES:
        run(dict(name=name, grid=grid, size=size, shift=shift, convention=conv, raster=raster,
                 geoloc_nodata=gnd, src_nodata=snd, dst_crs=crs, bounds=bounds,
                 dst_size=dst_size, methods=methods, approx=approx))


if __name__ == "__main__":
    main()
