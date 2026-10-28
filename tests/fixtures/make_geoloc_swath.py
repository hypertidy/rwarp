"""Generate tests/fixtures/geoloc_swath.rs: a synthetic curvilinear swath
warped with GDAL's geolocation transformer, as an oracle for rwarp.

Run from the crate root with GDAL's Python bindings available:

    python3 tests/fixtures/make_geoloc_swath.py

The swath is a small, curved, rotated longitude/latitude grid south of
Tasmania. Its data values encode the source pixel (col + 1000 * row), so a
nearest-neighbour warp shows exactly which source pixel GDAL chose for each
target pixel.

For each georeferencing convention the fixture records:

- the source pixel/line GDAL's transformer gives for every target pixel
  centre (gdal.Transformer, destination to source), with NaN where it fails;
- the nearest-neighbour warp (gdalwarp -geoloc -r near -et 0).
"""

import math

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

NCOL, NROW = 24, 18
DST_CRS = "+proj=laea +lat_0=-45 +lon_0=146 +datum=WGS84 +units=m +no_defs"
DST_NCOL, DST_NROW = 40, 32
# Target grid: 10 km pixels, chosen to overhang the swath on every side.
DST_GT = (-420000.0, 10000.0, 0.0, 330000.0, 0.0, -10000.0)
NODATA = -1


def swath():
    i, j = np.meshgrid(np.arange(NCOL, dtype=float), np.arange(NROW, dtype=float))
    lon = 142.0 + 0.30 * i + 0.08 * j + 0.004 * j * j
    lat = -42.0 - 0.22 * j + 0.05 * i - 0.002 * i * i
    return lon, lat


def mem_band(name, arr, dtype):
    ds = gdal.GetDriverByName("GTiff").Create(name, arr.shape[1], arr.shape[0], 1, dtype)
    ds.GetRasterBand(1).WriteArray(arr)
    return ds


def run(convention):
    lon, lat = swath()
    xds = mem_band("/vsimem/lon.tif", lon, gdal.GDT_Float64)
    yds = mem_band("/vsimem/lat.tif", lat, gdal.GDT_Float64)
    xds = None
    yds = None

    data = (np.arange(NCOL)[None, :] + 1000 * np.arange(NROW)[:, None]).astype(np.int32)
    src = mem_band("/vsimem/src.tif", data, gdal.GDT_Int32)
    wgs84 = osr.SpatialReference()
    wgs84.SetWellKnownGeogCS("WGS84")
    src.SetMetadata(
        {
            "SRS": wgs84.ExportToWkt(),
            "X_DATASET": "/vsimem/lon.tif",
            "X_BAND": "1",
            "Y_DATASET": "/vsimem/lat.tif",
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
            "GEOREFERENCING_CONVENTION": convention,
        },
        "GEOLOCATION",
    )
    src.GetRasterBand(1).SetNoDataValue(NODATA)
    src.FlushCache()

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromProj4(DST_CRS)
    dst = gdal.GetDriverByName("MEM").Create("", DST_NCOL, DST_NROW, 1, gdal.GDT_Int32)
    dst.SetGeoTransform(DST_GT)
    dst.SetProjection(dst_srs.ExportToWkt())
    dst.GetRasterBand(1).SetNoDataValue(NODATA)
    dst.GetRasterBand(1).Fill(NODATA)

    # Coordinates: destination pixel centres to source pixel/line.
    tr = gdal.Transformer(src, dst, ["METHOD=GEOLOC_ARRAY"])
    pts = [(c + 0.5, r + 0.5) for r in range(DST_NROW) for c in range(DST_NCOL)]
    out, ok = tr.TransformPoints(1, pts)
    sx = [p[0] if s else math.nan for p, s in zip(out, ok)]
    sy = [p[1] if s else math.nan for p, s in zip(out, ok)]

    # Values: nearest-neighbour warp, no approximation.
    gdal.Warp(
        dst,
        src,
        options=gdal.WarpOptions(
            geoloc=True, resampleAlg="near", errorThreshold=0, srcNodata=NODATA, dstNodata=NODATA
        ),
    )
    vals = dst.GetRasterBand(1).ReadAsArray().ravel().tolist()
    return sx, sy, vals


def run_dst_geoloc():
    """The other direction: an affine LAEA raster warped onto the swath,
    with the swath as the target's geolocation arrays (PIXEL_CENTER),
    through DST_METHOD=GEOLOC_ARRAY."""
    lon, lat = swath()
    mem_band("/vsimem/lon.tif", lon, gdal.GDT_Float64).FlushCache()
    mem_band("/vsimem/lat.tif", lat, gdal.GDT_Float64).FlushCache()
    wgs84 = osr.SpatialReference()
    wgs84.SetWellKnownGeogCS("WGS84")

    dst = gdal.GetDriverByName("MEM").Create("", NCOL, NROW, 1, gdal.GDT_Int32)
    dst.SetMetadata(
        {
            "SRS": wgs84.ExportToWkt(),
            "X_DATASET": "/vsimem/lon.tif",
            "X_BAND": "1",
            "Y_DATASET": "/vsimem/lat.tif",
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
            "GEOREFERENCING_CONVENTION": "PIXEL_CENTER",
        },
        "GEOLOCATION",
    )
    dst.GetRasterBand(1).SetNoDataValue(NODATA)
    dst.GetRasterBand(1).Fill(NODATA)

    laea = osr.SpatialReference()
    laea.ImportFromProj4(DST_CRS)
    src = gdal.GetDriverByName("MEM").Create("", DST_NCOL, DST_NROW, 1, gdal.GDT_Int32)
    src.SetGeoTransform(DST_GT)
    src.SetProjection(laea.ExportToWkt())
    data = (np.arange(DST_NCOL)[None, :] + 1000 * np.arange(DST_NROW)[:, None]).astype(np.int32)
    src.GetRasterBand(1).WriteArray(data)
    src.GetRasterBand(1).SetNoDataValue(NODATA)

    tr = gdal.Transformer(src, dst, ["DST_METHOD=GEOLOC_ARRAY"])
    # Target (swath) pixel centres to source: the forward geolocation path.
    pts = [(c + 0.5, r + 0.5) for r in range(NROW) for c in range(NCOL)]
    out, ok = tr.TransformPoints(1, pts)
    d2s_x = [p[0] if s else math.nan for p, s in zip(out, ok)]
    d2s_y = [p[1] if s else math.nan for p, s in zip(out, ok)]
    # Source pixel centres to target: the inverse geolocation path.
    pts = [(c + 0.5, r + 0.5) for r in range(DST_NROW) for c in range(DST_NCOL)]
    out, ok = tr.TransformPoints(0, pts)
    s2d_x = [p[0] if s else math.nan for p, s in zip(out, ok)]
    s2d_y = [p[1] if s else math.nan for p, s in zip(out, ok)]

    gdal.Warp(
        dst,
        src,
        options=gdal.WarpOptions(
            resampleAlg="near",
            errorThreshold=0,
            srcNodata=NODATA,
            dstNodata=NODATA,
            transformerOptions=["DST_METHOD=GEOLOC_ARRAY"],
        ),
    )
    vals = dst.GetRasterBand(1).ReadAsArray().ravel().tolist()
    return d2s_x, d2s_y, s2d_x, s2d_y, vals


def rs_f64(name, xs):
    body = ", ".join("f64::NAN" if math.isnan(v) else repr(float(v)) for v in xs)
    return f"pub const {name}: [f64; {len(xs)}] = [{body}];\n"


def rs_i32(name, xs):
    return f"pub const {name}: [i32; {len(xs)}] = [{', '.join(str(int(v)) for v in xs)}];\n"


def main():
    lon, lat = swath()
    parts = [
        "// Generated by tests/fixtures/make_geoloc_swath.py with GDAL "
        + gdal.__version__
        + ". Do not edit.\n",
        "#![allow(dead_code, clippy::all)]\n",
        f"pub const NCOL: usize = {NCOL};\n",
        f"pub const NROW: usize = {NROW};\n",
        f"pub const DST_NCOL: usize = {DST_NCOL};\n",
        f"pub const DST_NROW: usize = {DST_NROW};\n",
        f'pub const DST_CRS: &str = "{DST_CRS}";\n',
        "pub const DST_GT: [f64; 6] = [" + ", ".join(repr(float(v)) for v in DST_GT) + "];\n",
        f"pub const NODATA: i32 = {NODATA};\n",
        rs_f64("LON", lon.ravel().tolist()),
        rs_f64("LAT", lat.ravel().tolist()),
    ]
    for conv, tag in (("TOP_LEFT_CORNER", "TLC"), ("PIXEL_CENTER", "PC")):
        sx, sy, vals = run(conv)
        parts.append(rs_f64(f"{tag}_SRC_X", sx))
        parts.append(rs_f64(f"{tag}_SRC_Y", sy))
        parts.append(rs_i32(f"{tag}_NEAR", vals))
    d2s_x, d2s_y, s2d_x, s2d_y, vals = run_dst_geoloc()
    parts.append(rs_f64("DSTGEO_D2S_X", d2s_x))
    parts.append(rs_f64("DSTGEO_D2S_Y", d2s_y))
    parts.append(rs_f64("DSTGEO_S2D_X", s2d_x))
    parts.append(rs_f64("DSTGEO_S2D_Y", s2d_y))
    parts.append(rs_i32("DSTGEO_NEAR", vals))
    with open("tests/fixtures/geoloc_swath.rs", "w", encoding="ascii") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
