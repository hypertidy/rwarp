"""Regenerate the gdalwarp oracle fixtures for tests/kernels.rs.

    python3 tests/fixtures/kernels/gen.py

Needs the GDAL Python bindings. Writes src.txt (the 64x64 Float32 source)
and one <case>.txt per case with gdalwarp's output, plus the kernel scale
gdalwarp estimated (from its CPL_DEBUG "dfXScale = ..." line). Source and
destination share one extent and no CRS, so the transform is pure affine.
"""

import math
import os
import re

from osgeo import gdal

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
N = 64
NODATA = -9999.0


def source(holes):
    import array

    vals = array.array("f")
    for r in range(N):
        for c in range(N):
            # High-frequency content so kernel width matters.
            v = 100 * math.sin(c * 0.9) * math.cos(r * 0.7) + 3 * c + r
            if holes and (r * 7 + c * 3) % 23 == 0:
                v = NODATA
            vals.append(v)
    return vals


def mem_source(vals, nodata):
    ds = gdal.GetDriverByName("MEM").Create("", N, N, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, N, 0, -1])
    band = ds.GetRasterBand(1)
    band.WriteRaster(0, 0, N, N, vals.tobytes())
    if nodata:
        band.SetNoDataValue(NODATA)
    return ds


# name, alg, dst width, dst height, warp options, source has nodata
CASES = [
    ("bilinear_4x", "bilinear", 16, 16, {}, False),
    ("cubic_4x", "cubic", 16, 16, {}, False),
    ("lanczos_4x", "lanczos", 16, 16, {}, False),
    ("bilinear_2p78x", "bilinear", 23, 23, {}, False),
    ("cubic_2p78x", "cubic", 23, 23, {}, False),
    ("lanczos_2p78x", "lanczos", 23, 23, {}, False),
    ("bilinear_aniso", "bilinear", 16, 40, {}, False),
    ("cubic_aniso", "cubic", 16, 40, {}, False),
    ("lanczos_aniso", "lanczos", 16, 40, {}, False),
    # Between 1:1 and 2:1: bilinear keeps the 4-sample point kernel,
    # lanczos widens.
    ("bilinear_1p49x", "bilinear", 43, 43, {}, False),
    ("lanczos_1p49x", "lanczos", 43, 43, {}, False),
    # Widening turned off.
    ("bilinear_4x_point", "bilinear", 16, 16, {"XSCALE": "1", "YSCALE": "1"}, False),
    ("cubic_4x_point", "cubic", 16, 16, {"XSCALE": "1", "YSCALE": "1"}, False),
    ("lanczos_4x_point", "lanczos", 16, 16, {"XSCALE": "1", "YSCALE": "1"}, False),
    # Widened, skipping nodata.
    ("bilinear_4x_nodata", "bilinear", 16, 16, {}, True),
    ("cubic_4x_nodata", "cubic", 16, 16, {}, True),
    ("lanczos_4x_nodata", "lanczos", 16, 16, {}, True),
    # Rotated destination: the scale estimate takes its area branch.
    ("bilinear_rot30", "bilinear", 8, 8, {}, False),
    ("lanczos_rot30", "lanczos", 8, 8, {}, False),
    # Average (GDAL's box footprint).
    ("average_4x", "average", 16, 16, {}, False),
    ("average_2p78x", "average", 23, 23, {}, False),
    ("average_aniso", "average", 16, 40, {}, False),
    ("average_1p49x", "average", 43, 43, {}, False),
    ("average_upsample", "average", 128, 128, {}, False),
    ("average_overhang", "average", 20, 20, {}, False),
    ("average_4x_nodata", "average", 16, 16, {}, True),
    ("average_rot30", "average", 8, 8, {}, False),
]


def rot30_gt():
    """8x8 pixels of size 4 rotated 30 degrees about the source centre."""
    t, size = math.radians(30), 4.0
    g1, g2, g4, g5 = size * math.cos(t), size * math.sin(t), size * math.sin(t), -size * math.cos(t)
    return [32 - 4 * (g1 + g2), g1, g2, 32 - 4 * (g4 + g5), g4, g5]


def dst_dataset(name, w, h):
    """Destination for cases that need more than outputBounds."""
    if name.endswith("rot30"):
        gt = rot30_gt()
    elif name == "average_overhang":
        # Overhangs the source by 8 on every side.
        gt = [-8, 80 / w, 0, N + 8, 0, -80 / h]
    else:
        return None
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.GetRasterBand(1).SetNoDataValue(NODATA)
    ds.GetRasterBand(1).Fill(NODATA)
    return ds


def write_grid(path, header, w, h, vals):
    with open(path, "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write(f"{w} {h}\n")
        for r in range(h):
            f.write(" ".join(repr(float(v)) for v in vals[r * w:(r + 1) * w]) + "\n")


def main():
    import array

    write_grid(os.path.join(HERE, "src.txt"),
               [f"gdal {gdal.__version__}", "nodata cells hold -9999 in src_nodata.txt"],
               N, N, source(False))
    write_grid(os.path.join(HERE, "src_nodata.txt"), [f"gdal {gdal.__version__}"],
               N, N, source(True))

    gdal.SetConfigOption("CPL_DEBUG", "WARP")
    for name, alg, w, h, wo, holes in CASES:
        msgs = []
        gdal.PushErrorHandler(lambda cls, no, msg: msgs.append(msg))
        try:
            src = mem_source(source(holes), holes)
            dst = dst_dataset(name, w, h)
            if dst is None:
                out = gdal.Warp("", src, format="MEM",
                                outputBounds=[0, 0, N, N], width=w, height=h,
                                resampleAlg=alg, errorThreshold=0, outputType=gdal.GDT_Float32,
                                dstNodata=NODATA, warpOptions=wo)
            else:
                gdal.Warp(dst, src, resampleAlg=alg, errorThreshold=0, dstNodata=NODATA,
                          warpOptions=wo)
                out = dst
        finally:
            gdal.PopErrorHandler()
        scale = next((m for m in msgs if "dfXScale" in m), None)
        sx, sy = (re.findall(r"= ([0-9.]+)", scale) if scale else ("given", "given"))
        vals = array.array("f", out.GetRasterBand(1).ReadRaster(buf_type=gdal.GDT_Float32))
        opts = " ".join(f"{k}={v}" for k, v in wo.items()) or "-"
        write_grid(os.path.join(HERE, f"{name}.txt"),
                   [f"gdal {gdal.__version__}", f"alg {alg}", f"wo {opts}",
                    f"nodata_src {holes}", f"gdal_scale {sx} {sy}",
                    "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform())],
                   w, h, vals)
        print(name, sx, sy)


if __name__ == "__main__":
    main()
