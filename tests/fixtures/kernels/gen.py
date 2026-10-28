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

    main_mode()
    main_round()
    main_nodata()
    main_methods()


# ---- mode -----------------------------------------------------------------
#
# A categorical source (classes 1..7, many ties at 4:1) in four types:
# GDAL's mode has a histogram path for 8/16-bit working types and a
# list-of-values path for wider ones, which break ties differently.

MODE_TYPES = {"Byte": gdal.GDT_Byte, "Int16": gdal.GDT_Int16,
              "Int32": gdal.GDT_Int32, "Float32": gdal.GDT_Float32,
              "Int64": gdal.GDT_Int64, "UInt64": gdal.GDT_UInt64,
              "Float16": gdal.GDT_Float16}


def categorical(holes):
    vals = []
    for r in range(N):
        for c in range(N):
            v = ((r // 2) * 3 + (c // 3) * 5 + (r * c) % 3) % 7 + 1
            if holes and (r * 5 + c * 2) % 13 == 0:
                v = 0
            vals.append(v)
    return vals


def distinct_blocks():
    """Every 2x2 block holds four distinct values, the smallest and largest
    at varying positions, so 2:1 mode is a tie at count 1 everywhere."""
    order = [40, 10, 30, 20]
    vals = []
    for r in range(N):
        for c in range(N):
            b = ((r // 2) * 7 + (c // 2) * 3) % 4
            vals.append(order[((r % 2) * 2 + c % 2 + b) % 4] + (r // 2 + c // 2) % 5)
    return vals


def mem_categorical(dtype, holes, vals=None):
    import array

    ds = gdal.GetDriverByName("MEM").Create("", N, N, 1, dtype)
    ds.SetGeoTransform([0, 1, 0, N, 0, -1])
    band = ds.GetRasterBand(1)
    vals = categorical(holes) if vals is None else vals
    band.WriteRaster(0, 0, N, N, array.array("d", vals).tobytes(),
                     buf_type=gdal.GDT_Float64)
    if holes:
        band.SetNoDataValue(0)
    return ds


def main_mode():
    import array

    gdal.SetConfigOption("CPL_DEBUG", None)

    write_grid(os.path.join(HERE, "src_cat.txt"),
               [f"gdal {gdal.__version__}", "categorical, classes 1..7"], N, N, categorical(False))
    write_grid(os.path.join(HERE, "src_cat_nodata.txt"),
               [f"gdal {gdal.__version__}", "categorical, nodata 0"], N, N, categorical(True))
    cases = []
    for tname in MODE_TYPES:
        for ties in ("FIRST", "MIN", "MAX"):
            for size, w in (("4x", 16), ("2p78x", 23)):
                cases.append((f"mode_{tname.lower()}_{ties.lower()}_{size}", tname, ties, w, False, False))
    cases.append(("mode_byte_min_4x_nodata", "Byte", "MIN", 16, True, False))
    cases.append(("mode_byte_first_rot30", "Byte", "FIRST", 8, False, True))
    # Ties at a value's first occurrence, where GDAL's types disagree.
    write_grid(os.path.join(HERE, "src_distinct.txt"),
               [f"gdal {gdal.__version__}", "distinct values in every 2x2 block"], N, N, distinct_blocks())
    for tname in ("Byte", "Int32", "Float32"):
        for ties in ("FIRST", "MIN", "MAX"):
            cases.append((f"mode_distinct_{tname.lower()}_{ties.lower()}", tname, ties, 32, False, False))
    for name, tname, ties, w, holes, rotated in cases:
        dtype = MODE_TYPES[tname]
        src = mem_categorical(dtype, holes, distinct_blocks() if "distinct" in name else None)
        wo = {"MODE_TIES": ties}
        if rotated:
            out = gdal.GetDriverByName("MEM").Create("", w, w, 1, dtype)
            out.SetGeoTransform(rot30_gt())
            gdal.Warp(out, src, resampleAlg="mode", errorThreshold=0, warpOptions=wo)
        else:
            out = gdal.Warp("", src, format="MEM", outputBounds=[0, 0, N, N], width=w, height=w,
                            resampleAlg="mode", errorThreshold=0, warpOptions=wo,
                            dstNodata=0 if holes else None)
        vals = array.array("d", out.GetRasterBand(1).ReadRaster(buf_type=gdal.GDT_Float64))
        write_grid(os.path.join(HERE, f"{name}.txt"),
                   [f"gdal {gdal.__version__}", "alg mode", f"src_type {tname}", f"ties {ties}",
                    "src " + ("src_distinct" if "distinct" in name else "src_cat_nodata" if holes else "src_cat"),
                    f"nodata_src {holes}",
                    "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform())],
                   w, w, vals)
    print(len(cases), "mode cases")



# ---- rounding, clamping, nodata avoidance, source bounds -----------------

def pattern_round(r, c):
    """Column pairs whose mean is k + 0.5 for k of both signs."""
    return (c % 2) - 3 + 10 * ((r % 4) - 2)


def pattern_avoid(r, c):
    """Column pairs averaging -12 (the nodata value, never in the source) on
    even rows and 6 on odd rows."""
    if r % 2 == 0:
        return -13 if c % 2 == 0 else -11
    return 5 if c % 2 == 0 else 7


def pattern_avoid_float(r, c):
    """Column pairs averaging exactly 0.0 (the nodata value)."""
    return -1.0 if c % 2 == 0 else 1.0


def pattern_step(r, c):
    """A 0/255 step, which cubic overshoots both ways."""
    return 0 if c < 32 else 255


ROUND_PATTERNS = {"round": pattern_round, "avoid": pattern_avoid,
                  "avoid_float": pattern_avoid_float, "step": pattern_step}


def main_round():
    import array

    gdal.SetConfigOption("CPL_DEBUG", None)
    for pname, f in ROUND_PATTERNS.items():
        write_grid(os.path.join(HERE, f"src_{pname}.txt"), [f"gdal {gdal.__version__}"],
                   N, N, [f(r, c) for r in range(N) for c in range(N)])
    # name, src pattern, type, alg, bounds, width, nodata, warp options
    point = {"XSCALE": "1", "YSCALE": "1"}
    cases = [
        # Means at k + 0.5: GDAL rounds half up (-2.5 -> -2).
        ("round_int16_average_2x", "round", "Int16", "average", [0, 0, N, N], 32, None, {}),
        ("round_int16_bilinear_2x", "round", "Int16", "bilinear", [0, 0, N, N], 32, None, {}),
        ("round_int32_average_2x", "round", "Int32", "average", [0, 0, N, N], 32, None, {}),
        # Cubic overshoot on a step, half a pixel off: clamped to 0..255.
        ("clamp_byte_cubic_shift", "step", "Byte", "cubic", [0.5, 0, N + 0.5, N], N, None, point),
        ("clamp_int16_cubic_shift", "step", "Int16", "cubic", [0.5, 0, N + 0.5, N], N, None, point),
        # A resampled value equal to nodata is moved off it.
        ("avoid_int16_average_2x", "avoid", "Int16", "average", [0, 0, N, N], 32, -12, {}),
        ("avoid_int16_bilinear_2x", "avoid", "Int16", "bilinear", [0, 0, N, N], 32, -12, {}),
        ("avoid_float32_average_2x", "avoid_float", "Float32", "average", [0, 0, N, N], 32, 0, {}),
        # Destination pixel centres on the source edge: [0, n) is inside.
        ("bounds_bilinear_left", "round", "Int16", "bilinear", [-1, 0, N - 1, N], N, -9999, point),
        ("bounds_bilinear_left_half", "round", "Int16", "bilinear", [-0.5, 0, N - 0.5, N], N, -9999, point),
        ("bounds_bilinear_right_half", "round", "Int16", "bilinear", [0.5, 0, N + 0.5, N], N, -9999, point),
        ("bounds_lanczos_left", "round", "Int16", "lanczos", [-1, 0, N - 1, N], N, -9999, point),
    ]
    for name, pname, tname, alg, bounds, w, nodata, wo in cases:
        f = ROUND_PATTERNS[pname]
        dtype = MODE_TYPES[tname]
        src = gdal.GetDriverByName("MEM").Create("", N, N, 1, dtype)
        src.SetGeoTransform([0, 1, 0, N, 0, -1])
        src.GetRasterBand(1).WriteRaster(
            0, 0, N, N, array.array("d", [f(r, c) for r in range(N) for c in range(N)]).tobytes(),
            buf_type=gdal.GDT_Float64)
        if nodata is not None:
            src.GetRasterBand(1).SetNoDataValue(nodata)
        out = gdal.Warp("", src, format="MEM", outputBounds=bounds, width=w, height=N,
                        resampleAlg=alg, errorThreshold=0, warpOptions=wo,
                        dstNodata=nodata)
        vals = array.array("d", out.GetRasterBand(1).ReadRaster(buf_type=gdal.GDT_Float64))
        opts = " ".join(f"{k}={v}" for k, v in wo.items()) or "-"
        write_grid(os.path.join(HERE, f"{name}.txt"),
                   [f"gdal {gdal.__version__}", f"alg {alg}", f"src src_{pname}", f"src_type {tname}",
                    f"nodata {'-' if nodata is None else nodata}", f"wo {opts}",
                    "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform())],
                   w, N, vals)
    print(len(cases), "rounding cases")



# ---- nodata: point kernels, multiple bands, average thresholds ------------

def write_bands(path, header, w, h, bands):
    """Like write_grid, band after band, with a `bands` header line."""
    with open(path, "w") as f:
        for line in header + [f"bands {len(bands)}"]:
            f.write(f"# {line}\n")
        f.write(f"{w} {h}\n")
        for vals in bands:
            for r in range(h):
                f.write(" ".join(repr(float(v)) for v in vals[r * w:(r + 1) * w]) + "\n")


def multiband():
    """Three Float32 bands with nodata holes in different places, including
    pixels where every band is nodata."""
    base = source(False)
    holes = [lambda r, c: (r * 7 + c * 3) % 23 == 0,
             lambda r, c: (r * 5 + c * 11) % 19 == 0,
             lambda r, c: (r * 7 + c * 3) % 23 == 0 and (r + c) % 2 == 0 or (r * 3 + c) % 17 == 0]
    bands = []
    for b in range(3):
        bands.append([NODATA if holes[b](i // N, i % N) else base[i] + 10 * b for i in range(N * N)])
    # Every band nodata here.
    for b in range(3):
        for i in range(0, N * N, 29):
            bands[b][i] = NODATA
    return bands


def multiband_categorical():
    """Three bands of one categorical field, so tuples are (k, k, k)."""
    cat = categorical(False)
    return [list(cat), list(cat), list(cat)]


def mem_bands(bands, dtype, nodata):
    import array

    ds = gdal.GetDriverByName("MEM").Create("", N, N, len(bands), dtype)
    ds.SetGeoTransform([0, 1, 0, N, 0, -1])
    for b, vals in enumerate(bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteRaster(0, 0, N, N, array.array("d", vals).tobytes(), buf_type=gdal.GDT_Float64)
        if nodata is not None:
            # From the dataset, not -srcnodata, so gdalwarp leaves
            # UNIFIED_SRC_NODATA unset.
            band.SetNoDataValue(nodata)
    return ds


def main_nodata():
    import array

    gdal.SetConfigOption("CPL_DEBUG", None)
    write_bands(os.path.join(HERE, "src_mb.txt"), [f"gdal {gdal.__version__}"], N, N, multiband())
    write_bands(os.path.join(HERE, "src_mb_cat.txt"), [f"gdal {gdal.__version__}"], N, N,
                multiband_categorical())
    point = {"XSCALE": "1", "YSCALE": "1"}
    # name, src, type, alg, width, nodata, warp options
    cases = [
        # Single band, point kernels next to nodata: bilinear renormalises
        # over valid neighbours, cubic falls back to it, lanczos skips.
        ("nd1_bilinear_point", "src_nodata", "Float32", "bilinear", 43, NODATA, point),
        ("nd1_cubic_point", "src_nodata", "Float32", "cubic", 43, NODATA, point),
        ("nd1_lanczos_point", "src_nodata", "Float32", "lanczos", 43, NODATA, point),
        ("nd1_near", "src_nodata", "Float32", "near", 43, NODATA, {}),
        # Average thresholds, single band.
        ("thr_average_nodata25", "src_nodata", "Float32", "average", 16, NODATA,
         {"NODATA_VALUES_PCT_THRESHOLD": "5"}),
        ("thr_average_excl1", "src_cat", "Float32", "average", 16, None,
         {"EXCLUDED_VALUES": "1"}),
        ("thr_average_excl12_pct20", "src_cat", "Float32", "average", 16, None,
         {"EXCLUDED_VALUES": "(1),(2)", "EXCLUDED_VALUES_PCT_THRESHOLD": "20"}),
        # Three bands, excluded tuples.
        ("thr_average_mb_excl", "src_mb_cat", "Float32", "average", 16, None,
         {"EXCLUDED_VALUES": "(1,1,1),(2,2,2)", "EXCLUDED_VALUES_PCT_THRESHOLD": "20"}),
        ("thr_average_mb_nodata25", "src_mb", "Float32", "average", 16, NODATA,
         {"NODATA_VALUES_PCT_THRESHOLD": "25"}),
    ]
    for unified in ("unset", "YES", "NO"):
        u = {} if unified == "unset" else {"UNIFIED_SRC_NODATA": unified}
        tag = unified.lower()
        cases += [
            (f"mb_{tag}_near", "src_mb", "Float32", "near", 43, NODATA, dict(u)),
            (f"mb_{tag}_bilinear_point", "src_mb", "Float32", "bilinear", 43, NODATA, {**u, **point}),
            (f"mb_{tag}_cubic_point", "src_mb", "Float32", "cubic", 43, NODATA, {**u, **point}),
            (f"mb_{tag}_bilinear_4x", "src_mb", "Float32", "bilinear", 16, NODATA, dict(u)),
            (f"mb_{tag}_lanczos_4x", "src_mb", "Float32", "lanczos", 16, NODATA, dict(u)),
            (f"mb_{tag}_average_4x", "src_mb", "Float32", "average", 16, NODATA, dict(u)),
            (f"mb_{tag}_mode_4x", "src_mb", "Float32", "mode", 16, NODATA, dict(u)),
        ]
    srcs = {"src_nodata": lambda: [source(True)], "src_cat": lambda: [categorical(False)],
            "src_mb": multiband, "src_mb_cat": multiband_categorical}
    for name, sname, tname, alg, w, nodata, wo in cases:
        bands = srcs[sname]()
        src = mem_bands(bands, MODE_TYPES[tname], nodata)
        out = gdal.Warp("", src, format="MEM", outputBounds=[0, 0, N, N], width=w, height=w,
                        resampleAlg=alg, errorThreshold=0, warpOptions=wo,
                        dstNodata=NODATA if nodata is not None else None)
        res = [array.array("d", out.GetRasterBand(b + 1).ReadRaster(buf_type=gdal.GDT_Float64))
               for b in range(out.RasterCount)]
        opts = " ".join(f"{k}={v}" for k, v in wo.items()) or "-"
        write_bands(os.path.join(HERE, f"{name}.txt"),
                    [f"gdal {gdal.__version__}", f"alg {alg}", f"src {sname}", f"src_type {tname}",
                     f"nodata {'-' if nodata is None else nodata}", f"wo {opts}",
                     "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform())],
                    w, w, res)
    print(len(cases), "nodata cases")



# ---- more methods and types ------------------------------------------------

def big_ints(signed):
    """Values near 2**40 (exact in float64), signed or not."""
    base = 2 ** 40
    vals = []
    for r in range(N):
        for c in range(N):
            v = base + ((r * 37 + c * 11) % 97) * 1000 + (c % 3)
            vals.append(-v if signed and (r + c) % 5 == 0 else v)
    return vals


def main_methods():
    import array

    gdal.SetConfigOption("CPL_DEBUG", None)
    write_grid(os.path.join(HERE, "src_big_signed.txt"), [f"gdal {gdal.__version__}"], N, N, big_ints(True))
    write_grid(os.path.join(HERE, "src_big_unsigned.txt"), [f"gdal {gdal.__version__}"], N, N, big_ints(False))
    srcs = {"src": lambda: [source(False)], "src_nodata": lambda: [source(True)],
            "src_round": lambda: [[pattern_round(r, c) for r in range(N) for c in range(N)]],
            "src_mb": multiband, "src_cat": lambda: [categorical(False)],
            "src_big_signed": lambda: [big_ints(True)], "src_big_unsigned": lambda: [big_ints(False)]}
    point = {"XSCALE": "1", "YSCALE": "1"}
    # name, src, type, alg, width, nodata, warp options, destination
    cases = [
        ("cs_point_1p49x", "src", "Float32", "cubicspline", 43, NODATA, point, None),
        ("cs_1p49x", "src", "Float32", "cubicspline", 43, NODATA, {}, None),
        ("cs_4x", "src", "Float32", "cubicspline", 16, NODATA, {}, None),
        ("cs_2p78x", "src", "Float32", "cubicspline", 23, NODATA, {}, None),
        ("cs_4x_nodata", "src_nodata", "Float32", "cubicspline", 16, NODATA, {}, None),
        ("cs_point_nodata", "src_nodata", "Float32", "cubicspline", 43, NODATA, point, None),
        ("cs_int16_4x", "src_round", "Int16", "cubicspline", 16, None, {}, None),
        ("cs_rot30", "src", "Float32", "cubicspline", 8, NODATA, {}, "rot30"),
    ]
    for alg in ("min", "max", "med", "q1", "q3", "rms"):
        cases += [
            (f"stat_{alg}_4x", "src", "Float32", alg, 16, NODATA, {}, None),
            (f"stat_{alg}_2p78x", "src", "Float32", alg, 23, NODATA, {}, None),
            (f"stat_{alg}_4x_nodata", "src_nodata", "Float32", alg, 16, NODATA, {}, None),
            (f"stat_{alg}_rot30", "src", "Float32", alg, 8, NODATA, {}, "rot30"),
        ]
    cases += [
        ("stat_max_mb_4x", "src_mb", "Float32", "max", 16, NODATA, {}, None),
        ("stat_med_mb_4x", "src_mb", "Float32", "med", 16, NODATA, {}, None),
        ("stat_med_int16_2p78x", "src_round", "Int16", "med", 23, None, {}, None),
    ]
    for tname, sname in (("Int64", "src_big_signed"), ("UInt64", "src_big_unsigned")):
        for alg, w in (("near", 43), ("bilinear", 43), ("average", 16), ("mode", 16), ("max", 16)):
            cases.append((f"type_{tname.lower()}_{alg}", sname, tname, alg, w, None, {}, None))
    for alg, w in (("near", 43), ("bilinear", 43), ("cubic", 43), ("average", 16), ("mode", 16)):
        cases.append((f"type_float16_{alg}", "src", "Float16", alg, w, None, {}, None))

    for name, sname, tname, alg, w, nodata, wo, dst in cases:
        src = mem_bands(srcs[sname](), MODE_TYPES[tname], nodata)
        if dst == "rot30":
            out = gdal.GetDriverByName("MEM").Create("", w, w, src.RasterCount, MODE_TYPES[tname])
            out.SetGeoTransform(rot30_gt())
            for b in range(src.RasterCount):
                out.GetRasterBand(b + 1).SetNoDataValue(nodata)
                out.GetRasterBand(b + 1).Fill(nodata)
            gdal.Warp(out, src, resampleAlg=alg, errorThreshold=0, warpOptions=wo)
        else:
            out = gdal.Warp("", src, format="MEM", outputBounds=[0, 0, N, N], width=w, height=w,
                            resampleAlg=alg, errorThreshold=0, warpOptions=wo, dstNodata=nodata)
        res = [array.array("d", out.GetRasterBand(b + 1).ReadRaster(buf_type=gdal.GDT_Float64))
               for b in range(out.RasterCount)]
        opts = " ".join(f"{k}={v}" for k, v in wo.items()) or "-"
        write_bands(os.path.join(HERE, f"{name}.txt"),
                    [f"gdal {gdal.__version__}", f"alg {alg}", f"src {sname}", f"src_type {tname}",
                     f"nodata {'-' if nodata is None else nodata}", f"wo {opts}",
                     "dst_gt " + " ".join(repr(v) for v in out.GetGeoTransform())],
                    w, w, res)
    print(len(cases), "method/type cases")


if __name__ == "__main__":
    main()
