"""Regenerate the ESMF oracle fixtures for tests/esmf.rs.

    /opt/gdal-py/bin/python tests/fixtures/esmf/gen.py

Runs in ghcr.io/hypertidy/gdal-r-esmf (ESMPy, GDAL's Python bindings), or by
hand from the workflow .github/workflows/esmf-fixtures.yml. design/conservative.md
section 4 explains the suite.

Each case is a regular source grid and a destination grid, both given as CRS
plus geotransform. Each destination cell's corners are taken from source pixel
space through GDAL's coordinate transformation, as GenImgProjTransformer does,
and two ESMF first-order conservative maps are made, normalised by destination
area, as ESMF's S:

- <case>.cart.txt: Cartesian (CoordSys.CART), in source pixel coordinates.
  Destination cell edges are straight between their corners in source pixel
  space and areas are planar there. That is rwarp's cell definition at
  densify 1, so rwarp must match it to rounding.
- <case>.sph.txt: spherical (CoordSys.SPH_DEG), with corners in lon/lat and
  ESMF's great-circle edges. The test records rwarp's difference from it.
- <case>.tr.txt, for the cases between regular lon/lat grids: TempestRemap's
  first-order finite volume map (GenerateRLLMesh, GenerateOverlapMesh,
  GenerateOfflineMap --in_np 1). TempestRemap 2.2 builds lon/lat meshes with
  great-circle edges too (GenerateRLLMesh.cpp defines ONLY_GREAT_CIRCLES), so
  this is a second, independent implementation of ESMF's cells.

Fixture layout: '# key value' header lines, then 'n_b n_s', then n_b lines
'frac_b area_b' (destination row-major), then n_s lines 'row col S' (ESMF's
1-based destination and source indices, row-major over each grid).
"""

import math
import os
import shutil
import subprocess
import tempfile

import esmpy
import netCDF4
import numpy as np
from osgeo import osr

osr.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))

LAEA_TAS = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs"
# CORDEX EUR-44's rotated pole
ROTATED = ("+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=39.25 "
           "+lon_0=18 +datum=WGS84 +no_defs")

CASES = [
    # name, src_crs, src_gt, src_size, dst_crs, dst_gt, dst_size, note
    ("regular_coarser", "EPSG:4326", [100, 1, 0, -10, 0, -1], (60, 40),
     "EPSG:4326", [101.3, 2.5, 0, -11.7, 0, -2], (20, 15),
     "1 degree to 2.5 x 2 degrees, offset, inside the source"),
    ("regular_finer", "EPSG:4326", [100, 2, 0, -10, 0, -2], (30, 20),
     "EPSG:4326", [103.1, 0.75, 0, -13.4, 0, -0.75], (56, 40),
     "2 degrees to 0.75 degrees, inside the source"),
    ("regular_overhang", "EPSG:4326", [100, 1, 0, -10, 0, -1], (60, 40),
     "EPSG:4326", [97.6, 1, 0, -8.7, 0, -1], (30, 20),
     "1 degree shifted over the source's top-left corner: frac_b < 1"),
    ("rotated_pole", "EPSG:4326", [-20, 0.5, 0, 75, 0, -0.5], (160, 90),
     ROTATED, [-15, 1, 0, 10, 0, -1], (30, 20),
     "0.5 degree lon/lat to a 1 degree rotated-pole grid over Europe"),
    ("lonlat_3031", "EPSG:4326", [0, 0.5, 0, -55, 0, -0.5], (180, 60),
     "EPSG:3031", [500_000, 25_000, 0, 2_000_000, 0, -25_000], (40, 40),
     "0.5 degree lon/lat to EPSG:3031 at 25 km, 67 to 80 S"),
    ("lonlat_laea", "EPSG:4326", [130, 0.5, 0, -30, 0, -0.5], (70, 50),
     LAEA_TAS, [-800_000, 40_000, 0, 800_000, 0, -40_000], (40, 40),
     "0.5 degree lon/lat to LAEA centred on Tasmania at 40 km"),
]


def srs(defn):
    s = osr.SpatialReference()
    s.SetFromUserInput(defn)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def pixel_to_geo(gt, px, py):
    return gt[0] + px * gt[1] + py * gt[2], gt[3] + px * gt[4] + py * gt[5]


def geo_to_pixel(gt, x, y):
    det = gt[1] * gt[5] - gt[2] * gt[4]
    dx, dy = x - gt[0], y - gt[3]
    return (dx * gt[5] - dy * gt[2]) / det, (dy * gt[1] - dx * gt[4]) / det


def dst_points(case, px, py):
    """Destination pixel coordinates to source lon/lat (or CRS units) and
    source pixel coordinates, through GDAL's transformation."""
    _, src_crs, src_gt, _, dst_crs, dst_gt, _, _ = case
    x, y = pixel_to_geo(dst_gt, px, py)
    ct = osr.CoordinateTransformation(srs(dst_crs), srs(src_crs))
    pts = np.array(ct.TransformPoints(np.column_stack([x.ravel(), y.ravel()])))
    gx, gy = pts[:, 0].reshape(x.shape), pts[:, 1].reshape(x.shape)
    sx, sy = geo_to_pixel(src_gt, gx, gy)
    return (gx, gy), (sx, sy)


def lattice(ncol, nrow, offset):
    """Pixel coordinates indexed [col, row] (ESMF's order: x fastest)."""
    c = np.arange(ncol + (0 if offset else 1)) + (0.5 if offset else 0.0)
    r = np.arange(nrow + (0 if offset else 1)) + (0.5 if offset else 0.0)
    return np.meshgrid(c, r, indexing="ij")


def grid(corner, centre, coord_sys):
    ncol, nrow = centre[0].shape
    g = esmpy.Grid(np.array([ncol, nrow]),
                   staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
                   coord_sys=coord_sys)
    for stagger, (x, y) in [(esmpy.StaggerLoc.CORNER, corner),
                            (esmpy.StaggerLoc.CENTER, centre)]:
        lo, hi = g.lower_bounds[stagger], g.upper_bounds[stagger]
        g.get_coords(0, staggerloc=stagger)[...] = x[lo[0]:hi[0], lo[1]:hi[1]]
        g.get_coords(1, staggerloc=stagger)[...] = y[lo[0]:hi[0], lo[1]:hi[1]]
    return g


def conserve(src_grid, dst_grid):
    src = esmpy.Field(src_grid, staggerloc=esmpy.StaggerLoc.CENTER)
    dst = esmpy.Field(dst_grid, staggerloc=esmpy.StaggerLoc.CENTER)
    frac = esmpy.Field(dst_grid, staggerloc=esmpy.StaggerLoc.CENTER)
    area = esmpy.Field(dst_grid, staggerloc=esmpy.StaggerLoc.CENTER)
    r = esmpy.Regrid(src, dst,
                     regrid_method=esmpy.RegridMethod.CONSERVE,
                     norm_type=esmpy.NormType.DSTAREA,
                     unmapped_action=esmpy.UnmappedAction.IGNORE,
                     dst_frac_field=frac, factors=True)
    w = r.get_weights_dict(deep_copy=True)
    area.get_area()
    # [col, row] arrays to destination row-major
    out = (frac.data.T.ravel().copy(), area.data.T.ravel().copy(),
           w["row_dst"], w["col_src"], w["weights"])
    r.destroy()
    return out


def rll_mesh(path, gt, size):
    """A TempestRemap lon/lat mesh for a north-up regular grid."""
    (w, h), (lon0, dlon, lat0, dlat) = size, (gt[0], gt[1], gt[3], gt[5])
    assert gt[2] == 0 and gt[4] == 0 and dlon > 0 and dlat < 0
    run("GenerateRLLMesh", "--lon", str(w), "--lat", str(h),
        "--lon_begin", repr(lon0), "--lon_end", repr(lon0 + w * dlon),
        "--lat_begin", repr(lat0 + h * dlat), "--lat_end", repr(lat0),
        "--file", path)


def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def grid_index(gt, size, lon, lat):
    """0-based row-major index of the cell whose centre is (lon, lat)."""
    c = np.floor((lon - gt[0]) / gt[1]).astype(int)
    r = np.floor((lat - gt[3]) / gt[5]).astype(int)
    assert (c >= 0).all() and (c < size[0]).all() and (r >= 0).all() and (r < size[1]).all()
    return r * size[0] + c


def tempest(case, work):
    """TempestRemap's first-order map between regular lon/lat grids, in
    ESMF's terms: frac_b, area_b per destination cell (row-major) and
    1-based row, col, S."""
    _, _, src_gt, src_size, _, dst_gt, dst_size, _ = case
    a, b = os.path.join(work, "a.g"), os.path.join(work, "b.g")
    ov, m = os.path.join(work, "ov.g"), os.path.join(work, "map.nc")
    rll_mesh(a, src_gt, src_size)
    rll_mesh(b, dst_gt, dst_size)
    run("GenerateOverlapMesh", "--a", a, "--b", b, "--out", ov)
    run("GenerateOfflineMap", "--in_mesh", a, "--out_mesh", b, "--ov_mesh", ov,
        "--in_np", "1", "--out_map", m)
    with netCDF4.Dataset(m) as ds:
        get = lambda v: ds[v][:].data
        ia = grid_index(src_gt, src_size, get("xc_a"), get("yc_a"))
        ib = grid_index(dst_gt, dst_size, get("xc_b"), get("yc_b"))
        n_b = dst_size[0] * dst_size[1]
        assert ib.size == n_b and len(set(ib)) == n_b
        frac_b, area_b = np.zeros(n_b), np.zeros(n_b)
        frac_b[ib], area_b[ib] = get("frac_b"), get("area_b")
        row = ib[get("row") - 1] + 1
        col = ia[get("col") - 1] + 1
        return frac_b, area_b, row, col, get("S")


def write(path, case, kind, result):
    name, src_crs, src_gt, src_size, dst_crs, dst_gt, dst_size, note = case
    frac_b, area_b, row, col, s = result
    order = np.lexsort((col, row))
    fmt = "%.17g" if kind == "cart" else "%.12g"
    oracle = {
        "cart": f"ESMPy {esmpy.__version__} conserve, norm dstarea, Cartesian in source pixel space",
        "sph": f"ESMPy {esmpy.__version__} conserve, norm dstarea, spherical, great-circle edges",
        "tr": "TempestRemap first-order FV (--in_np 1), great-circle edges",
    }[kind]
    with open(path, "w") as f:
        f.write(f"# case {name}: {note}\n")
        f.write(f"# oracle {oracle}\n")
        f.write(f"# proj {osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}."
                f"{osr.GetPROJVersionMicro()}\n")
        f.write(f"# src_crs {src_crs}\n# src_gt {' '.join(repr(float(v)) for v in src_gt)}\n")
        f.write(f"# src_size {src_size[0]} {src_size[1]}\n")
        f.write(f"# dst_crs {dst_crs}\n# dst_gt {' '.join(repr(float(v)) for v in dst_gt)}\n")
        f.write(f"# dst_size {dst_size[0]} {dst_size[1]}\n")
        f.write(f"{frac_b.size} {s.size}\n")
        for fb, ab in zip(frac_b, area_b):
            f.write(f"{fmt % fb} {fmt % ab}\n")
        for k in order:
            f.write(f"{row[k]} {col[k]} {fmt % s[k]}\n")
    print(f"{os.path.basename(path)}: {frac_b.size} destination cells, {s.size} weights")


def main():
    esmpy.Manager(debug=False)
    for case in CASES:
        name, _, src_gt, (sw, sh), _, _, (dw, dh), _ = case
        # source: pixel space, and lon/lat (every case's source is EPSG:4326)
        s_corner, s_centre = lattice(sw, sh, False), lattice(sw, sh, True)
        s_corner_ll = pixel_to_geo(src_gt, *s_corner)
        s_centre_ll = pixel_to_geo(src_gt, *s_centre)
        # destination: through GDAL to source lon/lat and source pixel space
        d_corner_ll, d_corner_px = dst_points(case, *lattice(dw, dh, False))
        d_centre_ll, d_centre_px = dst_points(case, *lattice(dw, dh, True))

        # Cartesian with pixel y flipped to point up, so cells are
        # counter-clockwise; areas and overlaps are unchanged.
        up = lambda xy: (xy[0], -xy[1])
        cart = conserve(grid(up(s_corner), up(s_centre), esmpy.CoordSys.CART),
                        grid(up(d_corner_px), up(d_centre_px), esmpy.CoordSys.CART))
        write(os.path.join(HERE, f"{name}.cart.txt"), case, "cart", cart)
        sph = conserve(grid(s_corner_ll, s_centre_ll, esmpy.CoordSys.SPH_DEG),
                       grid(d_corner_ll, d_centre_ll, esmpy.CoordSys.SPH_DEG))
        write(os.path.join(HERE, f"{name}.sph.txt"), case, "sph", sph)
        if name.startswith("regular_"):
            work = tempfile.mkdtemp()
            write(os.path.join(HERE, f"{name}.tr.txt"), case, "tr", tempest(case, work))
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
