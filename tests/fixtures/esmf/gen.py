"""Regenerate the ESMF oracle fixtures for tests/esmf.rs, tests/esmf_tripolar.rs
and tests/esmf_mesh.rs.

    /opt/gdal-py/bin/python tests/fixtures/esmf/gen.py [case ...]

Runs in ghcr.io/hypertidy/gdal-r-esmf (ESMPy, GDAL's Python bindings), or by
hand from the workflow .github/workflows/esmf-fixtures.yml. design/conservative.md
section 4 explains the suite. With case names, only those cases are made
("tripolar" makes all the tripolar ones).

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
- <case>.tr.txt, for regular_coarser and regular_finer: TempestRemap's
  first-order finite volume map (GenerateRLLMesh, GenerateOverlapMesh,
  GenerateOfflineMap --in_np 1). TempestRemap 2.2 builds lon/lat meshes with
  great-circle edges too (GenerateRLLMesh.cpp defines ONLY_GREAT_CIRCLES), so
  this is a second, independent implementation of ESMF's cells. TempestRemap
  does not map between grids that only partly overlap ("Invalid overlap
  mesh"), so regular_overhang has none.

The tripolar cases (tests/esmf_tripolar.rs) have a curvilinear source: an
analytic tripolar cap, with a seam and a fold, given by its corners in
north-polar Lambert azimuthal equal-area on a sphere (tripolar.grid.txt),
the plane in which rwarp holds it as one sheet. Destinations are a regular
grid in the same plane and a regular lon/lat grid up to the pole:

- <case>.cart.txt: Cartesian, in that plane. Every cell's edges are straight
  between its corners there, as rwarp's source cells are and its destination
  cells are at densify 1, so rwarp must match it to rounding: this checks the
  walk through the seam, the fold and the poles against an independent code.
- <case>.sph.txt: spherical, the same corners in lon/lat with great-circle
  edges, source and destination. The test records the difference.

The onto_tripolar cases are the same grids the other way round: the regular
grid is the source and the tripolar cap the destination (rwarp's
mesh::MeshTarget).

The mesh cases (tests/esmf_mesh.rs) are between a mesh of polygons and a
regular grid, both ways (rwarp's mesh::PolyMesh): hexagons with every
third split into six triangles round its centre, its nodes shared and
moved a little, over part of a grid of unit cells and past its top-left
corner (mesh_hex.mesh.txt). Cartesian, in the grid's plane, where every
edge is straight on both sides, so rwarp must match it to rounding:

- mesh_hex_to.cart.txt: mesh to grid (the mesh's elements the source).
- mesh_hex_onto.cart.txt: grid to mesh.

Two more mesh pairs are on the sphere (spherical, ESMF's great-circle
edges), against a global 5 degree grid: a geodesic mesh, an icosahedron
with nodes on the poles and each face cut into 16 triangles (ICON's kind,
mesh_ico.mesh.txt), and its dual, hexagons and twelve pentagons, two
round the poles (MPAS's kind, mesh_dual.mesh.txt). Nodes are longitude
and latitude. rwarp's grid cells are bounded by parallels and ESMF's by
great circles, so the test records the difference; the faces are the
same on both sides:

- mesh_ico_to.sph.txt, mesh_ico_onto.sph.txt, mesh_dual_to.sph.txt,
  mesh_dual_onto.sph.txt.

ESMF splits a polygon into triangles and gives one weight per triangle,
so a mesh map's weights are summed per (row, col) before they are
written.

Fixture layout: '# key value' header lines, then 'n_b n_s', then n_b lines
'frac_b area_b' (destination row-major), then n_s lines 'row col S' (ESMF's
1-based destination and source indices, row-major over each grid).
"""

import math
import os
import shutil
import subprocess
import sys
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


def cell_field(g):
    """A field of one value per cell: a grid's centres or a mesh's elements."""
    if isinstance(g, esmpy.Mesh):
        return esmpy.Field(g, meshloc=esmpy.MeshLoc.ELEMENT)
    return esmpy.Field(g, staggerloc=esmpy.StaggerLoc.CENTER)


def conserve(src_grid, dst_grid):
    src = cell_field(src_grid)
    dst = cell_field(dst_grid)
    frac = cell_field(dst_grid)
    area = cell_field(dst_grid)
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
    """Run a tool; TempestRemap exits 0 after some errors, so callers check
    for its output too."""
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0 or "EXCEPTION" in p.stdout:
        raise RuntimeError(f"{' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")


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
    # The overlap wants the covered mesh first; the map reverses it.
    run("GenerateOverlapMesh", "--a", b, "--b", a, "--out", ov)
    run("GenerateOfflineMap", "--in_mesh", a, "--out_mesh", b, "--ov_mesh", ov,
        "--in_np", "1", "--out_map", m)
    if not os.path.exists(m):
        raise RuntimeError(f"GenerateOfflineMap wrote no {m}")
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
    oracle = {
        "cart": f"ESMPy {esmpy.__version__} conserve, norm dstarea, Cartesian in source pixel space",
        "sph": f"ESMPy {esmpy.__version__} conserve, norm dstarea, spherical, great-circle edges",
        "tr": "TempestRemap first-order FV (--in_np 1), great-circle edges",
    }[kind]
    header = [
        f"case {name}: {note}",
        f"oracle {oracle}",
        f"proj {osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}.{osr.GetPROJVersionMicro()}",
        f"src_crs {src_crs}",
        f"src_gt {' '.join(repr(float(v)) for v in src_gt)}",
        f"src_size {src_size[0]} {src_size[1]}",
        f"dst_crs {dst_crs}",
        f"dst_gt {' '.join(repr(float(v)) for v in dst_gt)}",
        f"dst_size {dst_size[0]} {dst_size[1]}",
    ]
    write_map(path, header, kind, result)


def write_map(path, header, kind, result):
    frac_b, area_b, row, col, s = result
    order = np.lexsort((col, row))
    fmt = "%.17g" if kind == "cart" else "%.12g"
    with open(path, "w") as f:
        for h in header:
            f.write(f"# {h}\n")
        f.write(f"{frac_b.size} {s.size}\n")
        for fb, ab in zip(frac_b, area_b):
            f.write(f"{fmt % fb} {fmt % ab}\n")
        for k in order:
            f.write(f"{row[k]} {col[k]} {fmt % s[k]}\n")
    print(f"{os.path.basename(path)}: {frac_b.size} destination cells, {s.size} weights")


# ---- a tripolar source ---------------------------------------------------------

# NEMO's sphere, and north-polar Lambert azimuthal equal-area on it: the plane
# in which rwarp holds a tripolar grid as one sheet (src/mesh.rs, "Seams and
# folds"). Planar areas there are areas on the sphere.
RADIUS = 6371229.0

# Elementary functions from the C library, element by element. numpy's own
# (SIMD) versions differ in the last bit between CPUs, which would make the
# tripolar fixtures depend on the runner.
_sin, _cos, _sinh, _cosh, _asin, _atan2, _hypot = (
    np.vectorize(f, otypes=[float])
    for f in (math.sin, math.cos, math.sinh, math.cosh, math.asin, math.atan2, math.hypot))


def laea(lon, lat):
    rho = 2 * RADIUS * _sin(np.radians(90 - lat) / 2)
    lam = np.radians(lon)
    return rho * _sin(lam), -rho * _cos(lam)


def laea_inv(x, y):
    rho = _hypot(x, y)
    lat = 90 - 2 * np.degrees(_asin(rho / (2 * RADIUS)))
    return np.degrees(_atan2(x, -y)), lat


# The tripolar cap: confocal ellipses x = a cosh(mu) cos(t), y = a sinh(mu)
# sin(t) about two grid poles at (+-a, 0). Row line 0 is the outer ellipse
# (mu = MU), the last row line the segment between the poles (mu = 0); column
# lines are at equal t. Elliptic coordinates are conformal, so cells keep
# their shape and shrink only towards the two poles, as in ORCA. The last row
# line folds onto itself (corner i on corner ncol - i: an F-point pivot), and
# column line ncol is column line 0, a seam running into one pole. Both are
# exact: mirrored corners are computed once. The cap is turned and shifted
# so that no grid line meets the geographic pole or follows a destination
# grid line; its outer row is at about 58 to 62 N.
TRIPOLAR = dict(ncol=48, nrow=16, a=1_500_000.0, mu=1.475, turn=20.0, shift=(90_000.0, -50_000.0))

TRIPOLAR_CASES = [
    # name, destination kind, dst_gt, dst_size, note
    ("tripolar_laea", "laea", [-3_900_000, 300_000, 0, -3_900_000, 0, 300_000], (26, 26),
     "tripolar cap to 300 km cells in the same plane, rows from the bottom up"),
    ("tripolar_lonlat", "lonlat", [0, 6, 0, 54, 0, 3], (60, 12),
     "tripolar cap to 6 x 3 degree lon/lat cells, rows from 54 N up to the pole"),
]

# The same grids the other way round: onto the tripolar cap.
ONTO_TRIPOLAR_CASES = [
    # name, source kind, src_gt, src_size, note
    ("onto_tripolar_laea", "laea", [-3_900_000, 300_000, 0, -3_900_000, 0, 300_000], (26, 26),
     "300 km cells in the cap's plane, rows from the bottom up, onto the tripolar cap"),
    ("onto_tripolar_lonlat", "lonlat", [0, 6, 0, 54, 0, 3], (60, 12),
     "6 x 3 degree lon/lat cells, rows from 54 N up to the pole, onto the tripolar cap"),
]


def tripolar_corners(ncol, nrow, a, mu, turn, shift):
    """Corner arrays x, y, shape (nrow + 1, ncol + 1), in the LAEA plane."""
    j = np.arange(nrow + 1)[:, None]
    i = np.arange(ncol + 1)[None, :] % ncol
    m = mu * (nrow - j) / nrow
    k = np.minimum(i, ncol - i)
    t = 2 * np.pi * k / ncol
    sign = np.where(i > ncol // 2, -1.0, 1.0)
    ex = a * _cosh(m) * _cos(t)
    ey = a * _sinh(m) * _sin(t) * sign
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    return ex * c - ey * s + shift[0], ex * s + ey * c + shift[1]


def centres(x, y):
    return tuple(0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, 1:] + v[1:, :-1]) for v in (x, y))


def write_grid(path, x, y):
    nrow, ncol = x.shape[0] - 1, x.shape[1] - 1
    p = TRIPOLAR
    with open(path, "w") as f:
        f.write("# grid tripolar cap, confocal ellipses about two poles; corners in north-polar LAEA (m)\n")
        f.write(f"# radius {RADIUS!r}\n")
        f.write(f"# params a {p['a']!r} mu {p['mu']!r} turn {p['turn']!r} shift {p['shift'][0]!r} {p['shift'][1]!r}\n")
        f.write(f"{ncol} {nrow}\n")
        for xv, yv in zip(x.ravel(), y.ravel()):
            f.write(f"{xv:.17g} {yv:.17g}\n")
    print(f"{os.path.basename(path)}: {ncol} x {nrow} cells")


def tripolar(names):
    """The tripolar cases: corners in tripolar.grid.txt, maps as for the
    regular cases. Every grid goes to ESMF indexed [col, row], with cells
    counter-clockwise in the plane (rows run up, or towards the pole)."""
    x, y = tripolar_corners(**TRIPOLAR)
    write_grid(os.path.join(HERE, "tripolar.grid.txt"), x, y)
    cap = {"cart": ((x.T, y.T), tuple(v.T for v in centres(x, y)))}
    cap["sph"] = tuple(laea_inv(*xy) for xy in cap["cart"])
    how = {"cart": "Cartesian in the LAEA plane (m)",
           "sph": "spherical, great-circle edges (area_b in steradians)"}
    systems = [("cart", esmpy.CoordSys.CART), ("sph", esmpy.CoordSys.SPH_DEG)]

    def regular(kind, gt, size):
        """A regular grid's corners and centres, in the plane and in lon/lat."""
        corner, centre = (pixel_to_geo(gt, *lattice(*size, off)) for off in (False, True))
        if kind == "laea":
            return {"cart": (corner, centre), "sph": (laea_inv(*corner), laea_inv(*centre))}
        return {"cart": (laea(*corner), laea(*centre)), "sph": (corner, centre)}

    def wanted(name):
        return not names or name in names or "tripolar" in names

    for name, kind, gt, size, note in TRIPOLAR_CASES:
        if not wanted(name):
            continue
        dst = regular(kind, gt, size)
        for sys_, coord_sys in systems:
            res = conserve(grid(*cap[sys_], coord_sys), grid(*dst[sys_], coord_sys))
            header = [
                f"case {name}: {note}",
                f"oracle ESMPy {esmpy.__version__} conserve, norm dstarea, {how[sys_]}",
                "src tripolar.grid.txt",
                f"dst_kind {kind}",
                f"dst_gt {' '.join(repr(float(v)) for v in gt)}",
                f"dst_size {size[0]} {size[1]}",
            ]
            write_map(os.path.join(HERE, f"{name}.{sys_}.txt"), header, sys_, res)
    for name, kind, gt, size, note in ONTO_TRIPOLAR_CASES:
        if not wanted(name):
            continue
        src = regular(kind, gt, size)
        for sys_, coord_sys in systems:
            res = conserve(grid(*src[sys_], coord_sys), grid(*cap[sys_], coord_sys))
            header = [
                f"case {name}: {note}",
                f"oracle ESMPy {esmpy.__version__} conserve, norm dstarea, {how[sys_]}",
                f"src_kind {kind}",
                f"src_gt {' '.join(repr(float(v)) for v in gt)}",
                f"src_size {size[0]} {size[1]}",
                "dst tripolar.grid.txt",
            ]
            write_map(os.path.join(HERE, f"{name}.{sys_}.txt"), header, sys_, res)


# ---- a mesh of polygons --------------------------------------------------------

# The grid: unit cells, pixel (col, row) at (x, -y); as in the regular cases,
# ESMF gets it with y up.
MESH_GRID_GT = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
MESH_GRID_SIZE = (14, 12)


def hex_mesh():
    """Pointy-top hexagons of circumradius 0.8 in rows 1.2 apart, from
    above and left of the grid's top-left corner to short of its right and
    bottom edges, every third split into six triangles round its centre.
    Nodes are shared (found by position), then each moved by up to 0.06 in
    x and y, by C library sines of where it was. Returns the nodes' x and
    y and the faces, anticlockwise, as lists of 0-based node indices."""
    rho = 0.8
    w = math.sqrt(3.0) * rho
    index, xs, ys, faces = {}, [], [], []

    def node(px, py):
        key = (round(px * 1e6), round(py * 1e6))
        if key not in index:
            index[key] = len(xs)
            xs.append(px)
            ys.append(py)
        return index[key]

    k = 0
    for j in range(-1, 10):
        for i in range(-1, 10):
            cx = -0.5 + i * w + (w / 2 if j % 2 else 0.0)
            cy = 0.6 - j * 1.5 * rho
            ring = [node(cx + rho * math.cos(math.radians(90 + 60 * m)),
                         cy + rho * math.sin(math.radians(90 + 60 * m))) for m in range(6)]
            if k % 3 == 2:
                c = node(cx, cy)
                faces.extend([c, ring[m], ring[(m + 1) % 6]] for m in range(6))
            else:
                faces.append(ring)
            k += 1
    for n in range(len(xs)):
        px, py = xs[n], ys[n]
        xs[n] = px + 0.06 * math.sin(3.1 * px + 1.7 * py)
        ys[n] = py + 0.06 * math.cos(2.3 * px - 1.1 * py)
    return xs, ys, faces


def write_mesh(path, xs, ys, faces, note="hexagons, every third split into six triangles; nodes in the grid's plane"):
    with open(path, "w") as f:
        f.write(f"# mesh {note}\n")
        f.write(f"{len(xs)} {len(faces)}\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:.17g} {y:.17g}\n")
        for face in faces:
            f.write(" ".join(str(v) for v in [len(face)] + list(face)) + "\n")
    print(f"{os.path.basename(path)}: {len(xs)} nodes, {len(faces)} faces")


def esmf_mesh(xs, ys, faces, coord_sys=esmpy.CoordSys.CART):
    m = esmpy.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=coord_sys)
    n = len(xs)
    m.add_nodes(n, np.arange(1, n + 1), np.column_stack([xs, ys]).ravel(), np.zeros(n, dtype=int))
    # element coordinates aren't used by conserve, but ESMF looks for them
    if coord_sys == esmpy.CoordSys.CART:
        centroids = [(sum(xs[v] for v in f) / len(f), sum(ys[v] for v in f) / len(f)) for f in faces]
    else:
        centroids = [lonlat(normal(vsum([vec(xs[v], ys[v]) for v in f]))) for f in faces]
    m.add_elements(len(faces), np.arange(1, len(faces) + 1), np.array([len(f) for f in faces]),
                   np.array([v for f in faces for v in f]), element_coords=np.array(centroids).ravel())
    return m


def summed(result):
    """One weight per (row, col): ESMF gives one per triangle of a polygon."""
    frac_b, area_b, row, col, s = result
    keys, inv = np.unique(np.column_stack([row, col]), axis=0, return_inverse=True)
    total = np.zeros(len(keys))
    np.add.at(total, inv.ravel(), s)
    return frac_b, area_b, keys[:, 0], keys[:, 1], total


# ---- meshes on the sphere -----------------------------------------------------

def vec(lon, lat):
    lam, phi = math.radians(lon), math.radians(lat)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def vsum(vs):
    return tuple(sum(v[k] for v in vs) for k in range(3))


def normal(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def lonlat(v):
    h = math.hypot(v[0], v[1])
    if h < 1e-15:
        return 0.0, math.copysign(90.0, v[2])
    return math.degrees(math.atan2(v[1], v[0])), math.degrees(math.atan2(v[2], h))


def anticlockwise(vs, face):
    """The face turned anticlockwise seen from outside, if it isn't."""
    a, b, c = (vs[k] for k in face[:3])
    u = tuple(b[k] - a[k] for k in range(3))
    w = tuple(c[k] - a[k] for k in range(3))
    n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
    return list(face) if sum(a[k] * n[k] for k in range(3)) > 0 else list(reversed(face))


def icosahedron(n):
    """An icosahedron with nodes on both poles, each face cut into n * n
    triangles through points on the sphere. Returns unit vectors and faces."""
    lat = math.degrees(math.atan(0.5))
    corners = [vec(0, 90), vec(0, -90)] + [vec(72 * k, lat) for k in range(5)] \
        + [vec(36 + 72 * k, -lat) for k in range(5)]
    u, l = (lambda k: 2 + k % 5), (lambda k: 7 + k % 5)
    big = []
    for k in range(5):
        big += [(0, u(k), u(k + 1)), (u(k), l(k), u(k + 1)), (u(k + 1), l(k), l(k + 1)), (1, l(k + 1), l(k))]
    index, vs, faces = {}, [], []

    def node(p):
        key = tuple(round(c * 1e9) for c in p)
        if key not in index:
            index[key] = len(vs)
            vs.append(p)
        return index[key]

    for a, b, c in big:
        a, b, c = corners[a], corners[b], corners[c]

        def at(i, j):
            s, t = i / n, j / n
            return normal(tuple(a[k] + (b[k] - a[k]) * s + (c[k] - a[k]) * t for k in range(3)))

        for i in range(n):
            for j in range(n - i):
                faces.append([node(at(i, j)), node(at(i + 1, j)), node(at(i, j + 1))])
                if i + j + 1 < n:
                    faces.append([node(at(i + 1, j)), node(at(i + 1, j + 1)), node(at(i, j + 1))])
    return vs, [anticlockwise(vs, f) for f in faces]


def dual(vs, faces):
    """A node at each face's centre, and a face round each node through
    the centres of the faces about it."""
    centres = [normal(vsum([vs[k] for k in f])) for f in faces]
    about = [[] for _ in vs]
    for t, f in enumerate(faces):
        for k in f:
            about[k].append(t)
    out = []
    for k, ts in enumerate(about):
        p = vs[k]
        axis = (0.0, 0.0, 1.0) if abs(p[2]) < 0.9 else (1.0, 0.0, 0.0)
        e1 = normal((axis[1] * p[2] - axis[2] * p[1], axis[2] * p[0] - axis[0] * p[2], axis[0] * p[1] - axis[1] * p[0]))
        e2 = (p[1] * e1[2] - p[2] * e1[1], p[2] * e1[0] - p[0] * e1[2], p[0] * e1[1] - p[1] * e1[0])
        ang = lambda t: math.atan2(sum(centres[t][i] * e2[i] for i in range(3)), sum(centres[t][i] * e1[i] for i in range(3)))
        out.append(anticlockwise(centres, sorted(ts, key=ang)))
    return centres, out


SPHERE_GRID_GT = [-180.0, 5.0, 0.0, 90.0, 0.0, -5.0]
SPHERE_GRID_SIZE = (72, 36)


def sphere_mesh_cases():
    vs, faces = icosahedron(4)
    meshes = [("mesh_ico", "the geodesic mesh (320 triangles, nodes on the poles)", vs, faces),
              ("mesh_dual", "its dual (162 hexagons and pentagons, two round the poles)", *dual(vs, faces))]
    size, gt = SPHERE_GRID_SIZE, SPHERE_GRID_GT
    corner, centre = (pixel_to_geo(gt, *lattice(*size, off)) for off in (False, True))
    g = grid(corner, centre, esmpy.CoordSys.SPH_DEG)
    oracle = f"oracle ESMPy {esmpy.__version__} conserve, norm dstarea, spherical, great-circle edges (area_b in steradians)"
    for name, note, vs, faces in meshes:
        xs, ys = zip(*(lonlat(v) for v in vs))
        write_mesh(os.path.join(HERE, f"{name}.mesh.txt"), list(xs), list(ys), faces,
                   "mesh on the sphere; nodes in longitude and latitude, edges great circles")
        mesh = esmf_mesh(list(xs), list(ys), faces, esmpy.CoordSys.SPH_DEG)
        common = [
            f"mesh {name}.mesh.txt",
            f"grid_gt {' '.join(repr(float(v)) for v in gt)}",
            f"grid_size {size[0]} {size[1]}",
        ]
        for kind, what, res in [
            ("to", f"{note} to a global 5 degree grid", conserve(mesh, g)),
            ("onto", f"a global 5 degree grid onto {note}", conserve(g, mesh)),
        ]:
            write_map(os.path.join(HERE, f"{name}_{kind}.sph.txt"), [f"case {name}_{kind}: {what}", oracle] + common,
                      "sph", summed(res))


def mesh_cases():
    xs, ys, faces = hex_mesh()
    write_mesh(os.path.join(HERE, "mesh_hex.mesh.txt"), xs, ys, faces)
    mesh = esmf_mesh(xs, ys, faces)
    size, gt = MESH_GRID_SIZE, MESH_GRID_GT
    corner, centre = (pixel_to_geo(gt, *lattice(*size, off)) for off in (False, True))
    g = grid(corner, centre, esmpy.CoordSys.CART)
    common = [
        "mesh mesh_hex.mesh.txt",
        f"grid_gt {' '.join(repr(float(v)) for v in gt)}",
        f"grid_size {size[0]} {size[1]}",
    ]
    oracle = f"oracle ESMPy {esmpy.__version__} conserve, norm dstarea, Cartesian in the grid's plane"
    for name, note, res in [
        ("mesh_hex_to", "the hexagon mesh to the grid", conserve(mesh, g)),
        ("mesh_hex_onto", "the grid onto the hexagon mesh", conserve(g, mesh)),
    ]:
        write_map(os.path.join(HERE, f"{name}.cart.txt"), [f"case {name}: {note}", oracle] + common,
                  "cart", summed(res))


def main():
    esmpy.Manager(debug=False)
    names = sys.argv[1:]
    for case in CASES:
        name, _, src_gt, (sw, sh), _, _, (dw, dh), _ = case
        if names and name not in names:
            continue
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
        if name in ("regular_coarser", "regular_finer"):
            work = tempfile.mkdtemp()
            write(os.path.join(HERE, f"{name}.tr.txt"), case, "tr", tempest(case, work))
            shutil.rmtree(work)
    if not names or any("tripolar" in n for n in names):
        tripolar(names)
    if not names or any("mesh" in n for n in names):
        mesh_cases()
        sphere_mesh_cases()


if __name__ == "__main__":
    main()
