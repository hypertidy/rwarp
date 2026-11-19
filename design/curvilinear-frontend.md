# Curvilinear grids: the coverage front end

Status: stages 1 and 2 implemented (`src/mesh.rs`, `src/mesh/walk.rs`,
`src/mesh/rect.rs`; 2026-10-31 and 2026-11-01), stage 3 built (#36,
#72), stage 4 started: walls that meet in the world (a seam, a fold) are
walked across (4a, 2026-11-17; section 3), and meshes come from CF bounds
and SCRIP corners with ocean-model halos dropped and overlapping cells
refused (4b, 2026-11-17; section 3), and weights on a tripolar grid are
compared with ESMF's (4c, 2026-11-17; section 3), and a mesh is a warp
source, for `exact_weights` and `warp()`'s exact footprint, through
`MeshTransformer` (4d, 2026-11-17; section 3), and a period is declared
for coordinates that jump at the seam (4e, 2026-11-17; section 3), and a
mesh is a warp target, through `MeshTarget` (4f, 2026-11-17; section 3),
and a whole ocean grid is a `MeshSet` of pieces in two polar projections
with damaged land left out (4g, 2026-11-17; section 3), and polygons in
longitude and latitude burn onto a set, on several threads and
projected there too (4h, 2026-11-17; section 3), and a set is a warp
source, through `SetTransformer` (4i, 2026-11-18; section 3).
Later stages proposed.
Stage 2 as built: `QuadMesh::ring_to_index` (cells found by a flood
from the first point's cell, or from crossed boundary sides, and each
segment clipped by every triangle it meets; cut points placed by their
parameter along the canonical edge; off-mesh stretches closed by the
boundary arc of zero winding), `QuadMesh::burn` (polygons, points),
`QuadMesh::cover_ring` (`CoverageWindow`), `Rectilinear`. Not yet: lines,
snapping of near-degenerate vertices (no property-test failure without
it so far), a boundary index (crossed boundary sides are found by a scan
when a segment leaves the mesh). Stage 1 as built: `QuadMesh::{from_corners, from_centres}`,
validation (bow-ties and folds refused, zero-area cells allowed),
`CellMap::{Bilinear, EqualArea}` with the equal-area split (corner mean
for convex cells; for non-convex cells the balanced point in the kernel
that keeps the four triangles furthest from degenerate), ownership by
canonical side orientation, `locate` with a walking hint, `locate_all`,
and CSR buckets built on first use. Not yet: the boundary index (it
belongs with the walk in stage 2). A declared period for coordinates that
jump at the seam came in stage 4 (4e, section 3); there is no `Topology`
enum. Decisions 1 to 5 below were taken
at their recommendations, provisionally; 1, 3 and 4 are now decided (N1,
N3 and N4 in `decisions.md`). `mesh` is public and unstable: it may break
at a 0.x minor release, not a patch.

Scope: how rwarp moves polygons (and its own
destination pixel outlines) into the index space of a curvilinear grid so
that controlledburn's unchanged engine gives exact coverage there. It is
stage 4 of hypertidy/controlledburn-rs#7, re-homed per wxylib's m20:
controlledburn stays dependency-free and in index space, and grid front
ends live in rwarp, where transformers, controlledburn and bigcurve meet.
The contract this front end works against (what controlledburn
guarantees, what a front end must do, conventions) is in controlledburn-rs
`design/grid-frontends.md`.

## Summary

- **Cells are the straight-edged quadrilaterals through their corners**,
  in the coordinates the corners are given in. GDAL's bilinear model has
  the same cells; it differs only in where interior points land in index
  space, and that is what makes its fractions wrong by about 1% of a cell.
- **Equal-area cell map.** Split each cell into four triangles at the
  mean of its corners, and the unit square into four matching triangles
  at a point chosen so each pair holds the same share of area. The map is
  then affine per triangle with one scale per cell, and controlledburn's
  index-space fractions are exact world-area fractions. Verified in
  controlledburn-rs `examples/grid_frontends.rs`: single precision against
  a world-space clip, where bilinear cells are off by 1.5%.
- **The front end is a walk, not a search.** Locate a ring's first vertex
  once, then walk each segment through the triangulated mesh by
  adjacency, inserting a vertex at every cell side and spoke it crosses,
  each mapped by its triangle's barycentric map. No inverse bilinear in
  the hot path.
- **Quad index stays in rwarp** (m20: rwarp is the composition point) and
  grows hints, all-matches, a boundary index and compact buckets.
- **bigcurve is needed only where edges change space** (polygon CRS
  differs from the grid's; destination pixel edges in the exact
  footprint). bigcurve owns the batched forward-map trait; rwarp
  implements it behind a feature.
- **The exact footprint with a geolocation source is this same front
  end** with tiny rings. Switching its inverse to the equal-area map makes
  it first-order conservative remapping in the source CRS, with no change
  to `CoverageWindow`. The antimeridian gap is fixed by unwrapping and
  adding shifted copies, also with no change to controlledburn.

## 1. Cell definition

#7 asks for the cell definition as an explicit parameter: straight edges
in a CRS, bilinear in index space, or geodesic. Two facts settle most of
it.

- A bilinear map restricted to a side of the unit square is linear, so a
  bilinear cell's sides are the straight segments between its corners.
  Bilinear and "straight in the corners' CRS" are the same cells.
- They differ inside: the bilinear Jacobian varies across a cell, so
  equal index areas are unequal world areas. That is the 1.5% error.

So the parameter is the **cell map**, the map from the unit square onto
the (fixed) quadrilateral:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellMap {
    /// GDAL's geolocation model. Default, for parity of point kernels.
    #[default]
    Bilinear,
    /// Four triangles of matching area share; exact area fractions.
    EqualArea,
}
```

Decision N1 (`decisions.md`, 2026-11-16): coverage uses `EqualArea` on
every mesh; a mesh's `CellMap` is the map of its point methods.

| cell map | index-space fractions are | use |
|---|---|---|
| `Bilinear` | fractions of the bilinear preimage: not world areas; needs densifying even for straight polygon edges | GDAL parity (point kernels, `geo_to_pixel`) |
| `EqualArea` | world-area fractions, exact for polygons with straight edges in the grid's CRS | coverage, area methods, conservative remapping |
| geodesic (reserved) | would need a spherical measure, not an index-space burn | not planned |

Straight in which CRS: the one the corners are given in. Corners in
longitude and latitude give cells straight in degrees, as GEOS reads
polygons. For grids near a pole, give corners in a polar projection and
move the polygons there (section 5); the front end does not guess.

### The equal-area split

For a cell with corners `P00, P10, P11, P01` (unit square corners
`(0,0), (1,0), (1,1), (0,1)`) and area `A`:

- world centre `C` = mean of the four corners;
- unit-square centre `u = 2 area(P01, P00, C) / A`,
  `v = 2 area(P00, P10, C) / A`;
- triangle `k` is side `k` plus the centre, in both spaces.

Each unit triangle then holds the same share of the square as its world
triangle holds of the cell, so all four affine maps scale area by
`1 / A`. Why the corner mean works: the summed area of the two triangles
on opposite sides is linear in `C`, and its values at those sides'
midpoints add up to `A`, so at the midpoint between them (the corner
mean) it is `A / 2`.

- **Convex cells:** `C` is inside, all four triangles are proper.
- **Non-convex but simple cells:** the solutions form a line through the
  corner mean (`area(side 0 triangle) + area(side 2 triangle) = A / 2`).
  Use the point of that line inside the cell's kernel (nearest the corner
  mean); every simple quadrilateral has a non-empty kernel. If the line
  misses the kernel, the cell cannot be split this way: fall back to the
  kernel point nearest the line, accept a non-exact cell, and add a note.
  Measure how often this happens on real grids before doing more.
- **Bow-ties** (self-intersecting): rejected at construction.
- **Zero-area cells** (collapsed corners): no triangles; they never
  receive coverage; a note when a ring touches one.

### Grids given by centres

With `GeolocConvention::PixelCenter`, rwarp's quadrilaterals join sample
*centres*, so a pixel straddles four of them. For coverage, the cells are
the pixels: build the `(nrow + 1) x (ncol + 1)` corner array first (mean
of four centres, outer corners extrapolated half a cell, the convention
rwarp already uses for cell edges), then the equal-area split on those
corners. With `TopLeftCorner` the samples are already corners.

## 2. Mapping a ring

The front end's job (controlledburn contract, items 1 to 3) for one ring
in the grid's CRS:

1. **Start.** Locate the first vertex inside the mesh (section 4), then
   its triangle. If the ring starts outside, find where it first enters
   through the mesh's outer boundary.
2. **Walk.** In the current triangle, intersect the segment with the two
   edges it has not entered by. Leave through the one hit first: a spoke
   leads to the next triangle of the same cell, a cell side to the
   neighbour cell's triangle on that side (from the grid's topology, not
   a search). A few orientation tests per step.
3. **Emit.** At each crossing, insert a vertex. Its index-space position
   is exact without the barycentric map: on a cell side at parameter `t`
   it is at `t` along the unit side (the map is linear along a side); on
   a spoke at `t` it is at `t` along the unit spoke. Ring vertices inside
   a triangle use the barycentric map.
4. **Hand over.** The index-space ring goes to `controlledburn::burn`
   (sparse, many polygons) or `CoverageWindow` (one small ring), and the
   `(row, col)` that come back are the grid's own.

A shared side or spoke is intersected once, in a canonical direction, so
neighbouring polygons and neighbouring destination pixels get bitwise the
same crossing.

### Off the mesh

Outside the mesh the map is undefined (rwarp's extrapolated ring of
quadrilaterals is for GDAL parity, not coverage). A ring that leaves must
be closed in index space along the mesh boundary from exit to re-entry;
the added pieces lie on boundary walls and change no cell (contract item
3). Which way round matters when the excursion encircles the mesh: take
the boundary arc whose turning around an interior point matches the
excursion's.

### Degeneracies

#6 in controlledburn shows what happens when these are left for later:

- vertex within tolerance of a side, spoke or corner: snap onto it with
  its exact `t`;
- segment exactly through a corner or the centre: pick the next triangle
  by orientation tests against the incident edges (a fan walk), never by
  comparing `t` values alone;
- segment along a side: stays on the side; the index ring then runs along
  the wall, which controlledburn handles since the wall-cell fix (#12,
  #15).

Plain `f64` orientation tests with snapping first; adaptive-precision
`orient2d` if the property tests find failures.

## 3. Folds, poles, periodic grids, the antimeridian

### Seams and folds: walls that meet in the world (built, 4a)

Built (2026-11-17), differently from the first plan here. That plan was a
`Topology` enum (`Bounded`, `PeriodicCols { period }`, `Tripolar { period }`)
whose neighbour rules the walk would follow, keeping a local frame
across the seam and burning an unwrapped index ring on a widened grid.

Given its corners in a projection in which it is one sheet, a periodic
grid's seam and a tripolar grid's fold are outer walls of the index
rectangle that meet in the world. The walk already finds the cells on
both sides, since a segment that leaves through an outer side is looked
for again from the boundary sides it meets. What went wrong was the
index ring: at the crossing it jumps from one wall to the other with no
gap along the segment, and the walk joined the two positions with a
straight index-space segment across the mesh (and, before, dropped the
second one). Now the jump is taken as an exit and re-entry with nothing
outside between. The boundary arc picked by the existing winding test
then runs along the walls: round the fold and the seam, which go out and
back in the world, not round the grid's real boundary, which winds round
it. Nothing is declared. Three details:

- A vertex on such a wall is placed in each piece's own cell, not only in
  the cell that owns it.
- The jump where the ring closes on itself (its first vertex on a seam)
  is checked too.
- Walls that agree only to rounding, with a sliver gap or overlap, are
  handled the same way.

Bounded meshes give the same output bit for bit: Natural Earth 10m land
burned on the warped global mesh of `examples/mesh_burn_par.rs`, at 1 and
0.25 degrees, gives the same hashes before and after.

Tested in `tests/mesh_glued.rs`, against the per-cell world clip, on
three meshes, each with its walls exact and to rounding:

- an annulus (a seam);
- a strip wrapped round one end of a slit (a fold, no seam);
- a disk with a seam and a fold (a tripolar grid as a polar projection
  shows it, with two poles, one where the seam meets the fold).

The cases:

- random polygons with holes, steered onto corners and centres;
- rings round the hole, round the whole fold, and round each pole;
- three and four seam crossings;
- `cover_ring` against `burn`;
- an ignored test of about 16,000 polygons.

What stage 4 still needs:

- **Coordinates that jump at the seam** (longitude, period 360): built,
  4e, below.
- **Tripolar grids in longitude and latitude** can't be planar meshes:
  the geographic pole is inside an ocean cell. An ORCA grid is given as a
  mesh in a polar projection. North polar Lambert azimuthal equal-area
  keeps planar areas true, with edges straight in that projection: not
  great circles, not rhumb lines. The difference from ESMF's great
  circles is second order in cell size, as the ESMF comparison measures
  (4c, below). One north-polar projection doesn't hold the whole globe, though
  (4b, below): near 80 S its cells are flattened 500 to 1 or more, and
  the bend of their long sides exceeds their width.
- The southern rows of a global ORCA grid: longitude with a declared
  period (4e, below) holds them. A whole ORCA file is a `MeshSet` in two
  polar LAEAs (4g, below).
- A mesh as the target of weights (the reverse map, onto a tripolar
  grid): built, 4f, below.

### CF bounds, SCRIP corners and halos (built, 4b)

Built (2026-11-17): `QuadMesh::from_cf_bounds(ncol, nrow, xb, yb, halo,
map)` in `src/mesh/cf.rs`, for each cell's own four vertices, row-major
with the vertex last: CF's `lon_bnds(j, i, 4)`, and SCRIP's
`grid_corner_lon(grid_size, 4)` with `grid_dims` giving `ncol, nrow`.

- **Halo.** `Halo { cols, rows, fold_through_centres }` names the columns
  and rows dropped at each end before the mesh is built. Mesh cell
  `(row, col)` is the file's `(row + rows[0], col + cols[0])`.
- **Vertex order.** CF's, or any rotation or reflection of it, found from
  the data: the first of the eight under which neighbours share corners.
- **Shared corners.** Neighbours must agree to a thousandth of the larger
  cell (rounding, `f32` storage), or the bounds are refused, with the two
  cells named and a hint when the copies are 360 or 2 pi apart. Corner
  `(j, i)` is taken from cell `(j, i)`, whose first vertex it is.
- **Overlap (N3).** `QuadMesh::find_overlap` probes each cell once, at
  the centroid of the larger triangle of the diagonal inside it, and
  reports a second owner. `from_cf_bounds` refuses on one. It isn't run
  by `from_corners`: on eORCA1 it costs about what building the mesh
  does (30 ms against 24 ms for 100,000 cells, the bucket index
  included).

NEMO's layout, as CMIP6 files have it, checked on IPSL's eORCA1
(F-point pivot) and ORCA2 (T-point pivot) from the public CMIP6 zarr
store:

- The last two columns repeat the first two.
- The first row is the closed southern boundary: land, with eORCA1's
  bounds collapsing each cell to a point, so they share no corners.
- The last row repeats a row below it, mirrored by a half-turn: row
  `ny - 2` at an F pivot, row `ny - 3` at a T pivot.
- At a T pivot the fold runs through the centres of row `ny - 2`, which
  mirrors itself: its cells are the same ground in pairs. The two pivot
  cells in it are collapsed (one to a segment, one to a point).
  IPSL's `areacello` gives one cell of each pair the whole area and the
  other 0.

`Halo::NEMO_F_PIVOT` and `Halo::NEMO_T_PIVOT` drop the last two columns,
the first row and the last row. Dropping columns `[0, 2]` rather than
`[1, 1]` keeps mesh column `c` the file's column `c`. At a T pivot, it
also keeps the collapsed pivot cell off the seam (both layouts work).
With `fold_through_centres` (T pivot) the last row is cut at the
midpoints of its column-line sides and the lower halves kept. The top
wall then folds onto itself, as at an F pivot, and a model cell of that
row is two mesh cells, one half each. The flag is refused when that row
doesn't mirror itself (an F pivot taken for a T). The other way round
(a T pivot taken for an F), the row's overlapping pairs are refused.

The walk needed four fixes for cells with no area, which a T pivot puts
inside the mesh (and which polar projections and collapsed bounds give
anyway):

- A ring crossing one jumps in index space at one world point. It's now
  linked straight through the collapsed cell. Where that cell is on the
  boundary (a T pivot on the seam), it goes through the cell to its wall,
  then across the walls. A jump the walk can't follow is an error, not a
  wrong ring.
- The equal-area split of a cell with a side of no length (a triangle)
  was inexact, with its centre on the collapsed corner and its unit
  centre outside the square. Sides of no length are now left out of the
  test that every triangle has area. Its triangle on that side has none
  in either space, and the split is exact.
- A vertex at a collapsed corner has an index point per triangle, so a
  piece takes the vertex's own position only when it lies in the piece's
  triangle, not merely its cell.
- A start vertex no cell owns (a corner whose owner by the low-index
  rule has no area) seeds the flood from the cells near it.

An area that is zero to rounding is now zero (`shoelace2` checks its
forward error bound), so a collapsed cell whose corners are not quite on
one line has no orientation of its own.

Tested in `tests/mesh_cf.rs`. ORCA-like disks are written as CF bounds,
halo and all, for both pivots and both column layouts:

- random polygons and 80,000 more round the pivots, against the world
  clip;
- the mesh tiling the disk once;
- the halved row's pairs summing to the file's cells;
- every halo left in refused with its cause;
- all eight vertex orders giving the same mesh bit for bit;
- shared corners within rounding taken, and beyond it refused;
- longitudes a period apart named.

`tests/mesh_walk.rs` adds a collapsed column and a cell collapsed to a
point. The bounded mesh hashes are unchanged.

On the real files, projected to spherical north-polar LAEA:

- **ORCA2, rows north of 49 N (180 x 34).** Every wrong layout is refused
  with its cause. The mesh builds in 2.4 ms. Ocean cell areas against
  `areacello` are 0.9998 in total; the median cell is -8e-5 and the worst
  -1.4% (the halved fold row). A cap north of 60 N, over the fold, both
  pivots and the seam, is covered to 2e-10 of its area.
- **eORCA1 north of 76 S (360 x 283).** The same refusals, and a build in
  54 ms. Areas are 0.9999 in total, with cells between -0.5% and +0.2%.
  The cap is covered to 3e-11.

Neither whole file builds, for reasons in the data, all on land:

- eORCA1 south of 76 S has 377 bow-ties and 59 flipped cells in that
  projection (its ice-shelf rows, flattened as above).
- IPSL's ORCA2 bounds are corrupt over land between 9 N and 49 N: 119
  bow-ties and 47 flipped cells, some of them spanning 30 degrees of
  longitude.

Both are refused, rightly for the walk, since a bad cell can spill a
ring into good ones. For whole files, that needs a choice: drop the
southern land rows (the halo's `rows[0]` does it), repair land bounds
from the centres, or let a cell mask make holes (swath gaps, "later"
below).

- **Poles.** Cells straight in degrees that touch latitude 90 are valid
  planar quadrilaterals (possibly triangles). A pole *inside* a cell is
  not meaningful in degrees; such grids should come with corners in a
  polar projection.
- **Halos.** Ocean-model files often keep duplicated cells: NEMO's
  cyclic columns, and at the north fold a row that mirrors one below it
  (T pivot: the last row mirrors row `jpj - 2`, and row `jpj - 1` mirrors
  itself; F pivot: the last row mirrors row `jpj - 1`). The copies keep
  orientation (a shift, a half-turn in index space), so the sign test
  below doesn't see them. `from_cf_bounds` takes the halo layout and
  drops the duplicates (built, 4b): the seam and fold are walls that
  meet, not data.
- **Other folds** (overlapping swath scan lines, MODIS bow-tie): cells
  overlap in world space, so location is ambiguous and a walk follows one
  sheet. Decided (N3, 2026-11-17): refused for coverage. Construction
  refuses cells of both orientations (built in stage 1); `find_overlap`
  finds copies that keep orientation (one probe per cell, refused with a
  second owner; built, 4b, and run by `from_cf_bounds`). An overlapping
  swath is used as one mesh per scan (the `mesh` module docs show it);
  how overlapping scans combine is the caller's choice.

### ESMF on a tripolar grid (built, 4c)

Built (2026-11-17): `tests/esmf_tripolar.rs`, with fixtures made by
`tests/fixtures/esmf/gen.py` (ESMPy 8.9.1 in `ghcr.io/hypertidy/gdal-r-esmf`).

- **Source.** An analytic tripolar cap, 48 x 16 cells, given by its
  corners in north-polar LAEA on NEMO's sphere (`tripolar.grid.txt`).
  - Shape: confocal ellipses about two grid poles. They are conformal, so
    cells keep their shape and shrink only towards the poles, as in ORCA.
    They are 13 to 363 km across, median 183 km.
  - Walls: an F-point fold between the poles, and a seam that runs into
    one of them.
  - Placement: turned and shifted so that no grid line meets the
    geographic pole.
- **Destinations.** 300 km cells in the same plane, and 6 x 3 degree
  lon/lat cells from 54 N to the pole. The top row of the lon/lat grid is
  triangles with a corner on the pole.
- **rwarp's weights.** `exact_weights` through `MeshTransformer` (4d,
  below), normalised by destination area (ESMF's `destarea`). The test
  first assembled them from `QuadMesh::cover_ring`, and still checks the
  two against each other.

What it shows:

- **Cartesian ESMF in the plane** has rwarp's cells exactly.
  - Agreement: the maps agree to rounding. S agrees to 3e-14 and frac_b
    to 4e-14, the entries are the same, and areas agree to the bit.
  - Coverage: destination cells cross the seam (8 to 10 of them) and the
    fold (13 to 38). They also cover both grid poles and the geographic
    pole.
  - Layout: the NEMO layout of the same grid, read with
    `Halo::NEMO_F_PIVOT`, is the same mesh to the bit. That layout is CF
    bounds with a land row, the cyclic columns and the mirrored fold row.
- **Spherical ESMF** joins the same corners, in lon/lat, by great circles.
  The differences come from that cell definition, and the test records
  them as ceilings:
  - S: up to 2.6e-3, and 1.7e-3 over destination cells both maps cover
    completely. The largest are on the source's outer row, whose long
    edges are furthest from the pole.
  - The regridded smooth field: 1e-4.
  - Cell areas: 3e-4 for the LAEA destination, 9e-4 for the lon/lat one.
- **Cells half the size.** On grids with cells half the size on both
  sides (measured, not committed), the differences shrink:
  - area differences fall to a quarter, second order in cell size as
    expected above;
  - the field's fall to about 0.3;
  - S's fall to a half, since an edge shifted by h^2 is a share h of a
    cell of side h.
- **Lon/lat destination.** Joining its corners straight in the plane
  (densify 1) is closer to ESMF than following the parallels (densify 8:
  S 1.1e-2). That matches the regular cases of `tests/esmf.rs`, because
  ESMF's great circles bow poleward of the parallels.

The reverse direction, onto the tripolar grid, is compared in 4f
(below).

### A mesh as a warp source (built, 4d)

Built (2026-11-17): `rwarp::mesh::MeshTransformer` (`src/mesh/source.rs`),
a `Transformer` whose source pixels are a `QuadMesh`'s cells, from a
destination grid's geotransform and CRS. The constructors match
`GenImgGeolocTransformer`'s: `new(mesh, mesh_crs, dst_gt, dst_crs)` with
the compiled-in CRS backend, and `with_backend` with caller-supplied
`CrsTransform`s.

- **No new path.** It reports the mesh as the source's cells
  (`Transformer::source_cells`), the path geolocation sources take since
  stage 3. So `exact_weights`, `Weights::apply` and `warp()` with
  `Footprint::Exact` work unchanged. They walk each destination outline
  onto the mesh in world coordinates, across a seam or fold (4a), and
  weight by world area.
- **Source windows.** `SourceCells::mesh` is `QuadMesh::window`, which
  clones the whole mesh or builds a window's cells as a mesh of its own.
  Walls meet in a window only if both are in it, and each cell keeps its
  equal-area split, so windows tile the whole mesh's weights to rounding.
- **Point kernels.** `transform` maps a destination pixel to world
  coordinates and then into the mesh's index space, through the mesh's
  own cell map (`to_index`), and back through `to_world`. Point kernels
  read neighbours in index space, not across the seam.
- **Tests.** `tests/mesh_transformer.rs`, on a tripolar cap:
  - every cell's weights sum to 1 (4e-16) with a destination over the
    whole mesh;
  - `warp()`'s `Average` is `apply(FracArea)` bit for bit;
  - 2 x 2 source windows give the whole mesh's weights (7e-15);
  - either destination row order gives the same weights;
  - transform round trips;
  - the CRS backend (libproj and proj4rs) gives the written-out
    projection's weights to 5e-14.

  `tests/esmf_tripolar.rs` now takes its weights from it, the same as
  the `cover_ring` assembly to 1e-13.
- **Not built here.** A mesh as the target of weights came in 4f, and a
  mesh in longitude whose coordinates jump at the seam with the declared
  period, 4e.

### A mesh as a warp target (built, 4f)

Built (2026-11-17): `rwarp::mesh::MeshTarget<T>` (`src/mesh/target.rs`),
a `Transformer` whose destination pixels are a `QuadMesh`'s cells.

- **A wrapper, not a pipeline.** A target pixel/line is a point of the
  mesh's index space. `QuadMesh::to_world` takes it to the mesh's world
  coordinates, and an inner transformer, whose destination geotransform
  is the identity (`WORLD_GT`), takes it to the source. So the source is
  whatever the inner transformer's is:
  - `from_grid` builds a `GenImgProjTransformer` with the source's size,
    so a longitude/latitude source wraps round its centre as gdalwarp's
    does (`CENTER_LONG`) and reports `source_wrap`;
  - a `GenImgGeolocTransformer` gives geolocation arrays as the source;
  - a `MeshTransformer` with `WORLD_GT` gives mesh to mesh.

  `source_wrap` and `source_cells` pass through, the latter with the mesh
  step in front of the inner `dst_to_world`.
- **The cells.** The lattice's points lie on the sides of index-space
  pixels, where both cell maps are linear, so each outline is the cell's
  own quadrilateral, `densify` points a side, taken to the source exactly
  and joined straight there. Points shared by neighbours are transformed
  once, so the outlines tile. Each cell is measured on its own, so the
  mesh's seam, fold and period play no part on this side, and a halo
  is dropped at construction. Areas are in the source's space, under the
  weights' measure (`lonlat_measure` on the sphere). A cell's sides are
  straight in the mesh's coordinates, not the source's, so near a pole
  in longitude and latitude it needs `densify`: the cap's cell over the
  pole is 69% off in area at densify 1 and 1.1% at densify 8, against
  ESMF's great circles.
- **Why not the mesh source's overlaps, transposed** (the first plan
  here)? They would be measured in the mesh's world coordinates, and
  only `exact_weights` would get them. This way `warp()` targets a mesh
  with any kernel, the source keeps its own measure (areas on the sphere
  for lon/lat), and any source works, a mesh included. In one plane the
  two agree (`tests/mesh_target.rs`).
- **Tests.**
  - `tests/mesh_target.rs`:
    - in one plane, grid to mesh overlaps are mesh to grid's, transposed
      (2e-14 of a pixel); each cell's `dst_area` is its area (2e-13);
    - mesh to mesh is symmetric (7e-15 of a cell);
    - from a global 2.5 degree lon/lat grid with areas on the sphere:
      - every cell is covered to 2e-13, across the source's seam and
        over the pole (the pole's cell takes all 144 cells of the top
        row);
      - the seam elsewhere gives the same overlaps (6e-15 of a pixel);
      - the cap's area on the sphere approaches its LAEA area at second
        order in `densify` (7e-4, 5e-5, 3e-6 at 2, 8, 32);
    - `warp()`'s `Average` is `apply(FracArea)` bit for bit;
    - `Bilinear` of a linear field is exact at cell centres;
    - transform round trips;
    - target windows tile the mesh exactly;
    - the CRS backend matches the written-out projection.
  - `tests/esmf_tripolar.rs`, the `onto_tripolar` fixtures: the same
    grids as 4c, the other way round.
    - Cartesian ESMF matches to 3e-13 (S, `frac_b`, field and areas):
      from the grid in the plane, as itself and as a mesh, and mesh to
      mesh from the lon/lat grid's corners in the plane.
    - Spherical ESMF is recorded. S differs by up to 3e-2 on the cap's
      smallest cells, 13 to 55 km across near its grid poles. There a
      300 km source edge's great circle and straight line part by a large
      share of a small cell (d / h). Over cells more than 100 km across,
      from the grid in the plane, S differs by 3e-3, as in 4c. Areas
      differ by up to 1e-3, on the outer row.

### A declared period (built, 4e)

Built (2026-11-17): `QuadMesh::with_period` and
`Rectilinear::with_period` (`src/mesh/period.rs`), with `unwrap_grid` and
`unwrap_bounds`. World x is periodic: a point and the same point a whole
number of periods away are one place, and the seam's two walls, a period
apart in the plane, are crossed by taking copies, not by gluing walls.

- **Polygons.** `burn` and `cover_ring` take each polygon, holes and
  all, at each whole period at which its exterior ring overlaps the
  grid's x range (overlap with area; touching doesn't count), walk each
  copy as it is, and burn the copies together as one geometry. Polygons
  are planar: one across the antimeridian needs continuous longitudes,
  and one round a pole is closed along it. No ring is unwrapped, since
  a jump of 360 is a polygon the other way round. A ring that would be
  taken at more than 16 periods is refused. `ring_to_index` takes its
  ring as it is.
- **Points.** `locate`, `locate_all`, `to_index` and `Rectilinear::cell`
  try the point as it is, then at each whole period within the grid's x
  range. `contains` and `local` are about one cell, in the plane; the
  walk uses a planar `locate`.
- **Mesh sources.** `QuadMesh::window` carries the period. The lattice
  takes it from the window's mesh, as a `SourceWrap` of the period with
  the poles a quarter period from y = 0 (latitude 90 and -90 in the
  longitude's unit). Each outline is then unwrapped, closed along the
  nearer pole if it goes round one, and walked at each whole turn, as
  for geolocation arrays (3b). No trait change: `SourceCells::lon_wrap`
  stays the geolocation switch.
- **Checks.** `with_period` refuses a period that isn't finite and
  positive, a cell half a period wide or more (a cell round a pole: the
  cell over the North Pole of a tripolar grid in longitude and latitude
  spans 180 to 290 degrees), and a cell that overlaps one a whole period
  away (the mesh goes round more than once), found by the overlap probes
  of 4b moved by whole periods. `find_overlap` includes the period once
  declared.
- **Unwrapping.** `unwrap_grid` moves each value of a point grid
  (corners or centres) to within half a period of the one before it
  along its row, and of the one above down the first column, as
  geolocation arrays are unwrapped (3b). `unwrap_bounds` does it cell by
  cell for CF bounds: each cell's first vertex to the previous cell's,
  its other vertices to its first. `from_corners` and `from_cf_bounds`
  name the helper when the corners they refuse look like longitudes that
  jump: CF bounds in `[0, 360)` jump inside a cell, which shows only as a
  fold.
- **Tests.** `tests/mesh_period.rs`:
  - generator 5: a 10 degree global grid with its columns rotated by 0,
    1, 7, 18 and 35, as a `QuadMesh` and a `Rectilinear`, burns 61
    polygons (random ones with holes, many across the seam and given at
    other periods, and a cap round the south pole) to the same fractions
    as controlledburn's regular burn over five periods, folded back:
    bitwise at `f32`. Each polygon's covered area is its area to 4e-7
    (the fractions are `f32`);
  - a curved mesh covers the same, renumbered, wherever its seam is;
  - polygons and points at other periods, and `burn_par`;
  - `cover_ring` across the seam against `burn`;
  - points at any whole period, and on the seam's far wall;
  - corners, centres and CF bounds stored with a jump, and NEMO's cyclic
    halo: dropped, the same mesh; left in, refused by `with_period`;
  - what `with_period` refuses;
  - `MeshTransformer` with a lon/lat destination from -180, from 0 and
    offset (pixels across every rotated seam), and with a north-polar
    LAEA destination whose centre pixel holds the pole: the weights of
    every rotation are mesh 0's to 1e-14, the cells north of 50 N are
    shared out to 4e-16, the pole's pixel takes all 36 top-row cells and
    covers its closed outline's area to 1e-9, `warp()`'s `Average` is
    `apply(FracArea)` bit for bit, and the CRS backend agrees with the
    projection written out (1e-15 in weights, 1e-14 in areas).
- **ORCA in longitude and latitude** (scratch, CMIP6's IPSL grids, with
  `unwrap_bounds`, `Halo { cols: [1, 1] }` and the period): eORCA1 is one
  sheet from 75 S (row 48; the ice-shelf rows south of it fold in
  longitude too) to the row below the pole's cell, 360 x 282 cells and
  99.9% of its ocean area. ORCA2 is one sheet up to 24 N, 180 x 93 cells
  and 79% of its ocean area; a bow-tie stops it there, in the band whose
  bounds were found corrupt in 4b. Weights from either to a 1 degree grid
  (from -180 or from 0) share out every cell to 9e-16, in 0.2 s (ORCA2)
  and 0.4 s (eORCA1) on one thread, release build.

### A whole ocean grid in pieces (built, 4g)

Built (2026-11-17, decided with the user: leave damaged land out, two
polar LAEAs): `rwarp::mesh::MeshSet` (`src/mesh/set.rs`), with `Band`
and `Piece`.

- **Bands.** `MeshSet::from_cf_bounds(ncol, nrow, xb, yb, halo, bands,
  droppable)` splits the rows (after the halo) into bands, each with a
  `CrsTransform` from the bounds' coordinates to its world. Each band's
  bounds are projected and its corners found there by `from_cf_bounds`'
  own steps (`cf_corners`: vertex order, shared corners, a T pivot's
  halved last row in the last band).
- **Cells left out.** In each band, cells that are bow-ties or turned
  against most of the band are left out where `droppable` allows (land),
  and are an error otherwise, named by their place in the file. Then
  overlaps, by the overlap probes of 4b across all the band's pieces:
  each droppable cell of an overlapping pair is left out, and the pieces
  made again (up to eight rounds).
- **Pieces.** The cells kept in a band are covered by rectangles, each
  row's runs merged with the rectangle above when they span the same
  columns. Each is a `QuadMesh` (`EqualArea`) of its own, so a cell left
  out is a hole that no walk passes through. A piece that spans the
  seam or the fold has them glued inside it (4a); between pieces nothing
  needs joining, since each walk measures its own cells.
- **Why not repair the land?** On both files a Laplace fill of the bad
  cells' corners, grown one to three cells within land, made more bad
  cells, not fewer (eORCA1 86 to 166 or more; ORCA2 122 to 189 or more).
  The bad cells are one-cell strips whose corners all belong to land
  cells that look fine.
- **Weights.** `weights_to(transformer, dst_ncol, dst_nrow, opts)` runs
  `exact_weights` through each piece's `MeshTransformer` and merges the
  results, sources in the set's numbering; a target pixel's `dst_area`
  is from the piece covering most of it. Each piece is measured only over
  the target pixels it reaches: the box of its boundary (full width when
  that spans half of it), grown on each side its cover touches until it
  touches none. A piece's cells are one connected region, so that is all
  of them, and the weights are the same bit for bit (densify 1 and 4) as
  over the whole grid. eORCA1 to 1 degree took 5.2 s without the windows
  and takes 1.0 s with them. `weights_onto(inner, src, opts)` runs a
  `MeshTarget` on each piece. Cells left out get no entries and a NaN
  `dst_area`.
- **Where bands meet** each side's cells are straight in their own
  projection and bow apart between corners: a gap covered by neither, up
  to R dlam^2 / 4 high on each side at the equator, about dlam^2 / (6
  dphi) of each cell next to it (2.3% for ORCA2's 2 by 0.5 degree cells
  at the equator; under 1% at 20 S, where its cells are 2 by 1.9).
- **Tests.** `tests/mesh_set.rs`, an analytic global grid in lon/lat CF
  bounds (48 columns, regular rows from 78 S to 60 N, a tripolar cap
  with a seam, a fold and the pole), two land blocks damaged by a moved
  corner, bands in south-polar and north-polar LAEA split at the equator:
  - the damaged land left out, and only land; each cell kept in exactly
    one piece; undamaged, one piece a band;
  - refusals (damage that may not be left out, bands that don't tile,
    the file's numbering under a halo);
  - to a 2.5 degree grid: every cell kept shared out to 7e-15; pixels
    next to the equator short by the gap (8.7% for these 7.5 degree
    cells, under its bound of 9.8%) and nowhere else (5e-14); away from
    the damage, the undamaged set's weights (5e-15);
  - windows: each piece's weights as over the whole grid, bit for bit;
  - onto the set, areas on the sphere: every cell covered to 1e-13;
  - NEMO's halo round the same cells gives the same set.
- **On the CMIP6 files** (scratch; split at 20 S): eORCA1 builds in 0.13
  s as 50 pieces, 716 land cells left out and no ocean, its ocean area
  0.9999 of `areacello`; to 1 degree every cell is shared out to 1e-12
  (1.0 s), and from 1 degree onto it covered to 5e-10 (0.4 s). ORCA2: 56
  pieces, 801 land cells left out and no ocean, 0.9994 of `areacello`
  (0.9998 without the halved fold row), shared out to 8e-14 (1.2 s),
  covered to 1e-9 (0.1 s). The worst covered are 0.03 km² slivers by the
  grid poles.
- **On threads** (2026-11-18): `weights_to_par` and `weights_onto_par`,
  the same tables bit for bit. Pieces with at least `1 / threads` of the
  cells (on eORCA1 the main piece of each band took 0.45 s and 0.24 s of
  1.1 s) are measured one after another on every thread through
  `exact_weights_par`; the rest at once, one a thread. To 1 degree at
  densify 8: eORCA1 3.0 s to 0.42 s on 8 threads, ORCA2 3.6 s to 0.52 s.
- **Not built here.** Burning came in 4h, `warp()` from a set in 4i.

### Burning onto a set (built, 4h)

Built (2026-11-17, decided with the user: bigcurve as a plain
dependency, edges straight in longitude and latitude):
`MeshSet::burn(geoms, bands, tolerance, opts)` in `src/mesh/set.rs`.

- **Geometries in the bounds' coordinates.** The burn takes the bands
  the set was built with (their rows are checked) and, for each, cuts,
  densifies and projects every polygon, then burns it on each piece
  whose box in that world it meets, as a `QuadMesh`. Records are moved
  to the set's numbering and merged in controlledburn's order (by
  geometry, then row and column). Pieces on one row are always apart (a
  rectangle holds one maximal run of each of its rows), so no run needs
  joining across pieces. Points are found in the first piece holding
  them. Lines are refused, as on a `QuadMesh`.
- **The far pole.** A polar LAEA sends its antipode to the edge of its
  disk, so a polygon round it (Antarctica in the north band, a
  whole-world box in either) comes out inside out. Each band cuts its
  polygons to the y range of its kept cells' vertices, widened by its
  tallest cell (Sutherland and Hodgman, one side at a time), before they
  go to its world. Cutting per piece instead was tried and dropped. It
  was slower, because neighbouring strips densified the same polygons
  again. It was also wrong on ORCA2's garbled land cells near 26 N,
  whose meshes are built from shared corners and reach well beyond
  their own bounds.
- **Densifying.** Each edge is cut into equal pieces no longer than the
  band's tallest cell, from its lexicographically smaller end (an edge
  shared by two rings is cut the same from both). Each piece then goes
  to bigcurve's `Resampler` with `Planar` edges and the caller's
  tolerance in world units, one batched forward call per level. The
  first cut stops d3-geo's test from passing over an edge whose ends
  land together, such as a parallel all the way round.
- **Cuts at 180 and along a pole.** In polar LAEA both sides of a cut at
  180 land on one line, within 1e-16. Where that line is also a piece's
  outer wall, the walk's choice of boundary arc (the one of least
  winding about an interior point) fails. A whole-world box burned
  nothing on the one-row piece whose west wall is the 180 meridian: each
  excursion with its closing arc wound round the piece. So each
  projected ring is split where it comes back to a point it has been at
  (within 1e-12 of its largest coordinate). Loops with under 1e-12 of
  its box's area are dropped. A loop with the ring's own turn keeps the
  ring's role, the others take the other, and holes go to the smallest
  outer ring round their first point. A whole-world box becomes the
  circle of its cut, and Antarctica cut at a parallel becomes its coast
  and a hole.
- **Tests.** `tests/mesh_set.rs`, on the analytic grid of 4g:
  - a whole-world box (from -180 and from 0) covers every cell kept in
    full, round the fold and the pole, including the piece walled at
    180;
  - five boxes on grid lines against closed forms, to 8e-8. The forms:
    each cell edge is a chord of its parallel in its band's LAEA, so a
    box edge along the parallel at `L` adds or takes the segment
    `rho(L)^2 (dlam - sin dlam) / 2`. The boxes:
    - the two polar caps (the far band gets nothing);
    - a box across the bands with a hole;
    - a box across 180, written past it and split at it, the same;
    - a cap cut along a piece's wall;
  - points in each band and in the cap, and none in the gap at the
    equator, south of the grid or in a cell left out;
  - refusals: lines, other bands, a tolerance that isn't positive, modes
    other than coverage.
- **On the CMIP6 files** (scratch; split at 20 S, tolerance 1 m):
  - a whole-world box covers every kept cell of eORCA1 and ORCA2 in full;
  - five boxes on 1 degree lines agree on every ocean cell, to 8e-6
    (eORCA1) and 6e-6 (ORCA2), with `weights_onto` from a 1 degree
    indicator at densify 64. The boxes: the Arctic past the fold, the
    Antarctic, across 180, across the split, and the far south. At
    densify 16 one 30 degree wide land cell of ORCA2 was 1e-3 out, and
    the gap fell sixteenfold at 64: the weights' outline error, not the
    burn's.
  - Natural Earth 10 m countries (258 geometries, 548k vertices) burned
    in 1.0 s onto eORCA1 and 3.1 s onto ORCA2 as built in #85; with the
    filter below, 0.74 s and 1.1 s. No cell sums past 1, and Antarctica
    lands only in the south band.
- **Not built.** Nothing more for burning; `burn_par` below.

### Burning on several threads (built, 4h follow-up)

Built (2026-11-17, decided with the user: the same signature as `burn`
plus `threads`, projection on the calling thread):
`MeshSet::burn_par(geoms, bands, tolerance, opts, threads)`. `burn` is
`burn_par` on one thread.

- **Band by band**, as `burn` goes, so the first error is the same one.
  - Each band's polygons go to its world on the calling thread, because
    a `CrsTransform` needn't be `Sync` (libproj's handles aren't).
  - The band's pieces then filter and shorten their polygons in
    parallel, largest pieces first.
  - The pieces' polygons are walked into their index spaces in chunks of
    about equal vertex counts (`vertex_chunks`, as `QuadMesh::burn_par`
    cuts them), largest chunks first, from one queue over all the
    band's pieces. So one big piece doesn't hold up the rest.
  - Each piece is burned with `controlledburn::burn_par` in order. That
    stage is cheap (0.02 s of ORCA2's 1.1 s), and controlledburn sizes
    its threads to the work.
- **The same records.** The walk is per geometry and the chunks are put
  back in order, so each piece hands controlledburn the same plane as
  on one thread. controlledburn's own `burn_par` gives `burn`'s records.
  With one thread nothing is spawned (for wasm), and the steps are
  `QuadMesh::burn`'s. `walk.rs` now shares `map_geoms`, `vertex_chunks`
  and a `burn_plane` (the controlledburn step) with the crate.
- **Tests.**
  - Unit (`src/mesh/set.rs`): on the damaged two-band grid, the 65
    polygons and 50 points burn the same on 0, 1, 2, 3 and 8 threads,
    filtered, and on 3 threads unfiltered.
  - `tests/mesh_set.rs`: boxes and points on the analytic grid, on 0, 2
    and 5 threads.
- **On the CMIP6 files** (scratch, 32 cores): Natural Earth 10 m
  countries, the same records on every count. The projection is the
  floor: 0.33 s at 1 m and 1.6 s at 0.01 m on ORCA2.

  | | 1 thread | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | eORCA1, 1 m | 0.75 s | 0.54 | 0.45 | 0.40 | 0.40 |
  | eORCA1, 0.01 m | 4.00 s | 2.86 | 2.32 | 2.06 | 2.02 |
  | ORCA2, 1 m | 1.12 s | 0.73 | 0.54 | 0.44 | 0.40 |
  | ORCA2, 0.01 m | 5.59 s | 3.65 | 2.69 | 2.20 | 1.99 |
- **Not built** here: projection in parallel. The user chose to keep
  `burn`'s signature; `burn_par_with` below projects on the threads.

### Projecting on the threads (built, 4h follow-up)

Built (2026-11-17):
`MeshSet::burn_par_with(geoms, bands, tolerance, opts, threads,
to_world)`. `to_world: Fn(usize) -> Result<T, String> + Sync`, `T:
CrsTransform`, builds band `k`'s transform, as `weights_onto`'s `inner`
builds a transformer for each band. `burn_par` is it with projection on
the calling thread.

- **A transform per thread.** A `CrsTransform` needn't be `Send` or
  `Sync` (libproj's handles are neither), so each thread that projects
  into band `k` builds its own, once, before it takes a polygon. A
  thread that starts after every polygon is taken builds none.
  `to_world` must give what `bands[k].to_world` gives. With one thread,
  or a band of one polygon, it isn't called, and the band projects on
  the calling thread as in `burn`.
- **The unit is a polygon**: each polygon of a multipolygon is cut,
  densified and split on its own (as `project` did them in turn), those
  with the most vertices first. A geometry's parts are put back in its
  polygons' order, so each piece sees what it saw on one thread.
- **Errors.** Every polygon of the band is projected, and the first that
  failed, in order, gives `burn`'s message. A `to_world` error is
  returned as `band k's transform: ...` once the other threads finish.
- **New in `crs`:** `CrsTransform` for `Box<C>` and `&C`, so the boxed
  transforms of `crs_pair` serve as `T`.
- **Tests.**
  - Unit (`src/mesh/set.rs`): on the damaged two-band grid, the 65
    polygons and 50 points burn the same with per-thread copies on 0, 2,
    3 and 8 threads (and unfiltered on 3). `to_world` is called at most
    once a band a thread, and not at all on one thread. Its error comes
    back. With a transform that refuses points north of 85 N, the first
    failing polygon gives the same message as on one thread, though the
    polygons with more vertices are projected first.
  - `tests/mesh_set.rs`: the analytic grid's boxes and points on 0, 2
    and 5 threads, and boxes through `crs_pair`'s transforms (libproj or
    proj4rs) on 0 and 3 threads.
- **On the CMIP6 files** (scratch, 32 cores, the machine shared):
  Natural Earth 10 m countries, the same records on every count, with
  `burn_par` / `burn_par_with` times. The bands' analytic polar LAEA:

  | | 2 threads | 4 | 8 | 16 | 32 |
  |---|---|---|---|---|---|
  | eORCA1, 1 m | 0.54 / 0.40 s | 0.47 / 0.22 | 0.40 / 0.15 | 0.40 / 0.13 | 0.41 / 0.14 |
  | eORCA1, 0.01 m | 2.76 / 2.03 s | 2.22 / 1.07 | 2.08 / 0.77 | 2.03 / 0.71 | 2.02 / 0.71 |
  | ORCA2, 1 m | 0.73 / 0.58 s | 0.53 / 0.31 | 0.44 / 0.18 | 0.40 / 0.13 | 0.40 / 0.12 |
  | ORCA2, 0.01 m | 3.63 / 2.84 s | 2.72 / 1.56 | 2.22 / 0.92 | 2.02 / 0.73 | 1.96 / 0.56 |

  Through libproj (`crs_pair`), ORCA2 at 1 m: 0.25 s on 8 threads,
  0.20 s on 16, 0.31 s on 32; at 0.01 m 1.01, 0.71 and 0.65 s.
- **Building costs.** A libproj transform takes about 5 ms to build, and
  libproj builds them more or less one at a time: 32 threads building at
  once took 0.2 s. On a light burn the threads left building hold up
  the end. Starting threads in a chain, each starting the next once its
  transform was built, took 32 threads at 1 m to 0.28 s but 8 threads
  to 0.31 s: transforms built one after another, about 5 ms each while
  the other threads project, ramp up too slowly. Kept: all threads
  started at once.
- **Not built.** Cheaper libproj copies. PROJ's `proj_clone` copies a
  transform into a new context without the database, but the `proj`
  crate (0.31) doesn't offer it.

### Walking only what reaches a piece (built, 4h follow-up)

Built (2026-11-17). In #85 each piece took every polygon whose box met
its own. A one-row piece round most of a latitude circle (ORCA2 has
dozens beside its garbled land near 26 N) has a box the size of its
band's disk, so each walked nearly all of the band's vertices: 525k
for a strip of 140 cells, 3.1 s in all for the countries.

- **Tiles.** Each piece gets square tiles over its box in its world:
  twice as many along the longer side as it has cells along its longer
  one (4 to 512), and one more all round. Each cell's corners' box
  marks its tiles, widened by one, and the marks are held as a
  summed-area table. A box or segment that meets no marked tile can't
  meet the piece, and is a tile or more from all of it.
- **Parts.** A polygon whose outer box meets no marked tile is skipped.
  It can't hold the piece, since then its box would hold the piece's
  box.
- **Runs.** In each ring, a run of segments apart from the piece is cut
  short, greedily. Each shortcut runs as far as it can while it is apart
  too, and closes a loop of winding 0 round the walk's interior point
  `p0` with the stretch it replaces. Being apart from the piece, the
  loop winds as often round all of it. Its crossings of `p0`'s row are a
  tile or more from `p0`, so the walk's own floating-point winding test
  counts them alike. A ring wholly apart is dropped if it winds round
  nothing (an outer ring takes its polygon with it), and is stood in for
  by a rectangle beyond the tiles if it winds round the piece.
- **Why it's the same.** The walk uses a stretch outside the mesh only
  through its crossings (none, apart) and through the winding of each
  excursion, closed by a boundary arc, round `p0`. The vertices it
  locates, the final locate hint and the ring's start are unchanged.
  So the index rings, and the burn, are the same bit for bit.
- **Tests.**
  - Unit tests in `src/mesh/set.rs`: on a damaged global grid in two
    polar LAEAs (with one-row pieces), 65 polygons (blobs with and
    without holes, caps, an annulus round the pole, a box split at 180,
    a sliver along a parallel, the whole world) burn the same with the
    filter and without, record for record. A ring of 2000 vertices
    across a strip keeps 14; one apart from it is dropped, and one round
    it becomes the rectangle.
  - On the CMIP6 files (scratch), the countries burn hashes the same as
    #85's at 1 m and 0.01 m. Times on eORCA1: 1.0 s to 0.74 s (6.0 to
    3.8 s at 0.01 m). On ORCA2: 3.0 s to 1.1 s (15.1 to 5.5 s).
- **What's left** on ORCA2's north band at 1 m:
  - projecting the polygons, 0.26 s (on the threads since
    `burn_par_with`);
  - the filter's scans, 0.23 s, one tile test per segment per piece
    near it (0.05 s with chunk boxes, below);
  - walking the edges near pieces, 0.53 s: about 1M vertices, against
    546k projected, since neighbouring strips each walk the edges beside
    them.

### Chunk boxes in the filter (built, 4h follow-up)

Built (2026-11-18). The filter tested every segment of a part that
reached a piece, and again every vertex of a far run for its shortcuts:
0.23 s of ORCA2's north band at 1 m on one thread, 1.18 s at 0.01 m.

- **Chunks.** Each ring, once projected, carries the box of each 32
  segments in turn (`Part::chunks`, both ends of each segment in it),
  shared by every piece of the band.
- **Apart at once.** A chunk whose box meets no marked tile has every
  segment apart: a segment's box lies in its chunk's, and a smaller box
  reaches fewer tiles. Only the segments of chunks that meet a marked
  tile are tested one by one, and a run is found by passing far chunks
  whole.
- **Shortcuts a chunk ahead.** In a run, a shortcut first tries a whole
  chunk ahead, then one vertex at a time to that chunk's end if that
  fails. A chunk's crossings of `p0`'s row are none when its box is all
  above or all below the row (the walk counts a crossing only for a
  segment whose ends' y straddle it), so the loop's winding costs a
  test a chunk there. A ring wholly apart gets its winding the same way.
  Each shortcut kept is still apart from the piece and closes a loop of
  winding 0 round `p0`, so the walk's index rings are the same. Which
  vertices are kept changes a little (ORCA2 at 1 m walks 995,430
  vertices, 995,469 before).
- **Chunk size.** 16 to 128 segments gave 0.044 to 0.050 s of filter on
  ORCA2 at 1 m, and 8 gave 0.059 s. 32 is kept.
- **Checks.** The unit test that the trimmed burn is the whole burn
  (the 65 polygons on the damaged two-band grid) passes unchanged, and
  the strip's 2000-vertex ring still keeps 14. The Natural Earth 10 m
  countries hash the same as main's on eORCA1 and ORCA2, at 1 m and
  0.01 m.
- **On one thread** (scratch): the filter on ORCA2's north band 0.23 s to
  0.05 s at 1 m and 1.18 to 0.16 s at 0.01 m; eORCA1's south band 0.39
  to 0.03 s at 0.01 m. ORCA2's countries 1.1 s to 0.96 s at 1 m and 5.6
  to 4.6 s at 0.01 m. On 8 threads with `burn_par_with`, 0.92 s to
  0.79 s at 0.01 m.
- **What's left** is the walk itself: neighbouring strips each walk the
  edges near them, about 1M vertices at 1 m on ORCA2 for 546k projected.

### warp() from a set (built, 4i)

Built (2026-11-18): `rwarp::mesh::SetTransformer` (`src/mesh/set_source.rs`),
a `Transformer` whose source pixels are a `MeshSet`'s cells in the set's
numbering, from a destination grid's geotransform and CRS. The
constructors match `MeshTransformer`'s: `new(set, band_crs, dst_gt,
dst_crs)` with the compiled-in backend, `with_backend(set, dst_gt, crs)`
with a pair of `CrsTransform`s a band. It borrows the set.

- **Several worlds in `SourceCells`.** The trait gains three defaulted
  methods: `worlds()` (1), `dst_to_world_in(world, x, y)`
  (`dst_to_world`) and `sheets(window)` (one `SourceSheet`: the window's
  `mesh` in world 0). A set reports a world a band and each piece's part
  of the window as a sheet, a mesh with its offset in the window. A mesh
  target passes them through from its source. Sources that are one sheet
  are unchanged, bit for bit (ORCA2's `weights_to` and `weights_onto`
  hash the same as main's, densify 1 and 4).
- **The lattice in each world.** `warp()`'s exact footprint and
  `exact_weights` share `Lattice`. It now keeps its transformed row lines
  and column interiors per world. With several worlds or sheets, each
  destination pixel's outline is built in every world and walked onto
  each of that world's sheets that it reaches. Their coverage, offset to
  buffer pixels, is one `Cover`: a `Window` when one sheet is reached,
  `Pieces` otherwise, so one reduction takes cells from every piece and
  band. Each buffer pixel's world area is kept for the reduction.
  `outline_area` (a weight table's `dst_area`) is the outline's area in
  the world whose sheets it covers most, as `weights_to` takes it.
- **Which sheets.** A sheet's box in its world, then the burn's tiles
  (`Tiles`, now shared). A one-row piece round a latitude circle has a
  box the size of its band's disk, and every outline in the band met it:
  ORCA2's `Average` took 1.6 s at densify 2 with boxes only, 0.88 s
  with the tiles. An outline whose box meets none of a sheet's tiles
  neither reaches nor holds a cell of it, so nothing changes.
- **Inside out.** A polar LAEA sends its far pole to its edge. A
  destination pixel round that pole (the middle pixel of an Antarctic
  grid, in the north band's world) comes out as a ring round the whole
  disk, turned the other way, and would cover every cell of the band.
  Each world's turn is found once, by majority, from small triangles
  (1e-3 pixel) at pixel centres along a row. An outline whose shoelace
  area has the other sign isn't measured in that world. Without the
  test, the Antarctic test below takes the north band at the pole. One-
  world sources don't get the test, so they are unchanged.
- **Point kernels.** `transform` maps a destination pixel to each band's
  world in turn and into the first piece whose cell holds it (the last
  piece first, with its cell as the hint), then adds the piece's offset:
  the set's index space. So bilinear and cubic read neighbours across
  pieces and across the bands' meeting, as across a file's arrays. The
  reverse goes through the piece that owns the cell (or a neighbour on
  a shared edge). Points in a cell left out, in the gap or off the grid
  fail.
- **Tests** (`tests/mesh_set.rs`, the analytic grid of 4g):
  - `exact_weights` through the set are `weights_to`'s, fractions equal
    (0) and target areas to 2e-14; 2 x 2 source windows (across the
    bands' meeting) tile the whole to 6e-15;
  - `Average` is `apply(FracArea)` of warp-compatible weights bit for
    bit (densify 1 and 3), a constant stays constant to 1e-14, `Mode` of
    the bands takes the band each pixel covers most;
  - nearest lands in the cell the pixel centre's longitude and latitude
    give (7206 pixels) and never in a cell left out; bilinear of a field
    linear in the indices is linear to 6e-14, over 1876 pixels that read
    more than one piece, 270 of them both bands;
  - an Antarctic grid in south-polar LAEA takes only the south band;
  - a set of one piece warps as its mesh through `MeshTransformer`, bit
    for bit, for `Average`, `Mode`, `Max`, bilinear, cubic and nearest.
- **On the CMIP6 files** (scratch, one thread, a smooth field of cell
  centres, NaN on land, to 1 degree): `Average` is `weights_to` and
  `apply` bit for bit, at densify 2 and 8. eORCA1 0.91 s (1.06 s by
  weights) and 1.98 s (2.60); ORCA2 0.88 s (1.21) and 2.28 s (3.58).
  Bilinear and nearest 0.02 s, `Mode` 0.9 s. A 50 km south-polar LAEA
  grid, 160 x 160 pixels round the pole: no pixel takes the north band.
- **Threads** came after (2026-11-18): `warp_to_par` takes a closure
  that builds a `SetTransformer` on each thread. ORCA2's `Average` at
  densify 8: 2.3 s on one thread, 0.34 s on 8, 0.14 s on 32, bit for bit.
  The lattice's turns are now found over the whole destination window,
  on a grid of up to 9 by 9 pixel centres, when it is built (they were
  found along its first rows), so every band of rows finds the same.
- **The file's layout** (2026-11-18): `MeshSet` keeps the file's size and
  halo. `from_file_layout` drops the halo from a field on the file's
  grid. `to_file_layout` puts a field in the set's numbering back,
  masking the halo with a fill. At a T-point pivot it gives each cell of
  the last row the area mean of its two halves, found by the fold side
  they share (the top side of one is the other's, reversed), leaving out
  a half that is the fill, NaN or a cell left out. NEMO's mirror
  `(ncol + 2 - c) % ncol` is taken first among the cells that match:
  ORCA2 has 19 cells by its second pivot collapsed to one point, which
  all match each other. All 170 pairs kept in ORCA2's last row are
  merged into both copies. `fill_halo` writes
  NEMO's halo for `NEMO_F_PIVOT` and `NEMO_T_PIVOT`: columns `nx - 2`
  and `nx - 1` from 0 and 1; the fold row from column `nx - 1 - i` of
  the row below (F) or column `nx - i` of the row two below, with the
  row below mirroring itself (T). Those were checked on both CMIP6 files
  by cell centres, 362 of 362 and 182 of 182 (`areacello` is masked in
  the halo). Tests: the ORCA-like grids of `tests/mesh_cf.rs`, both
  pivots, centres back to 1e-9 with the halo filled; a T pivot's halves'
  area mean, one half missing; refusals.
- **Vector components across the fold** (2026-11-18):
  `to_file_layout_vector` and `fill_halo_vector`, for components in the
  grid's directions (every band a component; components east and north
  are scalars). `fill_halo_vector` is NEMO's `lbc_lnk` with `psgn = -1`
  at T points: the fold's copies change sign, the cyclic columns'
  don't, the cyclic first and then the fold, as in NEMO 3.6's
  `lbclnk.F90` and `lbcnfd.F90` (4.0's `lbc_nfd_generic.h90` is the
  same). That includes the T pivot: column `nx / 2` of the row below
  the fold is its own mirror and changes sign, and column `nx - 1` of
  that row is column 1 turned round, though the cyclic halo says column
  1. A component that agrees with its own halo is zero at a pivot, and
  ORCA's pivots are on land. `to_file_layout_vector`'s T-pivot halves
  are in opposite directions (each is the lower half of its own copy),
  so the other half's value is turned round before the mean by area,
  and the second copy has the mean turned round. Checked on both CMIP6
  files with components of a solid-body rotation along each cell's own
  row and column directions from its bounds: the files' fold copies are
  their mirrors' turned round exactly (724 and 533 components), and
  through the set and back they come back to 0 and 1.1e-16 (the scalar
  helpers: 1.9 and 1.8). Tests: NEMO's `lbc_lnk` transcribed from the
  Fortran, both pivots, both signs, cell for cell; the same geometric
  check on the ORCA-like grids; the halves' mean turned round, one half
  missing.
- **U, V and F points** (2026-11-18): `fill_halo` and
  `fill_halo_vector` take a `Stagger` (T, U, V, F: an Arakawa C grid's
  points, each an array of the file's size), the set giving the pivot.
  The fold is NEMO 4.0's `lbc_nfd_generic.h90` on one process,
  transcribed loop for loop, 0-based and in its order (a row copied
  onto itself is copied in place). 3.6's `lbcnfd.F90` differs only in
  the last column of U and F points at an F pivot, which it takes from
  column 0; its mirror is column `nx - 3`, as 4.0 has it and as the
  grid's geometry says. Two differences from NEMO: at a T pivot, NEMO's
  V and F points leave one cell of row `ny - 2` (column 0 at V, `nx - 1`
  at F) as the cyclic columns had it before the fold, so in a field
  with its halo masked it stays masked; here it is the cyclic copy of
  the fold's, the same for a field that agrees with its own halo. And the cyclic
  columns go from 0 and 1 to `nx - 2` and `nx - 1`, as before (NEMO
  copies column `nx - 2` to 0). NEMO's in-place order shows at U points
  at a T pivot: column `nx / 2` of row `ny - 2` is kept and column
  `nx / 2 - 1` copied from it. Checked on both CMIP6 files: U, V and F
  positions from the T cells' bounds (east and north side midpoints,
  north-east corners), and components along the sides at U and V
  points, come back exactly with the halo masked and filled. Tests:
  NEMO's `lbc_lnk` transcribed from the Fortran at every point, both
  pivots, both signs, cell for cell; the same geometric check on the
  ORCA-like grids. Each of nine rules broken in turn fails both.
- **Sets of U, V and F cells** (2026-11-18): `Halo::nemo(pivot, cells)`
  (`Pivot::T`/`F`, `Stagger`), the T cells' being `NEMO_F_PIVOT` and
  `NEMO_T_PIVOT`. U cells' corners are V points, V cells' U points, F
  cells' T points; at a T pivot V and F cells drop the last two rows
  (the fold is the top side of row `ny - 3`), T and U cells the last
  with the fold through row `ny - 2`'s centres; at an F pivot T and U
  cells drop the last, V and F cells the last with the fold through row
  `ny - 2`'s centres. `Halo` gains `nemo: Option<Nemo>` (the pivot and
  the cells' point), as the shape alone doesn't say which grid it is (U
  cells at a T pivot drop what T cells do, V cells at an F pivot too).
  `MeshSet` takes the pivot for `fill_halo` from it, and `halves` NEMO's
  mirror for the point (`k - c` on the cyclic columns: `k` 2 for T, 1
  for U and V, 0 for F), a cell its own mirror (a pivot) with itself
  before any other. U cells at a T pivot and V cells at an F pivot have a
  side through each pivot (corner columns 1 and `nx / 2`), its own
  mirror: `cf_corners` cuts it at its lower end, not its midpoint, which
  rounding can put across the cell's bottom side and turn the cell
  inside out (a tolerance on the side's length instead also moved the
  cut on sides by ORCA2's T cells collapsed near the second pivot,
  changing two cells' areas by 3e-4; by index, T cells are bit for bit
  as before, ORCA2's weights hashes unchanged). Checked on both CMIP6
  files with each point's cells from the T cells' bounds (V points the
  T cells' north sides' midpoints, U points their east sides', T points
  their centres; any cell droppable): every set builds, the cap north of
  60 N is covered to 2e-7, centres come back to 2e-16 with the halo
  filled at the cells' point, and every kept fold-row cell (eORCA1's V
  and F, ORCA2's T and U) pairs with NEMO's mirror. Tests: each point's
  cells on the ORCA-like grids tile the disk once at both pivots, and
  every other halo of that shape is refused or leaves it short; sets of
  each go to the file and back; a fold row collapsed to one point pairs
  by the mirror for each point; a pivot's side apart by rounding.

### The exact footprint at the antimeridian

Built (stage 3, #36): `Lattice::cover` in `src/warp.rs`, for sources
whose transformer reports `Transformer::source_wrap` (a wrapped
geographic `GenImgProjTransformer`), tested in `tests/antimeridian.rs`.
The pole is chosen as the one whose row the outline comes nearest, which
needs no forward transform; a cell measured at two offsets is merged into
one entry. Two cases, both fixed in rwarp with no change to
`CoverageWindow`:

- **Seam crossing.** A destination pixel straddling the source seam comes
  back with x jumping by about the source width. Unwrap the ring (shift
  any jump larger than half a period), then add it to the window at
  offsets `-P`, `0` and `+P`. A closed ring entirely beside the window
  contributes nothing (its clamped pieces cancel row by row), so each
  copy books only its own columns.
- **Ring fold** (a destination pixel containing the pole). After
  unwrapping, the ring's net x displacement is one period and it does not
  close. Close it along the pole row of source index space (to `y_pole`,
  across one period, back), choosing the pole whose forward transform
  lands inside the destination pixel. Then add copies as above.

The same two steps serve the polygon front end on periodic grids.

## 4. World-to-cell location

`geoloc_transform.rs`'s `QuadIndex` buckets every quadrilateral's bbox in
a `sqrt(n)` by `sqrt(n)` grid and takes the first containing one in
row-major order. It stays in rwarp (m20 section 2.1: rwarp is where
front ends live; controlledburn takes index-space rings only). Changes
for coverage, none affecting GDAL parity:

- **Hints.** `locate(p, hint)` walks from a starting cell before using
  the buckets. Consecutive queries (a scanline, a bigcurve midpoint next
  to its parent) cost O(1). The front end needs one location per ring
  start and per point; everything else is the walk.
- **All matches.** `locate_all` returns every containing cell, for fold
  detection; `geo_to_pixel` keeps the row-major first match.
- **Ownership.** A point on a shared side belongs to one cell, by side
  identity, not by tolerance.
- **Boundary index.** The mesh's outer sides, for "where does this
  segment first enter". Built with #36: a bucket grid over the outer
  sides' boxes, queried in the sides' scan order, so the walk's output is
  unchanged.
- **Compact and lazy.** CSR buckets (offsets plus one `u32` array), built
  on first use.
- **Triangles.** For `EqualArea`, the containing triangle and its
  barycentric inverse (closed form, no root).

GDAL's backmap inverse, which wxy-warp carries (`geoloc.rs`), is an
option for parity on folded swaths (m20 rwarp roadmap item 4), separate
from this.

```rust
// as built (stages 1 and 4b); the first sketch had a `topology` argument
// on each constructor, dropped with 4a
pub struct QuadMesh { /* corners (nrow + 1) x (ncol + 1), cell map */ }

impl QuadMesh {
    pub fn from_corners(ncol: usize, nrow: usize, x: Vec<f64>, y: Vec<f64>,
                        map: CellMap) -> Result<Self, String>;
    /// Corners from centres, GDAL's PixelCenter rule.
    pub fn from_centres(ncol: usize, nrow: usize, x: &[f64], y: &[f64],
                        map: CellMap) -> Result<Self, String>;
    /// CF bounds or SCRIP corners, nrow x ncol x 4, halo dropped; corners
    /// must be shared by neighbours; refused when cells overlap.
    pub fn from_cf_bounds(ncol: usize, nrow: usize, xb: &[f64], yb: &[f64],
                          halo: Halo, map: CellMap) -> Result<Self, String>;
    pub fn find_overlap(&self) -> Option<[(usize, usize); 2]>;
    pub fn cell_area(&self, row: usize, col: usize) -> f64;
    pub fn locate(&self, p: [f64; 2], hint: Option<(usize, usize)>) -> Option<(usize, usize)>;
    pub fn locate_all(&self, p: [f64; 2]) -> Vec<(usize, usize)>;
}
```

`GeolocArrays` keeps its `QuadIndex` for the point inverse. Area methods
build a `QuadMesh` of the source window when they need one
(`GeolocArrays::window_mesh`, section 6); under N1 coverage always uses
the equal-area map, so `GeolocOptions` needs no `cell_map`.

## 5. Where bigcurve plugs in

With `EqualArea` cells and polygons in the grid's CRS, the walk is exact
and nothing is densified. bigcurve is needed where an edge is straight in
one space and used in another:

- **Polygons in another CRS.** Densify with the polygon's edge model
  (planar, or great circle for longitude and latitude), forward map = CRS
  transform then the grid's world-to-index, tolerance in cells (1e-3, a
  unit that is the same for every grid). Then walk the densified ring as
  above. Exact to the tolerance. Built as 5b, below, with planar edges
  and the tolerance in world units, densifying into the world (where the
  walk is exact) rather than into index space.
- **The exact footprint.** Destination pixel edges, straight in
  destination pixel space, into source space: the hybrid in the bigcurve
  follow-up (transform the lattice at densify 1 as scanlines, refine only
  edges failing a sagitta tolerance in source-pixel units). Built as 5a,
  below.
- **Bilinear cells**, for anyone who wants the GDAL model's measure.

bigcurve's open question on kinks (a curvilinear map is only piecewise
smooth) goes away if the walk cuts at sides and spokes first: refine only
within each piece, where the map is smooth. For the CRS change, densify in
the grid's CRS (where cells are straight), then walk.

**Batching.** Each new point costs a CRS transform and a location.
Per m20, bigcurve owns the batched forward-map trait (slices in,
per-point success out, the shape of `Transformer::transform`) and rwarp
implements it for its transformers behind a `bigcurve` feature. Two
requests for bigcurve stage 2:

- refine in level order, one forward call per level: each edge's
  refinement depends only on its own endpoints, so the vertices are
  bitwise those of d3-geo's depth-first recursion;
- carry a hint per point (the parent's cell) through to the forward map,
  so rwarp's locate stays O(1) per point.

### Adaptive footprint edges (built, 5a)

`Footprint::Adaptive { densify, tolerance }` in `warp()`, and
`WeightOptions::refine_tolerance` for `exact_weights` and everything
built on it (`MeshSet::weights_to`, `weights_onto` and their threads), in
`src/warp.rs` (`Lattice`).

- **On top of the fixed lattice.** The lattice is built as before:
  `densify` points a pixel along each row line, one scanline, and each
  interior level of the column lines its own scanline. Then every segment
  between two consecutive lattice points goes to bigcurve's `Resampler`
  with `Planar` edges in destination pixel space: a row line's segments in
  one run when it is the band's bottom line, the column lines' segments of
  a row in another. Each run is one forward call per level. A segment's
  new points depend only on its two ends, so the top and bottom pixels on
  a line share them, and the runs are made up the same whether a band
  starts at a row or reaches it, which keeps tiles and threads bit for bit
  the whole window's.
- **The forward map** is the lattice's: to source pixels, or to world `k`
  of the source's cells (`dst_to_world_in`), so a mesh or a set refines in
  its own world and each world on its own. The tolerance is in those
  units. A tolerance in cells for world sources was considered (a scale
  per world, from its cells' sizes) and left out: a source window's cells
  change with the chunk, and the scale with them, so tiles would no longer
  agree.
- **Failures.** bigcurve stops a branch at a midpoint the map cannot
  take, and says nothing. The forward map records the destination
  position of each failure, and its segment is marked failed (a row
  line's by its x, a column line's by its column and level), which fails
  the pixels that use it, as a failed lattice point does.
- **Measured** (`tests/adaptive.rs`; a polar source, 12 by 12 pixels):
  error 1.0e-2, 8.9e-4, 8.0e-5, 9.2e-6, 8.6e-7 at tolerances 1e-2 to
  1e-6 against densify 4096, the points growing as the tolerance to the
  power -0.507. A fixed densify converges at the same rate (its error
  falls as `densify^-2`, the points as `densify`), so the gain is only in
  where the points go: 48 by 4 pixels bending at one end reach densify
  64's error at 1e-4 with 13313 points against 27713. An arc of a circle
  bisects evenly, so on a mesh source in LAEA from longitude and latitude
  the worst overlap at 1 m is densify 64's bit for bit. Across the seam
  and round the pole (`tests/antimeridian.rs`) the pole pixel's cap is
  0.88, 0.22, 0.014 pixels over at 1e-2, 1e-3, 1e-4, within the tolerance
  times the outline's 360 pixel length. On the analytic set
  (`tests/mesh_set.rs`) 10 m is 1.6e-5 from densify 64, densify 4 6.3e-5,
  both worst in the small cells round the pole.
- **Approximate transformers.** A row line's new points at one level
  come as a scanline, which `ApproxTransformer` interpolates, so
  refinement below its error follows the interpolation. The docs say to
  use the exact transformer or a tolerance above its error.

### Polygons in another CRS (built, 5b)

`QuadMesh::burn_from(geoms, to_world, from_world, tolerance, opts)` and
`Rectilinear::burn_from` (and `_par`, the burn on threads), in
`src/mesh/reproject.rs`. They take a band's path of `MeshSet::burn`
(4h) for one grid:

- **Where the grid is in the polygons' CRS.** `from_world` takes the
  grid's vertices back. Their y range, widened by the tallest cell
  each way, is the cut, and the tallest cell's height is the first
  cut's step, as a band's `Clip` is built from its bounds. A set has
  its cells' bounds in the polygons' coordinates already. A single
  grid has only its world, so the inverse is asked for (`crs_pair(grid,
  geoms)` gives both). Vertices that don't go back are left out of
  the cut. A grid none of whose cells go back is refused.
- **Then as a band:** each ring cut to that y range, its edges cut at
  the step from their lexicographically smaller end, densified by
  bigcurve (planar edges, tolerance in world units), and split where it
  meets itself (`project`, shared with the set). Points are taken
  directly. Those beyond the y range, or that don't go, are on no cell.
- **A declared period** (a grid in longitude and latitude): `project`
  now unwraps each densified ring by it, each point to within half a
  period of the one before, before its turn is taken. A ring that then
  doesn't close went round a pole of the world, and is refused. A set's
  bands have no period, so they pass none.
- **Edges are planar** in the polygons' CRS, as on a set. Great-circle
  polygon edges would need the cut at a parallel to follow the great
  circle (Sutherland and Hodgman interpolates linearly) and a first cut
  along it. They are left for later.
- **Measured** (`tests/burn_from.rs`, the tripolar cap in a polar LAEA
  on the sphere, from longitude and latitude). The cap north of 70 N
  is short of `2 pi R^2 (1 - sin 70)` by 0.21 to 0.58 of the tolerance
  times its circle's length from 100 m to 1 cm. The chords' slivers
  predict a sixth to two thirds. The points go from 1028 to 113804,
  tolerance to the power -0.511. A circle bisects evenly, so they double
  or quadruple from one tolerance to the next. A box at 1 cm gives what
  its edges cut into 4096 pieces give, to 1.8e-7 (the fractions are
  `f32`), and is within 1.7e-9 of its closed-form area. The whole world
  covers every cell in full, and the south nothing. A box across 180 is
  the box split at it, to 4e-8 of its area. On a 10 degree lon/lat grid
  with a period, a LAEA polygon across the antimeridian burns as its
  copy turned 180 degrees, 18 columns over, bit for bit. A `Rectilinear`
  in LAEA is its `QuadMesh`.

## 6. Duality with the exact footprint

`Footprint::Exact` builds each destination pixel's outline from a shared
transformed lattice and measures its overlap with source pixels with
`CoverageWindow`. Each warp case is a burn:

| warp | as a burn | exact today |
|---|---|---|
| affine source | small rings onto a regular lattice | yes, to densify |
| geoloc destination | straight quads onto a regular lattice | yes, to densify |
| geoloc source | small rings onto a curvilinear grid | yes, to densify (#36), when the arrays' cells are whole |
| both geoloc | quads onto a curvilinear grid | yes, as above |

The fix is the same front end. Built (stage 3, #36, decided 2026-11-16:
on by default, fall back where the cells are not whole):

- **How the warp gets the cells.** `Transformer::source_cells()` returns
  a `&dyn SourceCells`, as `source_wrap` reports a wrap: `dst_to_world`
  (the pipeline without the source's world-to-pixel step) and `mesh` (a
  `QuadMesh` of a source window, mesh cell = window pixel).
  `GenImgGeolocTransformer` with a geolocation source and `GeolocArrays`
  on its own report it; `ApproxTransformer` passes it through and runs
  `dst_to_world` exactly (it has no inverse lookup, so it costs about
  the same; approximating it is a possible follow-up).
- **The cells.** Corners are `pixel_to_geo` at whole raster pixel/line,
  computed point by point (so neighbouring windows share their edges
  exactly), before longitudes are moved back into -180..180. This takes
  both conventions and any `PIXEL_STEP`/`LINE_STEP` the same way. For
  `TopLeftCorner` at whole-number steps and offsets it is GDAL's bilinear
  model exactly, since lines of constant pixel or line are straight
  within a square of samples. For `PixelCenter` the corners are the means
  of four centres (`from_centres`' rule), and GDAL's model bends each
  pixel side where it crosses a line of samples (a second-order
  difference; following it would take a mesh at twice the resolution).
- **The lattice.** `Lattice::new` takes the source window and builds its
  mesh once per warp call or chunk (about 70 bytes a source pixel). Rows
  go through `dst_to_world`, each outline is walked with
  `ring_to_index` (whose boundary arcs do the clamping `CoverageWindow`
  does in source pixel space) and covered in index space.
- **Weights.** Source cells differ in area, so `warp()`'s area methods
  weight by fraction times cell area; `exact_weights` gives `dst_area`
  as the outline's world area and per-entry `Weights::src_cell_area`. A
  non-planar `WeightOptions::measure` is refused.
- **Fallback.** `GeolocArrays::world_cells()` decides once, from the
  whole arrays: no missing sample, no fold (both orientations), no
  bow-tie, and with `lon_wrap` longitudes that unwrap consistently
  within 360 degrees. Otherwise the exact footprint measures in source
  pixel space as before, so chunks never mix measures.

The measure is planar in the arrays' coordinates: square degrees for
longitude and latitude, true area for arrays in an equal-area CRS.
Tested in `tests/geoloc_exact.rs`: each weight against an independent
clip, generator 8 (to 2e-13; the source-pixel measure fails it at 3e-3
to 6e-3), tiling, affine as geoloc, steps, conventions, windows,
fallbacks, `warp()` against `Weights::apply` bit for bit, and a curved
longitude/latitude grid to polar stereographic.

Across the antimeridian (stage 3b, #72), built:

- **Cells.** With `lon_wrap`, `world_cells()` unwraps the arrays'
  longitudes once by continuity (each sample within 180 degrees of the
  one before it along its row, and of the one above it down the first
  column) and keeps them. Corners are interpolated from the unwrapped
  samples, so every window's mesh is continuous and all windows share
  the same longitudes. The arrays are refused (fallback) if two
  neighbours are still 180 degrees or more apart (round a pole) or the
  samples span more than 360.
- **Outlines.** `SourceCells::lon_wrap` gives the lattice a wrap of 360
  with poles at latitude ±90, and the outline is unwrapped as in source
  pixel space (#70): closed along the nearer pole if it goes round one,
  then walked at each whole turn at which its longitudes meet the mesh's,
  two or more merged as `Cover::Pieces`. This also covers a swath near
  the seam that doesn't cross it, whose outlines may.
- **Weights.** `warp()`'s `Cover::Pieces` branch weights by cell area
  as the window branch does.

Tested in `tests/geoloc_exact.rs`: the same swath with its longitudes
stored from -180, 175, 185, 0 and -200, both conventions, against the
swath with nothing wrapped (to 1.1e-11, the cancellation in areas at
longitude 180 against 0.4 degree cells, which the unwrapped reference
shows against an exact clip too); each weight across the seam against a
clip in local coordinates (2.5e-12); generator 8 across the seam (3e-12);
every swath cell shared out whole; windows split at the seam; a
destination pixel round the pole (conservation 6e-15, its area exactly
360 by the corners' colatitude at densify 1), reaching the swath at two
turns; `warp()` against `Weights::apply` bit for bit on both. Swaths over
a pole stay on the fallback.

What is shared between the two uses is all in rwarp: `QuadMesh`, the
walk, the equal-area map, the periodic and fold handling. What each
hands to controlledburn is an index-space ring. What is not shared:
`Lattice` (the exact footprint's scanline transform), which is warp's
own and the shape bigcurve's batching should follow.

## 7. Output, precision, tests

### Output

The front end returns controlledburn's tables unchanged: `(row, col)` of
the grid's arrays as stored (no flip; a CF grid with latitude ascending
has its southernmost row at row 0) and fractions of each cell's world
area. It also exposes `cell_area`, so callers can weight, and rescales
line lengths per piece to world units (a piece's world length is known
from the walk).

**f32 or f64** is controlledburn's call (hypertidy/controlledburn#24
item 8). This front end makes the truncation visible per cell, since
cell areas vary across a curvilinear grid; `CoverageWindow` is `f64`
already, and the exact footprint rounds to `f32` only to match what
`burn` gave rwarp 0.2. Decided in controlledburn 0.3.0: edges stay `f32`
(about 6e-8 of a cell), documented; decision N6 in
[decisions.md](decisions.md).

### Property tests

Each generator has an exact oracle:

1. **Affine as curvilinear.** A rotated regular grid as corner arrays
   must match the affine front end (controlledburn's `GridSpec` through
   the inverse geotransform). This brings controlledburn's lattice-biased
   ring generators (vertices on walls, centres and corners) over.
2. **Rectilinear as curvilinear.** Must match the rectilinear front end.
3. **Smooth warps.** Corners from analytic maps (polar sector, sinusoidal
   shear) and random perturbations of a lattice, below 0.25 cell for
   convex cells and above for non-convex ones. Oracle: each cell clipped
   by the polygon in world space (Sutherland-Hodgman per triangle, as
   controlledburn's `examples/grid_frontends.rs` does), plus the area
   invariant `sum of area(Q) * fraction = area(P inside the mesh)`.
4. **Steering.** Vertices on corners, sides, spokes and centres; segments
   through corners and centres; edges along sides.
5. **Periodic.** Built (4e, `tests/mesh_period.rs`): a regular global
   grid declared periodic, columns rotated by k, matches the regular burn
   shifted by k, for polygons crossing the seam.
6. **Fold.** Built as the world clip on meshes with a fold
   (`tests/mesh_glued.rs`): a strip wrapped round one end of a slit, and
   a disk with a seam and a fold. The first plan, a regular grid stored
   folded, is not a mesh: its middle column line would be two different
   lines in the world.
7. **Degenerate.** Collapsed corners and triangles (no coverage, a note);
   bow-ties (construction error); non-convex cells where the equal-area
   line misses the kernel (a note, bounded error).
8. **Exact footprint.** For a destination pixel inside the source, the
   sum over source cells of weight times world cell area equals the
   pixel's outline area in the source CRS. Index-measure weights fail
   this by about 1% of a cell per boundary cell.

GDAL parity tests for `Bilinear` stay as they are.

## 8. Stages

| stage | content | oracle |
|---|---|---|
| 1 | `QuadMesh` (constructors, validation, `CellMap`, topology `Bounded`), `QuadIndex` upgrades (hints, `locate_all`, boundary index, CSR) | brute-force point in quad; existing geoloc parity tests unchanged |
| 2 | the walk and equal-area map; `burn_curvilinear` and a small-ring path through `CoverageWindow`; rectilinear front end alongside | generators 1 to 4 and 7 |
| 3 | exact footprint: equal-area walk for geoloc sources (built, #36); antimeridian unwrap and copies, ring-fold closure (built, #36); geoloc sources across the antimeridian (3b, built, #72) | generator 8; the antimeridian cases from #11's comparison |
| 4 | walls that meet in the world (seam, fold) for the polygon front end (4a, built); CF bounds and SCRIP corners, halo layout (duplicates dropped, a T pivot's row halved) and an overlap check for copies that keep orientation (N3) (4b, built); the ESMF comparison on a tripolar grid (4c, built); a mesh as a warp source, `MeshTransformer` (4d, built); a declared period for coordinates that jump at the seam, which holds the southern rows of a global ORCA grid (4e, built); a mesh as a warp target, `MeshTarget` (4f, built); a whole ocean grid as a `MeshSet` of pieces in two polar LAEAs, damaged land left out (4g, built); polygons in longitude and latitude burned onto a set, densified by bigcurve, on several threads with projection there too, the filter by chunk boxes (4h, built); a set as a warp source, `SetTransformer`, outlines walked onto pieces in several worlds (4i, built) | the world clip on an annulus, a fold and a tripolar disk (built); ORCA-like disks with their halos, both pivots: refused without the layout, the world clip with it (built), and eORCA1 and ORCA2 from CMIP6 (scratch); ESMF on an analytic tripolar cap, Cartesian to rounding and spherical recorded (built); a mesh source's weights against warp(), tiles, conservation (built); generator 5, a periodic mesh source across its seam and round a pole, eORCA1 and ORCA2 in longitude and latitude (built; ORCA in scratch); onto a mesh: the transpose in one plane, mesh to mesh, a lon/lat source across its seam and over the pole, ESMF onto the cap (built); an analytic global grid with damaged land, eORCA1 and ORCA2 whole (built; files in scratch); boxes on that grid against closed-form slivers, and on eORCA1 and ORCA2 against `weights_onto` (built; files in scratch) |
| 5 | adaptive footprint edges, `Footprint::Adaptive` and `WeightOptions::refine_tolerance` (5a, built); polygons in another CRS onto a `QuadMesh` or `Rectilinear`, `burn_from` (5b, built) | the error follows the tolerance, the points grow as its inverse square root, a fine tolerance is a fine densify; affine adds nothing; tiles, threads, `warp()` = `apply`; the seam, the pole, a mesh source, a set (5a, built); the cap's area against the closed form at the tolerance's rate, the points as its inverse square root, a fine tolerance = densify 4096; identity = `burn`; the far pole; 180; points; a period; `Rectilinear` = `QuadMesh` (5b, built) |
| later | overlapping sheets (refused, N3; one mesh per scan meanwhile), swath gaps (mesh holes), approx (cell-centre) mode | |

## Decisions for the maintainer

Decided since: 1 (N1, 2026-11-16, `EqualArea` for coverage), 2 (N2,
2026-11-19, keep the split; its fallback a debug panic, searched for by a
property test), 3 (N3, 2026-11-17, refuse), 4 (N4, 2026-11-17, public;
may break at a 0.x minor release), 5 (N5, 2026-11-19, plain `f64` with
snapping first), 6 (N6, controlledburn 0.3.0, `f32`). See `decisions.md`.

1. **`EqualArea` as the coverage default.** Recommended for coverage and
   area methods, with `Bilinear` kept for point kernels and
   `geo_to_pixel`. Or keep `Bilinear` everywhere unless asked.
2. **Non-convex cells** where the equal-area line misses the kernel:
   note and accept a non-exact cell (recommended), or refuse the grid.
3. **Folded grids other than tripolar**: refuse for coverage in the first
   version (recommended), or design for overlapping sheets now.
4. **Public surface**: `QuadMesh` and the front-end functions in rwarp's
   public API now, or crate-private until a second consumer (wxylib's
   cover module) asks.
5. **Robust predicates**: plain `f64` with snapping first (recommended),
   or adaptive `orient2d` from the start.
6. **f64 edge fractions** in controlledburn, which this front end makes
   more visible: a controlledburn decision, flagged here because it
   changes this front end's output type.
