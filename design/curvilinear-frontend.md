# Curvilinear grids: the coverage front end

Status: stages 1 and 2 implemented (`src/mesh.rs`, `src/mesh/walk.rs`,
`src/mesh/rect.rs`; 2026-10-31 and 2026-11-01), stage 3 built (#36,
#72), stage 4 started: walls that meet in the world (a seam, a fold) are
walked across (4a, 2026-11-17; section 3), and meshes come from CF bounds
and SCRIP corners with ocean-model halos dropped and overlapping cells
refused (4b, 2026-11-17; section 3). Later stages proposed.
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
and CSR buckets built on first use. Not yet: a declared period for
coordinates that jump at the seam (section 3; there is no `Topology`
enum), the boundary index (it belongs with the walk in stage 2). Decisions 1 to 5 below were taken
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

- **Coordinates that jump at the seam** (longitude, period 360). The two
  walls are then different places in the plane and nothing meets. As for
  the exact footprint (stage 3b), a ring would be walked at each whole
  period at which it meets the mesh, and the pieces burned together. A
  period declared on the mesh is the only remaining use of a topology.
  Generator 5.
- **Tripolar grids in longitude and latitude** can't be planar meshes:
  the geographic pole is inside an ocean cell. An ORCA grid is given as a
  mesh in a polar projection. North polar Lambert azimuthal equal-area
  keeps planar areas true, with edges straight in that projection: not
  great circles, not rhumb lines. The difference from ESMF's great
  circles is second order in cell size, and the ESMF comparison measures
  it. One north-polar projection doesn't hold the whole globe, though
  (4b, below): near 80 S its cells are flattened 500 to 1 or more, and
  the bend of their long sides exceeds their width.
- The ESMF comparison, and the southern rows of a global ORCA grid (a
  second mesh in another projection, or longitude with a declared
  period).

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
  above. Exact to the tolerance.
- **The exact footprint.** Destination pixel edges, straight in
  destination pixel space, into source space: the hybrid in the bigcurve
  follow-up (transform the lattice at densify 1 as scanlines, refine only
  edges failing a sagitta tolerance in source-pixel units).
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
5. **Periodic.** A regular global grid declared periodic, columns rotated
   by k, must match the regular burn shifted by k, for polygons crossing
   the seam.
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
| 4 | walls that meet in the world (seam, fold) for the polygon front end (4a, built); CF bounds and SCRIP corners, halo layout (duplicates dropped, a T pivot's row halved) and an overlap check for copies that keep orientation (N3) (4b, built); a declared period for coordinates that jump at the seam; the southern rows of a global ORCA grid; the ESMF comparison on a tripolar grid | the world clip on an annulus, a fold and a tripolar disk (built); ORCA-like disks with their halos, both pivots: refused without the layout, the world clip with it (built), and eORCA1 and ORCA2 from CMIP6 (scratch); generator 5 |
| 5 | bigcurve feature: reprojected polygons, adaptive footprint edges | convergence at bigcurve's square-root rate; equality with fixed densify at a fine count |
| later | overlapping sheets (refused, N3; one mesh per scan meanwhile), swath gaps (mesh holes), approx (cell-centre) mode | |

## Decisions for the maintainer

Decided since: 1 (N1, 2026-11-16, `EqualArea` for coverage), 3 (N3,
2026-11-17, refuse), 4 (N4, 2026-11-17, public; may break at a 0.x minor
release), 6 (N6, controlledburn 0.3.0, `f32`). See `decisions.md`.

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
