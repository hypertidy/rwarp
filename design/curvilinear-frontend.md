# Curvilinear grids: the coverage front end

Status: proposed. Scope: how rwarp moves polygons (and its own
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

### Topology

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Topology {
    #[default]
    Bounded,
    /// Global in longitude: column `ncol` is column 0 again.
    PeriodicCols { period: f64 },
    /// Periodic columns plus a fold on the last row: cell
    /// `(nrow - 1, i)`'s far side joins `(nrow - 1, ncol - 1 - i)`'s.
    Tripolar { period: f64 },
}
```

- **Periodic.** The walk keeps a local frame: crossing the seam shifts it
  by one period, so the segment and the cell stay continuous. Polygon
  coordinates are taken literally (planar in the coordinates given; with
  a periodic grid, 181 and -179 are the same place, and a segment from
  179 to -179 goes the long way, as GEOS reads it). In index space the
  ring is unwrapped (columns beyond `ncol` allowed), burned on a grid
  widened to cover it, and columns folded back modulo `ncol` after. No
  polygon splitting.
- **Tripolar** (ORCA and friends): the walk crosses the fold like any
  side, using the topology's neighbour. In index space the ring jumps at
  the fold; close each part along the top boundary, as for "Off the
  mesh". Oracle in section 7.
- **Poles.** Cells straight in degrees that touch latitude 90 are valid
  planar quadrilaterals (possibly triangles). A pole *inside* a cell is
  not meaningful in degrees; such grids should come with corners in a
  polar projection.
- **Other folds** (overlapping swath scan lines, MODIS bow-tie): cells
  overlap in world space, so location is ambiguous and a walk follows one
  sheet. Detect at construction (cells whose signed area has the minority
  sign, overlapping neighbours) and refuse for coverage in the first
  version.

### The exact footprint at the antimeridian

`Footprint::Exact` does not handle the antimeridian today. Two cases,
both fixable in rwarp with no change to `CoverageWindow`:

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
  segment first enter".
- **Compact and lazy.** CSR buckets (offsets plus one `u32` array), built
  on first use.
- **Triangles.** For `EqualArea`, the containing triangle and its
  barycentric inverse (closed form, no root).

GDAL's backmap inverse, which wxy-warp carries (`geoloc.rs`), is an
option for parity on folded swaths (m20 rwarp roadmap item 4), separate
from this.

```rust
pub struct QuadMesh { /* corners (nrow + 1) x (ncol + 1), topology, cell map */ }

impl QuadMesh {
    pub fn from_corners(ncol: usize, nrow: usize, x: Vec<f64>, y: Vec<f64>,
                        map: CellMap, topology: Topology) -> Result<Self, String>;
    /// Corners from centres, GDAL's PixelCenter rule.
    pub fn from_centres(ncol: usize, nrow: usize, x: &[f64], y: &[f64],
                        map: CellMap, topology: Topology) -> Result<Self, String>;
    /// CF bounds, nrow x ncol x 4; corners must be shared by neighbours.
    pub fn from_cf_bounds(ncol: usize, nrow: usize, xb: &[f64], yb: &[f64],
                          map: CellMap, topology: Topology) -> Result<Self, String>;
    pub fn cell_area(&self, row: usize, col: usize) -> f64;
    pub fn locate(&self, p: [f64; 2], hint: Option<(usize, usize)>) -> Option<(usize, usize)>;
    pub fn locate_all(&self, p: [f64; 2], out: &mut Vec<(usize, usize)>);
}
```

`GeolocArrays` would hold a `QuadMesh` in place of `QuadIndex`, with
`GeolocOptions` gaining `cell_map` (default `Bilinear`).

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
| geoloc source | small rings onto a curvilinear grid | no: rings reach source index space by inverse bilinear, the index measure |
| both geoloc | quads onto a curvilinear grid | no, as above |

The fix is the same front end: with `cell_map: EqualArea` on the source
`GeolocArrays`, the inverse used for area methods becomes the
equal-area map, outline vertices are inserted at source cell sides and
spokes, and `CoverageWindow` then returns world-area fractions in the
source CRS. That makes the exact footprint first-order conservative
remapping on curvilinear sources, as its docs already claim for affine
ones. Point kernels keep `Bilinear` for GDAL parity.

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
`burn` gave rwarp 0.2. Recommendation: `f64` edges in a controlledburn
0.3.0 if the C++ sibling can follow; otherwise document it.

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
6. **Fold.** A regular 2N by M grid stored folded as N by 2M (the second
   half reversed above the fold) must match the unfolded burn after the
   index remap: tripolar logic, tested without an ocean model.
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
| 3 | exact footprint: `cell_map: EqualArea` for geoloc sources; antimeridian unwrap and copies, ring-fold closure | generator 8; the antimeridian cases from #11's comparison |
| 4 | `PeriodicCols`, `Tripolar` for the polygon front end | generators 5 and 6 |
| 5 | bigcurve feature: reprojected polygons, adaptive footprint edges | convergence at bigcurve's square-root rate; equality with fixed densify at a fine count |
| later | overlapping folds, swath gaps (mesh holes), approx (cell-centre) mode | |

## Decisions for the maintainer

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
