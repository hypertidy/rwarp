# Conservative regridding: rwarp's pitch at ESMF's core

Status: design, 2026-10-28. Builds on wxylib design m20 (section 4.1, "a
conservative regridder, with GDAL parity as its floor"), m19 (the exact
footprint), and [curvilinear-frontend.md](curvilinear-frontend.md). Based on
rwarp 0.3.0.

## Summary

- **rwarp already does first-order conservative remapping.**
  `Footprint::Exact` measures each destination cell's overlap with each source
  cell exactly, in source index space, and averages by area. That is ESMF's
  `conserve` method in what it computes. What's missing is around it: the
  weights aren't exposed, the measure isn't spherical, there are no mesh
  sources or targets, and there's no ESMF oracle.
- **The aim is the part where geospatial work and model output meet,** not
  all of ESMF: conservative weights between regular, rotated, rectilinear,
  curvilinear and mesh grids, available as a table, verified against ESMF,
  light, and wasm-capable. Nothing like that exists in Rust or the browser.
- **Six stages,** each a PR with its own oracle: weights as values; an ESMF
  oracle suite; a spherical measure; curvilinear and periodic grids (per the
  curvilinear note); meshes; second order.
- **Two observations make the hard parts tractable.** Spherical area for
  lon/lat-straight edges is a change of integrand in the Green's-theorem code
  rwarp already uses. And mesh-to-grid weights are a controlledburn burn of the
  mesh's faces.

## 1. What ESMF provides, and what we take

ESMF (and xESMF, its Python face) regrids between logically rectangular grids,
meshes and location streams, in parallel, with several methods. Climate and
ocean modelling, CMIP post-processing, and model-observation comparison lean
on a small core of it:

| ESMF | Use | Take? |
|---|---|---|
| `conserve` (first order) | fluxes, precipitation, anything whose integral must be kept | **yes, the centre of this note** |
| `conserve_2nd` | smoother conservative fields | **yes** (stage 6; 6a built, section 8) |
| Weight files (SCRIP/ESMF netCDF: `col`, `row`, `S`, plus areas and fractions) | compute once, apply anywhere: CDO, NCO `ncremap`, xESMF | **yes, as values here; files in wxylib** |
| Normalisation `destarea` / `fracarea`, masks | partial coverage and land/sea masks | **yes** |
| Great-circle cell edges, spherical areas | the definition of "conserve" on the globe | **yes, as an option** (stage 3) |
| Meshes (UGRID, MPAS, ICON, ESMF mesh) | unstructured models | **yes** (stage 5) |
| `bilinear`, `patch`, `nearest_stod`, `nearest_dtos` | interpolation | have GDAL-style equivalents; spherical bilinear later, if asked |
| Extrapolation (creep fill, IDW) | filling unmapped cells | no, a separate concern |
| MPI parallel weight generation | very large grids | no: weights are computed once, and rwarp parallelises by chunk |
| Vertical and time regridding | | no |

## 2. Where rwarp stands

`Footprint::Exact` (rwarp #11, m19) builds the destination pixel lattice for
each row band, transforms it to source pixel space through the transformer as
scanlines, and measures each destination pixel's outline against the source
pixels with `controlledburn::CoverageWindow`. So:

- **Cell definition:** destination cells have edges straight in destination
  index space, densified by `densify` points per edge, mapped exactly by the
  transformer. Source cells are unit squares in source index space.
- **Measure:** planar area in source index space. For a source whose
  index-to-world map is affine (any regular grid in its own CRS), that is world
  area in the source CRS, up to a constant per grid.
- **Shared edges:** neighbouring destination pixels use the same transformed
  points for their shared edge, so their outlines tile without gaps or
  overlaps (to the densified edge). This is what makes conservation hold
  exactly (section 3.3).
- **Reduction:** the weights go straight into the reducer (mean over valid
  pixels, renormalised) and are dropped. Coverage within 1e-6 of 1 is taken as
  1, coverage at or below 1e-6 is dropped, and the rest is rounded to `f32`, to
  match what rwarp 0.2 got from `controlledburn::burn`.
- **Gaps:** the antimeridian (a straddling pixel's ring folds), no mesh
  sources, and destination grids limited to what a transformer can describe.
  Curvilinear destinations work through `GenImgGeolocTransformer`, but the
  cell definition is bilinear. The curvilinear note makes the equal-area
  definition exact.

## 3. Stage 1: weights as values

### 3.1 API

```rust
/// Overlap of each destination cell with each source cell, from the exact
/// footprint.
#[non_exhaustive]
pub struct Weights {
    /// Full-grid indices (offsets applied), one entry per overlapping pair.
    pub dst_row: Vec<u32>,
    pub dst_col: Vec<u32>,
    pub src_row: Vec<u32>,
    pub src_col: Vec<u32>,
    /// Area of the overlap, as a fraction of the source cell (source index
    /// space, where a source cell has area 1). Unrounded f64.
    pub frac: Vec<f64>,
    /// Per destination cell, in destination row-major order over the window
    /// computed: the area of its outline in source index space, inside the
    /// source buffer or not, from controlledburn's `ring_area`. NaN where the
    /// outline failed to transform.
    pub dst_area: Vec<f64>,
}

pub fn exact_weights(
    transformer: &impl Transformer,
    src: &SourceGrid,          // ncol, nrow, col_off, row_off: no data needed
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WeightOptions,      // densify, dst offset, drop tolerance; #[non_exhaustive]
) -> Weights;

impl Weights {
    /// ESMF's S for each entry under a normalisation.
    pub fn normalised(&self, norm: Normalization) -> Vec<f64>;
    /// Apply to one band: dst = sum_j S_ij * src_j, skipping source nodata
    /// and renormalising (FracArea) or not (DestArea).
    pub fn apply<S: Sample, D: Sample>(&self, src: &SourceBuffer<S>, ...) -> Vec<D>;
}

#[non_exhaustive]
pub enum Normalization { FracArea, DestArea }
```

- **No I/O.** Writing ESMF/SCRIP netCDF, xESMF's format or Parquet belongs to
  wxylib (m20, "no plan formats or I/O below wxylib"). The columns map
  directly: ESMF's 1-based `row`/`col` are `dst_row * dst_ncol_full + dst_col +
  1` and the source equivalent. `S` is `normalised(..)`.
- **`frac` relative to the source cell** keeps the unit independent of the
  destination, so `dst_area` gives both normalisations: `DestArea` S =
  `frac / dst_area`, `FracArea` S = `frac / sum(frac over the destination
  cell)`.
- **Tiling:** with `dst_col_off`/`dst_row_off`, as for `warp()`. Weights of
  separate tiles concatenate to the weights of the whole, because the lattice
  for a tile is the same lattice.

### 3.2 Same numbers as `warp()`

`warp()` with `Footprint::Exact` and `ResampleAlg::Average` becomes, in effect,
`exact_weights` then `apply(FracArea)`. Two details keep them the same:

- `warp()` keeps its 1e-6 snap and `f32` rounding for compatibility with
  0.2/0.3 output. `exact_weights` returns unrounded `f64`, and its drop
  tolerance is an option (default 0, so nothing is dropped).
- A test compares `apply` with the 0.2-style rounding on against `warp()`,
  bit for bit, on the existing exact-footprint fixtures.

### 3.3 Conservation without an oracle

Because neighbouring destination outlines share their transformed edge
points, the destination cells tile source index space. So for every source
cell that destination cells cover completely, `sum over i of frac_ij = 1`,
up to f64 rounding in the Green's-theorem sums. This is the first test, and
it needs no external reference:

- On regular-to-rotated, regular-to-polar-stereographic and lon/lat-to-LAEA
  cases, check source-cell column sums are 1 to about 1e-12, and that the
  global integral `sum_j v_j` equals `sum_i dst_value_i * dst_area_i`
  (`DestArea`) to the same order.
- Where the source cell is cut by the edge of the destination grid, the sum is
  the covered fraction. The test checks that it matches the cell's area inside
  the destination outline's union.

ESMF reports the same kind of check (`frac_b`, `area_b`), so this also
prepares the oracle comparison.

### 3.4 Memory

The number of entries is about `dst_cells x (source cells per destination
cell + perimeter overlap)`. A 0.25-degree global source onto a 3031 grid at
25 km is about 1.6 M destination cells with about 3 entries each, around
5 M entries of 24 bytes (four `u32` and an `f64`). A `for_each`-style callback
variant (`exact_weights_with(..., |entry| ...)`) streams them for larger
problems.

## 4. Stage 2: an ESMF oracle suite

The same pattern as the gdalwarp fixtures: a committed generator, run once in
a container with ESMPy (or `ESMF_RegridWeightGen`), with weight files reduced
to compact text fixtures, so `cargo test` needs no ESMF.

- **Cases:** regular lon/lat to regular (coarser, finer, offset); lon/lat to
  rotated pole; lon/lat to EPSG:3031 at 25 km; rectilinear (Gaussian
  latitudes) to regular; a curvilinear ocean grid with corner bounds to
  lon/lat (built in stage 4 as a whole analytic tripolar cap, to lon/lat and
  to its own plane, and from both onto the cap: `tests/esmf_tripolar.rs`,
  curvilinear-frontend.md section 3); a mesh (an ICON or MPAS fragment) to lon/lat
  (after stage 5).
- **What's compared:** weights per pair, `frac_b`, `area_b`, and the regridded
  field for a smooth analytic function and for a constant (which must stay
  exact).
- **The expected differences are part of the oracle.** ESMF's cells have
  great-circle edges and spherical areas. rwarp's edges are straight in
  destination index space and its areas are planar in source index space
  until stage 3. The suite records the size of that difference, per case, as a
  number to shrink, not a failure. wxylib's design docs measured up to 0.75%
  of a cell between corner-polygon and geolocation-interpolated outlines
  against ESMF. That's the scale to expect.
- **Environment:** neither ESMPy nor xESMF is installed on the development
  machine. NCO's `ncremap` is, and it can call ESMF's
  `ESMF_RegridWeightGen` or TempestRemap when they're present. A
  `ghcr.io/hypertidy` image with ESMPy is the simplest route.
  [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap) is a
  second, independent oracle with high-order conservative methods.

## 5. Stage 3: a spherical measure

ESMF's conservative weights measure areas on the sphere, with great-circle
cell edges. Two steps get there.

### 5.1 Spherical area for lon/lat-straight edges: a change of integrand

For a region R in longitude λ and latitude φ (radians), the area on the unit
sphere is `∬_R cos φ dλ dφ`. By Green's theorem (λ as x, φ as y) this is
`-∮ sin φ dλ` around R's boundary taken counter-clockwise, so each edge
contributes a closed form:

- For an edge from (λ0, φ0) to (λ1, φ1), straight in (λ, φ),
  `∫ sin φ dλ = (λ1 - λ0) (cos φ0 - cos φ1) / (φ1 - φ0)`, or
  `(λ1 - λ0) sin φ0` when φ1 = φ0. The edge contributes the negative of that.
  A unit check: a cell [λ0, λ1] x [φ0, φ1] gives
  `(λ1 - λ0)(sin φ1 - sin φ0)`, the familiar spherical cell area.
- `CoverageWindow` and `RowPrefix` already sum per-edge Green's-theorem terms
  with integrand y. With a source on a regular lon/lat grid, source index
  space is (λ, φ) up to an affine map, so a `SphericalLonLat` measure is a
  different per-edge term in the same accumulation. The cell partition, the
  clipping and the shared-edge argument (section 3.3) stay as they are.
- This is exact for edges that are straight in lon/lat (rhumb lines on the
  lon/lat plane), which is how many model grids define their cells. It's also
  the right measure for a regular lon/lat source regardless of the
  destination.

This needs a measure parameter in controlledburn's window
(`CoverageWindow::with_measure`, a planar default), so controlledburn stays
dependency-free and index-space-only (m20 section 2.1). Its input is still
rings in index space, plus the affine map from index to (λ, φ). The C++ port
is the same per-edge formula.

**Status (2026-10-29): done.** controlledburn's `Measure::CosLatitude` takes
the other form of Green's theorem, `area = ∮ x w(y) dy` with `w = cos φ(y)`,
which keeps `CoverageWindow`'s row accumulation unchanged: each piece books
the closed-form integrals of `w dy` and `(x - k) w dy` (x is linear in y
along a piece), and coverage is divided by the cell's measure. rwarp's
`WeightOptions::with_measure(lonlat_measure(src_gt))` uses it. Results:

- Exact: regular lon/lat to regular lon/lat weights equal the closed-form
  box overlaps on the sphere to 1e-14 (`tests/weights.rs`).
- Against ESMF (`tests/esmf.rs`), the remaining differences are the edge
  definition, not the measure. ESMF joins a cell's corners with great
  circles, which bow poleward of the parallels: 0.007 degrees over a 2.5
  degree cell at 40 S. So regular_coarser and regular_finer stay at about
  1e-3 and 5e-3 in S (the planar numbers were 7e-4 and 6e-3, partly
  compensating), regular_overhang drops from 6e-4 to 8e-7 (the bows of
  equal cells cancel), and the projected cases at densify 8 drop from
  1.2e-3 to 2.0e-4 (rotated pole) and 1.6e-3 to 5.2e-4 (EPSG:3031).
- Which edges are right depends on the grid: CF lon/lat cells are bounded
  by parallels, which is what rwarp now measures exactly; ESMF's great
  circles are a different cell. 5.2 adds great circles as an option, not a
  replacement.

### 5.2 Great-circle edges: bigcurve

A great-circle edge is a curve in (λ, φ). bigcurve's great-circle edge model
refines it to a tolerance measured in source cells. Then 5.1 measures the
refined ring. The error is bounded by bigcurve's tolerance and converges
quadratically. That matches ESMF's cell definition to a stated tolerance,
without spherical polygon clipping. Exact spherical clipping (S2-style, or
SCRIP's line integrals) remains a possible later engine if the tolerance
route isn't enough. The oracle suite would show that.

This depends on bigcurve stage 2 (the edge-model trait and a batched forward
map, m20 section 6).

**Status (2026-10-29): parked, because it would not match ESMF.** ESMF's
cells are great-circle quadrilaterals on *both* sides: its `area_b` equals the
spherical excess of the quadrilateral through the cell corners to 4e-12, and
that differs from the same cell bounded by parallels by up to 1.4e-4 (2.5 x 2
degree cells at 40 S; `tests/esmf.rs`,
`esmf_cells_are_great_circle_quadrilaterals`). The tops and bottoms of the
*source* lon/lat cells are great circles in ESMF too, and rwarp's source cells
are unit squares in source pixel space, which is what makes the coverage window
exact. An experiment that moved every densified destination edge point onto
its great circle (a wrapping transformer, densify up to 128) measured how much
of the ESMF gap that closes, in max |S| difference:

| case | sphere, straight edges | + great-circle destination edges |
|---|---|---|
| regular_coarser (1 to 2.5 x 2 degrees) | 1.1e-3 | 1.5e-4 |
| regular_finer (2 to 0.75 degrees) | 5.0e-3 | 5.5e-3 |
| regular_overhang (1 to 1 degree) | 7.9e-7 | 3.9e-4 |
| rotated_pole | 2.0e-4 (densify 8) | 7.0e-5 |
| lonlat_3031 | 5.2e-4 (densify 8) | 4.8e-4 |
| lonlat_laea | 1.9e-4 (densify 1) | 5.3e-4 |

Great-circle destination edges help only where destination cells are much
wider than source cells. Elsewhere the source cells' edges dominate, and
bending one side alone breaks the cancellation between equal bows. Matching
ESMF exactly needs curved source cells: spherical polygon overlay, a separate
engine from controlledburn's.

For raster data on a regular lon/lat grid the cells are bounded by parallels
(CF), so rwarp's definition is the data's, and it is exact
(`spherical_measure_gives_exact_lonlat_box_overlaps`, 1e-14). TempestRemap
cannot check that independently: version 2.2 builds lon/lat meshes with
great-circle edges (`#define ONLY_GREAT_CIRCLES` in GenerateRLLMesh.cpp) and
computes face areas by quadrature over great-circle triangles whatever the
edge type. It serves instead as a second implementation of ESMF's cells.

Great-circle edges stay worth having for grids defined that way: unstructured
meshes (stage 5) and curvilinear grids whose corners are joined by great
circles (stage 4). They come with those stages, through bigcurve-rs, whose
stage 1 (the great-circle engine, `Forward`, shared-edge identity) is done.

## 6. Stage 4: curvilinear and periodic grids

As [curvilinear-frontend.md](curvilinear-frontend.md) designs it. In
particular:

- the `CellMap::EqualArea` definition, which makes index-space fractions exact
  world-area fractions for quadrilateral cells (convex, and the non-convex
  cases the note accepts);
- `Topology::{PeriodicCols, Tripolar}`, the antimeridian (unwrap and add the
  ring at offsets −P, 0, +P) and poles. Built without a topology: walls
  that meet in the world are crossed with nothing declared (4a), and a
  period declared on the mesh does the rest (4e).

For conservative regridding, the curvilinear destination's cells come from
corner bounds (CF `bounds`, SCRIP corners), as ESMF defines them, not from
interpolated centres. The note's cell-definition parameter covers this.

Status (2026-11-17): a curvilinear *source* given by its corners is
built. `QuadMesh::from_cf_bounds` reads CF bounds and SCRIP corners
(4b), and `mesh::MeshTransformer` makes the mesh a source for
`exact_weights` and `warp()` (4d). Against ESMF the weights agree to
rounding in the mesh's plane, with the spherical differences recorded
(`tests/esmf_tripolar.rs`, 4c). A mesh in longitude declares its period
(`QuadMesh::with_period`, 4e): outlines across its seam and round a pole
are measured as for geolocation arrays, and the southern rows of an ORCA
grid are a mesh in longitude and latitude. A mesh is the destination
through `mesh::MeshTarget` (4f): each cell is a destination pixel
measured in the source's space, and against ESMF onto the tripolar cap
the weights agree to rounding in the plane, with the spherical
differences recorded (`onto_tripolar` fixtures). A whole ocean grid is a
`mesh::MeshSet` (4g): pieces in north-polar and south-polar LAEA, with
damaged land left out, weights from and onto it in the file's numbering
(eORCA1 and ORCA2 from CMIP6 whole, every ocean cell kept).

## 7. Stage 5: meshes

A mesh source or destination is a list of polygons (faces) with shared edges:
ICON triangles, MPAS hexagons, a UGRID mesh, an ESMF mesh file.

- **Mesh to grid:** burn every face into the destination grid with
  `controlledburn::burn` (coverage mode) after mapping faces to destination
  index space (the transformer's inverse, densified by bigcurve where edges
  are great circles). The output, (face, row, col, fraction of the cell), is
  the weight table transposed. controlledburn's sparse output and speed are
  the point here.
- **Grid to mesh:** the same table, normalised the other way (by face area).
- **Mesh to mesh:** a later step, needing polygon-polygon intersection, which
  is out of controlledburn's scope.
- **Shared-edge exactness** carries over: faces that share an edge share its
  vertices, so the per-cell sums telescope.

**Status (2026-11-18): 5a built**, `mesh::PolyMesh` (`src/mesh/poly.rs`),
mesh to grid and grid to mesh with edges straight in the mesh's
coordinates:

- **The mesh.** Nodes and faces as compressed rows (`PolyMesh::new`, ESMF's
  `elementConn`) or a padded table (`from_padded`: UGRID's
  `face_node_connectivity`, MPAS's `verticesOnCell`). With a period
  (`with_period`, 360 for longitude) each edge goes the shorter way round,
  and a face round a pole is closed along it by the grid.
- **The engine is the exact footprint's, not `controlledburn::burn`.** Each
  face is an outline, measured as `exact_weights` measures a destination
  pixel: its edges cut into `densify` pieces in the mesh's coordinates (or
  refined by bigcurve to `refine_tolerance`), through the transformer into
  the grid's pixels, and covered there by `cover_outline`, the part of
  `Lattice::cover` that measures an outline in source pixels (factored out;
  warp and weights hashes unchanged). That gives `f64` overlaps, the grid's
  measure (`Measure::CosLatitude` for areas on the sphere), the seam and
  the poles. `burn`'s fractions are `f32` and planar; its runs would pay
  for faces many cells across, where the window over a face's box costs
  its area.
- **One table, both ways.** `weights_onto` is `exact_weights`' table with
  face `f` as destination cell `(0, f)`. `weights_to` is the same overlaps
  turned round: grouped by grid cell, `frac` over the face's measure
  (`src_cell_area`), `dst_area` the cells'.
- **Shared edges.** A face's interior edge points are made from the edge's
  lexicographically smaller end, the other aligned to it by the period,
  so a shared edge gets bitwise the same points from both faces. bigcurve
  refines in the same canonical direction; its edge model here is planar
  with `align` and `reframe` by the period, so a shared edge across the
  seam refines to the same points too, and the transformer sees the same
  inputs (it must take each point on its own: not `ApproxTransformer`).
- **Measured** (`tests/poly_mesh.rs`, `tests/esmf_mesh.rs`):
  - ESMF, Cartesian, hexagons and triangles over part of a grid and past
    its corner: `S` within 1.6e-14 (mesh to grid) and 5.7e-14 (grid to
    mesh), `frac_b` and `area_b` to 6e-14.
  - Boxes in longitude and latitude on a 1 degree grid under the
    spherical measure: every overlap is the closed form to 1e-16 sr.
    Across 180 and round both poles (a cap of 36 nodes on 80 N, three
    nodes on 60 S), each face's area is its closed form to 1e-14 of it,
    and the grid stored from 0 to 360 gives the same weights.
  - Boxes in longitude and latitude onto a polar LAEA grid (100 km),
    refined: each box's area is within 1.35 to 1.58 times the tolerance of
    its area on the sphere from 1e-1 to 1e-6 pixels, the points growing as
    the tolerance to the power -0.493 (from 1e-3), and the faces tile the
    grid (every whole cell's `frac` sums to 1 to 1e-13) at every
    tolerance, across the mesh's seam too.
  - The mesh of a grid's own cells is the identity; triangles halve their
    cells; a mixed mesh of quads, triangles and hexagons tiles the grid,
    and `apply` keeps the integral both ways.

**Status (2026-11-18): 5b built**, `PolyMesh::with_great_circles`: edges
on great circles between nodes in longitude and latitude, as ICON's and
MPAS's are.

- **Points.** With `densify`, each edge is cut at equal angles (slerp)
  from its canonical end, so shared edges stay bitwise. A meridian (an
  arc whose ends share a longitude, or one from a pole) keeps its
  longitude exactly, with latitude in proportion. Refinement uses
  bigcurve's `GreatCircle` model (its own `align` and `reframe` by 360).
- **The equator.** In longitude and latitude a great circle across the
  equator is an S, its curve turning there, and refinement's test (the
  moved midpoint against the chord) can't see a turn at a piece's middle:
  a 20 degree edge crossing at its middle was never refined, its overlaps
  5e-3 out at a tolerance of 1e-6. densify 2 only moves the blind spot
  to an edge crossing a quarter of the way. So each edge across the
  equator is cut there first, at the point of its great circle on it
  (from the canonical ends, so shared). Each piece then curves one way,
  and the same edges are within 5.5e-7 of densify 2048. For a grid in
  another projection the turns are elsewhere, and the blind spot is
  bigcurve's own (its forced split of arcs over 30 degrees covers long
  ones).
- **Poles.** A node on a pole (ICON's icosahedron has one on each) is two
  points on it, at the longitudes of its neighbours, so in longitude and
  latitude the face runs along the pole between its two meridians, which
  is its shape there, whatever longitude the node is written with. A face
  round a pole (MPAS's pentagons) goes once round and is closed along it
  by the grid, as before.
- **Measured** (`tests/poly_mesh_sphere.rs`, meshes made in the test: an
  icosahedron with nodes on the poles cut into 320 triangles, and its
  dual of 162 hexagons and pentagons, two round the poles):
  - Triangles from the equator to a pole, bounded by two meridians and
    the equator, are lon/lat boxes: every overlap is a whole cell, at any
    densify, refined or not, wherever the pole's node is written.
  - Onto a global 2 degree grid, refined, each geodesic face is within
    6.0 to 7.6 tolerances of its spherical excess from 1e-1 to 1e-5
    pixels, the points growing as the tolerance to the power -0.488; at
    densify 8, 32 and 128 the error falls as densify to the power -2.00.
    The dual's faces, the pole pentagons included, converge alike (6 to
    12 tolerances). The faces tile the grid at every tolerance (every
    cell's `frac` sums to 1 to 2e-14), and their areas sum to 4 pi to
    1e-14 even when each is a tolerance out, since neighbours share
    their edges' errors.
  - The same faces with each great circle cut into 2048 chords by hand,
    as a planar mesh, agree to 1.5e-7.
  - ESMF, spherical (`tests/esmf_mesh.rs`, both meshes both ways with a
    global 5 degree grid): the faces' areas agree to 8e-8 at a tolerance
    of 1e-7 pixels (5e-9 at 1e-8: both are the spherical excess), and
    `frac_b` to 1e-13. The weights differ by the grid's cells, which
    rwarp bounds by parallels and ESMF by great circles, 1.3e-3 apart in
    area at 5 degrees: `S` to 6.9e-4 onto the mesh and 3.8e-3 from it,
    recorded as ceilings.

**Status (2026-11-18): 5c built**, a grid with world cells on the other
side: geolocation arrays (`GenImgGeolocTransformer`), a `QuadMesh`
(`MeshTransformer`), a `MeshSet` (`SetTransformer`), each with destination
geotransform `WORLD_GT`.

- **The engine is the exact footprint's again.** The lattice's measure
  against world cells is factored out as `warp::WorldCells` (the sheets,
  their areas, each world's turn, the wrap): `cover_one` for one sheet
  (the outline walked onto the mesh at each turn of a period or
  `lon_wrap`), and `begin`, `offer` once a world, `finish` for several.
  The lattice gives it its outlines as before; warp and weights hashes
  are unchanged, sets and pieces included.
- **Faces in each world.** A face's points go to each of the source's
  worlds (`dst_to_world_in`), refined there with the tolerance in world
  units. The face's area is its planar area there (with several worlds,
  in the one whose cells it covers most), `frac` a fraction of a cell's
  world area, and the measure must be planar.
- **Inside out.** In a set's polar projection a face near the far pole
  comes out inside out round it, and its box covers the world. A pixel's
  outline has one turn, so the lattice compares its turn in each world
  with the world's. A face may be written either way round, so `offer`
  takes the face's own turn too: the sign of its area in the mesh's
  coordinates, unwrapped by the period and closed along a pole if it
  goes round one (so a cap round the south pole has a turn, and in the
  north world is left out). Each world's turn is voted from small
  triangles at the faces' first nodes, as the lattice's is.
- **Measured** (`tests/poly_mesh_cells.rs`):
  - Quads, triangles and hexagons onto a curvilinear `QuadMesh` in the
    same plane: every fraction is the face's overlap with the cell,
    clipped independently (Sutherland and Hodgman), over its area, to
    1.1e-13, at densify 1 and refined. The cells' own burn of the faces
    agrees to `f32`, but for two slivers of 3e-7 where three faces meet
    at a cell's corner, which it leaves out.
  - A lon/lat `QuadMesh` with a period, and geolocation arrays with
    `lon_wrap` stored from -180, give the fractions of the same cells as a
    regular grid, to rounding (1e-12), across 180 and with a cap round
    the pole.
  - A set in a south-polar and a north-polar LAEA, under boxes all round
    written either way, and caps round both poles: every cell of the set
    is covered once over to 1e-12, the caps reach none, and the fractions
    are the set's burn of the boxes to 8.2e-7 (its `f32` and slivers).

**Status (2026-11-19): threads built**, `PolyMesh::weights_onto_par` and
`weights_to_par`, the serial table bit for bit:

- A face's results don't depend on the faces measured with it: its
  points are transformed point by point, a segment's refinement depends
  only on its ends, and the cells, the worlds' turns and the window are
  the same on every thread. So the faces are cut into about four ranges a
  thread, taken in turn (`mesh::set::in_parallel_with`), each thread with
  a transformer of its own and its own copy of the cells (`Measurer`),
  and the ranges are joined in order.
- `weights_to` groups its entries by grid cell with a counting sort,
  which keeps faces in order within a cell as the stable sort did.
- Checked bit for bit against serial at 1 to 7 threads, both ways, on
  every kind of grid (pixels, longitude and latitude across the seam and
  round the poles, refined LAEA, great circles, a curvilinear mesh,
  geolocation arrays, a set in two worlds, faces that fail); the serial
  tables themselves hash as before on 16 tables. A geodesic mesh of
  20480 great-circle triangles onto a global 0.25 degree grid, refined to
  1e-3 pixels: 0.52 s serial, 0.12 s on 8 threads (4.2x); to the grid
  0.63 s and 0.25 s (2.6x, the grouping by cell serial).

Mesh to mesh stays out of scope.

## 8. Stage 6: second order

ESMF's `conserve_2nd` adds a per-source-cell gradient term, with weights that
also depend on the overlap's centroid. Centroids are Green's-theorem integrals
too (`∮ x² dy`, `∮ y² dx`), so the same window machinery returns them. Gradient
estimation needs neighbouring source cells. It's a reconstruction step on top
of stage 1's weights, and doesn't change the coverage engine.

**Decided (2026-11-19, the maintainer, each at the recommendation):**

- **The moments live in controlledburn.** `CoverageWindow::set_moments`
  books each cell's first moments beside its coverage, planar and
  `CosLatitude` (controlledburn-rs #51, released in 0.3.3).
- **ESMF's centre for each source cell:** the centroid of the part of the
  cell the table covers, not the whole cell. It gives ESMF parity, and
  the mapped part is conserved exactly. Where a grid's edge or a mask cuts
  a source cell, the plane is centred off the cell's middle, which is less
  accurate there.
- **The table keeps the moments and `apply` takes the gradients from the
  field** (SCRIP's second and third weights, in effect). So a neighbour
  without data drops out per field, as nodata does in first order.
  `Weights::second_order` folds them into ESMF's single matrix, given the
  cells with data, for weight files.

**Decided next (2026-11-19, the maintainer, each at the recommendation):**

- **World-cell grids (geolocation, `MeshTransformer`, `MeshSet`): planes
  in the cells' index space** under the walk's equal-area map, where the
  overlaps are measured. The moments come from the same window, and the
  neighbours are the eight round a cell in the full grid, so a set's
  pieces, halos and the seam need nothing new. It conserves exactly, and
  it is exact for fields linear in the indices, not for fields linear in
  world coordinates (the map is affine only on each quarter of a cell);
  on a smooth grid it is second order. The other option, world
  coordinates as ESMF, needed a second pass clipping each overlap in
  world coordinates and neighbours carried between a set's worlds.
- **A `PolyMesh`'s faces as sources: planes in the grid's space**, where
  their overlaps are measured, neighbours the faces sharing a node
  (ESMF's), centres from the table. Apply and fold take the mesh.
- **A limiter, opt-in:** Barth-Jespersen at apply time, off by default
  (ESMF has none).

Built in three steps: 6b the world-cell grids and grids onto meshes, 6c
the faces as sources, 6d the limiter and spherical ESMF fixtures.

**Status (2026-11-20): 6a to 6d built** (`src/weights/second_order.rs`).

- **What's built:** `WeightOptions::with_moments`, `Weights::moment`,
  `Weights::apply_second_order` and `Weights::second_order`, for sources
  measured in their pixels: regular grids, planar or under
  `lonlat_measure`, across the seam and on threads.
- **The method is ESMF's.** The gradient is Green's theorem round the
  centres of the cell's eight neighbours (fewer where some have no data),
  sorted anticlockwise round the covered centre. It is zero where there
  are fewer than three, the centre is outside their polygon by more than
  1e-14, or the polygon has no area.
- **Neighbour centres** are each cell's centroid under the measure: ESMF's
  corner mean for the planar measure, and towards the equator on the
  sphere, so fields linear in longitude and latitude come out exactly
  there too. ESMF's spherical method works in 3D instead, so on the
  sphere the two are different second-order methods.
- **One finding: ESMF centres an overlap on the mean of its polygon's
  vertices** (`_calc_centroid_2D_2D_cart`, ESMCI_SM.C), not its centroid.
  The two agree for rectangles, triangles and parallelograms, and not for
  the general polygons a projected grid cuts from a lon/lat one. So:
  - For a grid onto a grid, rwarp's folded table is ESMF's to 6e-14
    (`regular_coarser`, and `regular_overhang` over the source's corner).
  - Projected (`lonlat_3031`), S differs by up to 0.022. On a field linear
    in source pixels ESMF is off by up to 0.035 there, and rwarp by 1e-14.
  - Given ESMF's vertex means in place of the moments, rwarp's table is
    ESMF's to 1.1e-12 over 26,507 weights. It differs only next to the 60
    destination cells with a corner exactly on a source wall, where a
    vertex mean depends on how the clip lists the vertices on the wall.
- **Checks** (`tests/second_order.rs`):
  - A linear field is exact to 1e-14, planar and on the sphere (first
    order is off by 0.16).
  - On a quadratic, halving the cell size quarters the error exactly. On
    a sine, second order is 57 times closer than first.
  - Under `DestArea` the integral is conserved to 1e-12, with cells cut by
    the edge and without data.
  - The folded table applied by `apply` is `apply_second_order` to 1e-12,
    with and without a mask.
  - Moments survive the seam's pieces, threads and tiles bit for bit.
- **6b, world cells** (`tests/second_order_world.rs`): moments through
  the walk (`WorldCells::cover_one`, and `offer` for a set's sheets and
  worlds) and through `PolyMesh::weights_onto`'s measure;
  `Weights::second_order` takes the window's world cell areas
  (`weights::source_cell_areas`).
  - Each entry's fraction and moments are the overlap's, clipped against
    the cell's four triangles in world coordinates and mapped through the
    split, to 3e-15 (2,083 entries on a curved grid; 2,408 across a
    curved global mesh's seam, where an outline reaches it at both
    ends). There a field linear in rows comes out exact to 4e-15.
  - A field linear in the grid's indices comes out exact to 7e-15 (first
    order 0.19).
  - A world quadratic on the curved grid: second order's error falls
    3.9 times as the cells halve (from 2e-5 to 5e-6), first order's
    2.3 times.
  - Conserved to 1e-12 under `DestArea` with cells without data; the
    folded table applied is `apply_second_order` to 1e-14.
  - The grid as a `MeshSet` in two bands, one turned 30 degrees, gives the
    mesh's table to 3e-12 (through `SetTransformer` and `weights_to`),
    the rounding of the split's areas in absolute coordinates.
  - Onto a `PolyMesh`, a field linear in pixels is exact on the faces to
    5e-14, and faces over a wrapping grid's seam centre its cells on their
    middles; threads keep the moments bit for bit.
  - Without moments every table and warp is as before, bit for bit
    (ORCA2 hashes).
- **6c, a `PolyMesh`'s faces as sources** (`tests/second_order_faces.rs`):
  `PolyMesh::weights_to` records each overlap's moments about its grid
  cell's corner (the source has none), over the face's measure, and
  `PolyMesh::apply_second_order` and `PolyMesh::second_order` take the
  mesh for its adjacency. The apply and fold loops are 6a's, with the
  stencil passed in, so 6a's and 6b's output is bit for bit as before.
  - A face's centre is the centroid of its covered part (from the table),
    and its neighbours are the faces sharing a node, centred the same
    way. A neighbour with no entries has no centre, so isn't one.
  - **The seam.** The grid's period isn't in the table, and a world-cell
    grid's index seam (ORCA's) has none anyway, so the stencil judges
    from the cells. A face whose cells (slivers under 1e-12 of it aside)
    aren't one 8-connected piece is first order: the seam cuts it, and
    its centre is in the middle of the grid. A neighbour whose cells'
    bounding box isn't next to the face's is left out: it is across the
    seam. So at a seam the gradients are one-sided, as a grid's are at
    its edge. Carrying the period would give two-sided gradients for a
    lon/lat grid only; not done.
  - Checks: the moments are `weights_onto`'s, turned round, to 1e-12. A
    field linear in the grid's space is exact on faces off the mesh's
    edge and whole in the window, with their neighbours: in pixels to
    2.5e-14 (327 cells, first order off by 2.3), on the sphere to 2.8e-14
    (1,033 cells), in a curved grid's indices to 1.4e-14 (408 cells).
    On a quadratic, halving mesh and grid divides the error by 3.95 then
    3.90 (first order's by 1.7). Conserved to 1e-15 with faces without
    data; the folded table applied is the same to 1e-12. Either side of
    a global grid's seam (faces sharing nodes on it) a field linear
    across it is exact; in a face the seam cuts, the value is the face's.
    Each guard (connected, next to, slivers) has a test that fails
    without it. Threads keep the moments bit for bit.
- **6d, the limiter** (`tests/second_order_limiter.rs`):
  `ApplyOptions::limiter`, `Limiter::BarthJespersen` (off by default).
  Each source cell's gradient is scaled by the largest factor in [0, 1]
  keeping its plane, at each of its overlaps' centroids (where it is
  used), within the values of the cell and the neighbours its gradient
  came from. Per band, and cells without data left out. Overlaps under
  1e-12 of their cell are rounding (2 to 4% of a table's entries, their
  centroids up to 45 cells off): left out of the factor (otherwise it
  flattens planes for nothing), and given their cell's value (so a
  destination cell with nothing else stays in bounds).
  - Fronts over grid cells (planar, on the sphere, finer and coarser
    destinations), a curved grid's world cells and a mesh's faces
    overshoot their bounds by 0.1 to 7.3 without it, and by at most
    1.3e-15 with it. Conserved to 1e-12 under `DestArea`.
  - A linear field inside the window is as without it to 3.6e-15 (the
    slivers' terms); on the window's edge, where a plane is one-sided,
    it is limited.
  - On a sine (wavelengths 75, 38 and 19 cells) it costs 16, 43 and 72%
    in rms, still 49, 19 and 7 times closer than first order.
  - Each guard (no nodata in the range, the cell's own value in it,
    slivers out of the factor, slivers' values) has a test that fails
    without it. With the limiter off every output is as before, bit for
    bit (38 hashes, with the faces' second order).
- **6d, on the sphere against ESMF** (`tests/esmf_second_order_sphere.rs`,
  fixtures `<case>.sph2.txt`: ESMF's spherical `conserve_2nd` for the
  three second-order grid cases and both sphere meshes both ways). The
  two are different methods, so the comparison is of accuracy, each
  against exact cell means in its own cells (quadrature, checked against
  closed forms to 1e-14), rms of a smooth field:
  - A grid onto a grid, inside: rwarp 2.8e-6, 1.9e-7, 4.5e-6
    (`regular_coarser`, `regular_overhang`, `lonlat_3031`), ESMF 2.6e-6,
    3.0e-7, 8.6e-5, first order 3.0e-4, 2.4e-4, 1.3e-3. Over every cell
    the ring on cut source cells dominates both, no better than first
    order there, and the two agree.
  - A grid onto a mesh: rwarp 1.3e-4 and 9.8e-5, ESMF 1.6e-3 and 4.8e-4
    (ESMF's vertex means), first order 4.0e-3 and 3.2e-3.
  - A mesh onto a grid, faces of 15 degrees onto cells of 5, first order
    7.5e-2 to 9.5e-2: away from the seam and the poles rwarp 9.7e-3 and
    1.6e-2, ESMF 1.7e-2 and 7.1e-2. Within 15 degrees of the seam rwarp
    5.3e-2 and 7.7e-2, near first order, ESMF 1.5e-2 and 6.1e-2: the
    faces the seam cuts are first order, and their neighbours' gradients
    one-sided.
  - The weights differ from ESMF's by more than first order's: by first
    order's for a grid onto a grid but next to the source's edge
    (one-sided gradients, `S` to 0.074), and elsewhere by the method
    (0.02 projected, 0.23 from a mesh).
- **Not built:** the grid's period in a face table, for two-sided
  gradients at a lon/lat grid's seam (and a face the seam cuts with a
  plane): the spherical fixtures show it is where a mesh onto a global
  grid loses most (6c's one-sided seam).

## 9. Interfaces and where things live

Per m20:

- **controlledburn:** the measure parameter on the window and burn (5.1). Still
  rings in index space, no CRSs.
- **bigcurve:** great-circle refinement and the batched forward map (5.2).
- **rwarp:** `exact_weights`, `Weights`, normalisation and `apply`, the
  curvilinear and mesh front ends, and the oracle suite.
- **wxylib:** weight files (ESMF/SCRIP netCDF, xESMF, Parquet), mesh readers,
  CLI, and storing weights as plans.

## 10. Open questions

All four decided 2026-11-19 at the recommendation (C1 to C4,
`decisions.md`): `u32` per axis; `FracArea`, said prominently; the
lattice shared; a destination-side measure waits for a use.

1. **Index types:** `u32` per axis (grids up to 4 billion per side) or one
   flattened `u64` per side as ESMF does? Per-axis is easier to read, and
   flattening is one multiply.
2. **Default normalisation:** `FracArea` matches `warp()` today. xESMF defaults
   to `conserve` (destarea) with `conserve_normed` for fracarea. Recommended:
   `FracArea`, and say so.
3. **Whether `warp()` should be rebuilt on `exact_weights`** internally (one
   code path) or share only the lattice (no change to `warp()`'s performance).
   Recommended: share the lattice and the per-pixel coverage function, and keep
   `warp()`'s direct reduction, which avoids materialising weights.
4. **Spherical measure as a destination-side option** too, for lon/lat
   destinations with projected sources. Only the source side is needed for
   ESMF parity, since `frac` is relative to the source cell.
