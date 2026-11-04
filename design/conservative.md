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
| `conserve_2nd` | smoother conservative fields | later (stage 6) |
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
  latitudes) to regular; a curvilinear ocean grid (a small tripolar patch with
  corner bounds) to lon/lat; a mesh (an ICON or MPAS fragment) to lon/lat
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
  ring at offsets −P, 0, +P) and poles.

For conservative regridding, the curvilinear destination's cells come from
corner bounds (CF `bounds`, SCRIP corners), as ESMF defines them, not from
interpolated centres. The note's cell-definition parameter covers this.

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

## 8. Stage 6: second order

ESMF's `conserve_2nd` adds a per-source-cell gradient term, with weights that
also depend on the overlap's centroid. Centroids are Green's-theorem integrals
too (`∮ x² dy`, `∮ y² dx`), so the same window machinery returns them. Gradient
estimation needs neighbouring source cells. It's a reconstruction step on top
of stage 1's weights, and doesn't change the coverage engine.

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
