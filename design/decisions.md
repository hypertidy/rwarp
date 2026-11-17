# Open decisions

Status: 2026-11-05; N1 decided 2026-11-16, N3 and N4 2026-11-17. The decisions the roadmap is waiting on, one section
each: what is being decided, the options with their costs, the
recommendation, and what the code does today. Several are already
implemented at their recommendation *provisionally*; for those the
decision is to confirm or to change course. Each has a tracking issue
(label `decision`).

The figures are computed, not sketched: `Rscript design/figures/make-figures.R`
from the repository root regenerates them, and every number printed on a
figure comes from that script.

Sources: [curvilinear-frontend.md](curvilinear-frontend.md) ("Decisions for
the maintainer", decisions N1 to N6 here) and
[conservative.md](conservative.md) (section 10, "Open questions", C1 to C4).
C5 is recorded because it shapes how the work is presented, and is decided.

| | decision | today | recommendation |
|---|---|---|---|
| N1 | cell map default for coverage | coverage uses `EqualArea` on every mesh; `Bilinear` default for point methods | decided: A, `EqualArea` for coverage (2026-11-16) |
| N2 | non-convex cells | equal-area split; fallback note never seen | keep, and test that it never fires |
| N3 | folds other than tripolar | folds that flip orientation refused at construction | decided: A, refuse; one mesh per scan for overlapping swaths; overlaps that keep orientation refused by `from_cf_bounds` (4b) (2026-11-17) |
| N4 | `mesh` public or crate-private | public, documented unstable | decided: A, public; may break at a 0.x minor release, not a patch (2026-11-17) |
| N5 | robust predicates | plain `f64`, no snapping yet | plain `f64`, snap first |
| N6 | `f64` edge fractions in controlledburn | `f32` in `burn`, `f64` in `CoverageWindow` | decided: `f32`, documented (0.3.0) |
| C1 | weight index types | `u32` per axis | settled; confirm |
| C2 | default normalisation | `FracArea` | `FracArea`, documented |
| C3 | `warp()` built on `exact_weights` | shares the `Lattice` | settled; confirm |
| C4 | spherical measure on the destination side | source side only | wait for a use |
| C5 | cells bounded by parallels, or by great circles | parallels (exact for CF lon/lat) | decided |

---

## N1. The cell map for coverage (decided)

**Decided (2026-11-16): option A.** Coverage uses the equal-area map on
every mesh: `ring_to_index`, `burn`, `burn_par` and `cover_ring` no
longer refuse a `Bilinear` mesh, and give the same records as on an
`EqualArea` mesh. A mesh's `CellMap` is now only the map of its point
methods (`to_index`, `to_world`, `local`, `world`), `Bilinear` by
default for parity with GDAL's point kernels and `geo_to_pixel`. A
bilinear coverage measure, if anyone wants one, would need densified
rings (bigcurve) and an option of its own.

![Bilinear vs EqualArea on one distorted cell](figures/cellmap.svg)

A curvilinear cell is the quadrilateral through its four corners. Bilinear
and equal-area maps give **the same cell**; they differ in where an interior
point of the unit square lands. Under the bilinear map (GDAL's geolocation
model) equal index areas are unequal world areas, so index-space coverage
fractions are wrong. The figure's cell is deliberately distorted (0.83 to
1.17 of an equal share). On ordinary grids the error is about 1.5% of a
cell (controlledburn-rs `examples/grid_frontends.rs`). The equal-area map
splits the cell into four triangles at the corner mean, and every index
area maps to an equal share (1.000 to 1.000).

**Before the decision:** `CellMap::Bilinear` was the default at
construction, and `QuadMesh::burn` / `ring_to_index` required `EqualArea`
(they refused a bilinear mesh), so coverage users had to ask for it.

| option | for | against |
|---|---|---|
| **A. `EqualArea` default for coverage**, `Bilinear` for point kernels and `geo_to_pixel` (recommended) | coverage and conservative remapping are exact by default; nobody gets 1% errors by forgetting a flag | two maps of one grid in one crate; a point's index position differs from GDAL's inside a cell (never at corners or along sides) |
| B. `Bilinear` default everywhere (today) | one model, GDAL-identical everywhere; no surprise for GDAL users | coverage callers must opt in, or `burn` refuses them |
| C. `EqualArea` default everywhere | one default | point kernels lose GDAL parity; the map has kinks along the four spokes, so interpolated fields get slope breaks |

## N2. Non-convex cells

![Non-convex cells and their split point](figures/nonconvex.svg)

For a non-convex (but simple) cell the corner mean can lie outside the cell.
The points that give four equal-share triangles lie on a line through the
corner mean (red), and the split point must also see every corner (the
kernel, blue). The design allowed for the line missing the kernel: fall
back to the nearest kernel point, accept a slightly inexact cell, and add a
note.

**New evidence:** a random search over 59,234 simple non-convex
quadrilaterals (`design/figures/nonconvex-search.R`) found **no case**
where the line misses the kernel. It may always meet it, which would make
the fallback dead code.

| option | for | against |
|---|---|---|
| **A. Keep the split, make the fallback a debug assertion, add the search as a property test** (recommended) | turns an unknown into a test; keeps the path if a counterexample turns up | none, beyond writing the test |
| B. Accept a non-exact cell with a note (the design as written) | robust whatever the truth is | carries code for a case that may never happen |
| C. Refuse grids with such cells | simplest | refuses real grids for a case not yet seen |

A proof that the line always meets the kernel would settle it; the search
is the cheap substitute.

## N3. Folds other than tripolar (decided)

**Decided (2026-11-17): option A.** Coverage refuses cells that overlap.
A swath whose scans overlap is used as one mesh per scan: each scan is
one sheet, each source pixel gets its own coverage, and where scans
cover the same ground, how to combine them (sum, mean, the first scan
only) is the caller's choice. The `mesh` module docs show it
("Overlapping swaths: one mesh per scan", a doctest).

Stage 4 (#37) extends the refusal to overlaps that keep orientation,
which the orientation test misses (built, 4b, 2026-11-17):

- `QuadMesh::from_corners` refuses a mesh whose cells have both
  orientations. That catches overlapping swath scans: towards the swath
  edges the cells between two scans flip.
- A copy that keeps orientation passes. NEMO's halos are that: the
  cyclic columns repeat at a shift in index space, and the north-fold
  row repeats rotated 180 degrees in index space. At an F pivot the last
  row mirrors row `jpj - 1`. At a T pivot it mirrors row `jpj - 2`, and
  row `jpj - 1` mirrors itself, its cells the same ground in pairs. A
  shift and a half-turn both keep orientation, so an ORCA mesh with its
  halos would be accepted and those cells counted twice. Seen on IPSL's
  eORCA1 (F) and ORCA2 (T) CMIP6 files.
- `QuadMesh::from_cf_bounds` takes the halo layout (`Halo`, with
  `NEMO_F_PIVOT` and `NEMO_T_PIVOT`) and drops the duplicates. At a T
  pivot it keeps the lower half of each cell of the row that mirrors
  itself. The seam and fold are walls that meet, which the walk crosses
  since 4a, not data. It then refuses the mesh if
  `QuadMesh::find_overlap` finds two cells that overlap. That check
  probes each cell once, at a point inside it, and reports a second
  owner; it catches copies of any kind, though not an overlap thinner
  than about half a cell. `from_corners` doesn't run it (it costs about
  what building the mesh does), and its docs point to it.

![Tripolar fold vs overlapping sheets](figures/folds.svg)

A tripolar grid (ORCA and similar) folds its top row onto itself: cell
`(top, i)` meets `(top, ncol - 1 - i)`. That is a neighbour rule, so the
walk crosses it like any side (stage 4, `Topology::Tripolar`). Other folds
are different in kind. In swaths with overlapping scan lines (the MODIS
bow-tie) two cells cover the same ground, so locating a point is ambiguous
and a walk can follow only one sheet.

**Today:** construction refuses folded (minority-sign) cells.

| option | for | against |
|---|---|---|
| **A. Refuse for coverage in the first version** (recommended) | keeps the walk single-sheet and exact; the error is explicit | swath users must cut or deduplicate scan lines first |
| B. Design for overlapping sheets now | serves raw swaths directly | location returns several cells and the walk branches; it isn't clear what "coverage" means where sheets overlap (sum, first, mean?); a lot of design for an unasked case |

## N4. Is `mesh` public? (decided)

**Decided (2026-11-17): option A.** `mesh` stays public, and "unstable"
says what it means: the module changes more often than the rest of the
crate and may break at any 0.x minor release; patch releases keep it,
as Cargo's semver rules expect. Since #71 `QuadMesh` is also in a public
signature (`SourceCells::mesh`, reached from `Transformer::source_cells`),
so B would have taken `SourceCells` private with it. The name stays
`mesh`; wxylib's m33 names its warp (rendering) meshes `rwarp::view::mesh`.
Stage 4 is expected to change `from_corners` (a topology), within that
promise.

**Before the decision:** `rwarp::mesh` was public and documented as
unstable until a second consumer used it.

| option | for | against |
|---|---|---|
| **A. Public, marked unstable** (today; recommended) | wxylib, the R package and others can try it now; feedback comes early | an unstable public API in a 0.x crate invites breakage reports; semver discipline gets fuzzier |
| B. Crate-private until wxylib's cover module asks | free to change; nothing to break | no outside use, so no feedback; the R bindings can't reach it |
| C. A separate crate (`rwarp-mesh`) | own version and stability promise | another crate to publish; it still needs rwarp's transformers for the exact footprint |

## N5. Robust predicates

The walk decides which triangle a segment enters next by orientation tests.
Near-degenerate input (a vertex within rounding of a side, spoke or corner)
is where plain floating point gives inconsistent answers.

**Today:** plain `f64` tests. Snapping of near-degenerate vertices isn't
implemented yet, and no property test has failed without it.

| option | for | against |
|---|---|---|
| **A. Plain `f64` with snapping first, adaptive only if tests fail** (recommended) | fast; simple; cut points are already placed by edge parameter, so shared edges agree bitwise | correctness rests on the property tests finding the bad cases |
| B. Adaptive-precision `orient2d` (Shewchuk) from the start | provably consistent orientation | slower in the hot path; a dependency (or a port) in rwarp; doesn't remove the need to place cut points consistently |

## N6. `f64` edge fractions in controlledburn (decided)

**Decided in controlledburn 0.3.0 (hypertidy/controlledburn-rs#23): option B.**
Edge fractions are computed in `f64` and stored as `f32`, about 6e-8 of a
cell, and controlledburn's contract chapter and `GridEdge` say so. Where
`f64` matters, `CoverageWindow` gives it over a window. rwarp's exact
footprint keeps rounding to `f32` to match `burn`.

This is controlledburn's decision; it is listed here because the
curvilinear front end makes it visible. `burn`'s edge table stores
fractions as `f32`, and `CoverageWindow` is already `f64`. On a curvilinear
grid cell areas vary, so an `f32` fraction's world-area error varies by
cell.

| option | for | against |
|---|---|---|
| A. `f64` edges in controlledburn 0.3.0 | one precision end to end; the exact footprint can drop its `f32` rounding | output type change (0.3.0); twice the memory for the edge table; the C++/R/Python package must change in step |
| **B. Keep `f32`, document it** (chosen) | no churn | the front end's exactness is capped at about 1e-7 of each cell |
| C. Generic over the float type | both | complexity in a crate that is dependency-free and small on purpose |

## C1. Weight index types

**Settled by implementation:** `Weights` stores `u32` row and column per
axis, as conservative.md recommended. ESMF's flattened 1-based index is
`row * ncol + col + 1`, so writers (wxylib) convert in one multiply. Grids
are limited to 4 billion cells per side. Confirm, or ask for `u64`
flattened.

## C2. Default normalisation

![FracArea vs DestArea on a partly covered cell](figures/normalisation.svg)

When a destination cell is only partly covered (the grid's edge, a
coastline, nodata), the two normalisations differ. `FracArea` averages
over what is there. `DestArea` divides by the whole cell, which counts
missing as zero.

**Today:** `Normalization::FracArea` is the default and matches `warp()`.
ESMF's default is `DSTAREA`; xESMF's `conservative` is `DSTAREA` and
`conservative_normed` is `FRACAREA`.

| option | for | against |
|---|---|---|
| **A. `FracArea` default, say so prominently** (today; recommended) | matches `warp()` and what raster users expect at coasts and edges; no artificial low values | differs from ESMF's default, so a side-by-side comparison has to set the option |
| B. `DestArea` default | matches ESMF and xESMF's `conservative` out of the box | partly covered cells come out low (8.0 rather than 12.0 in the figure), which raster users read as a bug |

## C3. `warp()` and `exact_weights`

**Settled by implementation:** `warp()`'s exact footprint and
`exact_weights` share the destination `Lattice` and its per-pixel coverage
function. `warp()` keeps its direct reduction (no weights materialised) and
its `f32` rounding for 0.2/0.3 compatibility, and a test holds the two
equal bit for bit. Confirm, or ask for one code path.

## C4. Spherical measure on the destination side

The spherical measure (stage 3a) applies on the source side, which is all
ESMF parity needs, since `frac` is relative to the source cell. A lon/lat
*destination* with a projected source would want `dst_area` on the sphere
too, for `DestArea` normalisation.

| option | for | against |
|---|---|---|
| **A. Wait for a use** (recommended) | no untested mode | `DestArea` for lon/lat destinations with projected sources stays planar in source space |
| B. Add it now | symmetry | another mode to test, with no oracle case yet |

## C5. Cells bounded by parallels, or by great circles (decided)

![A cell bounded by parallels vs by great circles](figures/cell-edges.svg)

ESMF and TempestRemap treat every cell edge as a great circle, so a lon/lat
cell's top and bottom bow poleward: a 40 x 30 degree cell loses 1.9% of
its area, and a 2.5 x 2 degree cell at 40 S loses 3.8e-5. CF lon/lat
rasters are cells bounded by parallels, and rwarp's spherical measure is
exact for those (1e-14 against the closed form). Matching ESMF would mean
curving the *source* cells too, which is spherical polygon overlay, a
different engine.

**Decided (2026-10-29):** parallels for lon/lat rasters, with the ESMF
difference attributed and tested (`tests/esmf.rs`). Great-circle edges
come with meshes and curvilinear grids defined that way (stages 4 and 5),
through bigcurve-rs. This is the pitch: rwarp is exact for the data's own
cell definition, and can say precisely how and why it differs from ESMF.
