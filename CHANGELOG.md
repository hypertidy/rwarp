# Changelog

All notable changes to rwarp. Dates are when the change reached `main`.

## Unreleased

### Changed

- The exact footprint (`Footprint::Exact`, `exact_weights`) over a
  geolocation source measures in the source's world coordinates (#36,
  stage 3 of `design/curvilinear-frontend.md`). Each destination pixel's
  outline goes to source world coordinates and is walked onto the source
  window's cells under the equal-area map. Each source pixel is the
  quadrilateral through its corners' positions. So weights are fractions
  of each cell's world area, and `Average` and `Mode` weight each source
  pixel by its world area. Before, the outline went through the inverse
  geolocation transform and was measured in source pixel space.
  - Effect: for a destination pixel inside the source, weight times cell
    area now sums to the outline's area to 2e-13 (relative). The old
    measure was off by up to 3e-3 on a polar-sector grid, and 6e-3 on a
    curved longitude/latitude grid to polar stereographic.
  - Across the antimeridian (#72): with `lon_wrap`, the arrays'
    longitudes are unwrapped by continuity, so cells across the seam are
    whole. Each outline is unwrapped too and walked at each whole turn
    that reaches the cells, and one that goes round a pole is closed
    along latitude 90 or -90. The same swath stored with its seam in
    different places gives the same weights (to 1e-11, the rounding of
    areas at longitude 180), and weight times cell area sums to the
    outline's area across the seam.
  - Fallback: arrays with a missing sample, arrays that fold, and, with
    `lon_wrap`, arrays whose longitudes don't unwrap consistently (a
    swath over a pole) or span more than 360 degrees keep the old
    measure, decided once per set of arrays
    (`GeolocArrays::world_cells`).
  - Values change for exact-footprint warps and weights from geolocation
    sources. Point kernels, the GDAL box footprint and affine sources are
    unchanged bit for bit. rwarp-r doesn't expose geolocation sources.
  - Speed: 3 to 9 times the old measure at densify 2 (4 to 6 microseconds
    a pixel in the timings for #36).
- `exact_weights_with`'s `CellWeights` gains `src_area`, each entry's
  source cell measure.
- The mesh walk (`QuadMesh::ring_to_index`, and so `burn`, `burn_par` and
  `cover_ring`) is 4 to 30 times faster, with the same output bit for bit
  (checked over 17,806 rings). The changes:
  - outer sides are found through a bucket index, instead of every outer
    side being tested for each segment that leaves the mesh;
  - a cell is clipped only if the segment meets its bounding box;
  - the flood's buffers are reused along a ring.
  Natural Earth 10m land onto the warped 1440 x 720 mesh of
  `examples/mesh_burn_par.rs` now takes 0.24 s with `burn` (1.15 s
  before), and 0.06 s at 8 threads (0.23 s).

- `mesh` says what "unstable" means (decision N4): the module may break
  at a 0.x minor release, not at a patch release. Its docs show one mesh
  per scan for a swath whose scans overlap (decision N3; a doctest), and
  `QuadMesh::from_corners` says that overlap keeping orientation (a row
  or column stored twice, such as an ocean model's halo) isn't detected
  yet.

### Fixed

- `QuadMesh::ring_to_index` (and so `burn`, `burn_par`, `cover_ring`) on
  meshes with cells of no area inside them, such as a T-point pivot of a
  tripolar grid, a collapsed column, or a pole row in a polar projection
  (stage 4b, #37). Four fixes:
  - A ring crossing such a cell jumps in index space at one world point.
    It was joined straight to the next piece's end, misplacing the
    following cell's coverage. It's now linked through the collapsed
    cell, and, where that cell is on the boundary, across the walls. A
    jump the walk can't follow is an error, not a wrong ring.
  - The equal-area split of a cell with a side of no length (a triangle)
    put its centre on the collapsed corner and its unit centre outside
    the unit square, so it was inexact and placed index points outside
    the cell. It's now exact.
  - A vertex at a collapsed corner took a position from another of the
    cell's triangles.
  - A ring starting on a corner whose owner had no area was taken as
    starting outside the mesh.
  - Bounded meshes without such cells give the same output bit for bit.
  - Also: a cell's area that is zero to rounding is now zero, so a cell
    collapsed onto a line, with its corners not quite on it, no longer
    takes an orientation and gets the mesh refused as folded.

- `QuadMesh::ring_to_index` for a ring that touches the mesh at a single
  boundary point from outside: it left and re-entered at the same point,
  and the walk closed it along the whole boundary, so `burn` and
  `cover_ring` covered every cell. Now the ring misses the mesh. A ring
  around the whole mesh that touches it at one point still covers it.

- The exact footprint (`Footprint::Exact`, `exact_weights`) across a
  geographic source's antimeridian and around a pole (#36, the cases #11's
  comparison found). A destination pixel whose outline crosses the
  source's seam is unwrapped and measured at each whole-period offset that
  reaches the source; one that goes around a pole is closed along the
  pole's row. On a global 1 degree grid to 41 x 41 polar stereographic
  pixels, the weights no longer depend on where the source's seam is (to
  5e-14), every pixel's weights sum to its outline's area, and the pole
  pixel's converge to the polar cap at second order in densify; before,
  weights across the seam were off by up to 0.92.

### Added

- `mesh::MeshSet`: a whole curvilinear grid as rectangular pieces, each a
  `QuadMesh` in the projection of its band of rows, for an ocean model's
  whole global grid (stage 4g of `design/curvilinear-frontend.md`, #37).
  - `MeshSet::from_cf_bounds(ncol, nrow, xb, yb, halo, bands, droppable)`
    projects each `Band` to its own world. It leaves out cells that are
    bow-ties there, turned the wrong way or overlapping others, where
    `droppable` allows (land), and refuses them otherwise. The rest are
    covered by `Piece`s, so a cell left out is a hole.
  - `weights_to` and `weights_onto` give conservative weights from and
    onto the whole grid as one `Weights` in the file's numbering, from
    `exact_weights` through a `MeshTransformer` or `MeshTarget` per
    piece. Each piece is measured only over the pixels it reaches.
  - For ORCA: bands in north-polar and south-polar LAEA on the model's
    sphere, both equal-area, so areas add up across them. Where they meet
    the cells bow apart, a gap of about dlam^2 / (6 dphi) of a cell (under
    1% at 20 S on ORCA2).
  - Checks (`tests/mesh_set.rs`): an analytic global tripolar grid with
    damaged land, conservation to 7e-15, the gap within its bound, the
    rest unchanged by the damage. On CMIP6's eORCA1 and ORCA2 whole (in
    scratch): every ocean cell kept, 716 and 801 land cells left out, to
    1 degree in about 1 s each.
  - `MeshSet::burn(geoms, bands, tolerance, opts)` (stage 4h) burns
    polygons and points in the bounds' coordinates (longitude and
    latitude) onto the set. It returns controlledburn's tables in the
    set's numbering, as fractions of each cell's area.
    - Edges are straight in the bounds' coordinates. bigcurve densifies
      them into each band's world to `tolerance` (world units).
    - Each band takes only the polygons' part within its cells' y
      range, so a polygon round the far pole isn't turned inside out.
    - Rings are split where a cut at 180 or along a pole brings them
      back onto themselves.
    - Checks: boxes on the analytic grid against closed-form areas, to
      8e-8. On eORCA1 and ORCA2 (scratch): a whole-world box covers
      every kept cell, and boxes agree with `weights_onto` to 8e-6.
      Natural Earth's 10 m countries burn in 0.7 to 1.1 s.
    - Each piece walks only what can reach it. Tiles mark where its cells
      are, so polygons elsewhere are skipped and runs of edges away from
      it cut short, with the same result bit for bit. On ORCA2 the
      countries went from 3.0 s to 1.1 s.
    - `MeshSet::burn_par(geoms, bands, tolerance, opts, threads)` gives
      the same records on several threads. Pieces filter, walk and burn
      their polygons in parallel. Projection stays on the calling
      thread, since a band's transform needn't be thread-safe. On 8
      threads ORCA2's countries take 0.44 s.
    - `MeshSet::burn_par_with(geoms, bands, tolerance, opts, threads,
      to_world)` projects on the threads too: each thread builds its
      own copy of band `k`'s transform with `to_world(k)`. The records
      are the same again. On 8 threads ORCA2's countries take 0.18 s at
      1 m and 0.92 s at 0.01 m (2.2 s with `burn_par`).
  - New dependency: bigcurve 0.3 (pure Rust, no dependencies of its own,
    builds for wasm).
- `CrsTransform` for `Box<C>` and `&C` (`C: CrsTransform + ?Sized`), so
  `crs_pair`'s boxed transforms go where a `T: CrsTransform` is asked
  for, such as `MeshSet::burn_par_with`'s `to_world`.
- `mesh::MeshTarget`: a `QuadMesh` as a warp target, for regridding onto
  a curvilinear grid given by its corners, such as an ocean model's
  tripolar grid (stage 4f of `design/curvilinear-frontend.md`, #37).
  - How: target pixel `(col, row)` is mesh cell `(row, col)`. A target
    pixel goes to the mesh's world coordinates and then through an inner
    transformer to the source. `from_grid(src_crs, src_gt, src_size,
    mesh, mesh_crs)` makes that a `GenImgProjTransformer`, wrapping a
    lon/lat source's longitudes as gdalwarp does. `new(mesh, inner)`
    takes any transformer whose destination geotransform is
    `mesh::WORLD_GT`: geolocation arrays, or another mesh
    (`MeshTransformer`), for mesh to mesh.
  - What it gives: `exact_weights` measures each cell as a target pixel
    in the source's space, under any measure (`lonlat_measure` for areas
    on the sphere), across a lon/lat source's seam and over its pole.
    `warp()` regrids onto the mesh with any kernel; point kernels sample
    each cell's centre.
  - Checks (`tests/mesh_target.rs`): in one plane, the overlaps are the
    mesh source's transposed (2e-14), and mesh to mesh is symmetric. From
    a global lon/lat grid every cell is covered in full and the seam's
    position doesn't matter. `warp()` is `Weights::apply` bit for bit.
  - Against ESMF onto the tripolar cap (`tests/esmf_tripolar.rs`, new
    `onto_tripolar` fixtures from `gen.py`; the existing fixtures are
    unchanged bit for bit): Cartesian in the plane to rounding (3e-13);
    spherical recorded, S up to 3e-2 on the cap's smallest cells and
    3e-3 over cells more than 100 km across.
- A declared period for coordinates that jump at the seam, such as
  longitude: `QuadMesh::with_period` and `Rectilinear::with_period`
  (stage 4e of `design/curvilinear-frontend.md`, #37).
  - Polygons: `burn`, `burn_par` and `cover_ring` take each polygon at
    each whole period at which it overlaps the grid, so polygons from -180
    to 180 burn onto a grid from 0 to 360, and across its seam.
  - Points: `locate`, `locate_all`, `to_index` and `Rectilinear::cell`
    find a point at any whole period.
  - A mesh source: `MeshTransformer` over a mesh with a period takes
    target pixels across its seam, and round a pole (closed along
    latitude 90 or -90).
  - `mesh::unwrap_grid` and `mesh::unwrap_bounds` make longitudes stored
    with a jump (`[-180, 180)`, `[0, 360)`) continuous; `from_corners` and
    `from_cf_bounds` name them when they refuse such corners.
  - Refused: a cell half a period wide or more (round a pole, as at a
    tripolar grid's north), and a mesh that goes round more than once.
    `find_overlap` includes the period once it is declared.
  - Checks (`tests/mesh_period.rs`): generator 5, a regular global grid
    with its columns rotated, against controlledburn's regular burn
    (bitwise at `f32`); the same weights wherever a mesh's seam is, for a
    lon/lat destination and a polar one whose centre pixel holds the
    pole; `warp()` against `Weights::apply`.
  - Meshes without a period are unchanged bit for bit.
- `mesh::MeshTransformer`: a `QuadMesh` as a warp source, for data with
  one value per cell on a curvilinear grid given by its corners, such as
  an ocean model's tripolar grid (stage 4d of
  `design/curvilinear-frontend.md`, #37).
  - Constructors: `new(mesh, mesh_crs, dst_gt, dst_crs)` with the CRS
    backend, or `with_backend` with your own `CrsTransform`s.
  - What it gives: it reports the mesh as the source's cells
    (`Transformer::source_cells`). So `exact_weights` gives conservative
    weights from the mesh, as fractions of each cell's world area across
    the seam and fold, and `warp()` with `Footprint::Exact` regrids with
    the area methods.
  - Point kernels go through the mesh's cell map (`QuadMesh::to_index`).
  - Checks: `warp()` with `Average` is `Weights::apply(FracArea)` bit for
    bit, every cell's weights sum to 1, and source windows tile the
    mesh (`tests/mesh_transformer.rs`).
- `QuadMesh::window`: a window's cells as a mesh of their own.
- ESMF on a tripolar grid (stage 4c of `design/curvilinear-frontend.md`,
  #37; `tests/esmf_tripolar.rs`). The weights come from `exact_weights`
  through `MeshTransformer`, on an analytic tripolar cap with a seam into
  one pole and an F-point fold. They are compared on two destinations: a grid
  in the cap's own plane, and lon/lat up to the pole.
  - Cartesian ESMF in the plane: the weights agree to rounding (3e-14).
  - Spherical ESMF: the differences are great-circle edges, recorded as
    ceilings. S differs by up to 2.6e-3 and areas by up to 9e-4, and the
    area difference falls with the square of cell size.
  - The NEMO layout of the same grid gives the same mesh to the bit.
  - Fixtures come from `tests/fixtures/esmf/gen.py`, which now takes case
    names. Its existing fixtures are unchanged bit for bit.
- `QuadMesh::from_cf_bounds`: a mesh from each cell's own four vertices,
  CF bounds (`lon_bnds(j, i, 4)`) or SCRIP corners
  (`grid_corner_lon(grid_size, 4)`), with a `Halo` of rows and columns
  dropped first (stage 4b, #37, decision N3).
  - Vertex order: found from the data (CF's or any rotation or
    reflection of it).
  - Shared corners: neighbours must share them, to a thousandth of a
    cell.
  - Overlap: refused, as a halo left in gives.
  - NEMO: `Halo::NEMO_F_PIVOT` and `Halo::NEMO_T_PIVOT` are the ORCA
    layouts as CMIP6 files have them (the last two columns, the first
    row and the last row). At a T pivot the row the fold runs through is
    halved, and its model cells are two mesh cells.
  - Real data: checked on IPSL's eORCA1 and ORCA2. Areas agree with
    `areacello` to 1e-4 in total, and a cap over the fold is covered to
    1e-10 of its area. Both files have land cells (eORCA1 south of 76 S,
    flattened by a north-polar projection; corrupt bounds in ORCA2) that
    are refused, so for now those rows are dropped.
- `QuadMesh::find_overlap`: two cells that overlap, a point inside one
  owned by the other too, for copies that keep orientation, which
  `from_corners` doesn't check.

- `QuadMesh::ring_to_index`, and so `burn`, `burn_par` and `cover_ring`,
  cross outer walls that meet in the world: a periodic grid's seam
  (column line `ncol` on column line 0) and a tripolar grid's fold (the
  last row line on itself, corner `i` on corner `ncol - i`), with
  nothing declared (stage 4a of `design/curvilinear-frontend.md`, #37).
  Give the grid's corners in a projection in which it is one sheet:
  for an ocean grid's tripolar north, Lambert azimuthal equal-area on
  the north pole, whose planar areas are true areas. The walls need to
  agree only to rounding.
  - Before: a ring crossing a seam or fold was joined across the mesh in
    index space, and cells along the crossing were off by up to a whole
    cell.
  - Bounded meshes: the same output bit for bit. Tested against the
    per-cell world clip on an annulus, a fold and a tripolar disk
    (`tests/mesh_glued.rs`).
  - Not yet: coordinates that jump at the seam (longitudes 360 apart).

- `Transformer::source_cells` and `transform::SourceCells`: a transformer
  reports a source whose pixels are cells in its own world coordinates.
  It provides the map from destination pixels to source world coordinates
  (`dst_to_world`) and the cells of a source window (`mesh`), for the
  exact footprint. Default `None`. `GenImgGeolocTransformer` with
  geolocation arrays on the source side, and `GeolocArrays` on its own,
  report it when `GeolocArrays::world_cells()` is `Ok`;
  `ApproxTransformer` passes it through, with `dst_to_world`
  unapproximated. `SourceCells::lon_wrap` (default `false`) says world x
  is longitude that wraps.
- `GeolocArrays::world_cells`, whether the exact footprint measures a
  source with these arrays in world coordinates (and if not, why), and
  `GeolocArrays::window_mesh`, a source window's cells.
- `Weights::src_cell_area`: each entry's source cell area, for a
  geolocation source (empty otherwise, and not serialised when empty).
  `Weights::src_area` uses it.
- `Transformer::source_wrap` and `transform::SourceWrap`: a transformer
  reports a source that wraps in longitude (its period in source pixels
  and the rows of the poles), for the exact footprint. Default `None`;
  `GenImgProjTransformer` reports it for a north-up geographic source of
  at most 360 degrees wrapped around a centre longitude (after
  `with_source_size`), and `ApproxTransformer` passes it through.

### Changed

- Decision N1 (#43): coverage uses the equal-area map on every mesh.
  `QuadMesh::ring_to_index`, `burn`, `burn_par` and `cover_ring` no longer
  refuse a mesh built with `CellMap::Bilinear`; they give the same records
  as on a `CellMap::EqualArea` mesh. A mesh's `CellMap` is now the map of
  its point methods only (`to_index`, `to_world`, `local`, `world`),
  `Bilinear` by default for GDAL parity.

## 0.4.1 (2026-11-16)

### Added

- `QuadMesh::burn_par` and `Rectilinear::burn_par`: the mesh burns on up to
  `threads` threads (0 for all available), with the same result as `burn`,
  record for record. The ring walk into index space runs on contiguous
  chunks of the geometries of about equal vertex counts, taken in turn by
  the threads, and the burn uses controlledburn's `burn_par`. Natural Earth
  10m land (6,837 polygons, 446,175 vertices) onto a warped 1440 x 720
  mesh: 1.16 s with `burn`; 0.58 s, 0.31 s and 0.24 s at 2, 4 and 8
  threads, and no faster past 8, as one geometry is never split (the
  largest, 19% of the vertices, takes 0.33 s alone).
  `examples/mesh_burn_par.rs` times it.

## 0.4.0 (2026-11-12)

### Changed

- controlledburn 0.3.1 (from 0.2.4). Breaking for rwarp because its types
  are in rwarp's API (`weights::Measure`, and `Geometry`, `BurnOptions` and
  `BurnResult` in `QuadMesh::burn` and `Rectilinear::burn`). Options are
  `#[non_exhaustive]` in 0.3, so build them with `BurnOptions::coverage()`
  and the `with_*` methods. A multi-part geometry now gets one record per
  cell in a mesh burn (`merge_parts`, on by default); `BurnOptions::parity`
  keeps one per part. Cutlines burn each polygon separately and are
  unchanged. Otherwise the test suite gives the same results as on 0.2.4.

## 0.3.3 (2026-11-12)

### Changed

- `ChunkOptions::split_antimeridian` no longer reads the whole source width
  for the columns at the jump (the gap). Each gap pixel needs only one edge
  of the source, so the gap becomes two chunks on the same target window,
  one source window at each edge, sampled every half pixel and padded for
  the gap's own kernel scale. A 384x384 Lambert azimuthal target centred on
  180 degrees over GEBCO at overview 3 reads 200,600 source pixels instead
  of 2,257,200, with the same values (exact transformer; nearest, bilinear
  and cubic; Int16, Float32 and Byte).
- The split is kept only when it reads under half of the full-width window,
  so a target around a pole (where every longitude is needed) keeps GDAL's
  plan, and it no longer depends on the fill ratio split being on.
- `warp_chunks` places only the pixels a chunk wrote (not equal to the fill,
  NaN equal to NaN), through the new `plan::merge_tile`. For chunks that don't
  overlap this is the plain copy it was.

## 0.3.2 (2026-11-04)

### Added

- The curvilinear front end, stage 2 of `design/curvilinear-frontend.md`
  (unstable): `QuadMesh::ring_to_index` walks a world ring into the
  mesh's index space, cutting each segment at every cell side and
  split spoke it crosses (Cyrus-Beck clipping per triangle over a flood
  of the cells it passes, so vertices, corners and centres on a segment
  need no special case). A cut point is placed by its parameter along
  the edge's canonical direction, so rings sharing an edge get bitwise
  the same point. Stretches outside the mesh are replaced by the boundary
  arc whose loop with the excursion does not wind around the mesh; a
  ring around the whole mesh gives its outline. `QuadMesh::burn` burns
  polygons (holes, multipolygons) and points through controlledburn with
  the mesh's own `(row, col)` and fractions of each cell's world area;
  `QuadMesh::cover_ring` does one small ring through a reused
  `CoverageWindow`. `mesh::Rectilinear` is the same for grids given by
  column and row edges (cut at every wall, piecewise linear per axis).
  Checked against world-space clipping of every cell (Sutherland-Hodgman
  per split triangle) on rotated, jittered, non-convex, polar, sheared
  and flipped meshes with polygons crossing the boundary, holes and
  vertices steered onto corners and centres; against the affine burn
  for a rotated grid; and rectilinear against the mesh, all to 2e-6
  (controlledburn's `f32` fractions). Lines and approx mode are refused
  for now. (2026-11-01)
- `mesh` module, stage 1 of `design/curvilinear-frontend.md` (unstable):
  `QuadMesh` holds a curvilinear grid as straight-edged quadrilateral
  cells (`from_corners`, `from_centres`), refuses bow-ties and folds,
  and maps between world coordinates and index space with
  `CellMap::Bilinear` (GDAL's model; equal to `GeolocArrays::geo_to_pixel`
  with corner samples, to 1e-9) or `CellMap::EqualArea` (four triangles
  per cell with matching area shares, so index-space area fractions are
  world-area fractions; checked to 1e-9 on jittered, non-convex, polar
  and sheared grids, where the bilinear measure is off by more than
  1e-3). Every point has at most one owning cell: side orientations are
  computed once in a canonical direction, and points on a shared wall go
  to the low-index cell. `locate` walks from a hint and falls back to CSR
  buckets built on first use; `locate_all` returns every owner (for
  checks). No CRS: builds with `--no-default-features`.
  (2026-10-31)
- A spherical measure for conservative weights (stage 3a of
  `design/conservative.md`): `WeightOptions::with_measure` with
  `weights::lonlat_measure(src_gt)` measures areas on the sphere for a
  source on a regular longitude/latitude grid, through controlledburn's
  `Measure::CosLatitude` (closed-form edge integrals of `cos(lat)`; the
  cell partition and conservation are unchanged). `Weights` records the
  `measure`, `Weights::src_area(k)` gives an entry's source cell measure,
  and `normalised`, `dst_covered` and `apply` weight overlaps by it; for the
  default planar measure every cell is 1 and results are unchanged, bit for
  bit. Exact against closed-form lon/lat box overlaps on the sphere (to
  1e-14). Needs controlledburn 0.2.4. (2026-10-29)
- ESMF oracle suite (stage 2 of `design/conservative.md`, `tests/esmf.rs`):
  six cases, each against ESMPy 8.9.1's first-order conservative weights
  in Cartesian source pixel space (rwarp's cell definition: equal to about
  1e-12) and on the sphere with great-circle edges (differences recorded as
  ceilings to shrink). Fixtures from `tests/fixtures/esmf/gen.py`, run in
  `ghcr.io/hypertidy/gdal-r-esmf` by `.github/workflows/esmf-fixtures.yml`.
  (2026-10-29)

### Changed

- The `serde` feature also enables controlledburn's, for `Measure` in
  `WeightOptions` and `Weights`. (2026-10-29)

## 0.3.1 (2026-10-28)

### Added

- Longitude wrapping of a geographic source, GDAL's `CENTER_LONG`:
  `GenImgProjTransformer::with_source_size` lets the transformer wrap
  source longitudes around the source raster's centre as gdalwarp does
  (`InsertCenterLong`, `OGRProjCT`), so a source stored as 170..190,
  -190..-170 or 0..360 is found by a target crossing 180 degrees.
  `CenterLong` (`Auto`, the default; `Off`; `At(lon)`, `#[non_exhaustive]`)
  with `with_center_long`, `with_target_extent` for gdalwarp's `-te`,
  `center_long()`, `source_center_long` for GDAL's centre, and
  `crs::is_geographic_degrees`. A transformer built without
  `with_source_size` wraps nothing, as before. Checked against GDAL
  3.14.0dev on 11 cases across 180 degrees (LAEA, Mercator, polar
  stereographic, lon/lat targets; chunked and whole): centres, exact
  transformer points, chunk lists and every output pixel, with both CRS
  backends. The LAEA chunk fixtures are back across 180 degrees.
  The `proj` feature now also depends on `proj-sys` (already a dependency
  of `proj`), to ask PROJ whether a CRS is geographic.
- `weights` module: conservative remapping weights from the exact
  footprint, as values (stage 1 of `design/conservative.md`).
  `exact_weights` returns `Weights`, one entry per overlapping (target
  cell, source cell) pair with full-grid indices and the overlap as an
  unrounded `f64` fraction of the source cell, plus each target cell's
  outline area in source pixel space (`dst_area`, `NaN` where it failed).
  `Weights::normalised` gives ESMF's `S` under `Normalization::FracArea`
  or `DestArea`; `Weights::apply` applies the weights to one or more bands,
  skipping source nodata; `Weights::esmf_row_col` gives ESMF/SCRIP's
  1-based flattened indices. `exact_weights_with` streams them per target
  cell. `WeightOptions` (densify, target offset for tiling, drop and snap
  tolerances, `f32` rounding; `WeightOptions::warp_compatible`),
  `ApplyOptions` and `GridWindow`, all `#[non_exhaustive]` with builders;
  serde derives behind `serde`. `apply(FracArea)` with warp-compatible
  weights reproduces `warp()` with `Footprint::Exact` and `Average` bit for
  bit. `warp()` shares the lattice and per-pixel coverage step and its
  output is unchanged. (2026-10-28)
- GDAL's backmap geolocation inverse, as an option beside the quad index:
  `GeolocOptions::with_inverse(GeolocInverse::Backmap)` (`GeolocInverse`:
  `QuadIndex`, the default; `Backmap`; `BackmapEstimate` for
  `GDAL_GEOLOC_USE_MAX_ACCURACY=NO`; `#[non_exhaustive]`), with
  `with_backmap_oversample` (`GEOLOC_BACKMAP_OVERSAMPLE_FACTOR`) and
  `GeolocArrays::backmap_grid`. Ported from GDAL 3.14 in its arithmetic
  order (float32 backmap, `GDALFillNodata` hole filling, the spiral
  search), by way of wxylib's `wxy-warp` (`geoloc.rs`, GDAL 3.12; the
  geolocation code is unchanged since), adding `PIXEL_OFFSET` / `STEP`.
  Checked against GDAL 3.14.0dev on 9 new cases (swath by centres and by
  corners, lon/lat and polar stereographic targets, a polar ring and a
  swath across the antimeridian, holes in the arrays with source nodata,
  overlapping scans): success and position at every target pixel, the
  backmap's estimate, and 24 warps (near, bilinear, cubic; `-et 0` and
  `0.125`) identical to gdalwarp, every pixel, with proj4rs and the
  fixtures' PROJ (with an older libproj, nearest identical and
  interpolated values within 2e-8 relative).
  Where the two inverses differ, and why, is in the curvilinear guide.
  (2026-10-28)

### Changed

- The chunk fixtures' special points are computed as GDAL computes them for
  a geographic source (not reprojected; see `ChunkOptions::special_points`).
  No chunk list changes.
- `warp()` does GDAL's `GWKAdjustSrcOffsetOnEdge`: when a corner of the
  source buffer fails to transform forward and a sample point lands on an
  invalid source pixel whose corners fail too, a valid neighbour (right,
  below, left, above) is used instead. This changes output only where the
  transformer has holes or edges inside the source, as geolocation arrays
  with missing samples do (12 of 4560 pixels in the holes test case,
  now identical to gdalwarp). `warp()` therefore calls the transformer
  source to destination too, for such pixels only. Point methods; GDAL's
  variant for alpha and cutline densities is not reproduced. (2026-10-28)
- The geolocation swath fixture is regenerated with GDAL 3.14.0dev (was
  3.8.4): positions moved by up to 3e-8 pixels (PROJ), nothing else.
- controlledburn 0.2.3 (corner passes, near-coincident edges, exact centre
  ties). The cutline centre rule now matches gdalwarp on centres exactly on
  diagonal edges: 5 pixels of the `mask_diamond_tie` fixture that differed
  now agree. They came from controlledburn 0.2.2's rounded intercepts; the
  parity page had put them down to GDAL.

## 0.3.0 (2026-10-28)

This release changes the API: options types are `#[non_exhaustive]`.

### Changed

- `warp()` retries a sample point that lands less than one pixel outside
  the source buffer by transforming it again on its own, as GDAL does
  (`GWKCheckAndComputeSrcOffsets`). Through `ApproxTransformer` that is
  exact, so interpolation error no longer drops pixels at the edge of a
  chunk's source window; with an exact transformer nothing changes.
  Point methods only (nearest, bilinear, cubic, cubic spline, lanczos), as
  in GDAL. (2026-10-28)
- `WarpOptions`, `GeolocOptions` and `AverageThresholds` are
  `#[non_exhaustive]`, as are `ResampleAlg`, `KernelScaling`, `Footprint`,
  `BandNodata`, `Nodata`, `DstNodata`, `DataType` and `GeolocConvention`.
  New options and methods can then come in minor releases. Outside rwarp,
  build options with `WarpOptions::new(alg)` and the `with_` methods
  (`GeolocOptions::default().with_convention(..)`, and so on) instead of
  struct literals, and add a `_` arm when matching on the enums. Fields
  stay public for reading and assignment.

### Added

- `WarpOptions::new` and `with_` builder methods on `WarpOptions`,
  `GeolocOptions` and `AverageThresholds`.
- `plan` module: `plan_chunks` and `ChunkPlanner` give gdalwarp's chunk
  list as values (`WarpPlan` of `Chunk`s: target window, source window,
  fill ratio, kernel scale), following GDAL 3.14's `CollectChunkList` and
  `ComputeSourceWindow` exactly: warp memory limit (`-wm`) and per-pixel
  cost, fill ratio splits, `OPTIMIZE_SIZE` on target blocks, special
  points, the forward probe, `SAMPLE_GRID`, `SAMPLE_STEPS`,
  `SOURCE_EXTRA`. `ChunkOptions` (builders, `#[non_exhaustive]`);
  `ChunkOptions::gdalwarp` for gdalwarp's planning exactly; an optional
  round-trip check for backends that do not fail at a projection's
  antipode; the antimeridian split of 0.2's `collect_chunk_list` as
  `split_antimeridian`. `warp_chunks` warps a plan from a source buffer
  in memory. Ported from wxylib's `wxy-warp` (`chunked.rs`). (2026-10-28)
- `serde` feature: `Serialize`/`Deserialize` for `WarpPlan` and `Chunk`.
  (2026-10-28)
- Parity fixtures for whole rasters warped in many chunks
  (`tests/chunks.rs`, `tests/fixtures/chunks/`): seven gdalwarp cases,
  chunk lists and output identical. (2026-10-28)
- Cutlines (gdalwarp `-cutline`): `WarpOptions::cutline` /
  `with_cutline`, and the `cutline` module with `Cutline` (polygons in
  source pixel/line), `CutlineRule::Centre` (GDAL's default, with its tie
  rules), `CutlineRule::AllTouched` (`CUTLINE_ALL_TOUCHED`), blend distance
  (`-cblend`), and `CutlineRule::Exact`, which weights each source pixel by
  the fraction of it the cutline covers (not a GDAL mode).
  `Cutline::from_geotransform` and `Cutline::from_transformer` (densified
  as gdalwarp does) map world geometry to source pixels;
  `Cutline::density` is the per-pixel density the warp uses. The cutline
  is rasterized over the source buffer only, with `controlledburn`.

### Removed

- `source_window::collect_chunk_list` and `ChunkPlan`: use
  `plan::plan_chunks` (with `ChunkOptions::split_antimeridian` for the
  same split at the antimeridian). `compute_source_window` stays.
  (2026-10-28)

## 0.2.0 (2026-10-28)

This release changes the API: `warp()` with `WarpOptions` and
`SourceBuffer` replaces the warp entry points of 0.1, and `ResampleAlg`
gains variants.

### Added

- `geoloc_transform`: geolocation-array transformers, for a curvilinear
  source (`gdalwarp -geoloc`), a curvilinear target
  (`DST_METHOD=GEOLOC_ARRAY`), or both. Both of GDAL's georeferencing
  conventions; offsets and steps; nodata; antimeridian handling.
  (2026-10-28)
- `warp()`, `WarpOptions`, `SourceBuffer`: one entry point for every sample
  type, method and option. `WarpOptions` carries a destination offset for
  warping one tile of a larger grid. (2026-10-28)
- `KernelScaling`: GDAL's kernel widening when downsampling, now the
  default; `Fixed` and `Point` alternatives. `estimate_scale` is public.
  (2026-10-28)
- `ResampleAlg::Average`, with `Footprint::GdalBox` (gdalwarp's) and
  `Footprint::Exact` (first-order conservative). `controlledburn` is a new
  dependency. (2026-10-28)
- `ResampleAlg::Mode` with `ModeTies` (`MODE_TIES`). (2026-10-28)

- `BandNodata` (`UNIFIED_SRC_NODATA`) and `AverageThresholds`
  (`NODATA_VALUES_PCT_THRESHOLD`, `EXCLUDED_VALUES`). (2026-10-28)
- `ResampleAlg::CubicSpline`, `Min`, `Max`, `Med`, `Q1`, `Q3` and `Rms`,
  and `ResampleAlg::is_area()`. (2026-10-28)
- `Sample` for `i64`, `u64`, and `half::f16` behind a new `f16` feature.
  (2026-10-28)
- `warp_to::<S, D>`: a different output type from the source's, through
  GDAL's working type (`DataType::working`) and `GDALCopyWords`
  conversion. (2026-10-28)
- `Nodata` and `DstNodata`: source and destination nodata per band, as
  gdalwarp's `-srcnodata` and `-dstnodata`. (2026-10-28)
- Examples: `tile_laea`, `curvilinear_swath`.
- A documentation site, with the browser demo under `/demo/`.

### Changed

- Integer outputs round half up and clamp, and a resampled value equal to
  nodata is moved off it, as gdalwarp does with an integer working type.
- Sample points outside `[0, n)` of the source are off the source for
  every point method, as in GDAL.
- Lanczos clips its window and renormalises at edges and nodata, and a
  target pixel whose centre lands on nodata gets no value, for every
  interpolating kernel. Some edge and nodata pixels change compared with
  0.1.
- Multi-band nodata follows GDAL's default (a pixel is nodata only where
  every band is); the 0.1 rule is `BandNodata::AnyBand`.
- `Sample` requires `PartialOrd`.
- `Footprint::Exact` is 2 to 6x faster. The destination pixel lattice is
  transformed once per row band (shared corners and edge points) instead
  of four points per pixel, and coverage comes from
  `controlledburn::CoverageWindow` (0.2.2) instead of one `burn` call per
  pixel. With an exact transformer the output is unchanged, except for
  pixels straddling the antimeridian, which the exact footprint does not
  handle (it gave a different wrong answer before). The lattice's row
  lines go to the transformer as scanlines, so an `ApproxTransformer` now
  approximates them, as for the box footprint; before, it transformed the
  exact footprint's points exactly.
- `WarpOptions` is not generic: `nodata: Option<T>` is replaced by
  `src_nodata: Nodata` and `dst_nodata: DstNodata`.
- Bilinear uses GDAL's general formula for types and cases where GDAL
  does, which can change the last bit (and so integer rounding near 2^40
  for 64-bit values).

### Deprecated

- `warp_resample_t` and `warp_resample_u8`, kept as shims with the 0.1
  point-kernel behaviour. Use `warp()`.

### Fixed

- Nearest neighbour mapped source coordinates in (-1, 0) to pixel 0.
- `ApproxTransformer` left one point untransformed and reported it failed
  when subdivision reached a left half of two points, which happens on
  strongly curved transforms or with a small error threshold (whole
  columns of a polar warp came out as nodata). GDAL transforms it.

## 0.1.2 (2026-03-11, not on crates.io)

- Use the system libproj through `proj` 0.31.

Not released to crates.io. On `main` after it (2026-09 to 2026-10):

- `rwarp-wasm`: browser shim and demo (OSM, WMTS, ArcGIS tile services,
  COG sources), `Warper.withTransforms` for JavaScript CRS transforms.
- A generic `Sample` kernel over `u8` to `f64`, interleaved bands, nodata as
  a value or an alpha band.
- The `CrsTransform` trait with `proj` and `proj4rs` backends; builds for
  `wasm32-unknown-unknown`.

## 0.1.1 (2026-03-06)

- `GcpTransformer`: polynomial transformer from ground control points,
  order 1 to 3.
- Use the published `vaster` crate; crate documentation.

## 0.1.0 (2026-03-06)

- First release, extracted from cogcache: `GenImgProjTransformer`,
  `ApproxTransformer`, `compute_source_window`, `collect_chunk_list` and
  the nearest, bilinear, cubic and lanczos kernels.
