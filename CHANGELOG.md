# Changelog

All notable changes to rwarp. Dates are when the change reached `main`.

## Unreleased

### Added

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
