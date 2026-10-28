# Changelog

All notable changes to rwarp. Dates are when the change reached `main`.

## Unreleased (0.3.0)

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
