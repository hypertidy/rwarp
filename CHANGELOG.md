# Changelog

All notable changes to rwarp. Dates are when the change reached `main`.

## Unreleased (0.2.0)

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

### Deprecated

- `warp_resample_t` and `warp_resample_u8`, kept as shims with the 0.1
  point-kernel behaviour. Use `warp()`.

### Fixed

- Nearest neighbour mapped source coordinates in (-1, 0) to pixel 0.

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
