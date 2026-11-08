# rwarp (development version)

* `plan_windows()` lists a plan's source windows in the order `warp_exec()`
  reads them, for fetching them all at once before the warp: `read_fn` is
  called per chunk, in turn, so over HTTP a reader that prefetches makes its
  requests overlap. At 0.4 s per request and 5 MB/s, a 384x384 Pacific view
  of GEBCO warps in 1.35 s this way against 3.06 s for gdalwarp (warp-sandbox
  `bench/latency-local.R`).

* `warp_plan()` gains `split_antimeridian` (default `TRUE`): a destination
  crossing the antimeridian of a global longitude/latitude source reads the
  two edges of the source instead of a full-width strip (about 11x fewer
  source pixels for a Pacific view of GEBCO), with the same values. A
  destination around a pole keeps gdalwarp's plan. `FALSE` is gdalwarp's plan
  everywhere. Needs rwarp (the crate) 0.3.3.

* `warp_exec()` runs its chunk loop in Rust and exposes the crate's warp
  (rwarp 0.3.2): all of gdalwarp's kernels (`near`, `bilinear`, `cubic`,
  `cubicspline`, `lanczos`, `average`, `mode`, `min`, `max`, `med`, `rms`),
  integer, double and raw (byte) data, multiple bands, source and destination
  nodata (one or per band), `band_nodata` (gdalwarp's `UNIFIED_SRC_NODATA`),
  `mask_band` and `mode_ties`. The data type follows what `read_fn` returns.

* `warp_exec(output = "nara")` returns a `nativeRaster` for 1-, 3- or 4-band
  raw data, packed in Rust, for `plot()`, `grid::grid.raster()` and
  `png::writePNG()`.

* `warp_plan()` gains `resample` and `scaling` (`"widen"`, gdalwarp's default;
  `"point"`, as `-wo XSCALE=1 -wo YSCALE=1`; `"fixed"`, with `scale`), so
  source windows are padded for the kernel, and each chunk carries its kernel
  scale (`scale_x`, `scale_y`). The plan records `src_dim`, `dst_off`,
  `dst_size`, `resample` and `scaling` as attributes, which `warp_exec()`
  uses by default.

* `warp_plan()` now plans as gdalwarp does: it uses GDAL's special points (so
  a pole inside the destination is found), builds the transformer with the
  source size and target extent (CENTER_LONG wrapping), and
  `resample_padding` defaults to `NULL`, gdalwarp's `SOURCE_EXTRA` rule,
  instead of `1L`.

* Against gdalwarp on the same GDAL build, on a lon/lat source downsampled to
  polar stereographic: every kernel matches exactly for Int16 and Byte data,
  and to 1e-9 for Float64, except `mode` on floats, where GDAL's tie-break
  differs from its own 8/16-bit path (rwarp keeps the 8/16-bit behaviour).

* `warp_exec(nodata = )` is deprecated: use `src_nodata` and `dst_nodata`.
  `warp_resample()` is unchanged.
