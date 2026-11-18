# rwarp (development version)

* Built on rwarp (the crate) 0.5.0, with controlledburn 0.3 and bigcurve
  0.3. The R functions give the same results: the crate's 0.4 and 0.5
  changes are in its mesh and weights API, the exact footprint, and new
  functions on several threads, which the package doesn't bind yet.

* rwarp has a hex logo, drawn from rwarp's own output: MUR sea surface
  temperature warped to Antarctic polar stereographic in a regular grid of
  tiles, and one row per tile for the plan. The site has favicons to match.
  The logo is made by the scripts in `design/logo/` of the repository.

* The website (<https://hypertidy.github.io/rwarp/r/>) is fuller: a Get
  started article, grouped articles, the changelog, a light/dark switch,
  and source links that work. The `sentinel2-warp` and `ghrsst-polar`
  articles read from the network, so they are now website-only and
  precomputed (`vignettes/articles/precompute.R`) instead of installed
  vignettes that ran only with `VWARP_VIGNETTES=true`. The `ghrsst-polar`
  example moved to 52 S, across the Antarctic Polar Front, at 10 km.

* New vignette, `vignette("rwarp")`: plan and warp `sst.tif` from vapour
  onto polar stereographic, plot the plan's chunks, and match gdalwarp
  (offline).

* `plan-inspect`'s antimeridian example now crosses the antimeridian, and
  shows the source windows the split reads against gdalwarp's full-width
  strip.

* The examples of `warp_plan()`, `warp_exec()` and `pixel_transform()` run
  (offline, nothing read from the network); the remote examples stay in
  `\dontrun{}`.

* `warp_plan()` can plan as gdalwarp plans (#67). New arguments describe the
  warp's working memory as gdalwarp counts it: `nbands`, `working_type`,
  `src_nodata`, `dst_nodata` (whether there's nodata; defaults to
  `src_nodata`) and `memory_limit` (gdalwarp's `-wm`, in MB). New
  `scale_origin = "output"` estimates each chunk's kernel scale as gdalwarp
  does. With these, `min_fill_ratio = 0.5` and `min_dst_size = 100L`, the
  chunks are gdalwarp's, and so are the values where the kernel widens: the
  `sentinel2-warp` vignette now matches gdalwarp at its default memory,
  without `-wm`. The defaults are unchanged.

* `warp_exec()` takes `nbands` from the plan by default.

* `warp_plan()` and `pixel_transform()` accept sizes and offsets as
  whole-number doubles, as `GDALRaster$getRasterXSize()` returns them.

* The `sentinel2-warp` vignette is rewritten (#55): two Sentinel-2 tiles from
  one overpass, in UTM zones 54 and 55, warped as three bands of float
  reflectance onto Australian Albers (EPSG:3577), combined in order as
  gdalwarp combines its inputs, and compared with gdalwarp's mosaic of the
  same sources (the same nodata, values to 1e-11). It also reads one window
  once and warps it again with `warp_buffer()`.

* `warp_exec()` and `warp_buffer()` document that `dst_nodata = NA` writes
  `NaN` for double data, not `NA`; `is.na()` is `TRUE` for both.

* New `warp_buffer()` warps one pre-fetched source window with the crate's
  `warp()` (#54). It has what `warp_exec()` has: every kernel, integer, double
  and raw data, several bands, source and destination nodata, `band_nodata`,
  `mask_band`, `mode_ties`, kernel scaling and `output = "nara"`. The buffer is
  laid out as `read_fn` returns it, or is a matrix or array in R orientation,
  and the result is a matrix or array, as from `warp_exec()`. On the window
  `warp_plan()` gives, it warps to the same pixels as `warp_exec()`.

* `warp_resample()` is deprecated in favour of `warp_buffer()`. It is
  integer-only, `"near"` or `"bilinear"`, one band, and returns a row-major
  vector.

* `warp_plan()` documents `fill_ratio`: GDAL's, which can be above 1.

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
