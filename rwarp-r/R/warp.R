#' Map destination pixel centres to fractional source pixel coordinates
#'
#' Implements GDAL's `GenImgProjTransform`: for every cell centre of the
#' destination grid, compute the corresponding fractional (sub-pixel) position
#' in the source raster via the three-step pipeline
#' `dst pixel -> dst geo -> PROJ -> src geo -> src pixel`.
#'
#' This is the low-level building block used by [warp_plan()] and
#' [warp_resample()].  It is exposed here so you can inspect the coordinate
#' mapping directly — useful for debugging CRS strings, checking extent
#' coverage, and understanding fill ratios.
#'
#' @param src_crs CRS of the source raster (WKT, EPSG code, PROJ string with
#'   `+type=crs`, ...).
#' @param src_gt Geotransform of the source raster: a 6-element numeric vector
#'   `c(xmin, xres, 0, ymax, 0, -yres)` (north-up convention, as returned by
#'   `GDALRaster$getGeoTransform()` or `vaster::extent_dim_to_gt()`).
#' @param dst_crs CRS of the destination raster.
#' @param dst_gt Geotransform of the destination raster.
#' @param dst_dim Integer vector `c(ncol, nrow)` of the destination grid.
#'
#' @return A list with numeric vectors `src_x` and `src_y` of length
#'   `prod(dst_dim)`, in row-major order.  `NA` where the destination point
#'   transforms to outside the PROJ domain or behind the horizon.
#'
#' @seealso [warp_plan()], [warp_resample()]
#' @export
#' @examples
#' \dontrun{
#' library(vaster)
#' src_gt  <- extent_dim_to_gt(c(-180, 180, -90, 90), c(3600L, 1800L))
#' dst_ext <- c(-20037508, 20037508, -20037508, 20037508)
#' dst_gt  <- extent_dim_to_gt(dst_ext, c(256L, 256L))
#'
#' coords <- pixel_transform(
#'   src_crs = "EPSG:4326", src_gt = src_gt,
#'   dst_crs = "EPSG:3857", dst_gt = dst_gt,
#'   dst_dim = c(256L, 256L)
#' )
#' # Proportion of destination pixels with valid source coordinates
#' mean(!is.na(coords$src_x))
#' }
pixel_transform <- function(src_crs, src_gt, dst_crs, dst_gt, dst_dim) {
  stopifnot(
    is.character(src_crs), length(src_crs) == 1L,
    is.numeric(src_gt),    length(src_gt)  == 6L,
    is.character(dst_crs), length(dst_crs) == 1L,
    is.numeric(dst_gt),    length(dst_gt)  == 6L,
    is.integer(dst_dim),   length(dst_dim) == 2L
  )
  .Call(wrap__pixel_transform, src_crs, src_gt, dst_crs, dst_gt, dst_dim)
}

#' Plan the source windows needed for a destination raster window
#'
#' Computes all source windows required to fill a destination grid window,
#' accounting for antimeridian splits and minimum fill ratios.  The returned
#' plan can be inspected before any I/O is performed, and the execution step
#' [warp_exec()] accepts any read function — GDAL, byte-ref COG tiles, Zarr,
#' etc.
#'
#' @param src_crs,src_gt,src_dim CRS, geotransform, and `c(ncol, nrow)` of the
#'   source raster.
#' @param dst_crs,dst_gt CRS and geotransform of the destination raster.
#' @param dst_off Integer `c(col_off, row_off)` origin of the destination
#'   window (0-based).  Default `c(0L, 0L)`.
#' @param dst_size Integer `c(ncol, nrow)` of the destination window.
#' @param resample_padding Extra source pixels added to every side of each
#'   source window (gdalwarp's `-wo SOURCE_EXTRA`).  `NULL` (the default) is
#'   gdalwarp's rule: none, or 10 when a sample point fails to transform.
#'   The kernel's own radius is always added, from `resample`.
#' @param min_fill_ratio Split a chunk with fill above 0 and below this
#'   fraction.  Default `0` (never split on fill).
#' @param min_dst_size Only split a chunk with a side longer than this, in
#'   pixels.  Default `1L`.
#' @param resample Resampling method the plan is for: one of `"near"`,
#'   `"bilinear"`, `"cubic"`, `"cubicspline"`, `"lanczos"`, `"average"`,
#'   `"mode"`, `"min"`, `"max"`, `"med"`, `"rms"` (gdalwarp's `-r` names).
#'   Source windows are padded by the kernel's radius, times the kernel
#'   scale when downsampling.  [warp_exec()] uses it by default.
#' @param scaling How the kernel is widened when downsampling:
#'   `"widen"` (gdalwarp's default: by the destination/source scale,
#'   estimated per chunk), `"point"` (never: gdalwarp
#'   `-wo XSCALE=1 -wo YSCALE=1`) or `"fixed"` (by `scale`, as gdalwarp
#'   `-wo XSCALE= -wo YSCALE=`).
#' @param scale With `scaling = "fixed"`, `c(x, y)`: destination pixels per
#'   source pixel (below 1 is downsampling).
#' @param split_antimeridian When a chunk's source window spans the whole
#'   source width (gdalwarp's 90% rule, as when the destination crosses the
#'   antimeridian of a global longitude/latitude source), split the chunk
#'   where source x jumps, into compact windows either side. That reads the
#'   two edges of the source instead of a full-width strip (the few columns
#'   at the jump read a small window at each edge). gdalwarp does not split.
#'   Default `TRUE`; `FALSE` is gdalwarp's plan.
#'
#' @return A list holding one chunk list, itself a list of chunk objects with
#'   fields:
#'   `dst_xoff`, `dst_yoff`, `dst_xsize`, `dst_ysize`,
#'   `src_xoff`, `src_yoff`, `src_xsize`, `src_ysize`,
#'   `fill_ratio`, `scale_x`, `scale_y` (the kernel scale for the chunk).
#'   Attributes `src_dim`, `dst_off`, `dst_size`, `resample` and `scaling` record what
#'   it was planned for.
#'   Returns `NULL` when no valid chunks are found (source does not overlap
#'   the destination window).
#'
#' @seealso [warp_exec()], [warp_resample()]
#' @export
#' @examples
#' \dontrun{
#' library(vaster)
#'
#' src_gt  <- extent_dim_to_gt(c(-180, 180, -90, 90), c(36000L, 18000L))
#' dst_ext <- c(-20037508, 20037508, -20037508, 20037508)
#' dst_gt  <- extent_dim_to_gt(dst_ext, c(512L, 512L))
#'
#' plan <- warp_plan(
#'   src_crs  = "EPSG:4326",
#'   src_gt   = src_gt,
#'   src_dim  = c(36000L, 18000L),
#'   dst_crs  = "EPSG:3857",
#'   dst_gt   = dst_gt,
#'   dst_size = c(512L, 512L)
#' )
#' length(plan)               # 1 normally; 2 at the antimeridian
#' plan[[1]][[1]]$fill_ratio  # how much of the dst tile is covered
#' }
warp_plan <- function(
    src_crs, src_gt, src_dim,
    dst_crs, dst_gt,
    dst_off  = c(0L, 0L),
    dst_size,
    resample_padding = NULL,
    min_fill_ratio   = 0,
    min_dst_size     = 1L,
    resample         = "near",
    scaling          = c("widen", "point", "fixed"),
    scale            = NULL,
    split_antimeridian = TRUE) {
  resample <- match.arg(resample, c("near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode", "min", "max", "med", "rms"))
  scaling  <- match.arg(scaling)
  if (scaling == "fixed" && (is.null(scale) || length(scale) != 2L)) {
    stop("scaling = \"fixed\" needs scale = c(x, y)")
  }
  stopifnot(
    is.character(src_crs), length(src_crs) == 1L,
    is.numeric(src_gt),    length(src_gt)  == 6L,
    is.integer(src_dim),   length(src_dim) == 2L,
    is.character(dst_crs), length(dst_crs) == 1L,
    is.numeric(dst_gt),    length(dst_gt)  == 6L,
    is.integer(dst_off),   length(dst_off) == 2L,
    is.integer(dst_size),  length(dst_size) == 2L
  )
  plan <- .Call(wrap__warp_plan,
    src_crs, src_gt, src_dim,
    dst_crs, dst_gt,
    dst_off, dst_size,
    if (is.null(resample_padding)) -1L else as.integer(resample_padding),
    as.double(min_fill_ratio),
    as.integer(min_dst_size),
    resample, scaling, as.double(if (is.null(scale)) numeric(0) else scale),
    isTRUE(split_antimeridian)
  )
  ## NULL (no chunk reads source) stays NULL, as before
  if (is.null(plan)) return(NULL)
  attr(plan, "src_dim")  <- src_dim
  attr(plan, "dst_off")  <- dst_off
  attr(plan, "dst_size") <- dst_size
  attr(plan, "resample") <- resample
  attr(plan, "scaling")  <- scaling
  plan
}

#' Execute a warp plan using a user-supplied read function
#'
#' Runs the chunk plan returned by [warp_plan()]: reads each chunk's source
#' window with `read_fn`, warps it with rwarp's kernels, and places it in the
#' output.  The loop runs in Rust and calls `read_fn` once per chunk.
#'
#' The data type follows what `read_fn` returns: an integer vector warps as
#' 32-bit integers, a double vector as 64-bit floats, a raw vector as bytes.
#' The output is the same type.
#'
#' @param plan Chunk plan as returned by [warp_plan()].  A chunk without
#'   `scale_x`/`scale_y` (a plan made by hand) uses a kernel scale of 1.
#' @param read_fn A function `f(xoff, yoff, xsize, ysize)` returning the
#'   source window as one vector of length `xsize * ysize * nbands`: each band
#'   row-major (as `GDALRaster$read()` gives it), bands one after another.
#'   Integer, double or raw.  Can call GDAL, a byte-ref tile reader, a Zarr
#'   chunk fetcher, or anything else.
#' @param src_crs,src_gt Source CRS and geotransform (as passed to
#'   [warp_plan()]).
#' @param dst_crs,dst_gt Destination CRS and geotransform of the full grid
#'   (as passed to [warp_plan()]).
#' @param dst_size Integer `c(ncol, nrow)` of the planned window.  Default:
#'   the plan's.
#' @param nbands Number of bands `read_fn` returns.  Default `1L`.
#' @param resample Resampling method (see [warp_plan()]).  Default: the
#'   method the plan was made for.  Planning for one method and warping with
#'   a wider one can clip kernels at window edges.
#' @param src_nodata Source nodata, one value or one per band.  `NA` (the
#'   default) means R's `NA` for integer data; NaN is always nodata in double
#'   data; raw has none.  `NULL` for none.
#' @param dst_nodata Value written where no source contributes, one value or
#'   one per band.  `NA` (the default) writes `NA` (integer and double) or 0
#'   (raw).  `NULL` writes 0.
#' @param band_nodata How nodata works across bands (gdalwarp
#'   `UNIFIED_SRC_NODATA`): `"partial"` (gdalwarp's default), `"unified"`, or
#'   `"per_band"`.
#' @param mask_band A band (1-based) whose zeros mark source pixels invalid,
#'   such as an alpha band.  `NULL` for none.
#' @param mode_ties For `resample = "mode"`, how ties break: `"first"`,
#'   `"min"` or `"max"` (gdalwarp `-wo MODE_TIES=`).
#' @param max_error Approx-transformer pixel error bound.  Default `0.125`
#'   (gdalwarp's `-et`).
#' @param output `"array"` (default): a matrix `c(nrow, ncol)` for one band,
#'   an array `c(nrow, ncol, nbands)` for more.  `"nara"`: a `nativeRaster`
#'   for raw data of 1 (grey), 3 (RGB) or 4 (RGBA) bands, ready for
#'   `plot()`, `grid::grid.raster()` or `png::writePNG()`.  Without an alpha
#'   band, pixels whose bands all equal `dst_nodata` are transparent.
#' @param verbose Print a line per chunk.  Default `FALSE`.
#' @param nodata Deprecated: sets both `src_nodata` and `dst_nodata`.
#'
#' @return See `output`.  The matrix/array is in standard R orientation
#'   (row 1 at the top).
#'
#' @seealso [warp_plan()], [warp_resample()]
#' @export
#' @examples
#' \dontrun{
#' library(gdalraster)
#' library(vaster)
#'
#' ## Sentinel-2 Band 4 COG
#' url <- paste0(
#'   "/vsicurl/https://e84-earth-search-sentinel-data.s3.us-west-2.",
#'   "amazonaws.com/sentinel-2-c1-l2a/55/G/CM/2026/2/",
#'   "S2C_T55GCM_20260227T000650_L2A/B04.tif"
#' )
#' ds <- new(GDALRaster, url, read_only = TRUE)
#'
#' src_crs <- ds$getProjectionRef()
#' src_gt  <- ds$getGeoTransform()
#' src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())
#'
#' dst_crs  <- "EPSG:4326"
#' dst_ext  <- reproj::reproj_extent(ds$bbox()[c(1,3,2,4)], "EPSG:4326", source = src_crs)
#' dst_dim  <- c(512L, 512L)
#' dst_gt   <- extent_dim_to_gt(dst_ext, dst_dim)
#'
#' read_fn <- function(xoff, yoff, xsize, ysize) {
#'   as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize))
#' }
#'
#' plan   <- warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt,
#'                     dst_size = dst_dim, resample = "cubic")
#' result <- warp_exec(plan, read_fn, src_crs, src_gt, dst_crs, dst_gt)
#'
#' ximage::ximage(result, dst_ext, col = grey.colors(256))
#' ds$close()
#' }
warp_exec <- function(
    plan, read_fn,
    src_crs, src_gt,
    dst_crs, dst_gt,
    dst_size    = attr(plan, "dst_size"),
    nbands      = 1L,
    resample    = attr(plan, "resample"),
    src_nodata  = NA,
    dst_nodata  = NA,
    band_nodata = c("partial", "unified", "per_band"),
    mask_band   = NULL,
    mode_ties   = c("first", "min", "max"),
    max_error   = 0.125,
    output      = c("array", "nara"),
    verbose     = FALSE,
    nodata      = NULL) {
  if (!is.null(nodata)) {
    .Deprecated(msg = "warp_exec(nodata = ) is deprecated; use src_nodata and dst_nodata")
    src_nodata <- dst_nodata <- nodata
  }
  band_nodata <- match.arg(band_nodata)
  mode_ties   <- match.arg(mode_ties)
  output      <- match.arg(output)
  if (is.null(resample)) resample <- "near"
  resample <- match.arg(resample, c("near", "bilinear", "cubic", "cubicspline",
                                    "lanczos", "average", "mode", "min", "max",
                                    "med", "rms"))
  if (is.null(dst_size)) stop("dst_size is needed (the plan has no dst_size attribute)")
  dst_off <- attr(plan, "dst_off")
  if (is.null(dst_off)) dst_off <- c(0L, 0L)
  stopifnot(is.function(read_fn), length(dst_size) == 2L)

  chunks <- unlist(plan, recursive = FALSE)
  geom <- vapply(chunks, function(ch) as.integer(c(
    ch$dst_xoff, ch$dst_yoff, ch$dst_xsize, ch$dst_ysize,
    ch$src_xoff, ch$src_yoff, ch$src_xsize, ch$src_ysize)), integer(8L))
  ## a chunk without a kernel scale (a plan made by hand) uses 1: no widening
  scales <- vapply(chunks, function(ch) c(
    if (is.null(ch$scale_x)) 1 else ch$scale_x,
    if (is.null(ch$scale_y)) 1 else ch$scale_y), numeric(2L))

  nd <- function(x) if (is.null(x)) numeric(0) else as.double(x)
  .Call(wrap__warp_exec_rs,
    as.integer(geom), as.double(scales),
    read_fn,
    src_crs, as.double(src_gt),
    if (is.null(attr(plan, "src_dim"))) integer(0) else as.integer(attr(plan, "src_dim")),
    dst_crs, as.double(dst_gt),
    as.integer(dst_off), as.integer(dst_size),
    as.integer(nbands),
    as.double(max_error),
    resample, mode_ties,
    nd(src_nodata), nd(dst_nodata),
    band_nodata,
    if (is.null(mask_band)) 0L else as.integer(mask_band),
    output == "nara",
    isTRUE(verbose)
  )
}

#' Warp and resample a source pixel buffer into a destination grid
#'
#' Low-level kernel.  Reprojects a pre-fetched source window into the
#' destination grid.  In most workflows you will call [warp_exec()] instead,
#' which handles the planning loop and compositing automatically.
#'
#' The source buffer must be a **row-major integer vector**.
#' Use `as.integer(as.vector(t(m)))` to convert from an R matrix `m`.
#'
#' @param src_crs,src_gt Source CRS and geotransform.
#' @param dst_crs,dst_gt Destination CRS and geotransform.
#' @param dst_dim Integer `c(ncol, nrow)` of the destination window.
#' @param src_pixels Integer vector (row-major) of source values.
#' @param src_ncol,src_nrow Dimensions of the source buffer.
#' @param src_col_off,src_row_off 0-based offset of the source buffer within
#'   the full source raster.
#' @param nodata No-data integer.  Default `-32768L`.
#' @param max_error Approx-transformer pixel error bound.  Default `0.125`.
#' @param resample `"near"` (default) or `"bilinear"`.
#'
#' @return Integer vector of length `prod(dst_dim)`, row-major.
#'   Convert to a matrix with
#'   `matrix(result, nrow = dst_dim[2], ncol = dst_dim[1], byrow = TRUE)`.
#'
#' @seealso [warp_plan()], [warp_exec()]
#' @export
warp_resample <- function(
    src_crs, src_gt,
    dst_crs, dst_gt, dst_dim,
    src_pixels,
    src_ncol, src_nrow,
    src_col_off, src_row_off,
    nodata    = -32768L,
    max_error = 0.125,
    resample  = "near") {

  stopifnot(
    is.integer(src_pixels),
    is.integer(dst_dim), length(dst_dim) == 2L
  )
  .Call(wrap__warp_resample,
    src_crs, as.double(src_gt),
    dst_crs, as.double(dst_gt), as.integer(dst_dim),
    src_pixels,
    as.integer(src_ncol), as.integer(src_nrow),
    as.integer(src_col_off), as.integer(src_row_off),
    as.integer(nodata), as.double(max_error),
    as.character(resample)
  )
}
