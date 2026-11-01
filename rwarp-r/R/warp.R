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
#' @param resample_padding Extra source pixels of context for interpolating
#'   kernels (1 for bilinear).  Default `1L`.
#' @param min_fill_ratio Split a chunk with fill above 0 and below this
#'   fraction.  Default `0` (never split on fill).
#' @param min_dst_size Only split a chunk with a side longer than this, in
#'   pixels.  Default `1L`.
#'
#' @return A list holding one chunk list, itself a list of chunk objects with
#'   fields:
#'   `dst_xoff`, `dst_yoff`, `dst_xsize`, `dst_ysize`,
#'   `src_xoff`, `src_yoff`, `src_xsize`, `src_ysize`,
#'   `fill_ratio`.
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
    resample_padding = 1L,
    min_fill_ratio   = 0,
    min_dst_size     = 1L) {
  stopifnot(
    is.character(src_crs), length(src_crs) == 1L,
    is.numeric(src_gt),    length(src_gt)  == 6L,
    is.integer(src_dim),   length(src_dim) == 2L,
    is.character(dst_crs), length(dst_crs) == 1L,
    is.numeric(dst_gt),    length(dst_gt)  == 6L,
    is.integer(dst_off),   length(dst_off) == 2L,
    is.integer(dst_size),  length(dst_size) == 2L
  )
  .Call(wrap__warp_plan,
    src_crs, src_gt, src_dim,
    dst_crs, dst_gt,
    dst_off, dst_size,
    as.integer(resample_padding),
    as.double(min_fill_ratio),
    as.integer(min_dst_size)
  )
}

#' Execute a warp plan using a user-supplied read function
#'
#' Iterates over the chunk plan returned by [warp_plan()], reads each source
#' window via `read_fn`, reprojects the pixels with [warp_resample()], and
#' composites the results ("first valid wins") into an output integer matrix.
#'
#' @param plan Chunk plan as returned by [warp_plan()].
#' @param read_fn A function `f(xoff, yoff, xsize, ysize)` that returns an
#'   **integer** vector in row-major order of length `xsize * ysize`.  The
#'   simplest implementation uses `GDALRaster$read()`.  Can also call a
#'   byte-ref tile reader, a Zarr chunk fetcher, or any other source.
#' @param src_crs,src_gt Source CRS and geotransform (same as passed to
#'   [warp_plan()]).
#' @param dst_crs,dst_gt Destination CRS and geotransform.
#' @param dst_size Integer `c(ncol, nrow)` of the full destination window.
#' @param nodata Integer no-data value.  Default `-32768L`.
#' @param max_error Approx-transformer pixel error bound.  Default `0.125`.
#' @param resample Resampling method: `"near"` (default) or `"bilinear"`.
#' @param verbose Print progress per chunk.  Default `FALSE`.
#'
#' @return Integer matrix of dimensions `c(dst_size[2], dst_size[1])` (rows,
#'   cols) in standard R matrix orientation.  Composite rule: first non-nodata
#'   value wins; use `result[result == nodata] <- NA` to mask.
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
#' ## Read function using GDAL
#' read_fn <- function(xoff, yoff, xsize, ysize) {
#'   as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize))
#' }
#'
#' plan   <- warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt, dst_size = dst_dim)
#' result <- warp_exec(plan, read_fn, src_crs, src_gt, dst_crs, dst_gt, dst_dim)
#'
#' ximage::ximage(result, dst_ext, col = grey.colors(256))
#' ds$close()
#' }
warp_exec <- function(
    plan, read_fn,
    src_crs, src_gt,
    dst_crs, dst_gt,
    dst_size,
    nodata    = -32768L,
    max_error = 0.125,
    resample  = "near",
    verbose   = FALSE) {

  ncol <- dst_size[1L]
  nrow <- dst_size[2L]
  out  <- matrix(nodata, nrow = nrow, ncol = ncol)

  for (chunk_list in plan) {
    for (chunk in chunk_list) {
      if (verbose) {
        message(sprintf(
          "chunk: src [%d,%d] %dx%d | fill %.3f",
          chunk$src_xoff, chunk$src_yoff,
          chunk$src_xsize, chunk$src_ysize,
          chunk$fill_ratio
        ))
      }
      src_buf <- read_fn(
        chunk$src_xoff, chunk$src_yoff,
        chunk$src_xsize, chunk$src_ysize
      )
      if (!is.integer(src_buf)) src_buf <- as.integer(src_buf)

      ## warp_resample() fills a grid from pixel 0, so shift dst_gt to the chunk
      chunk_gt <- dst_gt
      chunk_gt[1L] <- dst_gt[1L] + chunk$dst_xoff * dst_gt[2L] + chunk$dst_yoff * dst_gt[3L]
      chunk_gt[4L] <- dst_gt[4L] + chunk$dst_xoff * dst_gt[5L] + chunk$dst_yoff * dst_gt[6L]

      warped <- warp_resample(
        src_crs = src_crs, src_gt = src_gt,
        dst_crs = dst_crs, dst_gt = chunk_gt,
        dst_dim = as.integer(c(chunk$dst_xsize, chunk$dst_ysize)),
        src_pixels  = src_buf,
        src_ncol    = chunk$src_xsize,
        src_nrow    = chunk$src_ysize,
        src_col_off = chunk$src_xoff,
        src_row_off = chunk$src_yoff,
        nodata      = nodata,
        max_error   = max_error,
        resample    = resample
      )
      ## Place the warped tile into the output matrix (first valid wins)
      tile_m <- matrix(warped,
        nrow = chunk$dst_ysize, ncol = chunk$dst_xsize,
        byrow = TRUE
      )
      dst_rows <- chunk$dst_yoff + seq_len(chunk$dst_ysize)
      dst_cols <- chunk$dst_xoff + seq_len(chunk$dst_xsize)
      existing <- out[dst_rows, dst_cols, drop = FALSE]
      mask_new <- !is.na(tile_m) & tile_m != nodata
      mask_fill <- mask_new & (is.na(existing) | existing == nodata)
      existing[mask_fill] <- tile_m[mask_fill]
      out[dst_rows, dst_cols] <- existing
    }
  }
  out
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
