## warp_exec() against gdalwarp (via gdalraster) on synthetic sources: a
## global lon/lat grid warped to a coarser Antarctic polar stereographic
## grid, so the kernels widen and the plan has to find the pole (GDAL's
## special points). gdalwarp is the oracle; the comparison is only as exact
## as the GDAL build allows, so integer and byte data must match exactly and
## doubles to rounding.

nc <- 360L; nr <- 180L
src_gt  <- c(-180, 1, 0, 90, 0, -1)
src_crs <- "EPSG:4326"
dst_crs <- "EPSG:3031"
dst_dim <- c(48L, 40L)
dst_ext <- c(-4e6, 4e6, -3.5e6, 3.5e6)
dst_gt  <- c(dst_ext[1], diff(dst_ext[1:2]) / dst_dim[1], 0,
             dst_ext[4], 0, -diff(dst_ext[3:4]) / dst_dim[2])

field <- function(b = 1L) {
  xy <- expand.grid(x = seq_len(nc), y = seq_len(nr))
  set.seed(b)
  100 * sin(xy$x / (15 * b)) * cos(xy$y / (11 + b)) + stats::rnorm(nc * nr, sd = 20)
}

make_src <- function(vals, dt) {
  f <- tempfile(fileext = ".tif")
  gdalraster::create("GTiff", f, nc, nr, length(vals), dt, options = "COMPRESS=NONE")
  ds <- methods::new(gdalraster::GDALRaster, f, read_only = FALSE)
  ds$setGeoTransform(src_gt)
  ds$setProjection(gdalraster::srs_to_wkt(src_crs))
  for (b in seq_along(vals)) ds$write(b, 0, 0, nc, nr, vals[[b]])
  ds$close()
  f
}

gdal_ref <- function(f, alg, extra = character()) {
  out <- tempfile(fileext = ".tif")
  gdalraster::warp(f, out, t_srs = dst_crs, quiet = TRUE, cl_arg = c(
    "-te", dst_ext[c(1, 3, 2, 4)], "-ts", dst_dim, "-r", alg, extra))
  ds <- methods::new(gdalraster::GDALRaster, out)
  on.exit(ds$close())
  matrix(ds$read(1, 0, 0, dst_dim[1], dst_dim[2], dst_dim[1], dst_dim[2]),
         nrow = dst_dim[2], byrow = TRUE)
}

rwarp_ref <- function(f, alg, conv, scaling = "widen") {
  ds <- methods::new(gdalraster::GDALRaster, f)
  on.exit(ds$close())
  read_fn <- function(xoff, yoff, xsize, ysize) {
    conv(ds$read(1, xoff, yoff, xsize, ysize, xsize, ysize))
  }
  plan <- warp_plan(src_crs, src_gt, c(nc, nr), dst_crs, dst_gt,
                    dst_size = dst_dim, resample = alg, scaling = scaling)
  warp_exec(plan, read_fn, src_crs, src_gt, dst_crs, dst_gt,
            src_nodata = NULL, dst_nodata = NULL)
}

kernels <- c("near", "bilinear", "cubic", "cubicspline", "lanczos",
             "average", "mode", "min", "max", "med", "rms")

test_that("the plan finds the pole inside the destination (special points)", {
  plan <- warp_plan(src_crs, src_gt, c(nc, nr), dst_crs, dst_gt,
                    dst_size = dst_dim, resample = "lanczos")
  ch <- plan[[1]][[1]]
  ## the destination reaches the South Pole: the window runs to the last row
  expect_equal(ch$src_yoff + ch$src_ysize, nr)
  expect_lt(ch$scale_x, 1)
  expect_identical(attr(plan, "resample"), "lanczos")
})

test_that("Int16: every kernel matches gdalwarp exactly", {
  skip_if_not_installed("gdalraster")
  f <- make_src(list(round(field())), "Int16")
  for (alg in kernels) {
    expect_identical(rwarp_ref(f, alg, as.integer),
                     matrix(as.integer(gdal_ref(f, alg)), nrow = dst_dim[2]),
                     label = alg)
  }
})

test_that("Byte (raw): every kernel matches gdalwarp exactly", {
  skip_if_not_installed("gdalraster")
  f <- make_src(list(pmin(255, pmax(0, round(field() + 128)))), "Byte")
  for (alg in kernels) {
    r <- rwarp_ref(f, alg, as.raw)
    expect_type(r, "raw")
    expect_identical(array(as.integer(r), dim(r)),
                     matrix(as.integer(gdal_ref(f, alg)), nrow = dst_dim[2]),
                     label = alg)
  }
})

test_that("Float64: every kernel but mode matches gdalwarp to rounding", {
  skip_if_not_installed("gdalraster")
  f <- make_src(list(field()), "Float64")
  for (alg in setdiff(kernels, "mode")) {
    r <- rwarp_ref(f, alg, as.double)
    expect_type(r, "double")
    expect_lt(max(abs(r - gdal_ref(f, alg))), 1e-9, label = alg)
  }
  ## mode on 32/64-bit and float data: GDAL's tie-break differs from its
  ## 8/16-bit path (an upstream bug candidate); rwarp keeps the 8/16-bit
  ## behaviour for every type, so continuous data (all ties) differs.
})

test_that("scaling = \"point\" matches gdalwarp -wo XSCALE=1 -wo YSCALE=1", {
  skip_if_not_installed("gdalraster")
  f <- make_src(list(field()), "Float64")
  for (alg in c("bilinear", "cubic", "lanczos")) {
    expect_lt(max(abs(rwarp_ref(f, alg, as.double, scaling = "point") -
                        gdal_ref(f, alg, c("-wo", "XSCALE=1", "-wo", "YSCALE=1")))),
              1e-9, label = alg)
  }
})

test_that("multi-band raw data warps to an array and to a nativeRaster", {
  skip_if_not_installed("gdalraster")
  vals <- lapply(1:3, function(b) pmin(255, pmax(0, round(field(b) + 128))))
  f <- make_src(vals, "Byte")
  ds <- methods::new(gdalraster::GDALRaster, f)
  on.exit(ds$close())
  read3 <- function(xoff, yoff, xsize, ysize) {
    as.raw(unlist(lapply(1:3, function(b) ds$read(b, xoff, yoff, xsize, ysize, xsize, ysize))))
  }
  plan <- warp_plan(src_crs, src_gt, c(nc, nr), dst_crs, dst_gt,
                    dst_size = dst_dim, resample = "cubic")
  a <- warp_exec(plan, read3, src_crs, src_gt, dst_crs, dst_gt, nbands = 3L,
                 src_nodata = NULL, dst_nodata = NULL)
  expect_equal(dim(a), c(dst_dim[2], dst_dim[1], 3L))
  for (b in 1:3) {
    g <- gdal_ref(f, "cubic", c("-b", b))
    expect_identical(matrix(as.integer(a[, , b]), nrow = dst_dim[2]),
                     matrix(as.integer(g), nrow = dst_dim[2]), label = paste("band", b))
  }
  n <- warp_exec(plan, read3, src_crs, src_gt, dst_crs, dst_gt, nbands = 3L,
                 src_nodata = NULL, dst_nodata = NULL, output = "nara")
  expect_s3_class(n, "nativeRaster")
  expect_equal(dim(n), c(dst_dim[2], dst_dim[1]))
  ## R | G<<8 | B<<16 | A<<24, opaque
  ## nativeRaster pixels are row-major under dim c(nrow, ncol)
  px <- as.integer(n)[(5L - 1L) * dst_dim[1] + 7L]
  rgba <- c(bitwAnd(px, 255L), bitwAnd(bitwShiftR(px, 8L), 255L),
            bitwAnd(bitwShiftR(px, 16L), 255L), bitwAnd(bitwShiftR(px, 24L), 255L))
  expect_equal(rgba, c(as.integer(a[5, 7, ]), 255L))
  if (requireNamespace("vapour", quietly = TRUE)) {
    v <- vapour::gdal_raster_nara(f, target_crs = dst_crs, target_ext = dst_ext,
                                  target_dim = dst_dim, resample = "cubic")[[1]]
    expect_identical(as.integer(n), as.integer(v))
  }
})

test_that("integer NA is nodata in and out; double NaN likewise", {
  ## a 4x4 lon/lat source with one NA, warped onto itself
  gt <- c(0, 1, 0, 4, 0, -1)
  v  <- 1:16; v[6] <- NA
  plan <- warp_plan("EPSG:4326", gt, c(4L, 4L), "EPSG:4326", gt, dst_size = c(4L, 4L))
  r <- warp_exec(plan, function(...) v, "EPSG:4326", gt, "EPSG:4326", gt)
  expect_identical(r, matrix(v, 4, 4, byrow = TRUE))
  rd <- warp_exec(plan, function(...) as.double(v), "EPSG:4326", gt, "EPSG:4326", gt)
  expect_true(is.na(rd[2, 2]))
  expect_equal(rd, matrix(as.double(v), 4, 4, byrow = TRUE))
})

test_that("raw nara: pixels at dst_nodata are transparent without an alpha band", {
  gt <- c(0, 1, 0, 4, 0, -1)
  plan <- warp_plan("EPSG:4326", gt, c(4L, 4L), "EPSG:4326", c(0, 2, 0, 4, 0, -2),
                    dst_size = c(4L, 4L))
  ## the destination is twice as big: its right and bottom halves see no source
  n <- warp_exec(plan, function(...) as.raw(rep(200L, 16)), "EPSG:4326", gt,
                 "EPSG:4326", c(0, 2, 0, 4, 0, -2), dst_nodata = 0, output = "nara")
  alpha <- bitwAnd(bitwShiftR(as.integer(n), 24L), 255L)
  expect_true(any(alpha == 0L))
  expect_true(any(alpha == 255L))
})

test_that("read_fn errors are clear", {
  gt <- c(0, 1, 0, 4, 0, -1)
  plan <- warp_plan("EPSG:4326", gt, c(4L, 4L), "EPSG:4326", gt, dst_size = c(4L, 4L))
  expect_error(warp_exec(plan, function(...) 1:3, "EPSG:4326", gt, "EPSG:4326", gt),
               "expected 16")
  expect_error(warp_exec(plan, function(...) rep(TRUE, 16), "EPSG:4326", gt, "EPSG:4326", gt),
               "integer, double or raw")
  expect_error(warp_exec(plan, function(...) 1:16, "EPSG:4326", gt, "EPSG:4326", gt,
                         output = "nara"), "raw")
})
