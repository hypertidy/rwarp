## warp_buffer(): one pre-fetched source window, warped with the crate's
## warp(). warp_exec() is the reference (it matches gdalwarp, see
## test-warp-exec.R): a buffer holding the window a plan asks for must warp
## to the same pixels, every kernel, no GDAL needed.

nc <- 360L; nr <- 180L
src_gt  <- c(-180, 1, 0, 90, 0, -1)
src_crs <- "EPSG:4326"
dst_crs <- "EPSG:3031"
dst_dim <- c(48L, 40L)
dst_ext <- c(-4e6, 4e6, -3.5e6, 3.5e6)
dst_gt  <- c(dst_ext[1], diff(dst_ext[1:2]) / dst_dim[1], 0,
             dst_ext[4], 0, -diff(dst_ext[3:4]) / dst_dim[2])

## the source in R orientation, and its windows as read_fn returns them
set.seed(1)
xy <- expand.grid(x = seq_len(nc), y = seq_len(nr))
m  <- matrix(100 * sin(xy$x / 15) * cos(xy$y / 12) + stats::rnorm(nc * nr, sd = 20),
             nrow = nr, byrow = TRUE)
win <- function(src, xoff, yoff, xsize, ysize) {
  src[yoff + seq_len(ysize), xoff + seq_len(xsize), drop = FALSE]
}
reader <- function(src) {
  function(xoff, yoff, xsize, ysize) as.vector(t(win(src, xoff, yoff, xsize, ysize)))
}

kernels <- c("near", "bilinear", "cubic", "cubicspline", "lanczos",
             "average", "mode", "min", "max", "med", "rms")

test_that("the plan's window warps as warp_exec() does, every kernel, double and integer", {
  for (src in list(m, round(m) + 0L)) {
    for (alg in kernels) {
      plan <- warp_plan(src_crs, src_gt, c(nc, nr), dst_crs, dst_gt,
                        dst_size = dst_dim, resample = alg)
      expect_length(plan[[1]], 1L)
      ch <- plan[[1]][[1]]
      ref <- warp_exec(plan, reader(src), src_crs, src_gt, dst_crs, dst_gt,
                       src_nodata = NULL, dst_nodata = NULL)
      w <- win(src, ch$src_xoff, ch$src_yoff, ch$src_xsize, ch$src_ysize)
      ## a matrix, with the kernel scale estimated over the window
      a <- warp_buffer(w, src_crs, src_gt, dst_crs, dst_gt, dst_size = dst_dim,
                       src_off = c(ch$src_xoff, ch$src_yoff), src_dim = c(nc, nr),
                       resample = alg, src_nodata = NULL, dst_nodata = NULL)
      expect_identical(a, ref, label = paste(typeof(src), alg))
      ## a row-major vector, with the plan's kernel scale
      b <- warp_buffer(as.vector(t(w)), src_crs, src_gt, dst_crs, dst_gt,
                       dst_size = dst_dim, src_off = c(ch$src_xoff, ch$src_yoff),
                       src_size = c(ch$src_xsize, ch$src_ysize), src_dim = c(nc, nr),
                       resample = alg, scaling = "fixed", scale = c(ch$scale_x, ch$scale_y),
                       src_nodata = NULL, dst_nodata = NULL)
      expect_identical(b, ref, label = paste(typeof(src), alg, "fixed"))
    }
  }
})

test_that("several bands: an array and a band-sequential vector agree, band by band", {
  bands <- list(round(m), round(t(apply(m, 1, rev))), round(-m))
  arr <- array(unlist(bands), c(nr, nc, 3L))
  vec <- unlist(lapply(bands, function(b) as.vector(t(b))))
  a <- warp_buffer(arr, src_crs, src_gt, dst_crs, dst_gt, dst_size = dst_dim,
                   src_dim = c(nc, nr), resample = "cubic")
  v <- warp_buffer(vec, src_crs, src_gt, dst_crs, dst_gt, dst_size = dst_dim,
                   src_size = c(nc, nr), src_dim = c(nc, nr), resample = "cubic")
  expect_equal(dim(a), c(dst_dim[2], dst_dim[1], 3L))
  expect_identical(a, v)
  for (b in 1:3) {
    one <- warp_buffer(bands[[b]], src_crs, src_gt, dst_crs, dst_gt, dst_size = dst_dim,
                       src_dim = c(nc, nr), resample = "cubic")
    expect_identical(a[, , b], one, label = paste("band", b))
  }
})

test_that("a destination window at dst_off is that part of the whole", {
  ## with a kernel scale that doesn't depend on the window: "widen" estimates
  ## it over the window warped, so a part of this polar grid widens less
  ## (0.32 against 0.45 for the whole)
  for (alg in c("near", "bilinear", "cubic")) {
    whole <- warp_buffer(m, src_crs, src_gt, dst_crs, dst_gt, dst_size = dst_dim,
                         src_dim = c(nc, nr), resample = alg, scaling = "point")
    part <- warp_buffer(m, src_crs, src_gt, dst_crs, dst_gt, dst_size = c(20L, 15L),
                        dst_off = c(10L, 5L), src_dim = c(nc, nr), resample = alg,
                        scaling = "point")
    expect_identical(part, whole[5 + 1:15, 10 + 1:20], label = alg)
  }
})

test_that("integer NA is nodata in and out; double NaN likewise", {
  gt <- c(0, 1, 0, 4, 0, -1)
  v  <- 1:16; v[6] <- NA
  r <- warp_buffer(v, "EPSG:4326", gt, "EPSG:4326", gt, dst_size = c(4L, 4L),
                   src_size = c(4L, 4L))
  expect_identical(r, matrix(v, 4, 4, byrow = TRUE))
  rd <- warp_buffer(as.double(v), "EPSG:4326", gt, "EPSG:4326", gt,
                    dst_size = c(4L, 4L), src_size = c(4L, 4L), resample = "average")
  expect_equal(rd, matrix(as.double(v), 4, 4, byrow = TRUE))
  ## a destination twice the size: where there's no source, dst_nodata
  big <- warp_buffer(1:16, "EPSG:4326", gt, "EPSG:4326", c(0, 1, 0, 4, 0, -1),
                     dst_size = c(8L, 8L), src_size = c(4L, 4L), dst_nodata = -1L)
  expect_identical(big[1:4, 1:4], matrix(1:16, 4, 4, byrow = TRUE))
  expect_true(all(big[5:8, ] == -1L))
})

test_that("raw data gives raw, and a nativeRaster with output = \"nara\"", {
  gt <- c(0, 1, 0, 4, 0, -1)
  rgb <- as.raw(c(rep(10L, 16), rep(20L, 16), rep(30L, 16)))
  a <- warp_buffer(rgb, "EPSG:4326", gt, "EPSG:4326", gt, dst_size = c(4L, 4L),
                   src_size = c(4L, 4L))
  expect_type(a, "raw")
  expect_equal(dim(a), c(4L, 4L, 3L))
  n <- warp_buffer(rgb, "EPSG:4326", gt, "EPSG:4326", gt, dst_size = c(4L, 4L),
                   src_size = c(4L, 4L), output = "nara")
  expect_s3_class(n, "nativeRaster")
  ## R | G<<8 | B<<16 | A<<24, opaque
  px <- as.integer(n)[1]
  rgba <- c(bitwAnd(px, 255L), bitwAnd(bitwShiftR(px, 8L), 255L),
            bitwAnd(bitwShiftR(px, 16L), 255L), bitwAnd(bitwShiftR(px, 24L), 255L))
  expect_equal(rgba, c(10L, 20L, 30L, 255L))
})

test_that("errors are clear", {
  gt <- c(0, 1, 0, 4, 0, -1)
  wb <- function(src, ...) warp_buffer(src, "EPSG:4326", gt, "EPSG:4326", gt,
                                       dst_size = c(4L, 4L), ...)
  expect_error(wb(1:16), "src_size")
  expect_error(wb(1:15, src_size = c(4L, 4L)), "whole number")
  expect_error(wb(rep(TRUE, 16), src_size = c(4L, 4L)), "integer, double or raw")
  expect_error(wb(matrix(1:16, 4), src_size = c(2L, 8L)), "dim\\(src\\)")
  expect_error(wb(1:16, src_size = c(4L, 4L), output = "nara"), "raw")
  expect_error(wb(1:16, src_size = c(4L, 4L), scaling = "fixed"), "scale")
  expect_error(wb(1:16, src_size = c(4L, 4L), mask_band = 2L), "mask_band")
})
