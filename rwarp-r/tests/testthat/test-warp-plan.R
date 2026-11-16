## Basic geometry tests for warp_plan and pixel_transform.
## These use tiny grids and no-network sources so they pass in CI.

library(vaster)

## A 1-degree-per-pixel global geographic source
src_gt  <- c(-180, 1, 0, 90, 0, -1)
src_dim <- c(360L, 180L)
src_crs <- "EPSG:4326"

## A tiny 10x10 destination in the same CRS (identity warp)
dst_crs <- "EPSG:4326"
dst_gt  <- c(0, 1, 0, 10, 0, -1)
dst_dim <- c(10L, 10L)

test_that("warp_plan returns a list for overlapping grids", {
  plan <- warp_plan(
    src_crs = src_crs, src_gt = src_gt, src_dim = src_dim,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_off  = c(0L, 0L), dst_size = dst_dim
  )
  expect_type(plan, "list")
  expect_gte(length(plan), 1L)
  ch <- plan[[1]][[1]]
  expect_true(ch$fill_ratio > 0)
})

test_that("warp_plan returns NULL for non-overlapping grids", {
  ## Put destination grid in the Pacific (170-180 E), source is -1 to 10 lon
  dst_gt2 <- c(170, 1, 0, 10, 0, -1)
  plan <- warp_plan(
    src_crs = "EPSG:4326", src_gt = c(-1, 1, 0, 10, 0, -1),
    src_dim = c(10L, 10L),
    dst_crs = "EPSG:4326", dst_gt = dst_gt2,
    dst_off = c(0L, 0L), dst_size = c(10L, 10L)
  )
  expect_null(plan)
})

test_that("pixel_transform returns src_x and src_y", {
  coords <- pixel_transform(
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_dim = dst_dim
  )
  expect_named(coords, c("src_x", "src_y"))
  expect_length(coords$src_x, prod(dst_dim))
  expect_true(all(!is.na(coords$src_x)))
})

test_that("pixel_transform identity warp has expected pixel offsets", {
  ## With identical CRS and geotransform, dst pixel [0,0] -> src pixel [180,80]
  coords <- pixel_transform(
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_dim = dst_dim
  )
  ## dst cell 0,0 centre at (0.5, 9.5 deg) = src pixel (180.5, 80.5) -> 0-based = 180, 80
  expect_equal(coords$src_x[1], 180.5, tolerance = 0.01)
  expect_equal(coords$src_y[1],  80.5, tolerance = 0.01)
})

## The antimeridian: a destination centred on 180 degrees reads the two edges
## of a global lon/lat source, not a full-width strip, and warps to the same
## values (exact transformer).
test_that("split_antimeridian reads two edge windows and gives the same values", {
  dst_crs <- "+proj=laea +lat_0=-17 +lon_0=180"
  dst_gt  <- c(-1.5e6, 7812.5, 0, 1.5e6, 0, -7812.5)
  dst_dim <- c(384L, 384L)
  pl <- function(split, resample) warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt,
                                            dst_size = dst_dim, resample = resample,
                                            split_antimeridian = split)
  full  <- pl(FALSE, "bilinear")[[1]]
  split <- pl(TRUE, "bilinear")[[1]]
  px <- function(p) sum(vapply(p, function(c) c$src_xsize * c$src_ysize, 0))
  expect_length(full, 1L)
  expect_equal(full[[1]]$src_xsize, 360L)
  expect_true(all(vapply(split, function(c) c$src_xsize < 180L, TRUE)))
  expect_lt(px(split), px(full) / 4)
  ## the gap: two chunks on one target window
  key <- vapply(split, function(c) paste(c$dst_xoff, c$dst_yoff, c$dst_xsize, c$dst_ysize), "")
  expect_true(anyDuplicated(key) > 0)

  ## values: x * 1000 + y, so a pixel from the wrong column shows
  src <- outer(seq_len(src_dim[2]) - 1L, (seq_len(src_dim[1]) - 1L) * 1000L, `+`)
  read_fn <- function(xoff, yoff, xsize, ysize) {
    as.double(t(src[yoff + seq_len(ysize), xoff + seq_len(xsize), drop = FALSE]))
  }
  for (rs in c("near", "bilinear", "cubic")) {
    ex <- function(split) warp_exec(pl(split, rs), read_fn, src_crs, src_gt, dst_crs, dst_gt,
                                    dst_nodata = -1, max_error = 0)
    expect_identical(ex(TRUE), ex(FALSE), info = rs)
  }
})

test_that("sizes and offsets can be whole-number doubles, as GDAL bindings give them", {
  p_int <- warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt, dst_size = dst_dim)
  p_dbl <- warp_plan(src_crs, src_gt, as.double(src_dim), dst_crs, dst_gt,
                     dst_off = c(0, 0), dst_size = as.double(dst_dim))
  expect_identical(p_dbl, p_int)
  expect_error(warp_plan(src_crs, src_gt, c(360.5, 180), dst_crs, dst_gt, dst_size = dst_dim),
               "`src_dim` must be two whole numbers")
  expect_error(warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt, dst_off = c(-1, 0),
                         dst_size = dst_dim), "`dst_off`")
  expect_identical(pixel_transform(src_crs, src_gt, dst_crs, dst_gt, c(10, 10)),
                   pixel_transform(src_crs, src_gt, dst_crs, dst_gt, dst_dim))
})

test_that("the working memory a chunk takes sets where the destination splits", {
  ## global 0.1 degree source to a 2000 x 2000 polar grid
  sgt <- c(-180, 0.1, 0, 90, 0, -0.1); sdim <- c(3600L, 1800L)
  dgt <- c(-4e6, 4000, 0, 4e6, 0, -4000); ddim <- c(2000L, 2000L)
  plan <- function(...) warp_plan("EPSG:4326", sgt, sdim, "EPSG:3031", dgt,
                                  dst_size = ddim, resample = "cubic", ...)
  one <- plan()
  ## the defaults: one byte per pixel, which fits in 64 MB
  expect_identical(one, plan(nbands = 1L, working_type = "Byte", src_nodata = FALSE,
                             memory_limit = 64, scale_origin = "chunk"))
  expect_identical(attr(one, "nbands"), 1L)
  ## three Float64 bands with nodata masks: 195 bits a pixel, 97 MB for the
  ## destination alone
  three <- plan(nbands = 3, working_type = "Float64", src_nodata = -9999)
  expect_gt(length(three[[1]]), length(one[[1]]))
  expect_identical(attr(three, "nbands"), 3L)
  ## dst_nodata follows src_nodata unless given
  expect_identical(three, plan(nbands = 3, working_type = "Float64", src_nodata = TRUE,
                               dst_nodata = TRUE))
  ## a larger limit, fewer chunks
  expect_lt(length(plan(nbands = 3, working_type = "Float64", src_nodata = TRUE,
                        memory_limit = 512)[[1]]), length(three[[1]]))
  ## chunk-sized windows at the top-left give equal-sized chunks one scale
  out <- plan(nbands = 3, working_type = "Float64", src_nodata = TRUE, scale_origin = "output")
  sc <- vapply(out[[1]], function(ch) ch$scale_x, 0)
  sz <- vapply(out[[1]], function(ch) paste(ch$dst_xsize, ch$dst_ysize), "")
  expect_true(all(tapply(sc, sz, function(s) diff(range(s))) == 0))

  expect_error(plan(nbands = 0), "`nbands`")
  expect_error(plan(memory_limit = -1), "`memory_limit`")
  expect_error(plan(working_type = "CFloat64"), "should be one of")
})

test_that("warp_exec() takes nbands from the plan", {
  gt <- c(0, 1, 0, 4, 0, -1)
  plan <- warp_plan("EPSG:4326", gt, c(4L, 4L), "EPSG:4326", gt, dst_size = c(4L, 4L),
                    nbands = 2L)
  r <- warp_exec(plan, function(...) as.double(c(1:16, 101:116)), "EPSG:4326", gt,
                 "EPSG:4326", gt)
  expect_equal(dim(r), c(4L, 4L, 2L))
  expect_equal(r[, , 2], matrix(as.double(101:116), 4, 4, byrow = TRUE))
})
