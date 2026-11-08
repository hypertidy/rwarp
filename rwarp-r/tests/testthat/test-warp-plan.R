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

test_that("plan_windows lists every chunk's source window in exec order", {
  dst_gt <- c(-1.5e6, 7812.5, 0, 1.5e6, 0, -7812.5)
  plan <- warp_plan(src_crs, src_gt, src_dim, "+proj=laea +lat_0=-17 +lon_0=180", dst_gt,
                    dst_size = c(384L, 384L), resample = "bilinear")
  w <- plan_windows(plan)
  expect_identical(colnames(w), c("xoff", "yoff", "xsize", "ysize"))
  expect_equal(nrow(w), length(plan[[1]]))
  ## read_fn sees exactly these windows, in this order
  seen <- list()
  src <- matrix(1, src_dim[2], src_dim[1])
  invisible(warp_exec(plan, function(xoff, yoff, xsize, ysize) {
    seen[[length(seen) + 1L]] <<- c(xoff, yoff, xsize, ysize)
    rep(1, xsize * ysize)
  }, src_crs, src_gt, "+proj=laea +lat_0=-17 +lon_0=180", dst_gt))
  expect_equal(unname(w), do.call(rbind, seen), ignore_attr = TRUE)
  expect_equal(nrow(plan_windows(NULL)), 0L)
})
