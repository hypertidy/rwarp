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
