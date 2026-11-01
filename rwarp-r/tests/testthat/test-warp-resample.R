## Minimal round-trip test for warp_resample (no network).

src_gt  <- c(-180, 1, 0, 90, 0, -1)
src_dim <- c(360L, 180L)
src_crs <- "EPSG:4326"
dst_crs <- "EPSG:4326"
dst_gt  <- c(0, 1, 0, 10, 0, -1)
dst_dim <- c(5L, 5L)

## Synthetic source: sequential integers (row-major)
src_pixels <- as.integer(seq_len(prod(src_dim)))

test_that("warp_resample returns integer vector of correct length", {
  result <- warp_resample(
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_dim = dst_dim,
    src_pixels  = src_pixels,
    src_ncol    = src_dim[1], src_nrow = src_dim[2],
    src_col_off = 0L, src_row_off = 0L,
    nodata = -32768L
  )
  expect_type(result, "integer")
  expect_length(result, prod(dst_dim))
})

test_that("identity warp produces non-nodata values", {
  result <- warp_resample(
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_dim = dst_dim,
    src_pixels  = src_pixels,
    src_ncol    = src_dim[1], src_nrow = src_dim[2],
    src_col_off = 0L, src_row_off = 0L,
    nodata = -32768L
  )
  ## dst pixel 0 is src pixel (180, 80), 1-based row-major value 80 * 360 + 181
  expect_equal(result[1], 28981L)
  expect_false(anyNA(result))
  expect_false(any(result == -32768L))
})

test_that("warp_exec places every chunk of a split plan", {
  plan <- list(list(
    list(dst_xoff = 0L, dst_yoff = 0L, dst_xsize = 2L, dst_ysize = 5L,
         src_xoff = 0L, src_yoff = 0L, src_xsize = 360L, src_ysize = 180L,
         fill_ratio = 1),
    list(dst_xoff = 2L, dst_yoff = 0L, dst_xsize = 3L, dst_ysize = 5L,
         src_xoff = 0L, src_yoff = 0L, src_xsize = 360L, src_ysize = 180L,
         fill_ratio = 1)
  ))
  out <- warp_exec(
    plan,
    read_fn = function(xoff, yoff, xsize, ysize) src_pixels,
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_size = dst_dim
  )
  whole <- warp_resample(
    src_crs = src_crs, src_gt = src_gt,
    dst_crs = dst_crs, dst_gt = dst_gt,
    dst_dim = dst_dim,
    src_pixels  = src_pixels,
    src_ncol    = src_dim[1], src_nrow = src_dim[2],
    src_col_off = 0L, src_row_off = 0L,
    nodata = -32768L
  )
  expect_equal(out, matrix(whole, nrow = 5L, ncol = 5L, byrow = TRUE))
})
