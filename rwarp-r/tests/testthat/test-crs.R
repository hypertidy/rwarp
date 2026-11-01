test_that("crs_describe returns a list with expected names", {
  d <- crs_describe("EPSG:4326")
  expect_type(d, "list")
  expect_named(d, c("input", "id", "description", "definition",
                     "wkt2_2019", "wkt2_2019_simplified",
                     "wkt1_gdal", "wkt1_esri", "proj5", "proj4",
                     "projjson", "error"))
})

test_that("crs_describe identifies EPSG:4326", {
  d <- crs_describe("EPSG:4326")
  expect_identical(d$input, "EPSG:4326")
  expect_false(is.na(d$wkt2_2019))
  expect_true(is.na(d$error))
})

test_that("bare +proj= string is an operation, not a CRS", {
  d <- crs_describe("+proj=laea")
  expect_identical(d$id, "laea")
  expect_true(is.na(d$wkt1_gdal))
})

test_that("+type=crs suffix makes it a CRS", {
  d <- crs_describe("+proj=laea +datum=WGS84 +type=crs")
  expect_false(is.na(d$wkt2_2019))
  expect_false(is.na(d$wkt2_2019))
})

test_that("unknown string returns error element", {
  d <- crs_describe("NOT_A_REAL_CRS_XYZ")
  expect_false(is.na(d$error))
  expect_true(is.na(d$wkt2_2019))
})
