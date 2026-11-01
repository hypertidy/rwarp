## Sentinel-2 B04 COG -> EPSG:5070 (US Albers Equal Area)
## Single-date, single MGRS tile, 10 m resolution
##
## Requires: rwarp, gdalraster, vaster, reproj

library(rwarp)
library(gdalraster)
library(vaster)
library(reproj)

url <- paste0(
  "/vsicurl/https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
  "/sentinel-2-c1-l2a/55/G/CM/2026/2/S2C_T55GCM_20260227T000650_L2A/B04.tif"
)

ds      <- new(GDALRaster, url, read_only = TRUE)
src_crs <- ds$getProjectionRef()
src_gt  <- ds$getGeoTransform()
src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())
nodata  <- as.integer(ds$getNoDataValue(1L))
if (is.na(nodata)) nodata <- 0L

dst_crs  <- "EPSG:5070"
bbox_ll  <- ds$bbox()[c(1, 3, 2, 4)]
dst_ext  <- reproj_extent(bbox_ll, target = dst_crs, source = src_crs)
dst_dim  <- as.integer(c(
  ceiling((dst_ext[2] - dst_ext[1]) / 10),
  ceiling((dst_ext[4] - dst_ext[3]) / 10)
))
dst_gt   <- extent_dim_to_gt(dst_ext, dst_dim)

cat("Destination:", dst_dim, "pixels at 10 m\n")

plan <- warp_plan(
  src_crs = src_crs, src_gt = src_gt, src_dim = as.integer(src_dim),
  dst_crs = dst_crs, dst_gt = dst_gt,
  dst_off  = c(0L, 0L), dst_size = dst_dim
)

result <- warp_exec(
  plan, read_fn = function(xoff, yoff, xsize, ysize)
    as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize)),
  src_crs = src_crs, src_gt = src_gt,
  dst_crs = dst_crs, dst_gt = dst_gt,
  dst_size = dst_dim, nodata = nodata
)

ds$close()

result[result <= 0L] <- NA
ximage::ximage(result, dst_ext,
  main = "Sentinel-2 B04 | EPSG:5070 (Albers) | 10 m",
  col = grey.colors(256))
