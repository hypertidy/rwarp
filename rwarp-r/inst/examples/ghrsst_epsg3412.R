## GHRSST MUR v2 SST -> EPSG:3412 (Antarctic Polar Stereographic)
## 40 x 40 tile centred on lon=0, lat=-60, 25 km resolution
##
## Requires: rwarp, gdalraster, vaster, reproj

library(rwarp)
library(gdalraster)
library(vaster)
library(reproj)

url <- paste0(
  "/vsicurl/https://data.source.coop/ausantarctic/ghrsst-mur-v2/",
  "2025/11/26/",
  "20251126090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif"
)

ds      <- new(GDALRaster, url, read_only = TRUE)
src_crs <- ds$getProjectionRef()
src_gt  <- ds$getGeoTransform()
src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())
nodata  <- -32768L

## Destination: 40 x 40 at 25 km, centred 60 S / 0 E in EPSG:3412
dst_crs    <- "EPSG:3412"
center     <- reproj_xy(cbind(0, -60), target = dst_crs, source = "EPSG:4326")
tile_r     <- 500000
dst_ext    <- c(center[1] - tile_r, center[1] + tile_r,
                center[2] - tile_r, center[2] + tile_r)
dst_dim    <- c(40L, 40L)
dst_gt     <- extent_dim_to_gt(dst_ext, dst_dim)

plan <- warp_plan(
  src_crs = src_crs, src_gt = src_gt, src_dim = as.integer(src_dim),
  dst_crs = dst_crs, dst_gt = dst_gt,
  dst_off  = c(0L, 0L), dst_size = dst_dim,
  resample_padding = 1L
)

result <- warp_exec(
  plan, read_fn = function(xoff, yoff, xsize, ysize)
    as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize)),
  src_crs = src_crs, src_gt = src_gt,
  dst_crs = dst_crs, dst_gt = dst_gt,
  dst_size = dst_dim, nodata = nodata
)

ds$close()

## Scale to Celsius
sst <- result * 0.001 + 298.15 - 273.15
sst[result == nodata] <- NA

ximage::ximage(sst, dst_ext,
  main = "MUR SST 2025-11-26 | EPSG:3412 | 25 km",
  col  = rev(hcl.colors(256, "RdYlBu")),
  zlim = c(-2, 25))
