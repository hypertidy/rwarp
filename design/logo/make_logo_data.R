# Data for the rwarp hex logo, computed with rwarp (the R package).
#
#   Rscript make_logo_data.R OUT_DIR
#
# Writes OUT_DIR/mur.json (the field) and OUT_DIR/mur-tilings.json (the
# tiles), which make_logo.py draws. Reads from the network: one overview of
# the GHRSST MUR v2 COG for 2025-11-26 on Source Cooperative. Needs rwarp,
# gdalraster, isoband and jsonlite.

library(rwarp)
library(gdalraster)

out_dir <- commandArgs(TRUE)[1]
url <- paste0(
  "/vsicurl/https://data.source.coop/ausantarctic/ghrsst-mur-v2/2025/11/26/",
  "20251126090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif"
)

## the target: EPSG:3031 over the box of vapour's sst.tif (Get started)
ext <- c(1660000, 3730000, -5020000, -2550000)
grid_at <- function(res) {
  dim <- as.integer(c(ext[2] - ext[1], ext[4] - ext[3]) / res)
  list(dim = dim, gt = c(ext[1], res, 0, ext[4], 0, -res))
}

## the field: MUR at 0.04 degrees (overview 1) warped to 5 km, as 1 C bands
ds <- new(GDALRaster, paste0("vrt://", url, "?ovr=1"))
src_crs <- ds$getProjectionRef(); src_gt <- ds$getGeoTransform()
src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())
g5 <- grid_at(5000)
plan <- warp_plan(src_crs, src_gt, src_dim, "EPSG:3031", g5$gt,
                  dst_size = g5$dim, resample = "bilinear")
read_fn <- function(xo, yo, xs, ys) as.numeric(ds$read(1, xo, yo, xs, ys, xs, ys))
z <- warp_exec(plan, read_fn, src_crs, src_gt, "EPSG:3031", g5$gt)
ds$close()
z <- z * 0.001 + 25   # MUR's scale and offset: Celsius
lo <- floor(min(z, na.rm = TRUE)); hi <- ceiling(max(z, na.rm = TRUE))
lev <- seq(lo, hi, by = 1)
x <- seq_len(g5$dim[1]) - 0.5; y <- seq_len(g5$dim[2]) - 0.5   # row 1 at the top
b <- isoband::isobands(x, y, z, lev[-length(lev)], lev[-1])
bands <- lapply(seq_along(b), function(k) {
  list(lo = lev[k], x = round(b[[k]]$x, 2), y = round(b[[k]]$y, 2), id = b[[k]]$id)
})
jsonlite::write_json(
  list(field = list(ncol = g5$dim[1], nrow = g5$dim[2], lo = lo, hi = hi, bands = bands)),
  file.path(out_dir, "mur.json"), auto_unbox = TRUE, digits = NA
)

## the tiles: the 10 km target cut n x n, each planned against the
## full-resolution MUR grid (geometry only, nothing read)
full <- new(GDALRaster, url)
fcrs <- full$getProjectionRef(); fgt <- full$getGeoTransform()
fdim <- c(full$getRasterXSize(), full$getRasterYSize())
full$close()
g10 <- grid_at(10000)
tilings <- list()
for (n in 4:6) {
  xs <- round(seq(0, g10$dim[1], length.out = n + 1))
  ys <- round(seq(0, g10$dim[2], length.out = n + 1))
  tl <- list()
  for (j in seq_len(n)) for (i in seq_len(n)) {
    off <- c(xs[i], ys[j]); size <- c(xs[i + 1] - xs[i], ys[j + 1] - ys[j])
    p <- warp_plan(fcrs, fgt, fdim, "EPSG:3031", g10$gt, dst_off = as.integer(off),
                   dst_size = as.integer(size), resample = "bilinear")
    ch <- p[[1]][[1]]
    tl[[length(tl) + 1]] <- list(tile = length(tl) + 1, dst_xoff = off[1], dst_yoff = off[2],
                                 dst_xsize = size[1], dst_ysize = size[2],
                                 src_pixels = ch$src_xsize * ch$src_ysize)
  }
  tilings[[as.character(n)]] <- list(n = n, xs = xs, ys = ys,
                                     tiles = do.call(rbind, lapply(tl, as.data.frame)))
}
jsonlite::write_json(
  list(target = list(ncol = g10$dim[1], nrow = g10$dim[2]), tilings = tilings),
  file.path(out_dir, "mur-tilings.json"), auto_unbox = TRUE, digits = NA
)
