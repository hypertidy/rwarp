# R package

The [`rwarp-r/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-r)
subdirectory of this repository is an R package, also named **rwarp**, that
exposes the same planning and resampling pipeline via
[extendr](https://extendr.github.io/).

## Installation

Requires a Rust toolchain (>= 1.86), pkg-config, and PROJ (>= 9.6.2):

```r
# install.packages("remotes")
remotes::install_github("hypertidy/rwarp", subdir = "rwarp-r")
```

## Core functions

| Function | What it does |
|---|---|
| `warp_plan()` | Compute source windows for a destination grid (pure geometry, no I/O) |
| `warp_exec()` | Execute a plan with a user-supplied read function |
| `warp_buffer()` | Warp one source window you already have, as `warp_exec()` does each chunk |
| `warp_resample()` | Deprecated: use `warp_buffer()` |
| `pixel_transform()` | Map destination pixels to fractional source pixel coordinates |
| `crs_describe()` | Describe a CRS string in all PROJ output formats |

## Plan / execute pattern

The design mirrors the Rust API: planning and reading are always separate
steps. `warp_plan()` transforms grid corners and edge samples to find the
source window(s) that cover each destination tile. No pixels are touched.
`warp_exec()` calls a user-supplied `read_fn` once per chunk and composites
the results.

```r
library(rwarp)
library(gdalraster)

## --- open source ---
url <- "/vsicurl/https://example.com/my.tif"
ds  <- new(GDALRaster, url, read_only = TRUE)

src_crs <- ds$getProjectionRef()
src_gt  <- ds$getGeoTransform()
src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())

## --- define destination grid ---
dst_crs <- "EPSG:3857"
dst_ext <- c(-20037508, 20037508, -20037508, 20037508)
dst_dim <- c(512L, 512L)
# geotransform: [xmin, xres, 0, ymax, 0, -yres]
dst_gt  <- c(dst_ext[1],
             (dst_ext[2] - dst_ext[1]) / dst_dim[1], 0,
             dst_ext[4], 0,
             -(dst_ext[4] - dst_ext[3]) / dst_dim[2])

## --- 1. Plan (no I/O) ---
plan <- warp_plan(
  src_crs  = src_crs,  src_gt  = src_gt,  src_dim = as.integer(src_dim),
  dst_crs  = dst_crs,  dst_gt  = dst_gt,  dst_size = dst_dim
)

## inspect fill ratio before spending any network budget
plan[[1]][[1]]$fill_ratio

## --- 2. Execute with a GDAL reader ---
result <- warp_exec(
  plan,
  read_fn = function(xoff, yoff, xsize, ysize)
    as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize)),
  src_crs  = src_crs,  src_gt  = src_gt,
  dst_crs  = dst_crs,  dst_gt  = dst_gt,
  dst_size = dst_dim,
  src_nodata = -32768L, dst_nodata = -32768L
)
ds$close()

## result is an integer matrix, dst_dim[2] rows by dst_dim[1] columns
## (warp_buffer() returns the same shape)
ximage::ximage(result, dst_ext)
```

Swapping the reader is the point. The same plan works with an HTTP
byte-range reader, a Zarr chunk fetcher, or any other source that can
return a rectangular pixel block:

```r
## byte-range COG reader (sketch)
read_fn <- function(xoff, yoff, xsize, ysize) {
  byte_range <- cog_tile_range(cog_url, xoff, yoff, xsize, ysize)
  raw_bytes  <- http_get_range(cog_url, byte_range)
  decode_cog_tile(raw_bytes)
}

result <- warp_exec(plan, read_fn = read_fn, ...)
```

## Inspecting the plan

`warp_plan()` returns a list holding one chunk list. Each chunk has its
destination window, its source window, its fill ratio and the kernel
scale it was planned for:

```r
plan <- warp_plan(
  src_crs = "EPSG:4326", src_gt = c(-180, 0.1, 0, 90, 0, -0.1),
  src_dim = c(3600L, 1800L),
  dst_crs = "EPSG:3857",
  dst_gt  = c(-20037508, 78271.5, 0, 20037508, 0, -78271.5),
  dst_size = c(512L, 512L)
)

## how many chunks?
length(plan[[1]])

## the first chunk
str(plan[[1]][[1]])
# List of 11
#  $ dst_xoff  : int 0
#  $ dst_yoff  : int 0
#  $ dst_xsize : int 512
#  $ dst_ysize : int 512
#  $ src_xoff  : int 0
#  $ src_yoff  : int 0
#  $ src_xsize : int 3600
#  $ src_ysize : int 1800
#  $ fill_ratio: num 1.06
#  $ scale_x   : num 0.143
#  $ scale_y   : num 0.778
```

`warp_plan(min_fill_ratio = )` splits chunks with a low fill ratio into
smaller ones, so less is read. To skip chunks that contribute almost
nothing, drop them from the plan before executing it:

```r
plan[[1]] <- Filter(function(ch) ch$fill_ratio >= 0.05, plan[[1]])
result <- warp_exec(plan, read_fn = read_fn, ...)
```

The chunks matter to the values, not only to memory: when the kernel
widens to downsample, each chunk estimates its own scale. By default
`warp_plan()` plans for one byte per pixel of working memory and never
splits on fill ratio, so a destination that gdalwarp splits can be one
chunk here. To get gdalwarp's chunks, and its values, describe the warp as
gdalwarp sees it and use its rules:

```r
plan <- warp_plan(src_crs, src_gt, src_dim, dst_crs, dst_gt,
                  dst_size = dst_dim, resample = "cubic",
                  nbands = 3, working_type = "Float64", src_nodata = 0,
                  memory_limit = 64,                  # gdalwarp -wm
                  min_fill_ratio = 0.5, min_dst_size = 100L,
                  scale_origin = "output", split_antimeridian = FALSE)
```

## CRS diagnostics

`crs_describe()` calls PROJ directly and returns every string form PROJ
can produce for a given input. Use it to debug CRS strings before passing
them to warp functions:

```r
crs_describe("EPSG:3031")
# $proj4
# [1] "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
#
# $wkt2
# [1] "PROJCRS[\"WGS 84 / Antarctic Polar Stereographic\", ..."
#
# $projjson
# ...

## bare +proj= string without +type=crs is treated as an operation, not a CRS
crs_describe("+proj=merc +datum=WGS84")          # may fail
crs_describe("+proj=merc +datum=WGS84 +type=crs") # works
```

The CRS string pitfalls are covered in depth in the **CRS Strings**
vignette (`vignette("crs-strings", package = "rwarp")`).

## Vignettes

| Vignette | What it shows |
|---|---|
| `sentinel2-warp` | Two Sentinel-2 tiles in different UTM zones, three bands as float reflectance, onto one Albers grid; checked against gdalwarp |
| `ghrsst-polar` | Reproject global SST to Antarctic Polar Stereographic via byte-range concept |
| `crs-strings` | Diagnosing `+proj=` string pitfalls with `crs_describe()` |
| `plan-inspect` | Fill ratios, antimeridian splits, tiled execution loops |

```r
vignette("sentinel2-warp",  package = "rwarp")
vignette("plan-inspect",    package = "rwarp")
```

## Alongside vapour, gdalraster and terra

The usual way to reproject in R is through GDAL's warper:
`vapour::gdal_raster_data()` and the `gdal_raster_*` family,
`gdalraster::warp()` and `terra::project()`. For reading and writing
ordinary raster files they remain the natural choices. rwarp covers the
same reprojection, with the steps apart, which helps when:

- The read step should be **user-supplied**: any source, not just GDAL files.
- The plan should be **inspectable**: fill ratios and source windows are
  available before any I/O.
- The warp kernel should be **pure Rust**: no GDAL warp dependency, so it
  can compile to WebAssembly and run in the browser.

The two work together: the example above reads with gdalraster and warps
with rwarp.

## Source and issues

The R package lives in
[`rwarp-r/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-r).
Issues and pull requests are welcome at
[github.com/hypertidy/rwarp](https://github.com/hypertidy/rwarp).
