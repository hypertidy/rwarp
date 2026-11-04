# R package

The [`rwarp-r/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-r)
subdirectory of this repository is an R package, also named **rwarp**, that
exposes the same planning and resampling pipeline via
[extendr](https://extendr.github.io/).

## Installation

Requires a Rust toolchain (>= 1.75) and PROJ (>= 9.0):

```r
# install.packages("remotes")
remotes::install_github("hypertidy/rwarp", subdir = "rwarp-r")
```

## Core functions

| Function | What it does |
|---|---|
| `warp_plan()` | Compute source windows for a destination grid (pure geometry, no I/O) |
| `warp_exec()` | Execute a plan with a user-supplied read function |
| `warp_resample()` | Warp a pre-fetched pixel buffer directly |
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
  dst_size = dst_dim,  nodata  = -32768L
)
ds$close()

## result is a flat integer vector, length dst_dim[1] * dst_dim[2]
image(matrix(result, dst_dim[1], dst_dim[2]))
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

The list returned by `warp_plan()` has one entry per destination chunk,
each containing source offset, source size, and fill ratio:

```r
plan <- warp_plan(
  src_crs = "EPSG:4326", src_gt = c(-180, 0.1, 0, 90, 0, -0.1),
  src_dim = c(3600L, 1800L),
  dst_crs = "EPSG:3857",
  dst_gt  = c(-20037508, 78271.5, 0, 20037508, 0, -78271.5),
  dst_size = c(512L, 512L)
)

## how many source chunks?
length(plan)

## first chunk
str(plan[[1]])
# List of 4
#  $ xoff      : int ...
#  $ yoff      : int ...
#  $ fill_ratio: num ...   # 0-1; < 0.05 often not worth fetching
#  $ ...
```

Skip chunks with a very low fill ratio to avoid fetching data that
contributes almost nothing to the output:

```r
result <- warp_exec(
  plan,
  read_fn = function(xoff, yoff, xsize, ysize) {
    ## caller decides whether to actually read
    as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize))
  },
  min_fill = 0.05,   ## skip chunks below 5 % coverage
  ...
)
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
| `sentinel2-warp` | Reproject a Sentinel-2 tile from UTM to Albers Equal Area |
| `ghrsst-polar` | Reproject global SST to Antarctic Polar Stereographic via byte-range concept |
| `crs-strings` | Diagnosing `+proj=` string pitfalls with `crs_describe()` |
| `plan-inspect` | Fill ratios, antimeridian splits, tiled execution loops |

```r
vignette("sentinel2-warp",  package = "rwarp")
vignette("plan-inspect",    package = "rwarp")
```

## Relationship to vapour

rwarp has the same feature set in mind as `vapour::gdal_raster_data()` and
the `gdal_raster_*` family. A few things the pipeline adds:

- The read step is **user-supplied**: any source, not just GDAL files.
- The plan is **inspectable**: fill ratios and source windows are
  available before any I/O.
- The warp kernel is **pure Rust**: no GDAL warp dependency, so it
  can compile to WebAssembly and run in the browser.

For reading and writing ordinary raster files, vapour and gdalraster
remain the natural choices.

## Source and issues

The R package lives in
[`rwarp-r/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-r).
Issues and pull requests are welcome at
[github.com/hypertidy/rwarp](https://github.com/hypertidy/rwarp).
