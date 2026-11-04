# rwarp (R package)

[![R-CMD-check](https://github.com/hypertidy/rwarp/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/hypertidy/rwarp/actions/workflows/R-CMD-check.yaml)
[![Crate](https://img.shields.io/crates/v/rwarp.svg)](https://crates.io/crates/rwarp)

**Plannable raster warp via Rust ([rwarp](https://crates.io/crates/rwarp))**

This folder is the R package rwarp, bindings to the rwarp Rust crate in the root
of this repository: a GDAL-style warp pipeline
implemented in pure Rust using the proj crate for coordinate transforms.

The design has the same feature set in mind as `vapour::gdal_raster_data()` and
the `gdal_raster_*` family, with a pipeline where:

- The **chunk plan** (source windows, fill ratios) is computed first and can
  be inspected or cached before any I/O is performed.
- The **read function** is user-supplied and swappable: GDAL, HTTP byte-ref
  COG tiles, Zarr chunks, or any other source all use the same interface.
- The warp kernel itself is pure Rust (no GDAL warp dependency).

## Installation

Requires a Rust toolchain (>= 1.75) and PROJ (>= 9.0):

```r
# install.packages("remotes")
remotes::install_github("hypertidy/rwarp", subdir = "rwarp-r")
```

## Quick start

```r
library(rwarp)
library(gdalraster)
library(vaster)

url <- "/vsicurl/https://example.com/my.tif"
ds  <- new(GDALRaster, url, read_only = TRUE)

src_crs <- ds$getProjectionRef()
src_gt  <- ds$getGeoTransform()
src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())

dst_crs <- "EPSG:3857"
dst_ext <- c(-20037508, 20037508, -20037508, 20037508)
dst_dim <- c(512L, 512L)
dst_gt  <- extent_dim_to_gt(dst_ext, dst_dim)

## 1. Plan (no I/O)
plan <- warp_plan(
  src_crs = src_crs, src_gt = src_gt, src_dim = as.integer(src_dim),
  dst_crs = dst_crs, dst_gt = dst_gt, dst_size = dst_dim
)
plan[[1]][[1]]$fill_ratio   # inspect before fetching

## 2. Execute with a GDAL reader
result <- warp_exec(
  plan,
  read_fn = function(xoff, yoff, xsize, ysize)
    as.integer(ds$read(1L, xoff, yoff, xsize, ysize, xsize, ysize)),
  src_crs = src_crs, src_gt = src_gt,
  dst_crs = dst_crs, dst_gt = dst_gt,
  dst_size = dst_dim, nodata = -32768L
)
ximage::ximage(result, dst_ext)
ds$close()
```

## Core functions

| Function | Purpose |
|----------|---------|
| `warp_plan()` | Compute source windows for a destination grid (no I/O) |
| `warp_exec()` | Execute a plan with a user-supplied read function |
| `warp_resample()` | Low-level: warp a pre-fetched pixel buffer |
| `pixel_transform()` | Map destination pixels to fractional source pixel coords |
| `crs_describe()` | Describe a CRS string in all PROJ output formats |

## Vignettes

- **Sentinel-2: Reproject from UTM to Albers Equal Area** —
  single-tile, 10 m, GDAL read path.
- **GHRSST SST: Reproject to Antarctic Polar Stereographic** —
  global geographic source, polar target, byte-ref read path concept.
- **CRS Strings: What PROJ Actually Sees** —
  diagnosing `+proj=` string pitfalls with `crs_describe()`.
- **Warp Planning: Inspect Before You Read** —
  fill ratios, antimeridian splits, tiled execution loops.

## Design notes

### Plan/execute split

`warp_plan()` is pure geometry: it transforms grid corners and edge samples
to find the source window(s) that overlap the destination, accounting for
antimeridian wrapping and minimum fill ratios.  No pixels are read.

`warp_exec()` iterates the plan, calls `read_fn` per chunk, and composites
with a "first valid wins" rule.  Swap `read_fn` for any source.

### CRS strings

The `proj` Rust crate (as of 0.31) requires CRS strings to be recognisable
by PROJ's `proj_create()`.  EPSG codes (`"EPSG:4326"`) and WKT strings work
reliably.  Bare `+proj=` PROJ strings may be treated as coordinate
*operations* rather than CRS definitions; append `+type=crs` to fix this.
Use `crs_describe()` to diagnose any string.

### Known issues

- `warp_resample()` may write `i32::MIN`, which is `NA` in R, for out-of-bounds
  pixels instead of the supplied `nodata` value.
- Bilinear resampling has stripe artifacts at tile boundaries in some
  configurations (rwarp upstream issue).

## Related packages

- [rwarp](https://crates.io/crates/rwarp) — the underlying Rust crate
- [cogcache](https://github.com/hypertidy/cogcache) — earlier R prototype
  (wariirp -> cogcache -> rwarp provenance chain)
- [vapour](https://github.com/hypertidy/vapour) — GDAL bindings this aims
  — raster I/O package; vapour and rwarp work alongside each other
- [gdalraster](https://usdaforestservice.github.io/gdalraster/) — GDAL R
  bindings used in examples
