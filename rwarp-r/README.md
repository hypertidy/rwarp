# rwarp (R package)

[![R-CMD-check](https://github.com/hypertidy/rwarp/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/hypertidy/rwarp/actions/workflows/R-CMD-check.yaml)
[![Crate](https://img.shields.io/crates/v/rwarp.svg)](https://crates.io/crates/rwarp)
[![pkgdown site](https://img.shields.io/badge/docs-pkgdown-blue)](https://hypertidy.github.io/rwarp/r/)

**Plannable raster warp via Rust ([rwarp](https://crates.io/crates/rwarp))**

> Full documentation, function reference and vignettes:
> **<https://hypertidy.github.io/rwarp/r/>**

This folder is the R package rwarp, bindings to the rwarp Rust crate in the root
of this repository: a GDAL-style warp pipeline
implemented in pure Rust using the proj crate for coordinate transforms.

## The usual way to do this

Most raster reprojection in R goes through GDAL's warper:
`terra::project()`, `gdalraster::warp()`, `vapour::gdal_raster_data()`,
`sf::gdal_utils("warp")` and `stars::st_warp(use_gdal = TRUE)`. These are
good defaults. They read and write almost any format, they have been used
on a great deal of data, and they warp a file into a file, or into a raster
object, in one call. If they already do what you need, there is no reason
to change.

## The same things in rwarp

rwarp's arguments follow gdalwarp's options, so what you know carries over:

| You may know it as | In rwarp |
|---|---|
| `-t_srs`, `-te`, `-ts` (the target grid) | `dst_crs`, `dst_gt`, `dst_size` (`vaster::extent_dim_to_gt()` makes `dst_gt` from an extent and dimension) |
| `-r near`, `bilinear`, `cubic`, `cubicspline`, `lanczos`, `average`, `mode`, `min`, `max`, `med`, `rms` | `resample` |
| `-et 0.125` | `max_error` in `warp_exec()` |
| `-srcnodata`, `-dstnodata`, `-wo UNIFIED_SRC_NODATA` | `src_nodata`, `dst_nodata`, `band_nodata` in `warp_exec()` |
| a source alpha band (`-srcalpha`) | `mask_band` in `warp_exec()` |
| `-wo MODE_TIES` | `mode_ties` in `warp_exec()` |
| `-wo SOURCE_EXTRA`, `-wo XSCALE`, `-wo YSCALE` | `resample_padding`, `scaling`, `scale` in `warp_plan()` |

## When you might reach for rwarp

GDAL's warper plans, reads, maps coordinates and resamples in one call.
rwarp keeps those steps apart, which helps in a few situations:

- **You want to know what a warp will read before it reads anything.**
  `warp_plan()` lists each chunk's source window and fill ratio with no
  I/O, so you can count the reads, cache the plan, or run the chunks
  yourself in a tiled loop.
- **Your pixels don't come through GDAL**, or not only through it. The
  read function is yours: GDAL, HTTP byte-ref COG tiles, Zarr chunks, or
  any other source all use the same interface.
- **You want gdalwarp's results with the steps in view.** The warp kernel
  is pure Rust (no GDAL warp dependency), checked against gdalwarp's
  output.

The usual tools stay the simpler choice for warping files into files, or
into a terra or stars object. The Rust crate also has cutlines, geolocation
arrays and conservative weights that this package doesn't bind yet. For
cutlines and geolocation arrays in R, gdalwarp's `-cutline` and `-geoloc`
are there through `gdalraster::warp()` or `sf::gdal_utils()`. The usual
tools and rwarp also work side by side: the quick start below reads with
gdalraster and warps with rwarp.

## Installation

Requires a Rust toolchain (>= 1.86), pkg-config, and PROJ (>= 9.6.2):

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
- [cogcache](https://github.com/rgdal-dev/cogcache) — earlier R prototype
  (wariirp -> cogcache -> rwarp provenance chain)
- [vapour](https://github.com/hypertidy/vapour) — GDAL raster I/O and
  warping; vapour and rwarp work alongside each other
- [gdalraster](https://usdaforestservice.github.io/gdalraster/) — GDAL R
  bindings used in examples
