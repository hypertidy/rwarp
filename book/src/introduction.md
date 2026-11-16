# rwarp

rwarp reprojects and resamples rasters. It is GDAL's warp pipeline in
Rust: coordinate transforms, approximate transforms, source window
planning and resampling kernels, with no runtime GDAL dependency. It
follows GDAL's algorithms closely, written with reference to the GDAL
source and checked against `gdalwarp` output, but it is an independent
implementation, not affiliated with or endorsed by the GDAL project.

**Try it in the browser:** the [demo](demo/index.html) warps web map tiles
into a projection of your choice, client-side, compiled to WebAssembly.

## The usual way to do this

Most raster reprojection goes through GDAL's warper, whether you call it
yourself or not. On the command line it is `gdalwarp`, or
`gdal raster reproject` from GDAL 3.11. In R it is under `terra::project()`,
`gdalraster::warp()` and `vapour::gdal_raster_data()`, and in Python under
`rasterio.warp.reproject()` and rioxarray's `.rio.reproject()`. For
conservative regridding between model grids, the usual choice is ESMF,
through ESMPy or xESMF, or CDO's `remapcon`.

These are good defaults. They read and write almost any format, they have
been used on a great deal of data, and they warp one file into another in
a single call. If they already do what you need, there is no reason to
change.

## The same things in rwarp

rwarp has equivalents for most of what those tools do to the pixels, so
what you know carries over:

| You may know it as | In rwarp | See |
|---|---|---|
| gdalwarp's error threshold, `-et 0.125` by default | `ApproxTransformer` | [Transformers](guide/transformers.md) |
| how gdalwarp splits a warp into chunks, set by `-wm` | `plan_chunks`, `ChunkOptions` | [Planning reads](guide/planning.md) |
| `gdalwarp -r near`, `bilinear`, `cubic`, `lanczos`, `average`, `mode` | `warp()` with a `ResampleAlg` | [Resampling](guide/resampling.md) |
| ESMF's `conserve`; xESMF's `conserve` and `conserve_normed` | `rwarp::weights` | [Conservative weights](guide/weights.md) |
| `gdalwarp -geoloc`, geolocation arrays | `GeolocArrays` | [Curvilinear grids](guide/curvilinear.md) |
| `gdalwarp -cutline`, `-cblend` | `Cutline` | [Cutlines](guide/cutline.md) |

Where rwarp does something differently from GDAL, the
[GDAL parity](gdal-parity.md) page says what and why.

## When you might reach for rwarp

GDAL's warper plans, reads, maps coordinates and resamples in one
operation. rwarp keeps those steps apart, which helps in a few situations:

- **You want to know what a warp will read before it reads anything.**
  The plan lists the source pixels each target chunk needs, so a cache,
  tile server or planner can count the I/O first.
- **Your pixels don't come through GDAL.** rwarp does no I/O. You fetch
  the window it asks for from wherever the data lives: COG tiles, Zarr
  chunks, a byte-reference store, or a GDAL dataset if that is easiest.
- **GDAL is hard to bring along**, as in a web page. With the pure-Rust
  `proj4rs` backend rwarp builds for WebAssembly, which is what the demo
  runs on.
- **You regrid between the same grids many times.** Conservative weights
  are computed once and applied to every band, time step or variable.
- **You want to see exactly what gdalwarp does.** rwarp's fixtures pin
  down how ties are broken, where edges fall and when kernels widen.

The usual tools stay the simpler choice for warping files into files, and
ESMF for conservative weights with areas measured on the sphere (rwarp's
are planar, in source pixel space, for now). The two also work side by
side: the [R example](r-package.md) reads with gdalraster and warps with
rwarp.

## How it fits together

```text
target grid
    |
    v
[plan]  plan_chunks / compute_source_window
    |     which chunks, and which source pixels does each need?
    |     inspectable, serialisable, resumable
    v
[read]  the caller fetches bytes: COG tiles, Zarr chunks, byte-reference stores
    |
    v
[warp]  warp(): NearestNeighbour, Bilinear, Cubic, Lanczos, Average, Mode
```

That makes it a building block for caches, tile servers, browser viewers
and planners that need to know the I/O cost of a reprojection up front.

For readers who know GDAL's source, the pieces correspond to these:

| Piece | GDAL equivalent |
|---|---|
| `GenImgProjTransformer`: pixel, geo and CRS steps | `GDALGenImgProjTransform` |
| `ApproxTransformer`: adaptive interpolation | `GDALApproxTransform` |
| `GcpTransformer`: polynomial order 1-3 | `GDALGCPTransform` |
| `GeolocArrays`, `GenImgGeolocTransformer`: curvilinear source or target | geolocation arrays (`alg/gdalgeoloc.cpp`) |
| `plan_chunks` + `ChunkOptions`, `compute_source_window` | `GDALWarpOperation` planning (`CollectChunkList`, `ComputeSourceWindow`) |
| `warp()`, `warp_chunks()`: resampling kernels, kernel widening, nodata rules | `gdalwarpkernel.cpp` |

CRS transforms go through a small trait with two backends: libproj (the
default, full PROJ) or `proj4rs` (pure Rust, builds for
`wasm32-unknown-unknown`). See [Getting started](getting-started.md).

## Status

The core API has been stable since 0.3; 0.4 changed only the
controlledburn types that appear in rwarp's API (see the
[changelog](changelog.md)). Types and functions marked `#[non_exhaustive]`
may gain new variants in minor releases; everything else follows semver.
Each component is validated against GDAL output; the
[GDAL parity](gdal-parity.md) page lists what has been checked, how, and
where rwarp deliberately differs. Feedback and testing are very welcome at
[github.com/hypertidy/rwarp](https://github.com/hypertidy/rwarp).

## R package

An R package lives in the [`rwarp-r/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-r)
subdirectory of this repository. It wraps the same planning and resampling
pipeline via [extendr](https://extendr.github.io/), exposing `warp_plan()`,
`warp_exec()`, `warp_resample()`, `pixel_transform()` and `crs_describe()`.
See the [R package chapter](r-package.md) or install directly:

```r
remotes::install_github("hypertidy/rwarp", subdir = "rwarp-r")
```

## Python package

Python bindings are planned, through [PyO3](https://pyo3.rs/), with NumPy
arrays as the pixel buffer type and the same API shape as the R package.
The [Python package chapter](python-package.md) sketches the design;
contributions are welcome.
