# rwarp

rwarp is GDAL's warp pipeline in Rust: coordinate transforms, approximate
transforms, source window planning and resampling kernels, with no runtime
GDAL dependency. It follows GDAL's algorithms closely, written with
reference to the GDAL source and checked against `gdalwarp` output, but it
is an independent implementation, not affiliated with or endorsed by the
GDAL project.

**Try it in the browser:** the [demo](demo/index.html) warps web map tiles
into a projection of your choice, client-side, compiled to WebAssembly.

## Why another warper

GDAL's warper bundles planning, reading, coordinate mapping and resampling
into one operation. rwarp separates them, so you can see and control what
will be read before any pixel is touched:

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
[warp]  warp(): nearest, bilinear, cubic, lanczos, average, mode
```

That makes it a building block for caches, tile servers, browser viewers
and planners that need to know the I/O cost of a reprojection up front.

## What it covers

| Piece | GDAL equivalent |
|---|---|
| `GenImgProjTransformer`: pixel, geo and CRS steps | `GDALGenImgProjTransform` |
| `ApproxTransformer`: adaptive interpolation | `GDALApproxTransform` |
| `GcpTransformer`: polynomial order 1-3 | `GDALGCPTransform` |
| `GeolocArrays`, `GenImgGeolocTransformer`: curvilinear source or target | geolocation arrays (`alg/gdalgeoloc.cpp`) |
| `plan_chunks`, `compute_source_window` | `GDALWarpOperation` planning (`CollectChunkList`, `ComputeSourceWindow`) |
| `warp()`: resampling kernels, kernel widening, nodata rules | `gdalwarpkernel.cpp` |

CRS transforms go through a small trait with two backends: libproj (the
default, full PROJ) or `proj4rs` (pure Rust, builds for
`wasm32-unknown-unknown`). See [Getting started](getting-started.md).

## Status

Early development: the API will change. Each component is validated
against GDAL output; the [GDAL parity](gdal-parity.md) page lists what has
been checked, how, and where rwarp deliberately differs. Feedback and
testing are very welcome, at
[github.com/hypertidy/rwarp](https://github.com/hypertidy/rwarp).
