# Applications

## Browser demo

[Open the demo](demo/index.html). A Leaflet map in a projection of your
choice (a local Lambert azimuthal equal-area by default, centred wherever
you like) whose tiles are warped client-side from web map tiles, WMTS,
ArcGIS tile services or a Cloud Optimized GeoTIFF. rwarp runs in a Web
Worker, compiled to WebAssembly with the pure-Rust `proj4rs` backend: no
server, no libproj.

The shim is `rwarp-wasm/` in the repository. Its JavaScript API plans a
source window and warps RGBA or float tiles:

```js
const w = new Warper(srcCrs, srcGt, dstCrs, dstGt, maxError);
const win = w.source_window(dstW, dstH, srcW, srcH, padding);
const rgba = w.warp_rgba(srcRgba, srcW, srcH, xoff, yoff, dstW, dstH, "nearest");
```

`Warper.withTransforms` accepts CRS transforms as JavaScript functions, so
a page can lend rwarp a projection library (proj4js, a PROJ build) when
`proj4rs` declines a CRS. See `rwarp-wasm/README.md`.

## cogcache

rwarp was extracted from
[cogcache](https://github.com/rgdal-dev/cogcache), which plans warps from
a target tile back to the source chunks it needs, and caches both the raw
encoded source chunks and the warped target tiles. rwarp is the planning
and resampling core.

## Reference for array planning

The [ArrIR](https://github.com/charles-turner-bot/arrir) design for portable
array operation plans defines resampling independently of GDAL, and uses
rwarp and gdalwarp together to pin down what gdalwarp actually does: how
ties are broken, how edges are placed for curvilinear grids, where kernels
widen. rwarp's fixtures are the evidence for those answers.

## R package

The [`rwarp` R package](r-package.md) wraps the same planning and
resampling pipeline via [extendr](https://extendr.github.io/).
Install from the `rwarp-r/` subdirectory of this repository:

```r
remotes::install_github("hypertidy/rwarp", subdir = "rwarp-r")
```

It exposes `warp_plan()`, `warp_exec()`, `warp_resample()`,
`pixel_transform()` and `crs_describe()`. The plan/execute split and
the user-supplied read function work the same way as in the Rust API.

A [Python package](python-package.md) in `rwarp-py/` exposes the same
API via [PyO3](https://pyo3.rs/), with NumPy arrays as the pixel buffer
type. It is in early development.

## Using rwarp in your project

If you build on rwarp, an issue or pull request adding your project here
is welcome.
