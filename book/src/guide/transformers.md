# Transformers

A transformer maps coordinates between target pixels and source pixels. Every
part of rwarp that needs coordinates (planning, kernels) takes anything that
implements the `Transformer` trait, so they compose freely.

```rust,ignore
pub trait Transformer {
    /// dst_to_src = true: input is target pixel/line, output is source pixel/line.
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool>;
}
```

It follows GDAL's transformer callback: coordinate arrays are modified in
place, and one flag per point says whether it transformed. The warp kernels
always call it target to source.

## The transformers

| Transformer | Maps | Use it for |
|---|---|---|
| `GenImgProjTransformer` | pixel to geo (geotransform), CRS to CRS, geo to pixel | ordinary rasters with a geotransform on both sides |
| `ApproxTransformer<T>` | wraps any transformer, interpolating between exactly transformed points within a maximum error | speed; gdalwarp uses 0.125 pixels by default |
| `GcpTransformer` | polynomial fitted to ground control points, order 1 to 3 | unrectified imagery with GCPs |
| `GeolocArrays` | pixel to geo through 2D coordinate arrays | swaths and curvilinear model grids |
| `GenImgGeolocTransformer` | either side a geotransform or geolocation arrays, plus a CRS step | warping from, or onto, a curvilinear grid |

Geotransforms are in GDAL's order, `[x0, dx, rx, y0, ry, dy]`. Pixel
coordinates are continuous: pixel `i` covers `[i, i + 1)`, so its centre is
`i + 0.5`.

## CRS backends

The CRS step goes through the `CrsTransform` trait:

```rust,ignore
pub trait CrsTransform {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)>;
}
```

Its direction is fixed at construction. Geographic coordinates are always
longitude, latitude in degrees, whichever backend is used. `crs_pair` builds
the two directions with the backend compiled in; `with_backend`
constructors accept your own.

## Vector quantities

For velocity fields and other `(u, v)` quantities stored in grid
coordinates, reprojecting the scalar values is not enough: the vectors also
rotate with the local frame. `Transformer::jacobian` gives the local 2 x 2
derivative of a transform at a point (by finite differences unless an
implementation overrides it), the piece a vector-preserving warp needs.
