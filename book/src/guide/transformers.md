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

## Sources across 180 degrees

A lon/lat source stored as 170..190, as -190..-170, or as a 0..360 global
grid has longitudes the CRS step never returns: inverse projections give
-180..180. gdalwarp handles this by wrapping the source's longitudes
around its centre (GDAL's `CENTER_LONG`), and `GenImgProjTransformer`
does the same once it knows the source raster's size:

```rust,ignore
use rwarp::transform::{CenterLong, GenImgProjTransformer};

let t = GenImgProjTransformer::new("EPSG:4326", src_gt, dst_crs, dst_gt)?
    .with_source_size([src_width, src_height]);
// the middle of the source's corner longitudes, to six digits as GDAL
let centre = t.center_long();
```

Target to source, a longitude more than 180 degrees from the centre is
brought back by 360 (once), so a target crossing 180 finds both halves of
the source; source to target, the same is done before the CRS step. Only
a source CRS that is geographic in degrees is wrapped, as in GDAL. A source
wider than 360 degrees is not, unless the target is lon/lat too and its
extent crosses 180 (give it with `with_target_extent`, gdalwarp's `-te`).

`with_center_long(CenterLong::Off)` turns it off (gdalwarp does, for
cutlines) and `CenterLong::At(lon)` sets the centre. A transformer from
`with_backend` does not know its CRSs, so `CenterLong::Auto` does nothing
there; use `CenterLong::At(source_center_long(&src_gt, size, None).unwrap())`
for GDAL's value. Without `with_source_size`, nothing is wrapped, as in
rwarp 0.3.0.

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
