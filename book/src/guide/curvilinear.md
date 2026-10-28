# Curvilinear grids

Many datasets have no geotransform: satellite swaths, ocean models on
rotated or tripolar grids, and anything else whose cell positions are given
by 2D longitude and latitude arrays. rwarp handles these with geolocation
arrays, following GDAL's geolocation transformer (`alg/gdalgeoloc.cpp`).

## The model

Each geolocation sample is a point. Between four neighbouring samples,
positions are interpolated bilinearly; beyond the outermost samples, they
are extrapolated from the closest grid square. Going the other way, from a
coordinate to a pixel, rwarp finds the quadrilateral of samples that
contains the point and solves the exact inverse bilinear interpolation.

Where a sample sits within its pixel is set by `GeolocConvention`:

- **`PixelCenter`**: samples are cell centres, as for CF coordinate
  variables. Cell edges then fall at the midpoint between neighbouring
  centres (in the arrays' own coordinates), the outer edges half a spacing
  beyond the outermost centres, and each corner at the mean of four centres.
- **`TopLeftCorner`**: samples are top-left pixel corners. This is GDAL's
  default when `GEOREFERENCING_CONVENTION` is not set, so centre coordinates
  read through the default land half a cell off. Say which one you have.

`GeolocOptions` also carries pixel and line offsets and steps (for arrays
that sample every Nth pixel), a nodata value for the X array, and whether
longitudes wrap at the antimeridian.

## Warping from a swath

`GenImgGeolocTransformer` puts geolocation arrays on either side of the
usual CRS step. This is `examples/curvilinear_swath.rs`
(`cargo run --example curvilinear_swath`):

```rust,ignore
{{#include ../../../examples/curvilinear_swath.rs:8:}}
```

823 of the 1280 target pixels fall on the swath, the same count gdalwarp
gives for this grid.

## Warping onto a curvilinear grid

The same transformer works the other way round: put the geolocation arrays
on the target side to resample a regular grid onto a swath or model grid.
This is GDAL's `DST_METHOD=GEOLOC_ARRAY`, and rwarp matches gdalwarp there
too.

## Differences from GDAL

GDAL finds the containing quadrilateral from a backmap raster and a local
search; rwarp indexes every quadrilateral directly. On grids without folds
the answer is the same. Folded swaths, where quadrilaterals overlap, can
pick a different one. GDAL's approximate-only inverse
(`GDAL_GEOLOC_USE_MAX_ACCURACY=NO`), `SWAP_XY` and the quadtree inverse are
not implemented.
