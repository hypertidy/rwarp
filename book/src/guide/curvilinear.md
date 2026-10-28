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
contains the point and solves the exact inverse bilinear interpolation
(two ways to find it, below).

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

## Two inverse methods

From a coordinate to a pixel, both methods find a quadrilateral of
samples that contains the point and solve the same exact inverse bilinear
interpolation in it. They differ in how they find it, chosen with
`GeolocOptions::with_inverse`:

- **`GeolocInverse::QuadIndex`** (the default) indexes every quadrilateral,
  plus one ring of extrapolated ones outside the arrays, and takes the
  first that contains the point in row-major order. Its domain is the union
  of the quadrilaterals.
- **`GeolocInverse::Backmap`** is GDAL's method (`GEOLOC_INVERSE_METHOD`
  `BACKMAP`, GDAL's default): a float32 raster over the arrays' extent,
  filled by forward-projecting the arrays and patching holes as GDAL does,
  gives an approximate pixel, and a spiral search of up to 3 squares around
  it (5 near the poles) finds the quadrilateral. It is ported in GDAL's
  arithmetic order, so warps through it are identical to gdalwarp's, pixel
  for pixel. `GeolocInverse::BackmapEstimate` stops at the backmap's
  estimate (`GDAL_GEOLOC_USE_MAX_ACCURACY=NO`). The backmap's oversampling
  is `with_backmap_oversample` (`GEOLOC_BACKMAP_OVERSAMPLE_FACTOR`, 1.3).

```rust,ignore
let opts = GeolocOptions::default()
    .with_convention(GeolocConvention::PixelCenter)
    .with_lon_wrap(true) // what GDAL does for lon/lat in [-180, 180]
    .with_inverse(GeolocInverse::Backmap);
```

Use `Backmap` to reproduce gdalwarp, and `QuadIndex` where the answer
should not depend on a sampled raster. On a grid whose quadrilaterals do
not overlap, the two agree wherever both succeed inside the raster. On
the test cases (`tests/geoloc_backmap.rs`, 2600 to 4900 target pixels
each, nearest neighbour):

| Case | Target pixels that differ | Why |
|---|---|---|
| curved swath, by centres or corners, to lon/lat or polar stereographic | 0 | |
| swath across the antimeridian | 0 | |
| swath with holes in its arrays and source nodata | 0 | |
| overlapping scans ("bow-ties") | 19 of 2880 | where scans overlap, the quad index takes the earlier scan, GDAL the later; each pixel is one row apart |
| ring of cells around the south pole, crossing the antimeridian | 53 of 4900 | the arrays wrap 360 degrees with no quadrilateral across the seam between their last and first column. GDAL's backmap, averaged across the antimeridian, sends the search to the last columns for points whose quadrilateral is in the first few, beyond its radius: GDAL leaves 51 pixels empty along the antimeridian that the quad index fills. Over the seam itself the extrapolated quadrilaterals of both sides overlap, and 2 pixels go the other way |

Outside the raster GDAL's search also succeeds at more points (squares
further than one beyond the top and left edges, within its search radius),
which no kernel reads.

## Other differences from GDAL

- rwarp treats NaN samples as missing, as well as the X array's nodata
  value; GDAL compares with the nodata value only.
- Whether longitudes wrap at the antimeridian is the explicit
  `with_lon_wrap`. GDAL decides it from the SRS (geographic, every
  longitude in `[-180, 180]`) or forces it with
  `GEOLOC_NORMALIZE_LONGITUDE_MINUS_180_PLUS_180`.
- `SWAP_XY` and GDAL's experimental quadtree inverse
  (`GDAL_GEOLOC_INVERSE_METHOD=QUADTREE`) are not implemented. GDAL keeps
  the backmap of arrays of 24 megapixels or more in temporary files, with
  the same arithmetic; rwarp keeps it in memory.
