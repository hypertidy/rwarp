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

## Cells for coverage: `QuadMesh`

The geolocation transformer is GDAL's point model. For coverage and
area methods, `rwarp::mesh::QuadMesh` treats the same grid as cells: the
straight-edged quadrilaterals through the corners (`from_corners`, or
`from_centres` with corners at the mean of four centres). It maps world
points to index space and back, and finds the owning cell, optionally
walking from a hint.

`CellMap::Bilinear` is GDAL's interior map (and matches
`GeolocArrays::geo_to_pixel` with corner samples). `CellMap::EqualArea`
splits each cell into four triangles so that areas in index space are
the same fractions of the cell as in the world. That is what makes a
burn in index space give world-area coverage on a curvilinear grid
(`design/curvilinear-frontend.md`), so coverage always uses it: the
`CellMap` a mesh is built with, `Bilinear` by default, is the map of its
point methods (`to_index`, `to_world`). The module needs no CRS backend.

`QuadMesh::burn` takes polygons and points in the grid's coordinates
and returns controlledburn's tables with the grid's own `(row, col)`:
fractions are of each cell's world area, exact for polygon edges that
are straight in those coordinates (densify polygons from another CRS
first). It walks each ring through the cells' triangles, inserting a
vertex wherever the ring crosses a cell side or split spoke, and closes
any part outside the grid along the grid's boundary.
`QuadMesh::cover_ring` does the same for one small ring into a reused
`CoverageWindow`, and `mesh::Rectilinear` is the front end for grids
given by column and row edges. Lines and approx mode are not supported
yet.

```rust,ignore
use controlledburn::{BurnOptions, Coord, Geometry, Polygon};
use rwarp::mesh::{CellMap, QuadMesh};

let (ncol, nrow) = (3, 2);
let (mut x, mut y) = (Vec::new(), Vec::new());
for j in 0..=nrow {
    for i in 0..=ncol {
        let (r, a) = (10.0 + 5.0 * j as f64, (30.0 * i as f64).to_radians());
        x.push(r * a.cos());
        y.push(r * a.sin());
    }
}
let mesh = QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap();
let ring = vec![
    Coord { x: 5.0, y: 5.0 },
    Coord { x: 30.0, y: 5.0 },
    Coord { x: 30.0, y: 12.0 },
    Coord { x: 5.0, y: 12.0 },
    Coord { x: 5.0, y: 5.0 },
];
let res = mesh
    .burn(&[Geometry::Polygon(Polygon { rings: vec![ring] })], BurnOptions::coverage())
    .unwrap();
assert!(!res.edges.is_empty());
```

### Seams and folds

Outer walls that meet in the world are crossed like any side, with
nothing to declare. That covers a periodic grid whose seam has no jump in
its coordinates (column line `ncol` on column line 0) and a tripolar ocean
grid, whose last row line folds onto itself. Give the grid its corners in
a projection in which it is one sheet. For an ocean grid's tripolar
north, Lambert azimuthal equal-area on the north pole does this, and its
planar areas are true areas. It holds down to the Southern Ocean. Near
80 S it flattens cells 500 to 1, and an ice-shelf row there can become a
bow-tie, which is refused. A tripolar grid can't be a mesh in longitude
and latitude, since the geographic pole is inside one of its cells.
Coordinates that jump at the seam (longitudes 360 apart) are not handled
yet.

### CF bounds and ocean-model halos

`QuadMesh::from_cf_bounds` builds a mesh from each cell's own four
vertices: CF's `lon_bnds(j, i, 4)`, or SCRIP's `grid_corner_lon(grid_size,
4)` with `grid_dims` giving the columns, then the rows. The vertex order
is found from the data (CF's, or any rotation or reflection of it), and
neighbours must agree on the corners they share.

Ocean-model files store some cells twice (a halo), and those copies have
to go or they are counted twice. A `Halo` names the columns and rows to
drop. `Halo::NEMO_F_PIVOT` (eORCA1, ORCA1) and `Halo::NEMO_T_PIVOT`
(ORCA2) are NEMO's layout as CMIP6 files have it: the last two columns
repeat the first two; the first row is land with collapsed bounds; the
last row repeats one below it. At a T pivot the fold runs through the
centres of the row before that one, so each of its cells is the same
ground as its mirror. The mesh keeps the half of each below the fold, and
a model cell there is two mesh cells, one half each.

A wrong layout is refused rather than counted twice. Neighbours that
don't share corners are refused, and so are cells that overlap, which is
what a halo left in gives (`QuadMesh::find_overlap`). The flag for the
centre fold is refused on a row that doesn't mirror itself.

```rust,ignore
use rwarp::mesh::{CellMap, Halo, QuadMesh};

// vertices_longitude / vertices_latitude (ny, nx, 4) from an eORCA1 file,
// projected to Lambert azimuthal equal-area on the north pole
let mesh = QuadMesh::from_cf_bounds(nx, ny, &xb, &yb, Halo::NEMO_F_PIVOT, CellMap::EqualArea)?;
// mesh cell (row, col) is the file's (row + 1, col)
```

On IPSL's CMIP6 eORCA1 north of 76 S and ORCA2 north of 49 N, mesh cell
areas agree with `areacello` to 1e-4 in total. A cap north of 60 N, across
the fold, both pivots and the seam, is covered to 1e-10 of its area.
Neither whole file builds as it stands; the cells that stop them are all
land:

- eORCA1's rows south of 76 S are too flattened in that projection.
- IPSL's ORCA2 bounds are corrupt over land between 9 N and 49 N.

Drop such rows with the halo's `rows`, or repair the bounds, for now.

Against ESMF, on an analytic tripolar cap with a seam and a fold
(`tests/esmf_tripolar.rs`), with each destination cell's outline
covered by `cover_ring`:

- **ESMF in the same plane** (Cartesian) has the same cells. The weights
  agree to rounding.
- **ESMF on the sphere** joins corners by great circles, and rwarp joins
  them by straight lines in the projection. For cells 13 to 363 km across,
  weights differ by up to 2.6e-3 and cell areas by up to 9e-4. The area
  difference falls with the square of cell size.

There is no `Weights` from a mesh yet: the test builds them from
`cover_ring`, one destination cell at a time.

### Overlapping swaths

A mesh is one sheet: `from_corners` refuses cells of both orientations,
which is what a swath whose scans overlap (the MODIS bow-tie) gives when
its scans are stored as one array. Each scan alone is one sheet, so build
one mesh per scan and offset its rows to the swath's:

```rust,ignore
let n = rows * ncol; // pixels per scan
for s in 0..nscan {
    let (sx, sy) = (&x[s * n..(s + 1) * n], &y[s * n..(s + 1) * n]);
    let mesh = QuadMesh::from_centres(ncol, rows, sx, sy, CellMap::EqualArea).unwrap();
    let res = mesh.burn(&geoms, BurnOptions::coverage()).unwrap();
    // res holds the scan's rows: swath row = s * rows + row
}
```

Each source pixel gets its own coverage. Where scans overlap, the ground
is covered twice and each scan reports it in full; how to combine them
there (sum, mean, the first scan only) is the caller's choice. The full
example, which checks the area each scan covers, is in the
[`mesh` module docs](../api/rwarp/mesh/index.html#overlapping-swaths-one-mesh-per-scan).

A row or column stored twice, such as an ocean model's halo, keeps its
orientation, and `from_corners` doesn't check for it (the check costs
about what building the mesh does). `QuadMesh::find_overlap` finds it,
and `from_cf_bounds` refuses it. A declared period for longitudes is
the rest of stage 4 ([#37](https://github.com/hypertidy/rwarp/issues/37)).

`mesh` is public and unstable: it changes more often than the rest of
the crate and may break at any 0.x minor release, though not at a patch
release.

## Area methods from a swath

The exact footprint (`Footprint::Exact`, and `exact_weights`) measures a
geolocation source with these cells. Each target pixel's outline goes to
the source's world coordinates and is walked onto a `QuadMesh` of the
source window. Each pixel is the quadrilateral through the positions of
its four corners. So weights are fractions of each source pixel's world
area, and `Average` weights each pixel by its world area. With
`TopLeftCorner` samples at whole-number steps that is GDAL's bilinear
model exactly. With `PixelCenter` the corners are the means of four
centres, as `from_centres` makes them.

With `with_lon_wrap(true)`, a swath across the antimeridian works the
same way. Its longitudes are unwrapped by continuity, so cells across the
seam are whole, and so is each destination pixel's outline. A destination
pixel round a pole is closed along latitude 90 or -90. Where the arrays
store their seam makes no difference to the weights, beyond rounding.

`GeolocArrays::world_cells()` says whether this applies to a set of
arrays: not with a missing sample, not where the swath folds, and not for
a swath over a pole, whose longitudes don't unwrap consistently. Those
fall back to source pixel space. Point kernels are unchanged.

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
