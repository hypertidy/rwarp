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
Coordinates that jump at the seam, such as longitudes, put its two walls
in different places in the plane: declare the period
([below](#longitude-a-declared-period)).
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

### Longitude: a declared period

A grid in longitude and latitude has its seam's two walls 360 degrees
apart, so they don't meet in the plane. Declare the period with
`with_period(360.0)`, on a `QuadMesh` or a `Rectilinear`:

- `burn` takes each polygon at each whole period at which it overlaps
  the grid, and adds up the pieces. Polygons given from -180 to 180 burn
  onto a grid from 0 to 360, and one across the seam covers cells at both
  ends. Polygons are taken as they are in the plane. Give one across the
  antimeridian with continuous longitudes (170 to 190), and close one
  round a pole along the pole, as Natural Earth's Antarctica is.
- `locate`, `to_index` and `Rectilinear::cell` find a point at any whole
  period.
- `MeshTransformer` unwraps each target pixel's outline and walks it at
  each whole period at which it reaches the mesh. An outline round a pole
  is closed along latitude 90 or -90.

Where the seam is makes no difference beyond rounding.

The corners must be continuous. Longitudes stored with a jump, from
`[-180, 180)` or `[0, 360)`, are refused with a hint. `unwrap_grid`
(corners or centres) and `unwrap_bounds` (CF bounds, SCRIP corners)
unwrap them:

```rust,ignore
use rwarp::mesh::{unwrap_bounds, CellMap, Halo, QuadMesh};

// ORCA2's bounds as CMIP6 stores them, longitudes in [-180, 180): the
// cyclic columns and the northern rows dropped
unwrap_bounds(182, 149, &mut lon_bnds, 360.0)?;
let halo = Halo { cols: [1, 1], rows: [1, 55], ..Halo::NONE };
let mesh = QuadMesh::from_cf_bounds(182, 149, &lon_bnds, &lat_bnds, halo, CellMap::EqualArea)?
    .with_period(360.0)?;
```

`with_period` refuses a cell half a period wide or more, which is a cell
round a pole, such as the one over the North Pole on a tripolar grid.
Give a tripolar grid's north in a polar projection, as above. Areas in longitude and latitude are planar in
square degrees: a weight is the covered share of a cell's square degrees.

### Warping from a mesh

`MeshTransformer` makes a mesh the source of a warp, with one value per
cell: `new(mesh, mesh_crs, dst_gt, dst_crs)`. It is a `Transformer` whose
source pixels are the mesh's cells, so `exact_weights` gives conservative
weights from the mesh to the grid, and `warp` with `Footprint::Exact` and
an area method regrids the data. Each target pixel's outline goes to the
mesh's coordinates and is walked onto its cells, across the seam and the
fold. A weight is the share of a cell's area that the outline covers.

```rust,ignore
use rwarp::mesh::MeshTransformer;
use rwarp::weights::{exact_weights, ApplyOptions, Normalization, WeightOptions};

// the eORCA1 mesh above, in LAEA on the north pole, to 0.5 degrees north of 60 N;
// buf is a SourceBuffer of the mesh's cells, one value each
let laea = "+proj=laea +lat_0=90 +lon_0=0 +R=6371229 +units=m";
let dst_gt = [-180.0, 0.5, 0.0, 90.0, 0.0, -0.5];
let t = MeshTransformer::new(mesh, laea, dst_gt, "+proj=longlat +R=6371229")?;
let w = exact_weights(&t, &t.source_window(), 720, 60, &WeightOptions::new(4));
let sst: Vec<f32> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
```

The buffer holds the mesh's cells row-major, so mesh cell `(row, col)` is
source pixel `(col, row)`. Point kernels such as `Bilinear` map a target
pixel into the mesh's index space through its cell map. They read
neighbouring cells in index space, not across the seam or fold. A mesh
in longitude and latitude with a declared period takes target pixels
across its seam and round a pole, as above.

Against ESMF, on an analytic tripolar cap with a seam and a fold
(`tests/esmf_tripolar.rs`), with weights from `MeshTransformer`:

- **ESMF in the same plane** (Cartesian) has the same cells. The weights
  agree to rounding.
- **ESMF on the sphere** joins corners by great circles, and rwarp joins
  them by straight lines in the projection. For cells 13 to 363 km across,
  weights differ by up to 2.6e-3 and cell areas by up to 9e-4. The area
  difference falls with the square of cell size.

### Warping onto a mesh

`MeshTarget` makes a mesh the target: its pixels are the mesh's cells,
`ncol` by `nrow` (`target_size`), one value each. Each target pixel goes to
the mesh's coordinates and then to the source. `from_grid` does that for a
grid with a CRS, wrapping a longitude/latitude source's longitudes as
gdalwarp does, so cells across its seam or over a pole are measured
whole. `MeshTarget::new(mesh, inner)` takes any transformer whose
destination geotransform is `WORLD_GT`: geolocation arrays, or another
mesh through `MeshTransformer`.

```rust,ignore
use rwarp::mesh::MeshTarget;
use rwarp::weights::{exact_weights, lonlat_measure, GridWindow, WeightOptions};

// a global 0.25 degree grid onto the eORCA1 mesh above, in LAEA on the
// north pole; areas on the sphere
let laea = "+proj=laea +lat_0=90 +lon_0=0 +R=6371229 +units=m";
let src_gt = [-180.0, 0.25, 0.0, 90.0, 0.0, -0.25];
let t = MeshTarget::from_grid("+proj=longlat +R=6371229", src_gt, [1440, 720], mesh, laea)?;
let [w, h] = t.target_size();
let opts = WeightOptions::new(8).with_measure(lonlat_measure(src_gt).unwrap());
let wts = exact_weights(&t, &GridWindow::new(1440, 720), w, h, &opts);
```

Each cell is measured on its own, so the mesh's seam and fold don't
matter on this side. Its outline is the cell, sides straight in the
mesh's coordinates, taken to the source at `densify` points per side and
joined straight there. Near the pole, in longitude and latitude, that
needs a higher `densify`: the cell over the pole has its corners 90
degrees of longitude apart. `warp` onto the mesh works the same way, with
point kernels such as `Bilinear` sampling each cell's centre.

Against ESMF, on the same tripolar cap as the target:

- **ESMF in the same plane** (Cartesian) has the same cells, and the
  weights agree to rounding. That holds from a grid in the plane, and
  mesh to mesh from a lon/lat grid's corners in the plane.
- **ESMF on the sphere.** The weights differ more this way round, by up
  to 3e-2 on the cap's smallest cells, 13 to 55 km across near its grid
  poles. There, a long source edge drawn as a great circle rather than a
  straight line moves a large share of a small cell. From a grid in the
  plane, over cells more than 100 km across, the difference is 3e-3, as
  in the other direction.

### Whole ORCA files

One mesh can't hold a whole tripolar ocean grid. North-polar LAEA turns
its far south into bow-ties, and longitude and latitude can't hold the
cell over the North Pole. Model files also have land cells with damaged
bounds: IPSL's CMIP6 eORCA1 has thin flipped strips south of 75 S, and
its ORCA2 has garbled cells between 9 N and 49 N. No ocean cell is
damaged in either.

`MeshSet` takes the whole file. Its rows are split into bands, each in a
projection that holds it. Cells that aren't valid there, or overlap
others, are left out where a mask allows: give it the land. What's left
is covered by rectangular pieces, each a mesh of its own, so a cell left
out is a hole and every other cell keeps the file's geometry.

```rust,ignore
use rwarp::mesh::{Band, Halo, MeshSet, MeshTransformer};

// lon_b, lat_b: the file's bounds (182 x 149 x 4 for ORCA2); land: !(areacello > 0)
// south and north: lon/lat to south- and north-polar LAEA on the model's sphere,
// as CrsTransforms; crs: the same two as CRS strings
let bands = [
    Band { rows: 0..54, to_world: &south },   // to about 20 S
    Band { rows: 54..147, to_world: &north },
];
let set = MeshSet::from_cf_bounds(182, 149, &lon_b, &lat_b, Halo::NEMO_T_PIVOT, &bands, Some(&land))?;

// conservative weights to a 1 degree grid, sources in the set's numbering
let gt = [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0];
let opts = WeightOptions::new(2);
let w = set.weights_to(|band, mesh| MeshTransformer::new(mesh, crs[band], gt, "+proj=longlat +R=6371229"), 360, 180, &opts)?;
```

`weights_onto` goes the other way, onto the set's cells. Both bands are
equal-area on one sphere, so areas from either are square metres of it
and weights add up across them. Where the bands meet, each side's cells
are straight in their own projection and bow apart between corners,
leaving a thin gap covered by neither. At the equator it takes about
dlam² / (6 dphi) of each cell next to it, where dlam and dphi are the
cells' width and height in radians: 2.3% for ORCA2's 2 by 0.5 degree cells
there. So split where the cells are taller, south of the equatorial
refinement; at 20 S it is under 1%.

On IPSL's CMIP6 files (checked in scratch, not in CI), split at 20 S:

| | eORCA1 | ORCA2 |
|---|---|---|
| Built in | 0.13 s, 50 pieces | 0.05 s, 56 pieces |
| Land cells left out | 716 (no ocean) | 801 (no ocean) |
| Ocean area vs `areacello` | 0.9999 | 0.9994; 0.9998 without the halved fold row, whose cells `areacello` counts whole |
| To 1°: every cell shared out to | 1e-12 (1.0 s) | 8e-14 (1.2 s) |
| From 1°, onto the set: covered to | 5e-10 (0.4 s) | 1e-9 (0.1 s) |

The worst cells covered are 0.03 km² slivers next to the grid poles,
where 5e-10 is rounding on a cell a millionth of a source pixel.

`SetTransformer` makes the set the source of a warp, so `warp()` takes a
field on the whole grid, one value per cell in the set's numbering, with
any method:

```rust,ignore
use rwarp::mesh::SetTransformer;
use rwarp::warp::{warp, Footprint, ResampleAlg, SourceBuffer, WarpOptions};

// tos: the file's field less its halo, one value per cell of the set, NaN on land
let t = SetTransformer::new(&set, &crs, gt, "+proj=longlat +R=6371229")?;
let [ncol, nrow] = set.size();
let src = SourceBuffer { data: &tos, ncol, nrow, nbands: 1, col_off: 0, row_off: 0 };
let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify: 2 });
let out = warp(&t, &src, 360, 180, &opts);
```

With the exact footprint, each destination pixel's outline goes to both
bands' projections and is walked onto every piece it reaches. So
`Average`, `Mode`, `Min`, `Max` and the rest see every cell under the
pixel, each weighted by its area. `Average` is `weights_to` and `apply`
bit for bit, and quicker: 0.9 s on either file at densify 2, against 1.1
and 1.2 s. Point kernels find the piece whose cell holds the pixel's
centre and read neighbours in the set's numbering, across pieces and
bands, as GDAL reads geolocation arrays (0.02 s). A polar projection
can't hold its far pole: a pixel round the South Pole comes out round
the whole north band there, turned the other way. An outline turned
against its band isn't measured in it, so an Antarctic grid takes
nothing from the north band.

`warp_to_par` gives the same values on several threads, each with a
`SetTransformer` of its own: `|| SetTransformer::new(&set, &crs, gt,
dst_crs)`. ORCA2's `Average` at densify 8 takes 2.3 s on one thread,
0.34 s on 8 and 0.14 s on 32.

`MeshSet::burn` burns polygons and points in longitude and latitude
onto the set's cells. It takes the bands the set was built with and a
tolerance in the bands' world units, and returns controlledburn's tables
in the set's numbering:

```rust,ignore
use controlledburn::BurnMode;

// countries: Vec<Geometry> in lon/lat (edges straight in degrees)
let r = set.burn(&countries, &bands, 1.0, BurnMode::Coverage)?;
// r.runs and r.edges: (row, col) of the set's cells, fraction of each cell's area
```

Each band takes the part of each polygon within its cells' latitudes,
widened by its tallest cell. A polar LAEA can't hold the far pole, so
without that cut Antarctica in the north band, or a whole-world box,
would come out inside out. Edges are straight in longitude and latitude.
bigcurve densifies them into each band's projection until no chord is
further than the tolerance from the edge's image. A fraction is then
good to about the tolerance times the length of edge in the cell, over
the cell's area. A polygon cut at 180° or closed along a pole comes out
of a polar projection with both sides of the cut on one line. Each ring
is split where it meets itself, and that line is taken out before the
walk. Where the bands meet, the thin gap between their cells is burned
onto no cell.

On the same files (scratch), at a tolerance of 1 m:

| | eORCA1 | ORCA2 |
|---|---|---|
| A whole-world box | every cell kept covered in full | the same |
| Five boxes on 1° lines (Arctic, Antarctic, across 180°, across 20 S), against `weights_onto` from a 1° indicator at densify 64 | 8e-6 | 6e-6 |
| Natural Earth 10 m countries (258, 548k vertices) | 0.70 s | 0.96 s |
| The same with `burn_par` on 8 threads | 0.40 s | 0.44 s |
| The same with `burn_par_with` on 8 threads | 0.15 s | 0.18 s |

The comparison is two separate walks: a polygon into the mesh's index
space, and each cell's outline into the source's pixels. On a 30° wide
land cell of ORCA2 the two differed by 1e-3 at densify 16, and the gap
fell sixteenfold at densify 64, as the outlines' error should.
Antarctica lands only in the south band.

Each piece walks only what can reach it. A one-row piece round most of
a latitude circle has a box the size of its band's disk. A grid of tiles
marks where its cells are, so polygons elsewhere are skipped, and long
runs of edges away from it are cut short, 32 edges at a time where
their box is away from it. That doesn't change the result: the walk
sees the same crossings and the same winding round its interior
point. What's left of the time is projecting the polygons and
walking the edges that are near.

`MeshSet::burn_par(geoms, bands, tolerance, opts, threads)` gives the
same records on several threads. The pieces filter, walk and burn their
polygons in parallel. Projecting into each band stays on the calling
thread, since a libproj transform can't be shared between threads. That
is the floor: about 0.3 s for the countries at 1 m, and 1.6 s at
0.01 m, where 8 threads take ORCA2 from 5.6 s to 2.2 s.

`MeshSet::burn_par_with(geoms, bands, tolerance, opts, threads,
to_world)` projects on the threads too, each through a transform of its
own. `to_world(k)` builds band `k`'s, the same as `bands[k].to_world`:

```rust,ignore
let lonlat = "+proj=longlat +R=6371229";
let laea = ["+proj=laea +lat_0=-90 +R=6371229", "+proj=laea +lat_0=90 +R=6371229"];
let to_world = |k: usize| rwarp::crs::crs_pair(lonlat, laea[k]).map(|(_, f)| f);
let r = set.burn_par_with(&countries, &bands, 1.0, BurnMode::Coverage, 8, to_world)?;
```

The records are the same again. On 8 threads ORCA2 takes 0.92 s at
0.01 m. A libproj transform takes about 5 ms to build, and many threads
building at once slow each other down, so on a light burn the most
threads isn't always fastest. For the countries at 1 m through libproj,
16 threads took 0.20 s and 32 took 0.31 s.

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
and `from_cf_bounds` refuses it.

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
