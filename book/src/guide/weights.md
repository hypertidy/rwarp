# Conservative weights

`Footprint::Exact` measures how much of each source pixel every target
pixel's outline covers, then averages and throws the measurements away.
`rwarp::weights` keeps them: a sparse table with one entry per overlapping
(target cell, source cell) pair. That table is the weight matrix of
first-order conservative remapping, ESMF's `conserve` method. Compute it
once, then apply it to every band, time step or variable on the same grids.

```rust,ignore
use rwarp::weights::{exact_weights, ApplyOptions, GridWindow, Normalization, WeightOptions};

// The source grid needs no data, only its size (and offset, for a window).
let src = GridWindow::new(src_ncol, src_nrow);
let w = exact_weights(&transformer, &src, dst_ncol, dst_nrow, &WeightOptions::new(4));

// Apply to any pixel-interleaved buffer on that source grid.
let opts = ApplyOptions::new().with_src_nodata(Nodata::Value(-9999.0));
let out: Vec<f32> = w.apply(&source_buffer, Normalization::FracArea, &opts);
```

The transformer is the same one `warp()` takes (target pixel to source
pixel), so anything rwarp can warp it can make weights for.

## What's in a `Weights`

| Field | Per | Meaning |
|---|---|---|
| `dst_row`, `dst_col` | entry | target cell, full-grid indices (offsets applied) |
| `src_row`, `src_col` | entry | source cell, full-grid indices |
| `frac` | entry | area of the overlap as a fraction of the source cell, unrounded `f64` |
| `dst_area` | target cell | area of the target cell's outline, in source cells (in world units for a geolocation or mesh source); `NaN` if it failed to transform |
| `src_cell_area` | entry | for a geolocation or mesh source, the source cell's world area; from `PolyMesh::weights_to`, the face's area; empty otherwise |
| `dst_off`, `dst_size` | | the target window the weights are for |

Areas are measured in source pixel space, where a source cell has area 1.
For a source on a regular grid in its own CRS that is area in the source
CRS, up to a constant. Target cells have edges straight in target pixel
space, each split into `densify` segments whose end points go through the
transformer exactly.

A geolocation source (geolocation arrays on the source side of
`GenImgGeolocTransformer`) is measured in its own world coordinates
instead, since its pixels are not the same size there. The outline's
points go to source world coordinates and are joined by straight lines
there. Each source pixel is the quadrilateral through its corners' world
positions, and the outline is walked onto those cells as
`QuadMesh::burn` walks a polygon. `frac` is then a fraction of the
cell's world area, `Weights::src_area(k)` is that area, and `dst_area`
is the outline's world area. This is planar area in the arrays'
coordinates: square degrees for longitude and latitude, true area for
arrays in an equal-area CRS. Arrays whose cells are not whole fall back
to source pixel space; see [Limits](#limits).

A curvilinear grid given by its cells' corners, such as an ocean model's
tripolar grid, is a source through `MeshTransformer` (in `rwarp::mesh`).
It is measured the same way, in the mesh's world coordinates, and the
outlines are walked across the mesh's seam and fold. See
[Curvilinear grids](curvilinear.md#warping-from-a-mesh).

A mesh is a target through `MeshTarget`: each of its cells is a target
pixel, its outline the cell itself (sides straight in the mesh's
coordinates, densified), measured in the source's space. So `frac` is a
cell's share of each source pixel, under any measure the source takes
(`lonlat_measure` for areas on the sphere), and `dst_area` is the cell's
measure there. See [Curvilinear grids](curvilinear.md#warping-onto-a-mesh).

`frac` is relative to the source cell, not the target, so it doesn't depend
on how the target is normalised. `dst_area` gives either normalisation:

With `a` the source cell's area (`Weights::src_area`, 1 in source pixel
space):

- **`Normalization::FracArea`**: `S = frac a / (sum of frac a over the
  target cell)`, the area-weighted mean over the part of the cell that has
  source. This is what `warp()` computes. In ESMF it's `fracarea`; in
  xESMF, `conserve_normed`.
- **`Normalization::DestArea`**: `S = frac a / dst_area`. A cell half off the
  source gets half the weight, so `sum(value * dst_area)` over the target
  equals the integral of the source over the region the target covers.
  ESMF `destarea`, xESMF `conserve`.

**The default is `FracArea`**, which is not ESMF's: ESMF's default is
`destarea`, as is xESMF's `conservative`. To compare with them, ask for
`Normalization::DestArea`.

`Weights::normalised(norm)` returns `S` per entry. `Weights::apply`
also skips source nodata: `FracArea` renormalises over the valid entries,
and `DestArea` doesn't (a missing source cell counts as zero).

## Conservation

Neighbouring target cells build their outlines from the same transformed
points, so together they tile source pixel space with no gaps or overlaps.
For every source cell the target grid covers completely, its `frac` sums
to 1 over the target cells. Where the grid's edge cuts a source cell, the
sum is the covered fraction. This needs no reference to check, and
`tests/weights.rs` checks it against the coverage of the target grid's
outer boundary:

| Case | Worst \|sum - 1\| (covered) | Worst \|sum - covered fraction\| (edge) | Integral, `DestArea` (relative) |
|---|---|---|---|
| regular to rotated (0.3 rad, 2.3:1), densify 1 / 4 | 2.8e-14 / 1.2e-14 | 7.5e-15 / 5.2e-15 | 1.6e-15 / 1.8e-15 |
| lon/lat 0.5° to EPSG:3031 at 25 km, densify 1 / 4 | 6.0e-14 / 1.6e-14 | 3.3e-14 / 1.2e-14 | 3.8e-15 / 3.1e-15 |
| lon/lat 0.5° to LAEA at 20 km, densify 1 / 4 | 2.1e-15 / 4.4e-16 | 3.3e-16 / 3.3e-16 | 1.9e-15 / 2.4e-15 |

## Second order

First order gives each target cell the area-weighted mean of the source
values. ESMF's `conserve_2nd` (Kritsikis et al. 2017) gives each source
cell a plane instead, `f + g · (x − c)`, and each overlap the plane's
mean over it. The result is still conservative, and a smooth field comes
out far closer.

```rust,ignore
let opts = WeightOptions::new(densify).with_moments(true);
let w = exact_weights(&t, &src, ncol, nrow, &opts);
let values: Vec<f32> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
```

- **`with_moments(true)`** records each overlap's first moments
  (`Weights::moment`), from which its centroid follows. controlledburn's
  window computes them exactly, beside the area.
- **The gradient `g`** comes from the cell's eight neighbours, by Green's
  theorem round the polygon of their centres, as ESMF does. It is zero,
  so first order for that cell, where the neighbours don't surround the
  cell's centre: at a corner of the source window.
- **`c` is the centre of the part of the cell the table covers**, as in
  ESMF. The gradient terms then integrate to zero over that part, so
  under `DestArea` the integral is conserved exactly, even where a
  target's edge or a mask cuts a source cell.
- **`apply_second_order` estimates the gradients from the field it is
  given.** A neighbour without data drops out of them, as a cell without
  data drops out of `apply`.
- **`Weights::second_order(&src, valid, areas)` folds the gradients
  into one table**, as ESMF's weight files hold it. Some of its weights
  are negative. Use it to write files, for a fixed set of cells with
  data. `areas` is `None` for a source measured in its pixels, and
  `weights::source_cell_areas(&t, &src)` for one with world cells.

What it gives, from `tests/second_order.rs` and
`tests/esmf_second_order.rs`:

- **Linear fields come out exactly.** A field linear in source pixels is
  exact to 1e-14 in the interior, planar and on the sphere, where first
  order is off by 0.16.
- **The error is second order.** On a quadratic, halving the cell size
  quarters it exactly. On a smooth sine it is 57 times smaller than first
  order's.
- **The weights are ESMF's**, to 6e-14, for a grid onto a grid, inside
  the source and over its corner. Projected (0.5° lon/lat to EPSG:3031)
  they differ by up to 0.022. There every overlap is a general polygon,
  and ESMF centres it on the mean of its vertices rather than its
  centroid, so ESMF's second order is off by up to 0.035 on a linear
  field and rwarp's by 1e-14. With ESMF's vertex means in its place,
  rwarp's table is ESMF's to 1e-12, away from target corners that fall
  exactly on a source cell's wall.

Moments come from any grid source: a regular grid, planar or under
`lonlat_measure`, and a grid with world cells (geolocation arrays, a
`QuadMesh` through `MeshTransformer`, a `MeshSet`), onto a grid or onto
a mesh (`PolyMesh::weights_onto`, `MeshSet::weights_onto`). For world
cells the planes are in the cells' index space, where rwarp measures
their overlaps: each cell there is a unit square holding its world area
evenly, so the moments are exact centroids and the neighbours are the
eight round a cell in the full grid, whatever piece or band they're in.
That is exact for a field linear in the grid's indices, and second order
for a smooth field on a smooth grid, but not exact for a field linear in
world coordinates (ESMF works in world coordinates instead). On a curved
grid a world quadratic's error falls 3.9 times as the cells halve
(`tests/second_order_world.rs`).

From a `PolyMesh`'s faces to a grid (`weights_to` with moments), the
faces have no index space, so their planes are in the grid's, where the
overlaps are measured: its pixels, or a curvilinear grid's indices.

```rust,ignore
let w = mesh.weights_to(&t, &grid, &WeightOptions::new(densify).with_moments(true))?;
let values: Vec<f32> = mesh.apply_second_order(&w, &faces, Normalization::FracArea, &ApplyOptions::new());
let folded = mesh.second_order(&w, None);
```

A face's neighbours are the faces sharing a node with it, as in ESMF, each
centred on the part of it the table covers. A field linear in the grid's
space comes out exact to 3e-14 where the faces are whole in the grid
(first order is off by 2.3), and on a quadratic halving both mesh and
grid divides the error by 3.9 (`tests/second_order_faces.rs`). For a mesh
in longitude and latitude onto a grid in them, the planes are linear in
longitude and latitude. At the grid's seam the gradients are one-sided: a
face the seam cuts is first order, and a face's neighbour across it is
left out.

**A limiter.** Next to a sharp front a plane can reach past the values
round it, so a second-order value overshoots what it is drawn from: below
the lowest of its sources, or above the highest. ESMF has no limiter, and
none is the default. Ask for Barth and Jespersen's:

```rust,ignore
let opts = ApplyOptions::new().with_limiter(Limiter::BarthJespersen);
let values: Vec<f32> = w.apply_second_order(&buf, Normalization::FracArea, &opts);
```

Each source cell's gradient is scaled so that its plane, at each of its
overlaps' centroids, stays within the values of the cell and its
neighbours; under `FracArea` each target value is then within its
sources' and theirs. One factor per cell keeps the integral as it was.
It works for grids, grids with world cells and a mesh's faces, band by
band (`tests/second_order_limiter.rs`). It costs some accuracy on a smooth
field, as it flattens planes at extrema too: on a sine, 16 to 72% in rms,
the more the less the field is resolved, still 7 to 49 times closer than
first order. It depends on the field, so a folded table can't hold it.

**On the sphere** rwarp's planes are in longitude and latitude, under the
spherical measure, and ESMF's in three dimensions, so the two are
different second-order methods, and `tests/esmf_second_order_sphere.rs`
compares their accuracy, each against exact cell means in its own cells.
Onto a grid both are second order, close for lon/lat onto lon/lat, rwarp
19 times closer projected; onto a mesh rwarp is 5 to 12 times closer
(ESMF centres an overlap on its vertices' mean). From a mesh's faces rwarp
is 1.8 to 4.4 times closer away from the grid's seam, but close to first
order within 15° of it, where ESMF, with no seam, is 1.3 to 3.6 times
closer: a face the seam cuts is first order, as the table doesn't carry
the grid's period.

## Meshes of polygons

An unstructured mesh, ICON's triangles, MPAS's hexagons, a UGRID or ESMF
mesh, is a `PolyMesh` (in `rwarp::mesh`): its nodes' coordinates and, for
each face, its nodes in order round it. `weights_onto` gives the weights
from a grid onto the mesh, and `weights_to` from the mesh to the grid:

```rust,ignore
use rwarp::mesh::{PolyMesh, WORLD_GT};
use rwarp::transform::GenImgProjTransformer;
use rwarp::weights::{lonlat_measure, GridWindow, WeightOptions};

// faces as UGRID's face_node_connectivity: up to 6 nodes a face, 0-based,
// padded with -1; straight edges, the shorter way round in longitude
let mesh = PolyMesh::from_padded(lon, lat, &face_nodes, 6, 0)?.with_period(360.0)?;
// or, for an ICON or MPAS mesh, great circles between the nodes:
// let mesh = PolyMesh::from_padded(lon, lat, &face_nodes, 6, 0)?.with_great_circles()?;

// from the mesh's coordinates to the grid's pixels
let t = GenImgProjTransformer::new("EPSG:4326", grid_gt, "EPSG:4326", WORLD_GT)?
    .with_source_size([ncol, nrow]);
let grid = GridWindow::new(ncol, nrow);
let opts = WeightOptions::new(4).with_measure(lonlat_measure(grid_gt).unwrap());
let onto = mesh.weights_onto(&t, &grid, &opts)?; // grid to mesh
let to = mesh.weights_to(&t, &grid, &opts)?; // mesh to grid
```

Each face is measured as a target pixel is. Its edges are cut into
`densify` pieces in the mesh's coordinates (or refined to
`refine_tolerance`), go through the transformer to the grid's pixels, and
are measured there under the grid's measure, across the seam and round a
pole as above. Both directions come from the same overlaps, one turned
round, so the pair is consistent. Faces that share an edge get bitwise
the same points on it, so they tile the grid: where the mesh covers a
grid cell, its `frac` sums to 1 over the faces, as in
[Conservation](#conservation).

- **Onto the mesh**, face `f` is target cell `(0, f)`: `dst_size` is
  `[nface, 1]`, `dst_area` the faces' areas, and `apply` takes the grid's
  values and gives one per face.
- **From the mesh**, `src_col` is the face, `src_cell_area` its area, and
  `apply` takes the faces' values as a buffer `nface` wide and 1 high.

`weights_onto_par` and `weights_to_par` do the same on threads, with a
transformer built on each (`|| GenImgProjTransformer::new(...)`), and give
the same table bit for bit.

Edges are straight in the mesh's coordinates, or with
`with_great_circles()` great circles between nodes in longitude and
latitude, as ICON's and MPAS's are. Great circles need `densify` above 1
or a `refine_tolerance` to bend (at densify 1 an edge is its chord in the
grid's pixels). An edge across the equator is cut there first, and a node
on a pole joins its edges along their meridians, whatever longitude it is
written with. Each face's area then converges on its spherical excess and
the faces tile the grid at every tolerance.

Against ESMF, Cartesian (hexagons and triangles over part of a grid,
`tests/esmf_mesh.rs`), the weights agree to 6e-14 both ways. On the
sphere (a geodesic mesh and its dual with a global 5° grid) the faces'
areas agree to 8e-8, and the weights differ by the grid's cells, which
rwarp bounds by parallels and ESMF by great circles (`S` to 3.8e-3).

The grid can also be one whose cells are in its world coordinates, as for
`exact_weights`: geolocation arrays (`GenImgGeolocTransformer`), a
curvilinear grid by its corners (`MeshTransformer` over a `QuadMesh`) or a
whole ocean grid in pieces (`SetTransformer` over a `MeshSet`), each with
destination geotransform `WORLD_GT`, so that its destination coordinates
are the mesh's. Each face is then measured in the source's world and
walked onto its cells, `frac` is a fraction of a cell's world area
(`src_cell_area`), and the measure must be planar. For a set, each face
goes to every band's world and is left out of one where it comes out
inside out, such as round the far pole of a polar projection.

## Densify

Both settings conserve: conservation comes from the shared edges, not from
how straight they are. They define slightly different cells. With
`densify` 1 a target cell is the quadrilateral through its four transformed
corners in source pixel space; with more, its edges follow the curve the
target's straight edges map to. Under an affine transform the two agree to
rounding (4e-14). For lon/lat to LAEA at 20 km, `frac` differs by up to
7e-5 and `dst_area` by up to 6e-7 relative between densify 1 and 4. Cost
grows with `densify` (the transform dominates): 4 is about 4 times 1 when
target cells are smaller than source cells.

`WeightOptions::with_refine_tolerance(t)` (and `Footprint::Adaptive {
densify, tolerance }` in `warp()`) refines instead of guessing a count.
Each segment is bisected while its transformed midpoint strays more than
`t` from the chord, d3-geo's adaptive resampling through bigcurve, one
transformer call per level. `t` is in source pixels, or in world units
for a geolocation or mesh source, where it is one length for every cell.
The error in `frac` follows `t` about one to one, and the points grow as
`t` to the power -1/2. That is the rate a fixed `densify` has too, so the
gain is in where the points go: only to the edges that bend. On a test
map that bends at one end (`tests/adaptive.rs`), tolerance 1e-4 matches
densify 64's error with under half its points. Where the transform is
affine nothing is added, and the weights are `densify`'s bit for bit.
Tiles and threads still give the whole grid's weights, since a segment's
points depend only on its ends.

## Same numbers as `warp()`

`warp()` with `Footprint::Exact` and `ResampleAlg::Average` uses the same
lattice and the same per-pixel coverage step. For compatibility with
rwarp 0.2 it snaps coverage within 1e-6 of 1 to 1, drops coverage at or
below 1e-6 and rounds the rest to `f32`. `exact_weights` keeps every
positive overlap, unrounded, by default. With
`WeightOptions::warp_compatible(densify)` it does what `warp()` does, and
`apply(FracArea)` then gives `warp()`'s output bit for bit (tested on the
exact-footprint kernel fixtures and on lon/lat to LAEA with two bands,
nodata, NaN, every `BandNodata` rule, a mask band and Int16 to Byte).

`warp()` reduces directly and never stores weights. Use it for a one-off.
Use weights when the same grids are applied many times, when you need
`DestArea`, or when the weights are the product.

## Tiling and streaming

`WeightOptions::with_dst_offset(col, row)` computes the weights of a target
window, with full-grid indices. A window's lattice is the whole grid's, so
tiles give exactly the whole grid's entries: row bands concatenate, and
column offsets agree too when `densify` is a power of two.

`exact_weights_with(.., |cell| ..)` calls back once per target cell with
its entries, storing nothing, for problems too big to hold. Entries cost
24 bytes each (four `u32` and an `f64`). Over a 0.25° global source:

| Target | Entries per target cell | `warp()` exact | `exact_weights` | `apply` |
|---|---|---|---|---|
| LAEA 400 x 400 at 10 km | 2.2 (354 k, 9.8 MB) | 81 ms | 90 ms | 12 ms |
| LAEA 160 x 160 at 50 km | 45 (1.1 M, 28 MB) | 42 ms | 61 ms | 26 ms |
| EPSG:3031 200 x 240 at 12.5 km | 4.6 (220 k, 5.7 MB) | 39 ms | 41 ms | 6 ms |

(densify 1, release build, one thread, PROJ backend.)

`exact_weights_par(make, &src, dst_ncol, dst_nrow, &opts, threads)` gives
`exact_weights`' table entry for entry and bit for bit, on several
threads. As for `warp_par` ([Resampling](resampling.md#threads)), `make`
builds a transformer for each thread. Each thread takes bands of the
target's rows into a table of its own, and the tables are joined in row
order. A 0.1° global source to 1000 by 1000 pixels of south polar
stereographic at densify 2 (6.7 M entries) takes 3.0 s on one thread,
0.58 s on 8 and 0.29 s on 32. Through a `SetTransformer`, the same gives
a whole ocean grid's weights to a grid on threads: ORCA2 to 1° at
densify 8, 2.3 s on one thread and 0.35 s on 8.

## Weight files

rwarp has no I/O. Writing ESMF/SCRIP netCDF, xESMF's format or Parquet is
wxylib's job, and the columns map directly. With the full grids' widths:

| ESMF / SCRIP | rwarp |
|---|---|
| `row` (1-based, target) | `dst_row * dst_ncol_full + dst_col + 1` |
| `col` (1-based, source) | `src_row * src_ncol_full + src_col + 1` |
| `S` | `normalised(norm)` |
| `area_b` | `dst_area` (in source cells, not steradians) |
| `frac_b` | `dst_covered() / dst_area` |

`Weights::esmf_row_col(dst_ncol_full, src_ncol_full)` returns the two index
columns.

## Limits

- **The measure is the source side's**: planar in source pixel space, on
  the sphere for a regular longitude/latitude source (`lonlat_measure`),
  or planar in a geolocation or mesh source's world coordinates. A
  destination in longitude and latitude with a projected source has its
  `dst_area` planar in source pixel space, not on the sphere: a
  destination-side measure waits for a use (decision C4). ESMF measures
  on the sphere with great-circle edges on both sides.
- **The antimeridian needs the transformer to say the source wraps.** A
  target cell whose outline crosses a geographic source's seam is unwrapped
  and measured on both sides of it, and one around a pole is closed along
  the pole's row, when the transformer reports the source's longitude
  period (`Transformer::source_wrap`). `GenImgProjTransformer` does for a
  north-up source of at most 360 degrees once it has the source size
  (`with_source_size`, which also gives GDAL's `CENTER_LONG`); a
  transformer that does not report it gets no unwrapping, and such cells
  are wrong as before. A geolocation source with `with_lon_wrap(true)` is
  handled the same way in its world coordinates, a turn being 360
  degrees.
- **Geolocation sources need whole cells.** `GeolocArrays::world_cells()`
  says whether a source is measured in world coordinates, and if not, why.
  It isn't for arrays with a missing sample, arrays that fold (cells of
  both orientations, as where a scanning swath's scans overlap), or, with
  `lon_wrap`, a swath over a pole (its longitudes don't unwrap
  consistently) or one spanning more than 360 degrees. Arrays that cross
  the antimeridian are fine. Those it isn't for are measured in source
  pixel space as before, which on a curved grid is off by about 1% of a
  cell along each outline. The choice is made once per
  set of arrays, so every chunk of a warp uses the same measure.
- **Destination cells are what a transformer describes.** A curvilinear
  target works through `GenImgGeolocTransformer` with its bilinear cell
  definition, or as a mesh through `MeshTarget`, whose cells are the
  mesh's. Onto a mesh, the overlaps are measured in the source's space,
  so a cell's sides are only as straight there as `densify` makes them:
  near a pole, in longitude and latitude, use more.
