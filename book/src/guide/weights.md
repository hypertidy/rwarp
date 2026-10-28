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
| `dst_area` | target cell | area of the target cell's outline, in source cells; `NaN` if it failed to transform |
| `dst_off`, `dst_size` | | the target window the weights are for |

Areas are measured in source pixel space, where a source cell has area 1.
For a source on a regular grid in its own CRS that is area in the source
CRS, up to a constant. Target cells have edges straight in target pixel
space, each split into `densify` segments whose end points go through the
transformer exactly.

`frac` is relative to the source cell, not the target, so it doesn't depend
on how the target is normalised. `dst_area` gives either normalisation:

- **`Normalization::FracArea`**: `S = frac / (sum of frac over the target
  cell)`, the area-weighted mean over the part of the cell that has
  source. This is what `warp()` computes. In ESMF it's `fracarea`; in
  xESMF, `conserve_normed`.
- **`Normalization::DestArea`**: `S = frac / dst_area`. A cell half off the
  source gets half the weight, so `sum(value * dst_area)` over the target
  equals the integral of the source over the region the target covers.
  ESMF `destarea`, xESMF `conserve`.

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

- **Planar areas** in source pixel space. ESMF measures on the sphere with
  great-circle edges. A spherical measure is planned (design/conservative.md,
  stage 3).
- **No antimeridian handling.** A target cell whose outline crosses the
  source's wrap column folds across the whole source width, and its weights
  are wrong. Keep target grids clear of the source's seam for now, or use a
  source that is continuous across the target.
- **Destination cells are what a transformer describes.** Curvilinear grids
  work through `GenImgGeolocTransformer` with its bilinear cell definition.
  Cells from corner bounds and meshes are later stages.
