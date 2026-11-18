# Resampling

`warp` resamples a source window into a target, for any sample type
(`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64`) and any number of
pixel-interleaved bands, in one pass:

```rust,ignore
let out = warp(&transformer, &SourceBuffer { .. }, dst_ncol, dst_nrow, &opts);
```

`SourceBuffer` is the window you read, laid out `(row, col, band)`, with
its offset in the full source image. `WarpOptions` holds everything else;
`WarpOptions::default()` is nearest neighbour with GDAL's defaults. Build
it with `WarpOptions::new(alg)` and the `with_` methods, one per option:

```rust,ignore
let opts = WarpOptions::new(ResampleAlg::Average)
    .with_footprint(Footprint::Exact { densify: 1 })
    .with_src_nodata(Nodata::Value(-9999.0));
```

`WarpOptions` and the option enums are `#[non_exhaustive]`, so new options
can arrive in minor releases. Outside rwarp this means no struct literals
for `WarpOptions`, `GeolocOptions` or `AverageThresholds`, and a `_` arm
when matching on the enums.

| Option | Choices | GDAL equivalent |
|---|---|---|
| `alg` | `NearestNeighbour`, `Bilinear`, `Cubic`, `Lanczos`, `Average`, `Mode` | `-r` |
| `scaling` | `WidenWhenDownsampling` (default), `Fixed { x, y }`, `Point` | `-wo XSCALE=`, `-wo YSCALE=` |
| `footprint` | `GdalBox` (default), `Exact { densify }` | (average and mode only) |
| `mode_ties` | `First` (default), `Min`, `Max` | `-wo MODE_TIES=` |
| `average` | thresholds and excluded values | `NODATA_VALUES_PCT_THRESHOLD`, `EXCLUDED_VALUES`, `EXCLUDED_VALUES_PCT_THRESHOLD` |
| `band_nodata` | `Partial` (default), `Unified`, `PerBand`, `AnyBand` | `-wo UNIFIED_SRC_NODATA=` |
| `nodata`, `mask_band` | a nodata value; an alpha band | `-srcnodata`, `-dstnodata`, alpha |
| `dst_col_off`, `dst_row_off` | offset of this tile in a larger target | the chunk's destination offset |
| `cutline` | a `Cutline` (see [Cutlines](cutline.md)) | `-cutline`, `-cblend`, `-wo CUTLINE_ALL_TOUCHED=` |

## Threads

`warp_par` and `warp_to_par` give `warp`'s and `warp_to`'s values bit for
bit, on up to `threads` threads (0 for all available). They take a
closure that builds the transformer instead of the transformer:

```rust,ignore
let make = || GenImgProjTransformer::new(src_crs, src_gt, dst_crs, dst_gt)
    .map(|t| ApproxTransformer::new(t, 0.125));
let out = warp_par(make, &src, dst_ncol, dst_nrow, &opts, 8)?;
```

As with GDAL's `-wo NUM_THREADS`, each thread warps bands of the
target's rows through a transformer of its own, since a libproj
transformer can't be shared between threads. A `Sync` one can serve them
all as `|| Ok(&t)`. What a warp works out once is still worked out once:
the cutline's density over the source, and the kernel scale over the
whole target. So every row comes from the same steps as on one thread.
With one thread nothing is spawned. A whole plan of chunks runs on
threads with `warp_chunks_par` ([Planning reads](planning.md)).

A 0.1° global grid to 2000 by 2000 pixels of south polar stereographic,
through libproj and the approximate transformer, on a 32-core machine:

| | 1 thread | 8 | 32 |
|---|---|---|---|
| Bilinear | 0.48 s | 0.08 s | 0.04 s |
| Cubic | 0.89 s | 0.15 s | 0.06 s |
| Average | 0.86 s | 0.14 s | 0.05 s |
| Average, `Exact { densify: 2 }` | 3.44 s | 0.45 s | 0.15 s |

## Kernel widening when downsampling

When the output is coarser than the source, GDAL widens the bilinear, cubic
and lanczos kernels to cover each output pixel's source area. rwarp does
the same by default:

- **`WidenWhenDownsampling`** estimates the target/source scale over the
  target window, as GDAL does per warp chunk, then widens the kernel support
  to `ceil(radius / scale)` on any axis where the scale is below 1.
  Upsampling never narrows the kernel.
- **`Fixed { x, y }`** gives the scale, like GDAL's `XSCALE` and `YSCALE`.
  Use it to make results independent of how the target is tiled.
- **`Point`** keeps the fixed 2 x 2, 4 x 4 and 6 x 6 support at every
  ratio, which is `gdalwarp -wo XSCALE=1 -wo YSCALE=1`.

As in GDAL, bilinear and cubic widen only once the scale reaches 0.5 or
less; lanczos widens at any scale below 1.

One deliberate difference: rwarp estimates the scale over the window
actually being warped. GDAL takes its estimate at the top-left of the whole
output for every chunk, so a chunked gdalwarp can widen by the wrong amount
away from the origin (see [GDAL parity](../gdal-parity.md)).

## Average and mode

Both weigh the source pixels a target pixel's footprint covers, measuring
area in source pixel space:

- **`Footprint::GdalBox`** is GDAL's footprint: the axis-aligned box between
  the transformed top-left and bottom-right corners of the target pixel,
  with fractional weights at its edges. It matches gdalwarp, quirks
  included; under rotation or strong curvature the box can be far from the
  true footprint.
- **`Footprint::Exact { densify }`** transforms the target pixel's outline
  and intersects it exactly with the source pixels. With renormalisation
  over valid pixels, `Average` is then first-order conservative remapping.
  It costs about 25 times the box. The same overlaps are available as a
  weight table; see [Conservative weights](weights.md).

`Mode` counts covered area per value and breaks ties with `ModeTies`.
Counts are re-compared every time a value's count grows, in scan order, so
`First` is the tied value that reached the winning count first. GDAL does
this for 8- and 16-bit data; rwarp does it for every type.

## Output values

- **Integer outputs** are clamped to the type and rounded half up,
  `floor(v + 0.5)`, as GDAL does with an integer working type. Nearest and
  mode copy source values and never round.
- **Nodata avoidance:** a resampled value equal to the nodata value is moved
  off it by one step, so a valid pixel never reads back as missing.
- **Source bounds:** sample points outside `[0, n)` of the source are off
  the source for every method except average and mode, which use
  footprints.
