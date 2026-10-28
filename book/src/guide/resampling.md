# Resampling

`warp` resamples a source window into a target, for any sample type
(`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64`) and any number of
pixel-interleaved bands, in one pass:

```rust,ignore
let out = warp(&transformer, &SourceBuffer { .. }, dst_ncol, dst_nrow, &WarpOptions { .. });
```

`SourceBuffer` is the window you read, laid out `(row, col, band)`, with
its offset in the full source image. `WarpOptions` holds everything else;
`WarpOptions::default()` is nearest neighbour with GDAL's defaults.

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
  It costs about 25 times the box.

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
