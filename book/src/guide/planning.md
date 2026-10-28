# Planning reads

Planning answers "which source pixels does this target need?" before any
pixels are read. It needs only a transformer and the source raster's size.

## One window

```rust,ignore
let win = compute_source_window(&t, dst_off, dst_size, src_raster_size, resample_padding)?;
// win.xoff, win.yoff, win.xsize, win.ysize: the source window in pixels
// win.fill_ratio: how much of the unclamped window lies inside the source
```

This follows GDAL's `ComputeSourceWindow`: it samples the target window's
edges (and, if that fails, a grid over its interior), transforms them to the
source, and takes their bounding box, padded for the resampling kernel
(0 for nearest, 1 for bilinear, 2 for cubic, 3 for lanczos). It returns
`None` if too few points transform.

A low `fill_ratio` means much of the window falls outside the source, or
the target straddles a discontinuity such as the antimeridian.

## Splitting into chunks

```rust,ignore
let chunks = collect_chunk_list(&t, src_raster_size, dst_off, dst_size,
                                resample_padding, 0.5, 8, None);
for c in &chunks {
    // c.dst_off, c.dst_size: a piece of the target
    // c.src_window: what it needs from the source
}
```

This follows GDAL's `CollectChunkListInternal`: when a window's fill ratio
is below the threshold (GDAL uses 0.5), the target is split along its
longest side and each half is planned again. A tile straddling 180 degrees
then becomes two tiles, one each side, each with a compact source window
rather than one window spanning the whole globe (a 70 times reduction in
pixels read for a GEBCO tile over Fiji).

## Why planning is separate

Because the plan is just data (target windows and source windows), it can
be inspected, costed, cached, serialised and resumed. You can count the
COG tiles or Zarr chunks a reprojection touches, pick an overview level
from the scale before reading, or batch requests across many targets, all
before any I/O.
