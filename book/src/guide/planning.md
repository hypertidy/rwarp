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

`compute_source_window` is a quick estimate that pads one pixel beyond the
kernel radius and skips GDAL's forward probe and special points. For the
window exactly as gdalwarp computes it, use
`ChunkPlanner::new(&t, src_size, dst_size, opts).source_window(dst_off, dst_size)`.

## Splitting into chunks

```rust,ignore
use rwarp::plan::{plan_chunks, ChunkOptions};

let opts = ChunkOptions::gdalwarp(ResampleAlg::Bilinear)
    .with_memory_limit(64.0 * 1024.0 * 1024.0)            // gdalwarp -wm
    .with_pixel_cost(DataType::Float32, 1, true, true);   // with nodata
let plan = plan_chunks(&approx, src_size, dst_size, &opts);
for c in plan.live() {
    // c.dst_off, c.dst_size: a piece of the target
    // c.src_off, c.src_size: what it reads from the source
    // c.fill_ratio, c.scale: GDAL's fill ratio and kernel scale
    let tile = warp(&approx, &read(c.src_off, c.src_size), c.dst_size[0], c.dst_size[1],
                    &c.warp_options(&WarpOptions::new(ResampleAlg::Bilinear)));
}
```

This is gdalwarp's planning (`CollectChunkList`): the source window of
the whole target is computed, and while it would take more working memory
than the limit, or its fill ratio is below 0.5 (much of it outside the
source), the target is halved along its longer side and each half planned
again. Each chunk is then warped on its own, with the kernel scale GDAL
would estimate for it. Chunks are sorted top to bottom, then left to
right, as GDAL warps them.

The chunk boundaries matter to the output, not only to memory: they set
the kernel scale, the edges where kernels are clipped, and the scanlines
the approximate transformer interpolates over. With
`ChunkOptions::gdalwarp` and the same memory limit, the plan and the
output equal gdalwarp's for a whole raster (see
[GDAL parity](../gdal-parity.md)). `ChunkOptions::new` differs in two
documented places: each chunk's kernel scale is estimated over the chunk
(GDAL estimates every chunk at the output's top-left), and source
windows are padded for a fixed kernel scale.

A few inputs need the caller: GDAL's special points (longitude 0 at
latitude +-89.9999, in target pixels, which make every chunk sample a
grid when one falls in the target) need the CRSs, and the target's block
size matters only for a tiled target (`OPTIMIZE_SIZE`).

[`warp_chunks`](../api/rwarp/plan/fn.warp_chunks.html) runs a whole plan
from a source buffer in memory, which is how the parity tests use it.

### At the antimeridian

`with_split_antimeridian(true)` adds the split of 0.2's
`collect_chunk_list`: a chunk whose source window spans the whole source
width (as at 180 degrees) is split where source x jumps, so a tile
straddling 180 degrees becomes tiles each side with compact source
windows rather than one spanning the globe (a 70 times reduction in
pixels read for a GEBCO tile over Fiji). GDAL reaches a similar result
only through its memory limit.

## Why planning is separate

Because the plan is just data (target windows and source windows), it can
be inspected, costed, cached, serialised (the `serde` feature) and
resumed. You can count the
COG tiles or Zarr chunks a reprojection touches, pick an overview level
from the scale before reading, or batch requests across many targets, all
before any I/O.
