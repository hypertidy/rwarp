# The antimeridian: two edge windows, fetched at once

Status: 2026-11-08, hypertidy/rwarp#57 (crate 0.3.3). Where rwarp reads less
than GDAL, and what it takes to turn that into time.

## The case

A target centred on 180° over a global longitude/latitude source: here GEBCO
2025 at overview 3 (5400 × 2700) to a 384 × 384 Lambert azimuthal target,
`+proj=laea +lat_0=-17 +lon_0=180`, ±1500 km. Its pixels map to source columns
5168-5398 and 1.6-232: the two edges of the source.

GDAL's `ComputeSourceWindow()` takes the bounding box of the transformed
sample points. Here those points sit at both ends of the source, so the box
passes the 90% rule and becomes the whole width: one chunk reading
`Src=0,1393,5400x418`, 2,257,200 pixels, 7.6 MB over HTTP. rwarp's planner
follows GDAL, so it did the same.

## The split

`ChunkOptions::split_antimeridian` (and `warp_plan(split_antimeridian = TRUE)`
in R, the default there):

1. A chunk whose window takes the whole width is split at the target columns
   where source x jumps by more than half the source width
   (`find_discontinuity_range`). Each side is planned as usual and gets a
   compact window.
2. The columns at the jump, the gap, map near both edges. Each pixel needs
   only one, though, so the gap becomes **two chunks on the same target
   window**, one source window at each edge (`gap_windows`: sampled every half
   pixel, sorted by source half, padded for the gap's own kernel scale plus a
   pixel). The kernel already skips a target pixel whose source is outside the
   chunk's window, and `warp_chunks` (and the R binding) place only the pixels
   a chunk wrote (`plan::merge_tile`), so the two passes combine. For chunks
   that don't overlap, that's the plain copy it always was.
3. The split is kept only if it reads under half of the full-width window.
   Around a pole every longitude is needed and splitting saves nothing, so an
   EPSG:3031 target keeps GDAL's plan (and gdalwarp parity).

Result for the case above: 2,257,200 → **200,600** source pixels (11.3×),
7.4 → 1.2 MB. With an exact transformer the values are identical to the unsplit
warp (near, bilinear, cubic; Int16, Float32, Byte). With the default
`max_error = 0.125` they differ within that tolerance, as any change of
chunking does, because the approximation's nodes move. This matches gdalwarp
to 0.125 px, not to the bit.

## Fewer bytes was not faster, at first

The split makes four chunks, and `warp_exec()` calls `read_fn` once per chunk,
in turn. Each request to source.coop costs 0.4-0.75 s before the first byte,
at about 5 MB/s. Four queued requests lost to GDAL's one big one.

rwarp does no I/O, so the fix belongs to the reader. `plan_windows(plan)` lists
every chunk's source window in exec order. The reader fetches them all at
once, then answers `read_fn` from what it fetched. warp-sandbox does this with
one GDAL dataset handle per window on threads, sharing GDAL's HTTP cache.
Within one `RasterIO()` GDAL already fetches a window's tiles in parallel; this
overlaps the windows as well.

## Numbers

Live timings swing more than the effect, so these come from a local server
with a fixed 0.4 s per request and 5 MB/s per connection (warp-sandbox
`bench/latency-local.R`):

| 384² bilinear | rwarp, split + prefetch | rwarp, split | GDAL warp API |
|---|---|---|---|
| antimeridian | **1.35 s** | 1.79 s | 3.06 s |
| pole inside (EPSG:3031) | 3.37 s | 3.37 s | 3.18 s |
| small oblique | 1.33 s | 1.32 s | 1.33 s |

It takes both: the split saves bytes, prefetch saves round trips. Where the
reads are the same, so are the times. With the data in memory, rwarp's kernels
are still 1.5-3× slower than GDAL's (warp-sandbox `bench/README.md` section 3).

## Upstream

Drafts for GDAL, not filed: warp-sandbox `notes/` (mdsumner/warp-sandbox#1).
They cover the full-width strip, and a request that GTiff's `AdviseRead()`
(a no-op today) prefetch several windows, so callers wouldn't need threads.

Evidence: warp-sandbox `bench/README.md` sections 4 and 5, and the scripts
beside it. Tests: `tests/chunks.rs`
(`split_antimeridian_reads_compact_windows`) and `rwarp-r/tests/testthat/test-warp-plan.R`.
