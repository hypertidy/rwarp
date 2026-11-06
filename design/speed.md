# Speed against GDAL

Status: 2026-11-06. Measurements, a code reading, and a model. Parked: no
optimisation work started. Measured through the R package (hypertidy/rwarp#56)
in mdsumner/warp-sandbox (`bench/`), which compares rwarp with GDAL on the same
request at the same overview level.

## Measured

Output is identical to GDAL's in every case below (value for value).

**Kernels, data in memory** (no network, no decompression): IBCSO 2025 bed at
overview 2 (2400², Int16) to EPSG:3031, single-threaded on both sides;
seconds, rwarp / GDAL (`GDALWarpOperation` called directly):

| kernel | 512² | 1024² | 2048² |
|---|---|---|---|
| near | 0.032 / 0.016 (2.0x) | 0.074 / 0.026 (2.8x) | 0.205 / 0.077 (2.7x) |
| bilinear | 0.115 / 0.054 (2.1x) | 0.128 / 0.072 (1.8x) | 0.442 / 0.262 (1.7x) |
| cubic | 0.293 / 0.101 (2.9x) | 0.242 / 0.111 (2.2x) | 0.896 / 0.405 (2.2x) |
| **lanczos** | 0.724 / 0.133 (**5.4x**) | 1.208 / 0.246 (**4.9x**) | 2.524 / 0.708 (3.6x) |
| average | 0.094 / 0.048 (2.0x) | 0.182 / 0.118 (1.5x) | 0.509 / 0.348 (1.5x) |

rwarp's times include its plan (about 11 ms), the interleave from R and building
the R array; small next to the kernel for the interpolating methods.

**Fetching:** on a cold COG (GEBCO 2025, `/vsicurl/` on source.coop), rwarp's
plan and gdalwarp read the same source window (`0,244 675x93`, GDAL's kernel
log `Src=0,244,675x93`) with the same HTTP range requests. Cold timings over
the network are dominated by round trips and vary more between runs than
between engines.

## Code reading: why lanczos stands out

The generic interpolating path (`widened`, `src/warp.rs` ~L2344) is already
separable: per destination pixel it evaluates the x weights once into
`sc.wx` and one y weight per row. (The legacy `i32` path, `lanczos_sample`
~L1140, evaluates the x weight per tap; `warp_exec()` in the R package does not
use it.)

GDAL's `GWKResampleOptimizedLanczos` (`alg/gdalwarpkernel.cpp` ~L3918 at
`739fcf03bf`) also:

1. keeps the last fractional offsets (`dfLastDeltaX/Y`) and reuses the weight
   arrays when they repeat;
2. avoids `sin()` per weight: it computes `cos(pi/3 * scale)` once and steps
   through the weights with the triple-angle identity;
3. has narrower per-tap work than rwarp's `accumulate()` (a band loop,
   validity checks and shared-weight bookkeeping per tap).

Hypothesis, not yet profiled: per-tap overhead (3) is most of the general
1.5-3x gap, and lanczos shows it most because its widened window is the widest
(about 13 x 13 taps at scale 0.45, against about 9 x 9 for cubic); (2) adds to
it. Profile first: a Rust bench on this case (`cargo bench`, or `perf` on the
fixture), lanczos and cubic.

Exactness to keep in mind: the recurrence in (2) gives slightly different
floating-point weights from direct `sin()`; check against the gdalwarp
fixtures whether exact matches survive.

## Model: does a slower kernel matter over the network?

Brute force: time = round trips x RTT + bytes / bandwidth + decode + kernel.

From ecostats (Hobart) to source.coop a cold COG read takes about four
dependent HTTP exchanges in GDAL (file size, header, tile index, data), each
about 0.4-0.6 s, so about 2 s; the bytes for a 384² view are negligible. The
kernel penalty above is 0.02-1 s.

With the byte ranges known in advance (the reference Parquet files source.coop
publishes beside the COGs, `virtual/.../refs_*.parquet`) and the file size from
a catalogue, a reader can skip the dependent exchanges and issue one concurrent
round of data requests: about 1 round trip. So, cold and over the network, a
reader that saves 2-3 round trips (1-1.5 s) wins even with today's kernels;
large lanczos is about even. Once the data is local or cached, the kernels
decide, and GDAL is 1.5-5x faster. Most of that win comes from known byte
ranges rather than from async itself, and GDAL can close part of it too
(skipping the size request, or reading references).
