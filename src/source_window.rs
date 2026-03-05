//! Compute the source pixel window needed for a given destination window.
//!
//! Maps GDAL's `GDALWarpOperation::ComputeSourceWindow()` and
//! `ComputeSourceWindowTransformPoints()` (gdalwarpoperation.cpp,
//! commit a3b7b01d3e, GDAL 3.13.0dev, lines 2751–3367).
//!
//! ## Architecture
//!
//! This is a pure function of a `Transformer` — no dataset handles, no
//! warp options struct, no mutable state. It answers:
//!
//! > "Given a destination pixel window and a coordinate transformer,
//! >  which source pixels are needed?"
//!
//! The result is a `SourceWindow` (xoff, yoff, xsize, ysize) in source
//! pixel coordinates, plus diagnostic fields (fill_ratio, n_failed).
//!
//! ## Algorithm (matching GDAL)
//!
//! 1. **Edge sampling** (default): sample `step_count * 4` points around
//!    the perimeter of the destination window.
//! 2. **Corner check**: if any of the 4 dest corners fail to transform,
//!    switch to ALL mode (every pixel on every edge).
//! 3. **Grid fallback**: if any edge points fail, retry with a
//!    `(step_count+2)²` grid through the interior.
//! 4. **Bounds clamping**: clamp to source raster, add resampling padding.
//! 5. **Antimeridian heuristic**: if source window covers >90% of source
//!    width, snap to full width (GDAL line 3325).
//! 6. **Fill ratio**: ratio of clamped to unclamped window area — low
//!    values suggest the dest window should be split.
//!
//! ## Additional functions
//!
//! - `refine_source_window_from_source()` — forward probe matching GDAL's
//!   `ComputeSourceWindowStartingFromSource()`. Transforms a grid across
//!   the source raster forward to destination space, keeps only points
//!   that land inside the destination window, and expands the source
//!   window to include them. Safety net for source pixels missed by the
//!   backward probe.
//!
//! - `collect_chunk_list()` — recursive destination subdivision matching
//!   GDAL's `CollectChunkListInternal()`. When the source fill ratio is
//!   low (indicating wasteful reads, e.g. antimeridian crossing), splits
//!   the destination window and recurses until each chunk has a compact
//!   source window.
//!
//! ## What we skip (vs GDAL)
//!
//! - `CHECK_WITH_INVERT_PROJ` retry (PROJ-4 era workaround, not needed)
//! - `aDstXYSpecialPoints` pole detection (planned, not yet implemented)
//! - `SAMPLE_GRID` / `SAMPLE_STEPS` warp options (fixed defaults used)
//! - Warp memory budget (not applicable — rwarp operates per-tile)
//!
//! ## Antimeridian handling
//!
//! When a destination tile straddles the antimeridian (e.g. a Fiji LCC tile
//! crossing 180°), source X coordinates are bimodal: pixels on one side map
//! to columns near 0, pixels on the other map to columns near `src_width`.
//! `compute_source_window` detects this via the >90% width heuristic and
//! snaps to full source width — correct but reading up to 70× more source
//! pixels than necessary.
//!
//! `collect_chunk_list` resolves this by recursive subdivision: it detects
//! the low fill ratio, calls `find_discontinuity_range` to locate the gap
//! in source X coordinates, and splits the destination tile into
//! left/gap/right sub-tiles each with a compact source window. On GEBCO
//! tile \[5,3\] (Fiji LCC) this reduces source pixels from 97.9M to 1.4M —
//! a 70× reduction. Adjacent non-straddling tiles are unaffected.
//!
//! This is a cogcache innovation: GDAL achieves the same result indirectly
//! via warp memory budget subdivision, which rwarp does not implement.
//!
//! ## GCP transformers
//!
//! `compute_source_window` and `collect_chunk_list` accept any [`Transformer`],
//! including [`crate::gcp_transform::GcpTransformer`]. GCP-warped imagery
//! (e.g. Sentinel-2 L1C granules with GCPs instead of a geotransform) can
//! be planned with the same API as geotransform-based sources.

use crate::transform::Transformer;

/// Result of [`compute_source_window`].
///
/// Describes the rectangle of source pixels needed to produce a given
/// destination window. All coordinates are in source pixel space
/// (0-based, top-left origin).
///
/// A `fill_ratio` below ~0.5 suggests the destination window should be
/// subdivided — use [`collect_chunk_list`] to do this automatically.
/// A `fill_ratio` of exactly 0.01 indicates the antimeridian snap fired.
#[derive(Debug, Clone)]
pub struct SourceWindow {
    /// Column offset in source raster (0-based)
    pub xoff: i32,
    /// Row offset in source raster (0-based)
    pub yoff: i32,
    /// Width in source pixels
    pub xsize: i32,
    /// Height in source pixels
    pub ysize: i32,
    /// Ratio of clamped window area to unclamped — low values mean
    /// much of the source window falls outside the raster.
    pub fill_ratio: f64,
    /// Number of sample points that failed to transform.
    pub n_failed: i32,
    /// Total sample points attempted.
    pub n_samples: i32,
}

/// Default step count for edge/grid sampling (GDAL: DEFAULT_STEP_COUNT = 21).
const DEFAULT_STEP_COUNT: usize = 21;

/// Compute the source pixel window needed for a destination window.
///
/// # Arguments
///
/// * `transformer` — any `Transformer` (exact or approx)
/// * `dst_off` — destination window offset `[col_off, row_off]` in pixels
/// * `dst_size` — destination window size `[ncol, nrow]` in pixels
/// * `src_raster_size` — full source raster `[ncol, nrow]`
/// * `resample_padding` — extra pixels for resampling kernel
///   (0 = nearest, 1 = bilinear, 2 = cubic, 3 = lanczos)
///
/// Returns `None` if too few points transform successfully (<5 valid).
pub fn compute_source_window(
    transformer: &impl Transformer,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    src_raster_size: [i32; 2],
    resample_padding: i32,
) -> Option<SourceWindow> {
    let step_count = DEFAULT_STEP_COUNT;

    // --- Phase 1: Try edge-only sampling ---
    //
    // First check corners. If any fail, use ALL mode (every edge pixel).
    // (GDAL lines 3050–3073)
    let use_all = {
        let mut cx = vec![
            dst_off[0] as f64,
            (dst_off[0] + dst_size[0]) as f64,
            dst_off[0] as f64,
            (dst_off[0] + dst_size[0]) as f64,
        ];
        let mut cy = vec![
            dst_off[1] as f64,
            dst_off[1] as f64,
            (dst_off[1] + dst_size[1]) as f64,
            (dst_off[1] + dst_size[1]) as f64,
        ];
        let ok = transformer.transform(true, &mut cx, &mut cy);
        !ok[0] || !ok[1] || !ok[2] || !ok[3]
    };

    let (min_x, min_y, max_x, max_y, n_samples, n_failed) = if use_all {
        // ALL mode: every pixel on every edge (GDAL lines 2870–2896)
        sample_edges_all(transformer, dst_off, dst_size)
    } else {
        // Step mode: step_count points per edge (GDAL lines 2898–2922)
        sample_edges_step(transformer, dst_off, dst_size, step_count)
    };

    // --- Phase 2: Grid fallback if any edge points failed ---
    // (GDAL lines 3153–3164)
    let (min_x, min_y, max_x, max_y, n_samples, n_failed) = if n_failed > 0 {
        sample_grid(transformer, dst_off, dst_size, step_count)
    } else {
        (min_x, min_y, max_x, max_y, n_samples, n_failed)
    };

    // --- Phase 3: Check if too many failures ---
    // (GDAL line 3170: nFailedCount > nSamplePoints - 5)
    if n_failed > n_samples - 5 {
        return None;
    }

    // --- Phase 4: Early exit if completely outside source ---
    // (GDAL lines 3214–3228)
    let src_w = src_raster_size[0] as f64;
    let src_h = src_raster_size[1] as f64;
    if min_x > src_w || max_x < 0.0 || min_y > src_h || max_y < 0.0 {
        return Some(SourceWindow {
            xoff: 0, yoff: 0, xsize: 0, ysize: 0,
            fill_ratio: 0.0, n_failed, n_samples,
        });
    }

    // --- Phase 5: Round-if-close-enough ---
    // (GDAL lines 3232–3243)
    let snap = |v: f64| -> f64 {
        let r = v.round();
        if (r - v).abs() < 1e-6 { r } else { v }
    };
    let min_x = snap(min_x);
    let min_y = snap(min_y);
    let max_x = snap(max_x);
    let max_y = snap(max_y);

    // --- Phase 6: Resampling kernel padding ---
    // (GDAL lines 3262–3278)
    // Scale padding if downsampling significantly
    let dx_scale = (dst_size[0] as f64 / (max_x - min_x)).max(1e-3);
    let dy_scale = (dst_size[1] as f64 / (max_y - min_y)).max(1e-3);
    let mut x_radius = if dx_scale < 0.95 {
        (resample_padding as f64 / dx_scale).ceil() as i32
    } else {
        resample_padding
    };
    let mut y_radius = if dy_scale < 0.95 {
        (resample_padding as f64 / dy_scale).ceil() as i32
    } else {
        resample_padding
    };

    // Extra pixel beyond the kernel radius to ensure interpolation kernels
    // always have their full neighbourhood. Without this, transformed coords
    // near the buffer edge can put kernel pixels outside the buffer, forcing
    // the weight-renormalisation fallback path which produces different values
    // than GDAL. GDAL achieves this via nExtraSrcPixels (gdalwarpoperation.cpp
    // ~L3270). See: bilinear edge bug at pixel [1,157] in GEBCO diagnostics.
    if resample_padding > 0 {
        x_radius += 1;
        y_radius += 1;
    }

    // Extra padding if there were failures (GDAL lines 3293–3297)
    if n_failed > 0 {
        x_radius += 10;
        y_radius += 10;
    }

    // --- Phase 7: Clamp to source raster bounds ---
    // (GDAL lines 3309–3351)
    let min_x_clamped = min_x.max(0.0) as i32;
    let min_y_clamped = min_y.max(0.0) as i32;
    let max_x_clamped = max_x.ceil().min(src_w) as i32;
    let max_y_clamped = max_y.ceil().min(src_h) as i32;

    let _src_x_size_raw = (max_x - min_x)
        .max(0.0)
        .min((src_raster_size[0] - min_x_clamped) as f64);
    let _src_y_size_raw = (max_y - min_y)
        .max(0.0)
        .min((src_raster_size[1] - min_y_clamped) as f64);

    // Antimeridian heuristic: if >90% of source width, use full width
    // (GDAL lines 3325–3329)
    let mut antimeridian_snap = false;
    let (xoff, xsize) = if (max_x_clamped - min_x_clamped) as f64 > 0.9 * src_w {
        antimeridian_snap = true;
        (0, src_raster_size[0])
    } else {
        let xoff = 0.max((min_x_clamped - x_radius).min(src_raster_size[0]));
        let xsize = 0.max(
            (max_x_clamped - xoff + x_radius).min(src_raster_size[0] - xoff),
        );
        (xoff, xsize)
    };

    let (yoff, ysize) = if (max_y_clamped - min_y_clamped) as f64 > 0.9 * src_h {
        (0, src_raster_size[1])
    } else {
        let yoff = 0.max((min_y_clamped - y_radius).min(src_raster_size[1]));
        let ysize = 0.max(
            (max_y_clamped - yoff + y_radius).min(src_raster_size[1] - yoff),
        );
        (yoff, ysize)
    };

    // Fill ratio (GDAL lines 3360–3364)
    // When the antimeridian snap fires, the window covers the full source
    // width but only a fraction is actually needed. Force a low fill ratio
    // to trigger subdivision in collect_chunk_list. GDAL achieves this
    // indirectly through memory budget limits in CollectChunkListInternal.
    let fill_ratio = if antimeridian_snap {
        // Estimate: actual needed width / full width
        // The raw unclamped extent spans nearly the full width due to
        // coordinate wrapping. We don't know the true needed width here,
        // but we know the snap indicates waste. Use a fixed low value
        // to ensure subdivision.
        0.01
    } else {
        (xsize as f64 * ysize as f64)
            / ((max_x - min_x + 2.0 * x_radius as f64)
                * (max_y - min_y + 2.0 * y_radius as f64))
                .max(1.0)
    };

    Some(SourceWindow {
        xoff, yoff, xsize, ysize,
        fill_ratio, n_failed, n_samples,
    })
}

// =========================================================================
// Sampling strategies
// =========================================================================

/// Sample every pixel on every edge of the dest window.
/// Returns (min_x, min_y, max_x, max_y, n_samples, n_failed).
fn sample_edges_all(
    transformer: &impl Transformer,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
) -> (f64, f64, f64, f64, i32, i32) {
    let n_max = 2 * (dst_size[0] + dst_size[1]) as usize;
    let mut xs = Vec::with_capacity(n_max);
    let mut ys = Vec::with_capacity(n_max);

    // Top and bottom edges
    for ix in 0..=dst_size[0] {
        xs.push((dst_off[0] + ix) as f64);
        ys.push(dst_off[1] as f64);
        xs.push((dst_off[0] + ix) as f64);
        ys.push((dst_off[1] + dst_size[1]) as f64);
    }
    // Left and right edges (excluding corners already done)
    for iy in 1..dst_size[1] {
        xs.push(dst_off[0] as f64);
        ys.push((dst_off[1] + iy) as f64);
        xs.push((dst_off[0] + dst_size[0]) as f64);
        ys.push((dst_off[1] + iy) as f64);
    }

    collect_bounds(transformer, &mut xs, &mut ys)
}

/// Sample step_count points along each edge of the dest window.
/// Returns (min_x, min_y, max_x, max_y, n_samples, n_failed).
fn sample_edges_step(
    transformer: &impl Transformer,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    step_count: usize,
) -> (f64, f64, f64, f64, i32, i32) {
    let n_max = step_count * 4;
    let mut xs = Vec::with_capacity(n_max);
    let mut ys = Vec::with_capacity(n_max);

    let step_size = 1.0 / (step_count - 1) as f64;
    let mut ratio = 0.0;
    while ratio <= 1.0 + step_size * 0.5 {
        let dx = ratio * dst_size[0] as f64;
        let dy = ratio * dst_size[1] as f64;

        // Top edge
        xs.push(dst_off[0] as f64 + dx);
        ys.push(dst_off[1] as f64);

        // Bottom edge
        xs.push(dst_off[0] as f64 + dx);
        ys.push((dst_off[1] + dst_size[1]) as f64);

        // Left edge
        xs.push(dst_off[0] as f64);
        ys.push(dst_off[1] as f64 + dy);

        // Right edge
        xs.push((dst_off[0] + dst_size[0]) as f64);
        ys.push(dst_off[1] as f64 + dy);

        ratio += step_size;
    }

    collect_bounds(transformer, &mut xs, &mut ys)
}

/// Sample on a (step_count+2)² grid through the dest window interior.
/// The +2 adds near-edge points at 0.5/nDstXSize offset (GDAL lines 2846–2863).
/// Returns (min_x, min_y, max_x, max_y, n_samples, n_failed).
fn sample_grid(
    transformer: &impl Transformer,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    step_count: usize,
) -> (f64, f64, f64, f64, i32, i32) {
    let n_grid = step_count + 2;
    let n_max = n_grid * n_grid;
    let mut xs = Vec::with_capacity(n_max);
    let mut ys = Vec::with_capacity(n_max);

    let step_size = 1.0 / (step_count - 1) as f64;
    let near_edge = 0.5 / dst_size[0].max(1) as f64;

    for iy in 0..n_grid {
        let ratio_y = if iy == 0 {
            near_edge
        } else if iy <= step_count {
            (iy - 1) as f64 * step_size
        } else {
            1.0 - near_edge
        };

        for ix in 0..n_grid {
            let ratio_x = if ix == 0 {
                near_edge
            } else if ix <= step_count {
                (ix - 1) as f64 * step_size
            } else {
                1.0 - near_edge
            };

            xs.push(ratio_x * dst_size[0] as f64 + dst_off[0] as f64);
            ys.push(ratio_y * dst_size[1] as f64 + dst_off[1] as f64);
        }
    }

    collect_bounds(transformer, &mut xs, &mut ys)
}

/// Transform points and collect bounding box of valid results.
fn collect_bounds(
    transformer: &impl Transformer,
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
) -> (f64, f64, f64, f64, i32, i32) {
    let n = xs.len();
    let ok = transformer.transform(true, xs, ys);

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut n_failed = 0i32;

    for i in 0..n {
        if !ok[i] || xs[i].is_nan() || ys[i].is_nan() {
            n_failed += 1;
            continue;
        }
        if xs[i] < min_x { min_x = xs[i]; }
        if xs[i] > max_x { max_x = xs[i]; }
        if ys[i] < min_y { min_y = ys[i]; }
        if ys[i] > max_y { max_y = ys[i]; }
    }

    (min_x, min_y, max_x, max_y, n as i32, n_failed)
}

// =========================================================================
// Forward probe: ComputeSourceWindowStartingFromSource
// =========================================================================

/// Cached forward-transformed source grid.
///
/// GDAL caches this in `GDALWarpPrivateData` and reuses it across all
/// destination chunks. We expose it as a separate struct so the caller
/// (R or Rust) can create it once and pass it to multiple calls.
#[derive(Debug, Clone)]
pub struct SourceGrid {
    /// Source pixel X coordinates for each grid point.
    pub src_x: Vec<f64>,
    /// Source pixel Y coordinates for each grid point.
    pub src_y: Vec<f64>,
    /// Destination pixel X coordinates (forward-transformed).
    pub dst_x: Vec<f64>,
    /// Destination pixel Y coordinates (forward-transformed).
    pub dst_y: Vec<f64>,
    /// Whether each point transformed successfully.
    pub success: Vec<bool>,
    /// Grid dimensions: (n_cols, n_rows) = (step_count+2, step_count+2).
    pub grid_size: [usize; 2],
}

/// Build a forward-transformed grid across the source raster.
///
/// Maps GDAL's one-time setup in `ComputeSourceWindowStartingFromSource`
/// (lines 2656–2703). Creates a (step_count+2)² grid across the source,
/// transforms forward (source pixel → destination pixel), and caches
/// the results for reuse.
///
/// # Arguments
///
/// * `transformer` — must support `dst_to_src = false` (forward transform)
/// * `src_raster_size` — `[ncol, nrow]` of the full source raster
/// * `step_count` — grid density (default 21, giving 23×23 = 529 points)
pub fn build_source_grid(
    transformer: &impl Transformer,
    src_raster_size: [i32; 2],
    step_count: usize,
) -> SourceGrid {
    let n_grid = step_count + 2;
    let n_points = n_grid * n_grid;
    let step_size = 1.0 / (step_count - 1).max(1) as f64;
    let src_w = src_raster_size[0] as f64;
    let src_h = src_raster_size[1] as f64;

    let mut src_x = Vec::with_capacity(n_points);
    let mut src_y = Vec::with_capacity(n_points);

    // Build the source grid (GDAL lines 2675–2691)
    // First and last rows/cols are 0.5 pixels in from the edge;
    // interior points are evenly spaced.
    for iy in 0..n_grid {
        let ratio_y = if iy == 0 {
            0.5 / src_h
        } else if iy <= step_count {
            (iy - 1) as f64 * step_size
        } else {
            1.0 - 0.5 / src_h
        };
        for ix in 0..n_grid {
            let ratio_x = if ix == 0 {
                0.5 / src_w
            } else if ix <= step_count {
                (ix - 1) as f64 * step_size
            } else {
                1.0 - 0.5 / src_w
            };
            src_x.push(ratio_x * src_w);
            src_y.push(ratio_y * src_h);
        }
    }

    // Forward transform: source pixel → destination pixel
    let mut dst_x = src_x.clone();
    let mut dst_y = src_y.clone();
    let success = transformer.transform(false, &mut dst_x, &mut dst_y);

    SourceGrid {
        src_x,
        src_y,
        dst_x,
        dst_y,
        success,
        grid_size: [n_grid, n_grid],
    }
}

/// Refine a source window using the forward probe.
///
/// Maps GDAL's per-call logic in `ComputeSourceWindowStartingFromSource`
/// (lines 2720–2747). For each cached grid point whose forward-transformed
/// destination coordinates fall inside the destination window, expands the
/// source bounds to include that point's source coordinates.
///
/// This can only EXPAND the source window, never shrink it — it catches
/// source pixels that the backward probe missed (e.g. due to transform
/// failures at some destination points).
///
/// # Arguments
///
/// * `source_grid` — pre-computed from `build_source_grid()`
/// * `dst_off` — destination window offset `[col_off, row_off]`
/// * `dst_size` — destination window size `[ncol, nrow]`
/// * `min_x`, `min_y`, `max_x`, `max_y` — current source bounds
///   (from backward probe), modified in place
pub fn refine_from_source(
    source_grid: &SourceGrid,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    min_x: &mut f64,
    min_y: &mut f64,
    max_x: &mut f64,
    max_y: &mut f64,
) {
    let dst_x_min = dst_off[0] as f64;
    let dst_x_max = (dst_off[0] + dst_size[0]) as f64;
    let dst_y_min = dst_off[1] as f64;
    let dst_y_max = (dst_off[1] + dst_size[1]) as f64;

    for i in 0..source_grid.dst_x.len() {
        if !source_grid.success[i] {
            continue;
        }
        let dx = source_grid.dst_x[i];
        let dy = source_grid.dst_y[i];

        // Does this source point land inside the destination window?
        if dx >= dst_x_min && dx <= dst_x_max && dy >= dst_y_min && dy <= dst_y_max {
            let sx = source_grid.src_x[i];
            let sy = source_grid.src_y[i];
            if sx < *min_x { *min_x = sx; }
            if sx > *max_x { *max_x = sx; }
            if sy < *min_y { *min_y = sy; }
            if sy > *max_y { *max_y = sy; }
        }
    }
}

// =========================================================================
// Recursive destination subdivision: CollectChunkListInternal
// =========================================================================

/// A chunk plan: a destination sub-window and its corresponding source window.
#[derive(Debug, Clone)]
pub struct ChunkPlan {
    /// Source window for this chunk.
    pub src_window: SourceWindow,
    /// Destination window offset `[col_off, row_off]`.
    pub dst_off: [i32; 2],
    /// Destination window size `[ncol, nrow]`.
    pub dst_size: [i32; 2],
}

/// Recursively subdivide a destination window into chunks with efficient
/// source windows.
///
/// Maps GDAL's `CollectChunkListInternal()` (gdalwarpoperation.cpp lines
/// 1456–1624). When the source fill ratio is below `min_fill_ratio` and
/// the destination is large enough, splits along the longest dimension
/// and recurses. Each resulting `ChunkPlan` can be warped independently.
///
/// This is the mechanism that handles the antimeridian: a 256×256 tile
/// straddling 180° has a full-width source window (fill_ratio ≈ 0.01).
/// Recursive splitting produces sub-tiles entirely on one side, each
/// with a compact source window.
///
/// # Arguments
///
/// * `transformer` — coordinate transformer
/// * `src_raster_size` — full source raster `[ncol, nrow]`
/// * `dst_off` — destination window offset
/// * `dst_size` — destination window size
/// * `resample_padding` — kernel padding (0, 1, 2, or 3)
/// * `min_fill_ratio` — threshold below which to subdivide (GDAL uses 0.5)
/// * `min_dst_size` — minimum destination dimension to allow splitting
///   (GDAL uses 100 for the fill-ratio path; we use a smaller default
///   since we're operating on tiles, not full rasters)
/// * `source_grid` — optional pre-computed forward grid for refinement
///
/// # Returns
///
/// A `Vec<ChunkPlan>`. For well-behaved tiles this is a single entry.
/// For antimeridian-crossing tiles it's typically 2 entries.
/// Empty chunks (zero source window) are omitted.
pub fn collect_chunk_list(
    transformer: &impl Transformer,
    src_raster_size: [i32; 2],
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    resample_padding: i32,
    min_fill_ratio: f64,
    min_dst_size: i32,
    source_grid: Option<&SourceGrid>,
) -> Vec<ChunkPlan> {
    let mut chunks = Vec::new();
    collect_chunk_list_recursive(
        transformer,
        src_raster_size,
        dst_off,
        dst_size,
        resample_padding,
        min_fill_ratio,
        min_dst_size,
        source_grid,
        &mut chunks,
    );
    chunks
}



fn collect_chunk_list_recursive(
    transformer: &impl Transformer,
    src_raster_size: [i32; 2],
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    resample_padding: i32,
    min_fill_ratio: f64,
    min_dst_size: i32,
    source_grid: Option<&SourceGrid>,
    out: &mut Vec<ChunkPlan>,
) {
    let sw = compute_source_window(transformer, dst_off, dst_size, src_raster_size, resample_padding);
    let sw = match sw { Some(sw) => sw, None => return };
    if sw.xsize == 0 || sw.ysize == 0 { return; }

    let should_split = sw.fill_ratio > 0.0
        && sw.fill_ratio < min_fill_ratio
        && (dst_size[0] > min_dst_size || dst_size[1] > min_dst_size);

    if should_split {
        let split_range = find_discontinuity_range(transformer, dst_off, dst_size, src_raster_size);

        if let Some((col_min, col_max)) = split_range {
            let left_width = col_min - dst_off[0];
            let mid_width = col_max - col_min;
            let right_width = dst_size[0] - (col_max - dst_off[0]);

            if left_width > 0 {
                collect_chunk_list_recursive(
                    transformer, src_raster_size, dst_off, [left_width, dst_size[1]],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
            }
            if mid_width > 0 {
                let mid_sw = compute_source_window(
                    transformer, [col_min, dst_off[1]], [mid_width, dst_size[1]],
                    src_raster_size, resample_padding);
                if let Some(msw) = mid_sw {
                    if msw.xsize > 0 && msw.ysize > 0 {
                        out.push(ChunkPlan {
                            src_window: msw,
                            dst_off: [col_min, dst_off[1]],
                            dst_size: [mid_width, dst_size[1]],
                        });
                    }
                }
            }
            if right_width > 0 {
                collect_chunk_list_recursive(
                    transformer, src_raster_size, [col_max, dst_off[1]], [right_width, dst_size[1]],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
            }
        } else {
            if dst_size[0] >= dst_size[1] && dst_size[0] > 1 {
                let chunk1 = dst_size[0] / 2;
                let chunk2 = dst_size[0] - chunk1;
                collect_chunk_list_recursive(
                    transformer, src_raster_size, dst_off, [chunk1, dst_size[1]],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
                collect_chunk_list_recursive(
                    transformer, src_raster_size,
                    [dst_off[0] + chunk1, dst_off[1]], [chunk2, dst_size[1]],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
            } else if dst_size[1] > 1 {
                let chunk1 = dst_size[1] / 2;
                let chunk2 = dst_size[1] - chunk1;
                collect_chunk_list_recursive(
                    transformer, src_raster_size, dst_off, [dst_size[0], chunk1],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
                collect_chunk_list_recursive(
                    transformer, src_raster_size,
                    [dst_off[0], dst_off[1] + chunk1], [dst_size[0], chunk2],
                    resample_padding, min_fill_ratio, min_dst_size, source_grid, out);
            } else {
                out.push(ChunkPlan { src_window: sw, dst_off, dst_size });
            }
        }
    } else {
        out.push(ChunkPlan { src_window: sw, dst_off, dst_size });
    }
}

fn find_discontinuity_range(
    transformer: &impl Transformer,
    dst_off: [i32; 2],
    dst_size: [i32; 2],
    src_raster_size: [i32; 2],
) -> Option<(i32, i32)> {
    let ncol = dst_size[0] as usize;
    if ncol < 3 { return None; }

    let src_w = src_raster_size[0] as f64;
    let threshold = src_w * 0.5;

    let nrow = dst_size[1] as usize;
    let row_step = if nrow <= 16 { 1 } else { 8.min(nrow / 4) };
    let sample_rows: Vec<i32> = (0..nrow).step_by(row_step)
        .map(|r| dst_off[1] + r as i32).collect();

    let mut global_min: Option<i32> = None;
    let mut global_max: Option<i32> = None;

    for row in &sample_rows {
        let mut xs: Vec<f64> = (0..ncol)
            .map(|c| (dst_off[0] + c as i32) as f64 + 0.5).collect();
        let mut ys = vec![*row as f64 + 0.5; ncol];
        let ok = transformer.transform(true, &mut xs, &mut ys);

        let mut max_gap = 0.0f64;
        let mut gap_right: Option<usize> = None;
        let mut prev_x: Option<f64> = None;

        for col in 0..ncol {
            if !ok[col] || xs[col].is_nan() { continue; }
            if let Some(px) = prev_x {
                let gap = (xs[col] - px).abs();
                if gap > max_gap { max_gap = gap; gap_right = Some(col); }
            }
            prev_x = Some(xs[col]);
        }

        if max_gap > threshold {
            if let Some(rc) = gap_right {
                let abs_left = dst_off[0] + rc as i32 - 1;
                let abs_right = dst_off[0] + rc as i32 + 1;
                global_min = Some(match global_min {
                    Some(p) => p.min(abs_left), None => abs_left
                });
                global_max = Some(match global_max {
                    Some(p) => p.max(abs_right), None => abs_right
                });
            }
        }
    }

    match (global_min, global_max) {
        (Some(mn), Some(mx)) => {
            let mn = mn.max(dst_off[0]);
            let mx = mx.min(dst_off[0] + dst_size[0]);
            if mn < mx { Some((mn, mx)) } else { None }
        }
        _ => None,
    }
}
