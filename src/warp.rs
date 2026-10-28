//! Warp kernels: nearest-neighbour, bilinear, cubic, lanczos, average, mode.
//!
//! Maps GDAL's gdalwarpkernel.cpp (~line 5510 onwards) and
//! gdalresamplingkernels.h.
//!
//! Pattern:
//!   for each output scanline:
//!     build array of destination pixel centres (col + 0.5, row + 0.5)
//!     transform dst→src via Transformer
//!     for each output pixel:
//!       resample source pixels at the transformed coordinate
//!
//! The scanline loop is shared; only the per-pixel resampling differs.
//!
//! Two entry points: [`warp`], generic over [`Sample`] types, for
//! pixel-interleaved images of any band count in one pass, configured by
//! [`WarpOptions`]; and [`warp_resample`], a single `i32` band with a nodata
//! value and fixed (point) kernels, kept as the reference the generic kernel
//! is tested against. The interleaved path is 3-5x faster for RGBA because
//! it transforms and weights each output pixel once rather than once per
//! band.
//!
//! ## Downsampling and kernel scaling
//!
//! When downsampling, GDAL widens the bilinear, cubic and lanczos kernels so
//! that they cover the source area of a destination pixel (antialiasing).
//! [`KernelScaling`] selects the behaviour:
//!
//! - [`KernelScaling::WidenWhenDownsampling`] (default) estimates the
//!   destination/source scale per call, as GDAL does per warp chunk
//!   (`GDALWarpKernel::PerformWarp`, gdalwarpkernel.cpp ~L1092-1295), see
//!   [`estimate_scale`].
//! - [`KernelScaling::Fixed`] supplies the scale, like GDAL's `XSCALE` and
//!   `YSCALE` warp options. Useful when one call covers a region whose scale
//!   varies, or to make output independent of how the destination is tiled.
//! - [`KernelScaling::Point`] keeps the fixed 2×2 / 4×4 / 6×6 support at
//!   every ratio. This is what gdalwarp does with
//!   `-wo XSCALE=1 -wo YSCALE=1`.
//!
//! Following GDAL, widening applies per axis to scales below 1 (upsampling
//! never narrows the kernel): the support becomes `ceil(radius / scale)`
//! and the kernel is evaluated at `(i - delta) * scale`
//! (gdalwarpkernel.cpp ~L1319, `GWKResampleNoMasksT` ~L4840,
//! `GWKResample` ~L4027). Bilinear and cubic only widen once a scale drops
//! to 0.5 or below; between 1:1 and 2:1 GDAL keeps the 4-sample point
//! formula (`CanUse4SamplesFormula`, ~L1041). Lanczos widens at any scale
//! below 1.
//!
//! The widened kernel clips its window at the buffer edge and renormalises
//! by the weights of the valid pixels it covers, skipping nodata (GDAL's
//! `GWKResample`). The point kernels keep their edge and nodata fallbacks
//! described below.
//!
//! ## Average
//!
//! [`ResampleAlg::Average`] is the weighted mean of the source pixels a
//! destination pixel's footprint covers, over the valid ones only (nodata
//! pixels add neither value nor weight, so the mean is renormalised).
//! [`Footprint`] chooses how coverage is measured, in source pixel space:
//!
//! - [`Footprint::GdalBox`] (default) is GDAL's: transform the destination
//!   pixel's top-left and bottom-right corners, take the axis-aligned box
//!   between them, weight each source pixel by the fraction of it inside
//!   the box per axis (`GWKAverageOrModeThread`, gdalwarpkernel.cpp
//!   ~L6845-7030, ~L7960). It follows GDAL's quirks: only two corners are
//!   used, so a rotated pixel can collapse to a thin box; a box overhanging
//!   the buffer edge weights the edge pixel above 1; a target pixel on the
//!   antimeridian wraps past the right edge.
//! - [`Footprint::Exact`] intersects the transformed outline exactly with
//!   the source pixels, which with renormalisation is first-order
//!   conservative remapping with areas in source pixel space.
//!
//! For a destination pixel that maps to an axis-aligned rectangle the two
//! agree. The exact footprint costs about 25x the box (4 µs a pixel at
//! 4:1 in the timing test).
//!
//! ## Mode
//!
//! [`ResampleAlg::Mode`] uses the same footprints, with the covered area of
//! each distinct value as its count (accumulated in `f32`, as GDAL does, so
//! ties fall in the same places), per band over valid pixels.
//! [`ModeTies`] breaks ties, as GDAL's `MODE_TIES`. The count comparison
//! runs every time a value's count grows, in source scan order, which is
//! GDAL's 8/16-bit histogram path (gdalwarpkernel.cpp ~L8115-8170); GDAL's
//! path for wider types (`GWKModeRealType`, ~L7147-7235) skips it on a
//! value's first occurrence, so its `Min` / `Max` depend on the data type.
//! rwarp gives the 8/16-bit answer for every type.
//!
//! ## Output values: rounding, clamping, nodata, source bounds
//!
//! Every resampling method computes in `f64`. Integer outputs clamp to the
//! type and round half up, `floor(v + 0.5)` ([`Sample::from_f64`]), as GDAL
//! does with an integer working type for bilinear, cubic, lanczos and
//! average alike (`ClampRoundAndAvoidNoData` / `GWKClampValueT`,
//! gdalwarpkernel.cpp ~L1587, ~L1856). Nearest and mode copy a source value
//! and never round.
//!
//! A resampled value equal to the nodata value is moved off it by one step
//! ([`Sample::step_off`]: one less for integers, the next float up), so a
//! valid output pixel does not read back as missing (GDAL's `AvoidNoData`,
//! ~L1613; GDAL also warns once).
//!
//! Sample points outside `[0, n)` of the source buffer are off the source
//! for every method but the area ones, which use footprints
//! (`GWKCheckAndComputeSrcOffsets`, ~L5251). GDAL retries such points with
//! the exact transformer when an approximate one is in use; rwarp uses the
//! transformer it is given.
//!
//! ## Nodata handling (point kernels)
//!
//! **Nearest-neighbour**: nodata pixels in source are skipped; destination
//! retains its fill value.
//!
//! **Bilinear**: if any of the 4 contributing source pixels is nodata, the
//! destination pixel is set to nodata. At edges, valid pixels are weighted
//! and renormalized (weight sum < 1e-5 → nodata).
//!
//! **Cubic**: if any pixel in the 4×4 neighbourhood is nodata, falls back
//! to bilinear with the same nodata rule.
//!
//! **Lanczos**: the 6×6 window is clipped to the buffer and nodata pixels
//! are skipped, renormalising by the weight of the pixels used, as GDAL
//! does at every scale.
//!
//! **All interpolating kernels**: a destination pixel whose centre lands on
//! a nodata source pixel gets no value, whatever its neighbours.

use crate::transform::{transform_scanline, Transformer};

/// Resampling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleAlg {
    NearestNeighbour,  // 1 pixel,  radius 0
    Bilinear,          // 2×2,      radius 1
    Cubic,             // 4×4,      radius 2  (Catmull-Rom, a = -0.5)
    Lanczos,           // 6×6,      radius 3  (sinc windowed sinc)
    /// Area-weighted mean of the source pixels a destination pixel covers,
    /// over the valid ones; see [`Footprint`].
    Average,
    /// Most common value (by covered area) among the valid source pixels a
    /// destination pixel covers, per band; see [`Footprint`] and
    /// [`ModeTies`].
    Mode,
}

impl ResampleAlg {
    /// Filter radius in pixels (matches GWKGetFilterRadius).
    pub fn radius(self) -> i32 {
        match self {
            ResampleAlg::NearestNeighbour => 0,
            ResampleAlg::Bilinear => 1,
            ResampleAlg::Cubic => 2,
            ResampleAlg::Lanczos => 3,
            ResampleAlg::Average | ResampleAlg::Mode => 0,
        }
    }

    /// Kernel weight at distance `x` (GDAL's `apfGWKFilter`). Zero for
    /// nearest, which has no kernel.
    fn weight(self, x: f64) -> f64 {
        match self {
            ResampleAlg::NearestNeighbour | ResampleAlg::Average | ResampleAlg::Mode => 0.0,
            ResampleAlg::Bilinear => {
                let ax = x.abs();
                if ax <= 1.0 { 1.0 - ax } else { 0.0 }
            }
            ResampleAlg::Cubic => cubic_weight(x),
            ResampleAlg::Lanczos => lanczos_weight(x),
        }
    }
}

/// How the interpolating kernels respond to downsampling. See the module
/// documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelScaling {
    /// Fixed kernel support at every ratio (gdalwarp `-wo XSCALE=1
    /// -wo YSCALE=1`).
    Point,
    /// Widen the kernel by the destination/source scale estimated from the
    /// transformer over the destination window (GDAL's default).
    WidenWhenDownsampling,
    /// Widen the kernel by a given destination/source scale per axis
    /// (destination pixels per source pixel; below 1 is downsampling), as
    /// GDAL's `XSCALE` / `YSCALE` warp options.
    Fixed { x: f64, y: f64 },
}

/// How [`ResampleAlg::Average`] and [`ResampleAlg::Mode`] measure the
/// overlap of a destination pixel with source pixels. Both measure area in source pixel space, which
/// is planar area in the source CRS for an affine source; neither is area
/// in the target CRS or on the ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Footprint {
    /// GDAL's footprint (`GWKAverageOrModeThread`, gdalwarpkernel.cpp
    /// ~L7577): the axis-aligned box between the transformed top-left and
    /// bottom-right corners of the destination pixel, with fractional
    /// weights for the source pixels it partly covers. Exact when the
    /// destination pixel maps to an axis-aligned rectangle; under rotation,
    /// shear or curvature the box can be far from the true footprint.
    GdalBox,
    /// The destination pixel's outline, each edge split into `densify`
    /// segments, transformed to source pixel space and intersected exactly
    /// with the source pixels (coverage fractions from [`controlledburn`],
    /// stored as `f32`). With renormalisation over valid pixels this is
    /// first-order conservative remapping. No antimeridian handling.
    Exact { densify: usize },
}

/// How [`ResampleAlg::Mode`] breaks a tie between values with equal
/// covered area (GDAL's `MODE_TIES` warp option, GDAL >= 3.11).
///
/// Counts are re-compared each time a value's count grows, in source scan
/// order (row by row, then column), so `First` is the tied value that
/// *reached* the winning count first, not the one seen first:
/// `[[5, 3], [3, 5]]` gives 3. This is what GDAL does for 8- and 16-bit
/// data. For wider types (32/64-bit, floating) GDAL skips the comparison on
/// a value's first occurrence, so `Min` / `Max` are not applied to ties at
/// a count reached that way (all-distinct `[[5, 3], [7, 1]]` with `Min`:
/// GDAL gives 1 for Byte, 5 for Int32); rwarp gives the same answer for
/// every type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModeTies {
    #[default]
    First,
    Min,
    Max,
}

/// Options for [`warp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpOptions<T> {
    pub alg: ResampleAlg,
    pub scaling: KernelScaling,
    /// Footprint for [`ResampleAlg::Average`] and [`ResampleAlg::Mode`];
    /// ignored by the other methods.
    pub footprint: Footprint,
    /// Tie-breaking for [`ResampleAlg::Mode`]; ignored by the other
    /// methods.
    pub mode_ties: ModeTies,
    /// Source pixels where any band equals this are nodata, and it is the
    /// fill for destination pixels with no value. `None` fills with
    /// [`Sample::zero`].
    pub nodata: Option<T>,
    /// Band whose zero values mark source pixels as nodata (an alpha band).
    pub mask_band: Option<usize>,
    /// Offset of the destination window in the grid the transformer maps
    /// from, so a tile of a larger destination can be warped on its own.
    pub dst_col_off: usize,
    pub dst_row_off: usize,
}

impl<T: Sample> Default for WarpOptions<T> {
    fn default() -> Self {
        WarpOptions {
            alg: ResampleAlg::NearestNeighbour,
            scaling: KernelScaling::WidenWhenDownsampling,
            footprint: Footprint::GdalBox,
            mode_ties: ModeTies::First,
            nodata: None,
            mask_band: None,
            dst_col_off: 0,
            dst_row_off: 0,
        }
    }
}

/// A pixel-interleaved source buffer, laid out `(row, col, band)`, that is a
/// window of the full source image.
#[derive(Debug, Clone, Copy)]
pub struct SourceBuffer<'a, T> {
    pub data: &'a [T],
    pub ncol: usize,
    pub nrow: usize,
    pub nbands: usize,
    /// Offset of the buffer in the full source image (the transformer's
    /// pixel space).
    pub col_off: usize,
    pub row_off: usize,
}

/// Estimate the destination/source scale per axis of a destination window,
/// as GDAL does for each warp chunk (`GDALWarpKernel::PerformWarp`,
/// gdalwarpkernel.cpp ~L1092-1295).
///
/// The transformer maps a unit square at up to 10×10 points over the window
/// to source pixels. A square that stays a quasi-rectangle (edges within
/// ~20° of the axes, opposite sides within 10% in length) gives one scale per
/// axis from the reciprocal of its side lengths; otherwise both axes get the
/// reciprocal square root of its area. Samples below 10% of the per-axis
/// maximum (a polar discontinuity, say) are dropped and the rest averaged.
/// A scale whose reciprocal is within 0.05 of an integer is snapped to it.
/// Returns `(1.0, 1.0)` if nothing transforms.
///
/// GDAL samples the chunk at `0..nDstXSize` without adding the chunk's
/// offset (from reading the code; not yet checked against a multi-chunk
/// gdalwarp run). This adds `dst_col_off` / `dst_row_off`, so the estimate
/// is for the window actually warped. For a single-chunk warp, as in the
/// test fixtures, the two agree.
pub fn estimate_scale(
    transformer: &impl Transformer,
    dst_ncol: usize,
    dst_nrow: usize,
    dst_col_off: usize,
    dst_row_off: usize,
) -> (f64, f64) {
    const MAX_POINTS_PER_DIM: usize = 10;
    let npx = dst_ncol.min(MAX_POINTS_PER_DIM);
    let npy = dst_nrow.min(MAX_POINTS_PER_DIM);
    // Sample positions span the window, the last pulled in by a pixel so
    // its unit square stays inside.
    let sample = |i: usize, np: usize, n: usize| -> f64 {
        let shift = if i > 0 && i == np - 1 { -1.0 } else { 0.0 };
        shift + if np == 1 { 0.0 } else { i as f64 * n as f64 / (np - 1) as f64 }
    };
    let mut xs = Vec::with_capacity(4 * npx * npy);
    let mut ys = Vec::with_capacity(4 * npx * npy);
    for iy in 0..npy {
        let y = sample(iy, npy, dst_nrow) + dst_row_off as f64;
        for ix in 0..npx {
            let x = sample(ix, npx, dst_ncol) + dst_col_off as f64;
            // Corners 0..3: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
            xs.extend_from_slice(&[x, x + 1.0, x, x + 1.0]);
            ys.extend_from_slice(&[y, y, y + 1.0, y + 1.0]);
        }
    }
    let ok = transformer.transform(true, &mut xs, &mut ys);

    const TAN_MODEST_ANGLE: f64 = 0.35;
    const LENGTH_RELATIVE_TOLERANCE: f64 = 0.1;
    const EPSILON: f64 = 1e-10;
    let mut scales: Vec<(f64, f64)> = Vec::new();
    for i in (0..xs.len()).step_by(4) {
        if !(ok[i] && ok[i + 1] && ok[i + 2] && ok[i + 3]) {
            continue;
        }
        let (vx01, vy01) = (xs[i + 1] - xs[i], ys[i + 1] - ys[i]);
        let (vx23, vy23) = (xs[i + 3] - xs[i + 2], ys[i + 3] - ys[i + 2]);
        let (vx02, vy02) = (xs[i + 2] - xs[i], ys[i + 2] - ys[i]);
        let (vx13, vy13) = (xs[i + 3] - xs[i + 1], ys[i + 3] - ys[i + 1]);
        let len01 = vx01 * vx01 + vy01 * vy01;
        let len23 = vx23 * vx23 + vy23 * vy23;
        let len02 = vx02 * vx02 + vy02 * vy02;
        let len13 = vx13 * vx13 + vy13 * vy13;
        if vy01.abs() < TAN_MODEST_ANGLE * vx01
            && vy23.abs() < TAN_MODEST_ANGLE * vx23
            && (len01 - len23).abs() < LENGTH_RELATIVE_TOLERANCE * len01.abs()
            && (len02 - len13).abs() < LENGTH_RELATIVE_TOLERANCE * len02.abs()
        {
            // Geometric mean of opposite sides (still squared, hence two
            // square roots).
            let xlen = (len01 * len23).sqrt().sqrt();
            let ylen = (len02 * len13).sqrt().sqrt();
            if xlen > EPSILON && ylen > EPSILON {
                scales.push((1.0 / xlen, 1.0 / ylen));
            }
        } else {
            // Shoelace area of the quadrilateral 0, 1, 3, 2.
            let (px, py) = ([xs[i], xs[i + 1], xs[i + 3], xs[i + 2]], [ys[i], ys[i + 1], ys[i + 3], ys[i + 2]]);
            let mut sum = 0.0;
            for k in 0..4 {
                sum += px[k] * (py[(k + 1) % 4] - py[(k + 3) % 4]);
            }
            let factor = (0.5 * sum.abs()).sqrt();
            if factor > EPSILON {
                scales.push((1.0 / factor, 1.0 / factor));
            }
        }
    }

    let (mut sx, mut sy) = (0.0, 0.0);
    if !scales.is_empty() {
        // GDAL sorts by area scale before averaging; kept so the running
        // mean sums in the same order.
        scales.sort_by(|a, b| (a.0 * a.1).total_cmp(&(b.0 * b.1)));
        let xmax = scales.iter().fold(0.0f64, |m, s| m.max(s.0));
        let ymax = scales.iter().fold(0.0f64, |m, s| m.max(s.1));
        const THRESHOLD: f64 = 0.1;
        let mut n = 0;
        for &(x, y) in &scales {
            if x > THRESHOLD * xmax && y > THRESHOLD * ymax {
                n += 1;
                let inv = 1.0 / n as f64;
                sx += (x - sx) * inv;
                sy += (y - sy) * inv;
            }
        }
    }
    let snap = |s: f64| -> f64 {
        if s <= 0.0 {
            return 1.0;
        }
        if s < 1.0 {
            let r = 1.0 / s;
            let n = (r + 0.5) as i64;
            if (r - n as f64).abs() < 0.05 {
                return 1.0 / n as f64;
            }
        }
        s
    };
    (snap(sx), snap(sy))
}

/// Kernel parameters for a widened warp: the scale the kernel is evaluated
/// at and the support radius, per axis.
#[derive(Debug, Clone, Copy)]
struct Widened {
    alg: ResampleAlg,
    sx: f64,
    sy: f64,
    rx: i64,
    ry: i64,
}

/// Lanczos at its native support. GDAL runs lanczos through the same
/// clipped, renormalised kernel at every scale (`GWKResampleOptimizedLanczos`,
/// gdalwarpkernel.cpp ~L4188).
const LANCZOS_POINT: Widened = Widened { alg: ResampleAlg::Lanczos, sx: 1.0, sy: 1.0, rx: 3, ry: 3 };

impl Widened {
    /// `None` when GDAL would use the point kernel at this scale.
    fn new(alg: ResampleAlg, sx: f64, sy: f64) -> Option<Self> {
        let widen = match alg {
            ResampleAlg::NearestNeighbour | ResampleAlg::Average | ResampleAlg::Mode => false,
            ResampleAlg::Bilinear | ResampleAlg::Cubic => !(sx > 0.5 && sy > 0.5),
            ResampleAlg::Lanczos => sx < 1.0 || sy < 1.0,
        };
        if !widen {
            return None;
        }
        let f = alg.radius() as f64;
        let radius = |s: f64| if s < 1.0 { (f / s).ceil() as i64 } else { f as i64 };
        Some(Widened { alg, sx: sx.min(1.0), sy: sy.min(1.0), rx: radius(sx), ry: radius(sy) })
    }
}

// =========================================================================
// Public warp entry points
// =========================================================================

/// Warp with specified resampling algorithm.
///
/// # Arguments
/// * `transformer` - maps dst pixel coords → src pixel coords
/// * `src_pixels` - source pixel buffer (row-major, `src_ncol` wide)
/// * `src_ncol`, `src_nrow` - source buffer dimensions
/// * `src_col_off`, `src_row_off` - buffer offset in full source image
/// * `dst_ncol`, `dst_nrow` - destination dimensions
/// * `nodata` - fill value for unmapped pixels
/// * `alg` - resampling algorithm
pub fn warp_resample(
    transformer: &impl Transformer,
    src_pixels: &[i32],
    src_ncol: usize,
    src_nrow: usize,
    src_col_off: usize,
    src_row_off: usize,
    dst_ncol: usize,
    dst_nrow: usize,
    nodata: i32,
    alg: ResampleAlg,
) -> Vec<i32> {
    match alg {
        ResampleAlg::NearestNeighbour => warp_nearest(
            transformer, src_pixels, src_ncol, src_nrow,
            src_col_off, src_row_off, dst_ncol, dst_nrow, nodata,
        ),
        ResampleAlg::Bilinear => warp_interpolated(
            transformer, src_pixels, src_ncol, src_nrow,
            src_col_off, src_row_off, dst_ncol, dst_nrow, nodata,
            bilinear_sample,
        ),
        ResampleAlg::Cubic => warp_interpolated(
            transformer, src_pixels, src_ncol, src_nrow,
            src_col_off, src_row_off, dst_ncol, dst_nrow, nodata,
            cubic_sample,
        ),
        ResampleAlg::Lanczos => warp_interpolated(
            transformer, src_pixels, src_ncol, src_nrow,
            src_col_off, src_row_off, dst_ncol, dst_nrow, nodata,
            lanczos_sample,
        ),
        // Area methods have one implementation, the generic one.
        ResampleAlg::Average | ResampleAlg::Mode => {
            let src = SourceBuffer {
                data: src_pixels, ncol: src_ncol, nrow: src_nrow, nbands: 1,
                col_off: src_col_off, row_off: src_row_off,
            };
            let opts = WarpOptions { alg, nodata: Some(nodata), ..Default::default() };
            warp(transformer, &src, dst_ncol, dst_nrow, &opts)
        }
    }
}

// =========================================================================
// Nearest neighbour
// =========================================================================

/// Nearest-neighbour warp (unchanged from original).
///
/// GDAL equivalent: GWKNearestThread, gdalwarpkernel.cpp ~L5510.
pub fn warp_nearest(
    transformer: &impl Transformer,
    src_pixels: &[i32],
    src_ncol: usize,
    src_nrow: usize,
    src_col_off: usize,
    src_row_off: usize,
    dst_ncol: usize,
    dst_nrow: usize,
    nodata: i32,
) -> Vec<i32> {
    let mut output = vec![nodata; dst_ncol * dst_nrow];

    for dst_row in 0..dst_nrow {
        let (src_x, src_y, ok) = transform_scanline(
            transformer, dst_row, dst_ncol, 0, 0,
        );

        for dst_col in 0..dst_ncol {
            if !ok[dst_col] { continue; }

            // GDAL GWKCheckAndComputeSrcOffsets: reject coordinates left of
            // or above the source before casting, so that (-1, 0) is not
            // truncated to pixel 0; then truncate with epsilon.
            if src_x[dst_col] < src_col_off as f64 || src_y[dst_col] < src_row_off as f64 {
                continue;
            }
            let buf_col = (src_x[dst_col] + 1.0e-10) as i64 - src_col_off as i64;
            let buf_row = (src_y[dst_col] + 1.0e-10) as i64 - src_row_off as i64;

            if buf_col < 0 || buf_col >= src_ncol as i64
                || buf_row < 0 || buf_row >= src_nrow as i64
            {
                continue;
            }

            let src_idx = buf_row as usize * src_ncol + buf_col as usize;
            let val = src_pixels[src_idx];
            if val != nodata {
                output[dst_row * dst_ncol + dst_col] = val;
            }
        }
    }

    output
}

// =========================================================================
// Generic interpolated warp (bilinear, cubic, lanczos)
// =========================================================================

/// Type for per-pixel resampling functions.
///
/// Arguments: (src_pixels, src_ncol, src_nrow, buf_x, buf_y, nodata) → Option<f64>
/// where buf_x/buf_y are fractional coordinates in the source buffer.
type SampleFn = fn(&[i32], usize, usize, f64, f64, i32) -> Option<f64>;

/// Shared scanline loop for interpolated resampling.
fn warp_interpolated(
    transformer: &impl Transformer,
    src_pixels: &[i32],
    src_ncol: usize,
    src_nrow: usize,
    src_col_off: usize,
    src_row_off: usize,
    dst_ncol: usize,
    dst_nrow: usize,
    nodata: i32,
    sample_fn: SampleFn,
) -> Vec<i32> {
    let mut output = vec![nodata; dst_ncol * dst_nrow];

    for dst_row in 0..dst_nrow {
        let (src_x, src_y, ok) = transform_scanline(
            transformer, dst_row, dst_ncol, 0, 0,
        );

        for dst_col in 0..dst_ncol {
            if !ok[dst_col] { continue; }

            // Convert from full-image coords to buffer-relative coords
            let buf_x = src_x[dst_col] - src_col_off as f64;
            let buf_y = src_y[dst_col] - src_row_off as f64;

            // Sample points outside [0, n) are off the source
            // (GWKCheckAndComputeSrcOffsets).
            let Some((cx, cy)) = centre_pixel(buf_x, buf_y, src_ncol, src_nrow) else { continue };

            // Centre on a nodata pixel: no value (GWKRealCaseThread ~L5680).
            if src_pixels[cy * src_ncol + cx] == nodata { continue; }

            if let Some(val) = sample_fn(src_pixels, src_ncol, src_nrow, buf_x, buf_y, nodata) {
                output[dst_row * dst_ncol + dst_col] = settle(gdal_round(val), Some(nodata));
            }
        }
    }

    output
}

/// GDAL's rounding into an integer working type: clamp, then round half up
/// (`ClampRoundAndAvoidNoData` / `GWKClampValueT`, gdalwarpkernel.cpp
/// ~L1587, ~L1856). The same as [`Sample::from_f64`] for `i32`.
#[inline]
fn gdal_round(v: f64) -> i32 {
    i32::from_f64(v)
}

// =========================================================================
// Bilinear (2×2)
// =========================================================================

/// Bilinear interpolation at (buf_x, buf_y) in source buffer.
///
/// GDAL equivalent: GWKBilinearResampleNoMasks4SampleT
/// (gdalwarpkernel.cpp ~L3084).
///
/// Weight: (1 - |dx|)(1 - |dy|) for the 4 surrounding pixels.
fn bilinear_sample(
    src: &[i32], ncol: usize, nrow: usize,
    buf_x: f64, buf_y: f64, nodata: i32,
) -> Option<f64> {
    // GDAL: iSrcX = floor(dfSrcX - 0.5)
    let ix = (buf_x - 0.5).floor() as i64;
    let iy = (buf_y - 0.5).floor() as i64;

    // GDAL: dfRatioX = 1.5 - (dfSrcX - iSrcX)
    let rx = 1.5 - (buf_x - ix as f64);
    let ry = 1.5 - (buf_y - iy as f64);

    // Fast path: all 4 pixels in bounds and no nodata
    if ix >= 0 && ix + 1 < ncol as i64 && iy >= 0 && iy + 1 < nrow as i64 {
        let off = iy as usize * ncol + ix as usize;

        if src[off] == nodata || src[off + 1] == nodata
            || src[off + ncol] == nodata || src[off + ncol + 1] == nodata
        {
            return None;
        }

        let val = (src[off] as f64 * rx + src[off + 1] as f64 * (1.0 - rx)) * ry
                + (src[off + ncol] as f64 * rx + src[off + ncol + 1] as f64 * (1.0 - rx))
                    * (1.0 - ry);
        return Some(val);
    }

    // Edge path: weight only valid in-bounds pixels
    let mut acc = 0.0;
    let mut wsum = 0.0;

    for &(cx, cy, w) in &[
        (ix,     iy,     rx * ry),
        (ix + 1, iy,     (1.0 - rx) * ry),
        (ix,     iy + 1, rx * (1.0 - ry)),
        (ix + 1, iy + 1, (1.0 - rx) * (1.0 - ry)),
    ] {
        if cx >= 0 && cx < ncol as i64 && cy >= 0 && cy < nrow as i64 {
            let v = src[cy as usize * ncol + cx as usize];
            if v != nodata {
                acc += v as f64 * w;
                wsum += w;
            }
        }
    }

    if wsum < 1e-5 { None } else { Some(acc / wsum) }
}

// =========================================================================
// Cubic (4×4, Catmull-Rom)
// =========================================================================

/// Cubic convolution kernel weight (a = -0.5, Catmull-Rom).
///
/// GDAL equivalent: CubicKernel in gdalresamplingkernels.h.
/// Mitchell-Netravali (B=0, C=0.5).
#[inline]
fn cubic_weight(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 1.0 {
        let x2 = x * x;
        x2 * (1.5 * ax - 2.5) + 1.0
    } else if ax <= 2.0 {
        let x2 = x * x;
        x2 * (-0.5 * ax + 2.5) - 4.0 * ax + 2.0
    } else {
        0.0
    }
}

/// Bicubic interpolation at (buf_x, buf_y).
///
/// GDAL equivalent: GWKCubicResample4Sample (gdalwarpkernel.cpp ~L3262).
/// Separable: 4 weights in X, 4 in Y, convolved over 4×4 neighbourhood.
/// Falls back to bilinear at edges (matches GDAL).
fn cubic_sample(
    src: &[i32], ncol: usize, nrow: usize,
    buf_x: f64, buf_y: f64, nodata: i32,
) -> Option<f64> {
    let ix = (buf_x - 0.5).floor() as i64;
    let iy = (buf_y - 0.5).floor() as i64;
    let dx = buf_x - 0.5 - ix as f64;
    let dy = buf_y - 0.5 - iy as f64;

    // Check full 4×4 neighbourhood: ix-1..ix+2, iy-1..iy+2
    if ix - 1 < 0 || ix + 2 >= ncol as i64 || iy - 1 < 0 || iy + 2 >= nrow as i64 {
        return bilinear_sample(src, ncol, nrow, buf_x, buf_y, nodata);
    }

    // Separable cubic weights
    let wx: [f64; 4] = [
        cubic_weight(dx + 1.0),
        cubic_weight(dx),
        cubic_weight(dx - 1.0),
        cubic_weight(dx - 2.0),
    ];
    let wy: [f64; 4] = [
        cubic_weight(dy + 1.0),
        cubic_weight(dy),
        cubic_weight(dy - 1.0),
        cubic_weight(dy - 2.0),
    ];

    // Check for nodata in the 4×4 neighbourhood
    for jj in 0..4i64 {
        let row_start = (iy - 1 + jj) as usize * ncol;
        for ii in 0..4i64 {
            if src[row_start + (ix - 1 + ii) as usize] == nodata {
                return bilinear_sample(src, ncol, nrow, buf_x, buf_y, nodata);
            }
        }
    }

    // Convolve: row by row
    let mut acc = 0.0;
    for jj in 0..4i64 {
        let row_start = (iy - 1 + jj) as usize * ncol;
        let mut row_acc = 0.0;
        for ii in 0..4i64 {
            row_acc += src[row_start + (ix - 1 + ii) as usize] as f64 * wx[ii as usize];
        }
        acc += row_acc * wy[jj as usize];
    }

    Some(acc)
}

// =========================================================================
// Lanczos (6×6, windowed sinc)
// =========================================================================

/// Lanczos windowed sinc weight (a = 3).
///
/// GDAL equivalent: GWKLanczosSinc (gdalwarpkernel.cpp ~L3655).
/// L(x) = sinc(x) · sinc(x/3) for |x| < 3, 0 otherwise.
///
/// Uses GDAL's sin(3x) identity:
/// sin(πx) = 3·sin(πx/3) - 4·sin³(πx/3)
#[inline]
fn lanczos_weight(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let ax = x.abs();
    if ax >= 3.0 {
        return 0.0;
    }

    let pi_x = std::f64::consts::PI * x;
    let pi_x_over_3 = pi_x / 3.0;
    let pi_x2_over_3 = pi_x * pi_x_over_3;

    let sin_r = pi_x_over_3.sin();
    let sin_r2 = sin_r * sin_r;

    // sin(πx)·sin(πx/3) via triple angle identity
    let product = (3.0 - 4.0 * sin_r2) * sin_r2;

    product / pi_x2_over_3
}

/// Lanczos interpolation at (buf_x, buf_y).
///
/// GDAL equivalent: GWKResampleOptimizedLanczos (gdalwarpkernel.cpp
/// ~L4188). Separable 6×6 kernel (radius 3) clipped to the buffer; nodata
/// pixels are skipped and the result renormalised by the weight of the
/// pixels used (no cubic fallback). None if that weight is below 1e-6.
fn lanczos_sample(
    src: &[i32], ncol: usize, nrow: usize,
    buf_x: f64, buf_y: f64, nodata: i32,
) -> Option<f64> {
    let ix = (buf_x - 0.5).floor() as i64;
    let iy = (buf_y - 0.5).floor() as i64;
    let dx = buf_x - 0.5 - ix as f64;
    let dy = buf_y - 0.5 - iy as f64;
    let (ncol_i, nrow_i) = (ncol as i64, nrow as i64);
    let (i0, i1) = ((-2i64).max(-ix), 3i64.min(ncol_i - 1 - ix));
    let (j0, j1) = ((-2i64).max(-iy), 3i64.min(nrow_i - 1 - iy));

    let mut acc = 0.0;
    let mut wsum = 0.0;
    for j in j0..=j1 {
        let wy = lanczos_weight(j as f64 - dy);
        let row_start = ((iy + j) * ncol_i + ix) as usize;
        let mut row_acc = 0.0;
        let mut row_w = 0.0;
        for i in i0..=i1 {
            let v = src[(row_start as i64 + i) as usize];
            if v == nodata { continue; }
            let wx = lanczos_weight(i as f64 - dx);
            row_acc += v as f64 * wx;
            row_w += wx;
        }
        acc += row_acc * wy;
        wsum += row_w * wy;
    }
    if wsum < 1e-6 { return None; }
    if !(0.99999..=1.00001).contains(&wsum) { acc /= wsum; }
    Some(acc)
}

// =========================================================================
// Generic interleaved kernel (any Sample type, multi-band, one pass)
// =========================================================================

/// A pixel sample type the generic kernel can resample.
///
/// The kernels compute in `f64` and convert back with [`Sample::from_f64`].
/// Integer types clamp to the type's range and then round half up,
/// `floor(v + 0.5)`, which is what GDAL does when the warp's working type
/// is an integer type (`ClampRoundAndAvoidNoData`, gdalwarpkernel.cpp
/// ~L1856): -2.5 becomes -2. Floats pass through and treat NaN as nodata.
///
/// rwarp has one type for source and destination. gdalwarp works in the
/// union of the source, destination and nodata types, so a floating source
/// warped into an integer output is resampled as float and rounded on write
/// by `GDALCopyWords`, half away from zero (-2.5 becomes -3); with rwarp
/// that conversion is the caller's.
pub trait Sample: Copy + PartialOrd + 'static {
    fn to_f64(self) -> f64;
    /// Convert a resampled value; see the trait documentation.
    fn from_f64(v: f64) -> Self;
    /// The nearest value next to `self`, used to move a resampled value off
    /// the nodata value: one less for integers (one more at the type's
    /// minimum), the next float up (down at the maximum). GDAL's
    /// `AvoidNoData`, gdalwarpkernel.cpp ~L1613.
    fn step_off(self) -> Self;
    /// The value written for unmapped / nodata destination pixels when no
    /// nodata value is given.
    fn zero() -> Self;
    #[inline]
    fn is_nan(self) -> bool { false }
}

macro_rules! int_sample {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            #[inline] fn from_f64(v: f64) -> Self {
                // Clamping after the floor gives the same result as GDAL's
                // clamp-then-round; NaN casts to 0.
                (v + 0.5).floor().clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t
            }
            #[inline] fn step_off(self) -> Self {
                if self == <$t>::MIN { <$t>::MIN + 1 } else { self - 1 }
            }
            #[inline] fn zero() -> Self { 0 }
        }
    )*};
}
int_sample!(u8, i8, u16, i16, u32, i32);

impl Sample for f32 {
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn step_off(self) -> Self { if self == f32::MAX { self.next_down() } else { self.next_up() } }
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
}
impl Sample for f64 {
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn step_off(self) -> Self { if self == f64::MAX { self.next_down() } else { self.next_up() } }
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
}

/// The source pixel under a sample point, or `None` if the point is off the
/// buffer: GDAL tests the coordinate against `[0, n)` before truncating, so
/// `(-1, 0)` is outside, not pixel 0 (`GWKCheckAndComputeSrcOffsets`,
/// gdalwarpkernel.cpp ~L5251).
#[inline]
fn centre_pixel(buf_x: f64, buf_y: f64, ncol: usize, nrow: usize) -> Option<(usize, usize)> {
    if buf_x < 0.0 || buf_y < 0.0 || buf_x + 1.0e-10 > ncol as f64 || buf_y + 1.0e-10 > nrow as f64 {
        return None;
    }
    let cx = ((buf_x + 1.0e-10) as usize).min(ncol - 1);
    let cy = ((buf_y + 1.0e-10) as usize).min(nrow - 1);
    Some((cx, cy))
}

/// A resampled value that lands on the nodata value is moved off it, so it
/// does not read back as missing (GDAL's `AvoidNoData`, which also warns).
#[inline]
fn settle<T: Sample>(v: T, nodata: Option<T>) -> T {
    match nodata {
        Some(nd) if v == nd => v.step_off(),
        _ => v,
    }
}

/// Warp a pixel-interleaved image of any [`Sample`] type in a single pass.
///
/// Each destination pixel is transformed once and the kernel weights are
/// computed once, then applied to every band from contiguous memory. With
/// [`KernelScaling::Point`] (or at scales where GDAL does not widen) the
/// weights, edge fallbacks (lanczos -> cubic -> bilinear) and rounding are
/// those of [`warp_resample`], so single-band integer output is identical
/// to it.
///
/// Validity is per pixel. A source pixel is nodata if any of:
/// - `opts.mask_band` is given and the pixel's value in that band is zero
///   (an alpha band);
/// - `opts.nodata` is given and any band of the pixel equals it;
/// - any band is NaN (floating types).
///
/// Destination pixels that are unmapped or land on nodata are written as
/// `opts.nodata` if given, else [`Sample::zero`]. With a mask band, the mask
/// band of the output is itself resampled like any other band, so an alpha
/// channel survives the warp.
///
/// Returns `dst_ncol * dst_nrow * src.nbands` values, interleaved.
pub fn warp<T: Sample>(
    transformer: &impl Transformer,
    src: &SourceBuffer<'_, T>,
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions<T>,
) -> Vec<T> {
    let nbands = src.nbands;
    assert_eq!(src.data.len(), src.ncol * src.nrow * nbands, "source buffer size mismatch");
    assert!(opts.mask_band.map_or(true, |a| a < nbands), "mask_band out of range");

    let img = Interleaved {
        src: src.data, ncol: src.ncol, nrow: src.nrow, nb: nbands,
        mask: opts.mask_band, nodata: opts.nodata,
    };
    let alg = opts.alg;
    let (sx, sy) = match opts.scaling {
        KernelScaling::Point => (1.0, 1.0),
        KernelScaling::WidenWhenDownsampling => {
            estimate_scale(transformer, dst_ncol, dst_nrow, opts.dst_col_off, opts.dst_row_off)
        }
        KernelScaling::Fixed { x, y } => (x, y),
    };
    if matches!(alg, ResampleAlg::Average | ResampleAlg::Mode) {
        return warp_area(transformer, &img, (src.col_off as f64, src.row_off as f64), dst_ncol, dst_nrow, opts, (sx, sy));
    }
    let widened = Widened::new(alg, sx, sy);

    let fill = opts.nodata.unwrap_or_else(T::zero);
    let mut out = vec![fill; dst_ncol * dst_nrow * nbands];
    let mut acc = vec![0.0f64; nbands];
    let mut row_acc = vec![0.0f64; nbands];
    let mut wx: Vec<f64> = Vec::new();

    for dst_row in 0..dst_nrow {
        let (src_x, src_y, ok) =
            transform_scanline(transformer, dst_row, dst_ncol, opts.dst_col_off, opts.dst_row_off);

        for dst_col in 0..dst_ncol {
            if !ok[dst_col] { continue; }
            let buf_x = src_x[dst_col] - src.col_off as f64;
            let buf_y = src_y[dst_col] - src.row_off as f64;
            let o = (dst_row * dst_ncol + dst_col) * nbands;

            if alg == ResampleAlg::NearestNeighbour {
                // Test before casting, as GDAL does, so (-1, 0) is not
                // truncated to pixel 0.
                if buf_x < 0.0 || buf_y < 0.0 { continue; }
                let bc = (buf_x + 1.0e-10) as i64;
                let br = (buf_y + 1.0e-10) as i64;
                if bc < 0 || bc >= src.ncol as i64 || br < 0 || br >= src.nrow as i64 { continue; }
                let p = br as usize * src.ncol + bc as usize;
                if !img.valid(p) { continue; }
                out[o..o + nbands].copy_from_slice(img.px(p));
                continue;
            }

            // Sample points outside [0, n) are off the source
            // (GWKCheckAndComputeSrcOffsets).
            let Some((cx, cy)) = centre_pixel(buf_x, buf_y, src.ncol, src.nrow) else { continue };

            // GDAL skips a destination pixel whose centre lands on an
            // invalid source pixel before any kernel runs
            // (GWKRealCaseThread, gdalwarpkernel.cpp ~L5680).
            if !img.valid(cy * src.ncol + cx) { continue; }

            let hit = match (widened, alg) {
                (Some(w), _) => img.widened(buf_x, buf_y, &w, &mut acc, &mut row_acc, &mut wx),
                (None, ResampleAlg::Bilinear) => img.bilinear(buf_x, buf_y, &mut acc),
                (None, ResampleAlg::Cubic) => img.cubic(buf_x, buf_y, &mut acc, &mut row_acc),
                (None, ResampleAlg::Lanczos) => {
                    img.widened(buf_x, buf_y, &LANCZOS_POINT, &mut acc, &mut row_acc, &mut wx)
                }
                (None, ResampleAlg::NearestNeighbour | ResampleAlg::Average | ResampleAlg::Mode) => unreachable!(),
            };
            if hit {
                for b in 0..nbands {
                    out[o + b] = settle(T::from_f64(acc[b]), opts.nodata);
                }
            }
        }
    }
    out
}

/// The 0.1 entry point: [`warp`] with point kernels
/// ([`KernelScaling::Point`]), as it behaved before widening.
#[deprecated(since = "0.2.0", note = "use `warp` with `WarpOptions`")]
#[allow(clippy::too_many_arguments)]
pub fn warp_resample_t<T: Sample>(
    transformer: &impl Transformer,
    src: &[T],
    src_ncol: usize,
    src_nrow: usize,
    nbands: usize,
    src_col_off: usize,
    src_row_off: usize,
    dst_ncol: usize,
    dst_nrow: usize,
    nodata: Option<T>,
    mask_band: Option<usize>,
    alg: ResampleAlg,
) -> Vec<T> {
    let buf = SourceBuffer { data: src, ncol: src_ncol, nrow: src_nrow, nbands, col_off: src_col_off, row_off: src_row_off };
    let opts = WarpOptions { alg, scaling: KernelScaling::Point, nodata, mask_band, ..Default::default() };
    warp(transformer, &buf, dst_ncol, dst_nrow, &opts)
}

/// The 0.1 interleaved `u8` entry point: [`warp`] with point kernels, no
/// nodata value and `alpha_band` as the mask band.
#[deprecated(since = "0.2.0", note = "use `warp` with `WarpOptions`")]
#[allow(clippy::too_many_arguments)]
pub fn warp_resample_u8(
    transformer: &impl Transformer,
    src: &[u8],
    src_ncol: usize,
    src_nrow: usize,
    nbands: usize,
    src_col_off: usize,
    src_row_off: usize,
    dst_ncol: usize,
    dst_nrow: usize,
    alpha_band: Option<usize>,
    alg: ResampleAlg,
) -> Vec<u8> {
    #[allow(deprecated)]
    warp_resample_t(
        transformer, src, src_ncol, src_nrow, nbands, src_col_off, src_row_off,
        dst_ncol, dst_nrow, None, alpha_band, alg,
    )
}

/// View over a pixel-interleaved buffer with per-pixel validity.
struct Interleaved<'a, T: Sample> {
    src: &'a [T],
    ncol: usize,
    nrow: usize,
    nb: usize,
    mask: Option<usize>,
    nodata: Option<T>,
}

impl<'a, T: Sample> Interleaved<'a, T> {
    #[inline]
    fn px(&self, p: usize) -> &'a [T] {
        &self.src[p * self.nb..(p + 1) * self.nb]
    }

    #[inline]
    fn valid_px(&self, pix: &[T]) -> bool {
        if let Some(a) = self.mask {
            if pix[a] == T::zero() { return false; }
        }
        match self.nodata {
            Some(nd) => pix.iter().all(|&v| v != nd && !v.is_nan()),
            None => pix.iter().all(|&v| !v.is_nan()),
        }
    }

    #[inline]
    fn valid(&self, p: usize) -> bool {
        self.valid_px(self.px(p))
    }

    /// Mirrors `bilinear_sample`. Returns false for nodata / no support.
    fn bilinear(&self, buf_x: f64, buf_y: f64, acc: &mut [f64]) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let rx = 1.5 - (buf_x - ix as f64);
        let ry = 1.5 - (buf_y - iy as f64);
        let (ncol, nrow) = (self.ncol as i64, self.nrow as i64);

        if ix >= 0 && ix + 1 < ncol && iy >= 0 && iy + 1 < nrow {
            let p00 = iy as usize * self.ncol + ix as usize;
            let (p10, p01, p11) = (p00 + 1, p00 + self.ncol, p00 + self.ncol + 1);
            if !(self.valid(p00) && self.valid(p10) && self.valid(p01) && self.valid(p11)) {
                return false;
            }
            let (a, b, c, d) = (self.px(p00), self.px(p10), self.px(p01), self.px(p11));
            for ((((o, &a), &b), &c), &d) in acc.iter_mut().zip(a).zip(b).zip(c).zip(d) {
                *o = (a.to_f64() * rx + b.to_f64() * (1.0 - rx)) * ry
                    + (c.to_f64() * rx + d.to_f64() * (1.0 - rx)) * (1.0 - ry);
            }
            return true;
        }

        for v in acc.iter_mut() { *v = 0.0; }
        let mut wsum = 0.0;
        for &(cx, cy, w) in &[
            (ix,     iy,     rx * ry),
            (ix + 1, iy,     (1.0 - rx) * ry),
            (ix,     iy + 1, rx * (1.0 - ry)),
            (ix + 1, iy + 1, (1.0 - rx) * (1.0 - ry)),
        ] {
            if cx >= 0 && cx < ncol && cy >= 0 && cy < nrow {
                let p = cy as usize * self.ncol + cx as usize;
                if self.valid(p) {
                    for (o, &v) in acc.iter_mut().zip(self.px(p)) { *o += v.to_f64() * w; }
                    wsum += w;
                }
            }
        }
        if wsum < 1e-5 { return false; }
        for v in acc.iter_mut() { *v /= wsum; }
        true
    }

    /// Mirrors `cubic_sample`, including its bilinear fallback.
    fn cubic(&self, buf_x: f64, buf_y: f64, acc: &mut [f64], row_acc: &mut [f64]) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let dx = buf_x - 0.5 - ix as f64;
        let dy = buf_y - 0.5 - iy as f64;

        if ix - 1 < 0 || ix + 2 >= self.ncol as i64 || iy - 1 < 0 || iy + 2 >= self.nrow as i64 {
            return self.bilinear(buf_x, buf_y, acc);
        }
        let wx = [cubic_weight(dx + 1.0), cubic_weight(dx), cubic_weight(dx - 1.0), cubic_weight(dx - 2.0)];
        let wy = [cubic_weight(dy + 1.0), cubic_weight(dy), cubic_weight(dy - 1.0), cubic_weight(dy - 2.0)];
        // Any nodata in the 4x4 window: fall back to bilinear (matches GDAL).
        self.convolve(ix - 1, iy - 1, &wx, &wy, acc, row_acc) || self.bilinear(buf_x, buf_y, acc)
    }

    /// Kernel widened by the downsampling scale (GDAL's `GWKResample`,
    /// gdalwarpkernel.cpp ~L4027, and `GWKResampleNoMasksT` ~L4840).
    ///
    /// The window is `1 - r ..= r` around the pixel below-left of the
    /// sample point, clipped to the buffer; weights are the kernel at
    /// `(i - delta) * scale`. Nodata pixels are skipped and the result is
    /// normalised by the weight of the pixels used; false if that is below
    /// 1e-6.
    fn widened(
        &self, buf_x: f64, buf_y: f64, w: &Widened,
        acc: &mut [f64], row_acc: &mut [f64], wx: &mut Vec<f64>,
    ) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let (ncol, nrow) = (self.ncol as i64, self.nrow as i64);

        // For bilinear and cubic GDAL declines a centre past the buffer, or
        // a support wider than the buffer, and uses the 4-sample bilinear
        // formula instead (GWKResampleNoMasksT).
        if w.alg != ResampleAlg::Lanczos && (ix >= ncol || iy >= nrow || w.rx > ncol || w.ry > nrow) {
            return self.bilinear(buf_x, buf_y, acc);
        }
        let dx = buf_x - 0.5 - ix as f64;
        let dy = buf_y - 0.5 - iy as f64;
        let (i0, i1) = ((1 - w.rx).max(-ix), w.rx.min(ncol - 1 - ix));
        let (j0, j1) = ((1 - w.ry).max(-iy), w.ry.min(nrow - 1 - iy));

        wx.clear();
        wx.extend((i0..=i1).map(|i| w.alg.weight((i as f64 - dx) * w.sx)));

        for v in acc.iter_mut() { *v = 0.0; }
        let mut wsum = 0.0;
        let nb = self.nb;
        for j in j0..=j1 {
            let wy = w.alg.weight((j as f64 - dy) * w.sy);
            for v in row_acc.iter_mut() { *v = 0.0; }
            let mut row_w = 0.0;
            let start = (((iy + j) * ncol + ix + i0) as usize) * nb;
            let run = &self.src[start..start + wx.len() * nb];
            for (pix, &wxi) in run.chunks_exact(nb).zip(wx.iter()) {
                if !self.valid_px(pix) { continue; }
                for (r, &v) in row_acc.iter_mut().zip(pix) { *r += v.to_f64() * wxi; }
                row_w += wxi;
            }
            for (o, &r) in acc.iter_mut().zip(row_acc.iter()) { *o += r * wy; }
            wsum += row_w * wy;
        }
        if wsum < 1e-6 { return false; }
        if !(0.99999..=1.00001).contains(&wsum) {
            for v in acc.iter_mut() { *v /= wsum; }
        }
        true
    }

    /// Separable convolution over an in-bounds window whose top-left source
    /// pixel is `(x0, y0)`. Returns false (with `acc` unspecified) as soon as
    /// a nodata pixel is met, so the caller can fall back. Row-by-row
    /// accumulation order matches the per-band kernels so results are
    /// bit-identical.
    #[inline]
    fn convolve(&self, x0: i64, y0: i64, wx: &[f64], wy: &[f64], acc: &mut [f64], row_acc: &mut [f64]) -> bool {
        for v in acc.iter_mut() { *v = 0.0; }
        let nb = self.nb;
        for (jj, &wyj) in wy.iter().enumerate() {
            for v in row_acc.iter_mut() { *v = 0.0; }
            let start = ((y0 as usize + jj) * self.ncol + x0 as usize) * nb;
            let run = &self.src[start..start + wx.len() * nb];
            for (pix, &wxi) in run.chunks_exact(nb).zip(wx) {
                if !self.valid_px(pix) { return false; }
                for (r, &v) in row_acc.iter_mut().zip(pix) { *r += v.to_f64() * wxi; }
            }
            for (o, &r) in acc.iter_mut().zip(row_acc.iter()) { *o += r * wyj; }
        }
        true
    }
}

// =========================================================================
// Area methods (average, mode)
// =========================================================================

/// What an area method does with the valid source pixels under a
/// destination pixel's footprint and their weights.
trait AreaReducer<T: Sample> {
    fn reset(&mut self);
    fn add(&mut self, pix: &[T], w: f64);
    /// Write the result for one destination pixel; false if there is none.
    fn write(&self, out: &mut [T]) -> bool;
}

/// Running weighted mean per band (West's weighted incremental algorithm,
/// as GDAL's average uses).
struct AreaMean {
    mean: Vec<f64>,
    total: f64,
}

impl<T: Sample> AreaReducer<T> for AreaMean {
    fn reset(&mut self) {
        self.mean.fill(0.0);
        self.total = 0.0;
    }

    #[inline]
    fn add(&mut self, pix: &[T], w: f64) {
        self.total += w;
        let f = w / self.total;
        for (m, &v) in self.mean.iter_mut().zip(pix) {
            *m += f * (v.to_f64() - *m);
        }
    }

    fn write(&self, out: &mut [T]) -> bool {
        if self.total <= 0.0 { return false; }
        for (o, &v) in out.iter_mut().zip(&self.mean) { *o = T::from_f64(v); }
        true
    }
}

/// Weighted mode per band. Weights accumulate in `f32` like GDAL's
/// `pafCounts`, so ties fall where GDAL's do. The mode is re-decided every
/// time a value's count grows, including its first occurrence, which is
/// GDAL's histogram path for 8- and 16-bit types (gdalwarpkernel.cpp
/// ~L8115-8170); see [`ModeTies`] for how wider types differ in GDAL.
struct AreaMode<T> {
    ties: ModeTies,
    /// Per band: distinct values and their counts, and the mode's index.
    bins: Vec<Vec<(T, f32)>>,
    mode: Vec<Option<usize>>,
}

impl<T: Sample> AreaReducer<T> for AreaMode<T> {
    fn reset(&mut self) {
        for b in &mut self.bins { b.clear(); }
        self.mode.fill(None);
    }

    fn add(&mut self, pix: &[T], w: f64) {
        let w = w as f32;
        for ((&v, bins), mode) in pix.iter().zip(&mut self.bins).zip(&mut self.mode) {
            let i = match bins.iter().position(|b| b.0 == v) {
                Some(i) => { bins[i].1 += w; i }
                None => { bins.push((v, w)); bins.len() - 1 }
            };
            let better = match *mode {
                None => true,
                Some(m) => {
                    let (c, cm) = (bins[i].1, bins[m].1);
                    c > cm || (c == cm && match self.ties {
                        ModeTies::First => false,
                        ModeTies::Min => v < bins[m].0,
                        ModeTies::Max => v > bins[m].0,
                    })
                }
            };
            if better { *mode = Some(i); }
        }
    }

    fn write(&self, out: &mut [T]) -> bool {
        if self.mode.iter().any(Option::is_none) { return false; }
        for ((o, bins), m) in out.iter_mut().zip(&self.bins).zip(&self.mode) {
            *o = bins[m.unwrap()].0;
        }
        true
    }
}

/// Source window of a destination pixel under GDAL's box footprint
/// (`GWKAverageOrModeComputeSourceCoords`, gdalwarpkernel.cpp ~L6919):
/// fractional bounds in buffer pixels and the integer range they touch.
#[derive(Debug, Clone, Copy)]
struct SrcBox {
    x0: f64, x1: f64, y0: f64, y1: f64,
    ix0: i64, ix1: i64, iy0: i64, iy1: i64,
    wrap: bool,
}

impl SrcBox {
    /// GDAL's per-pixel weight: the fraction of the source pixel inside the
    /// box along each axis (`COMPUTE_WEIGHT`, ~L6849). Only the first and
    /// last row/column are partial. A box that overhangs the buffer edge
    /// gives its edge pixel a weight above 1, as in GDAL.
    #[inline]
    fn weight(&self, ix: i64, iy: i64) -> f64 {
        let wy = if iy == self.iy0 {
            if self.iy0 + 1 == self.iy1 { 1.0 } else { 1.0 - (self.y0 - self.iy0 as f64) }
        } else if iy + 1 == self.iy1 {
            1.0 - (self.iy1 as f64 - self.y1)
        } else {
            1.0
        };
        if ix == self.ix0 {
            if self.ix0 + 1 == self.ix1 { wy } else { wy * (1.0 - (self.x0 - self.ix0 as f64)) }
        } else if ix + 1 == self.ix1 {
            wy * (1.0 - (self.ix1 as f64 - self.x1))
        } else {
            wy
        }
    }
}

/// Warp with an area method, row by row.
fn warp_area<T: Sample>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    src_off: (f64, f64),
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions<T>,
    scale: (f64, f64),
) -> Vec<T> {
    let nb = img.nb;
    match opts.alg {
        ResampleAlg::Mode => {
            let red = AreaMode { ties: opts.mode_ties, bins: vec![Vec::new(); nb], mode: vec![None; nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red)
        }
        _ => {
            let red = AreaMean { mean: vec![0.0; nb], total: 0.0 };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn warp_area_with<T: Sample, R: AreaReducer<T>>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    src_off: (f64, f64),
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions<T>,
    scale: (f64, f64),
    mut red: R,
) -> Vec<T> {
    let nb = img.nb;
    let fill = opts.nodata.unwrap_or_else(T::zero);
    let mut out = vec![fill; dst_ncol * dst_nrow * nb];
    for dst_row in 0..dst_nrow {
        let row_out = &mut out[dst_row * dst_ncol * nb..(dst_row + 1) * dst_ncol * nb];
        match opts.footprint {
            Footprint::GdalBox => area_row_gdal_box(transformer, img, src_off, dst_row, dst_ncol, opts, scale, &mut red, row_out),
            Footprint::Exact { densify } => area_row_exact(transformer, img, src_off, dst_row, dst_ncol, opts, densify.max(1), &mut red, row_out),
        }
    }
    out
}

/// One destination row under GDAL's box footprint (`GWKAverageOrModeThread`
/// and `GWKAverageOrModeComputeLineCoords`, gdalwarpkernel.cpp ~L6873-7030,
/// ~L7960 for the average itself).
#[allow(clippy::too_many_arguments)]
fn area_row_gdal_box<T: Sample, R: AreaReducer<T>>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    (col_off, row_off): (f64, f64),
    dst_row: usize,
    dst_ncol: usize,
    opts: &WarpOptions<T>,
    (sx, sy): (f64, f64),
    red: &mut R,
    row_out: &mut [T],
) {
    let (xoff, yoff) = (opts.dst_col_off as f64, (dst_row + opts.dst_row_off) as f64);
    // Top-left and bottom-right corners of every pixel in the row.
    let mut x1: Vec<f64> = (0..dst_ncol).map(|c| c as f64 + xoff).collect();
    let mut y1 = vec![yoff; dst_ncol];
    let mut x2: Vec<f64> = (0..dst_ncol).map(|c| c as f64 + 1.0 + xoff).collect();
    let mut y2 = vec![yoff + 1.0; dst_ncol];
    let ok1 = transformer.transform(true, &mut x1, &mut y1);
    let ok2 = transformer.transform(true, &mut x2, &mut y2);

    let (ncol, nrow) = (img.ncol as i64, img.nrow as i64);
    // Corners well outside the buffer are rejected (GDAL issue #2365).
    let xmargin = 2.0 * 1f64.max((1.0 / sx).ceil());
    let ymargin = 2.0 * 1f64.max((1.0 / sy).ceil());
    const EPSILON: f64 = 1e-10;

    for c in 0..dst_ncol {
        if !(ok1[c] && ok2[c]) { continue; }
        let in_range = |v: f64, off: f64, n: f64, m: f64| v - off >= -m && v - off - n <= m;
        if !(in_range(x1[c], col_off, ncol as f64, xmargin) && in_range(x2[c], col_off, ncol as f64, xmargin)
            && in_range(y1[c], row_off, nrow as f64, ymargin) && in_range(y2[c], row_off, nrow as f64, ymargin))
        {
            continue;
        }
        // The transform need not preserve ordering (GDAL #5433).
        if x1[c] > x2[c] { std::mem::swap(&mut x1[c], &mut x2[c]); }

        // A target pixel straddling the antimeridian maps to the two ends
        // of the source row; GDAL detects it from the corners and a probe at
        // the top-edge midpoint (#6478), and wraps the box past the right
        // edge.
        let mut wrap = false;
        let threshold = 2i64.min(ncol / 10) as f64;
        if col_off == 0.0 && c + 1 < dst_ncol
            && 2.0 * x1[c] - x1[c + 1] < threshold
            && ncol as f64 - x2[c] < threshold
        {
            let (mut px, mut py) = ([c as f64 + 0.5 + xoff], [yoff]);
            let pok = transformer.transform(true, &mut px, &mut py);
            if pok[0] && px[0] < x1[c] {
                wrap = true;
                std::mem::swap(&mut x1[c], &mut x2[c]);
                x2[c] += ncol as f64;
            }
        }

        let (bx0, bx1) = (x1[c] - col_off, x2[c] - col_off);
        if !(bx1 > -EPSILON && bx0 < ncol as f64 + EPSILON) { continue; }
        let ix0 = (bx0 + EPSILON).floor().max(0.0) as i64;
        let mut ix1 = (bx1 - EPSILON).ceil().min(i32::MAX as f64) as i64;
        if !wrap { ix1 = ix1.min(ncol); }
        if ix0 == ix1 && ix1 < ncol { ix1 += 1; }

        if y1[c] > y2[c] { std::mem::swap(&mut y1[c], &mut y2[c]); }
        let (by0, by1) = (y1[c] - row_off, y2[c] - row_off);
        if !(by1 > -EPSILON && by0 < nrow as f64 + EPSILON) { continue; }
        let iy0 = (by0 + EPSILON).floor().max(0.0) as i64;
        let mut iy1 = ((by1 - EPSILON).ceil() as i64).min(nrow);
        if iy0 == iy1 && iy1 < nrow { iy1 += 1; }

        let b = SrcBox { x0: bx0, x1: bx1, y0: by0, y1: by1, ix0, ix1, iy0, iy1, wrap };
        red.reset();
        for iy in b.iy0..b.iy1 {
            for ix in b.ix0..b.ix1 {
                let col = if b.wrap { ix % ncol } else { ix };
                let p = (iy * ncol + col) as usize;
                if !img.valid(p) { continue; }
                let w = b.weight(ix, iy);
                if w > 0.0 { red.add(img.px(p), w); }
            }
        }
        let px = &mut row_out[c * img.nb..(c + 1) * img.nb];
        if red.write(px) {
            for v in px.iter_mut() { *v = settle(*v, opts.nodata); }
        }
    }
}

/// One destination row under the exact footprint: each pixel's outline,
/// densified and transformed, burned onto the source pixels it touches
/// with exact coverage fractions.
#[allow(clippy::too_many_arguments)]
fn area_row_exact<T: Sample, R: AreaReducer<T>>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    (col_off, row_off): (f64, f64),
    dst_row: usize,
    dst_ncol: usize,
    opts: &WarpOptions<T>,
    densify: usize,
    red: &mut R,
    row_out: &mut [T],
) {
    use controlledburn::{burn, BurnOptions, Coord, Geometry, GridSpec, Polygon};

    let (xoff, yoff) = (opts.dst_col_off as f64, (dst_row + opts.dst_row_off) as f64);
    let per = 4 * densify;
    let mut xs = Vec::with_capacity(per * dst_ncol);
    let mut ys = Vec::with_capacity(per * dst_ncol);
    for c in 0..dst_ncol {
        let (x0, y0) = (c as f64 + xoff, yoff);
        for k in 0..densify {
            let t = k as f64 / densify as f64;
            xs.push(x0 + t); ys.push(y0);             // top, left to right
        }
        for k in 0..densify {
            let t = k as f64 / densify as f64;
            xs.push(x0 + 1.0); ys.push(y0 + t);       // right, downwards
        }
        for k in 0..densify {
            let t = k as f64 / densify as f64;
            xs.push(x0 + 1.0 - t); ys.push(y0 + 1.0); // bottom, right to left
        }
        for k in 0..densify {
            let t = k as f64 / densify as f64;
            xs.push(x0); ys.push(y0 + 1.0 - t);       // left, upwards
        }
    }
    let ok = transformer.transform(true, &mut xs, &mut ys);

    let (ncol, nrow) = (img.ncol as i64, img.nrow as i64);
    for c in 0..dst_ncol {
        let r = c * per..(c + 1) * per;
        if !ok[r.clone()].iter().all(|&v| v) { continue; }
        // Source pixel space, y flipped so that row 0 is at the top of the
        // grid as controlledburn expects.
        let mut ring: Vec<Coord> = xs[r.clone()].iter().zip(&ys[r])
            .map(|(&x, &y)| Coord::new(x - col_off, -(y - row_off)))
            .collect();
        if !ring.iter().all(|p| p.is_finite()) { continue; }
        let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for p in &ring {
            minx = minx.min(p.x); maxx = maxx.max(p.x);
            miny = miny.min(-p.y); maxy = maxy.max(-p.y);
        }
        // Burn onto just the buffer pixels under the footprint's bbox.
        // (Burning a whole row in one call measured no faster.)
        let c0 = (minx.floor() as i64).max(0);
        let c1 = (maxx.ceil() as i64).min(ncol);
        let r0 = (miny.floor() as i64).max(0);
        let r1 = (maxy.ceil() as i64).min(nrow);
        if c0 >= c1 || r0 >= r1 { continue; }
        ring.push(ring[0]);
        let grid = GridSpec::new(c0 as f64, -(r1 as f64), c1 as f64, -(r0 as f64), (c1 - c0) as u32, (r1 - r0) as u32);
        let Ok(res) = burn(&[Geometry::Polygon(Polygon::new(vec![ring]))], &grid, BurnOptions::coverage()) else { continue };

        red.reset();
        let at = |row: i32, col: i64| ((r0 + row as i64) * ncol + c0 + col) as usize;
        for run in &res.runs {
            for col in run.col_start..run.col_end {
                let p = at(run.row, col as i64);
                if img.valid(p) { red.add(img.px(p), 1.0); }
            }
        }
        for e in &res.edges {
            let p = at(e.row, e.col as i64);
            if img.valid(p) && e.fraction > 0.0 { red.add(img.px(p), e.fraction as f64); }
        }
        let px = &mut row_out[c * img.nb..(c + 1) * img.nb];
        if red.write(px) {
            for v in px.iter_mut() { *v = settle(*v, opts.nodata); }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_weight_properties() {
        assert!((cubic_weight(0.0) - 1.0).abs() < 1e-12);
        assert!(cubic_weight(1.0).abs() < 1e-12);
        assert!(cubic_weight(2.0).abs() < 1e-12);
        assert!((cubic_weight(0.5) - cubic_weight(-0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_lanczos_weight_properties() {
        assert!((lanczos_weight(0.0) - 1.0).abs() < 1e-12);
        assert!(lanczos_weight(1.0).abs() < 1e-10);
        assert!(lanczos_weight(2.0).abs() < 1e-10);
        assert_eq!(lanczos_weight(3.0), 0.0);
        assert!((lanczos_weight(0.7) - lanczos_weight(-0.7)).abs() < 1e-12);
    }

    #[test]
    fn test_bilinear_on_flat_field() {
        let src = vec![42i32; 10 * 10];
        let val = bilinear_sample(&src, 10, 10, 5.3, 5.7, -9999);
        assert!((val.unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_at_pixel_centre() {
        let mut src = vec![0i32; 8 * 8];
        src[3 * 8 + 3] = 100;
        let val = bilinear_sample(&src, 8, 8, 3.5, 3.5, -9999);
        assert!((val.unwrap() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_on_flat_field() {
        let src = vec![100i32; 10 * 10];
        let val = cubic_sample(&src, 10, 10, 5.3, 5.7, -9999);
        assert!((val.unwrap() - 100.0).abs() < 1e-8);
    }

    #[test]
    fn test_lanczos_on_flat_field() {
        let src = vec![77i32; 12 * 12];
        let val = lanczos_sample(&src, 12, 12, 6.3, 6.7, -9999);
        assert!((val.unwrap() - 77.0).abs() < 1e-8);
    }

    // ---- interleaved kernel -------------------------------------------

    /// [`warp`] with point kernels, the configuration the per-band
    /// reference kernel implements.
    #[allow(clippy::too_many_arguments)]
    fn point<T: Sample>(
        t: &impl Transformer, src: &[T], w: usize, h: usize, nbands: usize,
        dw: usize, dh: usize, nodata: Option<T>, mask_band: Option<usize>, alg: ResampleAlg,
    ) -> Vec<T> {
        let buf = SourceBuffer { data: src, ncol: w, nrow: h, nbands, col_off: 0, row_off: 0 };
        let opts = WarpOptions { alg, scaling: KernelScaling::Point, nodata, mask_band, ..Default::default() };
        warp(t, &buf, dw, dh, &opts)
    }

    /// Identity-ish transformer with a fractional offset and slight scale so
    /// every kernel path (including edges) is exercised.
    struct Shift;
    impl Transformer for Shift {
        fn transform(&self, _d2s: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            for i in 0..x.len() {
                x[i] = x[i] * 1.03 + 0.37;
                y[i] = y[i] * 0.97 + 0.61;
            }
            vec![true; x.len()]
        }
    }

    fn synthetic_rgba(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for r in 0..h {
            for c in 0..w {
                let i = (r * w + c) * 4;
                v[i] = ((c * 7 + r * 3) % 256) as u8;
                v[i + 1] = ((c * c + r) % 256) as u8;
                v[i + 2] = ((r * 11) % 256) as u8;
                v[i + 3] = 255;
            }
        }
        v
    }

    #[test]
    fn interleaved_matches_per_band_for_all_kernels() {
        let (w, h) = (40, 33);
        let src = synthetic_rgba(w, h);
        let (dw, dh) = (37, 30);
        for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
            let fast = point(&Shift, &src, w, h, 4, dw, dh, None, Some(3), alg);
            for b in 0..4 {
                let band: Vec<i32> = (0..w * h).map(|i| src[i * 4 + b] as i32).collect();
                let slow = warp_resample(&Shift, &band, w, h, 0, 0, dw, dh, -1, alg);
                for i in 0..dw * dh {
                    let expect = if slow[i] == -1 { 0 } else { slow[i].clamp(0, 255) as u8 };
                    assert_eq!(fast[i * 4 + b], expect, "{alg:?} band {b} pixel {i}");
                }
            }
        }
    }

    #[test]
    fn interleaved_alpha_zero_is_nodata() {
        let (w, h) = (20, 20);
        let mut src = synthetic_rgba(w, h);
        // Punch a transparent hole in the middle.
        for r in 8..12 { for c in 8..12 { src[(r * w + c) * 4 + 3] = 0; } }
        struct Id;
        impl Transformer for Id {
            fn transform(&self, _: bool, x: &mut [f64], _y: &mut [f64]) -> Vec<bool> { vec![true; x.len()] }
        }
        for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
            let out = point(&Id, &src, w, h, 4, w, h, None, Some(3), alg);
            // Hole is transparent...
            assert_eq!(out[(10 * w + 10) * 4 + 3], 0, "{alg:?}");
            // ...and a pixel well away from it is untouched under identity.
            let p = (3 * w + 3) * 4;
            assert_eq!(&out[p..p + 4], &src[p..p + 4], "{alg:?}");
        }
    }

    #[test]
    fn interleaved_without_alpha_treats_all_valid() {
        let (w, h) = (10, 10);
        let src: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        struct Id;
        impl Transformer for Id {
            fn transform(&self, _: bool, x: &mut [f64], _y: &mut [f64]) -> Vec<bool> { vec![true; x.len()] }
        }
        let out = point(&Id, &src, w, h, 3, w, h, None, None, ResampleAlg::Bilinear);
        assert_eq!(out, src);
    }

    // ---- generic kernel -----------------------------------------------

    #[test]
    fn generic_i32_single_band_matches_reference_kernel() {
        let (w, h) = (40, 33);
        let mut band: Vec<i32> = (0..w * h).map(|i| ((i * 7919) % 1000) as i32 - 500).collect();
        // Sprinkle nodata.
        for i in (0..w * h).step_by(37) { band[i] = -9999; }
        for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
            let reference = warp_resample(&Shift, &band, w, h, 0, 0, 37, 30, -9999, alg);
            let generic = point(&Shift, &band, w, h, 1, 37, 30, Some(-9999), None, alg);
            assert_eq!(reference, generic, "{alg:?}");
        }
    }

    #[test]
    fn generic_f32_matches_i32_on_integer_data_and_keeps_fractions() {
        let (w, h) = (24, 20);
        let band: Vec<i32> = (0..w * h).map(|i| ((i * 31) % 200) as i32).collect();
        let bandf: Vec<f32> = band.iter().map(|&v| v as f32).collect();
        for alg in [ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
            let vi = point(&Shift, &band, w, h, 1, 20, 18, Some(-1), None, alg);
            let vf = point(&Shift, &bandf, w, h, 1, 20, 18, Some(-1.0), None, alg);
            let mut fractional = 0;
            for i in 0..vi.len() {
                if vi[i] == -1 { assert_eq!(vf[i], -1.0); continue; }
                // The integer path is the rounded float path; f32 storage can
                // land exactly on .5 where f64 was a hair below, so allow half.
                assert!((vf[i] as f64 - vi[i] as f64).abs() <= 0.5 + 1e-3, "{alg:?} pixel {i}: {} vs {}", vf[i], vi[i]);
                if vf[i].fract() != 0.0 { fractional += 1; }
            }
            assert!(fractional > 0, "{alg:?}: float output never carried a fraction");
        }
    }

    #[test]
    fn generic_nan_is_nodata_and_fill_is_nodata() {
        let (w, h) = (16, 16);
        let mut src: Vec<f32> = vec![10.0; w * h];
        for r in 6..10 { for c in 6..10 { src[r * w + c] = f32::NAN; } }
        struct Id;
        impl Transformer for Id {
            fn transform(&self, _: bool, x: &mut [f64], _y: &mut [f64]) -> Vec<bool> { vec![true; x.len()] }
        }
        for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic] {
            let out = point(&Id, &src, w, h, 1, w, h, Some(-9999.0), None, alg);
            assert_eq!(out[8 * w + 8], -9999.0, "{alg:?}: hole should be filled with nodata");
            assert_eq!(out[2 * w + 2], 10.0, "{alg:?}: flat field preserved");
            assert!(out.iter().all(|v| !v.is_nan()), "{alg:?}: NaN leaked into output");
        }
        // Without a nodata value the fill is zero.
        let out = point(&Id, &src, w, h, 1, w, h, None, None, ResampleAlg::Bilinear);
        assert_eq!(out[8 * w + 8], 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_entry_points_are_point_warps() {
        let (w, h) = (24, 20);
        let src = synthetic_rgba(w, h);
        for alg in [ResampleAlg::Bilinear, ResampleAlg::Lanczos] {
            let old = warp_resample_u8(&Shift, &src, w, h, 4, 0, 0, 10, 9, Some(3), alg);
            assert_eq!(old, point(&Shift, &src, w, h, 4, 10, 9, None, Some(3), alg), "{alg:?}");
        }
    }

    #[test]
    fn generic_u16_saturates_and_rounds() {
        assert_eq!(u16::from_f64(65535.7), 65535);
        assert_eq!(u16::from_f64(-3.0), 0);
        assert_eq!(u16::from_f64(2.5), 3);
        assert_eq!(i16::from_f64(-2.5), -2);
        assert_eq!(i16::from_f64(-2.6), -3);
        assert_eq!(i8::from_f64(-200.0), -128);
        assert_eq!(i32::from_f64(f64::NAN), 0);
        assert_eq!(i32::from_f64(-0.4), 0);
        assert_eq!(u8::from_f64(255.49), 255);
    }
}
