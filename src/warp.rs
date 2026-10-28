//! Warp kernels: nearest-neighbour, bilinear, cubic, cubic spline, lanczos,
//! and the area methods average, mode, min, max, median, quartiles and RMS.
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
//! agree. The exact footprint transforms the destination pixel lattice
//! once per row band (shared corners and edge points, each row line and
//! each densify level a scanline, so an approximate transformer applies),
//! and takes coverage from a reused [`controlledburn::CoverageWindow`]. It
//! costs about 3x the box at 3.5:1 in the timing test (0.8 microseconds a
//! pixel), less at higher ratios.
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
//! ## Min, max, median, quartiles, RMS
//!
//! [`ResampleAlg::Min`], [`Max`](ResampleAlg::Max), [`Med`](ResampleAlg::Med),
//! [`Q1`](ResampleAlg::Q1) and [`Q3`](ResampleAlg::Q3) take every valid
//! source pixel the footprint touches, unweighted, as GDAL does; the
//! quantiles are element `ceil(q n - 1)` of the sorted values
//! (gdalwarpkernel.cpp ~L8174-8340). They return source values unchanged.
//! [`ResampleAlg::Rms`] is `sqrt(sum(w v^2) / sum(w))` with
//! [`ResampleAlg::Average`]'s weights (~L8025).
//!
//! ## Cubic spline
//!
//! [`ResampleAlg::CubicSpline`] is the cubic B-spline (4×4) through the
//! general kernel at every scale, widening below scale 1 like lanczos; GDAL
//! has no 4-sample formula for it. GDAL's weight omits the term that would
//! make it zero beyond |x| = 2, so widened windows that reach there carry
//! small negative weights; rwarp matches that.
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
//! ## Nodata handling
//!
//! Validity is per band; [`BandNodata`] (GDAL's `UNIFIED_SRC_NODATA`)
//! decides how bands combine, and a zero in the mask band invalidates the
//! whole pixel. Per band, following gdalwarpkernel.cpp:
//!
//! **Nearest-neighbour**: the band of the source pixel is copied if valid;
//! otherwise the destination band keeps its fill value.
//!
//! **Bilinear** (point): invalid and out-of-buffer neighbours are dropped
//! and the rest renormalised; less than 1e-5 of weight left → no value
//! (`GWKBilinearResample4Sample`).
//!
//! **Cubic** (point): near the buffer edge, or with any invalid pixel in the
//! 4×4 window, falls back to bilinear for that band
//! (`GWKCubicResample4Sample`).
//!
//! **Lanczos, and widened kernels**: the window is clipped to the buffer and
//! invalid pixels skipped, renormalising by the weight used
//! (`GWKResampleOptimizedLanczos`, `GWKResample`).
//!
//! **All interpolating kernels**: a destination pixel whose centre lands on
//! a source pixel that is invalid as a whole (every band nodata, under the
//! default rule) gets no value, whatever its neighbours.
//!
//! **Average, mode**: invalid pixels add neither value nor weight to that
//! band; with [`AverageThresholds`] in use, average treats the bands
//! together.

use crate::transform::{transform_scanline, Transformer};

/// Resampling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Cubic B-spline (4×4, radius 2; B = 1, C = 0). Smooths rather than
    /// interpolates: the output at a source pixel centre is not that
    /// pixel's value.
    CubicSpline,
    /// Smallest valid source value under the footprint, per band.
    Min,
    /// Largest valid source value under the footprint, per band.
    Max,
    /// Median of the valid source values under the footprint, per band.
    Med,
    /// First quartile of the valid source values under the footprint.
    Q1,
    /// Third quartile of the valid source values under the footprint.
    Q3,
    /// Root mean square, weighted by coverage like [`ResampleAlg::Average`].
    Rms,
}

impl ResampleAlg {
    /// Whether the method reduces the source pixels under each destination
    /// pixel's footprint (see [`Footprint`]) rather than sampling at its
    /// centre.
    pub fn is_area(self) -> bool {
        matches!(
            self,
            ResampleAlg::Average | ResampleAlg::Mode | ResampleAlg::Min | ResampleAlg::Max
                | ResampleAlg::Med | ResampleAlg::Q1 | ResampleAlg::Q3 | ResampleAlg::Rms
        )
    }
}

impl ResampleAlg {
    /// Filter radius in pixels (matches GWKGetFilterRadius).
    pub fn radius(self) -> i32 {
        match self {
            ResampleAlg::NearestNeighbour => 0,
            ResampleAlg::Bilinear => 1,
            ResampleAlg::Cubic => 2,
            ResampleAlg::Lanczos => 3,
            ResampleAlg::CubicSpline => 2,
            _ => 0,
        }
    }

    /// Kernel weight at distance `x` (GDAL's `apfGWKFilter`). Zero for
    /// nearest, which has no kernel.
    fn weight(self, x: f64) -> f64 {
        match self {
            ResampleAlg::Bilinear => {
                let ax = x.abs();
                if ax <= 1.0 { 1.0 - ax } else { 0.0 }
            }
            ResampleAlg::Cubic => cubic_weight(x),
            ResampleAlg::Lanczos => lanczos_weight(x),
            ResampleAlg::CubicSpline => bspline_weight(x),
            _ => 0.0,
        }
    }
}

/// How the interpolating kernels respond to downsampling. See the module
/// documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// with the source pixels (coverage fractions from
    /// [`controlledburn::CoverageWindow`], rounded to `f32`). With
    /// renormalisation over valid pixels this is first-order conservative
    /// remapping. No antimeridian handling.
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

/// How nodata works across bands: GDAL's `UNIFIED_SRC_NODATA` warp option
/// (gdalwarper.cpp ~L1331, gdalwarpoperation.cpp ~L2262). A source pixel is
/// "nodata in a band" when that band equals [`WarpOptions::src_nodata`] or is
/// NaN. With one band, all four are the same. A zero in
/// [`WarpOptions::mask_band`] makes the whole pixel invalid under every rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BandNodata {
    /// GDAL's default (option unset, or `PARTIAL`): a source pixel is
    /// ignored only where every band is nodata, and each band is otherwise
    /// resampled from its own valid pixels. A destination pixel whose
    /// centre lands where every band is nodata gets no value.
    #[default]
    Partial,
    /// `UNIFIED_SRC_NODATA=YES`, which gdalwarp sets when `-srcnodata` is
    /// given explicitly: a source pixel is nodata only if every band is;
    /// otherwise all its band values are used as they are, including a band
    /// at its nodata value.
    Unified,
    /// `UNIFIED_SRC_NODATA=NO`: each band on its own; no whole-pixel test.
    PerBand,
    /// rwarp 0.1: a source pixel is nodata if any band is. Not a GDAL mode.
    AnyBand,
}

/// GDAL's thresholds for [`ResampleAlg::Average`] (warp options
/// `NODATA_VALUES_PCT_THRESHOLD`, `EXCLUDED_VALUES`,
/// `EXCLUDED_VALUES_PCT_THRESHOLD`; gdalwarpkernel.cpp ~L7760-7930). The
/// defaults (100, none, 50) leave average unchanged.
///
/// When either is in use, each footprint's weight is split into invalid
/// (any band nodata), excluded (all bands equal one of
/// `excluded_values`) and regular. If the invalid share is at least
/// `nodata_pct_threshold` percent the destination pixel gets no value;
/// otherwise if the excluded share is at least
/// `excluded_pct_threshold` percent it gets the excluded tuple seen most
/// often (by pixel count, the first listed on a tie); otherwise the mean of
/// the regular pixels.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AverageThresholds {
    pub nodata_pct_threshold: f64,
    /// Tuples of one value per band.
    pub excluded_values: Vec<Vec<f64>>,
    pub excluded_pct_threshold: f64,
}

impl Default for AverageThresholds {
    fn default() -> Self {
        AverageThresholds { nodata_pct_threshold: 100.0, excluded_values: Vec::new(), excluded_pct_threshold: 50.0 }
    }
}

impl AverageThresholds {
    /// `NODATA_VALUES_PCT_THRESHOLD`: percent.
    pub fn with_nodata_pct_threshold(mut self, pct: f64) -> Self {
        self.nodata_pct_threshold = pct;
        self
    }

    /// `EXCLUDED_VALUES` and `EXCLUDED_VALUES_PCT_THRESHOLD` (percent).
    pub fn with_excluded(mut self, values: Vec<Vec<f64>>, pct: f64) -> Self {
        self.excluded_values = values;
        self.excluded_pct_threshold = pct;
        self
    }

    fn active(&self) -> bool {
        !self.excluded_values.is_empty() || self.nodata_pct_threshold / 100.0 < 1.0 - 1e-10
    }
}

/// Source nodata: none, one value for every band, or one per band (GDAL
/// `-srcnodata`). Values are `f64`, as in GDAL; a band matches when its
/// value equals the band's nodata, and NaN is always nodata. If no band's
/// value is representable in the source type, all are ignored, as GDAL
/// does.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum Nodata {
    #[default]
    None,
    Value(f64),
    PerBand(Vec<f64>),
}

impl Nodata {
    fn per_band(&self, nb: usize) -> Vec<Option<f64>> {
        match self {
            Nodata::None => vec![None; nb],
            Nodata::Value(v) => vec![Some(*v); nb],
            Nodata::PerBand(v) => {
                assert_eq!(v.len(), nb, "one nodata value per band");
                v.iter().map(|&x| Some(x)).collect()
            }
        }
    }
}

/// Destination nodata (GDAL `-dstnodata`): the fill for destination pixels
/// with no value, and the value resampled results are moved off.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum DstNodata {
    /// The source nodata, as gdalwarp does when `-dstnodata` is not given.
    #[default]
    FromSource,
    /// No destination nodata: fill with zero.
    None,
    Value(f64),
    PerBand(Vec<f64>),
}

/// Options for [`warp`] and [`warp_to`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WarpOptions {
    pub alg: ResampleAlg,
    pub scaling: KernelScaling,
    /// Footprint for [`ResampleAlg::Average`] and [`ResampleAlg::Mode`];
    /// ignored by the other methods.
    pub footprint: Footprint,
    /// Tie-breaking for [`ResampleAlg::Mode`]; ignored by the other
    /// methods.
    pub mode_ties: ModeTies,
    /// Thresholds for [`ResampleAlg::Average`]; ignored by the other
    /// methods.
    pub average: AverageThresholds,
    /// How nodata works across bands.
    pub band_nodata: BandNodata,
    /// Source nodata; see [`Nodata`] and [`BandNodata`].
    pub src_nodata: Nodata,
    /// Destination nodata; see [`DstNodata`].
    pub dst_nodata: DstNodata,
    /// Band whose zero values mark source pixels as nodata (an alpha band).
    pub mask_band: Option<usize>,
    /// Offset of the destination window in the grid the transformer maps
    /// from, so a tile of a larger destination can be warped on its own.
    pub dst_col_off: usize,
    pub dst_row_off: usize,
}

impl Default for WarpOptions {
    fn default() -> Self {
        WarpOptions {
            alg: ResampleAlg::NearestNeighbour,
            scaling: KernelScaling::WidenWhenDownsampling,
            footprint: Footprint::GdalBox,
            mode_ties: ModeTies::First,
            average: AverageThresholds::default(),
            band_nodata: BandNodata::Partial,
            src_nodata: Nodata::None,
            dst_nodata: DstNodata::FromSource,
            mask_band: None,
            dst_col_off: 0,
            dst_row_off: 0,
        }
    }
}

impl WarpOptions {
    /// Default options with the given resampling method. Chain the `with_`
    /// methods to set others; `WarpOptions` is `#[non_exhaustive]`, so it
    /// can't be built with a struct literal outside this crate.
    pub fn new(alg: ResampleAlg) -> Self {
        WarpOptions { alg, ..Default::default() }
    }

    pub fn with_alg(mut self, alg: ResampleAlg) -> Self {
        self.alg = alg;
        self
    }

    pub fn with_scaling(mut self, scaling: KernelScaling) -> Self {
        self.scaling = scaling;
        self
    }

    pub fn with_footprint(mut self, footprint: Footprint) -> Self {
        self.footprint = footprint;
        self
    }

    pub fn with_mode_ties(mut self, mode_ties: ModeTies) -> Self {
        self.mode_ties = mode_ties;
        self
    }

    pub fn with_average(mut self, average: AverageThresholds) -> Self {
        self.average = average;
        self
    }

    pub fn with_band_nodata(mut self, band_nodata: BandNodata) -> Self {
        self.band_nodata = band_nodata;
        self
    }

    pub fn with_src_nodata(mut self, src_nodata: Nodata) -> Self {
        self.src_nodata = src_nodata;
        self
    }

    pub fn with_dst_nodata(mut self, dst_nodata: DstNodata) -> Self {
        self.dst_nodata = dst_nodata;
        self
    }

    pub fn with_mask_band(mut self, mask_band: Option<usize>) -> Self {
        self.mask_band = mask_band;
        self
    }

    /// Offset of the destination window; see [`WarpOptions::dst_col_off`].
    pub fn with_dst_offset(mut self, col_off: usize, row_off: usize) -> Self {
        self.dst_col_off = col_off;
        self.dst_row_off = row_off;
        self
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
/// offset, so in a multi-chunk gdalwarp every chunk is estimated at the
/// output's top-left (confirmed with GDAL 3.14.0dev: equal-sized chunks at
/// different latitudes of a lon/lat to Mercator warp log identical
/// scales). This adds `dst_col_off` / `dst_row_off`, so the estimate is for
/// the window actually warped. For a single-chunk warp, as in the test
/// fixtures, the two agree.
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

/// Cubic B-spline at its native support. GDAL has no 4-sample formula for
/// it and runs it through the general kernel at every scale (`GWKResample`,
/// `GWKResampleNoMasksT`).
const CUBICSPLINE_POINT: Widened = Widened { alg: ResampleAlg::CubicSpline, sx: 1.0, sy: 1.0, rx: 2, ry: 2 };

impl Widened {
    /// `None` when GDAL would use the point kernel at this scale.
    fn new(alg: ResampleAlg, sx: f64, sy: f64) -> Option<Self> {
        let widen = match alg {
            ResampleAlg::Bilinear | ResampleAlg::Cubic => !(sx > 0.5 && sy > 0.5),
            ResampleAlg::Lanczos | ResampleAlg::CubicSpline => sx < 1.0 || sy < 1.0,
            _ => false,
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
        // Area methods and cubic spline have one implementation, the
        // generic one.
        _ => {
            let src = SourceBuffer {
                data: src_pixels, ncol: src_ncol, nrow: src_nrow, nbands: 1,
                col_off: src_col_off, row_off: src_row_off,
            };
            let opts = WarpOptions { alg, src_nodata: Nodata::Value(nodata as f64), ..Default::default() };
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

    // A nodata value is always given here, which GDAL treats as a mask: it
    // uses the general accumulation (GWKBilinearResample4Sample) even when
    // all four neighbours are valid, and so does this.
    // Weight valid in-bounds pixels and renormalise.
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

    if wsum < 1e-5 { None } else if wsum == 1.0 { Some(acc) } else { Some(acc / wsum) }
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

/// Cubic B-spline weight (B = 1, C = 0), GDAL's `GWKBSpline`
/// (gdalwarpkernel.cpp ~L3817), unnormalised (the 1/6 is dropped; the
/// kernels normalise by the weight sum). Like GDAL's it omits the
/// `(x - 2)^3` term, so it is not zero beyond |x| = 2 on the positive side;
/// a widened window can reach there, and GDAL's output includes it.
#[inline]
fn bspline_weight(x: f64) -> f64 {
    let (xp2, xp1, xm1) = (x + 2.0, x + 1.0, x - 1.0);
    if xp2 <= 0.0 {
        return 0.0;
    }
    // Summed innermost first, as GDAL's nested expression does.
    let mut t = 0.0;
    if xp1 > 0.0 {
        let mut u = 0.0;
        if x > 0.0 {
            let w = if xm1 > 0.0 { -4.0 * xm1 * xm1 * xm1 } else { 0.0 };
            u = w + 6.0 * x * x * x;
        }
        t = u + -4.0 * xp1 * xp1 * xp1;
    }
    t + xp2 * xp2 * xp2
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
// Data types, the working type, and conversion to the output type
// =========================================================================

/// GDAL data types for which rwarp implements [`Sample`] (no complex types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataType {
    UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float16, Float32, Float64,
}

impl DataType {
    fn bits(self) -> u32 {
        use DataType::*;
        match self {
            UInt8 | Int8 => 8,
            UInt16 | Int16 | Float16 => 16,
            UInt32 | Int32 | Float32 => 32,
            UInt64 | Int64 | Float64 => 64,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, DataType::Float16 | DataType::Float32 | DataType::Float64)
    }

    fn is_signed(self) -> bool {
        !matches!(self, DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64)
    }

    /// Range of an integer type.
    fn int_range(self) -> (i128, i128) {
        use DataType::*;
        match self {
            UInt8 => (0, u8::MAX as i128),
            Int8 => (i8::MIN as i128, i8::MAX as i128),
            UInt16 => (0, u16::MAX as i128),
            Int16 => (i16::MIN as i128, i16::MAX as i128),
            UInt32 => (0, u32::MAX as i128),
            Int32 => (i32::MIN as i128, i32::MAX as i128),
            UInt64 => (0, u64::MAX as i128),
            Int64 => (i64::MIN as i128, i64::MAX as i128),
            Float16 | Float32 | Float64 => unreachable!("not an integer type"),
        }
    }

    /// `v` rounded to this floating type, as f64.
    fn round_float(self, v: f64) -> f64 {
        match self {
            DataType::Float32 => v as f32 as f64,
            DataType::Float16 => f16_round(v),
            _ => v,
        }
    }

    /// GDAL's `GDALIsValueExactAs`.
    fn is_exact(self, v: f64) -> bool {
        if self.is_float() {
            v.is_nan() || self.round_float(v) == v
        } else {
            let (lo, hi) = self.int_range();
            v.fract() == 0.0 && v >= lo as f64 && v <= hi as f64
        }
    }

    /// GDAL's `GDALFindDataType` (no complex; never Float16).
    fn find(bits: u32, signed: bool, floating: bool) -> DataType {
        use DataType::*;
        if floating {
            return if bits <= 32 { Float32 } else { Float64 };
        }
        match (signed, bits) {
            (false, 0..=8) => UInt8,
            (false, 9..=16) => UInt16,
            (false, 17..=32) => UInt32,
            (false, 33..=64) => UInt64,
            (true, 0..=8) => Int8,
            (true, 9..=16) => Int16,
            (true, 17..=32) => Int32,
            (true, 33..=64) => Int64,
            _ => Float64,
        }
    }

    /// GDAL's `GDALDataTypeUnion` (gdal_misc.cpp ~L136).
    fn union(a: Option<DataType>, b: DataType) -> DataType {
        let Some(a) = a else { return b };
        let (bits, signed, floating) = ([a.bits(), b.bits()], [a.is_signed(), b.is_signed()], [a.is_float(), b.is_float()]);
        // GetMinBitsForPair.
        let n = if floating[0] != floating[1] {
            let (f, i) = if floating[0] { (0, 1) } else { (1, 0) };
            bits[f].max(2 * bits[i])
        } else if signed[0] != signed[1] {
            if !signed[0] && bits[0] < bits[1] {
                bits[1]
            } else if !signed[1] && bits[1] < bits[0] {
                bits[0]
            } else {
                let (u, s) = if signed[0] { (1, 0) } else { (0, 1) };
                bits[s].max(2 * bits[u])
            }
        } else {
            bits[0].max(bits[1])
        };
        DataType::find(n, signed[0] || signed[1], floating[0] || floating[1])
    }

    /// GDAL's `GDALDataTypeUnionWithValue` (gdal_misc.cpp ~L178).
    fn union_with_value(a: DataType, v: f64) -> DataType {
        // GDAL does not return Float16 here.
        let a = if a == DataType::Float16 { DataType::Float32 } else { a };
        if a.is_exact(v) {
            return a;
        }
        // GDALFindDataTypeForValue / GetMinBitsForValue.
        let floating = v.round() != v || v > u64::MAX as f64 || v < i64::MIN as f64;
        let signed = floating || v < 0.0;
        let bits = if v.round() == v {
            if (0.0..=255.0).contains(&v) || (-128.0..=127.0).contains(&v) {
                8
            } else if (-32768.0..=32767.0).contains(&v) || (0.0..=65535.0).contains(&v) {
                16
            } else if (i32::MIN as f64..=i32::MAX as f64).contains(&v) || (0.0..=u32::MAX as f64).contains(&v) {
                32
            } else {
                64
            }
        } else if v as f32 as f64 == v {
            32
        } else {
            64
        };
        DataType::union(Some(a), DataType::find(bits, signed, floating))
    }

    /// The warp's working type: GDAL's `GDALWarpResolveWorkingDataType`
    /// (gdalwarper.cpp ~L1764), the union of the output and source types
    /// and every source and destination nodata value.
    pub fn working(src: DataType, dst: DataType, src_nodata: &[Option<f64>], dst_nodata: &[Option<f64>]) -> DataType {
        let mut w = None;
        for b in 0..src_nodata.len().max(1) {
            let mut t = DataType::union(w, dst);
            t = DataType::union(Some(t), src);
            if let Some(Some(v)) = src_nodata.get(b) { t = DataType::union_with_value(t, *v); }
            if let Some(Some(v)) = dst_nodata.get(b) { t = DataType::union_with_value(t, *v); }
            w = Some(t);
        }
        w.unwrap()
    }
}

/// Round to the nearest half-precision value (as GDAL's `GFloat16`).
fn f16_round(v: f64) -> f64 {
    #[cfg(feature = "f16")]
    { half::f16::from_f64(v).to_f64() }
    #[cfg(not(feature = "f16"))]
    { v }
}

/// A value held in the working type: integers exactly, floats as f64
/// rounded to the working type's precision.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Wv {
    I(i128),
    F(f64),
}

fn to_wv<S: Sample>(v: S) -> Wv {
    match v.to_i128() {
        Some(i) => Wv::I(i),
        None => Wv::F(v.to_f64()),
    }
}

/// The working type and the operations GDAL does in it.
#[derive(Debug, Clone, Copy)]
struct Working {
    dt: DataType,
}

impl Working {
    /// A computed value stored into the working type: integers clamp and
    /// round half up (`ClampRoundAndAvoidNoData`, gdalwarpkernel.cpp
    /// ~L1856); floats round to the type.
    fn store(&self, r: f64) -> Wv {
        if self.dt.is_float() {
            return Wv::F(self.dt.round_float(r));
        }
        let (lo, hi) = self.dt.int_range();
        let v = if r < lo as f64 {
            lo
        } else if r > hi as f64 {
            hi
        } else {
            let x = if self.dt.is_signed() { (r + 0.5).floor() } else { r + 0.5 };
            (x as i128).clamp(lo, hi)
        };
        Wv::I(v)
    }

    /// A source value copied into the working type (which contains the
    /// source type, so this is exact).
    fn copy(&self, w: Wv) -> Wv {
        match (w, self.dt.is_float()) {
            (Wv::I(i), true) => Wv::F(self.dt.round_float(i as f64)),
            (Wv::F(f), false) => self.store(f),
            (w, _) => w,
        }
    }

    fn equals(&self, w: Wv, nd: f64) -> bool {
        match w {
            Wv::I(i) => i as f64 == nd,
            Wv::F(f) => f == nd,
        }
    }

    /// The neighbouring working-type value (`AvoidNoData`, ~L1613).
    fn step_off(&self, w: Wv) -> Wv {
        match w {
            Wv::I(i) => {
                let (lo, _) = self.dt.int_range();
                Wv::I(if i == lo { lo + 1 } else { i - 1 })
            }
            Wv::F(f) => Wv::F(match self.dt {
                DataType::Float32 => {
                    let x = f as f32;
                    (if x == f32::MAX { x.next_down() } else { x.next_up() }) as f64
                }
                #[cfg(feature = "f16")]
                DataType::Float16 => half::f16::from_f64(f).step_off().to_f64(),
                _ => if f == f64::MAX { f.next_down() } else { f.next_up() },
            }),
        }
    }
}

/// A working-type value written to the output type: GDAL's
/// `GDALCopyWords` (gcore/gdal_priv_templates.hpp). Integers clamp; floats
/// into integers make NaN 0, round half away from zero (signed) or add 0.5
/// and truncate (unsigned), in the working type's precision, and clamp.
fn from_wv<D: Sample>(w: Wv, from: DataType) -> D {
    let to = D::DATA_TYPE;
    match w {
        Wv::I(i) => {
            if to.is_float() {
                D::from_i128(i)
            } else {
                let (lo, hi) = to.int_range();
                D::from_i128(i.clamp(lo, hi))
            }
        }
        Wv::F(f) => {
            if to.is_float() {
                return D::from_f64(f);
            }
            if f.is_nan() {
                return D::from_i128(0);
            }
            let (lo, hi) = to.int_range();
            let r = if from == DataType::Float32 {
                let x = f as f32;
                let x = if to.is_signed() && x < 0.0 { x - 0.5 } else { x + 0.5 };
                x.clamp(lo as f32, hi as f32) as i128
            } else {
                let x = if to.is_signed() && f < 0.0 { f - 0.5 } else { f + 0.5 };
                x.clamp(lo as f64, hi as f64) as i128
            };
            D::from_i128(r.clamp(lo, hi))
        }
    }
}

// =========================================================================
// Generic interleaved kernel (any Sample type, multi-band, one pass)
// =========================================================================

/// A pixel sample type the generic kernel can resample.
///
/// Implemented for `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
/// `f32`, `f64`, and `half::f16` with feature `f16`. Nearest and the
/// selecting area methods (mode, min, max, quantiles) copy values exactly;
/// the others compute in `f64` and convert back with [`Sample::from_f64`],
/// so 64-bit integers beyond 2^53 lose precision there, as in GDAL.
///
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
    /// Whether GDAL has a typed fast path for this type's unmasked bilinear
    /// (`GWKBilinearResampleNoMasks4SampleT`: Byte, Int16, UInt16, Float32,
    /// Float64), which sums in a different order from the general one.
    /// Only affects the last bit, which can flip integer rounding at large
    /// magnitudes.
    #[doc(hidden)]
    const GDAL_FAST_BILINEAR: bool = false;
    /// The GDAL data type.
    const DATA_TYPE: DataType;
    /// The value as an exact integer (integer types only).
    #[doc(hidden)]
    fn to_i128(self) -> Option<i128> { None }
    /// From an integer in range (rounding to nearest for floats).
    #[doc(hidden)]
    fn from_i128(v: i128) -> Self;
}

macro_rules! int_sample {
    ($($t:ty => $fast:expr, $dt:ident),*) => {$(
        impl Sample for $t {
            const GDAL_FAST_BILINEAR: bool = $fast;
            const DATA_TYPE: DataType = DataType::$dt;
            #[inline] fn to_i128(self) -> Option<i128> { Some(self as i128) }
            #[inline] fn from_i128(v: i128) -> Self { v as $t }
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
int_sample!(
    u8 => true, UInt8, i8 => false, Int8, u16 => true, UInt16, i16 => true, Int16,
    u32 => false, UInt32, i32 => false, Int32, u64 => false, UInt64, i64 => false, Int64
);

impl Sample for f32 {
    const GDAL_FAST_BILINEAR: bool = true;
    const DATA_TYPE: DataType = DataType::Float32;
    #[inline] fn from_i128(v: i128) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn step_off(self) -> Self { if self == f32::MAX { self.next_down() } else { self.next_up() } }
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
}
impl Sample for f64 {
    const GDAL_FAST_BILINEAR: bool = true;
    const DATA_TYPE: DataType = DataType::Float64;
    #[inline] fn from_i128(v: i128) -> Self { v as f64 }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn step_off(self) -> Self { if self == f64::MAX { self.next_down() } else { self.next_up() } }
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
}

/// GDAL's Float16 (feature `f16`). Resampled values are computed in `f64`
/// and rounded to the nearest half-precision value.
#[cfg(feature = "f16")]
impl Sample for half::f16 {
    const DATA_TYPE: DataType = DataType::Float16;
    #[inline] fn from_i128(v: i128) -> Self { half::f16::from_f32(v as f32) }
    #[inline] fn to_f64(self) -> f64 { self.to_f64() }
    #[inline] fn from_f64(v: f64) -> Self { half::f16::from_f64(v) }
    #[inline] fn step_off(self) -> Self {
        // The next value up (down at the maximum), stepping the bits: up is
        // one more for positive values, one less for negative, and the
        // smallest subnormal from zero.
        let bits = self.to_bits();
        let up = if self == half::f16::MAX {
            bits - 1
        } else if bits & 0x7fff == 0 {
            1
        } else if bits & 0x8000 == 0 {
            bits + 1
        } else {
            bits - 1
        };
        half::f16::from_bits(up)
    }
    #[inline] fn zero() -> Self { half::f16::ZERO }
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

/// Warp a pixel-interleaved image of any [`Sample`] type in a single pass,
/// to the same type. See [`warp_to`].
pub fn warp<T: Sample>(
    transformer: &impl Transformer,
    src: &SourceBuffer<'_, T>,
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions,
) -> Vec<T> {
    warp_to::<T, T>(transformer, src, dst_ncol, dst_nrow, opts)
}

/// Warp a pixel-interleaved image of type `S` to type `D` in a single pass.
///
/// Each destination pixel is transformed once and the kernel weights are
/// computed once, then applied to every band from contiguous memory.
///
/// Types follow gdalwarp: the warp works in the union of `S`, `D` and the
/// nodata values ([`DataType::working`]). Resampled values are stored into
/// that type (integers clamp and round half up), moved off the destination
/// nodata, and converted to `D` as `GDALCopyWords` does (integers clamp;
/// floats into integers round half away from zero, NaN becomes 0). Nearest
/// and the selecting area methods copy source values exactly. So with
/// `S = D` and nodata representable in it, the working type is `S` and
/// nothing is converted.
///
/// Validity is per band; see [`BandNodata`]. A source pixel is nodata in a
/// band if the band equals its [`Nodata`] value or is NaN; a zero in
/// `opts.mask_band` invalidates the whole pixel.
///
/// Destination pixels with no value are the destination nodata (see
/// [`DstNodata`]) converted to `D`, or zero.
///
/// Returns `dst_ncol * dst_nrow * src.nbands` values, interleaved.
pub fn warp_to<S: Sample, D: Sample>(
    transformer: &impl Transformer,
    src: &SourceBuffer<'_, S>,
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions,
) -> Vec<D> {
    let nbands = src.nbands;
    assert_eq!(src.data.len(), src.ncol * src.nrow * nbands, "source buffer size mismatch");
    assert!(opts.mask_band.map_or(true, |a| a < nbands), "mask_band out of range");

    let out_spec = Output::<D>::new(S::DATA_TYPE, nbands, opts);
    let img = Interleaved {
        src: src.data, ncol: src.ncol, nrow: src.nrow, nb: nbands,
        mask: opts.mask_band, nodata: out_spec.src_nodata::<S>(), rule: opts.band_nodata,
    };
    let alg = opts.alg;
    let (sx, sy) = match opts.scaling {
        KernelScaling::Point => (1.0, 1.0),
        KernelScaling::WidenWhenDownsampling => {
            estimate_scale(transformer, dst_ncol, dst_nrow, opts.dst_col_off, opts.dst_row_off)
        }
        KernelScaling::Fixed { x, y } => (x, y),
    };
    if alg.is_area() {
        return warp_area(transformer, &img, (src.col_off as f64, src.row_off as f64), dst_ncol, dst_nrow, opts, (sx, sy), &out_spec);
    }
    let widened = Widened::new(alg, sx, sy);

    let mut out = out_spec.filled(dst_ncol * dst_nrow);
    let mut sc = Scratch::new(nbands);

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
                let pix = img.px(br as usize * src.ncol + bc as usize);
                if !img.pixel_ok(pix) { continue; }
                for b in 0..nbands {
                    if img.band_ok(pix, b) { out[o + b] = out_spec.copied(pix[b], b); }
                }
                continue;
            }

            // Sample points outside [0, n) are off the source
            // (GWKCheckAndComputeSrcOffsets).
            let Some((cx, cy)) = centre_pixel(buf_x, buf_y, src.ncol, src.nrow) else { continue };

            // GDAL skips a destination pixel whose centre lands on an
            // invalid source pixel before any kernel runs
            // (GWKRealCaseThread, gdalwarpkernel.cpp ~L5680).
            if !img.valid(cy * src.ncol + cx) { continue; }

            let any = match (widened, alg) {
                (Some(w), _) => img.widened(buf_x, buf_y, &w, &mut sc),
                (None, ResampleAlg::Bilinear) => img.bilinear(buf_x, buf_y, &mut sc.acc, &mut sc.hit),
                (None, ResampleAlg::Cubic) => img.cubic(buf_x, buf_y, &mut sc),
                (None, ResampleAlg::Lanczos) => img.widened(buf_x, buf_y, &LANCZOS_POINT, &mut sc),
                (None, ResampleAlg::CubicSpline) => img.widened(buf_x, buf_y, &CUBICSPLINE_POINT, &mut sc),
                (None, _) => unreachable!("nearest and area methods are handled above"),
            };
            if any {
                for b in 0..nbands {
                    if sc.hit[b] { out[o + b] = out_spec.computed(sc.acc[b], b); }
                }
            }
        }
    }
    out
}

/// Writing results: the working type, the destination nodata per band, and
/// the fill.
struct Output<D> {
    working: Working,
    src_nd: Vec<Option<f64>>,
    dst_nd: Vec<Option<f64>>,
    fill: Vec<D>,
}

impl<D: Sample> Output<D> {
    fn new(src_dt: DataType, nb: usize, opts: &WarpOptions) -> Self {
        let mut src_nd = opts.src_nodata.per_band(nb);
        let dst_nd = match &opts.dst_nodata {
            DstNodata::FromSource => src_nd.clone(),
            DstNodata::None => vec![None; nb],
            DstNodata::Value(v) => vec![Some(*v); nb],
            DstNodata::PerBand(v) => {
                assert_eq!(v.len(), nb, "one destination nodata value per band");
                v.iter().map(|&x| Some(x)).collect()
            }
        };
        // GDAL ignores the source nodata if no band's value is
        // representable in the source type (GDALWarpResolveWorkingDataType).
        if src_nd.iter().all(|v| v.map_or(true, |v| !src_dt.is_exact(v))) {
            src_nd = vec![None; nb];
        }
        let working = Working { dt: DataType::working(src_dt, D::DATA_TYPE, &src_nd, &dst_nd) };
        let fill = dst_nd.iter()
            .map(|v| v.map_or_else(D::zero, |v| from_wv::<D>(Wv::F(v), DataType::Float64)))
            .collect();
        Output { working, src_nd, dst_nd, fill }
    }

    /// The source nodata as `S`, per band, where representable.
    fn src_nodata<S: Sample>(&self) -> Vec<Option<S>> {
        self.src_nd.iter()
            .map(|v| v.filter(|&v| S::DATA_TYPE.is_exact(v)).map(S::from_f64))
            .collect()
    }

    fn filled(&self, npix: usize) -> Vec<D> {
        let nb = self.fill.len();
        (0..npix * nb).map(|i| self.fill[i % nb]).collect()
    }

    #[inline]
    fn put(&self, w: Wv, b: usize) -> D {
        let w = match self.dst_nd[b] {
            Some(nd) if self.working.equals(w, nd) => self.working.step_off(w),
            _ => w,
        };
        from_wv::<D>(w, self.working.dt)
    }

    /// A computed value for band `b`.
    #[inline]
    fn computed(&self, r: f64, b: usize) -> D {
        self.put(self.working.store(r), b)
    }

    /// A source value copied into band `b`.
    #[inline]
    fn copied<S: Sample>(&self, v: S, b: usize) -> D {
        self.put(self.working.copy(to_wv(v)), b)
    }

    /// A reducer's result for band `b`, if any.
    #[inline]
    fn result<S: Sample>(&self, r: Res<S>, b: usize) -> Option<D> {
        match r {
            Res::None => None,
            Res::Computed(v) => Some(self.computed(v, b)),
            Res::Copied(v) => Some(self.copied(v, b)),
        }
    }
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
    let src_nodata = nodata.map_or(Nodata::None, |v| Nodata::Value(v.to_f64()));
    let opts = WarpOptions { alg, scaling: KernelScaling::Point, src_nodata, mask_band, ..Default::default() };
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

/// View over a pixel-interleaved buffer with per-band validity.
struct Interleaved<'a, T: Sample> {
    src: &'a [T],
    ncol: usize,
    nrow: usize,
    nb: usize,
    mask: Option<usize>,
    /// Source nodata per band, where representable in `T`.
    nodata: Vec<Option<T>>,
    rule: BandNodata,
}

/// Per-call scratch for the interpolating kernels: a value and a weight per
/// band, and whether the band got a value.
struct Scratch {
    acc: Vec<f64>,
    wsum: Vec<f64>,
    row_acc: Vec<f64>,
    row_w: Vec<f64>,
    hit: Vec<bool>,
    wx: Vec<f64>,
    fallback: Vec<f64>,
    fallback_hit: Vec<bool>,
}

impl Scratch {
    fn new(nb: usize) -> Self {
        Scratch {
            acc: vec![0.0; nb], wsum: vec![0.0; nb], row_acc: vec![0.0; nb], row_w: vec![0.0; nb],
            hit: vec![false; nb], wx: Vec::new(), fallback: vec![0.0; nb], fallback_hit: vec![false; nb],
        }
    }

    fn clear(&mut self) {
        self.acc.fill(0.0);
        self.wsum.fill(0.0);
        self.hit.fill(false);
    }
}

impl<'a, T: Sample> Interleaved<'a, T> {
    #[inline]
    fn px(&self, p: usize) -> &'a [T] {
        &self.src[p * self.nb..(p + 1) * self.nb]
    }

    #[inline]
    fn is_nd(&self, v: T, b: usize) -> bool {
        v.is_nan() || self.nodata[b] == Some(v)
    }

    #[inline]
    fn any_valid(&self, pix: &[T]) -> bool {
        pix.iter().enumerate().any(|(b, &v)| !self.is_nd(v, b))
    }

    #[inline]
    fn all_valid(&self, pix: &[T]) -> bool {
        pix.iter().enumerate().all(|(b, &v)| !self.is_nd(v, b))
    }

    /// Whether the source pixel takes part at all: GDAL's unified validity
    /// mask, which decides the centre check and, under
    /// [`BandNodata::Unified`] and [`BandNodata::AnyBand`], every band.
    #[inline]
    fn pixel_ok(&self, pix: &[T]) -> bool {
        if let Some(a) = self.mask {
            if pix[a] == T::zero() { return false; }
        }
        match self.rule {
            BandNodata::Partial | BandNodata::Unified => self.any_valid(pix),
            BandNodata::PerBand => true,
            BandNodata::AnyBand => self.all_valid(pix),
        }
    }

    /// Whether band `b` of a source pixel contributes to band `b`.
    #[inline]
    fn band_ok(&self, pix: &[T], b: usize) -> bool {
        match self.rule {
            BandNodata::Partial | BandNodata::PerBand => {
                !self.is_nd(pix[b], b) && self.mask.map_or(true, |a| pix[a] != T::zero())
            }
            BandNodata::Unified | BandNodata::AnyBand => self.pixel_ok(pix),
        }
    }

    /// Every band of the pixel contributes (the common case, and the fast
    /// path).
    #[inline]
    fn all_ok(&self, pix: &[T]) -> bool {
        match self.rule {
            BandNodata::Partial | BandNodata::PerBand => {
                self.mask.map_or(true, |a| pix[a] != T::zero()) && self.all_valid(pix)
            }
            BandNodata::Unified | BandNodata::AnyBand => self.pixel_ok(pix),
        }
    }

    #[inline]
    fn valid(&self, p: usize) -> bool {
        self.pixel_ok(self.px(p))
    }

    /// Hand source pixel `p` with weight `w` to an area reducer, with its
    /// validity.
    #[inline]
    fn reduce<R: AreaReducer<T>>(&self, red: &mut R, p: usize, w: f64, band_ok: &mut [bool]) {
        let pix = self.px(p);
        let pixel_ok = self.pixel_ok(pix);
        for (b, ok) in band_ok.iter_mut().enumerate() { *ok = pixel_ok && self.band_ok(pix, b); }
        red.add(pix, pixel_ok, band_ok, w);
    }

    /// Add `pix` with weight `w` to the bands it contributes to. The weight
    /// of pixels valid in every band goes to `shared`, the rest per band to
    /// `wsum`, so the common case costs one addition.
    #[inline]
    fn accumulate(&self, pix: &[T], w: f64, acc: &mut [f64], shared: &mut f64, wsum: &mut [f64]) {
        if self.all_ok(pix) {
            for (a, &v) in acc.iter_mut().zip(pix) {
                *a += v.to_f64() * w;
            }
            *shared += w;
        } else {
            for b in 0..self.nb {
                if self.band_ok(pix, b) {
                    acc[b] += pix[b].to_f64() * w;
                    wsum[b] += w;
                }
            }
        }
    }

    /// Bilinear at a sample point (GDAL's `GWKBilinearResample4Sample`,
    /// gdalwarpkernel.cpp ~L3090, and `GWKBilinearResampleNoMasks4SampleT`).
    /// Per band, invalid and out-of-buffer neighbours are dropped and the
    /// rest renormalised; a band whose neighbours weigh less than 1e-5 gets
    /// no value. Fills `out` / `hit` per band.
    fn bilinear(&self, buf_x: f64, buf_y: f64, out: &mut [f64], hit: &mut [bool]) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let rx = 1.5 - (buf_x - ix as f64);
        let ry = 1.5 - (buf_y - iy as f64);
        let (ncol, nrow) = (self.ncol as i64, self.nrow as i64);

        // GDAL's factored formula only for its fast-path types with no
        // validity masks; otherwise the general accumulation below.
        let fast = T::GDAL_FAST_BILINEAR && self.nodata.iter().all(Option::is_none) && self.mask.is_none();
        if fast && ix >= 0 && ix + 1 < ncol && iy >= 0 && iy + 1 < nrow {
            let p00 = iy as usize * self.ncol + ix as usize;
            let (a, b, c, d) = (self.px(p00), self.px(p00 + 1), self.px(p00 + self.ncol), self.px(p00 + self.ncol + 1));
            if self.all_ok(a) && self.all_ok(b) && self.all_ok(c) && self.all_ok(d) {
                for ((((o, &a), &b), &c), &d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
                    *o = (a.to_f64() * rx + b.to_f64() * (1.0 - rx)) * ry
                        + (c.to_f64() * rx + d.to_f64() * (1.0 - rx)) * (1.0 - ry);
                }
                hit.fill(true);
                return true;
            }
        }

        let mut any = false;
        for band in 0..self.nb {
            let (mut acc, mut div) = (0.0, 0.0);
            for &(cx, cy, w) in &[
                (ix,     iy,     rx * ry),
                (ix + 1, iy,     (1.0 - rx) * ry),
                (ix,     iy + 1, rx * (1.0 - ry)),
                (ix + 1, iy + 1, (1.0 - rx) * (1.0 - ry)),
            ] {
                if cx >= 0 && cx < ncol && cy >= 0 && cy < nrow {
                    let pix = self.px(cy as usize * self.ncol + cx as usize);
                    if self.band_ok(pix, band) {
                        acc += pix[band].to_f64() * w;
                        div += w;
                    }
                }
            }
            hit[band] = div >= 1e-5;
            if hit[band] {
                out[band] = if div == 1.0 { acc } else { acc / div };
                any = true;
            }
        }
        any
    }

    /// Cubic at a sample point (GDAL's `GWKCubicResample4Sample`, ~L3420).
    /// Near the buffer edge, or for a band with any invalid pixel in the
    /// 4×4 window, falls back to [`Self::bilinear`] for that band.
    fn cubic(&self, buf_x: f64, buf_y: f64, sc: &mut Scratch) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let dx = buf_x - 0.5 - ix as f64;
        let dy = buf_y - 0.5 - iy as f64;

        if ix - 1 < 0 || ix + 2 >= self.ncol as i64 || iy - 1 < 0 || iy + 2 >= self.nrow as i64 {
            return self.bilinear(buf_x, buf_y, &mut sc.acc, &mut sc.hit);
        }
        let wx = [cubic_weight(dx + 1.0), cubic_weight(dx), cubic_weight(dx - 1.0), cubic_weight(dx - 2.0)];
        let wy = [cubic_weight(dy + 1.0), cubic_weight(dy), cubic_weight(dy - 1.0), cubic_weight(dy - 2.0)];

        // Row-by-row accumulation, as the per-band reference kernel does.
        let nb = self.nb;
        sc.acc.fill(0.0);
        sc.hit.fill(true);
        for (jj, &wyj) in wy.iter().enumerate() {
            sc.row_acc.fill(0.0);
            let start = (((iy - 1) as usize + jj) * self.ncol + (ix - 1) as usize) * nb;
            for (pix, &wxi) in self.src[start..start + 4 * nb].chunks_exact(nb).zip(&wx) {
                if self.all_ok(pix) {
                    for (r, &v) in sc.row_acc.iter_mut().zip(pix) { *r += v.to_f64() * wxi; }
                } else {
                    for b in 0..nb {
                        if self.band_ok(pix, b) { sc.row_acc[b] += pix[b].to_f64() * wxi; } else { sc.hit[b] = false; }
                    }
                }
            }
            for (o, &r) in sc.acc.iter_mut().zip(sc.row_acc.iter()) { *o += r * wyj; }
        }
        if sc.hit.iter().all(|&h| h) { return true; }
        // Any band with an invalid pixel in the window: bilinear for it.
        self.bilinear(buf_x, buf_y, &mut sc.fallback, &mut sc.fallback_hit);
        let mut any = false;
        for b in 0..nb {
            if !sc.hit[b] {
                sc.hit[b] = sc.fallback_hit[b];
                sc.acc[b] = sc.fallback[b];
            }
            any |= sc.hit[b];
        }
        any
    }

    /// Kernel widened by the downsampling scale (GDAL's `GWKResample`,
    /// gdalwarpkernel.cpp ~L4027, and `GWKResampleNoMasksT` ~L4840), and
    /// lanczos at any scale.
    ///
    /// The window is `1 - r ..= r` around the pixel below-left of the
    /// sample point, clipped to the buffer; weights are the kernel at
    /// `(i - delta) * scale`. Per band, invalid pixels are skipped and the
    /// result is normalised by the weight of the pixels used; a band whose
    /// weight is below 1e-6 gets no value.
    fn widened(&self, buf_x: f64, buf_y: f64, w: &Widened, sc: &mut Scratch) -> bool {
        let ix = (buf_x - 0.5).floor() as i64;
        let iy = (buf_y - 0.5).floor() as i64;
        let (ncol, nrow) = (self.ncol as i64, self.nrow as i64);

        // For bilinear and cubic GDAL declines a centre past the buffer, or
        // a support wider than the buffer, and uses the 4-sample bilinear
        // formula instead (GWKResampleNoMasksT).
        if w.alg != ResampleAlg::Lanczos && (ix >= ncol || iy >= nrow || w.rx > ncol || w.ry > nrow) {
            return self.bilinear(buf_x, buf_y, &mut sc.acc, &mut sc.hit);
        }
        let dx = buf_x - 0.5 - ix as f64;
        let dy = buf_y - 0.5 - iy as f64;
        let (i0, i1) = ((1 - w.rx).max(-ix), w.rx.min(ncol - 1 - ix));
        let (j0, j1) = ((1 - w.ry).max(-iy), w.ry.min(nrow - 1 - iy));

        sc.wx.clear();
        sc.wx.extend((i0..=i1).map(|i| w.alg.weight((i as f64 - dx) * w.sx)));

        sc.clear();
        let nb = self.nb;
        for j in j0..=j1 {
            let wy = w.alg.weight((j as f64 - dy) * w.sy);
            sc.row_acc.fill(0.0);
            sc.row_w.fill(0.0);
            let mut row_shared = 0.0;
            let start = (((iy + j) * ncol + ix + i0) as usize) * nb;
            let run = &self.src[start..start + sc.wx.len() * nb];
            for (pix, &wxi) in run.chunks_exact(nb).zip(sc.wx.iter()) {
                self.accumulate(pix, wxi, &mut sc.row_acc, &mut row_shared, &mut sc.row_w);
            }
            for b in 0..nb {
                sc.acc[b] += sc.row_acc[b] * wy;
                sc.wsum[b] += (row_shared + sc.row_w[b]) * wy;
            }
        }
        let mut any = false;
        for b in 0..nb {
            let ws = sc.wsum[b];
            sc.hit[b] = ws >= 1e-6;
            if sc.hit[b] {
                if !(0.99999..=1.00001).contains(&ws) { sc.acc[b] /= ws; }
                any = true;
            }
        }
        any
    }
}

// =========================================================================
// Area methods (average, mode)
// =========================================================================

/// What an area method does with the source pixels under a destination
/// pixel's footprint and their weights.
trait AreaReducer<T: Sample> {
    fn reset(&mut self);
    /// A source pixel with weight `w > 0`: `pixel_ok` is the whole-pixel
    /// validity, `band_ok` the per-band validity (see [`BandNodata`]).
    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64);
    /// The result for one destination pixel, band by band.
    fn write(&self, out: &mut [Res<T>]);
}

/// An area method's result for one band: a computed value (stored into the
/// working type like any resampled value), a source value (copied
/// exactly), or none.
#[derive(Debug, Clone, Copy)]
enum Res<T> {
    None,
    Computed(f64),
    Copied(T),
}

/// Running weighted mean per band over its valid pixels (West's weighted
/// incremental algorithm, as GDAL's average uses).
struct AreaMean {
    mean: Vec<f64>,
    total: Vec<f64>,
}

impl<T: Sample> AreaReducer<T> for AreaMean {
    fn reset(&mut self) {
        self.mean.fill(0.0);
        self.total.fill(0.0);
    }

    #[inline]
    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64) {
        if !pixel_ok { return; }
        for b in 0..pix.len() {
            if band_ok[b] {
                self.total[b] += w;
                self.mean[b] += (w / self.total[b]) * (pix[b].to_f64() - self.mean[b]);
            }
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        for b in 0..out.len() {
            out[b] = if self.total[b] > 0.0 { Res::Computed(self.mean[b]) } else { Res::None };
        }
    }
}

/// Average with GDAL's nodata and excluded-value thresholds, all bands
/// together; see [`AverageThresholds`].
struct AreaMeanThresholds<'a> {
    th: &'a AverageThresholds,
    invalid: f64,
    excluded: f64,
    regular: f64,
    excluded_counts: Vec<usize>,
    mean: Vec<f64>,
}

impl<T: Sample> AreaReducer<T> for AreaMeanThresholds<'_> {
    fn reset(&mut self) {
        self.invalid = 0.0;
        self.excluded = 0.0;
        self.regular = 0.0;
        self.excluded_counts.fill(0);
        self.mean.fill(0.0);
    }

    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64) {
        if !pixel_ok || !band_ok.iter().all(|&ok| ok) {
            self.invalid += w;
            return;
        }
        let hit = self.th.excluded_values.iter()
            .position(|t| t.len() == pix.len() && t.iter().zip(pix).all(|(&e, &v)| e == v.to_f64()));
        if let Some(i) = hit {
            self.excluded_counts[i] += 1;
            self.excluded += w;
            return;
        }
        self.regular += w;
        for (m, &v) in self.mean.iter_mut().zip(pix) {
            *m += (w / self.regular) * (v.to_f64() - *m);
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        out.fill(Res::None);
        let total = self.invalid + self.excluded + self.regular;
        if self.invalid > 0.0 && self.invalid >= self.th.nodata_pct_threshold / 100.0 * total {
            return;
        }
        if self.excluded > 0.0 && self.excluded >= self.th.excluded_pct_threshold / 100.0 * total {
            // The most frequent excluded tuple, the first on a tie.
            let (mut best, mut count) = (0, 0);
            for (i, &c) in self.excluded_counts.iter().enumerate() {
                if c > count { best = i; count = c; }
            }
            for (o, &v) in out.iter_mut().zip(&self.th.excluded_values[best]) { *o = Res::Computed(v); }
            return;
        }
        if self.regular > 0.0 {
            for (o, &m) in out.iter_mut().zip(&self.mean) { *o = Res::Computed(m); }
        }
    }
}

/// Smallest or largest valid value per band. Coverage weights are ignored:
/// every source pixel the footprint touches counts, as in GDAL
/// (gdalwarpkernel.cpp ~L8174-8295).
struct AreaExtreme<T> {
    max: bool,
    best: Vec<Option<T>>,
}

impl<T: Sample> AreaReducer<T> for AreaExtreme<T> {
    fn reset(&mut self) {
        self.best.fill(None);
    }

    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], _w: f64) {
        if !pixel_ok { return; }
        for ((best, &v), &ok) in self.best.iter_mut().zip(pix).zip(band_ok) {
            if !ok { continue; }
            match best {
                Some(b) if (self.max && v > *b) || (!self.max && v < *b) => *b = v,
                None => *best = Some(v),
                _ => {}
            }
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        for (o, b) in out.iter_mut().zip(&self.best) {
            *o = b.map_or(Res::None, Res::Copied);
        }
    }
}

/// Median or quartile of the valid values per band, unweighted: GDAL sorts
/// the values the footprint touches and takes element
/// `ceil(q * n - 1)`, computed in `f32` (~L8297-8340).
struct AreaQuantile<T> {
    q: f32,
    vals: Vec<Vec<T>>,
}

impl<T: Sample> AreaReducer<T> for AreaQuantile<T> {
    fn reset(&mut self) {
        for v in &mut self.vals { v.clear(); }
    }

    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], _w: f64) {
        if !pixel_ok { return; }
        for ((vals, &v), &ok) in self.vals.iter_mut().zip(pix).zip(band_ok) {
            if ok { vals.push(v); }
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        for (o, vals) in out.iter_mut().zip(&self.vals) {
            if vals.is_empty() { *o = Res::None; continue; }
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let i = (self.q * sorted.len() as f32 - 1.0).ceil() as usize;
            *o = Res::Copied(sorted[i.min(sorted.len() - 1)]);
        }
    }
}

/// Root mean square per band, `sqrt(sum(w v^2) / sum(w))` over valid
/// pixels (~L8025-8075).
struct AreaRms {
    sumsq: Vec<f64>,
    total: Vec<f64>,
}

impl<T: Sample> AreaReducer<T> for AreaRms {
    fn reset(&mut self) {
        self.sumsq.fill(0.0);
        self.total.fill(0.0);
    }

    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64) {
        if !pixel_ok { return; }
        for b in 0..pix.len() {
            if band_ok[b] {
                let v = pix[b].to_f64();
                self.total[b] += w;
                self.sumsq[b] += v * v * w;
            }
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        for b in 0..out.len() {
            out[b] = if self.total[b] > 0.0 { Res::Computed((self.sumsq[b] / self.total[b]).sqrt()) } else { Res::None };
        }
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

    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64) {
        if !pixel_ok { return; }
        let w = w as f32;
        for (((&v, bins), mode), &ok) in pix.iter().zip(&mut self.bins).zip(&mut self.mode).zip(band_ok) {
            if !ok { continue; }
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

    fn write(&self, out: &mut [Res<T>]) {
        for ((o, bins), m) in out.iter_mut().zip(&self.bins).zip(&self.mode) {
            *o = m.map_or(Res::None, |m| Res::Copied(bins[m].0));
        }
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
#[allow(clippy::too_many_arguments)]
fn warp_area<T: Sample, D: Sample>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    src_off: (f64, f64),
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions,
    scale: (f64, f64),
    out_spec: &Output<D>,
) -> Vec<D> {
    let nb = img.nb;
    match opts.alg {
        ResampleAlg::Min | ResampleAlg::Max => {
            let red = AreaExtreme { max: opts.alg == ResampleAlg::Max, best: vec![None; nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
        ResampleAlg::Med | ResampleAlg::Q1 | ResampleAlg::Q3 => {
            let q = match opts.alg { ResampleAlg::Q1 => 0.25, ResampleAlg::Q3 => 0.75, _ => 0.5 };
            let red = AreaQuantile { q, vals: vec![Vec::new(); nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
        ResampleAlg::Rms => {
            let red = AreaRms { sumsq: vec![0.0; nb], total: vec![0.0; nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
        ResampleAlg::Mode => {
            let red = AreaMode { ties: opts.mode_ties, bins: vec![Vec::new(); nb], mode: vec![None; nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
        _ if opts.average.active() => {
            let red = AreaMeanThresholds {
                th: &opts.average, invalid: 0.0, excluded: 0.0, regular: 0.0,
                excluded_counts: vec![0; opts.average.excluded_values.len()], mean: vec![0.0; nb],
            };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
        _ => {
            let red = AreaMean { mean: vec![0.0; nb], total: vec![0.0; nb] };
            warp_area_with(transformer, img, src_off, dst_ncol, dst_nrow, opts, scale, red, out_spec)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn warp_area_with<T: Sample, D: Sample, R: AreaReducer<T>>(
    transformer: &impl Transformer,
    img: &Interleaved<'_, T>,
    src_off: (f64, f64),
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WarpOptions,
    scale: (f64, f64),
    mut red: R,
    out_spec: &Output<D>,
) -> Vec<D> {
    let nb = img.nb;
    let mut out = out_spec.filled(dst_ncol * dst_nrow);
    let mut row_res: Vec<Res<T>> = vec![Res::None; dst_ncol * nb];
    let mut exact = match opts.footprint {
        Footprint::Exact { densify } => Some(Lattice::new(transformer, dst_ncol, densify.max(1), opts)),
        Footprint::GdalBox => None,
    };
    for dst_row in 0..dst_nrow {
        row_res.fill(Res::None);
        match exact.as_mut() {
            None => area_row_gdal_box(transformer, img, src_off, dst_row, dst_ncol, opts, scale, &mut red, &mut row_res),
            Some(lat) => {
                lat.row(transformer, dst_row);
                area_row_exact(lat, img, src_off, dst_ncol, &mut red, &mut row_res);
            }
        }
        let row_out = &mut out[dst_row * dst_ncol * nb..(dst_row + 1) * dst_ncol * nb];
        for (i, (o, &r)) in row_out.iter_mut().zip(row_res.iter()).enumerate() {
            if let Some(v) = out_spec.result(r, i % nb) { *o = v; }
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
    opts: &WarpOptions,
    (sx, sy): (f64, f64),
    red: &mut R,
    row_out: &mut [Res<T>],
) {
    let band_ok = &mut vec![false; img.nb];
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
                let w = b.weight(ix, iy);
                if w > 0.0 { img.reduce(red, (iy * ncol + col) as usize, w, band_ok); }
            }
        }
        red.write(&mut row_out[c * img.nb..(c + 1) * img.nb]);
    }
}

/// Points of one transformed line of the destination lattice.
#[derive(Default)]
struct LatticeLine {
    x: Vec<f64>,
    y: Vec<f64>,
    ok: Vec<bool>,
}

impl LatticeLine {
    fn transform(&mut self, transformer: &impl Transformer) {
        self.ok = transformer.transform(true, &mut self.x, &mut self.y);
    }
}

/// The destination pixel lattice of one row band, transformed to source
/// pixel space. Each pixel's outline is assembled from shared points, so a
/// corner is transformed once rather than four times and an edge point once
/// rather than twice, and each row line goes to the transformer as a real
/// scanline (constant y, increasing x), the shape an
/// [`ApproxTransformer`](crate::approx::ApproxTransformer) interpolates.
struct Lattice {
    densify: usize,
    ncol: usize,
    xoff: f64,
    yoff: f64,
    /// Row lines at the top and bottom of the band: `ncol * densify + 1`
    /// points, left to right.
    top: LatticeLine,
    bot: LatticeLine,
    /// Interior points of the band's column lines, one scanline per
    /// interior level: line `c`, point `k` (1 to `densify - 1`, downwards)
    /// at `(k - 1) * (ncol + 1) + c`.
    vert: LatticeLine,
    /// Scratch for one interior level.
    level: LatticeLine,
    /// Scratch: one pixel's outline, and its coverage window.
    rx: Vec<f64>,
    ry: Vec<f64>,
    win: controlledburn::CoverageWindow,
}

impl Lattice {
    fn new(transformer: &impl Transformer, ncol: usize, densify: usize, opts: &WarpOptions) -> Self {
        let mut lat = Lattice {
            densify, ncol,
            xoff: opts.dst_col_off as f64, yoff: opts.dst_row_off as f64,
            top: LatticeLine::default(), bot: LatticeLine::default(), vert: LatticeLine::default(),
            level: LatticeLine::default(),
            rx: Vec::new(), ry: Vec::new(), win: controlledburn::CoverageWindow::default(),
        };
        lat.fill_row_line(0);
        lat.bot.transform(transformer);
        lat
    }

    /// Fill `bot` with the untransformed row line `row` of the destination.
    fn fill_row_line(&mut self, row: usize) {
        let (n, d) = (self.ncol * self.densify + 1, self.densify as f64);
        let y = self.yoff + row as f64;
        self.bot.x.clear();
        self.bot.y.clear();
        self.bot.x.extend((0..n).map(|k| self.xoff + k as f64 / d));
        self.bot.y.resize(n, y);
    }

    /// Advance to destination row `row`: the previous bottom line becomes
    /// the top line; transform the new bottom line and column interiors.
    fn row(&mut self, transformer: &impl Transformer, row: usize) {
        std::mem::swap(&mut self.top, &mut self.bot);
        self.fill_row_line(row + 1);
        self.bot.transform(transformer);
        let d = self.densify;
        if d > 1 {
            // Each interior level is a scanline too (constant y), so it goes
            // to the transformer on its own.
            self.vert.x.clear();
            self.vert.y.clear();
            self.vert.ok.clear();
            for k in 1..d {
                let y = self.yoff + row as f64 + k as f64 / d as f64;
                self.level.x.clear();
                self.level.y.clear();
                self.level.x.extend((0..=self.ncol).map(|c| self.xoff + c as f64));
                self.level.y.resize(self.ncol + 1, y);
                self.level.transform(transformer);
                self.vert.x.extend_from_slice(&self.level.x);
                self.vert.y.extend_from_slice(&self.level.y);
                self.vert.ok.extend_from_slice(&self.level.ok);
            }
        }
    }

    /// Outline of pixel `c` in the band, offset into the source buffer:
    /// top left to right, right edge down, bottom right to left, left edge
    /// up (as rwarp 0.2 built it). False if any point failed.
    fn ring(&mut self, c: usize, (col_off, row_off): (f64, f64)) -> bool {
        let d = self.densify;
        self.rx.clear();
        self.ry.clear();
        let mut ok = true;
        let (rx, ry) = (&mut self.rx, &mut self.ry);
        let mut push = |l: &LatticeLine, k: usize| {
            rx.push(l.x[k] - col_off);
            ry.push(l.y[k] - row_off);
            ok &= l.ok[k];
        };
        for k in 0..d { push(&self.top, c * d + k); }
        push(&self.top, (c + 1) * d);
        let m = self.ncol + 1;
        for k in 1..d { push(&self.vert, (k - 1) * m + c + 1); }
        for k in 0..d { push(&self.bot, (c + 1) * d - k); }
        push(&self.bot, c * d);
        for k in (1..d).rev() { push(&self.vert, (k - 1) * m + c); }
        ok
    }
}

/// One destination row under the exact footprint: each pixel's outline,
/// densified and transformed (as part of the shared [`Lattice`]), with
/// exact coverage of the source pixels it touches from a
/// [`controlledburn::CoverageWindow`].
///
/// Weights follow what `controlledburn::burn` gave rwarp 0.2: within 1e-6
/// of 1 counts as 1, at or below 1e-6 is dropped, other fractions are
/// rounded to `f32`; fully covered pixels are reduced first, then the
/// partly covered ones, each in source scan order.
fn area_row_exact<T: Sample, R: AreaReducer<T>>(
    lat: &mut Lattice,
    img: &Interleaved<'_, T>,
    src_off: (f64, f64),
    dst_ncol: usize,
    red: &mut R,
    row_out: &mut [Res<T>],
) {
    const TOL: f64 = 1e-6;
    let band_ok = &mut vec![false; img.nb];
    let (ncol, nrow) = (img.ncol as i64, img.nrow as i64);
    for c in 0..dst_ncol {
        if !lat.ring(c, src_off) { continue; }
        let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (&x, &y) in lat.rx.iter().zip(&lat.ry) {
            minx = minx.min(x); maxx = maxx.max(x);
            miny = miny.min(y); maxy = maxy.max(y);
        }
        if !(minx.is_finite() && maxx.is_finite() && miny.is_finite() && maxy.is_finite()) { continue; }
        // Only the buffer pixels under the footprint's bbox; the window
        // clamps pieces beyond its columns exactly.
        let c0 = (minx.floor() as i64).max(0);
        let c1 = (maxx.ceil() as i64).min(ncol);
        let r0 = (miny.floor() as i64).max(0);
        let r1 = (maxy.ceil() as i64).min(nrow);
        if c0 >= c1 || r0 >= r1 { continue; }
        let (w, h) = ((c1 - c0) as usize, (r1 - r0) as usize);
        lat.win.reset(c0, r0, w, h);
        lat.win.add_ring(&lat.rx, &lat.ry, 1.0);
        let cov = lat.win.resolve();

        red.reset();
        let at = |k: usize| ((r0 + (k / w) as i64) * ncol + c0 + (k % w) as i64) as usize;
        for (k, &f) in cov.iter().enumerate() {
            if f >= 1.0 - TOL { img.reduce(red, at(k), 1.0, band_ok); }
        }
        for (k, &f) in cov.iter().enumerate() {
            if f > TOL && f < 1.0 - TOL { img.reduce(red, at(k), f as f32 as f64, band_ok); }
        }
        red.write(&mut row_out[c * img.nb..(c + 1) * img.nb]);
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
        let src_nodata = nodata.map_or(Nodata::None, |v| Nodata::Value(v.to_f64()));
        let opts = WarpOptions { alg, scaling: KernelScaling::Point, src_nodata, mask_band, ..Default::default() };
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
