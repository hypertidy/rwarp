//! Warp kernels: nearest-neighbour, bilinear, cubic, lanczos.
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

use crate::transform::{transform_scanline, Transformer};

/// Resampling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleAlg {
    NearestNeighbour,  // 1 pixel,  radius 0
    Bilinear,          // 2×2,      radius 1
    Cubic,             // 4×4,      radius 2  (Catmull-Rom, a = -0.5)
    Lanczos,           // 6×6,      radius 3  (sinc windowed sinc)
}

impl ResampleAlg {
    /// Filter radius in pixels (matches GWKGetFilterRadius).
    pub fn radius(self) -> i32 {
        match self {
            ResampleAlg::NearestNeighbour => 0,
            ResampleAlg::Bilinear => 1,
            ResampleAlg::Cubic => 2,
            ResampleAlg::Lanczos => 3,
        }
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

            // GDAL line 5304: truncation with epsilon
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

            // Quick bounds check: is the pixel centre even near the buffer?
            if buf_x < -0.5 || buf_x >= src_ncol as f64 + 0.5
                || buf_y < -0.5 || buf_y >= src_nrow as f64 + 0.5
            {
                continue;
            }

            if let Some(val) = sample_fn(src_pixels, src_ncol, src_nrow, buf_x, buf_y, nodata) {
                output[dst_row * dst_ncol + dst_col] = gdal_round(val);
            }
        }
    }

    output
}

/// GDAL-style rounding: round half away from zero, matching GWKRoundValueT<int16>.
#[inline]
fn gdal_round(v: f64) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
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
/// Separable 6×6 kernel (radius 3). Falls back to cubic at edges.
fn lanczos_sample(
    src: &[i32], ncol: usize, nrow: usize,
    buf_x: f64, buf_y: f64, nodata: i32,
) -> Option<f64> {
    let ix = (buf_x - 0.5).floor() as i64;
    let iy = (buf_y - 0.5).floor() as i64;
    let dx = buf_x - 0.5 - ix as f64;
    let dy = buf_y - 0.5 - iy as f64;

    // 6×6 neighbourhood: ix-2..ix+3, iy-2..iy+3
    let x_start = ix - 2;
    let y_start = iy - 2;
    if x_start < 0 || x_start + 5 >= ncol as i64
        || y_start < 0 || y_start + 5 >= nrow as i64
    {
        return cubic_sample(src, ncol, nrow, buf_x, buf_y, nodata);
    }

    // Compute and normalise weights
    let mut wx = [0.0f64; 6];
    let mut wy = [0.0f64; 6];
    let mut wx_sum = 0.0;
    let mut wy_sum = 0.0;
    for i in 0..6 {
        wx[i] = lanczos_weight(dx - (i as f64 - 2.0));
        wy[i] = lanczos_weight(dy - (i as f64 - 2.0));
        wx_sum += wx[i];
        wy_sum += wy[i];
    }
    if wx_sum.abs() < 1e-10 || wy_sum.abs() < 1e-10 {
        return None;
    }
    for w in &mut wx { *w /= wx_sum; }
    for w in &mut wy { *w /= wy_sum; }

    // Check for nodata, convolve
    let mut acc = 0.0;
    for jj in 0..6usize {
        let row_start = (y_start + jj as i64) as usize * ncol;
        let mut row_acc = 0.0;
        for ii in 0..6usize {
            let v = src[row_start + (x_start + ii as i64) as usize];
            if v == nodata {
                return cubic_sample(src, ncol, nrow, buf_x, buf_y, nodata);
            }
            row_acc += v as f64 * wx[ii];
        }
        acc += row_acc * wy[jj];
    }

    Some(acc)
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
}
