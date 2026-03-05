//! GCP-based polynomial transformer.
//!
//! Matches GDAL's `GDALCreateGCPTransformer` / `GDALGCPTransform` from
//! `alg/gdal_crs.c`. Fits least-squares polynomials (order 1–3) from
//! Ground Control Points mapping pixel/line ↔ georeferenced coordinates.
//!
//! ## Polynomial terms
//!
//! - Order 1 (3 terms): `1, x, y`  (affine)
//! - Order 2 (6 terms): `1, x, y, x², xy, y²`
//! - Order 3 (10 terms): `1, x, y, x², xy, y², x³, x²y, xy², y³`
//!
//! ## Coordinate centring
//!
//! GDAL centres input coordinates by subtracting the mean before fitting.
//! This improves numerical conditioning when coordinates are large (e.g.
//! UTM eastings ~500000). We do the same: store the means and subtract
//! them before polynomial evaluation.
//!
//! ## Forward and reverse
//!
//! Both directions (pixel→geo and geo→pixel) are fit independently at
//! construction time. No iterative inversion at transform time.

use crate::Transformer;

/// Number of polynomial terms for each order.
fn n_terms(order: usize) -> usize {
    match order {
        1 => 3,
        2 => 6,
        3 => 10,
        _ => panic!("GCP polynomial order must be 1, 2, or 3"),
    }
}

/// Minimum number of GCPs required for each order.
fn min_gcps(order: usize) -> usize {
    n_terms(order)
}

/// Evaluate polynomial terms at (x, y) into `out`.
/// Terms are in GDAL order: 1, x, y, x², xy, y², x³, x²y, xy², y³
fn eval_terms(x: f64, y: f64, order: usize, out: &mut [f64]) {
    out[0] = 1.0;
    out[1] = x;
    out[2] = y;
    if order >= 2 {
        out[3] = x * x;
        out[4] = x * y;
        out[5] = y * y;
    }
    if order >= 3 {
        out[6] = x * x * x;
        out[7] = x * x * y;
        out[8] = x * y * y;
        out[9] = y * y * y;
    }
}

/// Evaluate a polynomial: dot product of coefficients and terms.
fn eval_poly(coeffs: &[f64], terms: &[f64]) -> f64 {
    coeffs.iter().zip(terms.iter()).map(|(c, t)| c * t).sum()
}

/// GCP polynomial transformer.
///
/// Holds forward (pixel/line → geo) and reverse (geo → pixel/line)
/// polynomial coefficients, plus the coordinate means for centring.
pub struct GcpTransformer {
    order: usize,

    // Forward: pixel/line → geo X/Y
    // Coefficients for centred (pixel - pixel_mean, line - line_mean)
    to_geo_x: Vec<f64>,
    to_geo_y: Vec<f64>,
    pixel_mean: f64,
    line_mean: f64,

    // Reverse: geo X/Y → pixel/line
    // Coefficients for centred (geo_x - geo_x_mean, geo_y - geo_y_mean)
    from_geo_x: Vec<f64>, // → pixel
    from_geo_y: Vec<f64>, // → line
    geo_x_mean: f64,
    geo_y_mean: f64,
}

impl GcpTransformer {
    /// Create a GCP polynomial transformer.
    ///
    /// `pixel`, `line`: source image coordinates of GCPs
    /// `geo_x`, `geo_y`: georeferenced coordinates of GCPs
    /// `order`: polynomial order (1, 2, or 3). Use 0 to auto-select
    ///   the highest order supported by the GCP count (max 2, matching GDAL).
    ///
    /// Returns `None` if the GCP count is insufficient or the system is
    /// singular.
    pub fn new(
        pixel: &[f64],
        line: &[f64],
        geo_x: &[f64],
        geo_y: &[f64],
        order: usize,
    ) -> Option<Self> {
        let n = pixel.len();
        assert_eq!(n, line.len());
        assert_eq!(n, geo_x.len());
        assert_eq!(n, geo_y.len());

        // Auto-select order if 0
        let order = if order == 0 {
            if n >= min_gcps(3) {
                // GDAL caps auto-select at 2, not 3
                2
            } else if n >= min_gcps(2) {
                2
            } else if n >= min_gcps(1) {
                1
            } else {
                return None;
            }
        } else {
            order
        };

        if n < min_gcps(order) {
            return None;
        }

        let nt = n_terms(order);

        // Compute means for centring
        let pixel_mean = pixel.iter().sum::<f64>() / n as f64;
        let line_mean = line.iter().sum::<f64>() / n as f64;
        let geo_x_mean = geo_x.iter().sum::<f64>() / n as f64;
        let geo_y_mean = geo_y.iter().sum::<f64>() / n as f64;

        // Forward fit: (pixel-mean, line-mean) → geo_x, geo_y
        let to_geo_x = fit_polynomial(
            pixel, line, geo_x, pixel_mean, line_mean, order, nt, n,
        )?;
        let to_geo_y = fit_polynomial(
            pixel, line, geo_y, pixel_mean, line_mean, order, nt, n,
        )?;

        // Reverse fit: (geo_x-mean, geo_y-mean) → pixel, line
        let from_geo_x = fit_polynomial(
            geo_x, geo_y, pixel, geo_x_mean, geo_y_mean, order, nt, n,
        )?;
        let from_geo_y = fit_polynomial(
            geo_x, geo_y, line, geo_x_mean, geo_y_mean, order, nt, n,
        )?;

        Some(GcpTransformer {
            order,
            to_geo_x,
            to_geo_y,
            pixel_mean,
            line_mean,
            from_geo_x,
            from_geo_y,
            geo_x_mean,
            geo_y_mean,
        })
    }

    /// Get the polynomial order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get forward coefficients (pixel/line → geo X).
    pub fn to_geo_x_coeffs(&self) -> &[f64] {
        &self.to_geo_x
    }

    /// Get forward coefficients (pixel/line → geo Y).
    pub fn to_geo_y_coeffs(&self) -> &[f64] {
        &self.to_geo_y
    }

    /// Get reverse coefficients (geo → pixel).
    pub fn from_geo_pixel_coeffs(&self) -> &[f64] {
        &self.from_geo_x
    }

    /// Get reverse coefficients (geo → line).
    pub fn from_geo_line_coeffs(&self) -> &[f64] {
        &self.from_geo_y
    }
}

impl Transformer for GcpTransformer {
    /// Transform coordinates.
    ///
    /// If `inverse` is false: pixel/line → geo (forward).
    /// If `inverse` is true: geo → pixel/line (reverse).
    ///
    /// Input x/y are modified in place. On failure (shouldn't happen for
    /// polynomial evaluation), values are set to NaN.
    fn transform(&self, inverse: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
    let nt = n_terms(self.order);
    let mut terms = vec![0.0; nt];
    let n = x.len();

    if !inverse {
        for i in 0..n {
            let cx = x[i] - self.pixel_mean;
            let cy = y[i] - self.line_mean;
            eval_terms(cx, cy, self.order, &mut terms);
            x[i] = eval_poly(&self.to_geo_x, &terms);
            y[i] = eval_poly(&self.to_geo_y, &terms);
        }
    } else {
        for i in 0..n {
            let cx = x[i] - self.geo_x_mean;
            let cy = y[i] - self.geo_y_mean;
            eval_terms(cx, cy, self.order, &mut terms);
            let new_x = eval_poly(&self.from_geo_x, &terms);
            let new_y = eval_poly(&self.from_geo_y, &terms);
            x[i] = new_x;
            y[i] = new_y;
        }
    }

    vec![true; n]
}
}

/// Fit a polynomial by least-squares.
///
/// Given input points `(in_x[i] - x_mean, in_y[i] - y_mean)` and target
/// values `out[i]`, solve the normal equations for the polynomial
/// coefficients.
///
/// Returns `None` if the system is singular.
fn fit_polynomial(
    in_x: &[f64],
    in_y: &[f64],
    out: &[f64],
    x_mean: f64,
    y_mean: f64,
    order: usize,
    nt: usize,
    n: usize,
) -> Option<Vec<f64>> {
    // Build normal equations: M * coeffs = rhs
    // M[i][j] = Σ_k term_i(k) * term_j(k)
    // rhs[i]  = Σ_k term_i(k) * out[k]

    let mut m = vec![0.0; nt * nt];
    let mut rhs = vec![0.0; nt];
    let mut terms = vec![0.0; nt];

    for k in 0..n {
        let cx = in_x[k] - x_mean;
        let cy = in_y[k] - y_mean;
        eval_terms(cx, cy, order, &mut terms);

        for i in 0..nt {
            rhs[i] += terms[i] * out[k];
            for j in 0..nt {
                m[i * nt + j] += terms[i] * terms[j];
            }
        }
    }

    // Solve by Gaussian elimination with partial pivoting
    solve_linear_system(&mut m, &mut rhs, nt)
}

/// Solve a linear system M * x = rhs in place.
///
/// M is nt×nt stored row-major in a flat vec. rhs is overwritten with
/// the solution. Returns `None` if the matrix is singular.
fn solve_linear_system(m: &mut [f64], rhs: &mut [f64], nt: usize) -> Option<Vec<f64>> {
    // Forward elimination with partial pivoting
    for col in 0..nt {
        // Find pivot
        let mut max_val = m[col * nt + col].abs();
        let mut max_row = col;
        for row in (col + 1)..nt {
            let val = m[row * nt + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..nt {
                m.swap(col * nt + j, max_row * nt + j);
            }
            rhs.swap(col, max_row);
        }

        // Eliminate below
        let pivot = m[col * nt + col];
        for row in (col + 1)..nt {
            let factor = m[row * nt + col] / pivot;
            for j in col..nt {
                m[row * nt + j] -= factor * m[col * nt + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut result = vec![0.0; nt];
    for col in (0..nt).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..nt {
            sum -= m[col * nt + j] * result[j];
        }
        result[col] = sum / m[col * nt + col];
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_roundtrip() {
        // A pure affine transform should be recovered exactly by order 1.
        // gt: x = 300000 + col*10 + row*0
        //     y = 5200000 + col*0 + row*(-10)
        let cols: Vec<f64> = vec![0.0, 100.0, 200.0, 0.0, 100.0, 200.0];
        let rows: Vec<f64> = vec![0.0, 0.0, 0.0, 100.0, 100.0, 100.0];
        let geo_x: Vec<f64> = cols.iter().map(|c| 300000.0 + c * 10.0).collect();
        let geo_y: Vec<f64> = rows.iter().map(|r| 5200000.0 + r * -10.0).collect();

        let t = GcpTransformer::new(&cols, &rows, &geo_x, &geo_y, 1).unwrap();

        // Forward: pixel/line → geo
        let mut px = vec![50.0, 150.0];
        let mut py = vec![50.0, 75.0];
        t.transform(false, &mut px, &mut py);
        assert!((px[0] - 300500.0).abs() < 1e-6);
        assert!((py[0] - 5199500.0).abs() < 1e-6);
        assert!((px[1] - 301500.0).abs() < 1e-6);
        assert!((py[1] - 5199250.0).abs() < 1e-6);

        // Reverse: geo → pixel/line
        let mut gx = vec![300500.0, 301500.0];
        let mut gy = vec![5199500.0, 5199250.0];
        t.transform(true, &mut gx, &mut gy);
        assert!((gx[0] - 50.0).abs() < 1e-6);
        assert!((gy[0] - 50.0).abs() < 1e-6);
        assert!((gx[1] - 150.0).abs() < 1e-6);
        assert!((gy[1] - 75.0).abs() < 1e-6);
    }

    #[test]
    fn test_order2_quadratic() {
        // Fit a known quadratic: geo_x = 100 + 2*col + 0.001*col²
        let mut cols = Vec::new();
        let mut rows = Vec::new();
        let mut geo_x = Vec::new();
        let mut geo_y = Vec::new();
        for c in (0..=200).step_by(40) {
            for r in (0..=200).step_by(40) {
                let cf = c as f64;
                let rf = r as f64;
                cols.push(cf);
                rows.push(rf);
                geo_x.push(100.0 + 2.0 * cf + 0.001 * cf * cf);
                geo_y.push(200.0 + 3.0 * rf + 0.0005 * rf * rf);
            }
        }

        let t = GcpTransformer::new(&cols, &rows, &geo_x, &geo_y, 2).unwrap();

        // Test at a point not in the training set
        let mut px = vec![73.0];
        let mut py = vec![117.0];
        let expected_x = 100.0 + 2.0 * 73.0 + 0.001 * 73.0 * 73.0;
        let expected_y = 200.0 + 3.0 * 117.0 + 0.0005 * 117.0 * 117.0;
        t.transform(false, &mut px, &mut py);
        assert!((px[0] - expected_x).abs() < 1e-6, "got {} expected {}", px[0], expected_x);
        assert!((py[0] - expected_y).abs() < 1e-6, "got {} expected {}", py[0], expected_y);
    }

    #[test]
    fn test_auto_order() {
        // With 6 GCPs, auto (0) should select order 2
        let cols: Vec<f64> = vec![0.0, 100.0, 200.0, 0.0, 100.0, 200.0];
        let rows: Vec<f64> = vec![0.0, 0.0, 0.0, 100.0, 100.0, 100.0];
        let geo_x: Vec<f64> = cols.iter().map(|c| 300000.0 + c * 10.0).collect();
        let geo_y: Vec<f64> = rows.iter().map(|r| 5200000.0 + r * -10.0).collect();

        let t = GcpTransformer::new(&cols, &rows, &geo_x, &geo_y, 0).unwrap();
        assert_eq!(t.order(), 2);
    }

    #[test]
    fn test_insufficient_gcps() {
        // 2 GCPs is not enough for any order
        let cols = vec![0.0, 100.0];
        let rows = vec![0.0, 100.0];
        let geo_x = vec![300000.0, 301000.0];
        let geo_y = vec![5200000.0, 5199000.0];

        assert!(GcpTransformer::new(&cols, &rows, &geo_x, &geo_y, 1).is_none());
        assert!(GcpTransformer::new(&cols, &rows, &geo_x, &geo_y, 0).is_none());
    }
}
