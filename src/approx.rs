//! Approximate transformer: adaptive interpolation of coordinate transforms.
//!
//! Maps GDAL's `GDALApproxTransform` + `GDALApproxTransformInternal`
//! (gdaltransformer.cpp, commit a3b7b01d3e, GDAL 3.13.0dev).
//!
//! ## How it works
//!
//! For a scanline of N destination pixels (all same y, increasing x):
//!   1. Transform 3 points exactly: first (S), middle (M), end (E)
//!   2. Compute linear interpolation slope from S to E
//!   3. Check: does the true M match the interpolated M?
//!   4. If error ≤ max_error → interpolate ALL points using slope from S→E
//!   5. If error > max_error → pre-transform midpoints of each half,
//!      then recurse on left [0..nMiddle) and right [nMiddle..nPoints)
//!
//! ## Key implementation details from GDAL source
//!
//! - Base case: `nPoints <= 5` → exact transform (not nPoints < 3)
//! - Precondition: all y values must be identical (it's a scanline), x must
//!   be non-degenerate. If violated → exact transform.
//! - Interpolation parameter: `dfDist = x[i] - x[0]` (actual input x distance,
//!   not array index fraction). For evenly-spaced pixels, equivalent to i/(n-1).
//! - Error metric: L1 in output space, using the slope-based interpolation.
//! - The recursion pre-computes the Start/Middle/End (SME) transformed values
//!   for each sub-call, passing them down to avoid redundant transforms.
//! - Split: `nMiddle = (nPoints-1)/2`. Left half: nMiddle points [0..nMiddle-1].
//!   Right half: nPoints-nMiddle points [nMiddle..nPoints-1]. No overlap.
//! - Separate forward/reverse error thresholds (we simplify to one for now).

use crate::transform::Transformer;

/// Approximate transformer wrapping an inner (exact) transformer.
pub struct ApproxTransformer<T: Transformer> {
    inner: T,
    max_error: f64,
}

impl<T: Transformer> ApproxTransformer<T> {
    pub fn new(inner: T, max_error: f64) -> Self {
        Self { inner, max_error }
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Entry point: check preconditions, transform SME, call internal.
    ///
    /// Maps to GDALApproxTransform() (line 4374).
    fn approx_entry(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
        success: &mut [bool],
        start: usize,
        len: usize,
    ) {
        // Base case: nPoints <= 5 → exact (line 4397)
        if len <= 5 {
            let ok = self.inner.transform(
                dst_to_src,
                &mut x[start..start + len],
                &mut y[start..start + len],
            );
            success[start..start + len].copy_from_slice(&ok);
            return;
        }

        let end = start + len - 1;
        let n_middle = (len - 1) / 2;
        let mid = start + n_middle;

        // Precondition checks (line 4393):
        // - All y values same (scanline)
        // - x values not degenerate
        // For robustness, we check start/mid/end only (like GDAL does)
        if y[start] != y[end] || y[start] != y[mid]
            || x[start] == x[end] || x[start] == x[mid]
        {
            let ok = self.inner.transform(
                dst_to_src,
                &mut x[start..start + len],
                &mut y[start..start + len],
            );
            success[start..start + len].copy_from_slice(&ok);
            return;
        }

        // Transform first, middle, last exactly (line 4407-4418)
        let mut tx = [x[start], x[mid], x[end]];
        let mut ty = [y[start], y[mid], y[end]];
        let ok3 = self.inner.transform(dst_to_src, &mut tx, &mut ty);

        if !ok3[0] || !ok3[1] || !ok3[2] {
            let ok = self.inner.transform(
                dst_to_src,
                &mut x[start..start + len],
                &mut y[start..start + len],
            );
            success[start..start + len].copy_from_slice(&ok);
            return;
        }

        // Call internal recursion with pre-computed SME
        self.approx_internal(dst_to_src, x, y, success, start, len, &tx, &ty);
    }

    /// Recursive core. Receives pre-computed Start/Middle/End transforms.
    ///
    /// Maps to GDALApproxTransformInternal() (line 4113).
    ///
    /// `sme_x` / `sme_y` are the transformed coords of [start, mid, end].
    fn approx_internal(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
        success: &mut [bool],
        start: usize,
        len: usize,
        sme_x: &[f64; 3],
        sme_y: &[f64; 3],
    ) {
        let n_middle = (len - 1) / 2;
        let mid = start + n_middle;
        let end = start + len - 1;

        // Compute slope for interpolation (line 4164-4169)
        // Uses actual input x distance, not array index
        let x_span = x[end] - x[start];
        let df_delta_x = (sme_x[2] - sme_x[0]) / x_span;
        let df_delta_y = (sme_y[2] - sme_y[0]) / x_span;

        // Error check at midpoint (line 4171-4175)
        // Interpolated midpoint vs exactly-transformed midpoint
        let mid_dist = x[mid] - x[start];
        let error =
            (sme_x[0] + df_delta_x * mid_dist - sme_x[1]).abs() +
            (sme_y[0] + df_delta_y * mid_dist - sme_y[1]).abs();

        if error > self.max_error {
            // Error too large: subdivide (line 4179+)
            //
            // GDAL pre-transforms the midpoints of each sub-half in a single
            // batch call to avoid redundant transforms. The three points are:
            //   xMiddle[0] = midpoint of left half  = x[(nMiddle-1)/2]
            //   xMiddle[1] = last of left half      = x[nMiddle-1]
            //   xMiddle[2] = midpoint of right half = x[nMiddle + (nPoints-nMiddle-1)/2]

            let left_len = n_middle;          // [start .. start + n_middle - 1]
            let right_len = len - n_middle;   // [start + n_middle .. end]

            // Check if sub-halves should use base transform (line 4195-4203)
            // This happens when they're too small or fail the scanline precondition
            let left_mid = start + (n_middle - 1) / 2;
            let right_mid = mid + (right_len - 1) / 2;

            let use_base_left = left_len <= 5
                || y[start] != y[start + left_len - 1]
                || y[start] != y[left_mid]
                || x[start] == x[start + left_len - 1]
                || x[start] == x[left_mid];

            let use_base_right = right_len <= 5
                || y[mid] != y[end]
                || y[mid] != y[right_mid]
                || x[mid] == x[end]
                || x[mid] == x[right_mid];

            // Pre-transform the 3 sub-midpoints (line 4205-4225)
            // Batch transform: up to 3 points depending on which halves need it
            let mut mx = [x[left_mid], x[start + left_len - 1], x[right_mid]];
            let mut my = [y[left_mid], y[start + left_len - 1], y[right_mid]];

            let mid_ok = if !use_base_left && !use_base_right {
                let ok = self.inner.transform(dst_to_src, &mut mx, &mut my);
                ok[0] && ok[1] && ok[2]
            } else if !use_base_left {
                let ok = self.inner.transform(dst_to_src, &mut mx[..2], &mut my[..2]);
                ok[0] && ok[1]
            } else if !use_base_right {
                let ok = self.inner.transform(dst_to_src, &mut mx[2..], &mut my[2..]);
                ok[0]
            } else {
                true // both halves use base transform, no midpoints needed
            };

            if !mid_ok {
                // Midpoint transforms failed: fall back to exact for interior,
                // write SME values directly (line 4229-4248)
                if left_len > 2 {
                    let ok = self.inner.transform(
                        dst_to_src,
                        &mut x[start + 1..start + left_len],
                        &mut y[start + 1..start + left_len],
                    );
                    for i in 0..left_len - 1 {
                        success[start + 1 + i] = ok[i];
                    }
                }
                if right_len > 2 {
                    let ok = self.inner.transform(
                        dst_to_src,
                        &mut x[mid + 1..end],
                        &mut y[mid + 1..end],
                    );
                    for i in 0..(right_len - 2) {
                        success[mid + 1 + i] = ok[i];
                    }
                }
                // Write the three known-good transforms
                x[start] = sme_x[0]; y[start] = sme_y[0]; success[start] = true;
                x[mid] = sme_x[1]; y[mid] = sme_y[1]; success[mid] = true;
                x[end] = sme_x[2]; y[end] = sme_y[2]; success[end] = true;
                return;
            }

            // Recurse on left half with pre-computed SME (line 4252-4268)
            if !use_base_left {
                let left_sme_x = [sme_x[0], mx[0], mx[1]];
                let left_sme_y = [sme_y[0], my[0], my[1]];
                self.approx_internal(
                    dst_to_src, x, y, success, start, left_len,
                    &left_sme_x, &left_sme_y,
                );
            } else {
                // Small left half: exact transform of interior + write endpoints
                if left_len > 2 {
                    let ok = self.inner.transform(
                        dst_to_src,
                        &mut x[start + 1..start + left_len],
                        &mut y[start + 1..start + left_len],
                    );
                    for i in 0..left_len - 1 {
                        success[start + 1 + i] = ok[i];
                    }
                }
                x[start] = sme_x[0];
                y[start] = sme_y[0];
                success[start] = true;
            }

            // Recurse on right half with pre-computed SME (line 4284-4298)
            if !use_base_right {
                let right_sme_x = [sme_x[1], mx[2], sme_x[2]];
                let right_sme_y = [sme_y[1], my[2], sme_y[2]];
                self.approx_internal(
                    dst_to_src, x, y, success, mid, right_len,
                    &right_sme_x, &right_sme_y,
                );
            } else {
                // Small right half: exact transform of interior + write endpoints
                if right_len > 2 {
                    let ok = self.inner.transform(
                        dst_to_src,
                        &mut x[mid + 1..end],
                        &mut y[mid + 1..end],
                    );
                    for i in 0..(right_len - 2) {
                        success[mid + 1 + i] = ok[i];
                    }
                }
                x[mid] = sme_x[1]; y[mid] = sme_y[1]; success[mid] = true;
                x[end] = sme_x[2]; y[end] = sme_y[2]; success[end] = true;
            }
        } else {
            // Error acceptable: linear interpolation (line 4332-4356)
            //
            // Uses slope * distance from start, NOT index fraction.
            // This is a single-segment interpolation from start to end.
            for i in (0..len).rev() {
                let df_dist = x[start + i] - x[start];
                x[start + i] = sme_x[0] + df_delta_x * df_dist;
                y[start + i] = sme_y[0] + df_delta_y * df_dist;
                success[start + i] = true;
            }
        }
    }
}

impl<T: Transformer> Transformer for ApproxTransformer<T> {
    fn transform(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
    ) -> Vec<bool> {
        let n = x.len();

        // Bypass if disabled
        if self.max_error <= 0.0 {
            return self.inner.transform(dst_to_src, x, y);
        }

        let mut success = vec![false; n];
        self.approx_entry(dst_to_src, x, y, &mut success, 0, n);
        success
    }
}
