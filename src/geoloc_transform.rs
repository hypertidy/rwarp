//! Geolocation-array transformer: pixel/line <-> georeferenced coordinates
//! through 2D coordinate arrays (curvilinear grids, satellite swaths).
//!
//! Follows GDAL's geolocation transformer (`alg/gdalgeoloc.cpp`, read at
//! GDAL master, October 2026; the logic has been stable since GDAL 3.5):
//!
//! | rwarp | GDAL |
//! |-------|------|
//! | [`GeolocArrays::pixel_line_to_xy`] | `GDALGeoLoc::PixelLineToXY` (double overload) |
//! | [`GeolocArrays::pixel_to_geo`] | `GDALGeoLoc::Transform`, `bDstToSrc = FALSE` |
//! | [`GeolocArrays::geo_to_pixel`] | `GDALGeoLoc::Transform`, `bDstToSrc = TRUE`, max-accuracy path |
//! | `inverse_bilinear` | `GDALInverseBilinearInterpolation` |
//! | `shift_geo_x` / `unshift_geo_x` | `ShiftGeoX` / `UnshiftGeoX` |
//!
//! ## The model
//!
//! Each geolocation sample is a **point**. Between four neighbouring
//! samples the position is interpolated bilinearly; beyond the outermost
//! samples it is extrapolated linearly from the closest grid square. Where a
//! sample sits relative to its pixel is set by [`GeolocConvention`]:
//!
//! - [`GeolocConvention::PixelCenter`]: samples are cell centres. Cell edges
//!   then fall at the arithmetic midpoint of neighbouring centres, in the
//!   geolocation arrays' own coordinates, the outer edges half a cell beyond
//!   the outermost centres, and each cell corner at the mean of four centres.
//! - [`GeolocConvention::TopLeftCorner`]: samples are top-left pixel corners.
//!   **This is GDAL's default** when `GEOREFERENCING_CONVENTION` is not set,
//!   so centre coordinates (as in CF) read through the default land half a
//!   cell off.
//!
//! Cell edges are straight lines in the geolocation coordinates (bilinear
//! interpolation is linear along each edge); they curve only after a CRS
//! transform.
//!
//! ## Inverse (geo -> pixel)
//!
//! GDAL first takes an approximate answer from a "backmap" raster, then
//! searches a few grid squares around it for the quadrilateral of four
//! samples that contains the point, and solves the exact inverse bilinear
//! interpolation there. rwarp skips the backmap: it indexes every
//! quadrilateral (including one ring of extrapolated quadrilaterals outside
//! the arrays) in a uniform bucket grid, takes the first containing
//! quadrilateral in row-major order, and solves the same exact inverse.
//!
//! For grids whose quadrilaterals do not overlap, the containing
//! quadrilateral is unique except on shared edges, where the inverse is
//! continuous, so the result is the same as GDAL's. Differences can appear
//! only where quadrilaterals overlap (folded swaths), where GDAL takes the
//! first match in a spiral search from its backmap estimate and rwarp takes
//! the first in row-major order, and where GDAL's limited search radius (3
//! squares, 5 near the poles) fails to find a quadrilateral that rwarp's
//! index finds. GDAL's approximate-only path
//! (`GDAL_GEOLOC_USE_MAX_ACCURACY=NO`) is not reproduced.
//!
//! ## Other departures from GDAL
//!
//! - NaN samples are treated as missing, as well as samples equal to the
//!   nodata value. GDAL compares only against the nodata value.
//! - Only the X array is checked for nodata, as GDAL does.
//! - `SWAP_XY` and the quadtree inverse (`GEOLOC_USE_QUADTREE`) are not
//!   implemented.
//! - Whether longitudes wrap at the antimeridian is an explicit option
//!   ([`GeolocOptions::lon_wrap`]). GDAL decides it from the SRS: geographic,
//!   with every longitude in `[-180, 180]`.

use crate::crs::{crs_pair, CrsTransform};
use crate::Transformer;
use vaster::inv_geotransform;

/// Where each geolocation sample sits relative to its pixel.
///
/// Maps GDAL's `GEOREFERENCING_CONVENTION` metadata item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GeolocConvention {
    /// Samples are top-left pixel corners. GDAL's default.
    #[default]
    TopLeftCorner,
    /// Samples are pixel centres, as for CF coordinate variables.
    PixelCenter,
}

impl GeolocConvention {
    /// GDAL's `dfGeorefConventionOffset`.
    fn offset(self) -> f64 {
        match self {
            GeolocConvention::TopLeftCorner => 0.0,
            GeolocConvention::PixelCenter => 0.5,
        }
    }
}

/// Options describing how geolocation arrays relate to the raster.
///
/// The defaults match GDAL's defaults: offsets 0, steps 1,
/// [`GeolocConvention::TopLeftCorner`], no nodata, no longitude wrapping.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GeolocOptions {
    /// `PIXEL_OFFSET`: raster pixel of geolocation column 0.
    pub pixel_offset: f64,
    /// `PIXEL_STEP`: raster pixels per geolocation column.
    pub pixel_step: f64,
    /// `LINE_OFFSET`: raster line of geolocation row 0.
    pub line_offset: f64,
    /// `LINE_STEP`: raster lines per geolocation row.
    pub line_step: f64,
    /// `GEOREFERENCING_CONVENTION`.
    pub convention: GeolocConvention,
    /// Nodata value of the X array. NaN is always treated as missing.
    pub nodata_x: Option<f64>,
    /// Longitudes in `[-180, 180]` that may cross the antimeridian.
    /// GDAL sets this when the geolocation SRS is geographic and every
    /// longitude lies in `[-180, 180]`.
    pub lon_wrap: bool,
}

impl Default for GeolocOptions {
    fn default() -> Self {
        Self {
            pixel_offset: 0.0,
            pixel_step: 1.0,
            line_offset: 0.0,
            line_step: 1.0,
            convention: GeolocConvention::TopLeftCorner,
            nodata_x: None,
            lon_wrap: false,
        }
    }
}

impl GeolocOptions {
    /// `PIXEL_OFFSET` and `PIXEL_STEP`.
    pub fn with_pixel(mut self, offset: f64, step: f64) -> Self {
        self.pixel_offset = offset;
        self.pixel_step = step;
        self
    }

    /// `LINE_OFFSET` and `LINE_STEP`.
    pub fn with_line(mut self, offset: f64, step: f64) -> Self {
        self.line_offset = offset;
        self.line_step = step;
        self
    }

    pub fn with_convention(mut self, convention: GeolocConvention) -> Self {
        self.convention = convention;
        self
    }

    pub fn with_nodata_x(mut self, nodata_x: Option<f64>) -> Self {
        self.nodata_x = nodata_x;
        self
    }

    pub fn with_lon_wrap(mut self, lon_wrap: bool) -> Self {
        self.lon_wrap = lon_wrap;
        self
    }
}

/// One quadrilateral of four neighbouring samples, in GDAL's corner naming:
/// 0 = (ix, iy), 1 = (ix, iy + 1), 2 = (ix + 1, iy), 3 = (ix + 1, iy + 1).
#[derive(Debug, Clone, Copy)]
struct Quad {
    ix: i64,
    iy: i64,
    x: [f64; 4],
    y: [f64; 4],
}

impl Quad {
    fn bbox(&self) -> (f64, f64, f64, f64) {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for k in 0..4 {
            x0 = x0.min(self.x[k]);
            x1 = x1.max(self.x[k]);
            y0 = y0.min(self.y[k]);
            y1 = y1.max(self.y[k]);
        }
        (x0, y0, x1, y1)
    }

    /// Point in the ring (0, 2, 3, 1), or on its boundary.
    ///
    /// Matches the effect of GDAL's `isPointInRing || isPointOnRingBoundary`
    /// on the ring `x0, x2, x3, x1, x0`.
    fn contains(&self, px: f64, py: f64) -> bool {
        let order = [0usize, 2, 3, 1];
        let scale = {
            let (a, b, c, d) = self.bbox();
            (c - a).abs().max((d - b).abs()).max(f64::MIN_POSITIVE)
        };
        let eps = 1e-12 * scale;
        let mut inside = false;
        for k in 0..4 {
            let (i, j) = (order[k], order[(k + 1) % 4]);
            let (xi, yi, xj, yj) = (self.x[i], self.y[i], self.x[j], self.y[j]);
            // On the segment?
            let cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi);
            let len = ((xj - xi).powi(2) + (yj - yi).powi(2)).sqrt();
            if cross.abs() <= eps * len.max(scale)
                && px >= xi.min(xj) - eps
                && px <= xi.max(xj) + eps
                && py >= yi.min(yj) - eps
                && py <= yi.max(yj) + eps
            {
                return true;
            }
            // Crossing number.
            if (yi > py) != (yj > py) {
                let x_at = xi + (py - yi) * (xj - xi) / (yj - yi);
                if px < x_at {
                    inside = !inside;
                }
            }
        }
        inside
    }
}

/// A uniform bucket grid over quadrilateral bounding boxes.
#[derive(Debug, Clone)]
struct QuadIndex {
    quads: Vec<Quad>,
    x0: f64,
    y0: f64,
    cell_w: f64,
    cell_h: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
}

impl QuadIndex {
    fn new(quads: Vec<Quad>) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for q in &quads {
            let (a, b, c, d) = q.bbox();
            x0 = x0.min(a);
            y0 = y0.min(b);
            x1 = x1.max(c);
            y1 = y1.max(d);
        }
        let n = (quads.len() as f64).sqrt().ceil().max(1.0) as usize;
        let nx = n.min(4096);
        let ny = n.min(4096);
        let cell_w = ((x1 - x0) / nx as f64).max(f64::MIN_POSITIVE);
        let cell_h = ((y1 - y0) / ny as f64).max(f64::MIN_POSITIVE);
        let mut buckets = vec![Vec::new(); nx * ny];
        if !quads.is_empty() {
            for (k, q) in quads.iter().enumerate() {
                let (a, b, c, d) = q.bbox();
                let (ca, cb) = Self::cell_of(a, b, x0, y0, cell_w, cell_h, nx, ny);
                let (cc, cd) = Self::cell_of(c, d, x0, y0, cell_w, cell_h, nx, ny);
                for cy in cb..=cd {
                    for cx in ca..=cc {
                        buckets[cy * nx + cx].push(k as u32);
                    }
                }
            }
        }
        Self { quads, x0, y0, cell_w, cell_h, nx, ny, buckets }
    }

    #[allow(clippy::too_many_arguments)]
    fn cell_of(
        x: f64,
        y: f64,
        x0: f64,
        y0: f64,
        cell_w: f64,
        cell_h: f64,
        nx: usize,
        ny: usize,
    ) -> (usize, usize) {
        let cx = (((x - x0) / cell_w).floor().max(0.0) as usize).min(nx - 1);
        let cy = (((y - y0) / cell_h).floor().max(0.0) as usize).min(ny - 1);
        (cx, cy)
    }

    /// The first quadrilateral, in row-major (iy, ix) order, containing the
    /// point. Buckets hold quadrilateral numbers in insertion order, which
    /// is row-major.
    fn find(&self, px: f64, py: f64) -> Option<&Quad> {
        if self.quads.is_empty() || !px.is_finite() || !py.is_finite() {
            return None;
        }
        let tol_x = self.cell_w * 1e-9;
        let tol_y = self.cell_h * 1e-9;
        let x_hi = self.x0 + self.cell_w * self.nx as f64;
        let y_hi = self.y0 + self.cell_h * self.ny as f64;
        if px < self.x0 - tol_x || px > x_hi + tol_x || py < self.y0 - tol_y || py > y_hi + tol_y {
            return None;
        }
        let (cx, cy) = Self::cell_of(
            px, py, self.x0, self.y0, self.cell_w, self.cell_h, self.nx, self.ny,
        );
        self.buckets[cy * self.nx + cx]
            .iter()
            .map(|&k| &self.quads[k as usize])
            .find(|q| q.contains(px, py))
    }
}

/// Geolocation arrays for one raster, with a lookup structure for the
/// inverse transform.
///
/// `x` and `y` are row-major, `ncol * nrow` values each: `x[iy * ncol + ix]`.
#[derive(Debug, Clone)]
pub struct GeolocArrays {
    x: Vec<f64>,
    y: Vec<f64>,
    ncol: usize,
    nrow: usize,
    opts: GeolocOptions,
    index: QuadIndex,
}

impl GeolocArrays {
    /// Build from row-major X and Y arrays of `ncol * nrow` values.
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        ncol: usize,
        nrow: usize,
        opts: GeolocOptions,
    ) -> Result<Self, String> {
        if ncol < 2 || nrow < 2 {
            return Err(format!(
                "geolocation arrays must be at least 2x2, got {ncol}x{nrow}"
            ));
        }
        if x.len() != ncol * nrow || y.len() != ncol * nrow {
            return Err(format!(
                "geolocation arrays have {} and {} values, expected {}",
                x.len(),
                y.len(),
                ncol * nrow
            ));
        }
        if opts.pixel_step == 0.0 || opts.line_step == 0.0 {
            return Err("PIXEL_STEP and LINE_STEP must be non-zero".to_string());
        }
        let mut g = Self {
            x,
            y,
            ncol,
            nrow,
            opts,
            index: QuadIndex::new(Vec::new()),
        };
        g.index = QuadIndex::new(g.build_quads());
        Ok(g)
    }

    /// The options these arrays were built with.
    pub fn options(&self) -> &GeolocOptions {
        &self.opts
    }

    /// Size of the geolocation arrays, `(ncol, nrow)`.
    pub fn size(&self) -> (usize, usize) {
        (self.ncol, self.nrow)
    }

    fn is_missing(&self, gx: f64) -> bool {
        gx.is_nan() || self.opts.nodata_x == Some(gx)
    }

    fn get(&self, ix: usize, iy: usize) -> (f64, f64) {
        let k = iy * self.ncol + ix;
        (self.x[k], self.y[k])
    }

    fn shift_geo_x(&self, x_ref: f64, x: f64) -> f64 {
        if !self.opts.lon_wrap {
            return x;
        }
        if x_ref < -170.0 && x > 170.0 {
            return x - 360.0;
        }
        if x_ref > 170.0 && x < -170.0 {
            return x + 360.0;
        }
        x
    }

    fn unshift_geo_x(&self, x: f64) -> f64 {
        if !self.opts.lon_wrap {
            return x;
        }
        if !(-180.0..=180.0).contains(&x) {
            let r = (x + 180.0) % 360.0;
            // C fmod keeps the sign of the dividend, as Rust's % does.
            return if r < 0.0 { r + 180.0 } else { r - 180.0 };
        }
        x
    }

    /// Georeferenced position at a fractional position in the geolocation
    /// arrays (column, row), bilinearly interpolated, extrapolated beyond
    /// the outermost samples from the closest grid square.
    ///
    /// Follows `GDALGeoLoc::PixelLineToXY` (double overload). `None` where a
    /// needed sample is missing.
    pub fn pixel_line_to_xy(&self, gpix: f64, gline: f64) -> Option<(f64, f64)> {
        let nx = self.ncol as i64;
        let ny = self.nrow as i64;
        // C's static_cast<int> of a clamped non-negative double truncates.
        let mut ix = gpix.clamp(0.0, (nx - 1) as f64) as i64;
        let mut iy = gline.clamp(0.0, (ny - 1) as f64) as i64;

        for _attempt in 0..2 {
            let (x00, y00) = self.get(ix as usize, iy as usize);
            if self.is_missing(x00) {
                return None;
            }
            if ix + 1 < nx && iy + 1 < ny {
                let (x10, y10) = self.get(ix as usize + 1, iy as usize);
                let (x01, y01) = self.get(ix as usize, iy as usize + 1);
                let (x11, y11) = self.get(ix as usize + 1, iy as usize + 1);
                if !self.is_missing(x10) && !self.is_missing(x01) && !self.is_missing(x11) {
                    let x10a = self.shift_geo_x(x00, x10);
                    let x01a = self.shift_geo_x(x00, x01);
                    let x11a = self.shift_geo_x(x00, x11);
                    let fx = gpix - ix as f64;
                    let fy = gline - iy as f64;
                    let gx = (1.0 - fy) * (x00 + fx * (x10a - x00))
                        + fy * (x01a + fx * (x11a - x01a));
                    let gy = (1.0 - fy) * (y00 + fx * (y10 - y00))
                        + fy * (y01 + fx * (y11 - y01));
                    return Some((self.unshift_geo_x(gx), gy));
                }
            }

            if ix == nx - 1 && ix >= 1 && iy + 1 < ny {
                // After the right edge: go one square left and retry.
                ix -= 1;
                continue;
            } else if iy == ny - 1 && iy >= 1 && ix + 1 < nx {
                // After the bottom edge: go one square up and retry.
                iy -= 1;
                continue;
            } else if ix == nx - 1 && iy == ny - 1 && ix >= 1 && iy >= 1 {
                // After both: go one square left and up and retry.
                ix -= 1;
                iy -= 1;
                continue;
            } else if ix + 1 < nx && !self.is_missing(self.get(ix as usize + 1, iy as usize).0) {
                let (x10, y10) = self.get(ix as usize + 1, iy as usize);
                let fx = gpix - ix as f64;
                let gx = x00 + fx * (self.shift_geo_x(x00, x10) - x00);
                let gy = y00 + fx * (y10 - y00);
                return Some((self.unshift_geo_x(gx), gy));
            } else if iy + 1 < ny && !self.is_missing(self.get(ix as usize, iy as usize + 1).0) {
                let (x01, y01) = self.get(ix as usize, iy as usize + 1);
                let fy = gline - iy as f64;
                let gx = x00 + fy * (self.shift_geo_x(x00, x01) - x00);
                let gy = y00 + fy * (y01 - y00);
                return Some((self.unshift_geo_x(gx), gy));
            } else {
                return Some((self.unshift_geo_x(x00), y00));
            }
        }
        // Both attempts used: GDAL leaves the output untouched and still
        // reports success here, which cannot happen for arrays of at least
        // 2x2 without missing values; report failure instead.
        None
    }

    /// Integer overload: the sample itself inside the arrays, otherwise
    /// the extrapolated position. Follows `GDALGeoLoc::PixelLineToXY`
    /// (int overload).
    fn pixel_line_to_xy_int(&self, ix: i64, iy: i64) -> Option<(f64, f64)> {
        if ix >= 0 && (ix as usize) < self.ncol && iy >= 0 && (iy as usize) < self.nrow {
            let (gx, gy) = self.get(ix as usize, iy as usize);
            if self.is_missing(gx) {
                return None;
            }
            return Some((self.unshift_geo_x(gx), gy));
        }
        self.pixel_line_to_xy(ix as f64, iy as f64)
    }

    /// Every quadrilateral GDAL's inverse search can reach: columns and rows
    /// `-1 ..= n - 1`, one ring beyond the arrays, in row-major order.
    /// A quadrilateral crossing the antimeridian is entered twice, shifted
    /// to each side, as GDAL tests both.
    fn build_quads(&self) -> Vec<Quad> {
        let mut quads = Vec::new();
        for iy in -1..(self.nrow as i64) {
            for ix in -1..(self.ncol as i64) {
                let c0 = self.pixel_line_to_xy_int(ix, iy);
                let c2 = self.pixel_line_to_xy_int(ix + 1, iy);
                let c1 = self.pixel_line_to_xy_int(ix, iy + 1);
                let c3 = self.pixel_line_to_xy_int(ix + 1, iy + 1);
                let (Some(c0), Some(c1), Some(c2), Some(c3)) = (c0, c1, c2, c3) else {
                    continue;
                };
                let mut q = Quad {
                    ix,
                    iy,
                    x: [c0.0, c1.0, c2.0, c3.0],
                    y: [c0.1, c1.1, c2.1, c3.1],
                };
                let crosses = self.opts.lon_wrap
                    && q.x.iter().all(|v| v.abs() > 170.0)
                    && ((q.x[1] - q.x[0]).abs() > 180.0
                        || (q.x[2] - q.x[0]).abs() > 180.0
                        || (q.x[3] - q.x[0]).abs() > 180.0);
                if crosses {
                    for v in q.x.iter_mut() {
                        if *v > 0.0 {
                            *v -= 360.0;
                        }
                    }
                    quads.push(q);
                    for v in q.x.iter_mut() {
                        *v += 360.0;
                    }
                }
                quads.push(q);
            }
        }
        quads
    }

    fn convention_offset(&self) -> f64 {
        self.opts.convention.offset()
    }

    /// Raster pixel/line to georeferenced coordinates.
    ///
    /// Follows `GDALGeoLoc::Transform` with `bDstToSrc = FALSE`. A raster
    /// position is pixel/line space (pixel `i` spans `[i, i + 1)`).
    pub fn pixel_to_geo(&self, pixel: f64, line: f64) -> Option<(f64, f64)> {
        if !pixel.is_finite() || !line.is_finite() {
            return None;
        }
        let off = self.convention_offset();
        let gpix = (pixel - self.opts.pixel_offset) / self.opts.pixel_step - off;
        let gline = (line - self.opts.line_offset) / self.opts.line_step - off;
        self.pixel_line_to_xy(gpix, gline)
    }

    /// Georeferenced coordinates to raster pixel/line.
    ///
    /// Follows `GDALGeoLoc::Transform` with `bDstToSrc = TRUE` on its
    /// max-accuracy path: the containing quadrilateral, then the exact
    /// inverse bilinear interpolation. See the module docs for how the
    /// quadrilateral is found.
    pub fn geo_to_pixel(&self, gx: f64, gy: f64) -> Option<(f64, f64)> {
        let q = self.index.find(gx, gy)?;
        let (mut i, mut j) = (q.ix as f64, q.iy as f64);
        inverse_bilinear(
            gx, gy, q.x[0], q.y[0], q.x[1], q.y[1], q.x[2], q.y[2], q.x[3], q.y[3], &mut i,
            &mut j,
        );
        let off = self.convention_offset();
        let pixel = (i + off) * self.opts.pixel_step + self.opts.pixel_offset;
        let line = (j + off) * self.opts.line_step + self.opts.line_offset;
        if pixel.is_finite() && line.is_finite() {
            Some((pixel, line))
        } else {
            None
        }
    }
}

/// On its own, a [`GeolocArrays`] is a transformer between raster
/// pixel/line ("source") and the geolocation arrays' coordinates
/// ("destination"), as GDAL's geolocation transformer is.
impl Transformer for GeolocArrays {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let mut ok = vec![false; x.len()];
        for k in 0..x.len() {
            let r = if dst_to_src {
                self.geo_to_pixel(x[k], y[k])
            } else {
                self.pixel_to_geo(x[k], y[k])
            };
            if let Some((a, b)) = r {
                x[k] = a;
                y[k] = b;
                ok[k] = true;
            } else {
                x[k] = f64::INFINITY;
                y[k] = f64::INFINITY;
            }
        }
        ok
    }
}

/// Exact inverse bilinear interpolation.
///
/// Follows `GDALInverseBilinearInterpolation`. On entry `(i, j)` are the
/// array coordinates of corner 0; on exit, those of `(x, y)`. Corners are
/// 0 = (i, j), 1 = (i, j + 1), 2 = (i + 1, j), 3 = (i + 1, j + 1), and
/// `(x, y)` is assumed to lie in the ring (0, 2, 3, 1).
#[allow(clippy::too_many_arguments)]
fn inverse_bilinear(
    x: f64,
    y: f64,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    i: &mut f64,
    j: &mut f64,
) {
    let a = (x0 - x) * (y0 - y2) - (y0 - y) * (x0 - x2);
    let b = (((x0 - x) * (y1 - y3) - (y0 - y) * (x1 - x3))
        + ((x1 - x) * (y0 - y2) - (y1 - y) * (x0 - x2)))
        / 2.0;
    let c = (x1 - x) * (y1 - y3) - (y1 - y) * (x1 - x3);
    let denom = a - 2.0 * b + c;
    let magnitude = a.abs() + b.abs() + c.abs();
    let s = if denom.abs() <= 1e-12 * magnitude {
        // Typically a rectangle, or a triangle.
        a / (a - c)
    } else {
        let sqrt_term = (b * b - a * c).sqrt();
        let s1 = ((a - b) + sqrt_term) / denom;
        let s2 = ((a - b) - sqrt_term) / denom;
        if !(0.0..=1.0).contains(&s1) {
            s2
        } else {
            s1
        }
    };

    let t_denom_x = (1.0 - s) * (x0 - x2) + s * (x1 - x3);
    if t_denom_x.abs() > 1e-12 * magnitude {
        *i += ((1.0 - s) * (x0 - x) + s * (x1 - x)) / t_denom_x;
    } else {
        let t_denom_y = (1.0 - s) * (y0 - y2) + s * (y1 - y3);
        if t_denom_y.abs() > 1e-12 * magnitude {
            *i += ((1.0 - s) * (y0 - y) + s * (y1 - y)) / t_denom_y;
        }
    }
    *j += s;
}

// ---------------------------------------------------------------------------
// Composing with a CRS transform
// ---------------------------------------------------------------------------

/// How one side of a warp maps raster pixel/line to georeferenced
/// coordinates in its CRS.
#[derive(Debug, Clone)]
pub enum PixelGeo {
    /// An affine geotransform (GDAL order).
    Affine { gt: [f64; 6], inv_gt: [f64; 6] },
    /// Geolocation arrays.
    Geoloc(GeolocArrays),
}

impl PixelGeo {
    /// An affine side. Fails if the geotransform is not invertible.
    pub fn affine(gt: [f64; 6]) -> Result<Self, String> {
        let inv_gt = inv_geotransform(&gt)
            .ok_or_else(|| "geotransform not invertible".to_string())?;
        Ok(PixelGeo::Affine { gt, inv_gt })
    }

    fn to_geo(&self, pixel: f64, line: f64) -> Option<(f64, f64)> {
        match self {
            PixelGeo::Affine { gt, .. } => Some((
                gt[0] + pixel * gt[1] + line * gt[2],
                gt[3] + pixel * gt[4] + line * gt[5],
            )),
            PixelGeo::Geoloc(g) => g.pixel_to_geo(pixel, line),
        }
    }

    fn to_pixel(&self, gx: f64, gy: f64) -> Option<(f64, f64)> {
        match self {
            PixelGeo::Affine { inv_gt, .. } => Some((
                inv_gt[0] + gx * inv_gt[1] + gy * inv_gt[2],
                inv_gt[3] + gx * inv_gt[4] + gy * inv_gt[5],
            )),
            PixelGeo::Geoloc(g) => g.geo_to_pixel(gx, gy),
        }
    }
}

/// The pixel <-> geo <-> CRS <-> geo <-> pixel pipeline, where either side
/// may be geolocation arrays rather than a geotransform.
///
/// This is GDAL's `GDALGenImgProjTransform` with a geolocation transformer
/// on one or both sides: a curvilinear source (`gdalwarp -geoloc`, or
/// `SRC_METHOD=GEOLOC_ARRAY`) and/or a curvilinear target
/// (`DST_METHOD=GEOLOC_ARRAY` with `DST_GEOLOC_ARRAY`).
///
/// Geolocation coordinates are in the side's CRS, longitude/latitude order
/// for geographic CRSs (see [`crate::crs`]).
pub struct GenImgGeolocTransformer {
    src: PixelGeo,
    dst: PixelGeo,
    crs_dst_to_src: Box<dyn CrsTransform>,
    crs_src_to_dst: Box<dyn CrsTransform>,
}

impl GenImgGeolocTransformer {
    /// Build with the compiled-in CRS backend.
    pub fn new(src: PixelGeo, src_crs: &str, dst: PixelGeo, dst_crs: &str) -> Result<Self, String> {
        let (crs_dst_to_src, crs_src_to_dst) = crs_pair(src_crs, dst_crs)?;
        Ok(Self::with_backend(src, dst, crs_dst_to_src, crs_src_to_dst))
    }

    /// Build with caller-supplied CRS transforms (`dst -> src`, `src -> dst`).
    pub fn with_backend(
        src: PixelGeo,
        dst: PixelGeo,
        crs_dst_to_src: Box<dyn CrsTransform>,
        crs_src_to_dst: Box<dyn CrsTransform>,
    ) -> Self {
        Self { src, dst, crs_dst_to_src, crs_src_to_dst }
    }
}

impl Transformer for GenImgGeolocTransformer {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (from, to, crs) = if dst_to_src {
            (&self.dst, &self.src, &self.crs_dst_to_src)
        } else {
            (&self.src, &self.dst, &self.crs_src_to_dst)
        };
        let mut ok = vec![false; x.len()];
        for k in 0..x.len() {
            let r = from
                .to_geo(x[k], y[k])
                .and_then(|(gx, gy)| crs.convert(gx, gy))
                .and_then(|(gx, gy)| to.to_pixel(gx, gy));
            if let Some((a, b)) = r {
                x[k] = a;
                y[k] = b;
                ok[k] = true;
            } else {
                x[k] = f64::INFINITY;
                y[k] = f64::INFINITY;
            }
        }
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Geolocation arrays sampled from an affine grid.
    fn affine_arrays(
        gt: [f64; 6],
        ncol: usize,
        nrow: usize,
        opts: GeolocOptions,
    ) -> GeolocArrays {
        let off = opts.convention.offset();
        let mut xs = Vec::with_capacity(ncol * nrow);
        let mut ys = Vec::with_capacity(ncol * nrow);
        for iy in 0..nrow {
            for ix in 0..ncol {
                let p = (ix as f64 + off) * opts.pixel_step + opts.pixel_offset;
                let l = (iy as f64 + off) * opts.line_step + opts.line_offset;
                xs.push(gt[0] + p * gt[1] + l * gt[2]);
                ys.push(gt[3] + p * gt[4] + l * gt[5]);
            }
        }
        GeolocArrays::new(xs, ys, ncol, nrow, opts).unwrap()
    }

    const GT: [f64; 6] = [140.0, 0.25, 0.0, -40.0, 0.0, -0.2];

    #[test]
    fn affine_arrays_reproduce_the_geotransform() {
        for convention in [GeolocConvention::TopLeftCorner, GeolocConvention::PixelCenter] {
            let opts = GeolocOptions { convention, ..Default::default() };
            let g = affine_arrays(GT, 12, 9, opts);
            // Inside, on edges, and beyond the samples (extrapolated).
            for &(p, l) in &[
                (0.5, 0.5),
                (3.25, 7.75),
                (0.0, 0.0),
                (12.0, 9.0),
                (-0.4, 4.0),
                (11.9, 8.9),
            ] {
                let (gx, gy) = g.pixel_to_geo(p, l).unwrap();
                assert!((gx - (GT[0] + p * GT[1])).abs() < 1e-12, "{convention:?} {p} {l}");
                assert!((gy - (GT[3] + l * GT[5])).abs() < 1e-12, "{convention:?} {p} {l}");
                let (pp, ll) = g.geo_to_pixel(gx, gy).unwrap();
                assert!((pp - p).abs() < 1e-9 && (ll - l).abs() < 1e-9, "{convention:?} {p} {l} -> {pp} {ll}");
            }
        }
    }

    #[test]
    fn steps_and_offsets_map_subsampled_arrays() {
        // A 4x3 array sampling every 4th pixel, starting at pixel 2.
        let opts = GeolocOptions {
            pixel_offset: 2.0,
            pixel_step: 4.0,
            line_offset: 1.0,
            line_step: 3.0,
            ..Default::default()
        };
        let g = affine_arrays(GT, 4, 3, opts);
        let (gx, gy) = g.pixel_to_geo(7.5, 5.0).unwrap();
        assert!((gx - (GT[0] + 7.5 * GT[1])).abs() < 1e-12);
        assert!((gy - (GT[3] + 5.0 * GT[5])).abs() < 1e-12);
        let (p, l) = g.geo_to_pixel(gx, gy).unwrap();
        assert!((p - 7.5).abs() < 1e-9 && (l - 5.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_center_edges_are_midpoints_of_centres() {
        // Irregular spacing along x: centres at 0, 1, 3, 7 (one row repeated).
        let cx = [0.0, 1.0, 3.0, 7.0];
        let xs: Vec<f64> = (0..2).flat_map(|_| cx).collect();
        let ys: Vec<f64> = (0..2).flat_map(|r| [r as f64; 4]).collect();
        let opts = GeolocOptions { convention: GeolocConvention::PixelCenter, ..Default::default() };
        let g = GeolocArrays::new(xs, ys, 4, 2, opts).unwrap();
        // Raster pixel edge between pixels 1 and 2 is at raster x = 2.0.
        assert!((g.pixel_to_geo(2.0, 0.5).unwrap().0 - 2.0).abs() < 1e-12);
        // Between pixels 2 and 3: midpoint of 3 and 7.
        assert!((g.pixel_to_geo(3.0, 0.5).unwrap().0 - 5.0).abs() < 1e-12);
        // Outer edges: half a spacing beyond the outermost centres.
        assert!((g.pixel_to_geo(0.0, 0.5).unwrap().0 - -0.5).abs() < 1e-12);
        assert!((g.pixel_to_geo(4.0, 0.5).unwrap().0 - 9.0).abs() < 1e-12);
    }

    #[test]
    fn nodata_and_nan_samples_are_missing() {
        let opts = GeolocOptions { nodata_x: Some(-999.0), ..Default::default() };
        let mut g = affine_arrays(GT, 5, 5, opts);
        g.x[0] = -999.0;
        g.x[12] = f64::NAN;
        g.index = QuadIndex::new(g.build_quads());
        assert!(g.pixel_to_geo(0.2, 0.2).is_none());
        assert!(g.pixel_to_geo(2.5, 2.5).is_none());
        assert!(g.pixel_to_geo(4.5, 0.5).is_some());
    }

    #[test]
    fn longitudes_wrap_across_the_antimeridian() {
        // Columns at 178, 179.5, -179, -177.5 degrees.
        let lon = [178.0, 179.5, -179.0, -177.5];
        let xs: Vec<f64> = (0..3).flat_map(|_| lon).collect();
        let ys: Vec<f64> = (0..3).flat_map(|r| [-10.0 - r as f64; 4]).collect();
        let opts = GeolocOptions {
            convention: GeolocConvention::PixelCenter,
            lon_wrap: true,
            ..Default::default()
        };
        let g = GeolocArrays::new(xs, ys, 4, 3, opts).unwrap();
        // Halfway between 179.5 and -179 (i.e. 180.25) is -179.75.
        let (gx, _) = g.pixel_to_geo(2.0, 1.5).unwrap();
        assert!((gx - -179.75).abs() < 1e-12, "{gx}");
        // And back, from either side of the antimeridian.
        let (p, _) = g.geo_to_pixel(-179.75, -11.0).unwrap();
        assert!((p - 2.0).abs() < 1e-9, "{p}");
        let (p, _) = g.geo_to_pixel(179.75, -11.0).unwrap();
        // 179.75 is a sixth of the way from 179.5 (centre of pixel 1) to 181
        // (-179, centre of pixel 2): raster x = 1.5 + 1/6.
        assert!((p - (1.5 + 1.0 / 6.0)).abs() < 1e-9, "{p}");
    }

    #[test]
    fn points_outside_every_quad_fail() {
        let g = affine_arrays(GT, 6, 6, GeolocOptions::default());
        assert!(g.geo_to_pixel(0.0, 0.0).is_none());
        assert!(g.geo_to_pixel(f64::NAN, -40.0).is_none());
    }

    #[test]
    fn inverse_bilinear_recovers_a_skewed_quad() {
        // Forward bilinear on a skewed quad, then invert.
        let (x0, y0, x1, y1, x2, y2, x3, y3) = (0.0, 0.0, 0.3, 1.1, 1.2, 0.1, 1.4, 1.3);
        let (s, t) = (0.37, 0.81); // i fraction, j fraction
        let x = (1.0 - t) * (x0 + s * (x2 - x0)) + t * (x1 + s * (x3 - x1));
        let y = (1.0 - t) * (y0 + s * (y2 - y0)) + t * (y1 + s * (y3 - y1));
        let (mut i, mut j) = (5.0, 7.0);
        inverse_bilinear(x, y, x0, y0, x1, y1, x2, y2, x3, y3, &mut i, &mut j);
        assert!((i - 5.37).abs() < 1e-12 && (j - 7.81).abs() < 1e-12, "{i} {j}");
    }
}
