//! Coordinate transformers for the warp pipeline.
//!
//! Maps GDAL's transformer hierarchy (gdaltransformer.cpp, commit a3b7b01d3e):
//!
//!   Transformer trait          ← GDALTransformerFunc callback signature
//!   GenImgProjTransformer      ← GDALGenImgProjTransform
//!
//! Default gdalwarp architecture (from gdalwarp_lib.cpp line 3260):
//!
//!   ApproxTransformer {                    ← GDALApproxTransform (max_error=0.125)
//!     wraps: GenImgProjTransformer {       ← GDALGenImgProjTransform
//!       step 1: dst pixel → dst geo         (apply dst geotransform)
//!       step 2: dst geo -> src geo          (CRS coordinate transform)
//!       step 3: src geo → src pixel         (apply inv src geotransform)
//!     }
//!   }
//!
//! The ApproxTransformer wraps the GenImgProjTransform and not just
//! the reprojection step. The error threshold of 0.125 is in the output
//! space of GenImgProjTransform = source pixel coordinates (since the
//! warp kernel always calls with bDstToSrc=TRUE).
//!
//! Evidence from gdalwarp_lib.cpp:
//!   line 2841: pfnTransformer = GDALGenImgProjTransform
//!   line 3260: hTransformArg = GDALCreateApproxTransformer(
//!                  GDALGenImgProjTransform, hTransformArg, dfErrorThreshold)
//!   line 3263: pfnTransformer = GDALApproxTransform
//!   line 1598: dfErrorThreshold = 0.125 (default)
//!
//! Note: There is ALSO a "REPROJECTION_APPROX_ERROR_IN_DST_SRS_UNIT" option
//! that wraps just the reprojection inside GenImgProjTransformer, but this
//! is NEVER used by default gdalwarp. It's a separate code path for other
//! callers.
//!
//! ## Cell-centre convention
//!
//! GDAL's warp kernel constructs destination pixel coordinates as
//! `iDstX + 0.5` (cell centres, not top-left corners). The high-level
//! [`transform_scanline`] function handles this automatically. The raw
//! [`Transformer::transform`] method receives whatever coordinates the
//! caller provides — integer pixel indices give top-left corner coordinates,
//! not centres. This is a silent half-pixel shift that is easy to miss.
//!
//! ## CRS backend
//!
//! The CRS -> CRS step is abstracted behind [`crate::crs::CrsTransform`].
//! [`GenImgProjTransformer::new`] builds the pair with whichever backend the
//! crate was compiled with (`proj` or `proj4rs` feature, see [`crate::crs`]);
//! [`GenImgProjTransformer::with_backend`] accepts any pair of transforms.
//!
//! Geographic coordinates are lon/lat in degrees with axis order normalised
//! to easting/northing, for every backend. Raw PROJ strings (`+proj=...`)
//! get `+type=crs` appended automatically so libproj interprets them as
//! CRS definitions rather than pipeline operations.
//!
//! ## Two transform objects
//!
//! A `CrsTransform` has a fixed direction, so we hold separate instances
//! for `dst->src` and `src->dst`. This doubles initialisation cost but is
//! negligible for tile-serving use cases where the transformer is created
//! once and reused across tiles.

use crate::crs::{crs_pair, is_geographic_degrees, CrsTransform};
use vaster::inv_geotransform;
// ---------------------------------------------------------------------------
// Transformer trait
// ---------------------------------------------------------------------------

/// Trait for coordinate transformers.
///
/// Matches the semantics of GDAL's GDALTransformerFunc:
/// - Takes arrays of x, y coordinates (modified in place)
/// - Returns per-point success flags
/// - `dst_to_src=true`: input is dest pixel coords, output is src pixel coords
/// - `dst_to_src=false`: input is src pixel coords, output is dest pixel coords
///
/// For warp, the kernel always calls with dst_to_src=true.
///
/// The trait is intentionally minimal — it maps coordinate arrays without
/// knowing what data values are. This makes it the right seam for future
/// transformer plugins: curvilinear grids, displacement fields, GCP
/// polynomials, and others all implement this trait and compose unchanged
/// with `ApproxTransformer`, `compute_source_window`, and `warp_resample`.
///
/// For vector-quantity warps (velocity fields, wind, ocean currents) the
/// optional [`Transformer::jacobian`] method provides the local frame
/// rotation needed to correctly reproject `(u, v)` components. Scalar
/// resampling via `warp_resample` does not call `jacobian`.
pub trait Transformer {
    fn transform(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
    ) -> Vec<bool>;

    /// Local Jacobian of the transform at a single point.
    ///
    /// Returns the 2×2 matrix:
    /// ```text
    /// [[∂x_out/∂x_in,  ∂x_out/∂y_in],
    ///  [∂y_out/∂x_in,  ∂y_out/∂y_in]]
    /// ```
    /// evaluated at `(x, y)` in the input coordinate space.
    ///
    /// Used by vector-quantity-preserving warp kernels to rotate `(u, v)`
    /// fields from the source index-grid frame into the destination CRS
    /// frame. This is the generalized form of the north-arrow correction:
    /// "north" (or any reference direction) rotates relative to the display
    /// grid as you move across a projection, and `(u, v)` components stored
    /// in index-grid coordinates need the same correction when reprojected.
    ///
    /// Scalar resampling via `warp_resample` does not call this method.
    ///
    /// The default implementation uses central finite differences (step 1e-5).
    /// Implementors with analytic Jacobians (e.g. from `proj_factors`) should
    /// override for accuracy and performance.
    fn jacobian(&self, dst_to_src: bool, x: f64, y: f64) -> Option<[[f64; 2]; 2]> {
        let h = 1e-5_f64;

        let mut xp = [x + h, x - h];
        let mut yp = [y,     y    ];
        let okx = self.transform(dst_to_src, &mut xp, &mut yp);
        if !okx[0] || !okx[1] { return None; }
        let dxdx = (xp[0] - xp[1]) / (2.0 * h);
        let dydx = (yp[0] - yp[1]) / (2.0 * h);

        let mut xq = [x,     x    ];
        let mut yq = [y + h, y - h];
        let oky = self.transform(dst_to_src, &mut xq, &mut yq);
        if !oky[0] || !oky[1] { return None; }
        let dxdy = (xq[0] - xq[1]) / (2.0 * h);
        let dydy = (yq[0] - yq[1]) / (2.0 * h);

        Some([[dxdx, dxdy], [dydx, dydy]])
    }
}

// ---------------------------------------------------------------------------
// GenImgProjTransformer
// ---------------------------------------------------------------------------

/// The three-step pixel ↔ geo ↔ CRS pipeline.
///
/// Maps GDAL's GDALGenImgProjTransform (gdaltransformer.cpp line ~3100).
///
/// Composes:
///   1. Apply geotransform: pixel → geo (always exact)
///   2. CRS transform: CRS -> CRS (always exact, per-point)
///   3. Apply inverse geotransform: geo → pixel (always exact)
///
/// The ApproxTransformer wraps this entire struct from outside (in lib.rs),
/// matching gdalwarp's default architecture.
///
/// ## Longitude wrapping (`CENTER_LONG`)
///
/// For a source CRS that is geographic in degrees, GDAL wraps source
/// longitudes into the 360 degrees around the source raster's centre, so a
/// target reaching across 180 degrees from a source near it still finds it
/// (a source stored as 170..190, or in 0..360). rwarp does the same once it
/// knows the source raster's size, which GDAL takes from the dataset:
///
/// ```
/// # use rwarp::transform::GenImgProjTransformer;
/// let src_gt = [170.0, 0.25, 0.0, -10.0, 0.0, -0.25]; // 170E to 190E
/// let t = GenImgProjTransformer::new(
///     "EPSG:4326", src_gt,
///     "+proj=laea +lat_0=-20 +lon_0=180", [-1e6, 1e4, 0.0, 1e6, 0.0, -1e4],
/// )?
/// .with_source_size([80, 40]);
/// assert_eq!(t.center_long(), Some(180.0));
/// # Ok::<(), String>(())
/// ```
///
/// See [`CenterLong`] for the rule and the alternatives. Without
/// [`with_source_size`](Self::with_source_size) nothing is wrapped, as
/// in rwarp 0.3.0.
pub struct GenImgProjTransformer {
    // Source grid
    src_gt: [f64; 6],
    src_inv_gt: [f64; 6],

    // Destination grid
    dst_gt: [f64; 6],
    dst_inv_gt: [f64; 6],

    // CRS coordinate transforms
    crs_dst_to_src: Box<dyn CrsTransform>,
    crs_src_to_dst: Box<dyn CrsTransform>,

    // CENTER_LONG: what is known of the CRSs (None with `with_backend`),
    // the inputs to GDAL's rule, and the longitude wrapped around.
    src_geographic: Option<bool>,
    dst_geographic: Option<bool>,
    src_size: Option<[usize; 2]>,
    target_extent: Option<[f64; 4]>,
    center_mode: CenterLong,
    center: Option<f64>,
}

/// How [`GenImgProjTransformer`] wraps the longitudes of a geographic
/// source: GDAL's `CENTER_LONG`.
///
/// With a centre `c`, a source longitude below `c - 180` gets 360 added
/// and one above `c + 180` gets 360 taken away (once; the bounds stay as
/// they are). Target to source, the wrap is applied to the longitude the
/// CRS step returns; source to target, to the longitude given to it. GDAL
/// does this in `OGRProjCT` (ogr/ogrct.cpp, `dfSourceWrapLong`, applied to
/// the source CRS and, through `GetInverse`, to the target CRS of the
/// reverse transform the warper uses).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum CenterLong {
    /// GDAL's default (`InsertCenterLong` in gdaltransformer.cpp, used by
    /// `GDALCreateGenImgProjTransformer2` and so by gdalwarp): when the
    /// source CRS is geographic in degrees, wrap around the middle of the
    /// smallest and largest longitudes of the source raster's four
    /// corners, rounded to six significant digits as GDAL passes it
    /// through `%g`. Nothing is wrapped if the corners span more than 360
    /// degrees, unless the target is geographic too and its extent
    /// (gdalwarp's `-te`, see
    /// [`GenImgProjTransformer::with_target_extent`]) crosses 180 degrees.
    ///
    /// Needs the source raster's size
    /// ([`GenImgProjTransformer::with_source_size`]), and a transformer
    /// made with [`GenImgProjTransformer::new`] (one made with
    /// [`GenImgProjTransformer::with_backend`] does not know its CRSs; use
    /// [`CenterLong::At`] with [`source_center_long`]).
    #[default]
    Auto,
    /// No wrapping (GDAL's `INSERT_CENTER_LONG=FALSE`, which gdalwarp uses
    /// to bring a cutline into source pixels).
    Off,
    /// Wrap around this longitude: a `CENTER_LONG` given in the source
    /// CRS's WKT (`EXTENSION["CENTER_LONG", ...]`) or to
    /// `GDALCreateReprojectionTransformerEx`. Ignored, as by GDAL, for a
    /// source known not to be geographic; applied for a transformer made
    /// with [`GenImgProjTransformer::with_backend`].
    At(f64),
}

/// GDAL's centre longitude for a geographic source raster
/// (`InsertCenterLong`, gdaltransformer.cpp): the middle of the smallest
/// and largest x of the four corners of the raster, `src_size` = `[width,
/// height]` pixels, rounded to six significant digits (GDAL formats it
/// with `%g`).
///
/// `None` when the corners span more than 360 degrees, unless
/// `geographic_target_extent` (`[xmin, ymin, xmax, ymax]` of a target in a
/// geographic CRS in degrees, gdalwarp's `-te`) crosses 180 degrees: xmin
/// in `[-179, 180)` with xmax above 180, or xmin below -180 with xmax in
/// `(-180, 179]`.
pub fn source_center_long(
    src_gt: &[f64; 6],
    src_size: [usize; 2],
    geographic_target_extent: Option<[f64; 4]>,
) -> Option<f64> {
    let (w, h) = (src_size[0] as f64, src_size[1] as f64);
    let corner = |p: f64, l: f64| src_gt[0] + p * src_gt[1] + l * src_gt[2];
    let xs = [corner(0.0, 0.0), corner(w, 0.0), corner(0.0, h), corner(w, h)];
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    if max - min > 360.0 {
        let crosses = geographic_target_extent.is_some_and(|te| {
            (te[0] >= -179.0 && te[0] < 180.0 && te[2] > 180.0) || (te[0] < -180.0 && te[2] > -180.0 && te[2] <= 179.0)
        });
        if !crosses {
            return None;
        }
    }
    Some(printf_g((max + min) / 2.0))
}

/// `CPLAtof(CPLSPrintf("%g", v))`: `v` to six significant digits.
fn printf_g(v: f64) -> f64 {
    format!("{v:.5e}").parse().unwrap_or(v)
}

/// GDAL's source longitude wrap (`OGRProjCT`, ogr/ogrct.cpp).
#[inline]
fn wrap_long(x: f64, center: f64) -> f64 {
    if x < center - 180.0 {
        x + 360.0
    } else if x > center + 180.0 {
        x - 360.0
    } else {
        x
    }
}

impl GenImgProjTransformer {
    /// Build a transformer from CRS strings using the compiled-in backend.
    pub fn new(
        src_crs: &str,
        src_gt: [f64; 6],
        dst_crs: &str,
        dst_gt: [f64; 6],
    ) -> Result<Self, String> {
        let (crs_dst_to_src, crs_src_to_dst) = crs_pair(src_crs, dst_crs)?;
        let mut t = Self::with_backend(src_gt, dst_gt, crs_dst_to_src, crs_src_to_dst)?;
        t.src_geographic = is_geographic_degrees(src_crs).ok();
        t.dst_geographic = is_geographic_degrees(dst_crs).ok();
        t.resolve_center_long();
        Ok(t)
    }

    /// Give the source raster's size, `[width, height]` in pixels, so the
    /// source longitudes can be wrapped as GDAL does by default
    /// ([`CenterLong::Auto`]).
    pub fn with_source_size(mut self, size: [usize; 2]) -> Self {
        self.src_size = Some(size);
        self.resolve_center_long();
        self
    }

    /// Give the target extent `[xmin, ymin, xmax, ymax]` in target CRS
    /// units, gdalwarp's `-te` (GDAL's `TARGET_EXTENT` transformer option).
    /// It only matters to [`CenterLong::Auto`] for a source wider than 360
    /// degrees and a geographic target.
    pub fn with_target_extent(mut self, extent: [f64; 4]) -> Self {
        self.target_extent = Some(extent);
        self.resolve_center_long();
        self
    }

    /// Choose how source longitudes are wrapped (default
    /// [`CenterLong::Auto`]).
    pub fn with_center_long(mut self, center: CenterLong) -> Self {
        self.center_mode = center;
        self.resolve_center_long();
        self
    }

    /// The longitude source longitudes are wrapped around, if any.
    pub fn center_long(&self) -> Option<f64> {
        self.center
    }

    fn resolve_center_long(&mut self) {
        self.center = match self.center_mode {
            CenterLong::Off => None,
            CenterLong::At(c) => (self.src_geographic != Some(false)).then_some(c),
            CenterLong::Auto => match (self.src_geographic, self.src_size) {
                (Some(true), Some(size)) => {
                    let te = self.target_extent.filter(|_| self.dst_geographic == Some(true));
                    source_center_long(&self.src_gt, size, te)
                }
                _ => None,
            },
        };
    }

    /// Build a transformer from caller-supplied CRS transforms.
    ///
    /// This is the seam for backends the crate does not ship: a JavaScript
    /// PROJ build reached from wasm, a curvilinear lookup table, a test
    /// double. `crs_dst_to_src` maps destination geo coords to source geo
    /// coords; `crs_src_to_dst` is its inverse.
    pub fn with_backend(
        src_gt: [f64; 6],
        dst_gt: [f64; 6],
        crs_dst_to_src: Box<dyn CrsTransform>,
        crs_src_to_dst: Box<dyn CrsTransform>,
    ) -> Result<Self, String> {
        let src_inv_gt = inv_geotransform(&src_gt)
            .ok_or_else(|| "Source geotransform not invertible".to_string())?;
        let dst_inv_gt = inv_geotransform(&dst_gt)
            .ok_or_else(|| "Dest geotransform not invertible".to_string())?;

        Ok(Self {
            src_gt,
            src_inv_gt,
            dst_gt,
            dst_inv_gt,
            crs_dst_to_src,
            crs_src_to_dst,
            src_geographic: None,
            dst_geographic: None,
            src_size: None,
            target_extent: None,
            center_mode: CenterLong::Auto,
            center: None,
        })
    }
}

impl Transformer for GenImgProjTransformer {
    /// Transform coordinates through the three-step pipeline.
    ///
    /// When `dst_to_src = true` (the warp kernel path):
    ///   1. Apply dst geotransform: pixel → geo
    ///   2. CRS transform: dst CRS -> src CRS
    ///   3. Apply inverse src geotransform: geo → pixel
    fn transform(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
    ) -> Vec<bool> {
        let n = x.len();

        if dst_to_src {
            // Step 1: dst pixel → dst geo (line 3113-3142)
            for i in 0..n {
                let pixel = x[i];
                let line = y[i];
                x[i] = self.dst_gt[0] + pixel * self.dst_gt[1] + line * self.dst_gt[2];
                y[i] = self.dst_gt[3] + pixel * self.dst_gt[4] + line * self.dst_gt[5];
            }

            // Step 2: CRS dst geo -> src geo (line 3148-3152)
            let mut success = vec![true; n];
            for i in 0..n {
                match self.crs_dst_to_src.convert(x[i], y[i]) {
                    Some((rx, ry)) => {
                        // CENTER_LONG: the reverse OGRProjCT wraps its
                        // target (our source) longitude on the way out
                        x[i] = match self.center {
                            Some(c) => wrap_long(rx, c),
                            None => rx,
                        };
                        y[i] = ry;
                    }
                    None => {
                        success[i] = false;
                    }
                }
            }

            // Step 3: src geo → src pixel (line 3158-3187)
            for i in 0..n {
                if !success[i] {
                    continue;
                }
                let geo_x = x[i];
                let geo_y = y[i];
                x[i] = self.src_inv_gt[0]
                    + geo_x * self.src_inv_gt[1]
                    + geo_y * self.src_inv_gt[2];
                y[i] = self.src_inv_gt[3]
                    + geo_x * self.src_inv_gt[4]
                    + geo_y * self.src_inv_gt[5];
            }

            success
        } else {
            // Reverse: src pixel → src geo → dst geo → dst pixel

            for i in 0..n {
                let pixel = x[i];
                let line = y[i];
                x[i] = self.src_gt[0] + pixel * self.src_gt[1] + line * self.src_gt[2];
                y[i] = self.src_gt[3] + pixel * self.src_gt[4] + line * self.src_gt[5];
            }

            // CENTER_LONG: the forward OGRProjCT wraps its source longitude
            // on the way in
            if let Some(c) = self.center {
                x.iter_mut().for_each(|v| *v = wrap_long(*v, c));
            }

            let mut success = vec![true; n];
            for i in 0..n {
                match self.crs_src_to_dst.convert(x[i], y[i]) {
                    Some((rx, ry)) => {
                        x[i] = rx;
                        y[i] = ry;
                    }
                    None => {
                        success[i] = false;
                    }
                }
            }

            for i in 0..n {
                if !success[i] {
                    continue;
                }
                let geo_x = x[i];
                let geo_y = y[i];
                x[i] = self.dst_inv_gt[0]
                    + geo_x * self.dst_inv_gt[1]
                    + geo_y * self.dst_inv_gt[2];
                y[i] = self.dst_inv_gt[3]
                    + geo_x * self.dst_inv_gt[4]
                    + geo_y * self.dst_inv_gt[5];
            }

            success
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: transform scanlines
// ---------------------------------------------------------------------------


/// Build destination pixel coordinates for a scanline and transform to source.
///
/// Matches GDAL's pattern at the top of GWKNearestThread (line 5560):
/// ```text
///   for (iDstX = 0; iDstX < nDstXSize; iDstX++) {
///       padfX[iDstX] = iDstX + 0.5 + nDstXOff;
///       padfY[iDstX] = iDstY + 0.5 + nDstYOff;
///   }
///   pfnTransformer(..., TRUE, nDstXSize, padfX, padfY, padfZ, pabSuccess);
/// ```
pub fn transform_scanline(
    transformer: &impl Transformer,
    dst_y: usize,
    dst_ncol: usize,
    dst_x_off: usize,
    dst_y_off: usize,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut x: Vec<f64> = (0..dst_ncol)
        .map(|col| (col + dst_x_off) as f64 + 0.5)
        .collect();
    let mut y = vec![(dst_y + dst_y_off) as f64 + 0.5; dst_ncol];

    let success = transformer.transform(true, &mut x, &mut y);

    (x, y, success)
}

/// Compute source pixel coordinates for every destination pixel.
pub fn transform_grid(
    transformer: &impl Transformer,
    dst_dim: &[usize; 2],
) -> (Vec<f64>, Vec<f64>) {
    let n = dst_dim[0] * dst_dim[1];
    let mut all_x = Vec::with_capacity(n);
    let mut all_y = Vec::with_capacity(n);

    for row in 0..dst_dim[1] {
        let (sx, sy, ok) = transform_scanline(transformer, row, dst_dim[0], 0, 0);
        for col in 0..dst_dim[0] {
            if ok[col] {
                all_x.push(sx[col]);
                all_y.push(sy[col]);
            } else {
                all_x.push(f64::NAN);
                all_y.push(f64::NAN);
            }
        }
    }

    (all_x, all_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity CRS step: what a geographic source to the same CRS does.
    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }

    const G360: [f64; 6] = [0.0, 1.0, 0.0, 90.0, 0.0, -1.0];
    const G180: [f64; 6] = [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0];

    #[test]
    fn center_is_gdal_percent_g() {
        // GDAL: "OGRCT: Wrap source at 147.128." for this raster
        let gt = [147.1234567, 0.001, 0.0, 0.0, 0.0, -0.001];
        assert_eq!(source_center_long(&gt, [10, 10], None), Some(147.128));
        assert_eq!(source_center_long(&[170.0, 0.25, 0.0, -10.0, 0.0, -0.25], [80, 40], None), Some(180.0));
        assert_eq!(source_center_long(&G360, [360, 180], None), Some(180.0));
        // rotated: the four corners count
        assert_eq!(source_center_long(&[10.0, 1.0, 1.0, 0.0, 0.0, -1.0], [20, 10], None), Some(25.0));
    }

    #[test]
    fn wider_than_360_wraps_only_for_a_geographic_target_across_180() {
        let gt = [-180.5, 1.0, 0.0, 90.0, 0.0, -1.0];
        assert_eq!(source_center_long(&gt, [361, 180], None), None);
        assert_eq!(source_center_long(&gt, [361, 180], Some([-180.0, -90.0, 180.0, 90.0])), None);
        assert_eq!(source_center_long(&gt, [361, 180], Some([90.0, -90.0, 270.0, 90.0])), Some(0.0));
        assert_eq!(source_center_long(&gt, [361, 180], Some([-270.0, -90.0, -90.0, 90.0])), Some(0.0));
        // GDAL's bounds: xmin from -179, xmax up to 179
        assert_eq!(source_center_long(&gt, [361, 180], Some([-179.5, -90.0, 181.0, 90.0])), None);
        assert_eq!(source_center_long(&gt, [361, 180], Some([-181.0, -90.0, 179.5, 90.0])), None);
    }

    #[test]
    fn wrap_keeps_both_bounds() {
        assert_eq!(wrap_long(0.0, 180.0), 0.0);
        assert_eq!(wrap_long(360.0, 180.0), 360.0);
        assert_eq!(wrap_long(-0.5, 180.0), 359.5);
        assert_eq!(wrap_long(-170.0, 180.0), 190.0);
        assert_eq!(wrap_long(190.0, -180.0), -170.0);
        assert_eq!(wrap_long(400.0, 0.0), 40.0);
        assert_eq!(wrap_long(-540.0, 0.0), -180.0); // once only
    }

    fn same(src_gt: [f64; 6], dst_gt: [f64; 6]) -> GenImgProjTransformer {
        GenImgProjTransformer::with_backend(src_gt, dst_gt, Box::new(Same), Box::new(Same)).unwrap()
    }

    #[test]
    fn explicit_center_wraps_both_ways() {
        // a 0..360 source and a -180..180 target, as GDAL with EPSG:4326 on
        // both sides: target lon -169.5 is source column 190.5
        let t = same(G360, G180).with_center_long(CenterLong::At(180.0));
        let (mut x, mut y) = ([10.5], [10.5]);
        assert!(t.transform(true, &mut x, &mut y)[0]);
        assert_eq!((x[0], y[0]), (190.5, 10.5));
        // source to target: lon 10.5 is in 0..360 already, so it stays
        let (mut x, mut y) = ([10.5], [10.5]);
        assert!(t.transform(false, &mut x, &mut y)[0]);
        assert_eq!(x[0], 190.5);
        // a source longitude outside 0..360 is brought in
        let (mut x, mut y) = ([-20.0], [0.0]);
        t.transform(false, &mut x, &mut y);
        assert_eq!(x[0], 180.0 + 340.0);
    }

    #[test]
    fn auto_needs_known_crs_and_size() {
        // with_backend does not know the CRSs: Auto does nothing
        let t = same(G360, G180).with_source_size([360, 180]);
        assert_eq!(t.center_long(), None);
        let t = t.with_center_long(CenterLong::At(180.0));
        assert_eq!(t.center_long(), Some(180.0));
        assert_eq!(t.with_center_long(CenterLong::Off).center_long(), None);
    }

    #[cfg(any(feature = "proj", feature = "proj4rs"))]
    #[test]
    fn auto_follows_gdal_for_crs_strings() {
        let t = GenImgProjTransformer::new("EPSG:4326", G360, "EPSG:4326", G180).unwrap();
        assert_eq!(t.center_long(), None, "no size, no wrap");
        let t = t.with_source_size([360, 180]);
        assert_eq!(t.center_long(), Some(180.0));
        let (mut x, mut y) = ([10.5], [10.5]);
        assert!(t.transform(true, &mut x, &mut y)[0]);
        assert!((x[0] - 190.5).abs() < 1e-9, "{}", x[0]);
        // a projected source is never wrapped, even when asked
        let merc = [-2e7, 1e5, 0.0, 2e7, 0.0, -1e5];
        let t = GenImgProjTransformer::new("EPSG:3857", merc, "EPSG:4326", G180)
            .unwrap()
            .with_source_size([400, 400]);
        assert_eq!(t.center_long(), None);
        assert_eq!(t.with_center_long(CenterLong::At(0.0)).center_long(), None);
        let t = GenImgProjTransformer::new("+proj=longlat +datum=WGS84 +no_defs", G360, "EPSG:3857", merc)
            .unwrap()
            .with_source_size([360, 180]);
        assert_eq!(t.center_long(), Some(180.0));
    }

    #[cfg(feature = "proj")]
    #[test]
    fn geographic_degrees_as_gdal_sees_it() {
        use crate::crs::is_geographic_degrees as g;
        assert!(g("EPSG:4326").unwrap());
        assert!(g("EPSG:4979").unwrap()); // 3D
        assert!(g("+proj=longlat +ellps=GRS80 +towgs84=0,0,0 +no_defs").unwrap()); // bound
        assert!(g("EPSG:4326+5773").unwrap()); // compound
        assert!(!g("EPSG:4807").unwrap()); // NTF (Paris), grads
        assert!(!g("EPSG:3857").unwrap());
        assert!(!g("EPSG:4978").unwrap()); // geocentric
        assert!(g("not a crs").is_err());
    }
}


