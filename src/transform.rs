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
//!       step 2: dst geo → src geo           (PROJ coordinate transform)
//!       step 3: src geo → src pixel         (apply inv src geotransform)
//!     }
//!   }
//!
//! The ApproxTransformer wraps the ENTIRE GenImgProjTransform, not just
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
pub trait Transformer {
    fn transform(
        &self,
        dst_to_src: bool,
        x: &mut [f64],
        y: &mut [f64],
    ) -> Vec<bool>;
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
///   2. PROJ: CRS → CRS (always exact, per-point)
///   3. Apply inverse geotransform: geo → pixel (always exact)
///
/// The ApproxTransformer wraps this entire struct from outside (in lib.rs),
/// matching gdalwarp's default architecture.
pub struct GenImgProjTransformer {
    // Source grid
    src_gt: [f64; 6],
    src_inv_gt: [f64; 6],

    // Destination grid
    dst_gt: [f64; 6],
    dst_inv_gt: [f64; 6],

    // PROJ coordinate transforms
    proj_dst_to_src: proj::Proj,
    proj_src_to_dst: proj::Proj,
}

impl GenImgProjTransformer {
    pub fn new(
        src_crs: &str,
        src_gt: [f64; 6],
        dst_crs: &str,
        dst_gt: [f64; 6],
    ) -> Result<Self, String> {
        let src_inv_gt = inv_geotransform(&src_gt)
            .ok_or_else(|| "Source geotransform not invertible".to_string())?;
        let dst_inv_gt = inv_geotransform(&dst_gt)
            .ok_or_else(|| "Dest geotransform not invertible".to_string())?;

        let proj_dst_to_src = proj::Proj::new_known_crs(dst_crs, src_crs, None)
            .map_err(|e| format!("PROJ dst→src failed: {}", e))?;
        let proj_src_to_dst = proj::Proj::new_known_crs(src_crs, dst_crs, None)
            .map_err(|e| format!("PROJ src→dst failed: {}", e))?;

        Ok(Self {
            src_gt,
            src_inv_gt,
            dst_gt,
            dst_inv_gt,
            proj_dst_to_src,
            proj_src_to_dst,
        })
    }
}

impl Transformer for GenImgProjTransformer {
    /// Transform coordinates through the three-step pipeline.
    ///
    /// When `dst_to_src = true` (the warp kernel path):
    ///   1. Apply dst geotransform: pixel → geo
    ///   2. PROJ: dst CRS → src CRS
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

            // Step 2: PROJ dst geo → src geo (line 3148-3152)
            let mut success = vec![true; n];
            for i in 0..n {
                match self.proj_dst_to_src.convert((x[i], y[i])) {
                    Ok((rx, ry)) => {
                        x[i] = rx;
                        y[i] = ry;
                    }
                    Err(_) => {
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

            let mut success = vec![true; n];
            for i in 0..n {
                match self.proj_src_to_dst.convert((x[i], y[i])) {
                    Ok((rx, ry)) => {
                        x[i] = rx;
                        y[i] = ry;
                    }
                    Err(_) => {
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
///   for (iDstX = 0; iDstX < nDstXSize; iDstX++) {
///       padfX[iDstX] = iDstX + 0.5 + nDstXOff;
///       padfY[iDstX] = iDstY + 0.5 + nDstYOff;
///   }
///   pfnTransformer(..., TRUE, nDstXSize, padfX, padfY, padfZ, pabSuccess);
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


