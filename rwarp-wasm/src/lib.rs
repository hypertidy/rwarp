//! wasm-bindgen shim over rwarp for in-browser tile reprojection.
//!
//! The pipeline per destination tile:
//!
//!   1. `Warper::new`            target grid + source grid -> ApproxTransformer
//!   2. `Warper::source_window`  which source pixels does this tile need?
//!   3. (JavaScript)             fetch + decode the source tiles covering that
//!                               window into one RGBA buffer
//!   4. `Warper::warp_rgba`      resample into the destination tile
//!
//! Everything below `#[wasm_bindgen]` is plain Rust so the core can be
//! tested natively with `cargo test`; the bindgen layer only converts types.

use rwarp::approx::ApproxTransformer;
use rwarp::CrsTransform;
use rwarp::source_window::{compute_source_window, SourceWindow};
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{warp_resample, ResampleAlg};
use wasm_bindgen::prelude::*;

/// Fill value for unmapped destination pixels. Outside u8 range, so every
/// source sample is treated as valid data by the kernels.
const NODATA: i32 = -1;

// ---------------------------------------------------------------------------
// Core (native-testable)
// ---------------------------------------------------------------------------

pub struct WarperCore {
    approx: ApproxTransformer<GenImgProjTransformer>,
}

impl WarperCore {
    pub fn new(
        src_crs: &str,
        src_gt: [f64; 6],
        dst_crs: &str,
        dst_gt: [f64; 6],
        max_error: f64,
    ) -> Result<Self, String> {
        let exact = GenImgProjTransformer::new(src_crs, src_gt, dst_crs, dst_gt)?;
        Ok(Self { approx: ApproxTransformer::new(exact, max_error) })
    }

    pub fn source_window(
        &self,
        dst_size: [i32; 2],
        src_raster_size: [i32; 2],
        padding: i32,
    ) -> Option<SourceWindow> {
        compute_source_window(&self.approx, [0, 0], dst_size, src_raster_size, padding)
    }

    /// Warp an interleaved RGBA buffer. Unmapped destination pixels are
    /// fully transparent. First cut: deinterleave, warp four bands,
    /// reinterleave. A single-pass interleaved kernel in rwarp will replace
    /// this without changing the signature.
    #[allow(clippy::too_many_arguments)]
    pub fn warp_rgba(
        &self,
        src: &[u8],
        src_w: usize,
        src_h: usize,
        src_xoff: usize,
        src_yoff: usize,
        dst_w: usize,
        dst_h: usize,
        alg: ResampleAlg,
    ) -> Result<Vec<u8>, String> {
        if src.len() != src_w * src_h * 4 {
            return Err(format!(
                "source buffer is {} bytes, expected {}x{}x4 = {}",
                src.len(), src_w, src_h, src_w * src_h * 4
            ));
        }
        let n_src = src_w * src_h;
        let n_dst = dst_w * dst_h;
        let mut out = vec![0u8; n_dst * 4];
        let mut band = vec![0i32; n_src];

        for b in 0..4 {
            for i in 0..n_src {
                band[i] = src[i * 4 + b] as i32;
            }
            let warped = warp_resample(
                &self.approx, &band, src_w, src_h, src_xoff, src_yoff,
                dst_w, dst_h, NODATA, alg,
            );
            for i in 0..n_dst {
                let v = warped[i];
                out[i * 4 + b] = if v == NODATA { 0 } else { v.clamp(0, 255) as u8 };
            }
        }
        // Any band unmapped -> whole pixel transparent. Cheap second pass
        // that also guards against band-wise nodata disagreement at edges
        // under interpolating kernels.
        if alg != ResampleAlg::NearestNeighbour {
            let mut mask = vec![true; n_dst];
            let mut probe = vec![255i32; n_src];
            let w = warp_resample(
                &self.approx, &probe, src_w, src_h, src_xoff, src_yoff,
                dst_w, dst_h, NODATA, alg,
            );
            for i in 0..n_dst {
                mask[i] = w[i] != NODATA;
            }
            probe.clear();
            for i in 0..n_dst {
                if !mask[i] {
                    out[i * 4..i * 4 + 4].fill(0);
                }
            }
        }
        Ok(out)
    }
}

pub fn parse_alg(s: &str) -> Result<ResampleAlg, String> {
    match s.to_ascii_lowercase().as_str() {
        "near" | "nearest" | "nearestneighbour" | "nearestneighbor" => Ok(ResampleAlg::NearestNeighbour),
        "bilinear" => Ok(ResampleAlg::Bilinear),
        "cubic" => Ok(ResampleAlg::Cubic),
        "lanczos" => Ok(ResampleAlg::Lanczos),
        other => Err(format!("unknown resampling algorithm {other:?}")),
    }
}

fn gt6(v: &[f64], what: &str) -> Result<[f64; 6], String> {
    v.try_into().map_err(|_| format!("{what} geotransform must have 6 elements, got {}", v.len()))
}

// ---------------------------------------------------------------------------
// wasm-bindgen surface
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Warper {
    core: WarperCore,
}

#[wasm_bindgen]
impl Warper {
    /// `src_gt` / `dst_gt`: GDAL geotransforms `[x0, dx, rx, y0, ry, dy]`.
    /// `max_error`: approximation threshold in source pixels (GDAL default 0.125).
    #[wasm_bindgen(constructor)]
    pub fn new(
        src_crs: &str,
        src_gt: &[f64],
        dst_crs: &str,
        dst_gt: &[f64],
        max_error: f64,
    ) -> Result<Warper, JsError> {
        let js = |e: String| JsError::new(&e);
        let core = WarperCore::new(
            src_crs, gt6(src_gt, "source").map_err(js)?,
            dst_crs, gt6(dst_gt, "destination").map_err(js)?, max_error,
        )
        .map_err(js)?;
        Ok(Warper { core })
    }

    /// Source pixel window needed for a `dst_w` x `dst_h` destination tile,
    /// given the full source raster is `src_w` x `src_h` pixels.
    /// Returns `[xoff, yoff, xsize, ysize]`, or `undefined` if the tile does
    /// not intersect the source at all.
    pub fn source_window(
        &self,
        dst_w: u32,
        dst_h: u32,
        src_w: u32,
        src_h: u32,
        padding: u32,
    ) -> Option<Vec<i32>> {
        self.core
            .source_window([dst_w as i32, dst_h as i32], [src_w as i32, src_h as i32], padding as i32)
            .filter(|w| w.xsize > 0 && w.ysize > 0)
            .map(|w| vec![w.xoff, w.yoff, w.xsize, w.ysize])
    }

    /// Warp an RGBA buffer (`src_w * src_h * 4` bytes) whose top-left pixel
    /// sits at `(src_xoff, src_yoff)` in the full source raster, into a
    /// `dst_w * dst_h * 4` RGBA buffer. `alg` is one of
    /// `nearest | bilinear | cubic | lanczos`.
    #[allow(clippy::too_many_arguments)]
    pub fn warp_rgba(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        src_xoff: u32,
        src_yoff: u32,
        dst_w: u32,
        dst_h: u32,
        alg: &str,
    ) -> Result<Vec<u8>, JsError> {
        let alg = parse_alg(alg).map_err(|e| JsError::new(&e))?;
        self.core
            .warp_rgba(
                src, src_w as usize, src_h as usize, src_xoff as usize, src_yoff as usize,
                dst_w as usize, dst_h as usize, alg,
            )
            .map_err(|e| JsError::new(&e))
    }
}

/// Project a lon/lat (degrees, WGS84) into `crs`. Returns `[x, y]` or
/// `undefined`. Lets the page place markers and read coordinates without a
/// JavaScript PROJ.
#[wasm_bindgen]
pub fn lonlat_to_crs(crs: &str, lon: f64, lat: f64) -> Option<Vec<f64>> {
    let t = rwarp::crs::Proj4rsBackend::new("EPSG:4326", crs).ok()?;
    t.convert(lon, lat).map(|(x, y)| vec![x, y])
}

/// Inverse of [`lonlat_to_crs`]: `[lon, lat]` in degrees or `undefined`.
#[wasm_bindgen]
pub fn crs_to_lonlat(crs: &str, x: f64, y: f64) -> Option<Vec<f64>> {
    let t = rwarp::crs::Proj4rsBackend::new(crs, "EPSG:4326").ok()?;
    t.convert(x, y).map(|(lon, lat)| vec![lon, lat])
}

// ---------------------------------------------------------------------------
// Native tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HALF: f64 = 20037508.342789244;
    const LAEA_TAS: &str = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";

    fn webmerc(z: u32) -> ([f64; 6], [i32; 2]) {
        let n = 256 * (1 << z);
        let px = 2.0 * HALF / n as f64;
        ([-HALF, px, 0.0, HALF, 0.0, -px], [n as i32, n as i32])
    }

    fn tile_gt(extent: f64) -> [f64; 6] {
        let px = extent / 256.0;
        [-extent / 2.0, px, 0.0, extent / 2.0, 0.0, -px]
    }

    /// Synthetic source: R = column gradient, G = row gradient, B = 128, A = 255.
    fn synthetic(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for r in 0..h {
            for c in 0..w {
                let i = (r * w + c) * 4;
                v[i] = (c * 255 / w.max(1)) as u8;
                v[i + 1] = (r * 255 / h.max(1)) as u8;
                v[i + 2] = 128;
                v[i + 3] = 255;
            }
        }
        v
    }

    #[test]
    fn full_tile_is_opaque_and_gradients_survive() {
        let (src_gt, src_size) = webmerc(8);
        let core = WarperCore::new("EPSG:3857", src_gt, LAEA_TAS, tile_gt(400_000.0), 0.125).unwrap();
        let win = core.source_window([256, 256], src_size, 1).unwrap();
        let (w, h) = (win.xsize as usize, win.ysize as usize);
        let src = synthetic(w, h);

        for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
            let out = core
                .warp_rgba(&src, w, h, win.xoff as usize, win.yoff as usize, 256, 256, alg)
                .unwrap();
            assert_eq!(out.len(), 256 * 256 * 4);
            let opaque = out.chunks(4).filter(|p| p[3] == 255).count();
            assert_eq!(opaque, 256 * 256, "{alg:?}: {} transparent pixels", 256 * 256 - opaque);
            // Blue is constant everywhere in the source, so it must be in the output.
            assert!(out.chunks(4).all(|p| p[2] == 128), "{alg:?}: blue not preserved");
            // Red increases left to right along the centre row.
            let row: Vec<u8> = (0..256).map(|c| out[(128 * 256 + c) * 4]).collect();
            assert!(row[255] > row[0] + 100, "{alg:?}: red gradient lost {} -> {}", row[0], row[255]);
            // Green increases top to bottom along the centre column.
            let col: Vec<u8> = (0..256).map(|r| out[(r * 256 + 128) * 4 + 1]).collect();
            assert!(col[255] > col[0] + 100, "{alg:?}: green gradient lost {} -> {}", col[0], col[255]);
        }
    }

    #[test]
    fn partial_source_gives_transparent_pixels() {
        // Hand the warper only the left half of the window it asked for.
        let (src_gt, src_size) = webmerc(8);
        let core = WarperCore::new("EPSG:3857", src_gt, LAEA_TAS, tile_gt(400_000.0), 0.125).unwrap();
        let win = core.source_window([256, 256], src_size, 1).unwrap();
        let (w, h) = (win.xsize as usize / 2, win.ysize as usize);
        let src = synthetic(w, h);
        let out = core
            .warp_rgba(&src, w, h, win.xoff as usize, win.yoff as usize, 256, 256, ResampleAlg::NearestNeighbour)
            .unwrap();
        let transparent = out.chunks(4).filter(|p| p[3] == 0).count();
        assert!(transparent > 256 * 100 && transparent < 256 * 156, "transparent = {transparent}");
        // Right side of the tile is the missing half.
        assert_eq!(out[(128 * 256 + 250) * 4 + 3], 0);
        assert_eq!(out[(128 * 256 + 5) * 4 + 3], 255);
    }

    #[test]
    fn tile_off_the_world_has_no_window() {
        // A tile 30,000 km from the LAEA origin is outside the projection domain.
        let (src_gt, src_size) = webmerc(8);
        let far = [30_000_000.0, 1000.0, 0.0, 30_000_000.0, 0.0, -1000.0];
        let core = WarperCore::new("EPSG:3857", src_gt, LAEA_TAS, far, 0.125).unwrap();
        let win = core.source_window([256, 256], src_size, 1);
        assert!(win.map_or(true, |w| w.n_failed == w.n_samples || w.xsize <= 0));
    }

    #[test]
    fn buffer_size_is_checked() {
        let (src_gt, _) = webmerc(8);
        let core = WarperCore::new("EPSG:3857", src_gt, LAEA_TAS, tile_gt(400_000.0), 0.125).unwrap();
        assert!(core.warp_rgba(&[0u8; 10], 4, 4, 0, 0, 2, 2, ResampleAlg::NearestNeighbour).is_err());
    }

    #[test]
    fn lonlat_helpers_roundtrip() {
        let p = lonlat_to_crs(LAEA_TAS, 147.3257, -42.8826).unwrap();
        assert!((p[0] - 26_700.0).abs() < 500.0 && (p[1] - -98_000.0).abs() < 500.0, "{p:?}");
        let ll = crs_to_lonlat(LAEA_TAS, p[0], p[1]).unwrap();
        assert!((ll[0] - 147.3257).abs() < 1e-6 && (ll[1] - -42.8826).abs() < 1e-6);
        assert!(lonlat_to_crs("+proj=nonsense", 0.0, 0.0).is_none());
    }

    #[test]
    fn alg_names() {
        assert_eq!(parse_alg("Nearest").unwrap(), ResampleAlg::NearestNeighbour);
        assert_eq!(parse_alg("bilinear").unwrap(), ResampleAlg::Bilinear);
        assert!(parse_alg("mode").is_err());
    }
}
