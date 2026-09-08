//! End-to-end: one slippy-map tile in a local LAEA, sourced from a Web
//! Mercator tile pyramid, on the pure-Rust `proj4rs` backend.
//!
//! This is the core loop of a browser reprojecting-tile app:
//!   target tile grid -> ApproxTransformer -> compute_source_window
//!   -> (fetch source tiles) -> warp_resample
//!
//! Run with: cargo test --no-default-features --features proj4rs

#![cfg(feature = "proj4rs")]

use rwarp::approx::ApproxTransformer;
use rwarp::source_window::compute_source_window;
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{warp_resample, ResampleAlg};
use rwarp::Transformer;

const WEBMERC_HALF: f64 = 20037508.342789244;

/// Web Mercator at zoom `z` as a single virtual raster.
fn webmerc_grid(z: u32) -> ([f64; 6], [i32; 2]) {
    let n = 256 * (1 << z);
    let px = 2.0 * WEBMERC_HALF / n as f64;
    ([-WEBMERC_HALF, px, 0.0, WEBMERC_HALF, 0.0, -px], [n as i32, n as i32])
}

const LAEA_TAS: &str = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";

/// A 256 px tile covering `extent_m` metres centred on the LAEA origin.
fn laea_tile_gt(extent_m: f64) -> [f64; 6] {
    let px = extent_m / 256.0;
    [-extent_m / 2.0, px, 0.0, extent_m / 2.0, 0.0, -px]
}

#[test]
fn laea_tile_source_window_lands_on_tasmania() {
    let z = 8;
    let (src_gt, src_size) = webmerc_grid(z);
    let dst_gt = laea_tile_gt(400_000.0);

    let exact = GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, dst_gt).unwrap();
    let approx = ApproxTransformer::new(exact, 0.125);

    let win = compute_source_window(&approx, [0, 0], [256, 256], src_size, 1)
        .expect("window");

    assert_eq!(win.n_failed, 0, "{win:?}");
    assert!(win.fill_ratio > 0.99, "tile fully inside source: {win:?}");

    // Where should Tasmania be at z8? lon 147 -> x = 16364000 m; lat -42 -> y = -5161000 m.
    let px = src_gt[1];
    let exp_col = ((16_364_000.0 + WEBMERC_HALF) / px) as i32;
    let exp_row = ((WEBMERC_HALF - -5_161_000.0) / px) as i32;
    let cx = win.xoff + win.xsize / 2;
    let cy = win.yoff + win.ysize / 2;
    assert!((cx - exp_col).abs() < 20, "centre col {cx} vs {exp_col}: {win:?}");
    assert!((cy - exp_row).abs() < 20, "centre row {cy} vs {exp_row}: {win:?}");

    // 400 km at ~611 m/px is ~655 px before Mercator stretch at 42S and
    // GDAL's safety margin on the window. Sanity bounds, not exact.
    assert!(win.xsize > 650 && win.xsize < 1100, "{win:?}");
    assert!(win.ysize > 650 && win.ysize < 1100, "{win:?}");
}

#[test]
fn approx_matches_exact_within_threshold() {
    let (src_gt, _) = webmerc_grid(8);
    let dst_gt = laea_tile_gt(400_000.0);
    let exact = GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, dst_gt).unwrap();
    let approx = ApproxTransformer::new(
        GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, dst_gt).unwrap(),
        0.125,
    );

    for row in [0usize, 100, 255] {
        let mut xe: Vec<f64> = (0..256).map(|c| c as f64 + 0.5).collect();
        let mut ye = vec![row as f64 + 0.5; 256];
        let mut xa = xe.clone();
        let mut ya = ye.clone();
        let oke = exact.transform(true, &mut xe, &mut ye);
        let oka = approx.transform(true, &mut xa, &mut ya);
        for i in 0..256 {
            assert!(oke[i] && oka[i]);
            assert!((xe[i] - xa[i]).abs() <= 0.125, "row {row} col {i}: dx {}", xe[i] - xa[i]);
            assert!((ye[i] - ya[i]).abs() <= 0.125, "row {row} col {i}: dy {}", ye[i] - ya[i]);
        }
    }
}

#[test]
fn nearest_warp_of_gradient_is_monotone_across_tile() {
    // Source: the window we would have fetched, filled with a column-index
    // gradient. After warping into LAEA, x should still increase left to
    // right along the tile's centre row.
    let (src_gt, src_size) = webmerc_grid(8);
    let dst_gt = laea_tile_gt(400_000.0);
    let approx = ApproxTransformer::new(
        GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, dst_gt).unwrap(),
        0.125,
    );
    let win = compute_source_window(&approx, [0, 0], [256, 256], src_size, 1).unwrap();

    let (w, h) = (win.xsize as usize, win.ysize as usize);
    let src: Vec<i32> = (0..w * h).map(|i| (win.xoff as usize + i % w) as i32).collect();

    let out = warp_resample(
        &approx, &src, w, h, win.xoff as usize, win.yoff as usize,
        256, 256, -1, ResampleAlg::NearestNeighbour,
    );
    assert_eq!(out.len(), 256 * 256);
    assert!(out.iter().all(|&v| v != -1), "no nodata inside a fully-covered tile");

    let mid = &out[128 * 256..129 * 256];
    assert!(mid.windows(2).all(|p| p[1] >= p[0]), "centre row monotone: {:?}", &mid[..8]);
    assert!(mid[255] - mid[0] > 600, "spans most of the window: {} .. {}", mid[0], mid[255]);
}

/// Before/after for the interleaved kernel. `cargo test --release --features proj4rs --no-default-features -- --ignored --nocapture timing`
#[test]
#[ignore]
fn timing_per_band_vs_interleaved() {
    use rwarp::warp::warp_resample_u8;
    let (src_gt, src_size) = webmerc_grid(8);
    let approx = ApproxTransformer::new(
        GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, laea_tile_gt(400_000.0)).unwrap(), 0.125);
    let win = compute_source_window(&approx, [0, 0], [256, 256], src_size, 1).unwrap();
    let (w, h) = (win.xsize as usize, win.ysize as usize);
    let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i % 253) as u8).collect();
    let (xo, yo) = (win.xoff as usize, win.yoff as usize);
    for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::Lanczos] {
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let mut band = vec![0i32; w * h];
            for b in 0..4 {
                for i in 0..w * h { band[i] = rgba[i * 4 + b] as i32; }
                let _ = warp_resample(&approx, &band, w, h, xo, yo, 256, 256, -1, alg);
            }
            if alg != ResampleAlg::NearestNeighbour {
                let probe = vec![255i32; w * h];
                let _ = warp_resample(&approx, &probe, w, h, xo, yo, 256, 256, -1, alg);
            }
        }
        let per_band = t.elapsed().as_secs_f64() * 100.0;
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = warp_resample_u8(&approx, &rgba, w, h, 4, xo, yo, 256, 256, Some(3), alg);
        }
        let inter = t.elapsed().as_secs_f64() * 100.0;
        eprintln!("{alg:?}: per-band {per_band:.2} ms  interleaved {inter:.2} ms  ({:.1}x)", per_band / inter);
    }
}
