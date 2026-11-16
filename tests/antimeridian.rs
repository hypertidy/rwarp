//! The exact footprint across a geographic source's seam and around a pole
//! (stage 3 of `design/curvilinear-frontend.md`, "The exact footprint at
//! the antimeridian"): a global longitude/latitude grid to a south polar
//! stereographic grid, as in #11's comparison, where the column of pixels
//! across 180 degrees and the pixel holding the pole were wrong.
//!
//! Oracles: the same data stored as -180..180 and as 0..360 (each puts its
//! seam on a different meridian) give the same weights; each destination
//! pixel's weights sum to its outline's area; the pole pixel's coverage is
//! the polar cap under its outline, integrated independently.

#![cfg(any(feature = "proj", feature = "proj4rs"))]

use std::collections::HashMap;

use rwarp::approx::ApproxTransformer;
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{warp, Footprint, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, GridWindow, WeightOptions};
use rwarp::Transformer;

/// 41 x 41 pixels of 100 km around the South Pole: the pole is the centre
/// of pixel (20, 20), and the column through it crosses 0 and 180 degrees.
const DST_N: usize = 41;
const DST_GT: [f64; 6] = [-2.05e6, 1e5, 0.0, 2.05e6, 0.0, -1e5];

/// A global 1 degree grid starting at `west`.
fn source(west: f64) -> GenImgProjTransformer {
    GenImgProjTransformer::new("EPSG:4326", [west, 1.0, 0.0, 90.0, 0.0, -1.0], "EPSG:3031", DST_GT)
        .unwrap()
        .with_source_size([360, 180])
}

type Table = HashMap<(u32, u32, u32, u32), f64>;

/// Weights keyed by (dst row, dst col, src row, src col), with source
/// columns moved by `col_shift` (mod 360), and each destination cell's area.
fn table(t: &impl Transformer, densify: usize, col_shift: u32) -> (Table, Vec<f64>) {
    let w = exact_weights(t, &GridWindow::new(360, 180), DST_N, DST_N, &WeightOptions::new(densify));
    let mut m = HashMap::new();
    for k in 0..w.len() {
        let key = (w.dst_row[k], w.dst_col[k], w.src_row[k], (w.src_col[k] + col_shift) % 360);
        assert!(m.insert(key, w.frac[k]).is_none(), "{key:?} twice");
    }
    (m, w.dst_area)
}

#[test]
fn source_wrap_is_reported_for_wrapped_geographic_sources() {
    let t = source(-180.0);
    let w = t.source_wrap().expect("a global lon/lat source wraps");
    assert_eq!(w.period, 360.0);
    assert_eq!(w.pole_rows, [0.0, 180.0]);
    assert_eq!(ApproxTransformer::new(source(-180.0), 0.125).source_wrap(), Some(w));
    // a half-degree source: 720 pixels a turn
    let half = GenImgProjTransformer::new("EPSG:4326", [-180.0, 0.5, 0.0, 90.0, 0.0, -0.5], "EPSG:3031", DST_GT)
        .unwrap()
        .with_source_size([720, 360]);
    assert_eq!(half.source_wrap().unwrap().period, 720.0);
    // not without the source size (no CENTER_LONG), not for a projected
    // source, not for a source wider than 360 degrees
    let no_size = GenImgProjTransformer::new("EPSG:4326", [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0], "EPSG:3031", DST_GT).unwrap();
    assert!(no_size.source_wrap().is_none());
    let projected = GenImgProjTransformer::new("EPSG:3031", DST_GT, "EPSG:4326", [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0])
        .unwrap()
        .with_source_size([41, 41]);
    assert!(projected.source_wrap().is_none());
    let wide = GenImgProjTransformer::new("EPSG:4326", [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0], "EPSG:3031", DST_GT)
        .unwrap()
        .with_source_size([400, 180]);
    assert!(wide.source_wrap().is_none());
}

#[test]
fn seam_shifted_sources_give_the_same_weights() {
    for densify in [1, 4] {
        let (a, area_a) = table(&source(-180.0), densify, 0);
        // column c of the 0..360 source is column c + 180 of the -180..180 one
        let (b, area_b) = table(&source(0.0), densify, 180);
        let mut worst: f64 = 0.0;
        for (key, fa) in &a {
            let fb = b.get(key).copied().unwrap_or(0.0);
            worst = worst.max((fa - fb).abs());
        }
        for (key, fb) in &b {
            if !a.contains_key(key) {
                worst = worst.max(fb.abs());
            }
        }
        eprintln!("densify {densify}: {} and {} entries, max |a - b| {worst:.1e}", a.len(), b.len());
        assert!(worst < 1e-9, "densify {densify}: {worst}");
        for (k, (x, y)) in area_a.iter().zip(&area_b).enumerate() {
            assert!((x - y).abs() < 1e-9 * x.max(1.0), "cell {k}: area {x} vs {y}");
        }
    }
}

#[test]
fn every_pixel_s_weights_sum_to_its_area() {
    // the planar measure in source pixels: each source cell has area 1, so
    // the weights of a pixel inside the source sum to its outline's area,
    // across the seam and around the pole too
    for (west, shift) in [(-180.0, 0), (0.0, 180)] {
        let (t, area) = table(&source(west), 4, shift);
        let mut sum = vec![0.0; DST_N * DST_N];
        for (&(r, c, _, _), &f) in &t {
            assert!(f > 0.0 && f <= 1.0 + 1e-9, "({r}, {c}): fraction {f}");
            sum[r as usize * DST_N + c as usize] += f;
        }
        for (k, (s, a)) in sum.iter().zip(&area).enumerate() {
            assert!((s - a).abs() < 1e-9 * a.max(1.0), "west {west}, cell ({}, {}): sum {s} area {a}", k / DST_N, k % DST_N);
        }
    }
}

#[test]
fn the_pole_pixel_covers_the_polar_cap() {
    // pixel (20, 20) holds the South Pole: its outline goes once around
    // it, and in source pixels its footprint is the cap between the
    // outline and the pole's row, 360 columns wide
    let t = source(-180.0);
    // the outline, very finely, to source pixels: the cap's area is the
    // integral of (pole row - y) dx along it, unwrapped
    let n = 400_000;
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        let s = 4.0 * k as f64 / n as f64;
        let (e, f) = match s as usize {
            0 => (s, 0.0),
            1 => (1.0, s - 1.0),
            2 => (3.0 - s, 1.0),
            _ => (0.0, 4.0 - s),
        };
        x.push(20.0 + e);
        y.push(20.0 + f);
    }
    assert!(t.transform(true, &mut x, &mut y).iter().all(|&ok| ok));
    let mut cap = 0.0;
    let mut prev = x[0];
    let mut ux = x[0];
    for k in 0..n {
        let k1 = (k + 1) % n;
        let d = x[k1] - prev;
        let d = d - (d / 360.0).round() * 360.0;
        prev = x[k1];
        let ux1 = ux + d;
        cap += 0.5 * ((180.0 - y[k]) + (180.0 - y[k1])) * (ux1 - ux);
        ux = ux1;
    }
    let cap = cap.abs();
    assert!((ux - x[0]).abs() > 359.0, "the outline goes round the pole");

    // the densified outline is straight between its points in source
    // pixels, so it converges to the cap at second order: the error falls
    // by 4 each time densify doubles (0.226, 0.057, 0.014, 0.0035 pixels
    // at 16, 32, 64, 128)
    let mut errs = Vec::new();
    for densify in [16, 32, 64] {
        let (w, area) = table(&t, densify, 0);
        let got: f64 = w.iter().filter(|(k, _)| (k.0, k.1) == (20, 20)).map(|(_, f)| f).sum();
        let cells = w.keys().filter(|k| (k.0, k.1) == (20, 20)).count();
        let pole_area = area[20 * DST_N + 20];
        eprintln!("densify {densify}: cap {cap:.6} source pixels, weights {got:.6} over {cells} cells, outline area {pole_area:.6}");
        let err = got - cap;
        assert!(err > 0.0 && err < 0.4 * cap / (densify * densify) as f64, "densify {densify}: {got} vs {cap}");
        assert!((pole_area - got).abs() < 1e-9 * cap);
        // the cap runs the whole way round: every column, down to the pole
        let cols: std::collections::HashSet<u32> = w.keys().filter(|k| (k.0, k.1) == (20, 20)).map(|k| k.3).collect();
        assert_eq!(cols.len(), 360);
        errs.push(err);
    }
    for e in errs.windows(2) {
        let order = e[0] / e[1];
        assert!((3.5..4.5).contains(&order), "error ratio {order}, not second order");
    }
}

#[test]
fn average_across_the_seam_and_at_the_pole() {
    // the same field stored both ways; the exact-footprint average must
    // agree everywhere (weights are rounded to f32, so to about 1e-6)
    let field = |lon: f64, lat: f64| 100.0 + 10.0 * (2.0 * lon.to_radians()).sin() * lat.to_radians().cos() + 0.3 * lat;
    let data = |west: f64| -> Vec<f64> {
        let mut v = Vec::with_capacity(360 * 180);
        for r in 0..180 {
            for c in 0..360 {
                v.push(field(west + c as f64 + 0.5, 89.5 - r as f64));
            }
        }
        v
    };
    let opts = WarpOptions::new(ResampleAlg::Average)
        .with_footprint(Footprint::Exact { densify: 4 })
        .with_src_nodata(Nodata::None);
    let mut out = Vec::new();
    for west in [-180.0, 0.0] {
        let d = data(west);
        let buf = SourceBuffer { data: &d, ncol: 360, nrow: 180, nbands: 1, col_off: 0, row_off: 0 };
        out.push(warp(&source(west), &buf, DST_N, DST_N, &opts));
    }
    let mut worst: f64 = 0.0;
    for (a, b) in out[0].iter().zip(&out[1]) {
        assert!(a.is_finite() && b.is_finite());
        worst = worst.max((a - b).abs() / a.abs());
    }
    eprintln!("average: max relative difference {worst:.1e}");
    assert!(worst < 2e-6, "{worst}");
}
