//! Cutline parity with `gdalwarp -cutline` on synthetic fixtures, and
//! analytic checks of the exact rule.
//!
//! The fixtures in `tests/fixtures/cutline/` are gdalwarp output, written by
//! `gen.py` there (which records the GDAL version). Most cases have no CRS,
//! so the transform is affine; the `*_reproj*` cases go from LAEA to
//! lon/lat.

use std::path::PathBuf;

use rwarp::cutline::{Cutline, CutlineRule, Ring};
use rwarp::warp::{warp, AverageThresholds, DstNodata, Footprint, KernelScaling, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::Transformer;

const N: usize = 64;
const NODATA: f32 = -9999.0;
const SRC_GT: [f64; 6] = [0.0, 1.0, 0.0, N as f64, 0.0, -1.0];

struct Grid {
    header: Vec<String>,
    ncol: usize,
    nrow: usize,
    vals: Vec<f32>,
}

impl Grid {
    fn header(&self, key: &str) -> &str {
        self.header
            .iter()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
            .unwrap_or_else(|| panic!("no '{key}' in fixture header"))
    }

    fn gt(&self, key: &str) -> [f64; 6] {
        let v: Vec<f64> = self.header(key).split_whitespace().map(|v| v.parse().unwrap()).collect();
        v.try_into().unwrap()
    }
}

fn read_grid(name: &str) -> Grid {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "cutline", &format!("{name}.txt")]
        .iter()
        .collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut header = Vec::new();
    let mut lines = text.lines();
    let dims = loop {
        let l = lines.next().expect("missing dimensions");
        match l.strip_prefix("# ") {
            Some(h) => header.push(h.to_string()),
            None => break l,
        }
    };
    let mut dims = dims.split_whitespace().map(|v| v.parse::<usize>().unwrap());
    let (ncol, nrow) = (dims.next().unwrap(), dims.next().unwrap());
    let vals: Vec<f32> = lines.flat_map(|l| l.split_whitespace()).map(|v| v.parse::<f64>().unwrap() as f32).collect();
    assert_eq!(vals.len(), ncol * nrow, "{name}: value count");
    Grid { header, ncol, nrow, vals }
}

/// The fixture's `cut`: JSON nested lists, polygons of rings of `[x, y]`.
fn parse_cut(s: &str) -> Vec<Vec<Ring>> {
    // Depth-first over brackets: depth 4 holds numbers.
    let mut polys: Vec<Vec<Ring>> = Vec::new();
    let mut depth = 0;
    let mut num = String::new();
    let mut pt: Vec<f64> = Vec::new();
    for ch in s.chars() {
        match ch {
            '[' => {
                depth += 1;
                match depth {
                    2 => polys.push(Vec::new()),
                    3 => polys.last_mut().unwrap().push(Vec::new()),
                    _ => {}
                }
            }
            ']' | ',' => {
                if !num.trim().is_empty() {
                    pt.push(num.trim().parse().unwrap());
                }
                num.clear();
                if ch == ']' {
                    if depth == 4 {
                        polys.last_mut().unwrap().last_mut().unwrap().push([pt[0], pt[1]]);
                        pt.clear();
                    }
                    depth -= 1;
                }
            }
            c => num.push(c),
        }
    }
    polys
}

/// Destination pixel -> source pixel through the geotransforms: the
/// destination's `gt`, then the inverse of the source's.
struct Affine {
    gt: [f64; 6],
}

impl Transformer for Affine {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        assert!(dst_to_src);
        let g = &self.gt;
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            let (gx, gy) = (g[0] + *x * g[1] + *y * g[2], g[3] + *x * g[4] + *y * g[5]);
            *x = gx;
            *y = N as f64 - gy;
        }
        vec![true; x.len()]
    }
}

fn alg_of(name: &str) -> ResampleAlg {
    match name {
        "near" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "lanczos" => ResampleAlg::Lanczos,
        "cubicspline" => ResampleAlg::CubicSpline,
        "average" => ResampleAlg::Average,
        "mode" => ResampleAlg::Mode,
        "max" => ResampleAlg::Max,
        "med" => ResampleAlg::Med,
        "rms" => ResampleAlg::Rms,
        _ => panic!("unknown alg {name}"),
    }
}

fn options(expect: &Grid, cutline: Cutline) -> WarpOptions {
    let holes = expect.header("nodata_src") == "True";
    let mut opts = WarpOptions::new(alg_of(expect.header("alg")))
        .with_src_nodata(if holes { Nodata::Value(NODATA as f64) } else { Nodata::None })
        .with_dst_nodata(if expect.header("dst_nodata") == "True" { DstNodata::Value(NODATA as f64) } else { DstNodata::None });
    match expect.header("wo") {
        "-" => {}
        "XSCALE=1 YSCALE=1" => opts = opts.with_scaling(KernelScaling::Point),
        "NODATA_VALUES_PCT_THRESHOLD=50" => {
            opts = opts.with_average(AverageThresholds::default().with_nodata_pct_threshold(50.0))
        }
        other => panic!("unhandled warp options {other}"),
    }
    let rule = match expect.header("rule") {
        "centre" => CutlineRule::Centre,
        "all_touched" => CutlineRule::AllTouched,
        other => panic!("rule {other}"),
    };
    let blend: f64 = expect.header("blend").parse().unwrap();
    opts.with_cutline(cutline.with_rule(rule).with_blend_dist(blend))
}

/// rwarp's output for an affine fixture, and the fixture.
fn run_affine(name: &str) -> (Grid, Vec<f32>) {
    let expect = read_grid(name);
    let holes = expect.header("nodata_src") == "True";
    let src = read_grid(if holes { "src_nodata" } else { "src" });
    let cut = Cutline::from_geotransform(parse_cut(expect.header("cut")), SRC_GT).unwrap();
    let opts = options(&expect, cut);
    let buf = SourceBuffer { data: &src.vals[..], ncol: N, nrow: N, nbands: 1, col_off: 0, row_off: 0 };
    let got = warp(&Affine { gt: expect.gt("dst_gt") }, &buf, expect.ncol, expect.nrow, &opts);
    (expect, got)
}

/// Pixels further than `tol` (relative to max(1, |value|)) from GDAL's.
fn mismatches(expect: &Grid, got: &[f32], tol: f64) -> Vec<(usize, usize, f32, f32)> {
    expect
        .vals
        .iter()
        .zip(got)
        .enumerate()
        .filter(|(_, (&e, &g))| (e as f64 - g as f64).abs() / (e.abs() as f64).max(1.0) > tol)
        .map(|(i, (&e, &g))| (i % expect.ncol, i / expect.ncol, e, g))
        .collect()
}

const TOL: f64 = 1e-6;

macro_rules! parity {
    ($($case:ident),* $(,)?) => {$(
        #[test]
        fn $case() {
            let name = stringify!($case);
            let (expect, got) = run_affine(name);
            let bad = mismatches(&expect, &got, TOL);
            assert!(bad.is_empty(), "{name}: {} pixels differ, first (col, row, gdal, rwarp): {:?}",
                bad.len(), &bad[..bad.len().min(8)]);
        }
    )*};
}

// Which source pixels are in: identity nearest warps.
parity!(
    mask_rect_tie, mask_hole_tie, mask_hole_tie_rev, mask_u_tie, mask_tri,
    mask_star_hole, mask_multi,
    mask_at_rect_tie, mask_at_hole_tie, mask_at_u_tie, mask_at_tri,
    mask_at_star_hole, mask_at_multi, mask_at_aligned, mask_at_corner_diag,
);

// Every method, downsampled 2.78:1, with a concave cutline.
parity!(
    near_star, bilinear_star, cubic_star, lanczos_star, cubicspline_star, average_star,
    max_star, med_star, rms_star,
    bilinear_star_point, cubic_star_point, lanczos_star_point, cubicspline_star_point,
    cubic_star_up, bilinear_star_up,
);

// Holes, multipolygons, source nodata.
parity!(
    near_multi_nodata, bilinear_multi_nodata, cubic_multi_nodata, lanczos_multi_nodata,
    average_multi_nodata, bilinear_star_hole, average_star_hole,
    average_multi_nodata_pct,
);

// All touched.
parity!(near_star_at, bilinear_star_at, cubic_star_at, average_star_at);

// Blend distance, with and without a destination nodata.
parity!(
    near_star_blend, bilinear_star_blend, cubic_star_blend, lanczos_star_blend, average_star_blend,
    max_star_blend,
    near_star_blend_nodst, bilinear_star_blend_nodst, cubic_star_blend_nodst, lanczos_star_blend_nodst,
    cubicspline_star_blend_nodst,
    near_id_blend_nodst, bilinear_id_blend_nodst, bilinear_star_blend_at_nodst, bilinear_multi_blend_nodata,
);

// Rotated destination.
parity!(near_star_rot30, bilinear_star_rot30, cubic_star_rot30, average_star_rot30, bilinear_star_rot30_blend_nodst);

/// Centre rule on a diamond whose edges run exactly through pixel centres
/// (28 per edge): GDAL leaves the centres on both left edges out and those
/// on both right edges in, and so does rwarp. controlledburn 0.2.2 decided
/// ties from rounded intercepts and kept 5 of the upper-left edge's
/// centres; 0.2.3 decides them exactly.
#[test]
fn centre_ties_on_a_diagonal_follow_gdal() {
    let (expect, got) = run_affine("mask_diamond_tie");
    let bad = mismatches(&expect, &got, TOL);
    assert!(bad.is_empty(), "{bad:?}");
}

/// All touched, where GDAL's line walker differs from coverage:
///
/// - a pixel touched only at a corner: GDAL burns the pixel below-right of
///   each corner the boundary passes through, inside the polygon or not
///   (the lower-right edge of the diamond, x + y = 93, through the corner
///   of 28 pixels outside it); coverage gives those pixels no area;
/// - an edge within GDAL's 1e-4 of a pixel wall: GDAL leaves the pixels it
///   grazes out, coverage gives them a sliver and burns them;
/// - an edge GDAL treats as vertical (|dx| < 0.01 pixels): GDAL burns one
///   column (at the end with larger x) all along it, including rows where
///   the edge is in the column to the left and the polygon never reaches
///   the burned column.
#[test]
fn all_touched_differs_from_gdal_at_corners_and_wall_tolerances() {
    let (expect, got) = run_affine("mask_at_diamond_tie");
    let bad = mismatches(&expect, &got, TOL);
    assert!(bad.len() == 28 && bad.iter().all(|&(c, r, _, g)| g == NODATA && c + r == 93), "{bad:?}");

    let (expect, got) = run_affine("mask_at_near_wall");
    let bad = mismatches(&expect, &got, TOL);
    // The right edge runs from x = 29.99995 to 30.00002 and crosses into
    // column 30 at y = 32: rows 32 to 39 of column 30 have a sliver.
    assert!(bad.len() == 8 && bad.iter().all(|&(c, r, e, _)| e == NODATA && c == 30 && (32..40).contains(&r)), "{bad:?}");

    let (expect, got) = run_affine("mask_at_near_vertical");
    let bad = mismatches(&expect, &got, TOL);
    // x = 20.995 + 0.009 (y - 5.3) / 44.9 reaches 21 at y = 30.2.
    assert!(!bad.is_empty() && bad.iter().all(|&(c, r, _, g)| g == NODATA && c == 21 && r <= 30), "{bad:?}");
}

#[cfg(any(feature = "proj", feature = "proj4rs"))]
mod reproj {
    use super::*;
    use rwarp::transform::GenImgProjTransformer;

    fn run(name: &str) -> (Grid, Vec<f32>) {
        let expect = read_grid(name);
        let src = read_grid("src");
        let (src_crs, dst_crs) = (expect.header("src_crs"), expect.header("dst_crs"));
        let src_gt = expect.gt("src_gt");
        let polys = parse_cut(expect.header("cut"));
        let cut = match expect.header("crs") {
            "src" => Cutline::from_geotransform(polys, src_gt).unwrap(),
            _ => {
                let t = GenImgProjTransformer::new(src_crs, src_gt, dst_crs, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
                Cutline::from_transformer(polys, &t).unwrap()
            }
        };
        let opts = options(&expect, cut);
        let t = GenImgProjTransformer::new(src_crs, src_gt, dst_crs, expect.gt("dst_gt")).unwrap();
        let buf = SourceBuffer { data: &src.vals[..], ncol: N, nrow: N, nbands: 1, col_off: 0, row_off: 0 };
        let got = warp(&t, &buf, expect.ncol, expect.nrow, &opts);
        (expect, got)
    }

    // Identical to gdalwarp here with libproj 9 and with proj4rs, the
    // densified lon/lat cutline included. The CRS step can differ slightly
    // between PROJ builds, so allow a sample point on a pixel edge to land
    // in the neighbour.
    macro_rules! reproj {
        ($($case:ident),* $(,)?) => {$(
            #[test]
            fn $case() {
                let name = stringify!($case);
                let (expect, got) = run(name);
                let bad = mismatches(&expect, &got, 1e-4);
                assert!(bad.len() <= 2, "{name}: {} pixels differ: {:?}", bad.len(), &bad[..bad.len().min(8)]);
                let valid = expect.vals.iter().filter(|&&v| v != NODATA).count();
                assert!(valid > 200, "{name}: only {valid} valid pixels");
            }
        )*};
    }

    reproj!(near_reproj, bilinear_reproj, average_reproj, near_reproj_dstcut, bilinear_reproj_dstcut);
}

// ---- the exact rule -------------------------------------------------------

/// A source of 1 on the left half and 3 on the right; with a cutline that
/// takes half of each of two pixels, an average over them is analytic.
#[test]
fn exact_half_pixel_coverage_is_half_weight_in_average() {
    // 4 x 1 source: values 1, 1, 3, 3.
    let src = [1.0f32, 1.0, 3.0, 3.0];
    let buf = SourceBuffer { data: &src[..], ncol: 4, nrow: 1, nbands: 1, col_off: 0, row_off: 0 };
    // One destination pixel over the whole source.
    let t = Scale { sx: 4.0, sy: 1.0 };
    // Cutline covering x in [1.5, 4]: pixel 1 half, pixels 2 and 3 fully.
    let cut = Cutline::new(vec![vec![vec![[1.5, -1.0], [4.0, -1.0], [4.0, 2.0], [1.5, 2.0]]]]);
    let run = |rule| {
        warp(&t, &buf, 1, 1, &WarpOptions::new(ResampleAlg::Average).with_cutline(cut.clone().with_rule(rule)))[0]
    };
    // (0.5 * 1 + 3 + 3) / 2.5 = 2.6
    assert!((run(CutlineRule::Exact) - 2.6).abs() < 1e-6);
    // Centre rule: pixel 1's centre (1.5) is on the cutline's left edge: out.
    assert_eq!(run(CutlineRule::Centre), 3.0);
    // All touched: pixel 1 counts in full: (1 + 3 + 3) / 3.
    assert!((run(CutlineRule::AllTouched) - 7.0 / 3.0).abs() < 1e-6);
}

#[test]
fn exact_average_is_the_mean_over_the_polygon() {
    // A linear ramp v = x + 10 y; the cutline is a triangle, and the exact
    // rule with the exact footprint gives the polygon's mean value up to
    // the per-pixel approximation (the value is constant over a pixel), so
    // compare with the mean of pixel values weighted by coverage.
    let (n, m) = (16usize, 12usize);
    let src: Vec<f64> = (0..n * m).map(|i| (i % n) as f64 + 10.0 * (i / n) as f64).collect();
    let buf = SourceBuffer { data: &src[..], ncol: n, nrow: m, nbands: 1, col_off: 0, row_off: 0 };
    let tri = vec![vec![vec![[1.3, 2.2], [14.1, 4.7], [5.9, 10.6]]]];
    let cut = Cutline::new(tri).with_rule(CutlineRule::Exact);
    let cov = cut.density(0, 0, n, m);
    let (mut s, mut w) = (0.0, 0.0);
    for (i, &c) in cov.iter().enumerate() {
        s += src[i] * c as f64;
        w += c as f64;
    }
    // Area of the triangle.
    let area: f64 = 0.5 * ((14.1 - 1.3) * (10.6 - 2.2) - (5.9 - 1.3) * (4.7 - 2.2f64)).abs();
    assert!((w - area).abs() < 1e-4, "{w} {area}");
    let t = Scale { sx: n as f64, sy: m as f64 };
    for footprint in [Footprint::GdalBox, Footprint::Exact { densify: 1 }] {
        let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(footprint).with_cutline(cut.clone());
        let got = warp(&t, &buf, 1, 1, &opts)[0];
        assert!((got - s / w).abs() < 1e-5, "{got} {}", s / w);
    }
}

#[test]
fn exact_point_kernels_weight_taps_by_coverage() {
    // Two columns, 0 and 10; bilinear halfway between them gives 5. With
    // the right column half covered its tap weighs half: 10 * 0.25 / 0.75.
    let src = [0.0f32, 10.0, 0.0, 10.0];
    let buf = SourceBuffer { data: &src[..], ncol: 2, nrow: 2, nbands: 1, col_off: 0, row_off: 0 };
    // One destination pixel, its centre at source (1, 1).
    struct At;
    impl Transformer for At {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            x.fill(1.0);
            y.fill(1.0);
            vec![true; x.len()]
        }
    }
    let cut = Cutline::new(vec![vec![vec![[0.0, 0.0], [1.5, 0.0], [1.5, 2.0], [0.0, 2.0]]]]).with_rule(CutlineRule::Exact);
    let opts = WarpOptions::new(ResampleAlg::Bilinear).with_scaling(KernelScaling::Point).with_cutline(cut);
    let got = warp(&At, &buf, 1, 1, &opts)[0];
    assert!((got - 10.0 / 3.0).abs() < 1e-6, "{got}");
}

#[test]
fn exact_nearest_takes_pixels_at_least_half_covered() {
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let buf = SourceBuffer { data: &src[..], ncol: 4, nrow: 1, nbands: 1, col_off: 0, row_off: 0 };
    let t = Scale { sx: 1.0, sy: 1.0 };
    // Covers 0.6 of pixel 1 and 0.4 of pixel 3.
    let cut = Cutline::new(vec![vec![vec![[1.4, 0.0], [3.4, 0.0], [3.4, 1.0], [1.4, 1.0]]]]).with_rule(CutlineRule::Exact);
    let opts = WarpOptions::new(ResampleAlg::NearestNeighbour).with_dst_nodata(DstNodata::Value(0.0)).with_cutline(cut);
    assert_eq!(warp(&t, &buf, 4, 1, &opts), [0.0, 2.0, 3.0, 0.0]);
}

#[test]
fn cutline_outside_or_containing_the_window() {
    let src = [5.0f32; 16];
    // A window at (10, 10) of the full source.
    let buf = SourceBuffer { data: &src[..], ncol: 4, nrow: 4, nbands: 1, col_off: 10, row_off: 10 };
    struct Off;
    impl Transformer for Off {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            x.iter_mut().for_each(|v| *v += 10.0);
            y.iter_mut().for_each(|v| *v += 10.0);
            vec![true; x.len()]
        }
    }
    let far = Cutline::new(vec![vec![vec![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]]]);
    let opts = WarpOptions::new(ResampleAlg::Bilinear).with_dst_nodata(DstNodata::Value(-1.0));
    assert_eq!(warp(&Off, &buf, 4, 4, &opts.clone().with_cutline(far)), [-1.0; 16]);
    let all = Cutline::new(vec![vec![vec![[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]]]);
    assert_eq!(warp(&Off, &buf, 4, 4, &opts.with_cutline(all)), [5.0; 16]);
}

/// Destination pixel -> source pixel by scaling: a destination of
/// `ncol / sx` by `nrow / sy` over the source.
struct Scale {
    sx: f64,
    sy: f64,
}

impl Transformer for Scale {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut().for_each(|v| *v *= self.sx);
        y.iter_mut().for_each(|v| *v *= self.sy);
        vec![true; x.len()]
    }
}
