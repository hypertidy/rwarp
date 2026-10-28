//! Kernel parity with gdalwarp on synthetic fixtures.
//!
//! The fixtures in `tests/fixtures/kernels/` are gdalwarp output, written by
//! `gen.py` there (which records the GDAL version). Source and destination
//! share an extent with no CRS, so the transform is affine and only the
//! kernels are under test.

use std::path::PathBuf;

use rwarp::warp::{
    estimate_scale, warp, AverageThresholds, BandNodata, Footprint, KernelScaling, ModeTies, ResampleAlg, Sample,
    SourceBuffer, WarpOptions,
};
use rwarp::Transformer;

const N: usize = 64;
const NODATA: f32 = -9999.0;

struct Grid {
    header: Vec<String>,
    ncol: usize,
    nrow: usize,
    vals: Vec<f32>,
}

impl Grid {
    fn header_opt(&self, key: &str) -> Option<&str> {
        self.header.iter().find_map(|l| l.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
    }

    fn header(&self, key: &str) -> &str {
        self.header_opt(key).unwrap_or_else(|| panic!("no '{key}' in fixture header"))
    }

    /// Bands in the fixture (written band after band).
    fn nbands(&self) -> usize {
        self.header_opt("bands").map_or(1, |v| v.parse().unwrap())
    }

    /// Values pixel-interleaved, as `warp` takes and returns them.
    fn interleaved(&self) -> Vec<f32> {
        let (n, nb) = (self.ncol * self.nrow, self.nbands());
        (0..n * nb).map(|i| self.vals[(i % nb) * n + i / nb]).collect()
    }
}

fn read_grid(name: &str) -> Grid {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "kernels", &format!("{name}.txt")]
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
    let vals: Vec<f32> = lines
        .flat_map(|l| l.split_whitespace())
        .map(|v| v.parse::<f64>().unwrap() as f32)
        .collect();
    let g = Grid { header, ncol, nrow, vals };
    assert_eq!(g.vals.len(), ncol * nrow * g.nbands(), "{name}: value count");
    g
}

/// Destination pixel -> source pixel through the geotransforms gdalwarp
/// uses: the destination's `gt`, then the inverse of the source's
/// `[0, 1, 0, 64, 0, -1]`.
struct Affine {
    gt: [f64; 6],
}

impl Affine {
    /// Destination covering the source extent at `dw` x `dh`.
    fn new(dw: usize, dh: usize) -> Self {
        Affine { gt: [0.0, N as f64 / dw as f64, 0.0, N as f64, 0.0, -(N as f64) / dh as f64] }
    }

    fn of(g: &Grid) -> Self {
        let v: Vec<f64> = g.header("dst_gt").split_whitespace().map(|v| v.parse().unwrap()).collect();
        Affine { gt: v.try_into().unwrap() }
    }
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
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "lanczos" => ResampleAlg::Lanczos,
        "average" => ResampleAlg::Average,
        "mode" => ResampleAlg::Mode,
        "near" => ResampleAlg::NearestNeighbour,
        "cubicspline" => ResampleAlg::CubicSpline,
        "min" => ResampleAlg::Min,
        "max" => ResampleAlg::Max,
        "med" => ResampleAlg::Med,
        "q1" => ResampleAlg::Q1,
        "q3" => ResampleAlg::Q3,
        "rms" => ResampleAlg::Rms,
        _ => panic!("unknown alg {name}"),
    }
}

/// Warp the fixture source the way the case's header says, returning
/// rwarp's output and gdalwarp's.
fn run_case(name: &str) -> (Grid, Vec<f32>) {
    let expect = read_grid(name);
    let holes = expect.header("nodata_src") == "True";
    let src = read_grid(if holes { "src_nodata" } else { "src" });
    let scaling = match expect.header("wo") {
        "-" => KernelScaling::WidenWhenDownsampling,
        "XSCALE=1 YSCALE=1" => KernelScaling::Point,
        other => panic!("unhandled warp options {other}"),
    };
    let buf = SourceBuffer { data: &src.vals[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions {
        alg: alg_of(expect.header("alg")),
        scaling,
        nodata: Some(NODATA),
        ..Default::default()
    };
    let got = warp(&Affine::of(&expect), &buf, expect.ncol, expect.nrow, &opts);
    (expect, got)
}

/// Assert every pixel within `tol` (relative to max(1, |value|)).
fn assert_close(name: &str, expect: &Grid, got: &[f32], tol: f64) {
    let mut worst = (0.0f64, 0usize);
    for (i, (&e, &g)) in expect.vals.iter().zip(got).enumerate() {
        let d = (e as f64 - g as f64).abs() / (e.abs() as f64).max(1.0);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    let (d, i) = worst;
    assert!(
        d <= tol,
        "{name}: pixel ({}, {}) gdal {} rwarp {} (relative {d:e})",
        i % expect.ncol, i / expect.ncol, expect.vals[i], got[i]
    );
}

// Float32 output: rwarp and GDAL accumulate in f64 in different orders (and
// GDAL's widened lanczos uses a sine recurrence), so allow a few f32 ulps.
const TOL: f64 = 1e-6;

macro_rules! parity {
    ($($case:ident),* $(,)?) => {$(
        #[test]
        fn $case() {
            let name = stringify!($case);
            let (expect, got) = run_case(name);
            assert_close(name, &expect, &got, TOL);
        }
    )*};
}

parity!(
    bilinear_4x, cubic_4x, lanczos_4x,
    bilinear_2p78x, cubic_2p78x, lanczos_2p78x,
    bilinear_aniso, cubic_aniso, lanczos_aniso,
    bilinear_1p49x, lanczos_1p49x,
    bilinear_4x_point, cubic_4x_point, lanczos_4x_point,
    bilinear_4x_nodata, cubic_4x_nodata, lanczos_4x_nodata,
    bilinear_rot30, lanczos_rot30,
    average_4x, average_2p78x, average_aniso, average_1p49x,
    average_upsample, average_overhang, average_4x_nodata, average_rot30,
);

#[test]
fn estimated_scale_matches_gdal() {
    for name in ["bilinear_4x", "bilinear_2p78x", "bilinear_aniso", "bilinear_1p49x", "bilinear_rot30", "average_upsample"] {
        let g = read_grid(name);
        let mut s = g.header("gdal_scale").split_whitespace().map(|v| v.parse::<f64>().unwrap());
        let (gx, gy) = (s.next().unwrap(), s.next().unwrap());
        let (sx, sy) = estimate_scale(&Affine::of(&g), g.ncol, g.nrow, 0, 0);
        // GDAL logs the scale with %f.
        assert!((sx - gx).abs() < 5e-7 && (sy - gy).abs() < 5e-7, "{name}: gdal ({gx}, {gy}) rwarp ({sx}, {sy})");
    }
}

#[test]
fn fixed_scale_equals_estimated_when_they_agree() {
    let (expect, _) = run_case("cubic_2p78x");
    let src = read_grid("src");
    let buf = SourceBuffer { data: &src.vals[..], ncol: N, nrow: N, nbands: 1, col_off: 0, row_off: 0 };
    let t = Affine::new(expect.ncol, expect.nrow);
    let (sx, sy) = estimate_scale(&t, expect.ncol, expect.nrow, 0, 0);
    let opts = |scaling| WarpOptions { alg: ResampleAlg::Cubic, scaling, nodata: Some(NODATA), ..Default::default() };
    let auto = warp(&t, &buf, expect.ncol, expect.nrow, &opts(KernelScaling::WidenWhenDownsampling));
    let fixed = warp(&t, &buf, expect.ncol, expect.nrow, &opts(KernelScaling::Fixed { x: sx, y: sy }));
    assert_eq!(auto, fixed);
}

#[test]
fn widening_changes_output_and_point_does_not_widen() {
    let src = read_grid("src");
    let buf = SourceBuffer { data: &src.vals[..], ncol: N, nrow: N, nbands: 1, col_off: 0, row_off: 0 };
    let t = Affine::new(16, 16);
    let opts = |scaling| WarpOptions { alg: ResampleAlg::Bilinear, scaling, nodata: Some(NODATA), ..Default::default() };
    let point = warp(&t, &buf, 16, 16, &opts(KernelScaling::Point));
    let fixed_one = warp(&t, &buf, 16, 16, &opts(KernelScaling::Fixed { x: 1.0, y: 1.0 }));
    let widened = warp(&t, &buf, 16, 16, &opts(KernelScaling::WidenWhenDownsampling));
    assert_eq!(point, fixed_one, "scale 1 is the point kernel");
    assert_ne!(point, widened);
}

// ---- average: exact footprint ---------------------------------------------

fn average(t: &impl Transformer, src: &Grid, dw: usize, dh: usize, footprint: Footprint) -> Vec<f32> {
    let buf = SourceBuffer { data: &src.vals[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions { alg: ResampleAlg::Average, footprint, nodata: Some(NODATA), ..Default::default() };
    warp(t, &buf, dw, dh, &opts)
}

/// Where the destination pixel maps to an axis-aligned rectangle, GDAL's
/// box is the exact footprint, so the two agree (up to the f32 coverage
/// fractions controlledburn stores).
#[test]
fn exact_equals_gdal_box_when_axis_aligned() {
    for name in ["average_4x", "average_2p78x", "average_aniso", "average_upsample", "average_4x_nodata"] {
        let g = read_grid(name);
        let src = read_grid(if g.header("nodata_src") == "True" { "src_nodata" } else { "src" });
        let t = Affine::of(&g);
        let exact = average(&t, &src, g.ncol, g.nrow, Footprint::Exact { densify: 1 });
        assert_close(name, &g, &exact, 1e-5);
    }
}

/// A source whose value is its column index, and one destination pixel
/// rotated 45 degrees into a diamond centred on x = 5: by symmetry the
/// exact area-weighted mean is 4.5. GDAL's box degenerates to the column
/// at x = 5 (the top-left and bottom-right corners share it) and gives 5.
#[test]
fn rotated_diamond_exact_vs_gdal_box() {
    let (w, h) = (10usize, 10usize);
    let vals: Vec<f32> = (0..w * h).map(|i| (i % w) as f32).collect();
    let src = Grid { header: vec![], ncol: w, nrow: h, vals };
    // A 4x4 pixel rotated 45 degrees. Through `Affine` (source y = 64 -
    // geo y) its corners land at source (5, 1), (7.83, 3.83), (2.17, 3.83)
    // and (5, 6.66): a diamond centred on x = 5, inside the 10x10 source.
    let s = 2f64.sqrt() * 2.0;
    let t = Affine { gt: [5.0, s, -s, N as f64 - 1.0, -s, -s] };
    let exact = average(&t, &src, 1, 1, Footprint::Exact { densify: 1 });
    let gdal_box = average(&t, &src, 1, 1, Footprint::GdalBox);
    assert!((exact[0] - 4.5).abs() < 1e-5, "exact {}", exact[0]);
    assert_eq!(gdal_box[0], 5.0);
}

/// Under an affine rotation the footprint is a quadrilateral, so densifying
/// its edges changes nothing; and GDAL's box differs from it.
#[test]
fn exact_under_rotation_is_densify_invariant_and_differs_from_box() {
    let src = read_grid("src");
    let g = read_grid("average_rot30");
    let t = Affine::of(&g);
    let a = average(&t, &src, g.ncol, g.nrow, Footprint::Exact { densify: 1 });
    let b = average(&t, &src, g.ncol, g.nrow, Footprint::Exact { densify: 8 });
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-4 * x.abs().max(1.0), "{x} vs {y}");
    }
    // And the box differs from the exact footprint under rotation.
    let boxed = average(&t, &src, g.ncol, g.nrow, Footprint::GdalBox);
    assert!(a.iter().zip(&boxed).any(|(x, y)| (x - y).abs() > 1.0));
}

/// Cost of the exact footprint against GDAL's box.
/// `cargo test --release --test kernels -- --ignored --nocapture timing`
#[test]
#[ignore]
fn timing_average_footprints() {
    let (w, h) = (1024usize, 1024usize);
    let vals: Vec<f32> = (0..w * h).map(|i| (i % 997) as f32).collect();
    let buf = SourceBuffer { data: &vals[..], ncol: w, nrow: h, nbands: 1, col_off: 0, row_off: 0 };
    // 256x256 over the 1024x1024 source (4:1), rotated slightly.
    struct Rot;
    impl Transformer for Rot {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            let (c, s) = (0.1f64.cos(), 0.1f64.sin());
            for (x, y) in x.iter_mut().zip(y.iter_mut()) {
                let (u, v) = (*x * 3.5, *y * 3.5);
                *x = 60.0 + c * u - s * v;
                *y = 60.0 + s * u + c * v;
            }
            vec![true; x.len()]
        }
    }
    for footprint in [Footprint::GdalBox, Footprint::Exact { densify: 1 }, Footprint::Exact { densify: 4 }] {
        let opts = WarpOptions { alg: ResampleAlg::Average, footprint, ..Default::default() };
        let t = std::time::Instant::now();
        let _ = warp(&Rot, &buf, 256, 256, &opts);
        eprintln!("{footprint:?}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

// ---- mode -----------------------------------------------------------------

fn ties_of(s: &str) -> ModeTies {
    match s {
        "FIRST" => ModeTies::First,
        "MIN" => ModeTies::Min,
        "MAX" => ModeTies::Max,
        _ => panic!("unknown ties {s}"),
    }
}

/// Mode of the case's source, loaded as `T`, returned as f64.
fn mode_as<T: Sample>(g: &Grid, footprint: Footprint, ties: ModeTies) -> Vec<f64> {
    let holes = g.header("nodata_src") == "True";
    let src = read_grid(g.header("src"));
    let data: Vec<T> = src.vals.iter().map(|&v| T::from_f64(v as f64)).collect();
    let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions {
        alg: ResampleAlg::Mode,
        footprint,
        mode_ties: ties,
        nodata: if holes { Some(T::from_f64(0.0)) } else { None },
        ..Default::default()
    };
    warp(&Affine::of(g), &buf, g.ncol, g.nrow, &opts).iter().map(|v| v.to_f64()).collect()
}

fn mode_case(g: &Grid, ty: &str, footprint: Footprint, ties: ModeTies) -> Vec<f64> {
    match ty {
        "Byte" => mode_as::<u8>(g, footprint, ties),
        "Int16" => mode_as::<i16>(g, footprint, ties),
        "Int32" => mode_as::<i32>(g, footprint, ties),
        "Float32" => mode_as::<f32>(g, footprint, ties),
        "Int64" => mode_as::<i64>(g, footprint, ties),
        "UInt64" => mode_as::<u64>(g, footprint, ties),
        #[cfg(feature = "f16")]
        "Float16" => mode_as::<half::f16>(g, footprint, ties),
        _ => panic!("unknown type {ty}"),
    }
}

fn assert_same(name: &str, expect: &[f32], got: &[f64]) {
    let bad: Vec<usize> = (0..expect.len()).filter(|&i| expect[i] as f64 != got[i]).collect();
    assert!(bad.is_empty(), "{name}: {} pixels differ, first {} (gdal {} rwarp {})",
        bad.len(), bad[0], expect[bad[0]], got[bad[0]]);
}

fn mode_fixtures(prefix: &str) -> Vec<String> {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "kernels"].iter().collect();
    let mut names: Vec<String> = std::fs::read_dir(dir).unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".txt"))
        .map(|n| n.trim_end_matches(".txt").to_string())
        .collect();
    names.sort();
    names
}

/// Every categorical case, all seven types (Float16 with feature `f16`) and
/// all three tie rules, 4:1 and 2.78:1 (fractional weights), nodata and a
/// rotated destination: identical to gdalwarp -r mode.
#[test]
fn mode_matches_gdal() {
    let names: Vec<String> = mode_fixtures("mode_").into_iter()
        .filter(|n| !n.contains("distinct"))
        .filter(|n| cfg!(feature = "f16") || !n.contains("float16"))
        .collect();
    assert_eq!(names.len(), if cfg!(feature = "f16") { 44 } else { 38 });
    for name in names {
        let g = read_grid(&name);
        let got = mode_case(&g, g.header("src_type"), Footprint::GdalBox, ties_of(g.header("ties")));
        assert_same(&name, &g.vals, &got);
    }
}

/// Ties at a value's first occurrence: every 2x2 block holds four distinct
/// values. GDAL applies MIN/MAX there for Byte but not for Int32/Float32 (its
/// list-of-values path never compares a new entry against the mode).
/// rwarp gives GDAL's Byte answer for every type.
#[test]
fn mode_first_occurrence_ties_follow_gdal_byte_for_every_type() {
    for ties in ["first", "min", "max"] {
        let byte = read_grid(&format!("mode_distinct_byte_{ties}"));
        for ty in ["Byte", "Int32", "Float32"] {
            let got = mode_case(&byte, ty, Footprint::GdalBox, ties_of(byte.header("ties")));
            assert_same(&format!("distinct {ties} as {ty}"), &byte.vals, &got);
        }
        // The GDAL inconsistency this departs from.
        let int32 = read_grid(&format!("mode_distinct_int32_{ties}"));
        let differ = byte.vals.iter().zip(&int32.vals).filter(|(a, b)| a != b).count();
        if ties == "first" {
            assert_eq!(differ, 0);
        } else {
            assert!(differ > 0, "{ties}: expected GDAL's Int32 path to differ from Byte");
        }
    }
}

/// Where the destination pixel maps to an axis-aligned rectangle the exact
/// footprint has the box's weights (as f32 fractions), so the mode agrees.
#[test]
fn mode_exact_equals_gdal_box_when_axis_aligned() {
    for name in ["mode_byte_first_4x", "mode_byte_min_4x", "mode_int32_max_4x", "mode_byte_min_4x_nodata"] {
        let g = read_grid(name);
        let got = mode_case(&g, g.header("src_type"), Footprint::Exact { densify: 1 }, ties_of(g.header("ties")));
        assert_same(name, &g.vals, &got);
    }
}

#[test]
fn mode_first_is_first_to_reach_the_count() {
    // [[5, 3], [3, 5]] -> one pixel: 5 is seen first, 3 reaches 2 first.
    let data: Vec<u8> = vec![5, 3, 3, 5];
    let buf = SourceBuffer { data: &data[..], ncol: 2, nrow: 2, nbands: 1, col_off: 0, row_off: 0 };
    struct Half;
    impl Transformer for Half {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            for v in x.iter_mut().chain(y.iter_mut()) { *v *= 2.0; }
            vec![true; x.len()]
        }
    }
    let run = |ties| {
        let opts = WarpOptions { alg: ResampleAlg::Mode, mode_ties: ties, ..Default::default() };
        warp(&Half, &buf, 1, 1, &opts)[0]
    };
    assert_eq!((run(ModeTies::First), run(ModeTies::Min), run(ModeTies::Max)), (3, 3, 5));
}

// ---- rounding, clamping, nodata avoidance, source bounds ------------------

fn run_typed<T: Sample>(g: &Grid) -> Vec<f64> {
    let src = read_grid(g.header("src"));
    let data: Vec<T> = src.vals.iter().map(|&v| T::from_f64(v as f64)).collect();
    let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: 0, row_off: 0 };
    let nodata = match g.header("nodata") {
        "-" => None,
        v => Some(T::from_f64(v.parse().unwrap())),
    };
    let scaling = match g.header("wo") {
        "-" => KernelScaling::WidenWhenDownsampling,
        "XSCALE=1 YSCALE=1" => KernelScaling::Point,
        other => panic!("unhandled warp options {other}"),
    };
    let opts = WarpOptions { alg: alg_of(g.header("alg")), scaling, nodata, ..Default::default() };
    warp(&Affine::of(g), &buf, g.ncol, g.nrow, &opts).iter().map(|v| v.to_f64()).collect()
}

/// Integer outputs clamp then round half up (-22.5 -> -22, cubic overshoot
/// clamped to the type), a resampled value equal to nodata steps off it
/// (-12 -> -13; float 0.0 -> the next float up), and sample points outside
/// [0, n) get no value. All identical to gdalwarp.
#[test]
fn rounding_clamping_avoidance_and_bounds_match_gdal() {
    let names: Vec<String> = ["round_", "clamp_", "avoid_", "bounds_"]
        .iter()
        .flat_map(|p| mode_fixtures(p))
        .collect();
    assert_eq!(names.len(), 12);
    for name in names {
        let g = read_grid(&name);
        let got = match g.header("src_type") {
            "Byte" => run_typed::<u8>(&g),
            "Int16" => run_typed::<i16>(&g),
            "Int32" => run_typed::<i32>(&g),
            "Float32" => run_typed::<f32>(&g),
            other => panic!("unknown type {other}"),
        };
        assert_same(&name, &g.vals, &got);
    }
}

// ---- nodata: point kernels, multiple bands, average thresholds ------------

/// `WarpOptions` from a fixture's `wo` header, the way GDAL reads the warp
/// options.
fn options_from_wo(g: &Grid, nbands: usize) -> WarpOptions<f32> {
    let mut o = WarpOptions {
        alg: alg_of(g.header("alg")),
        nodata: match g.header("nodata") { "-" => None, v => Some(v.parse().unwrap()) },
        ..Default::default()
    };
    let wo = g.header("wo");
    if wo == "-" { return o; }
    for kv in wo.split(' ') {
        let (k, v) = kv.split_once('=').unwrap();
        match k {
            "XSCALE" | "YSCALE" => o.scaling = KernelScaling::Point,
            "UNIFIED_SRC_NODATA" => o.band_nodata = match v {
                "YES" => BandNodata::Unified,
                "NO" => BandNodata::PerBand,
                _ => BandNodata::Partial,
            },
            "NODATA_VALUES_PCT_THRESHOLD" => o.average.nodata_pct_threshold = v.parse().unwrap(),
            "EXCLUDED_VALUES_PCT_THRESHOLD" => o.average.excluded_pct_threshold = v.parse().unwrap(),
            "EXCLUDED_VALUES" => {
                // GDAL tokenizes on "(,)" and groups by band count.
                let vals: Vec<f64> = v.split(['(', ',', ')']).filter(|t| !t.is_empty())
                    .map(|t| t.parse().unwrap()).collect();
                o.average.excluded_values = vals.chunks(nbands).map(<[f64]>::to_vec).collect();
            }
            _ => panic!("unhandled warp option {k}"),
        }
    }
    o
}

fn run_nodata_case(name: &str) {
    let g = read_grid(name);
    let src = read_grid(g.header("src"));
    let nb = src.nbands();
    let data = src.interleaved();
    let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: nb, col_off: 0, row_off: 0 };
    let opts = options_from_wo(&g, nb);
    let got = warp(&Affine::of(&g), &buf, g.ncol, g.nrow, &opts);
    let expect = g.interleaved();
    let mut worst = (0.0f64, 0usize);
    for (i, (&e, &v)) in expect.iter().zip(&got).enumerate() {
        let d = (e as f64 - v as f64).abs() / (e.abs() as f64).max(1.0);
        if d > worst.0 { worst = (d, i); }
    }
    let (d, i) = worst;
    let px = i / nb;
    assert!(d <= TOL, "{name}: pixel ({}, {}) band {} gdal {} rwarp {} (relative {d:e})",
        px % g.ncol, px / g.ncol, i % nb, expect[i], got[i]);
}

/// Single-band point kernels next to nodata, average thresholds and
/// excluded values, and three bands under each of GDAL's
/// UNIFIED_SRC_NODATA modes (unset, YES, NO) for near, point bilinear and
/// cubic, widened bilinear and lanczos, average and mode: all match
/// gdalwarp.
#[test]
fn nodata_cases_match_gdal() {
    let names: Vec<String> = ["nd1_", "thr_", "mb_"].iter().flat_map(|p| mode_fixtures(p)).collect();
    assert_eq!(names.len(), 30);
    let mut failures = Vec::new();
    for name in &names {
        if let Err(e) = std::panic::catch_unwind(|| run_nodata_case(name)) {
            failures.push(e.downcast_ref::<String>().cloned().unwrap_or_default());
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), names.len(), failures.join("\n"));
}

#[test]
fn any_band_rule_is_the_0_1_behaviour() {
    // A pixel with one band at nodata: AnyBand drops it entirely, Partial
    // keeps its other bands.
    let src = read_grid("src_mb");
    let data = src.interleaved();
    let buf = SourceBuffer { data: &data[..], ncol: N, nrow: N, nbands: 3, col_off: 0, row_off: 0 };
    let run = |band_nodata| {
        let o = WarpOptions { alg: ResampleAlg::NearestNeighbour, nodata: Some(NODATA), band_nodata, ..Default::default() };
        warp(&Affine::new(N, N), &buf, N, N, &o)
    };
    let (partial, any) = (run(BandNodata::Partial), run(BandNodata::AnyBand));
    assert_eq!(partial, data, "nearest at 1:1 copies every band");
    for p in 0..N * N {
        let pix = &data[p * 3..p * 3 + 3];
        if pix.contains(&NODATA) {
            assert_eq!(&any[p * 3..p * 3 + 3], &[NODATA; 3]);
        }
    }
}

/// The fixtures separate the rules: each multi-band case fails under a rule
/// other than the one it was generated with.
#[test]
fn nodata_fixtures_distinguish_the_rules() {
    for name in ["mb_unset_bilinear_point", "mb_yes_average_4x", "mb_no_cubic_point"] {
        let g = read_grid(name);
        let src = read_grid(g.header("src"));
        let nb = src.nbands();
        let data = src.interleaved();
        let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: nb, col_off: 0, row_off: 0 };
        let right = options_from_wo(&g, nb);
        let wrong = WarpOptions {
            band_nodata: if right.band_nodata == BandNodata::AnyBand { BandNodata::Partial } else { BandNodata::AnyBand },
            ..right.clone()
        };
        let expect = g.interleaved();
        let got = warp(&Affine::of(&g), &buf, g.ncol, g.nrow, &wrong);
        let differ = expect.iter().zip(&got).filter(|(e, v)| (**e as f64 - **v as f64).abs() > 1e-3).count();
        assert!(differ > 0, "{name}: the AnyBand rule should not reproduce GDAL");
    }
    let _ = AverageThresholds::default();
}

/// GDAL's point bilinear renormalises over valid neighbours: the
/// single-band fixture has valid outputs whose 2x2 neighbourhood includes
/// nodata (rwarp 0.1 gave those nodata).
#[test]
fn point_bilinear_fixture_has_renormalised_pixels() {
    let g = read_grid("nd1_bilinear_point");
    let src = read_grid("src_nodata");
    let t = Affine::of(&g);
    let mut renormalised = 0;
    for r in 0..g.nrow {
        let (mut x, mut y): (Vec<f64>, Vec<f64>) = ((0..g.ncol).map(|c| c as f64 + 0.5).collect(), vec![r as f64 + 0.5; g.ncol]);
        t.transform(true, &mut x, &mut y);
        for c in 0..g.ncol {
            let (ix, iy) = ((x[c] - 0.5).floor() as i64, (y[c] - 0.5).floor() as i64);
            if ix < 0 || iy < 0 || ix + 1 >= N as i64 || iy + 1 >= N as i64 { continue; }
            let at = |dx: i64, dy: i64| src.vals[((iy + dy) * N as i64 + ix + dx) as usize];
            let any_nd = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)].contains(&NODATA);
            if any_nd && g.vals[r * g.ncol + c] != NODATA { renormalised += 1; }
        }
    }
    assert!(renormalised > 0);
}

// ---- more methods and types -------------------------------------------------

/// Warp a fixture case with its source loaded as `T` (any band count),
/// returning the output interleaved, as f64.
fn run_bands<T: Sample>(g: &Grid) -> Vec<f64> {
    let src = read_grid(g.header("src"));
    let nb = src.nbands();
    // Exact (not via f32): 64-bit sources hold values beyond f32.
    let data: Vec<T> = exact_values(g.header("src"), &src).iter().map(|&v| T::from_f64(v)).collect();
    let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: nb, col_off: 0, row_off: 0 };
    let o = options_from_wo(g, nb);
    let opts = WarpOptions {
        alg: o.alg, scaling: o.scaling, band_nodata: o.band_nodata,
        nodata: o.nodata.map(|v| T::from_f64(v as f64)),
        ..Default::default()
    };
    warp(&Affine::of(g), &buf, g.ncol, g.nrow, &opts).iter().map(|v| v.to_f64()).collect()
}

/// Cubic spline (point, widened, nodata, Int16, rotated); min, max, med,
/// q1, q3 and rms (4:1, 2.78:1, nodata, rotated, three bands, Int16); and
/// near, bilinear, average, mode and max on Int64 and UInt64 values near
/// 2^40, and near, bilinear, cubic, average and mode on Float16 (feature
/// `f16`): all match gdalwarp. Integers exactly, Float32 within 1e-6
/// relative, Float16 within one half-precision step.
#[test]
fn methods_and_types_match_gdal() {
    let names: Vec<String> = ["cs_", "stat_", "type_"].iter().flat_map(|p| mode_fixtures(p))
        .filter(|n| cfg!(feature = "f16") || !n.contains("float16"))
        .collect();
    assert_eq!(names.len(), if cfg!(feature = "f16") { 50 } else { 45 });
    let mut failures = Vec::new();
    for name in &names {
        let g = read_grid(name);
        let ty = g.header("src_type");
        let (got, tol): (Vec<f64>, f64) = match ty {
            "Float32" => (run_bands::<f32>(&g), 1e-6),
            "Int16" => (run_bands::<i16>(&g), 0.0),
            "Int64" => (run_bands::<i64>(&g), 0.0),
            "UInt64" => (run_bands::<u64>(&g), 0.0),
            #[cfg(feature = "f16")]
            "Float16" => (run_bands::<half::f16>(&g), 1.0 / 1024.0),
            other => panic!("unknown type {other}"),
        };
        let expect: Vec<f64> = g.interleaved().iter().map(|&v| v as f64).collect();
        // Int64/UInt64 values exceed f32; compare against the f64 text.
        let expect = if ty.ends_with("64") { exact_values(name, &g) } else { expect };
        let bad = expect.iter().zip(&got).enumerate()
            .filter(|(_, (e, v))| (**e - **v).abs() > tol * e.abs().max(1.0))
            .map(|(i, (e, v))| format!("{name}: value {i} gdal {e} rwarp {v}"))
            .next();
        if let Some(b) = bad { failures.push(b); }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), names.len(), failures.join("\n"));
}

/// A fixture's values parsed as f64 (not via f32), interleaved.
fn exact_values(name: &str, g: &Grid) -> Vec<f64> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "kernels", &format!("{name}.txt")].iter().collect();
    let text = std::fs::read_to_string(path).unwrap();
    let vals: Vec<f64> = text.lines().filter(|l| !l.starts_with('#')).skip(1)
        .flat_map(|l| l.split_whitespace()).map(|v| v.parse().unwrap()).collect();
    let (n, nb) = (g.ncol * g.nrow, g.nbands());
    (0..n * nb).map(|i| vals[(i % nb) * n + i / nb]).collect()
}
