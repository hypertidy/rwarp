//! Kernel parity with gdalwarp on synthetic fixtures.
//!
//! The fixtures in `tests/fixtures/kernels/` are gdalwarp output, written by
//! `gen.py` there (which records the GDAL version). Source and destination
//! share an extent with no CRS, so the transform is affine and only the
//! kernels are under test.

use std::path::PathBuf;

use rwarp::warp::{estimate_scale, warp, KernelScaling, ResampleAlg, SourceBuffer, WarpOptions};
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
    fn header(&self, key: &str) -> &str {
        self.header
            .iter()
            .find_map(|l| l.strip_prefix(key).map(str::trim))
            .unwrap_or_else(|| panic!("no '{key}' in fixture header"))
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
    assert_eq!(vals.len(), ncol * nrow, "{name}: value count");
    Grid { header, ncol, nrow, vals }
}

/// Destination pixel -> source pixel through the geotransforms gdalwarp
/// uses: source `[0, 1, 0, 64, 0, -1]`, destination the same extent at
/// `dw` x `dh`.
struct Affine {
    dx: f64,
    dy: f64,
}

impl Affine {
    fn new(dw: usize, dh: usize) -> Self {
        Affine { dx: N as f64 / dw as f64, dy: N as f64 / dh as f64 }
    }
}

impl Transformer for Affine {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        assert!(dst_to_src);
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            *x *= self.dx;
            let geo_y = N as f64 - *y * self.dy;
            *y = N as f64 - geo_y;
        }
        vec![true; x.len()]
    }
}

fn alg_of(name: &str) -> ResampleAlg {
    match name {
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "lanczos" => ResampleAlg::Lanczos,
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
    let got = warp(&Affine::new(expect.ncol, expect.nrow), &buf, expect.ncol, expect.nrow, &opts);
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
);

#[test]
fn estimated_scale_matches_gdal() {
    for name in ["bilinear_4x", "bilinear_2p78x", "bilinear_aniso", "bilinear_1p49x"] {
        let g = read_grid(name);
        let mut s = g.header("gdal_scale").split_whitespace().map(|v| v.parse::<f64>().unwrap());
        let (gx, gy) = (s.next().unwrap(), s.next().unwrap());
        let (sx, sy) = estimate_scale(&Affine::new(g.ncol, g.nrow), g.ncol, g.nrow, 0, 0);
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
