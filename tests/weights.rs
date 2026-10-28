//! Exact-footprint weights (`rwarp::weights`): equivalence with `warp()`,
//! conservation, tiling, densify.
//!
//! Conservation needs no external reference (design/conservative.md, 3.3):
//! neighbouring destination outlines share their transformed points, so
//! the destination cells tile source pixel space and each source cell's
//! `frac` sums to the part of it inside the destination grid. That part is
//! measured independently, as the coverage of the destination grid's outer
//! boundary (built from the same points) over the source.
//!
//! `cargo test --test weights -- --nocapture` prints the worst errors.

use std::collections::HashMap;
use std::path::PathBuf;

use controlledburn::CoverageWindow;
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{warp_to, BandNodata, DstNodata, Footprint, Nodata, ResampleAlg, Sample, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, exact_weights_with, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::Transformer;

// ---- fixtures (as in tests/kernels.rs) --------------------------------------

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
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
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
    let vals = lines.flat_map(|l| l.split_whitespace()).map(|v| v.parse::<f64>().unwrap() as f32).collect();
    Grid { header, ncol, nrow, vals }
}

/// Destination pixel -> source pixel through a destination geotransform
/// and the source's `[0, 1, 0, 64, 0, -1]`, as in tests/kernels.rs.
struct Affine {
    gt: [f64; 6],
}

impl Affine {
    fn of(g: &Grid) -> Self {
        let v: Vec<f64> = g.header("dst_gt").split_whitespace().map(|v| v.parse().unwrap()).collect();
        Affine { gt: v.try_into().unwrap() }
    }
}

impl Transformer for Affine {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let g = &self.gt;
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            let (gx, gy) = (g[0] + *x * g[1] + *y * g[2], g[3] + *x * g[4] + *y * g[5]);
            *x = gx;
            *y = N as f64 - gy;
        }
        vec![true; x.len()]
    }
}

/// A rotation and scale straight into source pixels.
struct Rotated {
    angle: f64,
    scale: f64,
    origin: (f64, f64),
}

impl Transformer for Rotated {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (s, c) = self.angle.sin_cos();
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            let (u, v) = (*x * self.scale, *y * self.scale);
            *x = self.origin.0 + c * u - s * v;
            *y = self.origin.1 + s * u + c * v;
        }
        vec![true; x.len()]
    }
}

// ---- warp() equivalence --------------------------------------------------------

/// `warp_to` with the exact footprint and average, and the same through
/// warp-compatible weights and `apply(FracArea)`.
fn both<S: Sample, D: Sample>(
    t: &impl Transformer,
    buf: &SourceBuffer<'_, S>,
    dw: usize,
    dh: usize,
    densify: usize,
    opts: WarpOptions,
) -> (Vec<D>, Vec<D>) {
    let opts = opts.with_footprint(Footprint::Exact { densify });
    let warped = warp_to::<S, D>(t, buf, dw, dh, &opts);
    let wopts = WeightOptions::warp_compatible(densify).with_dst_offset(opts.dst_col_off, opts.dst_row_off);
    let w = exact_weights(t, &GridWindow::from(buf), dw, dh, &wopts);
    let applied = w.apply::<S, D>(buf, Normalization::FracArea, &ApplyOptions::from(&opts));
    (warped, applied)
}

fn assert_bits<D: Sample>(name: &str, a: &[D], b: &[D]) {
    assert_eq!(a.len(), b.len(), "{name}");
    let diff = a.iter().zip(b).filter(|(x, y)| x.to_f64().to_bits() != y.to_f64().to_bits()).count();
    assert_eq!(diff, 0, "{name}: {diff} of {} values differ", a.len());
}

#[test]
fn apply_frac_area_reproduces_warp_exact_on_kernel_fixtures() {
    let names = [
        "average_4x", "average_2p78x", "average_1p49x", "average_aniso", "average_upsample",
        "average_4x_nodata", "average_overhang", "average_rot30",
    ];
    let mut n = 0;
    for name in names {
        let g = read_grid(name);
        let src = read_grid(if g.header("nodata_src") == "True" { "src_nodata" } else { "src" });
        let buf = SourceBuffer { data: &src.vals[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: 0, row_off: 0 };
        for densify in [1, 3, 8] {
            let opts = WarpOptions::new(ResampleAlg::Average).with_src_nodata(Nodata::Value(NODATA as f64));
            let (a, b) = both::<f32, f32>(&Affine::of(&g), &buf, g.ncol, g.nrow, densify, opts);
            assert_bits(&format!("{name} densify {densify}"), &a, &b);
            n += a.len();
        }
    }
    eprintln!("kernel fixtures: {n} values identical to warp()");
}

/// A 45-degree diamond over a column-index source (kernels.rs): one pixel.
#[test]
fn apply_frac_area_reproduces_warp_on_rotated_diamond() {
    let vals: Vec<f32> = (0..100).map(|i| (i % 10) as f32).collect();
    let buf = SourceBuffer { data: &vals[..], ncol: 10, nrow: 10, nbands: 1, col_off: 0, row_off: 0 };
    let s = 2f64.sqrt() * 2.0;
    let t = Affine { gt: [5.0, s, -s, N as f64 - 1.0, -s, -s] };
    let (a, b) = both::<f32, f32>(&t, &buf, 1, 1, 1, WarpOptions::new(ResampleAlg::Average));
    assert_bits("diamond", &a, &b);
    assert!((b[0] - 4.5).abs() < 1e-5);
}

const LAEA_TAS: &str = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";

/// A smooth field on a 0.5-degree global lon/lat grid, with holes.
fn lonlat_field(ncol: usize, nrow: usize, nb: usize, holes: bool) -> Vec<f32> {
    let mut v = Vec::with_capacity(ncol * nrow * nb);
    for r in 0..nrow {
        for c in 0..ncol {
            for b in 0..nb {
                let x = (c as f64 * 0.37 + b as f64).sin() * 40.0 + (r as f64 * 0.23).cos() * 25.0 + 100.0 * (b + 1) as f64;
                let hole = holes && ((r * 7 + c * 3 + b) % 11 == 0);
                let nan = holes && b == 0 && (r * 5 + c) % 17 == 0;
                v.push(if nan { f32::NAN } else if hole { NODATA } else { x as f32 });
            }
        }
    }
    v
}

const LONLAT_HALF: [f64; 6] = [-180.0, 0.5, 0.0, 90.0, 0.0, -0.5];

/// Lon/lat to LAEA, through PROJ: a source window with offsets, a
/// destination window with offsets, two bands with nodata and NaN, every
/// band-nodata rule, a mask band, and type conversion.
#[test]
fn apply_frac_area_reproduces_warp_reprojected() {
    let dst_gt = [-1_000_000.0, 20_000.0, 0.0, 1_000_000.0, 0.0, -20_000.0];
    let t = GenImgProjTransformer::new("EPSG:4326", LONLAT_HALF, LAEA_TAS, dst_gt).unwrap();
    // The source window over Tasmania and around (lon 125..170, lat -25..-60).
    let (col_off, row_off, ncol, nrow) = (610usize, 230usize, 90usize, 70usize);
    let nb = 2;
    let data = lonlat_field(ncol, nrow, nb, true);
    let buf = SourceBuffer { data: &data[..], ncol, nrow, nbands: nb, col_off, row_off };
    let mut n = 0;
    for densify in [1, 4] {
        for rule in [BandNodata::Partial, BandNodata::Unified, BandNodata::PerBand, BandNodata::AnyBand] {
            let base = WarpOptions::new(ResampleAlg::Average)
                .with_src_nodata(Nodata::Value(NODATA as f64))
                .with_band_nodata(rule)
                .with_dst_offset(7, 11);
            let (a, b) = both::<f32, f32>(&t, &buf, 80, 70, densify, base.clone());
            assert_bits(&format!("laea {rule:?} densify {densify}"), &a, &b);
            n += a.len();
            let (a, b) = both::<f32, f64>(&t, &buf, 80, 70, densify, base.clone().with_mask_band(Some(1)));
            assert_bits(&format!("laea mask {rule:?} densify {densify}"), &a, &b);
            n += a.len();
        }
        // Integer source and destination: stored in the working type,
        // rounded, moved off the destination nodata.
        let ints: Vec<i16> = data.iter().map(|&v| if v.is_nan() { -9999 } else { v as i16 }).collect();
        let ibuf = SourceBuffer { data: &ints[..], ncol, nrow, nbands: nb, col_off, row_off };
        let opts = WarpOptions::new(ResampleAlg::Average)
            .with_src_nodata(Nodata::Value(-9999.0))
            .with_dst_nodata(DstNodata::Value(100.0));
        let (a, b) = both::<i16, u8>(&t, &ibuf, 100, 100, densify, opts);
        assert_bits(&format!("laea i16 -> u8 densify {densify}"), &a, &b);
        n += a.len();
    }
    eprintln!("reprojected: {n} values identical to warp()");
}

// ---- conservation --------------------------------------------------------------

/// Coverage of the source window's cells by the whole destination grid:
/// its outer boundary, through the same lattice points the weights use.
fn union_coverage(t: &impl Transformer, dw: usize, dh: usize, densify: usize, src: &GridWindow) -> Vec<f64> {
    let d = densify;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    let mut pt = |px: f64, py: f64| {
        x.push(px);
        y.push(py);
    };
    // The same expressions as the lattice (offset 0): row lines at x = k /
    // d, column lines at y = row + k / d.
    for k in 0..dw * d { pt(k as f64 / d as f64, 0.0); }
    for r in 0..dh {
        pt(dw as f64, r as f64);
        for k in 1..d { pt(dw as f64, r as f64 + k as f64 / d as f64); }
    }
    for k in (1..=dw * d).rev() { pt(k as f64 / d as f64, dh as f64); }
    for r in (0..dh).rev() {
        pt(0.0, (r + 1) as f64);
        for k in (1..d).rev() { pt(0.0, r as f64 + k as f64 / d as f64); }
    }
    let ok = t.transform(true, &mut x, &mut y);
    assert!(ok.iter().all(|&o| o), "boundary failed to transform");
    for v in x.iter_mut() { *v -= src.col_off as f64; }
    for v in y.iter_mut() { *v -= src.row_off as f64; }
    let mut win = CoverageWindow::default();
    win.reset(0, 0, src.ncol, src.nrow);
    win.add_ring(&x, &y, 1.0);
    win.resolve().to_vec()
}

/// Sum of `frac` per source cell of the window.
fn column_sums(w: &Weights, src: &GridWindow) -> Vec<f64> {
    let mut s = vec![0.0; src.ncol * src.nrow];
    for k in 0..w.len() {
        s[(w.src_row[k] as usize - src.row_off) * src.ncol + (w.src_col[k] as usize - src.col_off)] += w.frac[k];
    }
    s
}

// Fields are read through Debug, in the report.
#[allow(dead_code)]
#[derive(Debug)]
struct Conservation {
    entries: usize,
    full: usize,
    partial: usize,
    /// Worst |column sum - 1| over source cells the destination covers
    /// completely.
    full_err: f64,
    /// Worst |column sum - covered fraction| over cells cut by the
    /// destination grid's edge.
    partial_err: f64,
    /// Relative error of the global integral with DestArea.
    integral_err: f64,
}

fn check_conservation(name: &str, t: &impl Transformer, src: &GridWindow, dw: usize, dh: usize, densify: usize) -> (Weights, Conservation) {
    let w = exact_weights(t, src, dw, dh, &WeightOptions::new(densify));
    assert!(w.dst_area.iter().all(|a| a.is_finite()), "{name}: a destination cell failed");
    assert!(w.frac.iter().all(|&f| f > 0.0 && f <= 1.0 + 1e-12), "{name}: frac out of range");
    let u = union_coverage(t, dw, dh, densify, src);
    let s = column_sums(&w, src);
    let (mut full, mut partial, mut full_err, mut partial_err) = (0, 0, 0f64, 0f64);
    for (&uj, &sj) in u.iter().zip(&s) {
        if (uj - 1.0).abs() < 1e-9 {
            full += 1;
            full_err = full_err.max((sj - 1.0).abs());
        } else if uj > 1e-12 {
            partial += 1;
            partial_err = partial_err.max((sj - uj).abs());
        } else {
            assert!(sj.abs() < 1e-12, "{name}: uncovered source cell has weight {sj}");
        }
    }
    // Global integral with DestArea: sum_i dst_i * dst_area_i is
    // sum_j v_j * (covered fraction of j).
    let data: Vec<f64> = (0..src.ncol * src.nrow)
        .map(|p| 50.0 + 30.0 * ((p % src.ncol) as f64 * 0.11).sin() * ((p / src.ncol) as f64 * 0.07).cos())
        .collect();
    let buf = SourceBuffer { data: &data[..], ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: src.col_off, row_off: src.row_off };
    let dst: Vec<f64> = w.apply(&buf, Normalization::DestArea, &ApplyOptions::new().with_dst_nodata(DstNodata::None));
    let got: f64 = dst.iter().zip(&w.dst_area).map(|(v, a)| v * a).sum();
    let want: f64 = data.iter().zip(&u).map(|(v, u)| v * u).sum();
    let integral_err = ((got - want) / want).abs();
    let c = Conservation { entries: w.len(), full, partial, full_err, partial_err, integral_err };
    eprintln!("{name} densify {densify}: {c:?}");
    assert!(full > 100, "{name}: too few fully covered source cells ({full})");
    assert!(full_err < 1e-12, "{name}: {c:?}");
    assert!(partial_err < 1e-12, "{name}: {c:?}");
    assert!(integral_err < 1e-12, "{name}: {c:?}");
    (w, c)
}

/// Rotated by 0.3 rad at 2.3 source pixels per destination pixel, hanging
/// over the source's left edge.
#[test]
fn conservation_regular_to_rotated() {
    let t = Rotated { angle: 0.3, scale: 2.3, origin: (20.0, 5.0) };
    let src = GridWindow::new(200, 200);
    for densify in [1, 4] {
        check_conservation("rotated", &t, &src, 60, 50, densify);
    }
}

/// Lon/lat at 0.5 degrees to EPSG:3031 at 25 km, east of the pole (x from
/// 500 to 3000 km), clear of the antimeridian.
#[test]
fn conservation_lonlat_to_polar_stereographic() {
    let dst_gt = [500_000.0, 25_000.0, 0.0, 1_500_000.0, 0.0, -25_000.0];
    let t = GenImgProjTransformer::new("EPSG:4326", LONLAT_HALF, "EPSG:3031", dst_gt).unwrap();
    let src = GridWindow::new(720, 360);
    for densify in [1, 4] {
        check_conservation("3031", &t, &src, 100, 120, densify);
    }
}

/// Lon/lat at 0.5 degrees to LAEA centred on Tasmania, 2000 km at 20 km.
#[test]
fn conservation_lonlat_to_laea() {
    let dst_gt = [-1_000_000.0, 20_000.0, 0.0, 1_000_000.0, 0.0, -20_000.0];
    let t = GenImgProjTransformer::new("EPSG:4326", LONLAT_HALF, LAEA_TAS, dst_gt).unwrap();
    // A source window is enough; the grid's outline stays inside it.
    let src = GridWindow::new(160, 120).with_offset(580, 200);
    let (w1, _) = check_conservation("laea", &t, &src, 100, 100, 1);
    let (w4, _) = check_conservation("laea", &t, &src, 100, 100, 4);
    // Densify: both conserve; the cells differ, since destination edges
    // are curves in lon/lat and densify 1 replaces them with chords.
    let (df, da) = densify_difference(&w1, &w4);
    eprintln!("laea densify 1 vs 4: max |frac| difference {df:.3e}, max relative dst_area difference {da:.3e}");
    assert!(df > 1e-9 && df < 1e-2, "{df}");
    assert!(da > 1e-9 && da < 1e-3, "{da}");
}

/// Under an affine transform destination edges are straight in source
/// pixel space, so densifying them changes nothing beyond rounding.
#[test]
fn densify_is_invariant_under_affine() {
    let t = Rotated { angle: 0.3, scale: 2.3, origin: (20.0, 5.0) };
    let src = GridWindow::new(200, 200);
    let w1 = exact_weights(&t, &src, 60, 50, &WeightOptions::new(1));
    let w4 = exact_weights(&t, &src, 60, 50, &WeightOptions::new(4));
    let (df, da) = densify_difference(&w1, &w4);
    eprintln!("affine densify 1 vs 4: max |frac| difference {df:.3e}, max relative dst_area difference {da:.3e}");
    assert!(df < 1e-12 && da < 1e-12);
}

/// Largest difference in `frac` over all (destination, source) pairs
/// (missing counts as 0), and in `dst_area` relative to itself.
fn densify_difference(a: &Weights, b: &Weights) -> (f64, f64) {
    let key = |w: &Weights, k: usize| (w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]);
    let mut m: HashMap<_, (f64, f64)> = HashMap::new();
    for k in 0..a.len() { m.entry(key(a, k)).or_default().0 = a.frac[k]; }
    for k in 0..b.len() { m.entry(key(b, k)).or_default().1 = b.frac[k]; }
    let df = m.values().map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let da = a.dst_area.iter().zip(&b.dst_area).map(|(x, y)| ((x - y) / x).abs()).fold(0.0, f64::max);
    (df, da)
}

// ---- tiling --------------------------------------------------------------------

fn entries(w: &Weights) -> Vec<(u32, u32, u32, u32, u64)> {
    (0..w.len()).map(|k| (w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k], w.frac[k].to_bits())).collect()
}

/// Weights of destination tiles are the whole grid's, exactly: row bands
/// concatenate, and 2 x 2 tiles give the same entries and areas.
#[test]
fn tiles_concatenate_to_whole_grid() {
    let dst_gt = [-1_000_000.0, 20_000.0, 0.0, 1_000_000.0, 0.0, -20_000.0];
    let t = GenImgProjTransformer::new("EPSG:4326", LONLAT_HALF, LAEA_TAS, dst_gt).unwrap();
    let src = GridWindow::new(160, 120).with_offset(580, 200);
    let (dw, dh) = (100, 100);
    // Row bands concatenate for any densify.
    for densify in [1, 3, 4] {
        let whole = exact_weights(&t, &src, dw, dh, &WeightOptions::new(densify));
        let mut cat = Weights::default();
        for (r0, r1) in [(0, 37), (37, 38), (38, 100)] {
            let w = exact_weights(&t, &src, dw, r1 - r0, &WeightOptions::new(densify).with_dst_offset(0, r0));
            cat.dst_row.extend(&w.dst_row);
            cat.dst_col.extend(&w.dst_col);
            cat.src_row.extend(&w.src_row);
            cat.src_col.extend(&w.src_col);
            cat.frac.extend(&w.frac);
            cat.dst_area.extend(&w.dst_area);
        }
        assert_eq!(entries(&cat), entries(&whole), "row bands, densify {densify}");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&cat.dst_area), bits(&whole.dst_area), "row bands, densify {densify}");
    }
    // Column offsets too when densify is a power of two.
    for densify in [1, 4] {
        let whole = exact_weights(&t, &src, dw, dh, &WeightOptions::new(densify));
        let mut all = Vec::new();
        let mut area = vec![f64::NAN; dw * dh];
        for (c0, c1) in [(0, 41), (41, 100)] {
            for (r0, r1) in [(0, 61), (61, 100)] {
                let o = WeightOptions::new(densify).with_dst_offset(c0, r0);
                let w = exact_weights(&t, &src, c1 - c0, r1 - r0, &o);
                all.extend(entries(&w));
                for r in 0..r1 - r0 {
                    for c in 0..c1 - c0 {
                        area[(r0 + r) * dw + c0 + c] = w.dst_area[r * (c1 - c0) + c];
                    }
                }
            }
        }
        all.sort_by_key(|e| (e.0, e.1));
        assert_eq!(all, entries(&whole), "2 x 2 tiles, densify {densify}");
        assert!(area.iter().zip(&whole.dst_area).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

// ---- API -----------------------------------------------------------------------

#[test]
fn normalised_and_esmf_indices() {
    let t = Rotated { angle: 0.3, scale: 2.3, origin: (20.0, 5.0) };
    let src = GridWindow::new(200, 200);
    let w = exact_weights(&t, &src, 60, 50, &WeightOptions::new(1));
    let fa = w.normalised(Normalization::FracArea);
    let da = w.normalised(Normalization::DestArea);
    let mut sfa = vec![0.0; 60 * 50];
    let mut sda = vec![0.0; 60 * 50];
    for k in 0..w.len() {
        let c = w.dst_row[k] as usize * 60 + w.dst_col[k] as usize;
        sfa[c] += fa[k];
        sda[c] += da[k];
    }
    let cov = w.dst_covered();
    for c in 0..60 * 50 {
        if cov[c] > 0.0 {
            assert!((sfa[c] - 1.0).abs() < 1e-12);
            // DestArea sums to the covered fraction of the cell: 1 inside
            // the source, less where the cell hangs over its edge.
            assert!((sda[c] - cov[c] / w.dst_area[c]).abs() < 1e-12 && sda[c] <= 1.0 + 1e-12);
        }
    }
    assert!(sda.iter().any(|&s| s > 0.0 && s < 0.99), "some cells overhang the source");
    let (row, col) = w.esmf_row_col(60, 200);
    assert_eq!(row[0], w.dst_row[0] as u64 * 60 + w.dst_col[0] as u64 + 1);
    assert_eq!(col[0], w.src_row[0] as u64 * 200 + w.src_col[0] as u64 + 1);
}

/// FracArea renormalises over valid source cells; DestArea doesn't.
#[test]
fn apply_skips_nodata() {
    // One destination cell covering a 2 x 2 source block exactly.
    let t = Rotated { angle: 0.0, scale: 2.0, origin: (0.0, 0.0) };
    let w = exact_weights(&t, &GridWindow::new(2, 2), 1, 1, &WeightOptions::new(1));
    assert_eq!(w.len(), 4);
    assert_eq!(w.dst_area, vec![4.0]);
    let data = [1.0f32, 2.0, 3.0, -1.0];
    let buf = SourceBuffer { data: &data[..], ncol: 2, nrow: 2, nbands: 1, col_off: 0, row_off: 0 };
    let o = ApplyOptions::new().with_src_nodata(Nodata::Value(-1.0));
    let fa: Vec<f64> = w.apply(&buf, Normalization::FracArea, &o);
    let da: Vec<f64> = w.apply(&buf, Normalization::DestArea, &o);
    assert_eq!(fa, vec![2.0]);
    assert_eq!(da, vec![1.5]);
    // No valid source: the destination nodata.
    let data = [-1.0f32; 4];
    let buf = SourceBuffer { data: &data[..], ..buf };
    let fa: Vec<f32> = w.apply(&buf, Normalization::FracArea, &o);
    assert_eq!(fa, vec![-1.0]);
}

/// Defaults keep every positive overlap unrounded; the warp-compatible
/// options snap, drop and round.
#[test]
fn options_drop_snap_round() {
    let t = Rotated { angle: 0.3, scale: 2.3, origin: (20.0, 5.0) };
    let src = GridWindow::new(200, 200);
    let raw = exact_weights(&t, &src, 30, 30, &WeightOptions::default());
    let compat = exact_weights(&t, &src, 30, 30, &WeightOptions::warp_compatible(1));
    assert!(raw.len() >= compat.len());
    assert!(raw.frac.iter().any(|&f| f as f32 as f64 != f));
    assert!(compat.frac.iter().all(|&f| f == 1.0 || (f > 1e-6 && f as f32 as f64 == f)));
    // Streaming gives the same cells.
    let mut n = 0;
    let mut cells = 0;
    exact_weights_with(&t, &src, 30, 30, &WeightOptions::default(), |c| {
        n += c.frac.len();
        cells += 1;
    });
    assert_eq!((n, cells), (raw.len(), 900));
}

#[cfg(feature = "serde")]
#[test]
fn weights_serde_round_trip() {
    let t = Rotated { angle: 0.3, scale: 2.3, origin: (20.0, 5.0) };
    let w = exact_weights(&t, &GridWindow::new(200, 200), 10, 10, &WeightOptions::new(2));
    let s = serde_json::to_string(&w).unwrap();
    assert_eq!(serde_json::from_str::<Weights>(&s).unwrap(), w);
}

// ---- timing --------------------------------------------------------------------

/// Cost against `warp()`'s exact footprint, on a 0.25-degree global source.
/// `cargo test --release --test weights -- --ignored --nocapture timing`
#[test]
#[ignore]
fn timing_weights_vs_warp() {
    let src_gt = [-180.0, 0.25, 0.0, 90.0, 0.0, -0.25];
    let (sw, sh) = (1440usize, 720usize);
    let data = lonlat_field(sw, sh, 1, false);
    let buf = SourceBuffer { data: &data[..], ncol: sw, nrow: sh, nbands: 1, col_off: 0, row_off: 0 };
    let cases: [(&str, &str, [f64; 6], usize, usize); 3] = [
        ("laea 400x400 at 10 km", LAEA_TAS, [-2e6, 10_000.0, 0.0, 2e6, 0.0, -10_000.0], 400, 400),
        ("laea 160x160 at 50 km", LAEA_TAS, [-4e6, 50_000.0, 0.0, 4e6, 0.0, -50_000.0], 160, 160),
        ("3031 200x240 at 12.5 km", "EPSG:3031", [500_000.0, 12_500.0, 0.0, 1_500_000.0, 0.0, -12_500.0], 200, 240),
    ];
    for (name, crs, gt, dw, dh) in cases {
        let t = GenImgProjTransformer::new("EPSG:4326", src_gt, crs, gt).unwrap();
        for densify in [1, 4] {
            let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify });
            let t0 = std::time::Instant::now();
            let _: Vec<f32> = warp_to(&t, &buf, dw, dh, &opts);
            let tw = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = std::time::Instant::now();
            let w = exact_weights(&t, &GridWindow::from(&buf), dw, dh, &WeightOptions::new(densify));
            let te = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = std::time::Instant::now();
            let _: Vec<f32> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
            let ta = t0.elapsed().as_secs_f64() * 1e3;
            let bytes = w.len() * 24 + w.dst_area.len() * 8;
            eprintln!(
                "{name} densify {densify}: {} entries, {:.2} per dst cell, {:.1} MB; warp {tw:.1} ms, exact_weights {te:.1} ms, apply {ta:.1} ms",
                w.len(), w.len() as f64 / (dw * dh) as f64, bytes as f64 / 1e6
            );
        }
    }
}
