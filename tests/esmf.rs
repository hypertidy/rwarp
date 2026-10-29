//! ESMF oracle suite for `rwarp::weights` (design/conservative.md, section 4).
//!
//! The fixtures in `tests/fixtures/esmf/` are ESMF first-order conservative
//! maps, normalised by destination area, made by `gen.py` with ESMPy in
//! ghcr.io/hypertidy/gdal-r-esmf. Two per case:
//!
//! - `<case>.cart.txt`: Cartesian, in source pixel space, with destination
//!   corners from GDAL's transformation. That is rwarp's cell definition at
//!   densify 1, so the weights, destination fractions and areas must agree
//!   to rounding.
//! - `<case>.sph.txt`: spherical, with great-circle edges. rwarp measures
//!   planar area in source pixel space until stage 3, so they differ. The
//!   differences are recorded as ceilings per case: numbers to shrink, not
//!   failures. A change that grows one fails; one that shrinks it should
//!   lower the ceiling.
//!
//! `cargo test --test esmf -- --nocapture` prints the measured differences.
//!
//! The destination corners come from GDAL's PROJ, so the suite needs the
//! `proj` backend; proj4rs differs at rounding level and lacks `ob_tran`.
#![cfg(feature = "proj")]

use std::collections::HashMap;
use std::path::PathBuf;

use rwarp::transform::GenImgProjTransformer;
use rwarp::weights::{exact_weights, GridWindow, Normalization, WeightOptions};

struct Fixture {
    header: Vec<String>,
    frac_b: Vec<f64>,
    area_b: Vec<f64>,
    /// (ESMF row, ESMF col) -> S
    s: HashMap<(u64, u64), f64>,
}

impl Fixture {
    fn header(&self, key: &str) -> &str {
        self.header
            .iter()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
            .unwrap_or_else(|| panic!("no '{key}' in fixture header"))
    }

    fn nums<const N: usize>(&self, key: &str) -> [f64; N] {
        let v: Vec<f64> = self.header(key).split_whitespace().map(|x| x.parse().unwrap()).collect();
        v.try_into().unwrap_or_else(|_| panic!("'{key}' needs {N} numbers"))
    }
}

fn read_fixture(name: &str) -> Fixture {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "esmf", name].iter().collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut header = Vec::new();
    let mut lines = text.lines();
    let counts = loop {
        let l = lines.next().expect("missing counts");
        match l.strip_prefix("# ") {
            Some(h) => header.push(h.to_string()),
            None => break l,
        }
    };
    let mut counts = counts.split_whitespace().map(|v| v.parse::<usize>().unwrap());
    let (n_b, n_s) = (counts.next().unwrap(), counts.next().unwrap());
    let (mut frac_b, mut area_b) = (Vec::with_capacity(n_b), Vec::with_capacity(n_b));
    for l in lines.by_ref().take(n_b) {
        let mut f = l.split_whitespace().map(|v| v.parse::<f64>().unwrap());
        frac_b.push(f.next().unwrap());
        area_b.push(f.next().unwrap());
    }
    let mut s = HashMap::with_capacity(n_s);
    for l in lines {
        let mut f = l.split_whitespace();
        let row = f.next().unwrap().parse().unwrap();
        let col = f.next().unwrap().parse().unwrap();
        s.insert((row, col), f.next().unwrap().parse().unwrap());
    }
    assert_eq!((frac_b.len(), s.len()), (n_b, n_s), "{name}: truncated fixture");
    Fixture { header, frac_b, area_b, s }
}

/// rwarp's side of one case, in ESMF's terms.
struct Ours {
    s: HashMap<(u64, u64), f64>,
    /// Covered fraction and area (source pixels) per destination cell.
    frac_b: Vec<f64>,
    area_b: Vec<f64>,
}

fn ours(fx: &Fixture, densify: usize) -> Ours {
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let [dw, dh] = fx.nums::<2>("dst_size").map(|v| v as usize);
    let t = GenImgProjTransformer::new(fx.header("src_crs"), fx.nums("src_gt"), fx.header("dst_crs"), fx.nums("dst_gt")).unwrap();
    let w = exact_weights(&t, &GridWindow::new(sw, sh), dw, dh, &WeightOptions::new(densify));
    assert_eq!(w.dst_size, [dw, dh]);
    let s = w.normalised(Normalization::DestArea);
    let (row, col) = w.esmf_row_col(dw, sw);
    let covered = w.dst_covered();
    Ours {
        s: row.into_iter().zip(col).zip(s).collect(),
        frac_b: covered.iter().zip(&w.dst_area).map(|(c, a)| c / a).collect(),
        area_b: w.dst_area,
    }
}

/// A smooth field on the source cell centres, in lon/lat (every case's
/// source is EPSG:4326).
fn field(fx: &Fixture) -> Vec<f64> {
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let gt: [f64; 6] = fx.nums("src_gt");
    let mut v = Vec::with_capacity(sw * sh);
    for r in 0..sh {
        for c in 0..sw {
            let (x, y) = (c as f64 + 0.5, r as f64 + 0.5);
            let lon = (gt[0] + x * gt[1] + y * gt[2]).to_radians();
            let lat = (gt[3] + x * gt[4] + y * gt[5]).to_radians();
            v.push(2.0 + lat.cos().powi(2) * (2.0 * lon).cos() + 0.5 * (3.0 * lat).sin());
        }
    }
    v
}

fn regrid(s: &HashMap<(u64, u64), f64>, src: &[f64], n_b: usize) -> Vec<f64> {
    let mut d = vec![0.0; n_b];
    for (&(row, col), &w) in s {
        d[row as usize - 1] += w * src[col as usize - 1];
    }
    d
}

#[derive(Debug, Default)]
struct Diff {
    /// max |S ours - S ESMF| over the union of entries (a missing entry is 0)
    s: f64,
    /// entries in one map only, and the largest S among them
    only: (usize, usize, f64),
    /// max |frac_b ours - frac_b ESMF|
    frac_b: f64,
    /// max relative difference of the regridded smooth field, over
    /// destination cells both cover completely
    field: f64,
    /// max |row sum - frac_b| of the ESMF map: ESMF's own constant check
    esmf_rows: f64,
}

fn diff(fx: &Fixture, o: &Ours) -> Diff {
    let mut d = Diff::default();
    for (k, &e) in &fx.s {
        match o.s.get(k) {
            Some(&r) => d.s = d.s.max((r - e).abs()),
            None => {
                d.only.1 += 1;
                d.only.2 = d.only.2.max(e.abs());
                d.s = d.s.max(e.abs());
            }
        }
    }
    for (k, &r) in &o.s {
        if !fx.s.contains_key(k) {
            d.only.0 += 1;
            d.only.2 = d.only.2.max(r.abs());
            d.s = d.s.max(r.abs());
        }
    }
    let n_b = fx.frac_b.len();
    assert_eq!(o.frac_b.len(), n_b);
    d.frac_b = o.frac_b.iter().zip(&fx.frac_b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let src = field(fx);
    let (fo, fe) = (regrid(&o.s, &src, n_b), regrid(&fx.s, &src, n_b));
    let ones = vec![1.0; src.len()];
    let rows = regrid(&fx.s, &ones, n_b);
    for i in 0..n_b {
        d.esmf_rows = d.esmf_rows.max((rows[i] - fx.frac_b[i]).abs());
        if fx.frac_b[i] > 1.0 - 1e-9 && o.frac_b[i] > 1.0 - 1e-9 {
            d.field = d.field.max(((fo[i] - fe[i]) / fe[i]).abs());
        }
    }
    d
}

const CASES: [&str; 6] = ["regular_coarser", "regular_finer", "regular_overhang", "rotated_pole", "lonlat_3031", "lonlat_laea"];

/// Cartesian ESMF in source pixel space is rwarp's cell definition at
/// densify 1: weights, fractions and areas agree to rounding.
#[test]
fn cartesian_esmf_matches_exactly() {
    for name in CASES {
        let fx = read_fixture(&format!("{name}.cart.txt"));
        let o = ours(&fx, 1);
        let d = diff(&fx, &o);
        let area = o.area_b.iter().zip(&fx.area_b).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        eprintln!("{name:16} cart: {d:?}, max relative area difference {area:.2e}");
        assert!(d.s < 1e-9, "{name}: S differs by {:e}", d.s);
        assert!(d.frac_b < 1e-9, "{name}: frac_b differs by {:e}", d.frac_b);
        assert!(area < 1e-9, "{name}: area differs by {area:e}");
        assert!(d.field < 1e-9, "{name}: field differs by {:e}", d.field);
        assert!(d.esmf_rows < 1e-9, "{name}: ESMF row sums off by {:e}", d.esmf_rows);
    }
}

/// Spherical ESMF (great-circle edges, spherical areas) against rwarp's
/// planar measure: the recorded differences, per case, at densify 1 and 8.
/// Ceilings are [S, frac_b, field], about 1.5 times what was measured
/// (ESMF 8.9.1, rwarp 0.3.1), over both densities.
///
/// What they show: the S differences (up to 0.6% at regular_finer, where
/// one destination cell spans parts of several source rows) are the planar
/// against spherical measure, and stage 3's spherical measure is what
/// should close them. Densify 8 is further from ESMF than densify 1 for the
/// projected cases: ESMF joins a destination cell's four corners with great
/// circles, close to densify 1's chords in lon/lat, while densify 8
/// follows the edge as straight in the destination projection. frac_b
/// agrees to rounding except where the destination overhangs the source.
#[test]
fn spherical_esmf_differences_are_recorded() {
    #[rustfmt::skip]
    let ceilings: [(&str, [f64; 3]); 6] = [
        ("regular_coarser",  [1.0e-3, 1e-12, 7.0e-5]),
        ("regular_finer",    [9.5e-3, 1e-12, 5.0e-4]),
        ("regular_overhang", [9.0e-4, 5e-4,  2.0e-5]),
        ("rotated_pole",     [1.9e-3, 1e-12, 7.5e-5]),
        ("lonlat_3031",      [2.5e-3, 1e-12, 1.6e-5]),
        ("lonlat_laea",      [1.9e-3, 1e-12, 1.2e-5]),
    ];
    for (name, [cs, cf, cv]) in ceilings {
        let fx = read_fixture(&format!("{name}.sph.txt"));
        for densify in [1, 8] {
            let d = diff(&fx, &ours(&fx, densify));
            eprintln!("{name:16} sph densify {densify}: {d:?}");
            assert!(d.esmf_rows < 1e-9, "{name}: ESMF row sums off by {:e}", d.esmf_rows);
            assert!(d.s <= cs, "{name} densify {densify}: S differs by {:e} > {cs:e}", d.s);
            assert!(d.frac_b <= cf, "{name} densify {densify}: frac_b differs by {:e} > {cf:e}", d.frac_b);
            assert!(d.field <= cv, "{name} densify {densify}: field differs by {:e} > {cv:e}", d.field);
        }
    }
}
