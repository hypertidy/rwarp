//! The ESMF oracle fixtures (`tests/fixtures/esmf/`, made by `gen.py`) and
//! the comparison of a map with them, shared by `tests/esmf.rs`,
//! `tests/esmf_tripolar.rs` and `tests/esmf_mesh.rs`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

pub struct Fixture {
    pub header: Vec<String>,
    pub frac_b: Vec<f64>,
    pub area_b: Vec<f64>,
    /// (ESMF row, ESMF col) -> S
    pub s: HashMap<(u64, u64), f64>,
}

impl Fixture {
    pub fn header(&self, key: &str) -> &str {
        self.header
            .iter()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
            .unwrap_or_else(|| panic!("no '{key}' in fixture header"))
    }

    pub fn nums<const N: usize>(&self, key: &str) -> [f64; N] {
        let v: Vec<f64> = self
            .header(key)
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        v.try_into()
            .unwrap_or_else(|_| panic!("'{key}' needs {N} numbers"))
    }
}

pub fn fixture_path(name: &str) -> PathBuf {
    [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "esmf",
        name,
    ]
    .iter()
    .collect()
}

pub fn read_fixture(name: &str) -> Fixture {
    let path = fixture_path(name);
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
    let mut counts = counts
        .split_whitespace()
        .map(|v| v.parse::<usize>().unwrap());
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
    assert_eq!(
        (frac_b.len(), s.len()),
        (n_b, n_s),
        "{name}: truncated fixture"
    );
    Fixture {
        header,
        frac_b,
        area_b,
        s,
    }
}

/// rwarp's side of one case, in ESMF's terms.
pub struct Ours {
    pub s: HashMap<(u64, u64), f64>,
    /// Covered fraction and area per destination cell.
    pub frac_b: Vec<f64>,
    pub area_b: Vec<f64>,
}

pub fn regrid(s: &HashMap<(u64, u64), f64>, src: &[f64], n_b: usize) -> Vec<f64> {
    let mut d = vec![0.0; n_b];
    for (&(row, col), &w) in s {
        d[row as usize - 1] += w * src[col as usize - 1];
    }
    d
}

#[derive(Debug, Default)]
pub struct Diff {
    /// max |S ours - S ESMF| over the union of entries (a missing entry is 0)
    pub s: f64,
    /// entries in one map only, and the largest S among them
    pub only: (usize, usize, f64),
    /// max |frac_b ours - frac_b ESMF|
    pub frac_b: f64,
    /// max relative difference of the regridded smooth field, over
    /// destination cells both cover completely
    pub field: f64,
    /// max |row sum - frac_b| of the ESMF map: ESMF's own constant check
    pub esmf_rows: f64,
}

/// Compare `o` with the fixture, regridding `src` (one value per source
/// cell, in ESMF's order) with both maps.
pub fn diff(fx: &Fixture, o: &Ours, src: &[f64]) -> Diff {
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
    d.frac_b = o
        .frac_b
        .iter()
        .zip(&fx.frac_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let (fo, fe) = (regrid(&o.s, src, n_b), regrid(&fx.s, src, n_b));
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

/// The smooth test field on the sphere, at longitude and latitude in degrees.
pub fn smooth(lon: f64, lat: f64) -> f64 {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    2.0 + lat.cos().powi(2) * (2.0 * lon).cos() + 0.5 * (3.0 * lat).sin()
}
