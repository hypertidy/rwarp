//! ESMF's second-order conservative maps (`conserve_2nd`) against
//! `Weights::second_order` (design/conservative.md, section 8).
//!
//! The fixtures `tests/fixtures/esmf/<case>.cart2.txt` are made by `gen.py`
//! with ESMPy, Cartesian in source pixel space as the first-order
//! `<case>.cart.txt` are, so the cells are rwarp's at densify 1. The method
//! is the same: each source cell's gradient by Green's theorem round its
//! neighbours' centres, centred on the part of the cell the map covers,
//! with the same fallbacks at the grid's edge. One thing differs: ESMF
//! centres an overlap on the mean of its polygon's vertices, rwarp on its
//! centroid. Those agree for a grid onto a grid (rectangles), so there the
//! weights must agree to rounding. For the projected case they don't, and
//! the tests show both sides of it:
//!
//! - given ESMF's vertex means, rwarp's table is ESMF's to rounding (away
//!   from destination corners on a source wall, where a vertex mean
//!   depends on how a clip lists the vertices there);
//! - with true centroids, rwarp is exact for a field linear in source
//!   pixels, and ESMF is not.
//!
//! `cargo test --test esmf_second_order -- --nocapture` prints the
//! differences.
#![cfg(feature = "proj")]

mod esmf_common;

use esmf_common::{diff, read_fixture, regrid, smooth, Fixture, Ours};
use rwarp::transform::GenImgProjTransformer;
use rwarp::weights::{exact_weights, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::Transformer;

fn transformer(fx: &Fixture) -> GenImgProjTransformer {
    GenImgProjTransformer::new(fx.header("src_crs"), fx.nums("src_gt"), fx.header("dst_crs"), fx.nums("dst_gt")).unwrap()
}

/// rwarp's first-order table with moments, at densify 1.
fn table(fx: &Fixture) -> Weights {
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let [dw, dh] = fx.nums::<2>("dst_size").map(|v| v as usize);
    exact_weights(&transformer(fx), &GridWindow::new(sw, sh), dw, dh, &WeightOptions::new(1).with_moments(true))
}

/// `w`'s moments replaced by ESMF's overlap centres: the mean of the
/// overlap polygon's vertices (ESMCI_SM.C, `_calc_centroid_2D_2D_cart`),
/// consecutive vertices within 1e-15 taken as one, as ESMF's
/// `remove_0len_edges2D` does. Each overlap is the destination cell's
/// quadrilateral (its corners through the transformer, as at densify 1)
/// clipped by the source cell.
fn with_vertex_means(fx: &Fixture, mut w: Weights) -> Weights {
    let t = transformer(fx);
    for k in 0..w.len() {
        let (dr, dc) = (w.dst_row[k] as f64, w.dst_col[k] as f64);
        let (mut x, mut y) = (vec![dc, dc + 1.0, dc + 1.0, dc], vec![dr, dr, dr + 1.0, dr + 1.0]);
        t.transform(true, &mut x, &mut y);
        let (i, j) = (w.src_col[k] as f64, w.src_row[k] as f64);
        let mut p: Vec<[f64; 2]> = x.iter().zip(&y).map(|(&x, &y)| [x - i, y - j]).collect();
        // Sutherland-Hodgman against the unit cell
        for (a, b, c) in [(1.0, 0.0, 0.0), (-1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, -1.0, 1.0)] {
            let f = |q: [f64; 2]| a * q[0] + b * q[1] + c;
            let mut out = Vec::new();
            for n in 0..p.len() {
                let (s, e) = (p[n], p[(n + 1) % p.len()]);
                let (fs, fe) = (f(s), f(e));
                if fs >= 0.0 {
                    out.push(s);
                }
                if (fs >= 0.0) != (fe >= 0.0) {
                    let u = fs / (fs - fe);
                    out.push([s[0] + u * (e[0] - s[0]), s[1] + u * (e[1] - s[1])]);
                }
            }
            p = out;
        }
        let same = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15;
        p.dedup_by(|b, a| same(*a, *b));
        while p.len() > 1 && same(p[0], p[p.len() - 1]) {
            p.pop();
        }
        if p.len() >= 3 {
            let m = p.iter().fold([0.0, 0.0], |s, q| [s[0] + q[0], s[1] + q[1]]);
            let n = p.len() as f64;
            w.moment[k] = [w.frac[k] * m[0] / n, w.frac[k] * m[1] / n];
        }
    }
    w
}

/// `w` folded, in ESMF's terms.
fn ours(fx: &Fixture, w: &Weights) -> Ours {
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let [dw, _] = fx.nums::<2>("dst_size").map(|v| v as usize);
    let src = GridWindow::new(sw, sh);
    let folded = w.second_order(&src, None, None);
    let s = folded.normalised(Normalization::DestArea);
    let (row, col) = folded.esmf_row_col(dw, sw);
    let covered = w.dst_covered();
    Ours {
        s: row.into_iter().zip(col).zip(s).collect(),
        frac_b: covered.iter().zip(&w.dst_area).map(|(c, a)| c / a).collect(),
        area_b: w.dst_area.clone(),
    }
}

/// A smooth field on the source cell centres, in lon/lat.
fn field(fx: &Fixture) -> Vec<f64> {
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let gt: [f64; 6] = fx.nums("src_gt");
    let mut v = Vec::with_capacity(sw * sh);
    for r in 0..sh {
        for c in 0..sw {
            let (x, y) = (c as f64 + 0.5, r as f64 + 0.5);
            v.push(smooth(gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5]));
        }
    }
    v
}

const CASES: [&str; 3] = ["regular_coarser", "regular_overhang", "lonlat_3031"];

/// The destination cells to compare where ESMF's overlap centres are
/// taken: all but those that share a source cell with a destination cell
/// that has a corner on a source cell's wall. There an overlap's vertices
/// include points on the wall, which ESMF's clip (the source cell clipped
/// by the destination cell) and this test's (the other way round) list
/// differently, and a vertex mean, unlike a centroid, depends on that.
fn away_from_walls(fx: &Fixture, w: &Weights) -> Vec<bool> {
    let t = transformer(fx);
    let [dw, dh] = fx.nums::<2>("dst_size").map(|v| v as usize);
    let mut on_wall = vec![false; dw * dh];
    for r in 0..dh {
        for c in 0..dw {
            let (c, r0) = (c as f64, r as f64);
            let (mut x, mut y) = (vec![c, c + 1.0, c + 1.0, c], vec![r0, r0, r0 + 1.0, r0 + 1.0]);
            t.transform(true, &mut x, &mut y);
            on_wall[r * dw + c as usize] = x.iter().chain(&y).any(|v| (v - v.round()).abs() < 1e-9);
        }
    }
    let cell = |k: usize| w.dst_row[k] as usize * dw + w.dst_col[k] as usize;
    let near: std::collections::HashSet<(u32, u32)> =
        (0..w.len()).filter(|&k| on_wall[cell(k)]).map(|k| (w.src_row[k], w.src_col[k])).collect();
    let mut ok = vec![true; dw * dh];
    for k in 0..w.len() {
        if near.contains(&(w.src_row[k], w.src_col[k])) {
            ok[cell(k)] = false;
        }
    }
    ok
}

/// With ESMF's overlap centres, rwarp's second-order table is ESMF's, to
/// rounding: the same gradient, the same centre of each source cell's
/// covered part, the same fallbacks at the grid's edge.
#[test]
fn with_esmfs_overlap_centres_the_tables_are_the_same() {
    for name in CASES {
        let fx = read_fixture(&format!("{name}.cart2.txt"));
        let w = with_vertex_means(&fx, table(&fx));
        let o = ours(&fx, &w);
        let ok = away_from_walls(&fx, &w);
        let (mut worst, mut n) = (0.0f64, 0);
        for (&(row, col), &e) in &fx.s {
            if ok[row as usize - 1] {
                worst = worst.max((o.s.get(&(row, col)).copied().unwrap_or(0.0) - e).abs());
                n += 1;
            }
        }
        for (&(row, col), &r) in &o.s {
            if ok[row as usize - 1] && !fx.s.contains_key(&(row, col)) {
                worst = worst.max(r.abs());
            }
        }
        let skipped = ok.iter().filter(|&&k| !k).count();
        eprintln!("{name:16} cart2, ESMF's overlap centres: S differs by {worst:.1e} over {n} weights ({skipped} cells by a wall skipped)");
        assert!(n > 5000 && worst < 1e-11, "{name}: S differs by {worst:e} over {n} weights");
        if name != "lonlat_3031" {
            assert_eq!(skipped, 0, "{name}");
        }
    }
}

/// With the overlaps' true centroids (rwarp's moments), the tables are the
/// same where every overlap is a rectangle (a grid on a grid), and differ
/// where the overlaps are other polygons, whose vertex mean is not their
/// centroid. There rwarp comes out exact for fields linear in source
/// pixels, and ESMF does not. Ceilings for the difference in S.
#[test]
fn with_true_centroids_rwarp_is_exact_where_esmf_is_not() {
    for (name, ceiling) in [("regular_coarser", 1e-12), ("regular_overhang", 1e-12), ("lonlat_3031", 0.035)] {
        let fx = read_fixture(&format!("{name}.cart2.txt"));
        let w = table(&fx);
        let o = ours(&fx, &w);
        let d = diff(&fx, &o, &field(&fx));
        // a field linear in source pixels, at the source cells' centres
        let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
        let lin = |x: f64, y: f64| 2.0 + 0.3 * x - 0.7 * y;
        let src: Vec<f64> = (0..sw * sh).map(|k| lin((k % sw) as f64 + 0.5, (k / sw) as f64 + 0.5)).collect();
        let n_b = fx.frac_b.len();
        let (a, b) = (regrid(&o.s, &src, n_b), regrid(&fx.s, &src, n_b));
        // the exact mean over each destination cell: the field at its centroid
        let t = transformer(&fx);
        let [dw, _] = fx.nums::<2>("dst_size").map(|v| v as usize);
        let (mut ours_err, mut esmf_err, mut n) = (0.0f64, 0.0f64, 0);
        for cell in 0..n_b {
            // destination cells whose sources are wholly covered and have
            // all their neighbours: rows of the first-order table
            if !interior(&w, cell, sw, sh) {
                continue;
            }
            let (dr, dc) = ((cell / dw) as f64, (cell % dw) as f64);
            let (mut x, mut y) = (vec![dc, dc + 1.0, dc + 1.0, dc], vec![dr, dr, dr + 1.0, dr + 1.0]);
            t.transform(true, &mut x, &mut y);
            let want = lin(polygon_centroid(&x, &y)[0], polygon_centroid(&x, &y)[1]) * o.frac_b[cell];
            ours_err = ours_err.max((a[cell] - want).abs());
            esmf_err = esmf_err.max((b[cell] - want).abs());
            n += 1;
        }
        eprintln!("{name:16} cart2: S differs by {:.1e}; on a linear field over {n} cells, rwarp off by {ours_err:.1e}, ESMF {esmf_err:.1e}", d.s);
        assert!(d.s <= ceiling, "{name}: S differs by {:e} > {ceiling:e}", d.s);
        assert!(n > 10 && ours_err < 1e-11, "{name}: rwarp off by {ours_err:e} over {n} cells");
        if name == "lonlat_3031" {
            assert!(esmf_err > 1e-4, "ESMF off by only {esmf_err:e}: is its overlap centre still the vertex mean?");
        }
    }
}

/// The centroid of a polygon by the shoelace formulas.
fn polygon_centroid(x: &[f64], y: &[f64]) -> [f64; 2] {
    let n = x.len();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let j = (i + 1) % n;
        let c = (x[i] - x[0]) * (y[j] - y[0]) - (x[j] - x[0]) * (y[i] - y[0]);
        a += c;
        cx += (x[i] + x[j] - 2.0 * x[0]) * c;
        cy += (y[i] + y[j] - 2.0 * y[0]) * c;
    }
    [x[0] + cx / (3.0 * a), y[0] + cy / (3.0 * a)]
}

/// Whether every source cell of destination cell `cell` is wholly covered
/// by the table and has all eight neighbours in the source.
fn interior(w: &Weights, cell: usize, sw: usize, sh: usize) -> bool {
    let mut covered = std::collections::HashMap::<(u32, u32), f64>::new();
    for k in 0..w.len() {
        *covered.entry((w.src_row[k], w.src_col[k])).or_default() += w.frac[k];
    }
    let dw = w.dst_size[0];
    let mut any = false;
    for k in 0..w.len() {
        if (w.dst_row[k] as usize) * dw + w.dst_col[k] as usize != cell {
            continue;
        }
        any = true;
        let (r, c) = (w.src_row[k] as usize, w.src_col[k] as usize);
        if (covered[&(w.src_row[k], w.src_col[k])] - 1.0).abs() > 1e-12 || r == 0 || c == 0 || r + 1 == sh || c + 1 == sw {
            return false;
        }
    }
    any
}

