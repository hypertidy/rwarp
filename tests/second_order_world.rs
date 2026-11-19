//! Second order from sources with world cells (conservative stage 6b): a
//! curvilinear grid through `MeshTransformer`, the same grid as a
//! `MeshSet` in two bands, and grids onto a `PolyMesh` and a set. The
//! planes are in the cells' index space under the walk's equal-area map
//! (decided 2026-11-19), so:
//!
//! - each entry's fraction and moments are the overlap's area and first
//!   moments in that space, worked out here without the walk, by clipping
//!   each destination cell against the source cell's four triangles in
//!   world coordinates and mapping the pieces through the split;
//! - a field linear in the grid's indices comes out exactly, and a world
//!   quadratic's error falls as the square of the cell size;
//! - the integral is conserved and the folded table is the same method;
//! - the set's pieces, bands and worlds, threads, and the mesh paths keep
//!   the moments.

use std::collections::HashMap;

use rwarp::mesh::{Band, CellMap, Halo, MeshSet, MeshTransformer, PolyMesh, QuadMesh, SetTransformer, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::SourceBuffer;
use rwarp::weights::{
    exact_weights, exact_weights_par, source_cell_areas, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights,
};
use rwarp::CrsTransform;

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// A turn by 30 degrees about (145, -36), or back: area-preserving, so a
/// band held in it measures as one held in longitude and latitude.
struct Turn(bool);
impl CrsTransform for Turn {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let a = if self.0 { -30f64 } else { 30f64 }.to_radians();
        let (s, c) = a.sin_cos();
        let (u, v) = (x - 145.0, y + 36.0);
        Some((145.0 + c * u - s * v, -36.0 + s * u + c * v))
    }
}

/// The curved grid's corner at `(s, t)` in the unit square: smooth, its
/// cells far from parallelograms, rows going south.
fn curve(s: f64, t: f64) -> [f64; 2] {
    [140.0 + 10.0 * s + 0.6 * (1.3 + 2.5 * t).sin() * s, -32.0 - 8.0 * t + 0.5 * (0.7 + 3.0 * s).sin() + 0.3 * s * t]
}

/// The grid's corner `(j, i)` with `ncol` by `nrow` cells.
fn corner(ncol: usize, nrow: usize, j: usize, i: usize) -> [f64; 2] {
    curve(i as f64 / ncol as f64, j as f64 / nrow as f64)
}

fn curved(ncol: usize, nrow: usize) -> QuadMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = corner(ncol, nrow, j, i);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap()
}

/// The destination: a regular grid over [141, 149] x [-39, -33] in the
/// grid's own coordinates, `d` degrees a cell.
fn dst(d: f64) -> ([f64; 6], usize, usize) {
    ([141.0, d, 0.0, -33.0, 0.0, -d], (8.0 / d).round() as usize, (6.0 / d).round() as usize)
}

fn transformer(mesh: QuadMesh, gt: [f64; 6]) -> MeshTransformer {
    MeshTransformer::with_backend(mesh, gt, Box::new(Same), Box::new(Same)).unwrap()
}

/// Destination cell `(r, c)` of `gt` as a ring, anticlockwise.
fn dst_cell(gt: [f64; 6], r: usize, c: usize) -> Vec<[f64; 2]> {
    let at = |c: f64, r: f64| [gt[0] + c * gt[1], gt[3] + r * gt[5]];
    let (c, r) = (c as f64, r as f64);
    vec![at(c, r + 1.0), at(c + 1.0, r + 1.0), at(c + 1.0, r), at(c, r)]
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// `poly` clipped to the triangle `tri` (Sutherland-Hodgman).
fn clip(poly: &[[f64; 2]], tri: [[f64; 2]; 3]) -> Vec<[f64; 2]> {
    let tri = if cross(tri[0], tri[1], tri[2]) < 0.0 { [tri[0], tri[2], tri[1]] } else { tri };
    let mut out = poly.to_vec();
    for e in 0..3 {
        let (a, b) = (tri[e], tri[(e + 1) % 3]);
        let inp = std::mem::take(&mut out);
        for i in 0..inp.len() {
            let (p, q) = (inp[i], inp[(i + 1) % inp.len()]);
            let (sp, sq) = (cross(a, b, p), cross(a, b, q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// A polygon's area and first moments (shoelace).
fn moments(p: &[[f64; 2]]) -> [f64; 3] {
    let (mut a, mut mx, mut my) = (0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let (u, v) = (p[i], p[(i + 1) % p.len()]);
        let cr = u[0] * v[1] - v[0] * u[1];
        a += cr;
        mx += (u[0] + v[0]) * cr;
        my += (u[1] + v[1]) * cr;
    }
    [a / 2.0, mx / 6.0, my / 6.0]
}

/// The overlap of the world ring `ring` with cell `(r, c)` of `mesh`, in
/// the cell's unit square under the equal-area split: its area and first
/// moments about the cell's corner. Worked out in coordinates about the
/// cell's first corner.
fn overlap_in_index(mesh: &QuadMesh, r: usize, c: usize, ring: &[[f64; 2]]) -> [f64; 3] {
    let k = mesh.corners(r, c);
    let o = k[0];
    let rel = |p: [f64; 2]| [p[0] - o[0], p[1] - o[1]];
    let (centre, unit) = mesh.split_centre(r, c);
    let (wc, ring): ([f64; 2], Vec<[f64; 2]>) = (rel(centre), ring.iter().map(|&p| rel(p)).collect());
    let units = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut sum = [0.0; 3];
    for s in 0..4 {
        let (wa, wb) = (rel(k[s]), rel(k[(s + 1) % 4]));
        let (ua, ub) = (units[s], units[(s + 1) % 4]);
        let piece = clip(&ring, [wa, wb, wc]);
        if piece.len() < 3 {
            continue;
        }
        // world to the unit square, affine on the triangle
        let det = cross(wa, wb, wc);
        if det == 0.0 {
            continue;
        }
        let to_unit = |p: [f64; 2]| {
            let l1 = cross(wa, p, wc) / det;
            let l2 = cross(wa, wb, p) / det;
            [
                ua[0] + l1 * (ub[0] - ua[0]) + l2 * (unit[0] - ua[0]),
                ua[1] + l1 * (ub[1] - ua[1]) + l2 * (unit[1] - ua[1]),
            ]
        };
        // the map turns the piece over where rows go south: its moments
        // carry its area's sign
        let m = moments(&piece.iter().map(|&p| to_unit(p)).collect::<Vec<_>>());
        let sign = m[0].signum();
        for i in 0..3 {
            sum[i] += sign * m[i];
        }
    }
    sum
}

/// The destination cells whose sources the table covers wholly and whose
/// sources' neighbours are all in the window: where a field linear in the
/// grid's indices must come out exactly.
fn interior(w: &Weights, src: &GridWindow) -> Vec<bool> {
    let n = w.dst_size[0] * w.dst_size[1];
    let mut covered = HashMap::<(u32, u32), f64>::new();
    for k in 0..w.len() {
        *covered.entry((w.src_row[k], w.src_col[k])).or_default() += w.frac[k];
    }
    let inner = |r: u32, c: u32| {
        let (r, c) = (r as usize, c as usize);
        r > src.row_off && c > src.col_off && r + 1 < src.row_off + src.nrow && c + 1 < src.col_off + src.ncol
    };
    let (mut ok, mut any) = (vec![true; n], vec![false; n]);
    for k in 0..w.len() {
        let cell = (w.dst_row[k] as usize - w.dst_off[1]) * w.dst_size[0] + (w.dst_col[k] as usize - w.dst_off[0]);
        any[cell] = true;
        let full = (covered[&(w.src_row[k], w.src_col[k])] - 1.0).abs() < 1e-12;
        if !full || !inner(w.src_row[k], w.src_col[k]) {
            ok[cell] = false;
        }
    }
    ok.iter().zip(&any).map(|(&o, &a)| o && a).collect()
}

fn buffer<'a>(src: &GridWindow, data: &'a [f64]) -> SourceBuffer<'a, f64> {
    SourceBuffer { data, ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: src.col_off, row_off: src.row_off }
}

/// A field linear in the grid's indices: its cell means are its values
/// at the cells' middles in index space.
fn index_linear(u: f64, v: f64) -> f64 {
    3.0 + 0.7 * u - 0.4 * v
}

/// The largest difference of the table's fractions and moments from the
/// clipped overlaps', for destination cells of `gt`.
fn against_clips(w: &Weights, mesh: &QuadMesh, gt: [f64; 6]) -> (f64, f64) {
    let (mut df, mut dm) = (0.0f64, 0.0f64);
    for k in 0..w.len() {
        let ring = dst_cell(gt, w.dst_row[k] as usize, w.dst_col[k] as usize);
        let want = overlap_in_index(mesh, w.src_row[k] as usize, w.src_col[k] as usize, &ring);
        df = df.max((w.frac[k] - want[0]).abs());
        dm = dm.max((w.moment[k][0] - want[1]).abs()).max((w.moment[k][1] - want[2]).abs());
    }
    (df, dm)
}

#[test]
fn moments_are_the_overlaps_in_index_space() {
    let (gt, nc, nr) = dst(0.25);
    let mesh = curved(25, 20);
    let t = transformer(mesh.clone(), gt);
    let w = exact_weights(&t, &t.source_window(), nc, nr, &WeightOptions::new(1).with_moments(true));
    assert_eq!(w.moment.len(), w.len());
    let (df, dm) = against_clips(&w, &mesh, gt);
    eprintln!("{} entries: fractions to {df:.1e}, moments to {dm:.1e} of the clipped overlaps", w.len());
    assert!(df < 1e-12 && dm < 1e-12, "{df:e} {dm:e}");
    // a cell the destination covers wholly is centred on its middle
    let mut sums = HashMap::<(u32, u32), [f64; 3]>::new();
    for k in 0..w.len() {
        let s = sums.entry((w.src_row[k], w.src_col[k])).or_default();
        s[0] += w.frac[k];
        s[1] += w.moment[k][0];
        s[2] += w.moment[k][1];
    }
    let (mut whole, mut off) = (0, 0.0f64);
    for s in sums.values().filter(|s| (s[0] - 1.0).abs() < 1e-12) {
        whole += 1;
        off = off.max((s[1] / s[0] - 0.5).abs()).max((s[2] / s[0] - 0.5).abs());
    }
    eprintln!("{whole} cells covered wholly, centred on their middles to {off:.1e}");
    assert!(whole > 100 && off < 1e-12, "{off:e}");
}

#[test]
fn fields_linear_in_the_indices_come_out_exactly() {
    let (gt, nc, nr) = dst(0.25);
    let t = transformer(curved(25, 20), gt);
    let src = t.source_window();
    let w = exact_weights(&t, &src, nc, nr, &WeightOptions::new(1).with_moments(true));
    let data: Vec<f64> =
        (0..src.nrow * src.ncol).map(|p| index_linear((p % src.ncol) as f64 + 0.5, (p / src.ncol) as f64 + 0.5)).collect();
    let buf = buffer(&src, &data);
    let second: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
    // the exact mean: each overlap's area in world times the field at its
    // centroid in index space, from the clips
    let mesh = t.source_mesh();
    let inside = interior(&w, &src);
    let (mut e1, mut e2, mut n) = (0.0f64, 0.0f64, 0);
    let mut by_cell = HashMap::<usize, Vec<usize>>::new();
    for k in 0..w.len() {
        by_cell.entry(w.dst_row[k] as usize * nc + w.dst_col[k] as usize).or_default().push(k);
    }
    for (&cell, ks) in &by_cell {
        if !inside[cell] {
            continue;
        }
        let ring = dst_cell(gt, cell / nc, cell % nc);
        let (mut a, mut s) = (0.0, 0.0);
        for &k in ks {
            let (r, c) = (w.src_row[k] as usize, w.src_col[k] as usize);
            let m = overlap_in_index(mesh, r, c, &ring);
            let area = mesh.cell_area(r, c) * m[0];
            a += area;
            s += area * index_linear(c as f64 + m[1] / m[0], r as f64 + m[2] / m[0]);
        }
        let want = s / a;
        e1 = e1.max((first[cell] - want).abs());
        e2 = e2.max((second[cell] - want).abs());
        n += 1;
    }
    eprintln!("{n} interior cells: first order off by {e1:.2e}, second by {e2:.2e}");
    assert!(n > 300);
    assert!(e2 < 1e-12 && e1 > 1e-3, "{e1:e} {e2:e}");
}

/// The mean over a polygon in world coordinates of a quadratic at length
/// scale `s` about (145, -36), exactly (second moments
/// by the shoelace), about the polygon's first point.
fn quadratic_mean(s: f64, p: &[[f64; 2]]) -> f64 {
    let o = p[0];
    let q: Vec<[f64; 2]> = p.iter().map(|v| [v[0] - o[0], v[1] - o[1]]).collect();
    let (mut a, mut mx, mut my, mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..q.len() {
        let (u, v) = (q[i], q[(i + 1) % q.len()]);
        let cr = u[0] * v[1] - v[0] * u[1];
        a += cr / 2.0;
        mx += (u[0] + v[0]) * cr / 6.0;
        my += (u[1] + v[1]) * cr / 6.0;
        xx += (u[0] * u[0] + u[0] * v[0] + v[0] * v[0]) * cr / 12.0;
        yy += (u[1] * u[1] + u[1] * v[1] + v[1] * v[1]) * cr / 12.0;
        xy += (u[0] * v[1] + 2.0 * u[0] * u[1] + 2.0 * v[0] * v[1] + v[0] * u[1]) * cr / 24.0;
    }
    // the quadratic in the shifted coordinates
    let (x0, y0) = ((o[0] - 145.0) / s, (o[1] + 36.0) / s);
    let (ex, ey, exx, exy, eyy) = (mx / a / s, my / a / s, xx / a / (s * s), xy / a / (s * s), yy / a / (s * s));
    // E[(x0 + X)^2 + 0.5 (x0 + X)(y0 + Y) - 0.8 (y0 + Y)^2]
    (x0 * x0 + 2.0 * x0 * ex + exx) + 0.5 * (x0 * y0 + x0 * ey + y0 * ex + exy) - 0.8 * (y0 * y0 + 2.0 * y0 * ey + eyy)
}

/// RMS errors of first and second order over the interior cells, for the
/// quadratic at length scale `s` from a grid of `k` cells a degree.
fn rms_errors(k: usize, s: f64) -> (f64, f64) {
    let (gt, nc, nr) = dst(0.25);
    let t = transformer(curved(10 * k, 8 * k), gt);
    let src = t.source_window();
    let mesh = t.source_mesh();
    let w = exact_weights(&t, &src, nc, nr, &WeightOptions::new(1).with_moments(true));
    let data: Vec<f64> = (0..src.nrow * src.ncol)
        .map(|p| {
            let c = mesh.corners(p / src.ncol, p % src.ncol);
            quadratic_mean(s, &c)
        })
        .collect();
    let buf = buffer(&src, &data);
    let second: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
    let inside = interior(&w, &src);
    let (mut e1, mut e2, mut n) = (0.0, 0.0, 0.0);
    for r in 0..nr {
        for c in 0..nc {
            let cell = r * nc + c;
            if inside[cell] {
                let want = quadratic_mean(s, &dst_cell(gt, r, c));
                e1 += (first[cell] - want).powi(2);
                e2 += (second[cell] - want).powi(2);
                n += 1.0;
            }
        }
    }
    ((e1 / n).sqrt(), (e2 / n).sqrt())
}

#[test]
fn a_world_quadratic_is_second_order() {
    // planes in index space are off world planes by how far the cells are
    // from parallelograms, which falls as the square of the cell size too
    let (a1, a2) = rms_errors(2, 3.0);
    let (b1, b2) = rms_errors(4, 3.0);
    let (c1, c2) = rms_errors(8, 3.0);
    eprintln!("first order {a1:.3e} {b1:.3e} {c1:.3e}; second {a2:.3e} {b2:.3e} {c2:.3e}");
    eprintln!("second-order ratios {:.2} {:.2}", a2 / b2, b2 / c2);
    assert!(b2 / c2 > 3.5 && b2 / c2 < 4.5, "x{}", b2 / c2);
    assert!(c2 < 0.05 * c1, "{c1:e} {c2:e}");
}

#[test]
fn the_integral_is_conserved_and_the_folded_table_is_the_same_method() {
    let (gt, nc, nr) = dst(0.25);
    let t = transformer(curved(25, 20), gt);
    let src = t.source_window();
    let w = exact_weights(&t, &src, nc, nr, &WeightOptions::new(2).with_moments(true));
    let data: Vec<f64> = (0..src.nrow * src.ncol)
        .map(|p| {
            let (r, c) = ((p / src.ncol) as f64, (p % src.ncol) as f64);
            10.0 + (c / 4.0).sin() * (r / 3.0).cos()
        })
        .collect();
    let areas = source_cell_areas(&t, &src).unwrap();
    for mask in [false, true] {
        let valid: Vec<bool> = (0..data.len()).map(|p| !mask || (p * 7) % 11 != 3).collect();
        let masked: Vec<f64> = data.iter().zip(&valid).map(|(&v, &ok)| if ok { v } else { -999.0 }).collect();
        let buf = buffer(&src, &masked);
        let opts = ApplyOptions::new().with_src_nodata(rwarp::warp::Nodata::Value(-999.0));
        // conserved: the sum over destination cells is the table's
        // covered part of each source cell, times its value
        let v: Vec<f64> = w.apply_second_order(&buf, Normalization::DestArea, &opts);
        let got: f64 =
            v.iter().zip(&w.dst_area).filter(|(x, a)| x.is_finite() && **x != -999.0 && a.is_finite()).map(|(x, a)| x * a).sum();
        let mut want = 0.0;
        for k in 0..w.len() {
            let p = (w.src_row[k] as usize) * src.ncol + w.src_col[k] as usize;
            if valid[p] {
                want += w.frac[k] * w.src_area(k) * data[p];
            }
        }
        eprintln!("mask {mask}: integral {got:.12e}, the table's {want:.12e}");
        assert!((got - want).abs() < 1e-12 * want.abs(), "{got} vs {want}");
        // the folded table, applied, is apply_second_order
        let folded = w.second_order(&src, mask.then_some(&valid[..]), Some(&areas));
        assert_eq!(folded.src_cell_area.len(), folded.len());
        assert!(folded.frac.iter().any(|&f| f < 0.0));
        for norm in [Normalization::FracArea, Normalization::DestArea] {
            let a: Vec<f64> = w.apply_second_order(&buf, norm, &opts);
            let b: Vec<f64> = folded.apply(&buf, norm, &opts);
            let d = a.iter().zip(&b).filter(|(x, _)| x.is_finite()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            eprintln!("mask {mask}, {norm:?}: folded table to {d:.1e}");
            assert!(d < 1e-12, "{d:e}");
        }
    }
}

#[test]
#[should_panic(expected = "source_cell_areas")]
fn the_fold_needs_the_cell_areas() {
    let (gt, nc, nr) = dst(1.0);
    let t = transformer(curved(10, 8), gt);
    let w = exact_weights(&t, &t.source_window(), nc, nr, &WeightOptions::new(1).with_moments(true));
    w.second_order(&t.source_window(), None, None);
}

#[test]
fn threads_keep_the_moments() {
    let (gt, nc, nr) = dst(0.25);
    let mesh = curved(25, 20);
    let opts = WeightOptions::new(2).with_moments(true);
    let t = transformer(mesh.clone(), gt);
    let one = exact_weights(&t, &t.source_window(), nc, nr, &opts);
    let par = exact_weights_par(|| Ok(transformer(mesh.clone(), gt)), &t.source_window(), nc, nr, &opts, 4).unwrap();
    assert_eq!(one.frac, par.frac);
    assert_eq!(one.moment, par.moment);
}

/// The curved grid as CF bounds, `ncol` by `nrow`.
fn bounds(ncol: usize, nrow: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    for j in 0..nrow {
        for i in 0..ncol {
            for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)] {
                let p = corner(ncol, nrow, jj, ii);
                xb.push(p[0]);
                yb.push(p[1]);
            }
        }
    }
    (xb, yb)
}

/// Each entry by its cells, with its fraction and moments.
fn by_entry(w: &Weights) -> HashMap<(u32, u32, u32, u32), [f64; 3]> {
    (0..w.len())
        .map(|k| ((w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]), [w.frac[k], w.moment[k][0], w.moment[k][1]]))
        .collect()
}

fn max_diff(a: &HashMap<(u32, u32, u32, u32), [f64; 3]>, b: &HashMap<(u32, u32, u32, u32), [f64; 3]>) -> f64 {
    let mut d = 0.0f64;
    for (k, x) in a {
        let y = b.get(k).copied().unwrap_or([0.0; 3]);
        for i in 0..3 {
            d = d.max((x[i] - y[i]).abs());
        }
    }
    for (k, y) in b {
        if !a.contains_key(k) {
            d = d.max(y[0].abs());
        }
    }
    d
}

#[test]
fn a_set_in_two_worlds_is_its_mesh() {
    // the grid as a set: the north half in longitude and latitude, the
    // south half turned 30 degrees (each band a world of its own); the
    // split is affine-invariant, so its tables are the mesh's
    let (ncol, nrow) = (25, 20);
    let (gt, nc, nr) = dst(0.25);
    let (xb, yb) = bounds(ncol, nrow);
    let (same, turn) = (Same, Turn(false));
    let bands = [Band { rows: 0..10, to_world: &same }, Band { rows: 10..nrow, to_world: &turn }];
    let set = MeshSet::from_cf_bounds(ncol, nrow, &xb, &yb, Halo::NONE, &bands, None).unwrap();
    let opts = WeightOptions::new(2).with_moments(true);
    let t = transformer(curved(ncol, nrow), gt);
    let want = by_entry(&exact_weights(&t, &t.source_window(), nc, nr, &opts));
    // through the set's transformer: outlines in both worlds at once
    let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> =
        vec![(Box::new(Same), Box::new(Same)), (Box::new(Turn(false)), Box::new(Turn(true)))];
    let st = SetTransformer::with_backend(&set, gt, crs).unwrap();
    let through = exact_weights(&st, &st.source_window(), nc, nr, &opts);
    let d1 = max_diff(&by_entry(&through), &want);
    // piece by piece
    let pieces = set
        .weights_to(
            |k, mesh| {
                let (to, back): (Box<dyn CrsTransform>, Box<dyn CrsTransform>) =
                    if k == 0 { (Box::new(Same), Box::new(Same)) } else { (Box::new(Turn(false)), Box::new(Turn(true))) };
                MeshTransformer::with_backend(mesh, gt, to, back)
            },
            nc,
            nr,
            &opts,
        )
        .unwrap();
    let d2 = max_diff(&by_entry(&pieces), &want);
    // to rounding: the turn's, and the split's, whose areas are taken in
    // absolute coordinates (to about 1e-12 of a cell this far from 0)
    eprintln!("the set's tables are the mesh's to {d1:.1e} (through SetTransformer), {d2:.1e} (weights_to)");
    assert!(d1 < 1e-11 && d2 < 1e-11, "{d1:e} {d2:e}");
    // and so second order across the bands' meeting, with the set's areas
    let src = st.source_window();
    let data: Vec<f64> = (0..src.nrow * src.ncol).map(|p| index_linear((p % ncol) as f64 + 0.5, (p / ncol) as f64 + 0.5)).collect();
    let buf = buffer(&src, &data);
    let a: Vec<f64> = through.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let b: Vec<f64> = exact_weights(&t, &t.source_window(), nc, nr, &opts).apply_second_order(
        &buf,
        Normalization::FracArea,
        &ApplyOptions::new(),
    );
    let d = a.iter().zip(&b).filter(|(x, _)| x.is_finite()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    eprintln!("second order through the set: the mesh's to {d:.1e}");
    assert!(d < 1e-11, "{d:e}");
    let areas = source_cell_areas(&st, &src).unwrap();
    let mesh_areas = source_cell_areas(&t, &t.source_window()).unwrap();
    // the same areas, to the shoelace's rounding in absolute coordinates
    let da = areas.iter().zip(&mesh_areas).map(|(x, y)| (x - y).abs() / y).fold(0.0, f64::max);
    eprintln!("the set's cell areas are the mesh's to {da:.1e}");
    assert!(da < 1e-10, "{da:e}");
    through.second_order(&src, None, Some(&areas));
}

/// Triangles over [`lo`, `hi`]^2 in world coordinates, `n` squares a
/// side, each cut on a diagonal.
fn triangles(lo: [f64; 2], hi: [f64; 2], n: usize) -> PolyMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=n {
        for i in 0..=n {
            x.push(lo[0] + (hi[0] - lo[0]) * i as f64 / n as f64);
            y.push(lo[1] + (hi[1] - lo[1]) * j as f64 / n as f64);
        }
    }
    let node = |j: usize, i: usize| j * (n + 1) + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for j in 0..n {
        for i in 0..n {
            nodes.extend([node(j, i), node(j, i + 1), node(j + 1, i + 1)]);
            start.push(nodes.len());
            nodes.extend([node(j, i), node(j + 1, i + 1), node(j + 1, i)]);
            start.push(nodes.len());
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap()
}

#[test]
fn a_grid_onto_a_mesh_has_moments() {
    // a regular grid in pixels: a field linear in them comes out exactly
    // on the faces that the interior covers
    let gt = [140.0, 0.25, 0.0, -32.0, 0.0, -0.25];
    let grid = GridWindow::new(40, 32);
    let inner = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let mesh = triangles([141.1, -38.9], [148.9, -33.1], 13);
    let opts = WeightOptions::new(1).with_moments(true);
    let w = mesh.weights_onto(&inner, &grid, &opts).unwrap();
    assert_eq!(w.moment.len(), w.len());
    let make = || GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same));
    let par = mesh.weights_onto_par(make, &grid, &opts, 3).unwrap();
    assert_eq!(par.moment, w.moment);
    let f = |u: f64, v: f64| 2.0 - 0.3 * u + 0.45 * v;
    let data: Vec<f64> = (0..grid.ncol * grid.nrow).map(|p| f((p % grid.ncol) as f64 + 0.5, (p / grid.ncol) as f64 + 0.5)).collect();
    let buf = buffer(&grid, &data);
    let second: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let inside = interior(&w, &grid);
    let (mut e, mut n) = (0.0f64, 0);
    for face in 0..mesh.nface() {
        if !inside[face] {
            continue;
        }
        // the face's centroid in grid pixels
        let px: Vec<[f64; 2]> =
            mesh.face(face).iter().map(|&k| mesh.node(k)).map(|p| [(p[0] - gt[0]) / gt[1], (p[1] - gt[3]) / gt[5]]).collect();
        let m = moments(&px);
        e = e.max((second[face] - f(m[1] / m[0], m[2] / m[0])).abs());
        n += 1;
    }
    eprintln!("{n} faces inside: a field linear in pixels to {e:.1e}");
    assert!(n > 200 && e < 1e-12, "{e:e}");
    // from the curved grid's world cells: the cells the faces cover wholly
    // are centred on their middles in index space
    let t = transformer(curved(25, 20), WORLD_GT);
    let w = mesh.weights_onto(&t, &t.source_window(), &opts).unwrap();
    let mut sums = HashMap::<(u32, u32), [f64; 3]>::new();
    for k in 0..w.len() {
        let s = sums.entry((w.src_row[k], w.src_col[k])).or_default();
        s[0] += w.frac[k];
        s[1] += w.moment[k][0];
        s[2] += w.moment[k][1];
    }
    let (mut whole, mut off) = (0, 0.0f64);
    for s in sums.values().filter(|s| (s[0] - 1.0).abs() < 1e-12) {
        whole += 1;
        off = off.max((s[1] / s[0] - 0.5).abs()).max((s[2] / s[0] - 0.5).abs());
    }
    eprintln!("world cells onto the mesh: {whole} covered wholly, centred to {off:.1e}");
    assert!(whole > 100 && off < 1e-12, "{off:e}");
    // the faces as sources have them too (tests/second_order_faces.rs)
    let to = mesh.weights_to(&inner, &grid, &opts).unwrap();
    assert_eq!(to.moment.len(), to.len());
}

#[test]
fn a_grid_onto_a_set_has_moments() {
    let (ncol, nrow) = (25, 20);
    let (xb, yb) = bounds(ncol, nrow);
    let (same, turn) = (Same, Turn(false));
    let bands = [Band { rows: 0..10, to_world: &same }, Band { rows: 10..nrow, to_world: &turn }];
    let set = MeshSet::from_cf_bounds(ncol, nrow, &xb, &yb, Halo::NONE, &bands, None).unwrap();
    let gt = [139.0, 0.2, 0.0, -31.0, 0.0, -0.2];
    let grid = GridWindow::new(65, 50);
    let w = set
        .weights_onto(
            |k| {
                let (to, back): (Box<dyn CrsTransform>, Box<dyn CrsTransform>) =
                    if k == 0 { (Box::new(Same), Box::new(Same)) } else { (Box::new(Turn(true)), Box::new(Turn(false))) };
                GenImgProjTransformer::with_backend(gt, WORLD_GT, to, back)
            },
            &grid,
            &WeightOptions::new(2).with_moments(true),
        )
        .unwrap();
    assert_eq!(w.moment.len(), w.len());
    let mut sums = HashMap::<(u32, u32), [f64; 3]>::new();
    for k in 0..w.len() {
        let s = sums.entry((w.src_row[k], w.src_col[k])).or_default();
        s[0] += w.frac[k];
        s[1] += w.moment[k][0];
        s[2] += w.moment[k][1];
    }
    let (mut whole, mut off) = (0, 0.0f64);
    for s in sums.values().filter(|s| (s[0] - 1.0).abs() < 1e-12) {
        whole += 1;
        off = off.max((s[1] / s[0] - 0.5).abs()).max((s[2] / s[0] - 0.5).abs());
    }
    eprintln!("a grid onto the set: {whole} grid cells covered wholly, centred to {off:.1e}");
    assert!(whole > 500 && off < 1e-12, "{off:e}");
}

/// A curved mesh once round the world, with its period: `ncol` columns
/// from -180 to 180, rows from 60 N to 60 S.
fn round(ncol: usize, nrow: usize) -> QuadMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let a = std::f64::consts::TAU * i as f64 / ncol as f64;
            x.push(-180.0 + 360.0 * i as f64 / ncol as f64 + 2.0 * a.sin() * (0.3 * j as f64).cos());
            y.push(60.0 - 120.0 * j as f64 / nrow as f64 + 1.5 * (2.0 * a).sin());
        }
    }
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap().with_period(360.0).unwrap()
}

#[test]
fn moments_survive_the_seam_of_a_mesh() {
    // destination columns of 8 degrees from 184 W: the first straddles the
    // seam and reaches the mesh at both ends
    let gt = [-184.0, 8.0, 0.0, 50.0, 0.0, -5.0];
    let (nc, nr) = (45, 20);
    let mesh = round(36, 12);
    let t = transformer(mesh.clone(), gt);
    let w = exact_weights(&t, &t.source_window(), nc, nr, &WeightOptions::new(2).with_moments(true));
    let (mut df, mut dm, mut both) = (0.0f64, 0.0f64, [false; 2]);
    for k in 0..w.len() {
        let ring = dst_cell(gt, w.dst_row[k] as usize, w.dst_col[k] as usize);
        let (sr, sc) = (w.src_row[k] as usize, w.src_col[k] as usize);
        let mut want = [0.0; 3];
        for shift in [-360.0, 0.0, 360.0] {
            let moved: Vec<[f64; 2]> = ring.iter().map(|p| [p[0] + shift, p[1]]).collect();
            let m = overlap_in_index(&mesh, sr, sc, &moved);
            for i in 0..3 {
                want[i] += m[i];
            }
        }
        if w.dst_col[k] == 0 {
            both[(sc > 18) as usize] = true;
        }
        df = df.max((w.frac[k] - want[0]).abs());
        dm = dm.max((w.moment[k][0] - want[1]).abs()).max((w.moment[k][1] - want[2]).abs());
    }
    eprintln!("{} entries across the seam: fractions to {df:.1e}, moments to {dm:.1e} of the clipped overlaps", w.len());
    assert!(both[0] && both[1], "the first column reaches both ends");
    assert!(df < 1e-12 && dm < 1e-12, "{df:e} {dm:e}");
    // a field linear in the rows comes out exactly, the seam's columns too
    let src = t.source_window();
    let f = |v: f64| 4.0 + 0.25 * v;
    let data: Vec<f64> = (0..src.nrow * src.ncol).map(|p| f((p / src.ncol) as f64 + 0.5)).collect();
    let v: Vec<f64> = w.apply_second_order(&buffer(&src, &data), Normalization::FracArea, &ApplyOptions::new());
    let mut covered = HashMap::<(u32, u32), f64>::new();
    for k in 0..w.len() {
        *covered.entry((w.src_row[k], w.src_col[k])).or_default() += w.frac[k];
    }
    let mut cells = HashMap::<usize, Vec<usize>>::new();
    for k in 0..w.len() {
        cells.entry(w.dst_row[k] as usize * nc + w.dst_col[k] as usize).or_default().push(k);
    }
    let (mut e, mut n) = (0.0f64, 0);
    for (&cell, ks) in &cells {
        let rows_inside = ks.iter().all(|&k| w.src_row[k] > 0 && (w.src_row[k] as usize) + 1 < src.nrow);
        if !rows_inside || ks.iter().any(|&k| (covered[&(w.src_row[k], w.src_col[k])] - 1.0).abs() > 1e-12) {
            continue;
        }
        let ring = dst_cell(gt, cell / nc, cell % nc);
        let (mut a, mut s) = (0.0, 0.0);
        for &k in ks {
            let (sr, sc) = (w.src_row[k] as usize, w.src_col[k] as usize);
            for shift in [-360.0, 0.0, 360.0] {
                let moved: Vec<[f64; 2]> = ring.iter().map(|p| [p[0] + shift, p[1]]).collect();
                let m = overlap_in_index(&mesh, sr, sc, &moved);
                if m[0] > 0.0 {
                    let area = mesh.cell_area(sr, sc) * m[0];
                    a += area;
                    s += area * f(sr as f64 + m[2] / m[0]);
                }
            }
        }
        e = e.max((v[cell] - s / a).abs());
        n += 1;
    }
    eprintln!("{n} cells: a field linear in rows to {e:.1e}");
    assert!(n > 300 && e < 1e-12, "{e:e}");
}

#[test]
fn a_mesh_across_a_grids_seam_has_moments() {
    // faces from 170 E to 170 W onto a global grid that wraps: those over
    // the seam reach it at both ends
    let gt = [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0];
    let grid = GridWindow::new(180, 90);
    let inner = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([180, 90]);
    let mesh = triangles([170.0, -10.0], [190.0, 10.0], 10).with_period(360.0).unwrap();
    let w = mesh.weights_onto(&inner, &grid, &WeightOptions::new(2).with_moments(true)).unwrap();
    let mut sums = HashMap::<(u32, u32), [f64; 3]>::new();
    for k in 0..w.len() {
        let s = sums.entry((w.src_row[k], w.src_col[k])).or_default();
        s[0] += w.frac[k];
        s[1] += w.moment[k][0];
        s[2] += w.moment[k][1];
    }
    let (mut whole, mut ends, mut off) = (0, [0; 2], 0.0f64);
    for (&(_, c), s) in sums.iter().filter(|(_, s)| (s[0] - 1.0).abs() < 1e-12) {
        whole += 1;
        ends[(c > 90) as usize] += 1;
        off = off.max((s[1] / s[0] - 0.5).abs()).max((s[2] / s[0] - 0.5).abs());
    }
    eprintln!("across the grid's seam: {whole} cells covered wholly ({} west, {} east), centred to {off:.1e}", ends[0], ends[1]);
    assert!(whole == 100 && ends == [50, 50] && off < 1e-12, "{whole} {ends:?} {off:e}");
}

/// A table with no entries can't say whether its source has world cells,
/// so the fold takes the areas for it, and gives an empty table.
#[test]
fn an_empty_table_folds_with_areas() {
    let t = transformer(curved(20, 15), WORLD_GT);
    let src = t.source_window();
    // destination pixels in world coordinates, nowhere near the grid
    let w = exact_weights(&t, &src, 4, 4, &WeightOptions::new(1).with_moments(true));
    assert!(w.is_empty());
    let areas = source_cell_areas(&t, &src).unwrap();
    let folded = w.second_order(&src, None, Some(&areas));
    assert!(folded.is_empty());
}
