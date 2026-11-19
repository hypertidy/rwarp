//! Second order from a `PolyMesh`'s faces as sources (conservative stage
//! 6c): `PolyMesh::weights_to` with moments, `PolyMesh::apply_second_order`
//! and `PolyMesh::second_order`. The planes are in the grid's space, where
//! the overlaps are measured (decided 2026-11-19), so:
//!
//! - each entry's moments are the overlap's, the same both ways
//!   (`weights_onto`'s were checked against a clip without the walk in
//!   6b);
//! - a field linear in the grid's space comes out exactly where the faces
//!   and their neighbours are whole in the window: in pixels, on the
//!   sphere, and in a curvilinear grid's index space; a quadratic's error
//!   falls as the square of the size;
//! - the integral is conserved and the folded table is the same method;
//! - a face the grid's seam cuts is first order, and the faces either
//!   side of the seam keep one-sided gradients;
//! - threads keep the moments bit for bit.

use std::collections::HashMap;

use controlledburn::{CoverageWindow, Measure};
use rwarp::mesh::{CellMap, MeshTransformer, PolyMesh, QuadMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::{Nodata, SourceBuffer};
use rwarp::weights::{exact_weights, lonlat_measure, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// A rotation and scale from the mesh's coordinates straight into grid
/// pixels.
#[derive(Clone, Copy)]
struct Rotated {
    angle: f64,
    scale: f64,
    origin: [f64; 2],
}

impl Rotated {
    fn at(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let (u, v) = (p[0] * self.scale, p[1] * self.scale);
        [self.origin[0] + c * u - s * v, self.origin[1] + s * u + c * v]
    }

    fn back(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let (u, v) = (p[0] - self.origin[0], p[1] - self.origin[1]);
        [(c * u + s * v) / self.scale, (-s * u + c * v) / self.scale]
    }
}

impl Transformer for Rotated {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            [*x, *y] = self.at([*x, *y]);
        }
        vec![true; x.len()]
    }
}

/// A lattice of `nx` by `ny` boxes from `(x0, y0)`, `dx` by `dy` each, its
/// inner nodes moved a little, every third box cut into two triangles.
fn mixed(x0: f64, y0: f64, dx: f64, dy: f64, nx: usize, ny: usize) -> PolyMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..=nx {
            let inner = i > 0 && i < nx && k > 0 && k < ny;
            let j = if inner { 0.15 * (1.7 * i as f64 + 2.3 * k as f64).sin() } else { 0.0 };
            let l = if inner { 0.15 * (0.9 * i as f64 - 1.9 * k as f64).cos() } else { 0.0 };
            x.push(x0 + dx * (i as f64 + j));
            y.push(y0 + dy * (k as f64 + l));
        }
    }
    let n = |i: usize, k: usize| k * (nx + 1) + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        for i in 0..nx {
            let f = [n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)];
            if (i + 2 * k) % 3 == 0 {
                nodes.extend([f[0], f[1], f[2]]);
                start.push(nodes.len());
                nodes.extend([f[0], f[2], f[3]]);
            } else {
                nodes.extend(f);
            }
            start.push(nodes.len());
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap()
}

/// Face `f`'s nodes, through `map`.
fn ring(m: &PolyMesh, f: usize, map: impl Fn([f64; 2]) -> [f64; 2]) -> Vec<[f64; 2]> {
    m.face(f).iter().map(|&n| map(m.node(n))).collect()
}

/// Face `f`'s nodes, their longitudes unwrapped from the first's.
fn unwrapped(m: &PolyMesh, f: usize) -> Vec<[f64; 2]> {
    let r = ring(m, f, |p| p);
    let x0 = r[0][0];
    r.iter().map(|p| [p[0] - 360.0 * ((p[0] - x0) / 360.0).round(), p[1]]).collect()
}

/// A polygon's area and centroid (shoelace, about its first point).
fn area_centroid(p: &[[f64; 2]]) -> (f64, [f64; 2]) {
    let o = p[0];
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let (u, v) = (p[i], p[(i + 1) % p.len()]);
        let (u, v) = ([u[0] - o[0], u[1] - o[1]], [v[0] - o[0], v[1] - o[1]]);
        let cr = u[0] * v[1] - v[0] * u[1];
        a += cr;
        cx += (u[0] + v[0]) * cr;
        cy += (u[1] + v[1]) * cr;
    }
    (a / 2.0, [o[0] + cx / (3.0 * a), o[1] + cy / (3.0 * a)])
}

/// The mean of a quadratic `q` over a polygon, exactly: each triangle of a
/// fan from its first point by the mean at its sides' midpoints.
fn quad_mean(p: &[[f64; 2]], q: impl Fn([f64; 2]) -> f64) -> f64 {
    let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    let (mut a, mut s) = (0.0, 0.0);
    for i in 1..p.len() - 1 {
        let (u, v, w) = (p[0], p[i], p[i + 1]);
        let t = ((v[0] - u[0]) * (w[1] - u[1]) - (w[0] - u[0]) * (v[1] - u[1])) / 2.0;
        a += t;
        s += t * (q(mid(u, v)) + q(mid(v, w)) + q(mid(w, u))) / 3.0;
    }
    s / a
}

/// The centroid of a polygon (grid pixels, full grid) under `measure`,
/// from controlledburn's window moments.
fn centroid(measure: Measure, p: &[[f64; 2]]) -> [f64; 2] {
    let (xs, ys): (Vec<f64>, Vec<f64>) = p.iter().map(|q| (q[0], q[1])).unzip();
    let mut w = CoverageWindow::with_measure(measure);
    w.set_moments(true);
    assert!(w.reset_to_ring(&xs, &ys));
    w.add_ring(&xs, &ys, 1.0);
    w.resolve();
    let (mut a, mut mx, mut my) = (0.0, 0.0, 0.0);
    for k in 0..w.coverage().len() {
        let (i, j) = ((w.col0() + (k % w.ncol()) as i64) as f64, (w.row0() + (k / w.ncol()) as i64) as f64);
        let cell = measure.cell_area(j as i64);
        a += w.coverage()[k] * cell;
        mx += (i * w.coverage()[k] + w.moment_x()[k]) * cell;
        my += (j * w.coverage()[k] + w.moment_y()[k]) * cell;
    }
    [mx / a, my / a]
}

fn faces<'a>(data: &'a [f64]) -> SourceBuffer<'a, f64> {
    SourceBuffer { data, ncol: data.len(), nrow: 1, nbands: 1, col_off: 0, row_off: 0 }
}

/// The faces sharing a node with each face.
fn neighbours(m: &PolyMesh) -> Vec<Vec<usize>> {
    let mut at: HashMap<usize, Vec<usize>> = HashMap::new();
    for f in 0..m.nface() {
        for &n in m.face(f) {
            at.entry(n).or_default().push(f);
        }
    }
    (0..m.nface())
        .map(|f| {
            let mut v: Vec<usize> = m.face(f).iter().flat_map(|n| at[n].iter().copied()).filter(|&g| g != f).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect()
}

/// Whether each face has an edge no other face shares: on the mesh's
/// boundary, its neighbours all to one side.
fn on_boundary(m: &PolyMesh) -> Vec<bool> {
    let edge = |f: usize, i: usize| {
        let v = m.face(f);
        let (a, b) = (v[i], v[(i + 1) % v.len()]);
        (a.min(b), a.max(b))
    };
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for f in 0..m.nface() {
        for i in 0..m.face(f).len() {
            *count.entry(edge(f, i)).or_default() += 1;
        }
    }
    (0..m.nface()).map(|f| (0..m.face(f).len()).any(|i| count[&edge(f, i)] == 1)).collect()
}

/// The grid cells (row-major over the window) where a field linear in the
/// grid's space must come out exactly: covered wholly, by faces off the
/// mesh's boundary and whole in the window, as are all their neighbours.
fn exact_cells(w: &Weights, m: &PolyMesh) -> Vec<bool> {
    let mut whole = vec![0.0; m.nface()];
    for k in 0..w.len() {
        whole[w.src_col[k] as usize] += w.frac[k];
    }
    // to 1e-9: a grid with world cells has its overlaps' areas from the
    // walk and its faces' by shoelace, which round differently
    let whole: Vec<bool> = whole.iter().map(|s| (s - 1.0).abs() < 1e-9).collect();
    let (nbrs, edge) = (neighbours(m), on_boundary(m));
    let good: Vec<bool> = (0..m.nface()).map(|f| whole[f] && !edge[f] && nbrs[f].iter().all(|&g| whole[g])).collect();
    let covered = w.dst_covered();
    let mut ok: Vec<bool> = covered.iter().zip(&w.dst_area).map(|(c, a)| (c - a).abs() < 1e-9 * a).collect();
    for k in 0..w.len() {
        let cell = (w.dst_row[k] as usize - w.dst_off[1]) * w.dst_size[0] + (w.dst_col[k] as usize - w.dst_off[0]);
        if !good[w.src_col[k] as usize] {
            ok[cell] = false;
        }
    }
    ok
}

/// The largest difference from `want(row, col)` over the cells `ok`
/// marks, for second and first order; and how many there are.
fn worst(w: &Weights, ok: &[bool], second: &[f64], first: &[f64], want: impl Fn(usize, usize) -> f64) -> (f64, f64, usize) {
    let (mut a, mut b, mut n) = (0.0f64, 0.0f64, 0);
    for (cell, _) in ok.iter().enumerate().filter(|c| *c.1) {
        let (r, c) = (w.dst_off[1] + cell / w.dst_size[0], w.dst_off[0] + cell % w.dst_size[0]);
        let v = want(r, c);
        a = a.max((second[cell] - v).abs());
        b = b.max((first[cell] - v).abs());
        n += 1;
    }
    (a, b, n)
}

fn linear(p: [f64; 2]) -> f64 {
    3.0 + 0.7 * p[0] - 1.3 * p[1]
}

/// The planar case: the mesh over [0, 12] x [0, 10] turned into a grid
/// window it overhangs.
fn planar() -> (PolyMesh, Rotated, GridWindow) {
    (mixed(0.0, 0.0, 1.0, 1.0, 12, 10), Rotated { angle: 0.35, scale: 3.1, origin: [6.0, 2.0] }, GridWindow::new(40, 30).with_offset(3, 2))
}

#[test]
fn moments_are_the_same_both_ways() {
    let (mesh, t, grid) = planar();
    let opts = WeightOptions::new(2).with_moments(true);
    let to = mesh.weights_to(&t, &grid, &opts).unwrap();
    let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
    assert_eq!(to.moment.len(), to.len());
    assert_eq!(to.len(), onto.len());
    // each overlap's centroid about its grid cell's corner
    let at = |m: [f64; 2], f: f64| [m[0] / f, m[1] / f];
    let onto: HashMap<(u32, u32, u32), [f64; 2]> =
        (0..onto.len()).map(|k| ((onto.dst_col[k], onto.src_row[k], onto.src_col[k]), at(onto.moment[k], onto.frac[k]))).collect();
    let mut worst = 0.0f64;
    for k in 0..to.len() {
        let a = at(to.moment[k], to.frac[k]);
        let b = onto[&(to.src_col[k], to.dst_row[k], to.dst_col[k])];
        worst = worst.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs());
        // in the cell, slivers of rounding aside
        if to.frac[k] > 1e-12 {
            assert!((-1e-12..=1.0 + 1e-12).contains(&a[0]) && (-1e-12..=1.0 + 1e-12).contains(&a[1]), "{a:?}");
        }
    }
    assert!(worst < 1e-12, "{worst}");
    // without moments the table is as before
    let plain = mesh.weights_to(&t, &grid, &WeightOptions::new(2)).unwrap();
    assert!(plain.moment.is_empty());
    assert_eq!((&plain.frac, &plain.src_col, &plain.dst_col), (&to.frac, &to.src_col, &to.dst_col));
}

#[test]
fn linear_fields_come_out_exactly_in_pixels() {
    let (mesh, t, grid) = planar();
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1).with_moments(true)).unwrap();
    let data: Vec<f64> = (0..mesh.nface()).map(|f| linear(t.at(area_centroid(&ring(&mesh, f, |p| p)).1))).collect();
    let ok = exact_cells(&w, &mesh);
    for norm in [Normalization::FracArea, Normalization::DestArea] {
        let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), norm, &ApplyOptions::new());
        let first: Vec<f64> = w.apply(&faces(&data), norm, &ApplyOptions::new());
        let (a, b, n) = worst(&w, &ok, &second, &first, |r, c| linear([c as f64 + 0.5, r as f64 + 0.5]));
        println!("pixels {norm:?}: {n} cells, second order off by {a:.1e}, first {b:.2}");
        assert!(n > 300 && a < 1e-12 && b > 0.1, "{n} {a} {b}");
    }
}

#[test]
fn linear_fields_come_out_exactly_on_the_sphere() {
    // a mesh in longitude and latitude, its edges straight there, onto a
    // grid in them under the sphere's measure
    let gt = [100.0, 0.5, 0.0, -20.0, 0.0, -0.5];
    let measure = lonlat_measure(gt).unwrap();
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let mesh = mixed(101.0, -43.0, 1.5, 1.5, 12, 14);
    let grid = GridWindow::new(40, 48);
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1).with_measure(measure).with_moments(true)).unwrap();
    let px = |p: [f64; 2]| [(p[0] - gt[0]) / gt[1], (p[1] - gt[3]) / gt[5]];
    let data: Vec<f64> = (0..mesh.nface()).map(|f| linear(centroid(measure, &ring(&mesh, f, px)))).collect();
    let ok = exact_cells(&w, &mesh);
    let cell = |r: usize, c: usize| {
        let (x, y) = (c as f64, r as f64);
        centroid(measure, &[[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]])
    };
    let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let (a, b, n) = worst(&w, &ok, &second, &first, |r, c| linear(cell(r, c)));
    println!("sphere: {n} cells, second order off by {a:.1e}, first {b:.2}");
    assert!(n > 800 && a < 1e-12 && b > 0.1, "{n} {a} {b}");
}

/// The curved grid's corner at `(s, t)` in the unit square, as in
/// `second_order_world.rs`.
fn curve(s: f64, t: f64) -> [f64; 2] {
    [140.0 + 10.0 * s + 0.6 * (1.3 + 2.5 * t).sin() * s, -32.0 - 8.0 * t + 0.5 * (0.7 + 3.0 * s).sin() + 0.3 * s * t]
}

fn curved(ncol: usize, nrow: usize) -> QuadMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = curve(i as f64 / ncol as f64, j as f64 / nrow as f64);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap()
}

#[test]
fn linear_fields_come_out_exactly_in_a_curved_grid_s_indices() {
    // a curvilinear grid by its corners: the faces' planes are in its
    // index space under the walk's equal-area map
    let t = MeshTransformer::with_backend(curved(40, 30), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let grid = t.source_window();
    let mesh = mixed(141.0, -38.8, 0.6, 0.45, 13, 12);
    let opts = WeightOptions::new(3).with_moments(true);
    let w = mesh.weights_to(&t, &grid, &opts).unwrap();
    assert_eq!(w.moment.len(), w.len());
    // each face's mean of a field linear in index space is the field at
    // its centroid there, weighted by world area: from weights_onto's
    // overlaps, each cell a unit square holding its world area evenly
    let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
    let mut sum = vec![[0.0f64; 3]; mesh.nface()];
    for k in 0..onto.len() {
        let (s, a) = (&mut sum[onto.dst_col[k] as usize], onto.src_area(k));
        s[0] += onto.frac[k] * a;
        s[1] += (onto.frac[k] * onto.src_col[k] as f64 + onto.moment[k][0]) * a;
        s[2] += (onto.frac[k] * onto.src_row[k] as f64 + onto.moment[k][1]) * a;
    }
    let data: Vec<f64> = sum.iter().map(|s| linear([s[1] / s[0], s[2] / s[0]])).collect();
    let ok = exact_cells(&w, &mesh);
    let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let (a, b, n) = worst(&w, &ok, &second, &first, |r, c| linear([c as f64 + 0.5, r as f64 + 0.5]));
    println!("curved grid's indices: {n} cells, second order off by {a:.1e}, first {b:.2}");
    assert!(n > 300 && a < 1e-12 && b > 0.1, "{n} {a} {b}");
}

#[test]
fn a_quadratic_s_error_falls_as_the_square_of_the_size() {
    // the field in the mesh's coordinates; mesh and grid both halved
    let q = |p: [f64; 2]| {
        let (x, y) = (p[0] - 6.0, p[1] - 5.0);
        1.0 + 0.3 * x - 0.2 * y + 0.05 * x * x + 0.03 * x * y - 0.04 * y * y
    };
    let mut errs = Vec::new();
    for level in 0..3 {
        let h = 0.5f64.powi(level);
        let n = 1usize << level;
        let mesh = mixed(0.0, 0.0, h, h, 12 * n, 10 * n);
        let t = Rotated { angle: 0.35, scale: 3.1 / h, origin: [12.0 * n as f64, 0.0] };
        let grid = GridWindow::new(48 * n, 44 * n);
        let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1).with_moments(true)).unwrap();
        let data: Vec<f64> = (0..mesh.nface()).map(|f| quad_mean(&ring(&mesh, f, |p| p), q)).collect();
        let ok = exact_cells(&w, &mesh);
        let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
        let first: Vec<f64> = w.apply(&faces(&data), Normalization::FracArea, &ApplyOptions::new());
        let cell = |r: usize, c: usize| {
            let (x, y) = (c as f64, r as f64);
            let p: Vec<[f64; 2]> = [[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]].iter().map(|&p| t.back(p)).collect();
            quad_mean(&p, q)
        };
        let (a, b, cells) = worst(&w, &ok, &second, &first, cell);
        println!("size {h}: {cells} cells, second order off by {a:.2e}, first {b:.2e}");
        assert!(cells > 200);
        errs.push((a, b));
    }
    for k in 1..errs.len() {
        let (s, f) = (errs[k - 1].0 / errs[k].0, errs[k - 1].1 / errs[k].1);
        println!("halved: second order's error falls {s:.2} times, first order's {f:.2}");
        assert!(s > 3.0 && f < 2.6, "{s} {f}");
    }
}

#[test]
fn the_integral_is_conserved() {
    let (mesh, t, grid) = planar();
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(2).with_moments(true)).unwrap();
    let mut data: Vec<f64> =
        (0..mesh.nface()).map(|f| (t.at(area_centroid(&ring(&mesh, f, |p| p)).1)[0] / 4.0).sin() + 2.0).collect();
    for f in [3usize, 40, 41, 77] {
        data[f] = -999.0;
    }
    let opts = ApplyOptions::new().with_src_nodata(Nodata::Value(-999.0));
    let v: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::DestArea, &opts);
    let got: f64 = v.iter().zip(&w.dst_area).filter(|(x, a)| **x != -999.0 && a.is_finite()).map(|(x, a)| x * a).sum();
    let want: f64 = (0..w.len())
        .filter(|&k| data[w.src_col[k] as usize] != -999.0)
        .map(|k| data[w.src_col[k] as usize] * w.frac[k] * w.src_area(k))
        .sum();
    println!("conserved: {got} vs {want}, {:.1e}", (got - want).abs() / want);
    assert!((got - want).abs() < 1e-12 * want.abs(), "{got} vs {want}");
}

#[test]
fn the_folded_table_is_the_same_method() {
    let (mesh, t, grid) = planar();
    for measure in [Measure::Planar, Measure::cos_latitude((-50.0f64).to_radians(), (-0.25f64).to_radians())] {
        let w = mesh.weights_to(&t, &grid, &WeightOptions::new(2).with_measure(measure).with_moments(true)).unwrap();
        let mut data: Vec<f64> =
            (0..mesh.nface()).map(|f| (t.at(area_centroid(&ring(&mesh, f, |p| p)).1)[1] / 3.0).cos() * 2.0).collect();
        for mask in [false, true] {
            let mut valid = vec![true; mesh.nface()];
            if mask {
                for f in [3usize, 40, 41, 77] {
                    valid[f] = false;
                    data[f] = -999.0;
                }
            }
            let folded = mesh.second_order(&w, mask.then_some(&valid[..]));
            assert!(folded.moment.is_empty() && folded.frac.iter().any(|&f| f < 0.0));
            assert_eq!(folded.src_cell_area.len(), folded.len());
            assert!(folded.src_row.iter().all(|&r| r == 0));
            let opts = ApplyOptions::new().with_src_nodata(Nodata::Value(-999.0));
            for norm in [Normalization::DestArea, Normalization::FracArea] {
                let a: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), norm, &opts);
                let b: Vec<f64> = folded.apply(&faces(&data), norm, &opts);
                let worst = a.iter().zip(&b).map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() }).fold(0.0, f64::max);
                assert!(worst < 1e-12, "{measure:?} mask {mask} {norm:?}: {worst}");
            }
        }
    }
}

/// A global grid of 10 degree cells from -180, its longitudes wrapping.
fn global() -> (GenImgProjTransformer, GridWindow) {
    let gt = [-180.0, 10.0, 0.0, 90.0, 0.0, -10.0];
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([36, 18]);
    assert!(t.source_wrap().is_some());
    (t, GridWindow::new(36, 18))
}

/// Boxes of 30 degrees from longitude `lon0` round the globe, latitudes
/// -60 to 60: the last of each row reaches the first's west side.
fn belt(lon0: f64) -> PolyMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..5 {
        for i in 0..12 {
            let lon = lon0 + 30.0 * i as f64;
            x.push(if lon >= 180.0 { lon - 360.0 } else { lon });
            y.push(-60.0 + 30.0 * k as f64);
        }
    }
    let n = |i: usize, k: usize| k * 12 + i % 12;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..4 {
        for i in 0..12 {
            nodes.extend([n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)]);
            start.push(nodes.len());
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap()
}

/// The pixel the grid puts longitude and latitude at, unwrapped from -180.
fn global_px(p: [f64; 2]) -> [f64; 2] {
    [(p[0] + 180.0) / 10.0, (90.0 - p[1]) / 10.0]
}

#[test]
fn faces_either_side_of_the_seam_keep_one_sided_gradients() {
    // nodes on the seam: the faces either side of it share them, but are
    // at the grid's two ends, so neither is the other's neighbour there
    let (t, grid) = global();
    let mesh = belt(-180.0);
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1).with_moments(true)).unwrap();
    // a field linear across the seam: in the grid's columns with its east
    // half taken a period west, so a face's neighbour over the seam has the
    // value of a face beside it, not of one at the far end
    let across = |p: [f64; 2]| linear([if p[0] >= 18.0 { p[0] - 36.0 } else { p[0] }, p[1]]);
    let data: Vec<f64> = (0..mesh.nface())
        .map(|f| {
            // each face's centre in the grid's columns, on its side of the seam
            let c = area_centroid(&unwrapped(&mesh, f)).1;
            let lon = if c[0] < -180.0 { c[0] + 360.0 } else { c[0] };
            across(global_px([lon, c[1]]))
        })
        .collect();
    let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&faces(&data), Normalization::FracArea, &ApplyOptions::new());
    // the middle rows of faces (latitudes -30 to 30, rows 6 to 11) have
    // neighbours above and below, so the cells there within two faces of
    // the seam are exact
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for r in 6..12 {
        for c in (0..6).chain(30..36) {
            let v = across([c as f64 + 0.5, r as f64 + 0.5]);
            a = a.max((second[r * 36 + c] - v).abs());
            b = b.max((first[r * 36 + c] - v).abs());
        }
    }
    println!("either side of the seam: second order off by {a:.1e}, first {b:.2}");
    assert!(a < 1e-12 && b > 0.1, "{a} {b}");
}

#[test]
fn a_face_the_seam_cuts_is_first_order() {
    // the belt from -165: each row's last face is 165 to 195, cut by the
    // seam, its cells at both ends of the grid
    let (t, grid) = global();
    let mesh = belt(-165.0);
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1).with_moments(true)).unwrap();
    let cut = |f: usize| f % 12 == 11;
    let data: Vec<f64> = (0..mesh.nface())
        .map(|f| if cut(f) { 5.0 + f as f64 } else { linear(global_px(area_centroid(&unwrapped(&mesh, f)).1)) })
        .collect();
    let second: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&faces(&data), Normalization::FracArea, &ApplyOptions::new());
    for r in 6..12 {
        // a cell wholly in a cut face is its value: no gradient
        for c in [35, 0] {
            let k = r * 36 + c;
            assert!((second[k] - first[k]).abs() < 1e-13, "row {r} col {c}: {} vs {}", second[k], first[k]);
        }
        // the faces beside it keep a one-sided gradient: exact for a
        // linear field
        for c in [2, 3, 32, 33] {
            let v = linear([c as f64 + 0.5, r as f64 + 0.5]);
            assert!((second[r * 36 + c] - v).abs() < 1e-12, "row {r} col {c}: {} vs {v}", second[r * 36 + c]);
            assert!((first[r * 36 + c] - v).abs() > 0.1);
        }
    }
}

#[test]
fn threads_keep_the_moments() {
    let (mesh, t, grid) = planar();
    let opts = WeightOptions::new(2).with_moments(true);
    let one = mesh.weights_to(&t, &grid, &opts).unwrap();
    let par = mesh.weights_to_par(|| Ok(t), &grid, &opts, 3).unwrap();
    assert_eq!(one, par);
    let (t, grid) = global();
    let mesh = belt(-165.0);
    let one = mesh.weights_to(&t, &grid, &opts).unwrap();
    let par = mesh.weights_to_par(|| Ok(global().0), &grid, &opts, 3).unwrap();
    assert_eq!(one, par);
}

#[test]
#[should_panic(expected = "second order needs each entry's moments")]
fn without_moments_it_panics() {
    let (mesh, t, grid) = planar();
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(1)).unwrap();
    let data = vec![1.0; mesh.nface()];
    let _: Vec<f64> = mesh.apply_second_order(&w, &faces(&data), Normalization::FracArea, &ApplyOptions::new());
}

#[test]
#[should_panic(expected = "not a table from")]
fn a_grid_s_table_is_refused() {
    let (mesh, t, _) = planar();
    let w = exact_weights(&t, &GridWindow::new(8, 8), 4, 4, &WeightOptions::new(1).with_moments(true));
    assert!(!w.is_empty());
    let _ = mesh.second_order(&w, None);
}


