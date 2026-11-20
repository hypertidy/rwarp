//! Second order on the sphere against ESMF's spherical `conserve_2nd`
//! (`tests/fixtures/esmf/<case>.sph2.txt`, made by `gen.py`), conservative
//! stage 6d.
//!
//! The two are different second-order methods. ESMF's works in three
//! dimensions, its cells bounded by great circles. rwarp's planes are in
//! the source's longitude and latitude (6a, 6b), or for a mesh's faces as
//! sources in the grid's (6c), under the spherical measure, its grid cells
//! bounded by parallels. So the weights differ, by more than first order's
//! (which differ by the cells alone, `tests/esmf.rs`, `tests/esmf_mesh.rs`),
//! and the comparison is of accuracy: each method's remap of a smooth
//! field's exact cell means, in its own cells, against the exact means of
//! its destination cells. The means are integrals by quadrature: over a
//! cell straight in longitude and latitude under the measure for rwarp's,
//! and over a cell with great-circle edges for ESMF's, in a gnomonic plane
//! where those edges are straight.
//!
//! `cargo test --test esmf_second_order_sphere -- --nocapture` prints the
//! errors.
#![cfg(feature = "proj")]

mod esmf_common;

use std::sync::OnceLock;

use esmf_common::{diff, fixture_path, read_fixture, regrid, smooth, Fixture, Ours};
use rwarp::mesh::{PolyMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::SourceBuffer;
use rwarp::weights::{exact_weights, lonlat_measure, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

// ---- exact cell means -------------------------------------------------------------

/// Gauss-Legendre points and weights on [0, 1].
fn gauss(n: usize) -> Vec<(f64, f64)> {
    let legendre = |x: f64| {
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        (p1, n as f64 * (x * p1 - p0) / (x * x - 1.0))
    };
    (1..=n)
        .map(|i| {
            let mut x = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
            for _ in 0..50 {
                let (p, dp) = legendre(x);
                x -= p / dp;
            }
            let (_, dp) = legendre(x);
            ((1.0 + x) / 2.0, 1.0 / ((1.0 - x * x) * dp * dp))
        })
        .collect()
}

/// The integrals of `f` and of 1 over a plane polygon (convex) under the
/// density `rho`: a fan of triangles, each by 10 x 10 Gauss points
/// collapsed onto it.
fn polygon_integrals(p: &[[f64; 2]], rho: impl Fn([f64; 2]) -> f64, f: impl Fn([f64; 2]) -> f64) -> (f64, f64) {
    let g = gauss(10);
    let (mut a, mut s) = (0.0, 0.0);
    for i in 1..p.len() - 1 {
        let (o, b, c) = (p[0], p[i], p[i + 1]);
        let det = (b[0] - o[0]) * (c[1] - b[1]) - (b[1] - o[1]) * (c[0] - b[0]);
        for &(u, wu) in &g {
            for &(v, wv) in &g {
                let q = [o[0] + u * (b[0] - o[0]) + u * v * (c[0] - b[0]), o[1] + u * (b[1] - o[1]) + u * v * (c[1] - b[1])];
                let w = wu * wv * u * det * rho(q);
                a += w;
                s += w * f(q);
            }
        }
    }
    (s, a)
}

/// The smooth field's mean over a cell straight in longitude and latitude
/// (degrees), under the spherical measure: rwarp's cells.
fn straight_mean(p: &[[f64; 2]]) -> f64 {
    let (s, a) = polygon_integrals(p, |q| q[1].to_radians().cos(), |q| smooth(q[0], q[1]));
    s / a
}

fn unit(p: [f64; 2]) -> [f64; 3] {
    let (lon, lat) = (p[0].to_radians(), p[1].to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn normal(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The integrals of `f` (on unit vectors) and of 1 over a cell with
/// great-circle edges through its corners (degrees), on the unit sphere.
/// In the gnomonic plane about the corners' mean the edges are straight,
/// and the sphere's area is `(1 + x^2 + y^2)^(-3/2)` of the plane's.
fn great_circle_integrals(p: &[[f64; 2]], f: impl Fn([f64; 3]) -> f64) -> (f64, f64) {
    let v: Vec<[f64; 3]> = p.iter().map(|&q| unit(q)).collect();
    let c = normal(v.iter().fold([0.0; 3], |s, u| [s[0] + u[0], s[1] + u[1], s[2] + u[2]]));
    let axis = if c[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let e1 = normal(cross(axis, c));
    let e2 = cross(c, e1);
    let plane: Vec<[f64; 2]> = v.iter().map(|&u| [dot(u, e1) / dot(u, c), dot(u, e2) / dot(u, c)]).collect();
    polygon_integrals(
        &plane,
        |q| (1.0 + q[0] * q[0] + q[1] * q[1]).powf(-1.5),
        |q| f(normal([c[0] + q[0] * e1[0] + q[1] * e2[0], c[1] + q[0] * e1[1] + q[1] * e2[1], c[2] + q[0] * e1[2] + q[1] * e2[2]])),
    )
}

/// The smooth field's mean over a cell with great-circle edges: ESMF's
/// cells, and both sides' mesh faces.
fn great_circle_mean(p: &[[f64; 2]]) -> f64 {
    let (s, a) = great_circle_integrals(p, |u| smooth(u[1].atan2(u[0]).to_degrees(), u[2].asin().to_degrees()));
    s / a
}

/// A regular grid's cells' corners in longitude and latitude, row-major.
fn grid_cells(gt: [f64; 6], ncol: usize, nrow: usize) -> Vec<Vec<[f64; 2]>> {
    let at = |c: usize, r: usize| [gt[0] + c as f64 * gt[1], gt[3] + r as f64 * gt[5]];
    (0..nrow * ncol).map(|k| {
        let (r, c) = (k / ncol, k % ncol);
        vec![at(c, r), at(c + 1, r), at(c + 1, r + 1), at(c, r + 1)]
    }).collect()
}

/// The quadrature against closed forms: the smooth field over a box in
/// longitude and latitude, and a triangle's area (its spherical excess)
/// and the integral of `z` over it (half the sum of its edges' angles
/// times their unit normals).
#[test]
fn the_cell_means_are_exact() {
    // the box: integrals in degrees of longitude and latitude
    let (l0, l1, p0, p1) = (101.3f64, 103.8f64, -13.7f64, -11.7f64);
    let (s, a) = polygon_integrals(&[[l0, p0], [l1, p0], [l1, p1], [l0, p1]], |q| q[1].to_radians().cos(), |q| smooth(q[0], q[1]));
    let (r0, r1, f0, f1) = (l0.to_radians(), l1.to_radians(), p0.to_radians(), p1.to_radians());
    let deg2 = (180.0 / std::f64::consts::PI).powi(2);
    let area = (r1 - r0) * (f1.sin() - f0.sin());
    let cos3 = |f: f64| f.sin() - f.sin().powi(3) / 3.0;
    let sin3cos = |f: f64| -(4.0 * f).cos() / 8.0 - (2.0 * f).cos() / 4.0;
    let want = 2.0 * area + ((2.0 * r1).sin() - (2.0 * r0).sin()) / 2.0 * (cos3(f1) - cos3(f0)) + 0.5 * (r1 - r0) * (sin3cos(f1) - sin3cos(f0));
    assert!((a / deg2 - area).abs() < 1e-14 * area && (s / deg2 - want).abs() < 1e-13 * want, "{} {area} {} {want}", a / deg2, s / deg2);
    // a triangle of great circles, anticlockwise from outside
    let p = [[10.0, 20.0], [40.0, 25.0], [22.0, 50.0]];
    let v: Vec<[f64; 3]> = p.iter().map(|&q| unit(q)).collect();
    let (a3, b3, c3) = (v[0], v[1], v[2]);
    let excess = 2.0 * (dot(a3, cross(b3, c3)).abs()).atan2(1.0 + dot(a3, b3) + dot(b3, c3) + dot(c3, a3));
    let mut z = 0.0;
    for i in 0..3 {
        let (u, w) = (v[i], v[(i + 1) % 3]);
        z += 0.5 * dot(u, w).acos() * normal(cross(u, w))[2];
    }
    let (sz, area) = great_circle_integrals(&p, |u| u[2]);
    assert!((area - excess).abs() < 1e-14 && (sz - z).abs() < 1e-14, "{area} {excess} {sz} {z}");
}

// ---- the cases ------------------------------------------------------------------

/// One case's two sides: rwarp's table with moments and source values,
/// ESMF's source values, and the exact destination means, each in its own
/// cells; and how rwarp applies its table.
struct Case {
    name: String,
    w: Weights,
    esmf_src: Vec<f64>,
    ours_want: Vec<f64>,
    esmf_want: Vec<f64>,
    /// rwarp's first and second order, by destination cell.
    ours: [Vec<f64>; 2],
    /// rwarp's folded second-order table in ESMF's terms.
    folded: Ours,
    /// The grids' widths for ESMF's flattened indices.
    dst_ncol: usize,
    src_ncol: usize,
    /// The mesh, for a mesh's case.
    faces: Option<PolyMesh>,
}

fn in_esmf_terms(w: &Weights, dst_ncol: usize, src_ncol: usize) -> Ours {
    let s = w.normalised(Normalization::DestArea);
    let (row, col) = w.esmf_row_col(dst_ncol, src_ncol);
    Ours { s: row.into_iter().zip(col).zip(s).collect(), frac_b: w.dst_covered().iter().zip(&w.dst_area).map(|(c, a)| c / a).collect(), area_b: w.dst_area.clone() }
}

/// A grid onto a grid (`tests/esmf.rs`'s cases), at densify 1 under the
/// spherical measure: rwarp's destination cells are their corners joined
/// straight in the source's longitude and latitude, ESMF's by great
/// circles.
fn grid_case(name: &str) -> Case {
    let fx = read_fixture(&format!("{name}.sph2.txt"));
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let [dw, dh] = fx.nums::<2>("dst_size").map(|v| v as usize);
    let src_gt: [f64; 6] = fx.nums("src_gt");
    let t = GenImgProjTransformer::new(fx.header("src_crs"), src_gt, fx.header("dst_crs"), fx.nums("dst_gt")).unwrap();
    let opts = WeightOptions::new(1).with_measure(lonlat_measure(src_gt).unwrap()).with_moments(true);
    let src = GridWindow::new(sw, sh);
    let w = exact_weights(&t, &src, dw, dh, &opts);
    let cells = grid_cells(src_gt, sw, sh);
    let ours_src: Vec<f64> = cells.iter().map(|c| straight_mean(c)).collect();
    let esmf_src: Vec<f64> = cells.iter().map(|c| great_circle_mean(c)).collect();
    // each destination cell's corners in the source's longitude and latitude
    let (mut ours_want, mut esmf_want) = (Vec::new(), Vec::new());
    for r in 0..dh {
        for c in 0..dw {
            let (c, r) = (c as f64, r as f64);
            let (mut x, mut y) = (vec![c, c + 1.0, c + 1.0, c], vec![r, r, r + 1.0, r + 1.0]);
            t.transform(true, &mut x, &mut y);
            let p: Vec<[f64; 2]> = x.iter().zip(&y).map(|(&x, &y)| [src_gt[0] + x * src_gt[1], src_gt[3] + y * src_gt[5]]).collect();
            ours_want.push(straight_mean(&p));
            esmf_want.push(great_circle_mean(&p));
        }
    }
    let buf = SourceBuffer { data: &ours_src[..], ncol: sw, nrow: sh, nbands: 1, col_off: 0, row_off: 0 };
    let a = ApplyOptions::new();
    let ours = [w.apply(&buf, Normalization::FracArea, &a), w.apply_second_order(&buf, Normalization::FracArea, &a)];
    let folded = in_esmf_terms(&w.second_order(&src, None, None), dw, sw);
    Case { name: name.into(), w, esmf_src, ours_want, esmf_want, ours, folded, dst_ncol: dw, src_ncol: sw, faces: None }
}

/// A fixture's mesh, as `tests/esmf_mesh.rs` reads it.
fn read_mesh(name: &str) -> PolyMesh {
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut head = lines.next().unwrap().split_whitespace().map(|v| v.parse::<usize>().unwrap());
    let (nn, nf) = (head.next().unwrap(), head.next().unwrap());
    let (mut x, mut y) = (Vec::with_capacity(nn), Vec::with_capacity(nn));
    for l in lines.by_ref().take(nn) {
        let mut v = l.split_whitespace().map(|v| v.parse::<f64>().unwrap());
        x.push(v.next().unwrap());
        y.push(v.next().unwrap());
    }
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for l in lines.take(nf) {
        let v: Vec<usize> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
        nodes.extend(&v[1..]);
        start.push(nodes.len());
    }
    PolyMesh::new(x, y, start, nodes).unwrap()
}

/// A mesh on the sphere and the global 5 degree grid, `kind` "to" (the
/// faces the sources, 6c) or "onto" (the grid's cells, 6b), as
/// `tests/esmf_mesh.rs` sets them up. The faces have great-circle edges
/// on both sides; the grid's cells are bounded by parallels in rwarp and
/// by great circles in ESMF.
fn mesh_case(name: &str, kind: &str) -> Case {
    let fx: Fixture = read_fixture(&format!("{name}_{kind}.sph2.txt"));
    let m = read_mesh(&format!("{name}.mesh.txt")).with_great_circles().unwrap();
    let gt: [f64; 6] = fx.nums("grid_gt");
    let [gw, gh] = fx.nums::<2>("grid_size").map(|v| v as usize);
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([gw, gh]);
    let opts = WeightOptions::new(1).with_measure(lonlat_measure(gt).unwrap()).with_refine_tolerance(1e-7).with_moments(true);
    let g = GridWindow::new(gw, gh);
    let faces: Vec<f64> = (0..m.nface()).map(|f| great_circle_mean(&m.face(f).iter().map(|&n| m.node(n)).collect::<Vec<_>>())).collect();
    let cells = grid_cells(gt, gw, gh);
    let (ours_grid, esmf_grid): (Vec<f64>, Vec<f64>) = cells.iter().map(|c| (straight_mean(c), great_circle_mean(c))).unzip();
    let a = ApplyOptions::new();
    if kind == "to" {
        let w = m.weights_to(&t, &g, &opts).unwrap();
        let buf = SourceBuffer { data: &faces[..], ncol: faces.len(), nrow: 1, nbands: 1, col_off: 0, row_off: 0 };
        let ours = [w.apply(&buf, Normalization::FracArea, &a), m.apply_second_order(&w, &buf, Normalization::FracArea, &a)];
        let folded = in_esmf_terms(&m.second_order(&w, None), gw, m.nface());
        Case { name: format!("{name}_{kind}"), w, esmf_src: faces.clone(), ours_want: ours_grid, esmf_want: esmf_grid, ours, folded, dst_ncol: gw, src_ncol: m.nface(), faces: Some(m) }
    } else {
        let w = m.weights_onto(&t, &g, &opts).unwrap();
        let buf = SourceBuffer { data: &ours_grid[..], ncol: gw, nrow: gh, nbands: 1, col_off: 0, row_off: 0 };
        let ours = [w.apply(&buf, Normalization::FracArea, &a), w.apply_second_order(&buf, Normalization::FracArea, &a)];
        let folded = in_esmf_terms(&w.second_order(&g, None, None), m.nface(), gw);
        Case { name: format!("{name}_{kind}"), w, esmf_src: esmf_grid, ours_want: faces.clone(), esmf_want: faces, ours, folded, dst_ncol: m.nface(), src_ncol: gw, faces: Some(m) }
    }
}

/// Where a destination cell is, for picking cells: a grid cell's middle,
/// or a face's nodes' mean on the sphere, in degrees.
fn place(c: &Case, k: usize) -> [f64; 2] {
    match &c.faces {
        Some(m) if c.name.ends_with("_onto") => {
            let u = normal(m.face(k).iter().map(|&n| unit(m.node(n))).fold([0.0; 3], |s, w| [s[0] + w[0], s[1] + w[1], s[2] + w[2]]));
            [u[1].atan2(u[0]).to_degrees(), u[2].asin().to_degrees()]
        }
        _ => [-180.0 + 5.0 * (k % c.dst_ncol) as f64 + 2.5, 90.0 - 5.0 * (k / c.dst_ncol) as f64 - 2.5],
    }
}

/// The rms errors of rwarp's first and second order and ESMF's, over the
/// destination cells both cover wholly that `pick` takes, and how many.
fn rms(c: &Case, pick: impl Fn(usize) -> bool) -> ([f64; 4], usize) {
    let (first, second) = (read_fixture(&format!("{}.sph.txt", c.name)), read_fixture(&format!("{}.sph2.txt", c.name)));
    let n_b = first.frac_b.len();
    let esmf = [regrid(&first.s, &c.esmf_src, n_b), regrid(&second.s, &c.esmf_src, n_b)];
    let covered = c.w.dst_covered();
    let (mut e, mut n) = ([0.0f64; 4], 0);
    for k in 0..n_b {
        let whole = first.frac_b[k] > 1.0 - 1e-9 && second.frac_b[k] > 1.0 - 1e-9 && (covered[k] / c.w.dst_area[k] - 1.0).abs() < 1e-9;
        if !whole || !pick(k) {
            continue;
        }
        let d = [c.ours[0][k] - c.ours_want[k], c.ours[1][k] - c.ours_want[k], esmf[0][k] - c.esmf_want[k], esmf[1][k] - c.esmf_want[k]];
        for (e, d) in e.iter_mut().zip(d) {
            *e += d * d;
        }
        n += 1;
    }
    (e.map(|e| (e / n as f64).sqrt()), n)
}

/// The destination cells whose sources are wholly covered and have all
/// their neighbours: elsewhere both methods centre a plane on the covered
/// part of a cell (ESMF's way, `second_order.rs`), first order there.
fn interior(c: &Case) -> Vec<bool> {
    let fx = read_fixture(&format!("{}.sph2.txt", c.name));
    let [sw, sh] = fx.nums::<2>("src_size").map(|v| v as usize);
    let mut covered = std::collections::HashMap::<(u32, u32), f64>::new();
    for k in 0..c.w.len() {
        *covered.entry((c.w.src_row[k], c.w.src_col[k])).or_default() += c.w.frac[k];
    }
    let mut ok = vec![true; fx.frac_b.len()];
    for k in 0..c.w.len() {
        let (r, col) = (c.w.src_row[k] as usize, c.w.src_col[k] as usize);
        let whole = (covered[&(c.w.src_row[k], c.w.src_col[k])] - 1.0).abs() < 1e-9;
        if !whole || r == 0 || col == 0 || r + 1 == sh || col + 1 == sw {
            ok[c.w.dst_row[k] as usize * c.dst_ncol + c.w.dst_col[k] as usize] = false;
        }
    }
    ok
}

fn report(c: &Case, what: &str, (e, n): ([f64; 4], usize)) -> [f64; 4] {
    eprintln!(
        "{:18} {what:28} {n:5} cells: rms rwarp first {:.2e}, second {:.2e}; ESMF first {:.2e}, second {:.2e}",
        c.name, e[0], e[1], e[2], e[3]
    );
    e
}

static CASES: OnceLock<Vec<Case>> = OnceLock::new();

fn cases() -> &'static [Case] {
    CASES.get_or_init(|| {
        let mut v: Vec<Case> = ["regular_coarser", "regular_overhang", "lonlat_3031"].iter().map(|n| grid_case(n)).collect();
        for name in ["mesh_ico", "mesh_dual"] {
            for kind in ["onto", "to"] {
                v.push(mesh_case(name, kind));
            }
        }
        v
    })
}

fn case(name: &str) -> &'static Case {
    cases().iter().find(|c| c.name == name).unwrap()
}

/// A grid onto a grid (6a), under the spherical measure. Inside, where
/// each source cell is whole and has its neighbours, both are second
/// order: rwarp 100 to 1,300 times closer than first order, ESMF 15 to
/// 800. For lon/lat onto lon/lat they are close (`regular_coarser`, ESMF
/// 7% the closer; `regular_overhang`, rwarp 1.6 times); projected
/// (`lonlat_3031`) rwarp is 19 times closer, as in Cartesian
/// (`tests/esmf_second_order.rs`): ESMF centres an overlap on its
/// vertices' mean. Over every cell the ring of destination cells on cut
/// source cells dominates both, and there neither is better than first
/// order (each centres a plane on the covered part of a cell): the two
/// agree there. Ceilings for rwarp's second order inside, 1.5 times
/// measured.
#[test]
fn a_grid_onto_a_grid() {
    for (name, ceiling) in [("regular_coarser", 4.2e-6), ("regular_overhang", 3.0e-7), ("lonlat_3031", 7.0e-6)] {
        let c = case(name);
        report(c, "every cell", rms(c, |_| true));
        let ok = interior(c);
        let e = report(c, "inside", rms(c, |k| ok[k]));
        assert!(e[1] < 0.02 * e[0] && e[3] < 0.1 * e[2], "{name}: {e:?}");
        assert!(e[1] < 1.1 * e[3] && e[1] < ceiling, "{name}: {e:?}");
    }
}

/// A grid onto a mesh's faces (6b): the overlaps are general polygons, so
/// ESMF's vertex means put its second order 5 to 12 times further off
/// than rwarp's, everywhere, at the grid's seam (where rwarp's gradients
/// are one-sided) and near the poles too. Ceilings 1.5 times measured.
#[test]
fn a_grid_onto_a_mesh() {
    for (name, ceiling) in [("mesh_ico_onto", 2.0e-4), ("mesh_dual_onto", 1.5e-4)] {
        let c = case(name);
        let e = report(c, "every face", rms(c, |_| true));
        assert!(e[1] < 0.05 * e[0] && e[3] < 0.5 * e[2], "{name}: {e:?}");
        assert!(e[1] < 0.25 * e[3] && e[1] < ceiling, "{name}: {e:?}");
    }
}

/// A mesh's faces onto a grid (6c), faces of about 15 degrees onto cells
/// of 5, so first order is far off (0.07 to 0.09 rms). Away from the
/// grid's seam and the poles rwarp's second order is 1.8 to 4.4 times
/// closer than ESMF's. Near the seam it is close to first order: a face
/// the seam cuts is first order (the table doesn't carry the grid's
/// period), and its neighbours' gradients one-sided, so within 15 degrees
/// of it ESMF, which works on the sphere with no seam, is 1.3 to 3.6 times
/// closer. Ceilings 1.5 times measured, there and away from it.
#[test]
fn a_mesh_onto_a_grid() {
    for (name, away, near) in [("mesh_ico_to", 1.5e-2, 8.0e-2), ("mesh_dual_to", 2.4e-2, 1.2e-1)] {
        let c = case(name);
        report(c, "every cell", rms(c, |_| true));
        let a = report(c, "45+ from the seam, |lat| < 60", rms(c, |k| {
            let p = place(c, k);
            180.0 - p[0].abs() >= 45.0 && p[1].abs() < 60.0
        }));
        let n = report(c, "within 15 of the seam", rms(c, |k| 180.0 - place(c, k)[0].abs() < 15.0));
        assert!(a[1] < 0.25 * a[0] && a[3] < 0.8 * a[2], "{name}: {a:?}");
        assert!(a[1] < a[3] && a[1] < away, "{name} away from the seam: {a:?}");
        assert!(n[1] < n[0] && n[1] < near, "{name} near the seam: {n:?}");
    }
}

/// The weights' differences, recorded as ceilings (1.5 times measured):
/// larger than first order's (which differ by the cells alone), as two
/// methods' would be. For a grid onto a grid they are first order's but
/// next to the source's edge, where the gradients are one-sided
/// (`regular_overhang`); elsewhere, ESMF's vertex means, and from a mesh
/// the faces the grid's seam cuts.
#[test]
fn spherical_second_order_differences_are_recorded() {
    #[rustfmt::skip]
    let ceilings = [
        ("regular_coarser",  [1.7e-3, 1.5e-4]),
        ("regular_overhang", [1.1e-1, 3.2e-3]),
        ("lonlat_3031",      [3.0e-2, 3.5e-4]),
        ("mesh_ico_onto",    [1.7e-2, 3.8e-3]),
        ("mesh_ico_to",      [3.5e-1, 1.5e-1]),
        ("mesh_dual_onto",   [5.4e-3, 1.8e-3]),
        ("mesh_dual_to",     [3.5e-1, 2.1e-1]),
    ];
    for (name, [cs, cf]) in ceilings {
        let c = case(name);
        let fx = read_fixture(&format!("{name}.sph2.txt"));
        let d = diff(&fx, &c.folded, &c.esmf_src);
        let d1 = diff(&read_fixture(&format!("{name}.sph.txt")), &in_esmf_terms(&c.w, c.dst_ncol, c.src_ncol), &c.esmf_src);
        eprintln!("{name:18} S differs by {:.1e} (first order {:.1e}), a smooth field by {:.1e} (first order {:.1e})", d.s, d1.s, d.field, d1.field);
        // frac_b as first order's: the cells differ where the grid's corner cuts them
        assert!(d.esmf_rows < 1e-9 && d.frac_b < 2e-6, "{name}: {d:?}");
        assert!(d.s <= cs && d.field <= cf, "{name}: {d:?}");
    }
}
