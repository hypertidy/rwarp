//! The grid's period in a table (`Weights::period`): with moments, a table
//! on a grid that wraps round the globe every whole number of its columns
//! records that number, and second order takes neighbours across its
//! seam. So:
//!
//! - it is recorded for a grid that wraps, as a source or onto a mesh or
//!   from one, on one thread or several, and not without moments, for a
//!   grid that doesn't wrap, or for a period that isn't whole;
//! - second order doesn't see the seam: a mesh onto a global grid, and the
//!   grid onto the mesh, come out the same with the seam on the far side
//!   of the globe, with and without the limiter, and folded. Without the
//!   period they don't;
//! - a face whose cells take every column (round a pole) has no centre
//!   in the grid's columns, so is first order and no one's neighbour.

mod esmf_common;

use esmf_common::{fixture_path, smooth};
use rwarp::mesh::{PolyMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::SourceBuffer;
use rwarp::weights::{
    exact_weights, exact_weights_par, lonlat_measure, ApplyOptions, GridWindow, Limiter, Normalization, WeightOptions, Weights,
};
use rwarp::{CrsTransform, Transformer};

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// A global grid of 5 degree cells from longitude `lon0`, wrapping: 72
/// columns, its seam at `lon0`.
fn grid(lon0: f64) -> (GenImgProjTransformer, GridWindow, [f64; 6]) {
    let gt = [lon0, 5.0, 0.0, 90.0, 0.0, -5.0];
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(lon0 + 180.0))
        .with_source_size([72, 36]);
    (t, GridWindow::new(72, 36), gt)
}

/// The column of grid `grid(0.0)` that is column `c` of `grid(-180.0)`.
fn turned(c: usize) -> usize {
    (c + 36) % 72
}

/// A fixture's mesh on the sphere, as `tests/esmf_mesh.rs` reads it.
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
    PolyMesh::new(x, y, start, nodes).unwrap().with_great_circles().unwrap()
}

fn opts(gt: [f64; 6]) -> WeightOptions {
    WeightOptions::new(1).with_measure(lonlat_measure(gt).unwrap()).with_refine_tolerance(1e-7).with_moments(true)
}

/// One value per face: the smooth field at the face's nodes' mean on the
/// sphere.
fn face_values(m: &PolyMesh) -> Vec<f64> {
    (0..m.nface())
        .map(|f| {
            let mut u = [0.0f64; 3];
            for &n in m.face(f) {
                let [lon, lat] = m.node(n).map(f64::to_radians);
                u = [u[0] + lat.cos() * lon.cos(), u[1] + lat.cos() * lon.sin(), u[2] + lat.sin()];
            }
            let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            smooth(u[1].atan2(u[0]).to_degrees(), (u[2] / r).asin().to_degrees())
        })
        .collect()
}

/// The smooth field at each cell's middle, row-major, for `grid(lon0)`.
fn cell_values(lon0: f64) -> Vec<f64> {
    (0..36 * 72).map(|k| smooth(lon0 + 5.0 * (k % 72) as f64 + 2.5, 90.0 - 5.0 * (k / 72) as f64 - 2.5)).collect()
}

fn buffer(data: &[f64], ncol: usize) -> SourceBuffer<'_, f64> {
    SourceBuffer { data, ncol, nrow: data.len() / ncol, nbands: 1, col_off: 0, row_off: 0 }
}

fn without_period(w: &Weights) -> Weights {
    let mut w = w.clone();
    w.period = None;
    w
}

#[test]
fn the_period_is_recorded_with_moments() {
    let m = read_mesh("mesh_ico.mesh.txt");
    let (t, g, gt) = grid(-180.0);
    let (with, plain) = (opts(gt), WeightOptions::new(1));
    let par = || Ok(grid(-180.0).0);
    // the grid as a source, onto cells of 1 by 1 in its world coordinates
    assert_eq!(exact_weights(&t, &g, 10, 10, &with).period, Some(72));
    assert_eq!(exact_weights_par(par, &g, 10, 10, &with, 3).unwrap().period, Some(72));
    assert_eq!(exact_weights(&t, &g, 10, 10, &plain).period, None);
    // onto a mesh and from one
    assert_eq!(m.weights_onto(&t, &g, &with).unwrap().period, Some(72));
    assert_eq!(m.weights_onto_par(par, &g, &with, 3).unwrap().period, Some(72));
    assert_eq!(m.weights_to(&t, &g, &with).unwrap().period, Some(72));
    assert_eq!(m.weights_to_par(par, &g, &with, 3).unwrap().period, Some(72));
    assert_eq!(m.weights_to(&t, &g, &plain).unwrap().period, None);
    assert_eq!(m.weights_onto(&t, &g, &plain).unwrap().period, None);
    // a grid that doesn't wrap
    let flat = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    assert_eq!(exact_weights(&flat, &g, 10, 10, &with).period, None);
    assert_eq!(m.weights_to(&flat, &g, &with).unwrap().period, None);
    // 0.7 degree cells: 514 of them wrap, every 514.29 columns
    let gt7 = [-180.0, 0.7, 0.0, 90.0, 0.0, -0.7];
    let t7 = GenImgProjTransformer::with_backend(gt7, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([514, 257]);
    assert!(t7.source_wrap().is_some());
    assert_eq!(exact_weights(&t7, &GridWindow::new(514, 257), 10, 10, &opts(gt7)).period, None);
}

/// The largest difference between values on `grid(-180.0)` and
/// `grid(0.0)`, cell for cell.
fn turned_diff(a: &[f64], b: &[f64]) -> f64 {
    (0..36 * 72).map(|k| (a[k] - b[k / 72 * 72 + turned(k % 72)]).abs()).fold(0.0, f64::max)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// The seam on the far side of the globe changes nothing but the faces'
/// outlines, refined to 1e-7 pixels in each (so first order differs by
/// 4e-10): second order by as little.
#[test]
fn second_order_doesn_t_see_the_seam() {
    let norm = Normalization::FracArea;
    for name in ["mesh_ico", "mesh_dual"] {
        let m = read_mesh(&format!("{name}.mesh.txt"));
        let ((ta, ga, gta), (tb, gb, gtb)) = (grid(-180.0), grid(0.0));
        let faces = face_values(&m);
        let fb = buffer(&faces, faces.len());
        // the faces onto the grid, the seam at 180 and at 0
        let (wa, wb) = (m.weights_to(&ta, &ga, &opts(gta)).unwrap(), m.weights_to(&tb, &gb, &opts(gtb)).unwrap());
        let first = turned_diff(&wa.apply::<f64, f64>(&fb, norm, &ApplyOptions::new()), &wb.apply(&fb, norm, &ApplyOptions::new()));
        for limiter in [Limiter::Off, Limiter::BarthJespersen] {
            let a = ApplyOptions::new().with_limiter(limiter);
            let d = turned_diff(&m.apply_second_order(&wa, &fb, norm, &a), &m.apply_second_order(&wb, &fb, norm, &a));
            let d0 = turned_diff(
                &m.apply_second_order(&without_period(&wa), &fb, norm, &a),
                &m.apply_second_order(&without_period(&wb), &fb, norm, &a),
            );
            println!("{name} to the grid, {limiter:?}: turned the grid's cells differ by {d:.1e}, without the period {d0:.1e} (first order {first:.1e})");
            assert!(d < 1e-9 && d0 > 1e-2, "{name} {limiter:?}: {d} {d0}");
        }
        // folded, as applied (under DestArea: FracArea's running mean in
        // Weights::apply is rough with a folded table's negative weights)
        let (a, dest) = (ApplyOptions::new(), Normalization::DestArea);
        let fold = max_diff(&m.second_order(&wa, None).apply::<f64, f64>(&fb, dest, &a), &m.apply_second_order(&wa, &fb, dest, &a));
        assert!(fold < 1e-12, "{name} faces' fold: {fold}");

        // the grid onto the faces
        let (ca, cb) = (cell_values(-180.0), cell_values(0.0));
        let (ba, bb) = (buffer(&ca, 72), buffer(&cb, 72));
        let (wa, wb) = (m.weights_onto(&ta, &ga, &opts(gta)).unwrap(), m.weights_onto(&tb, &gb, &opts(gtb)).unwrap());
        for limiter in [Limiter::Off, Limiter::BarthJespersen] {
            let a = ApplyOptions::new().with_limiter(limiter);
            let d = max_diff(&wa.apply_second_order::<f64, f64>(&ba, norm, &a), &wb.apply_second_order(&bb, norm, &a));
            let d0 = max_diff(
                &without_period(&wa).apply_second_order::<f64, f64>(&ba, norm, &a),
                &without_period(&wb).apply_second_order(&bb, norm, &a),
            );
            println!("{name} onto the faces, {limiter:?}: with the seam turned they differ by {d:.1e}, without the period {d0:.1e}");
            assert!(d < 1e-9 && d0 > 1e-4, "{name} {limiter:?}: {d} {d0}");
        }
        let fold = max_diff(&wa.second_order(&ga, None, None).apply::<f64, f64>(&ba, dest, &a), &wa.apply_second_order(&ba, dest, &a));
        assert!(fold < 1e-12, "{name} grid's fold: {fold}");
    }
}

#[test]
fn a_face_round_a_pole_is_first_order() {
    // mesh_dual's poles are in pentagons, each round one, its cells taking
    // every column of the top or bottom rows
    let m = read_mesh("mesh_dual.mesh.txt");
    let (t, g, gt) = grid(-180.0);
    let w = m.weights_to(&t, &g, &opts(gt)).unwrap();
    let mut cols = vec![std::collections::BTreeSet::new(); m.nface()];
    for k in 0..w.len() {
        if w.frac[k] > 1e-12 {
            cols[w.src_col[k] as usize].insert(w.dst_col[k]);
        }
    }
    let round: Vec<usize> = (0..m.nface()).filter(|&f| cols[f].len() == 72).collect();
    assert_eq!(round.len(), 2, "{round:?}");
    let faces = face_values(&m);
    let fb = buffer(&faces, faces.len());
    let a = ApplyOptions::new();
    let first: Vec<f64> = w.apply(&fb, Normalization::FracArea, &a);
    let second: Vec<f64> = m.apply_second_order(&w, &fb, Normalization::FracArea, &a);
    // first order in the cells only it covers: the top and bottom rows
    for k in (0..72).chain(35 * 72..36 * 72) {
        assert!((second[k] - first[k]).abs() < 1e-14, "cell {k}: {} vs {}", second[k], first[k]);
    }
    // and no one's neighbour: in the fold, only in its own cells. Without
    // the period it has a centre in the middle of the grid, and is
    let own = |w: &Weights, f: usize| {
        let mut c: Vec<(u32, u32)> = (0..w.len()).filter(|&k| w.src_col[k] as usize == f).map(|k| (w.dst_row[k], w.dst_col[k])).collect();
        c.sort_unstable();
        c.dedup();
        c
    };
    for &f in &round {
        assert_eq!(own(&m.second_order(&w, None), f), own(&w, f), "face {f}");
        assert!(own(&m.second_order(&without_period(&w), None), f).len() > own(&w, f).len(), "face {f}");
    }
}
