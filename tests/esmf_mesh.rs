//! ESMF between a mesh of polygons and a grid, both ways: conservative
//! weights from [`PolyMesh::weights_to`] and [`PolyMesh::weights_onto`]
//! against ESMF's (conservative stage 5, `design/conservative.md`
//! section 7, #39).
//!
//! The mesh is `tests/fixtures/esmf/gen.py`'s (`mesh_hex.mesh.txt`):
//! hexagons with every third split into six triangles round its centre,
//! its nodes shared and moved a little, over part of a 14 by 12 grid of
//! unit cells and past its top-left corner, so some faces are partly off
//! the grid and some cells partly or not at all under the mesh. ESMF's
//! maps are Cartesian in the grid's plane, where every edge is straight
//! on both sides, as rwarp's are at densify 1: they must agree to
//! rounding.
//!
//! - `mesh_hex_to`: mesh to grid. `S = frac * face area / cell area`
//!   ([`Normalization::DestArea`]).
//! - `mesh_hex_onto`: grid to mesh. `S = frac / face area`.
//!
//! On the sphere (`on_the_sphere`), a geodesic mesh with nodes on the
//! poles and its dual, hexagons and pentagons with two round the poles,
//! to and from a global 5 degree grid, with great-circle edges: the faces
//! are the same on both sides and their areas agree, but the grid's cells
//! are not (rwarp's are bounded by parallels, ESMF's by great circles),
//! so the weights' differences are recorded as ceilings.
//!
//! `cargo test --test esmf_mesh -- --nocapture` prints the differences.

mod esmf_common;

use esmf_common::{diff, fixture_path, read_fixture, Fixture, Ours};
use rwarp::mesh::{PolyMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::weights::{lonlat_measure, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::CrsTransform;

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// A fixture's mesh: `nnode nface`, the nodes, then each face as its
/// node count and 0-based nodes.
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
        assert_eq!(v[0], v.len() - 1);
        nodes.extend(&v[1..]);
        start.push(nodes.len());
    }
    assert_eq!(start.len(), nf + 1, "truncated mesh");
    PolyMesh::new(x, y, start, nodes).unwrap()
}

fn mesh() -> PolyMesh {
    read_mesh("mesh_hex.mesh.txt")
}

fn grid(fx: &Fixture) -> (GenImgProjTransformer, GridWindow) {
    let [w, h] = fx.nums::<2>("grid_size").map(|v| v as usize);
    let t = GenImgProjTransformer::with_backend(fx.nums("grid_gt"), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    (t, GridWindow::new(w, h))
}

/// Our map in ESMF's terms: 1-based flattened indices, `S` under
/// destination-area normalisation, each destination cell's covered
/// fraction and area.
fn ours(w: &Weights, dst_ncol: usize, src_ncol: usize) -> Ours {
    let s = w.normalised(Normalization::DestArea);
    let (row, col) = w.esmf_row_col(dst_ncol, src_ncol);
    Ours {
        s: row.into_iter().zip(col).zip(s).collect(),
        frac_b: w.dst_covered().iter().zip(&w.dst_area).map(|(c, a)| c / a).collect(),
        area_b: w.dst_area.clone(),
    }
}

/// A field with no symmetry, on the faces or the cells.
fn field(n: usize) -> Vec<f64> {
    (0..n).map(|k| 2.0 + (0.37 * k as f64).sin()).collect()
}

fn check(name: &str, fx: &Fixture, o: &Ours, nsrc: usize) {
    let d = diff(fx, o, &field(nsrc));
    let area = o.area_b.iter().zip(&fx.area_b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!(
        "{name}: max |dS| {:.1e} (only ours {}, only ESMF {}, largest such {:.1e}); |d frac_b| {:.1e}; |d area_b| {:.1e}; field {:.1e}; ESMF rows {:.1e}",
        d.s, d.only.0, d.only.1, d.only.2, d.frac_b, area, d.field, d.esmf_rows
    );
    assert!(d.s < 1e-13 && d.frac_b < 1e-13 && area < 1e-13 && d.field < 1e-13, "{name}: {d:?}");
    assert!(d.only.2 < 1e-13, "{name}: an entry in one map only");
}

#[test]
fn mesh_to_grid_matches_esmf() {
    let fx = read_fixture("mesh_hex_to.cart.txt");
    let m = mesh();
    let (t, g) = grid(&fx);
    let w = m.weights_to(&t, &g, &WeightOptions::new(1)).unwrap();
    check("mesh_hex_to", &fx, &ours(&w, g.ncol, m.nface()), m.nface());
}

#[test]
fn grid_onto_mesh_matches_esmf() {
    let fx = read_fixture("mesh_hex_onto.cart.txt");
    let m = mesh();
    let (t, g) = grid(&fx);
    let w = m.weights_onto(&t, &g, &WeightOptions::new(1)).unwrap();
    check("mesh_hex_onto", &fx, &ours(&w, m.nface(), g.ncol), g.ncol * g.nrow);
}

// ---- on the sphere -------------------------------------------------------------------

/// A spherical case: the mesh with great-circle edges, refined to 1e-7
/// pixels, and the global grid in longitude and latitude, wrapped, under
/// the spherical measure. Areas are in steradians, as ESMF's.
fn sphere(name: &str, kind: &str) -> (Fixture, Ours, f64) {
    let fx = read_fixture(&format!("{name}_{kind}.sph.txt"));
    let m = read_mesh(&format!("{name}.mesh.txt")).with_great_circles().unwrap();
    let gt: [f64; 6] = fx.nums("grid_gt");
    let [w, h] = fx.nums::<2>("grid_size").map(|v| v as usize);
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([w, h]);
    let opts = WeightOptions::new(1).with_measure(lonlat_measure(gt).unwrap()).with_refine_tolerance(1e-7);
    let g = GridWindow::new(w, h);
    let (wts, n_dst, n_src) = if kind == "to" {
        (m.weights_to(&t, &g, &opts).unwrap(), w, m.nface())
    } else {
        (m.weights_onto(&t, &g, &opts).unwrap(), m.nface(), w)
    };
    let mut o = ours(&wts, n_dst, n_src);
    let px = gt[1].to_radians() * gt[5].to_radians().abs();
    o.area_b.iter_mut().for_each(|a| *a *= px);
    let nsrc = if kind == "to" { m.nface() } else { w * h };
    (fx, o, nsrc as f64)
}

fn report(name: &str, fx: &Fixture, o: &Ours, nsrc: usize) -> (esmf_common::Diff, f64) {
    let d = diff(fx, o, &field(nsrc));
    let area = o.area_b.iter().zip(&fx.area_b).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
    eprintln!(
        "{name}: max |dS| {:.1e} (only ours {}, only ESMF {}, largest such {:.1e}); |d frac_b| {:.1e}; area_b relative {:.1e}; field {:.1e}; ESMF rows {:.1e}",
        d.s, d.only.0, d.only.1, d.only.2, d.frac_b, area, d.field, d.esmf_rows
    );
    (d, area)
}

/// The faces are the same on both sides, so onto the mesh their areas
/// agree (both are their spherical excess: to 8e-8 at a tolerance of
/// 1e-7 pixels, 5e-9 at 1e-8), and both cover the whole sphere. The grid's
/// cells are not the same: rwarp's are bounded by parallels and ESMF's by
/// great circles, 1.3e-3 apart in area at 5 degrees, and the weights'
/// differences are those of the cells, recorded here as ceilings: `S` to
/// 6.9e-4 onto the mesh and 3.8e-3 from it, a smooth field to 1.9e-3 and
/// 5.0e-3.
#[test]
fn on_the_sphere() {
    // name, kind: S, area_b (relative), field
    let ceilings = [
        ("mesh_ico", "onto", 1e-3, 1e-7, 3e-3),
        ("mesh_ico", "to", 5e-3, 2e-3, 7e-3),
        ("mesh_dual", "onto", 1e-3, 1e-7, 3e-3),
        ("mesh_dual", "to", 5e-3, 2e-3, 7e-3),
    ];
    for (name, kind, s, area_b, f) in ceilings {
        let (fx, o, nsrc) = sphere(name, kind);
        let case = format!("{name}_{kind}");
        let (d, area) = report(&case, &fx, &o, nsrc as usize);
        assert!(d.frac_b < 1e-12 && d.only.2 < 1e-13, "{case}: {d:?}");
        assert!(d.s < s && area < area_b && d.field < f, "{case}: {d:?}, area_b {area:e}");
    }
}
