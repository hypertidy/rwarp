//! `QuadMesh` against exact oracles: brute-force ownership, round trips,
//! the equal-area identity, and GDAL-parity of the bilinear inverse.
//!
//! Meshes: a jittered lattice (convex cells), a strongly jittered one
//! (some cells non-convex), a polar sector, a sheared sinusoid, and the
//! same grids with rows stored the other way (clockwise cells).

use rwarp::geoloc_transform::{GeolocArrays, GeolocOptions};
use rwarp::mesh::{CellMap, QuadMesh};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

type Corners = (usize, usize, Vec<f64>, Vec<f64>);

fn lattice(ncol: usize, nrow: usize, f: impl Fn(f64, f64) -> [f64; 2]) -> Corners {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = f(i as f64, j as f64);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    (ncol, nrow, x, y)
}

fn jittered(seed: u64, amount: f64) -> Corners {
    let mut g = Lcg(seed);
    let (nc, nr) = (9, 7);
    let jit: Vec<[f64; 2]> = (0..(nc + 1) * (nr + 1))
        .map(|_| [(g.u() - 0.5) * 2.0 * amount, (g.u() - 0.5) * 2.0 * amount])
        .collect();
    lattice(nc, nr, |i, j| {
        let k = j as usize * (nc + 1) + i as usize;
        [0.7 * i + jit[k][0] * 0.7, 3.0 - 0.5 * j + jit[k][1] * 0.5]
    })
}

fn polar() -> Corners {
    lattice(12, 6, |i, j| {
        let (r, a) = (100.0 + 20.0 * j, (15.0 * i).to_radians());
        [r * a.cos(), r * a.sin()]
    })
}

fn sinusoid() -> Corners {
    lattice(10, 8, |i, j| {
        [
            i + 0.3 * (j * 0.7).sin(),
            j + 0.25 * (i * 0.9).cos() + 0.1 * i,
        ]
    })
}

/// A lattice with corners pushed diagonally into a neighbour: the cells
/// beside them are non-convex, and in some the corner mean lies outside the
/// cell.
fn dented() -> Corners {
    let (nc, nr, mut x, mut y) = lattice(6, 5, |i, j| [i, j]);
    for &(j, i, d) in &[(1usize, 1usize, 0.7), (2, 4, 0.6), (4, 2, -0.65)] {
        let k = j * (nc + 1) + i;
        x[k] += d;
        y[k] += d;
    }
    (nc, nr, x, y)
}

fn flipped(c: Corners) -> Corners {
    let (nc, nr, x, y) = c;
    let w = nc + 1;
    let mut fx = Vec::with_capacity(x.len());
    let mut fy = Vec::with_capacity(y.len());
    for j in (0..=nr).rev() {
        fx.extend_from_slice(&x[j * w..(j + 1) * w]);
        fy.extend_from_slice(&y[j * w..(j + 1) * w]);
    }
    (nc, nr, fx, fy)
}

fn meshes(map: CellMap) -> Vec<(&'static str, QuadMesh)> {
    let mut out = Vec::new();
    let cases: Vec<(&'static str, Corners)> = vec![
        ("jitter", jittered(7, 0.2)),
        ("jitter_strong", jittered(11, 0.45)),
        ("polar", polar()),
        ("sinusoid", sinusoid()),
        ("dented", dented()),
        ("jitter_flipped", flipped(jittered(7, 0.2))),
        ("polar_flipped", flipped(polar())),
    ];
    for (name, (nc, nr, x, y)) in cases {
        out.push((
            name,
            QuadMesh::from_corners(nc, nr, x, y, map).unwrap_or_else(|e| panic!("{name}: {e}")),
        ));
    }
    out
}

fn bbox(m: &QuadMesh) -> [f64; 4] {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            for p in m.corners(r, c) {
                b = [
                    b[0].min(p[0]),
                    b[1].min(p[1]),
                    b[2].max(p[0]),
                    b[3].max(p[1]),
                ];
            }
        }
    }
    b
}

fn brute(m: &QuadMesh, p: [f64; 2]) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            if m.contains(r, c, p) {
                v.push((r, c));
            }
        }
    }
    v
}

/// Random points over the bounding box, plus every corner and every side
/// midpoint: the places where ownership is decided by a tie.
fn probes(m: &QuadMesh, g: &mut Lcg, n: usize) -> Vec<[f64; 2]> {
    let b = bbox(m);
    let mut v: Vec<[f64; 2]> = (0..n)
        .map(|_| [b[0] + g.u() * (b[2] - b[0]), b[1] + g.u() * (b[3] - b[1])])
        .collect();
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let q = m.corners(r, c);
            for k in 0..4 {
                v.push(q[k]);
                let (a, bb) = (q[k], q[(k + 1) % 4]);
                v.push([0.5 * (a[0] + bb[0]), 0.5 * (a[1] + bb[1])]);
            }
            v.push(m.split_centre(r, c).0);
        }
    }
    v
}

#[test]
fn every_point_has_at_most_one_owner_and_locate_finds_it() {
    for map in [CellMap::Bilinear, CellMap::EqualArea] {
        for (name, m) in meshes(map) {
            let mut g = Lcg(3);
            let pts = probes(&m, &mut g, 4000);
            let mut owned = 0;
            for p in &pts {
                let b = brute(&m, *p);
                assert!(b.len() <= 1, "{name}: {p:?} owned by {b:?}");
                let want = b.first().copied();
                assert_eq!(m.locate(*p, None), want, "{name}: locate {p:?}");
                assert_eq!(m.locate_all(*p), b, "{name}: locate_all {p:?}");
                // any hint, near or far, gives the same answer
                let hint = (
                    (g.u() * m.nrow() as f64) as usize,
                    (g.u() * m.ncol() as f64) as usize,
                );
                assert_eq!(
                    m.locate(*p, Some(hint)),
                    want,
                    "{name}: locate {p:?} from {hint:?}"
                );
                owned += want.is_some() as usize;
            }
            assert!(
                owned > pts.len() / 3,
                "{name}: only {owned} of {} owned",
                pts.len()
            );
        }
    }
}

#[test]
fn interior_corners_belong_to_exactly_one_cell() {
    for (name, m) in meshes(CellMap::EqualArea) {
        for r in 1..m.nrow() {
            for c in 1..m.ncol() {
                let p = m.corners(r, c)[0];
                assert_eq!(brute(&m, p), vec![(r, c)], "{name}: corner ({r}, {c})");
            }
        }
    }
}

#[test]
fn index_and_world_round_trip() {
    for map in [CellMap::Bilinear, CellMap::EqualArea] {
        for (name, m) in meshes(map) {
            if map == CellMap::Bilinear && name == "dented" {
                // a bilinear map of a non-convex cell leaves the cell: the
                // GDAL model is not a map onto its own quadrilateral there,
                // one more reason for the equal-area map
                continue;
            }
            let mut g = Lcg(5);
            let mut outside = 0;
            let scale = {
                let b = bbox(&m);
                (b[2] - b[0]).max(b[3] - b[1])
            };
            for _ in 0..3000 {
                let idx = [g.u() * m.ncol() as f64, g.u() * m.nrow() as f64];
                let w = m.to_world(idx).unwrap();
                let Some(back) = m.to_index(w, None) else {
                    outside += 1;
                    continue;
                };
                // no gaps: the point is owned by the cell it came from
                let cell = (idx[1].floor() as usize, idx[0].floor() as usize);
                assert_eq!(m.locate(w, None), Some(cell), "{name} {map:?}: {idx:?}");
                assert!(
                    (back[0] - idx[0]).abs() < 1e-9 && (back[1] - idx[1]).abs() < 1e-9,
                    "{name} {map:?}: {idx:?} -> {w:?} -> {back:?}"
                );
                let w2 = m.to_world(back).unwrap();
                assert!(
                    (w2[0] - w[0]).abs() < 1e-12 * scale && (w2[1] - w[1]).abs() < 1e-12 * scale,
                    "{name} {map:?}"
                );
            }
            assert_eq!(outside, 0, "{name} {map:?}");
        }
    }
}

fn tri_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

/// A random point of a triangle.
fn in_tri(g: &mut Lcg, a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let (mut s, mut t) = (g.u(), g.u());
    if s + t > 1.0 {
        (s, t) = (1.0 - s, 1.0 - t);
    }
    [
        a[0] + s * (b[0] - a[0]) + t * (c[0] - a[0]),
        a[1] + s * (b[1] - a[1]) + t * (c[1] - a[1]),
    ]
}

/// The worst relative error of `index area * cell area = world area` over
/// small triangles inside the cells' fan triangles.
fn area_identity_error(m: &QuadMesh, g: &mut Lcg) -> f64 {
    let mut worst: f64 = 0.0;
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let q = m.corners(r, c);
            let (x, _) = m.split_centre(r, c);
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                let p: Vec<[f64; 2]> = (0..3).map(|_| in_tri(g, x, a, b)).collect();
                let wa = tri_area(p[0], p[1], p[2]);
                let ip: Vec<[f64; 2]> = p.iter().map(|&w| m.local(r, c, w).unwrap_or_else(|| panic!("({r}, {c}) tri {k}: {w:?} not owned; corners {q:?}, centre {x:?}, inexact {:?}", m.inexact_cells()))).collect();
                let ia = tri_area(ip[0], ip[1], ip[2]);
                if wa.abs() > 1e-6 * m.cell_area(r, c) {
                    worst = worst.max((ia.abs() * m.cell_area(r, c) / wa.abs() - 1.0).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn equal_area_cells_preserve_area_fractions() {
    for (name, m) in meshes(CellMap::EqualArea) {
        let e = area_identity_error(&m, &mut Lcg(9));
        assert!(e < 1e-9, "{name}: {e}");
        assert!(
            m.inexact_cells().is_empty(),
            "{name}: {:?}",
            m.inexact_cells()
        );
        if name == "dented" {
            // the case exercises the non-convex split: some corner means
            // fall outside their cell's kernel
            let moved = (0..m.nrow())
                .flat_map(|r| (0..m.ncol()).map(move |c| (r, c)))
                .filter(|&(r, c)| {
                    let q = m.corners(r, c);
                    let mean = [
                        (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0,
                        (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0,
                    ];
                    m.split_centre(r, c).0 != mean
                })
                .count();
            assert!(moved >= 1, "{moved} cells needed the non-convex split");
        }
        // the unit triangles' shares are the world triangles' shares
        for r in 0..m.nrow() {
            for c in 0..m.ncol() {
                let q = m.corners(r, c);
                let (x, u) = m.split_centre(r, c);
                let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                for k in 0..4 {
                    let ws = tri_area(x, q[k], q[(k + 1) % 4]).abs() / m.cell_area(r, c);
                    let us = tri_area(u, unit[k], unit[(k + 1) % 4]).abs();
                    assert!(
                        (ws - us).abs() < 1e-12,
                        "{name} ({r}, {c}) triangle {k}: {ws} vs {us}"
                    );
                }
            }
        }
    }
}

#[test]
fn bilinear_cells_do_not_preserve_area_on_curved_grids() {
    // the reason for the equal-area map: on the polar sector the bilinear
    // measure is off by a visible fraction of a cell
    let (_, m) = meshes(CellMap::Bilinear)
        .into_iter()
        .find(|(n, _)| *n == "polar")
        .unwrap();
    let e = area_identity_error(&m, &mut Lcg(9));
    assert!(e > 1e-3, "{e}");
}

#[test]
fn bilinear_matches_the_geolocation_inverse() {
    for (name, m) in meshes(CellMap::Bilinear) {
        let w = m.ncol() + 1;
        let (mut gx, mut gy) = (Vec::new(), Vec::new());
        for j in 0..=m.nrow() {
            for i in 0..=m.ncol() {
                let p = if j < m.nrow() && i < m.ncol() {
                    m.corners(j, i)[0]
                } else if i == m.ncol() && j < m.nrow() {
                    m.corners(j, i - 1)[1]
                } else if j == m.nrow() && i < m.ncol() {
                    m.corners(j - 1, i)[3]
                } else {
                    m.corners(j - 1, i - 1)[2]
                };
                gx.push(p[0]);
                gy.push(p[1]);
            }
        }
        let geo = GeolocArrays::new(gx, gy, w, m.nrow() + 1, GeolocOptions::default()).unwrap();
        let mut g = Lcg(13);
        let mut n = 0;
        for _ in 0..2000 {
            let idx = [g.u() * m.ncol() as f64, g.u() * m.nrow() as f64];
            let p = m.to_world(idx).unwrap();
            let Some(a) = m.to_index(p, None) else {
                continue;
            };
            let (pi, li) = geo.geo_to_pixel(p[0], p[1]).unwrap();
            assert!(
                (a[0] - pi).abs() < 1e-9 && (a[1] - li).abs() < 1e-9,
                "{name}: {a:?} vs ({pi}, {li})"
            );
            n += 1;
        }
        assert!(n > 1900, "{name}");
    }
}

#[test]
fn corners_from_centres() {
    // centres of a rotated regular lattice give its exact corners
    let (nc, nr) = (5, 4);
    let (s, c) = (0.3f64.sin(), 0.3f64.cos());
    let at = |i: f64, j: f64| [2.0 + c * i - s * j, -1.0 + s * i + c * j];
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..nr {
        for i in 0..nc {
            let p = at(i as f64 + 0.5, j as f64 + 0.5);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    let m = QuadMesh::from_centres(nc, nr, &x, &y, CellMap::EqualArea).unwrap();
    for r in 0..nr {
        for cc in 0..nc {
            let q = m.corners(r, cc);
            let want = [
                at(cc as f64, r as f64),
                at(cc as f64 + 1.0, r as f64),
                at(cc as f64 + 1.0, r as f64 + 1.0),
                at(cc as f64, r as f64 + 1.0),
            ];
            for k in 0..4 {
                assert!(
                    (q[k][0] - want[k][0]).abs() < 1e-12 && (q[k][1] - want[k][1]).abs() < 1e-12,
                    "({r}, {cc}) corner {k}"
                );
            }
            assert!((m.cell_area(r, cc) - 1.0).abs() < 1e-12);
        }
    }
}
