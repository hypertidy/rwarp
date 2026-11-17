//! Meshes whose outer walls meet in the world: a periodic seam (column
//! line `ncol` on column line 0), a fold (the last row line on itself, in
//! reverse) and both together, as a tripolar grid is in a polar
//! projection. The walk crosses such walls with nothing declared; each
//! cell's coverage is checked against the world clip of its four split
//! triangles, which knows nothing of the mesh's topology.

use controlledburn::{BurnOptions, BurnResult, Coord, CoverageWindow, Geometry, Polygon};
use rwarp::mesh::{CellMap, QuadMesh};
use std::f64::consts::TAU;

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

fn lattice(ncol: usize, nrow: usize, f: impl Fn(usize, usize) -> [f64; 2]) -> Corners {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = f(i, j);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    (ncol, nrow, x, y)
}

/// A full annulus, radius 100 to 220, column lines along the angle. With
/// `exact`, column line `ncol` is column line 0 bitwise; otherwise it is
/// computed at an angle of 2 pi, so the two walls agree only to rounding.
fn annulus(exact: bool) -> Corners {
    let (nc, nr) = (24, 6);
    lattice(nc, nr, |i, j| {
        let i = if exact { i % nc } else { i };
        let a = i as f64 / nc as f64 * TAU;
        let r = 100.0 + 20.0 * j as f64;
        [r * a.cos(), r * a.sin()]
    })
}

/// A disk with a seam and a fold, as a tripolar grid is in a polar
/// projection. Row line 0 is a circle of radius 200; row lines between are
/// nested ellipses; the last row line is the segment from (60, 0) to
/// (-60, 0) and back, so corner `i` is corner `ncol - i` there. Its ends
/// are the two poles: (60, 0) where the seam meets the fold (`i = 0`), and
/// (-60, 0) (`i = ncol / 2`). With `exact`, the seam and the fold agree
/// bitwise; otherwise only to rounding.
fn tripolar(exact: bool) -> Corners {
    let (nc, nr) = (24, 8);
    lattice(nc, nr, |i, j| {
        let i = if exact { i % nc } else { i };
        let a = i as f64 / nc as f64 * TAU;
        let folded = if exact {
            (i.min(nc - i) as f64 / nc as f64) * TAU
        } else {
            a
        };
        let t = j as f64 / nr as f64;
        let (big, s) = (200.0, 60.0);
        let x = (1.0 - t) * big * a.cos() + t * s * folded.cos();
        let y = if j == nr && exact {
            0.0
        } else {
            (1.0 - t) * big * a.sin()
        };
        [x, y]
    })
}

/// A fold with no seam: the ellipses of [`tripolar`] over angles -A to A
/// (A < pi). The last row line runs from (60 cos A, 0) to (60, 0) and back,
/// and column lines 0 and `ncol` are real walls either side of a slit.
fn folded() -> Corners {
    let (nc, nr) = (20, 6);
    let big_a = 2.0f64;
    lattice(nc, nr, |i, j| {
        // |theta| is the same for i and ncol - i, bitwise
        let th = big_a * (2 * i as i64 - nc as i64).abs() as f64 / nc as f64;
        let sign = if 2 * i < nc { -1.0 } else { 1.0 };
        let t = j as f64 / nr as f64;
        let (big, s) = (200.0, 60.0);
        let x = ((1.0 - t) * big + t * s) * th.cos();
        let y = if j == nr {
            0.0
        } else {
            sign * (1.0 - t) * big * th.sin()
        };
        [x, y]
    })
}

fn mesh(c: Corners) -> QuadMesh {
    QuadMesh::from_corners(c.0, c.1, c.2, c.3, CellMap::EqualArea).unwrap()
}

fn meshes() -> Vec<(&'static str, QuadMesh)> {
    vec![
        ("annulus", mesh(annulus(true))),
        ("annulus, seam to rounding", mesh(annulus(false))),
        ("folded", mesh(folded())),
        ("tripolar", mesh(tripolar(true))),
        ("tripolar, seam and fold to rounding", mesh(tripolar(false))),
    ]
}

fn shoelace(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n)
        .map(|k| p[k][0] * p[(k + 1) % n][1] - p[(k + 1) % n][0] * p[k][1])
        .sum::<f64>()
        * 0.5
}

/// Sutherland-Hodgman: ring `p` clipped by convex counter-clockwise `clip`.
fn clip(p: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = p.to_vec();
    let n = clip.len();
    for k in 0..n {
        let (a, b) = (clip[k], clip[(k + 1) % n]);
        let side = |q: [f64; 2]| (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
        let input = std::mem::take(&mut out);
        let m = input.len();
        for i in 0..m {
            let (s, e) = (input[(i + m - 1) % m], input[i]);
            let (ss, se) = (side(s), side(e));
            if se >= 0.0 {
                if ss < 0.0 {
                    let t = ss / (ss - se);
                    out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
                }
                out.push(e);
            } else if ss >= 0.0 {
                let t = ss / (ss - se);
                out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// Each cell's covered fraction of its world area for a polygon (first
/// ring exterior, the rest holes), clipped against its split triangles.
fn oracle(m: &QuadMesh, rings: &[Vec<[f64; 2]>]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrow() * m.ncol()];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let a = m.cell_area(r, c);
            if a == 0.0 {
                continue;
            }
            let q = m.corners(r, c);
            let x = m.split_centre(r, c).0;
            let mut cov = 0.0;
            for k in 0..4 {
                let mut tri = vec![x, q[k], q[(k + 1) % 4]];
                if shoelace(&tri) < 0.0 {
                    tri.reverse();
                }
                for (i, ring) in rings.iter().enumerate() {
                    let part = shoelace(ring).signum() * shoelace(&clip(ring, &tri));
                    cov += if i == 0 { part } else { -part };
                }
            }
            out[r * m.ncol() + c] = cov / a;
        }
    }
    out
}

fn dense(res: &BurnResult, ncol: usize, nrow: usize) -> Vec<f64> {
    let mut d = vec![0.0; ncol * nrow];
    for run in &res.runs {
        for c in run.col_start..run.col_end {
            d[run.row as usize * ncol + c as usize] += 1.0;
        }
    }
    for e in &res.edges {
        d[e.row as usize * ncol + e.col as usize] += e.fraction as f64;
    }
    d
}

fn polygon(rings: &[Vec<[f64; 2]>]) -> Geometry {
    Geometry::Polygon(Polygon {
        rings: rings
            .iter()
            .map(|r| {
                let mut v: Vec<Coord> = r.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
                v.push(v[0]);
                v
            })
            .collect(),
    })
}

fn check(name: &str, m: &QuadMesh, rings: &[Vec<[f64; 2]>]) {
    let res = m
        .burn(&[polygon(rings)], BurnOptions::coverage())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let got = dense(&res, m.ncol(), m.nrow());
    let want = oracle(m, rings);
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-6,
            "{name}: cell ({}, {}) burn {g} oracle {w}",
            k / m.ncol(),
            k % m.ncol()
        );
    }
}

fn circle(c: [f64; 2], r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let a = k as f64 / n as f64 * TAU + 0.1;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

/// A star-shaped ring around `c`, some vertices pulled onto `snap` points.
fn star(g: &mut Lcg, c: [f64; 2], r: f64, n: usize, snap: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let a = (k as f64 + 0.3 * g.u()) / n as f64 * TAU;
        let rr = r * (0.35 + 0.65 * g.u());
        let mut p = [c[0] + rr * a.cos(), c[1] + rr * a.sin()];
        if !snap.is_empty() && g.u() < 0.3 {
            let s = snap
                .iter()
                .min_by(|s, t| {
                    let ds = (s[0] - p[0]).powi(2) + (s[1] - p[1]).powi(2);
                    let dt = (t[0] - p[0]).powi(2) + (t[1] - p[1]).powi(2);
                    ds.total_cmp(&dt)
                })
                .copied()
                .unwrap();
            if ((s[0] - p[0]).powi(2) + (s[1] - p[1]).powi(2)).sqrt() < 0.3 * r {
                p = s;
            }
        }
        out.push(p);
    }
    out.dedup();
    out
}

/// Mesh corners and split centres, and points on the seam and fold.
fn snaps(m: &QuadMesh) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            out.extend(m.corners(r, c));
            out.push(m.split_centre(r, c).0);
        }
    }
    out
}

/// Random polygons centred near the glued walls (the positive x axis and
/// the fold segment) and anywhere over the mesh, some with a hole, half
/// steered onto corners and split centres.
fn cases(m: &QuadMesh, seed: u64, n: usize) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut g = Lcg(seed);
    let sn = snaps(m);
    let mut out = Vec::new();
    for k in 0..n {
        let c = match k % 3 {
            0 => [60.0 + 160.0 * g.u(), 20.0 * (g.u() - 0.5)], // the seam
            1 => [120.0 * (g.u() - 0.5), 20.0 * (g.u() - 0.5)], // the fold
            _ => {
                let (r, a) = (230.0 * g.u(), TAU * g.u());
                [r * a.cos(), r * a.sin()]
            }
        };
        let r = 8.0 + 60.0 * g.u();
        let nv = 5 + (g.u() * 14.0) as usize;
        let ext = star(&mut g, c, r, nv, if k % 2 == 0 { &sn } else { &[] });
        if ext.len() < 3 {
            continue;
        }
        let mut rings = vec![ext];
        if k % 4 == 0 {
            rings.push(circle(c, 0.25 * r, 6).into_iter().rev().collect());
        }
        out.push(rings);
    }
    out
}

#[test]
fn random_polygons_across_seams_and_folds_match_the_world_clip() {
    for (name, m) in meshes() {
        for (i, rings) in cases(&m, 20261117, 90).iter().enumerate() {
            if !simple(rings) {
                continue;
            }
            check(&format!("{name} case {i}"), &m, rings);
        }
    }
}

#[test]
fn rings_around_the_hole_the_fold_and_the_poles() {
    // a comb crossing the annulus's seam four times
    let comb = vec![
        [120.0, -15.0],
        [120.0, 15.0],
        [135.0, 15.0],
        [135.0, -5.0],
        [150.0, -5.0],
        [150.0, 15.0],
        [165.0, 15.0],
        [165.0, -15.0],
    ];
    for (name, m) in meshes() {
        let mut rings: Vec<Vec<Vec<[f64; 2]>>> = vec![
            vec![circle([0.0, 0.0], 150.0, 41)], // round the hole, or the fold: one seam crossing
            vec![circle([0.0, 0.0], 80.0, 23)],  // tripolar: the whole fold, one seam crossing
            vec![circle([0.0, 0.0], 30.0, 17)],  // across the fold twice
            vec![circle([60.0, 0.0], 20.0, 19)], // round the pole where the seam meets the fold
            vec![circle([-60.0, 0.0], 20.0, 19)], // round the other pole: one fold crossing
            vec![circle([0.0, 0.0], 300.0, 29)], // the whole mesh
            vec![
                circle([0.0, 0.0], 300.0, 29),
                circle([0.0, 0.0], 150.0, 41).into_iter().rev().collect(),
            ],
            vec![comb.clone()],
        ];
        // a ring round the hole (the fold) that crosses the seam three
        // times: a zigzag across it, then the long way round at r 195
        let polar = |r: f64, a: f64| [r * a.cos(), r * a.sin()];
        let mut wiggle = vec![
            polar(140.0, -0.3),
            polar(140.0, 0.2),
            polar(160.0, 0.2),
            polar(160.0, -0.2),
            polar(180.0, -0.2),
            polar(180.0, 0.3),
        ];
        wiggle.extend((0..=12).map(|k| polar(195.0, 0.3 + k as f64 / 12.0 * (TAU - 0.6))));
        rings.push(vec![wiggle]);
        for (i, r) in rings.iter().enumerate() {
            check(&format!("{name} ring {i}"), &m, r);
        }
    }
}

#[test]
fn cover_ring_across_seams_and_folds_matches_burn() {
    let mut win = CoverageWindow::default();
    for (name, m) in meshes() {
        for (i, rings) in cases(&m, 7, 45).iter().enumerate() {
            let ring = &rings[0];
            let d = dense(
                &m.burn(
                    &[polygon(std::slice::from_ref(ring))],
                    BurnOptions::coverage(),
                )
                .unwrap(),
                m.ncol(),
                m.nrow(),
            );
            let hit = m.cover_ring(ring, &mut win).unwrap();
            let mut w = vec![0.0; d.len()];
            if hit {
                let cov = win.coverage();
                for jr in 0..win.nrow() {
                    for jc in 0..win.ncol() {
                        let (r, c) = (win.row0() + jr as i64, win.col0() + jc as i64);
                        if r >= 0 && c >= 0 && (r as usize) < m.nrow() && (c as usize) < m.ncol() {
                            w[r as usize * m.ncol() + c as usize] = cov[jr * win.ncol() + jc];
                        }
                    }
                }
            }
            for k in 0..d.len() {
                assert!(
                    (d[k] - w[k]).abs() < 2e-6,
                    "{name} case {i}: cell {k}: burn {} window {}",
                    d[k],
                    w[k]
                );
            }
        }
    }
}

#[test]
fn glued_meshes_are_one_sheet() {
    // every cell centre has one owner: the seam and fold add no overlap
    for (name, m) in meshes() {
        for r in 0..m.nrow() {
            for c in 0..m.ncol() {
                let p = m.split_centre(r, c).0;
                assert_eq!(m.locate_all(p), vec![(r, c)], "{name}: cell ({r}, {c})");
            }
        }
    }
}

/// Many more polygons (8 seeds of 400 per mesh, about 16,000 valid):
/// `cargo test --release --test mesh_glued -- --ignored`.
#[test]
#[ignore]
fn many_polygons_across_seams_and_folds() {
    let mut n = 0;
    for seed in 1..=8u64 {
        for (name, m) in meshes() {
            for (i, rings) in cases(&m, seed * 7919, 400).iter().enumerate() {
                if !simple(rings) {
                    continue;
                }
                check(&format!("{name} seed {seed} case {i}"), &m, rings);
                n += 1;
            }
        }
    }
    println!("{n} polygons");
}

/// No two edges of the rings meet, except neighbours at their shared vertex.
fn simple(rings: &[Vec<[f64; 2]>]) -> bool {
    let mut e = Vec::new();
    for (ri, r) in rings.iter().enumerate() {
        for k in 0..r.len() {
            e.push((ri, k, r.len(), r[k], r[(k + 1) % r.len()]));
        }
    }
    let o = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    for i in 0..e.len() {
        for j in i + 1..e.len() {
            let (ri, ki, ni, a, b) = e[i];
            let (rj, kj, _, c, d) = e[j];
            if ri == rj && (kj == ki + 1 || (ki == 0 && kj == ni - 1)) {
                continue;
            }
            let (d1, d2, d3, d4) = (o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b));
            if d1 * d2 <= 0.0 && d3 * d4 <= 0.0 {
                return false;
            }
        }
    }
    true
}
