//! The curvilinear and rectilinear front ends (stage 2) against exact
//! oracles: world-space clipping of each cell, the affine burn of a
//! rotated regular grid, and the rectilinear front end.
//!
//! Generators 1 to 4 and 7 of `design/curvilinear-frontend.md` section 7:
//! affine as curvilinear, rectilinear as curvilinear, smooth warps (polar
//! sector, sinusoid, jittered lattices, flipped storage), steering
//! (vertices on corners, split centres and sides), and degenerate cells.

use controlledburn::{BurnOptions, BurnResult, Coord, CoverageWindow, Geometry, GridSpec, Polygon};
use rwarp::mesh::{CellMap, QuadMesh, Rectilinear};

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
    let (mut fx, mut fy) = (Vec::new(), Vec::new());
    for j in (0..=nr).rev() {
        fx.extend_from_slice(&x[j * w..(j + 1) * w]);
        fy.extend_from_slice(&y[j * w..(j + 1) * w]);
    }
    (nc, nr, fx, fy)
}

/// A rotated regular grid: cell size `d`, angle `a`, origin `o`.
const ROT: (f64, f64, [f64; 2]) = (1.7, 0.52, [3.0, -2.0]);

fn rot(i: f64, j: f64) -> [f64; 2] {
    let (d, a, o) = ROT;
    let (ca, sa) = (a.cos(), a.sin());
    [o[0] + d * (i * ca - j * sa), o[1] + d * (i * sa + j * ca)]
}

fn unrot(p: [f64; 2]) -> [f64; 2] {
    let (d, a, o) = ROT;
    let (ca, sa) = (a.cos(), a.sin());
    let (x, y) = ((p[0] - o[0]) / d, (p[1] - o[1]) / d);
    [x * ca + y * sa, -x * sa + y * ca]
}

fn mesh(c: Corners) -> QuadMesh {
    let (nc, nr, x, y) = c;
    QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap()
}

fn meshes() -> Vec<(&'static str, QuadMesh)> {
    vec![
        ("rotated", mesh(lattice(12, 9, rot))),
        ("jitter", mesh(jittered(7, 0.2))),
        ("jitter_strong", mesh(jittered(11, 0.45))),
        ("polar", mesh(polar())),
        ("sinusoid", mesh(sinusoid())),
        ("dented", mesh(dented())),
        ("jitter_flipped", mesh(flipped(jittered(7, 0.2)))),
        ("polar_flipped", mesh(flipped(polar()))),
    ]
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

/// A star-shaped ring around `c`, some vertices pulled onto `snap`
/// points (mesh corners and split centres) to steer it through
/// degenerate positions.
fn star(g: &mut Lcg, c: [f64; 2], r: f64, n: usize, snap: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let a = (k as f64 + 0.3 * g.u()) / n as f64 * std::f64::consts::TAU;
        let rr = r * (0.35 + 0.65 * g.u());
        let mut p = [c[0] + rr * a.cos(), c[1] + rr * a.sin()];
        if !snap.is_empty() && g.u() < 0.3 {
            // nearest snap point, if close
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
    // drop repeats made by snapping
    out.dedup();
    out
}

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

/// World-clip oracle: each cell's covered fraction of its world area for
/// a polygon (first ring exterior, the rest holes), clipped against the
/// cell's four split triangles. `None` for cells whose split is inexact.
fn oracle(m: &QuadMesh, rings: &[Vec<[f64; 2]>]) -> Vec<Option<f64>> {
    let inexact = m.inexact_cells();
    let mut out = vec![Some(0.0); m.nrow() * m.ncol()];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            if inexact.contains(&(r, c)) {
                out[r * m.ncol() + c] = None;
                continue;
            }
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
                    let s = shoelace(ring).signum();
                    let part = s * shoelace(&clip(ring, &tri));
                    cov += if i == 0 { part } else { -part };
                }
            }
            out[r * m.ncol() + c] = Some(cov / a);
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

fn check(name: &str, m: &QuadMesh, rings: &[Vec<[f64; 2]>], tol: f64) {
    let res = m
        .burn(&[polygon(rings)], BurnOptions::coverage())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let got = dense(&res, m.ncol(), m.nrow());
    let want = oracle(m, rings);
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        if let Some(w) = w {
            assert!(
                (g - w).abs() <= tol,
                "{name}: cell ({}, {}) burn {g} oracle {w}",
                k / m.ncol(),
                k % m.ncol()
            );
        }
    }
}

/// Polygons around the mesh: inside, across the boundary, with holes,
/// steered onto corners and centres.
fn cases(m: &QuadMesh, seed: u64, n: usize) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut g = Lcg(seed);
    let b = bbox(m);
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    let sn = snaps(m);
    let mut out = Vec::new();
    for k in 0..n {
        let c = [
            b[0] + w * (-0.1 + 1.2 * g.u()),
            b[1] + h * (-0.1 + 1.2 * g.u()),
        ];
        let r = (0.08 + 0.4 * g.u()) * w.min(h);
        let nv = 5 + (g.u() * 14.0) as usize;
        let ext = star(&mut g, c, r, nv, if k % 2 == 0 { &sn } else { &[] });
        if ext.len() < 3 {
            continue;
        }
        let mut rings = vec![ext];
        if k % 3 == 0 {
            let hole: Vec<[f64; 2]> = (0..6)
                .rev()
                .map(|i| {
                    let a = i as f64 / 6.0 * std::f64::consts::TAU;
                    [c[0] + 0.25 * r * a.cos(), c[1] + 0.25 * r * a.sin()]
                })
                .collect();
            rings.push(hole);
        }
        out.push(rings);
    }
    out
}

#[test]
fn smooth_warps_match_the_world_clip() {
    for (name, m) in meshes() {
        for (i, rings) in cases(&m, 20261101, 60).iter().enumerate() {
            check(&format!("{name} case {i}"), &m, rings, 2e-6);
        }
    }
}

#[test]
fn rotated_grid_matches_the_affine_burn() {
    let m = mesh(lattice(12, 9, rot));
    let grid = GridSpec::new(0.0, 0.0, 12.0, 9.0, 12, 9);
    for (i, rings) in cases(&m, 7, 80).iter().enumerate() {
        let got = dense(
            &m.burn(&[polygon(rings)], BurnOptions::coverage()).unwrap(),
            12,
            9,
        );
        // the same polygon taken into index space by the inverse affine map
        let idx: Vec<Vec<[f64; 2]>> = rings
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&p| {
                        let q = unrot(p);
                        [q[0], 9.0 - q[1]]
                    })
                    .collect()
            })
            .collect();
        let want = dense(
            &controlledburn::burn(&[polygon(&idx)], &grid, BurnOptions::coverage()).unwrap(),
            12,
            9,
        );
        for k in 0..got.len() {
            assert!(
                (got[k] - want[k]).abs() <= 2e-6,
                "case {i}: cell {k}: walk {} affine {}",
                got[k],
                want[k]
            );
        }
    }
}

#[test]
fn rectilinear_front_end_matches_the_mesh_and_the_clip() {
    let xs = vec![0.0, 0.4, 1.5, 1.7, 3.0, 4.4, 4.5, 6.0];
    let ys = vec![5.0, 4.2, 3.0, 2.9, 1.0, 0.0];
    let rect = Rectilinear::new(xs.clone(), ys.clone()).unwrap();
    let (nc, nr) = (rect.ncol(), rect.nrow());
    let m = mesh(lattice(nc, nr, |i, j| [xs[i as usize], ys[j as usize]]));
    for (i, rings) in cases(&m, 99, 80).iter().enumerate() {
        let g = [polygon(rings)];
        let a = dense(&rect.burn(&g, BurnOptions::coverage()).unwrap(), nc, nr);
        let b = dense(&m.burn(&g, BurnOptions::coverage()).unwrap(), nc, nr);
        let w = oracle(&m, rings);
        for k in 0..a.len() {
            assert!(
                (a[k] - b[k]).abs() <= 2e-6,
                "case {i}: cell {k}: rect {} mesh {}",
                a[k],
                b[k]
            );
            assert!(
                (a[k] - w[k].unwrap()).abs() <= 2e-6,
                "case {i}: cell {k}: rect {} clip {:?}",
                a[k],
                w[k]
            );
        }
    }
}

#[test]
fn rings_around_and_beside_the_mesh() {
    let m = mesh(polar());
    let b = bbox(&m);
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let big = (b[2] - b[0]).max(b[3] - b[1]);
    // a ring containing the whole mesh: every cell full
    let around = vec![
        [cx - big, cy - big],
        [cx + big, cy - big],
        [cx + big, cy + big],
        [cx - big, cy + big],
    ];
    let d = dense(
        &m.burn(
            &[polygon(std::slice::from_ref(&around))],
            BurnOptions::coverage(),
        )
        .unwrap(),
        m.ncol(),
        m.nrow(),
    );
    assert!(d.iter().all(|&v| (v - 1.0).abs() < 1e-6), "{d:?}");
    // the same ring as a hole in a bigger one: nothing
    let outer = vec![
        [cx - 2.0 * big, cy - 2.0 * big],
        [cx + 2.0 * big, cy - 2.0 * big],
        [cx + 2.0 * big, cy + 2.0 * big],
        [cx - 2.0 * big, cy + 2.0 * big],
    ];
    let mut hole = around.clone();
    hole.reverse();
    let d = dense(
        &m.burn(&[polygon(&[outer, hole])], BurnOptions::coverage())
            .unwrap(),
        m.ncol(),
        m.nrow(),
    );
    assert!(d.iter().all(|&v| v.abs() < 1e-6));
    // a ring beside the mesh: nothing, and an empty index ring
    let beside = vec![
        [b[2] + 10.0, b[1]],
        [b[2] + 20.0, b[1]],
        [b[2] + 20.0, b[3]],
    ];
    assert!(m.ring_to_index(&beside).unwrap().is_empty());
    // a C that wraps around the outside of the sector and dips in at both
    // ends: excursions whose boundary arcs must go the short way
    let c = vec![
        [0.0, 115.0],
        [-60.0, 0.0],
        [0.0, -60.0],
        [250.0, -60.0],
        [250.0, 250.0],
        [0.0, 250.0],
        [0.0, 200.0],
        [200.0, 200.0],
        [200.0, -20.0],
        [-20.0, -20.0],
        [-20.0, 0.0],
        [20.0, 115.0],
    ];
    check("C", &m, &[c], 2e-6);
}

#[test]
fn steering_through_corners_centres_and_along_sides() {
    let m = mesh(jittered(3, 0.2));
    let q = |r: usize, c: usize| m.corners(r, c);
    let x = |r: usize, c: usize| m.split_centre(r, c).0;
    // along sides (a cell outline), through corners and centres
    let outline = q(2, 3).to_vec();
    check("outline", &m, &[outline], 2e-6);
    let diag = vec![
        q(1, 1)[0],
        x(1, 1),
        q(1, 1)[2],
        q(3, 4)[2],
        x(3, 2),
        q(3, 1)[3],
    ];
    check("diag", &m, &[diag], 2e-6);
    // a block of cells along their sides
    let mut ring = vec![q(1, 2)[0]];
    for c in 2..=3 {
        ring.push(q(1, c)[1]);
    }
    for r in 1..=4 {
        ring.push(q(r, 3)[2]);
    }
    for c in (2..=3).rev() {
        ring.push(q(4, c)[3]);
    }
    for r in (2..=4).rev() {
        ring.push(q(r, 2)[0]);
    }
    ring.dedup();
    let res = dense(
        &m.burn(&[polygon(&[ring.clone()])], BurnOptions::coverage())
            .unwrap(),
        m.ncol(),
        m.nrow(),
    );
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let want = if (1..=4).contains(&r) && (2..=3).contains(&c) {
                1.0
            } else {
                0.0
            };
            assert!(
                (res[r * m.ncol() + c] - want).abs() < 1e-6,
                "block ({r}, {c}) = {}",
                res[r * m.ncol() + c]
            );
        }
    }
}

#[test]
fn neighbouring_polygons_share_their_crossings() {
    // two polygons sharing an edge: their coverages add up to their union's
    let m = mesh(sinusoid());
    let a = vec![[1.3, 1.1], [6.2, 1.7], [5.1, 6.3], [2.2, 5.5]];
    let b = vec![[6.2, 1.7], [9.1, 2.4], [8.8, 7.0], [5.1, 6.3]];
    let u = vec![
        [1.3, 1.1],
        [6.2, 1.7],
        [9.1, 2.4],
        [8.8, 7.0],
        [5.1, 6.3],
        [2.2, 5.5],
    ];
    let res = m
        .burn(
            &[polygon(&[a]), polygon(&[b]), polygon(&[u])],
            BurnOptions::coverage(),
        )
        .unwrap();
    let mut by = vec![vec![0.0f64; m.ncol() * m.nrow()]; 3];
    for run in &res.runs {
        for c in run.col_start..run.col_end {
            by[run.id as usize][run.row as usize * m.ncol() + c as usize] += 1.0;
        }
    }
    for e in &res.edges {
        by[e.id as usize][e.row as usize * m.ncol() + e.col as usize] += e.fraction as f64;
    }
    for (k, ((a, b), u)) in by[0].iter().zip(&by[1]).zip(&by[2]).enumerate() {
        assert!((a + b - u).abs() < 2e-6, "cell {k}");
    }
}

#[test]
fn cover_ring_matches_burn() {
    let m = mesh(polar());
    let mut win = CoverageWindow::default();
    for (i, rings) in cases(&m, 5, 40).iter().enumerate() {
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
                "case {i}: cell {k}: burn {} window {}",
                d[k],
                w[k]
            );
        }
    }
}

#[test]
fn collapsed_cells_take_no_coverage() {
    // corner (2, 2) moved onto corner (2, 3): cell (1, 2) and (2, 2) become triangles
    let (nc, nr, mut x, mut y) = lattice(5, 4, |i, j| [i, j]);
    let w = nc + 1;
    x[2 * w + 2] = x[2 * w + 3];
    y[2 * w + 2] = y[2 * w + 3];
    // and a fully collapsed cell: (0, 4) with its far corners merged onto the near ones
    let m = QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap();
    let ring = vec![[0.5, 0.5], [4.5, 0.6], [4.4, 3.5], [0.6, 3.4]];
    check("collapsed", &m, &[ring], 2e-6);
}

#[test]
fn points_and_refusals() {
    let m = mesh(jittered(7, 0.2));
    let p = m.split_centre(3, 4).0;
    let res = m
        .burn(
            &[
                Geometry::Point(Coord { x: p[0], y: p[1] }),
                Geometry::Point(Coord { x: 1e6, y: 0.0 }),
            ],
            BurnOptions::coverage(),
        )
        .unwrap();
    assert_eq!(res.points.len(), 1);
    assert_eq!(
        (res.points[0].row, res.points[0].col, res.points[0].id),
        (3, 4, 0)
    );
    let line = Geometry::LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]);
    assert!(m.burn(&[line], BurnOptions::coverage()).is_err());
    assert!(m.burn(&[], BurnOptions::approx()).is_err());
    let (nc, nr, x, y) = jittered(7, 0.2);
    let bil = QuadMesh::from_corners(nc, nr, x, y, CellMap::Bilinear).unwrap();
    assert!(bil
        .ring_to_index(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        .is_err());
    assert!(Rectilinear::new(vec![0.0, 1.0, 1.0], vec![0.0, 1.0]).is_err());
}

#[test]
fn multipolygon_parts_merge_per_cell() {
    // controlledburn 0.3: a multi-part geometry gets one record per cell
    // (`merge_parts`, on by default); `BurnOptions::parity` keeps one per part
    let m = mesh(lattice(4, 3, |i, j| [i, j]));
    let sq = |x0: f64, x1: f64| polygon(&[vec![[x0, 1.2], [x1, 1.2], [x1, 1.8], [x0, 1.8]]]);
    let parts: Vec<Polygon> = [sq(1.1, 1.4), sq(1.6, 1.9)]
        .into_iter()
        .map(|g| match g {
            Geometry::Polygon(p) => p,
            _ => unreachable!(),
        })
        .collect();
    let g = [Geometry::MultiPolygon(parts)];
    let merged = m.burn(&g, BurnOptions::coverage()).unwrap();
    let split = m
        .burn(&g, BurnOptions::parity(controlledburn::BurnMode::Coverage))
        .unwrap();
    assert_eq!((merged.edges.len(), split.edges.len()), (1, 2));
    assert!((merged.edges[0].fraction as f64 - 0.36).abs() < 1e-6);
    assert_eq!(dense(&merged, 4, 3), dense(&split, 4, 3));
}

/// Many polygons (enough vertices to split across threads), multi-parts
/// and points, in mesh order.
fn many(m: &QuadMesh, seed: u64) -> Vec<Geometry> {
    let polys: Vec<Geometry> = cases(m, seed, 3000).iter().map(|r| polygon(r)).collect();
    let mut out = Vec::new();
    for (k, p) in polys.into_iter().enumerate() {
        if k % 97 == 0 {
            let p0 = m.split_centre(k % m.nrow(), k % m.ncol()).0;
            out.push(Geometry::Point(Coord { x: p0[0], y: p0[1] }));
        }
        match (k % 50, out.pop()) {
            (0, Some(Geometry::Polygon(a))) => match p {
                Geometry::Polygon(b) => out.push(Geometry::MultiPolygon(vec![a, b])),
                _ => unreachable!(),
            },
            (_, Some(prev)) => {
                out.push(prev);
                out.push(p);
            }
            (_, None) => out.push(p),
        }
    }
    out
}

#[test]
fn burn_par_matches_burn() {
    for (name, m) in [("polar", mesh(polar())), ("jitter", mesh(jittered(7, 0.2)))] {
        let g = many(&m, 2026);
        let nv: usize = g
            .iter()
            .map(|g| match g {
                Geometry::Polygon(p) => p.rings.iter().map(Vec::len).sum(),
                Geometry::MultiPolygon(ps) => ps.iter().flat_map(|p| &p.rings).map(Vec::len).sum(),
                _ => 1,
            })
            .sum();
        assert!(nv > 3 * 8192, "{name}: {nv} vertices is too few to split");
        let one = m.burn(&g, BurnOptions::coverage()).unwrap();
        assert!(!one.points.is_empty() && !one.edges.is_empty());
        for threads in [0, 2, 3, 8] {
            assert_eq!(
                m.burn_par(&g, BurnOptions::coverage(), threads).unwrap(),
                one,
                "{name} at {threads}"
            );
        }
        let parity = BurnOptions::parity(controlledburn::BurnMode::Coverage);
        assert_eq!(
            m.burn_par(&g, parity, 4).unwrap(),
            m.burn(&g, parity).unwrap(),
            "{name} parity"
        );
    }
    let xs: Vec<f64> = (0..=40)
        .map(|i| i as f64 * 0.25 + (i as f64 * 0.7).sin() * 0.05)
        .collect();
    let ys: Vec<f64> = (0..=30).rev().map(|j| j as f64 * 0.2).collect();
    let rect = Rectilinear::new(xs.clone(), ys.clone()).unwrap();
    let m = mesh(lattice(rect.ncol(), rect.nrow(), |i, j| {
        [xs[i as usize], ys[j as usize]]
    }));
    let g = many(&m, 7);
    assert_eq!(
        rect.burn_par(&g, BurnOptions::coverage(), 4).unwrap(),
        rect.burn(&g, BurnOptions::coverage()).unwrap()
    );
}

#[test]
fn burn_par_reports_the_first_error() {
    let m = mesh(polar());
    let mut g = many(&m, 3);
    g.insert(
        g.len() * 2 / 3,
        Geometry::LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]),
    );
    assert_eq!(
        m.burn_par(&g, BurnOptions::coverage(), 4).unwrap_err(),
        m.burn(&g, BurnOptions::coverage()).unwrap_err()
    );
}
