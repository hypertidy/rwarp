//! Meshes from each cell's own four vertices (CF bounds, SCRIP corners):
//! halos dropped, the vertex order found, shared corners checked, and
//! overlapping cells refused (decision N3).
//!
//! The ORCA-like grids here are NEMO's layout as CMIP6 files have it
//! (checked on IPSL's eORCA1 and ORCA2): the last two columns repeat the
//! first two, the first row is cells collapsed to points, and the last
//! row repeats one below it, mirrored; at a T-point pivot the row before
//! it mirrors itself. Coverage is checked against the world clip of each
//! cell's split triangles, as in `tests/mesh_glued.rs`.

use controlledburn::{BurnOptions, BurnResult, Coord, Geometry, Polygon};
use rwarp::mesh::{Band, CellMap, Halo, MeshSet, QuadMesh};
use rwarp::CrsTransform;
use std::f64::consts::TAU;

/// Cell bounds, row-major with the vertex last (CF, SCRIP).
struct Bounds {
    ncol: usize,
    nrow: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Bounds {
    /// From each cell's vertices, in CF's order.
    fn from_cells(ncol: usize, nrow: usize, f: impl Fn(usize, usize) -> [[f64; 2]; 4]) -> Bounds {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for j in 0..nrow {
            for i in 0..ncol {
                for p in f(i, j) {
                    x.push(p[0]);
                    y.push(p[1]);
                }
            }
        }
        Bounds { ncol, nrow, x, y }
    }

    /// From a corner function on column line `s`, row line `j`.
    fn from_corners(ncol: usize, nrow: usize, f: impl Fn(usize, usize) -> [f64; 2]) -> Bounds {
        Self::from_cells(ncol, nrow, |i, j| {
            [f(i, j), f(i + 1, j), f(i + 1, j + 1), f(i, j + 1)]
        })
    }

    fn mesh(&self, halo: Halo) -> Result<QuadMesh, String> {
        QuadMesh::from_cf_bounds(
            self.ncol,
            self.nrow,
            &self.x,
            &self.y,
            halo,
            CellMap::EqualArea,
        )
    }
}

/// An ORCA-like grid, halo and all: `nc` columns plus the cyclic halo, and
/// `ny` rows counting the collapsed first row and the north-fold halo row.
/// The shape is the tripolar disk of `tests/mesh_glued.rs`: row lines from
/// a circle of radius 200 in to the fold, the segment from (60, 0) to
/// (-60, 0), and on past it by the half-turn about the pivots. The fold is
/// the top side of row `ny - 2` (an F-point pivot), or with `t_pivot` runs
/// through that row's centres. The pivots are where NEMO has them: column
/// line 1 (F), or column 1's centre (T), and half way round.
fn orca(nc: usize, ny: usize, t_pivot: bool) -> Bounds {
    let p0 = if t_pivot { 1.5 } else { 1.0 };
    let h = 1.0 / (ny as f64 - if t_pivot { 2.5 } else { 2.0 });
    let corner = |s: usize, j: usize| -> [f64; 2] {
        let a = (s as f64 - p0) / nc as f64 * TAU;
        let tau = 1.0 - (j as f64 - 1.0) * h;
        [
            (tau.abs() * 200.0 + (1.0 - tau.abs()) * 60.0) * a.cos(),
            tau * 200.0 * a.sin(),
        ]
    };
    Bounds::from_cells(nc + 2, ny, |i, j| {
        if j == 0 {
            // the first row: each cell collapsed to a point beyond the disk
            let q = corner(i, 1);
            [[q[0] * 1.1, q[1] * 1.1]; 4]
        } else {
            [
                corner(i, j),
                corner(i + 1, j),
                corner(i + 1, j + 1),
                corner(i, j + 1),
            ]
        }
    })
}

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
        if m.cell_area(k / m.ncol(), k % m.ncol()) == 0.0 {
            // no area: its fraction weighs nothing
            continue;
        }
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

/// Random polygons near the seam, the fold and anywhere over the disk.
fn cases(m: &QuadMesh, seed: u64, n: usize) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut g = Lcg(seed);
    let sn = snaps(m);
    let mut out = Vec::new();
    for k in 0..n {
        let c = match k % 3 {
            0 => {
                // the seam: column line 0, at angle -1/nc of a turn or so
                let (r, a) = (60.0 + 160.0 * g.u(), -0.05 * TAU * g.u());
                [r * a.cos(), r * a.sin()]
            }
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

fn mesh_area(m: &QuadMesh) -> f64 {
    (0..m.nrow())
        .flat_map(|r| (0..m.ncol()).map(move |c| (r, c)))
        .map(|(r, c)| m.cell_area(r, c))
        .sum()
}

#[test]
fn orca_grids_with_their_halos_match_the_world_clip() {
    let (nc, ny) = (24, 10);
    for t_pivot in [false, true] {
        let b = orca(nc, ny, t_pivot);
        // NEMO's halo as given, and the cyclic halo split one each side
        // (which puts a T pivot's collapsed cell on the seam)
        for cols in [[0, 2], [1, 1]] {
            let name = format!("t_pivot {t_pivot}, cols {cols:?}");
            let halo = Halo {
                cols,
                rows: [1, 1],
                fold_through_centres: t_pivot,
            };
            let m = b.mesh(halo).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((m.ncol(), m.nrow()), (nc, ny - 2), "{name}");
            assert_eq!(m.find_overlap(), None, "{name}");
            // the disk inside row line 1 (a 24-gon), tiled once
            let rim: Vec<[f64; 2]> = (0..nc).map(|c| m.corners(0, c)[0]).collect();
            let rim_area = shoelace(&rim).abs();
            assert!((mesh_area(&m) - rim_area).abs() < 1e-9 * rim_area, "{name}");
            let mut n = 0;
            for (i, rings) in cases(&m, 20261117, 150).iter().enumerate() {
                if !simple(rings) {
                    continue;
                }
                check(&format!("{name} case {i}"), &m, rings);
                n += 1;
            }
            assert!(n > 100, "{name}: {n} cases");
            // across the fold and round both poles
            check(
                &format!("{name} round the fold"),
                &m,
                &[circle([0.0, 0.0], 100.0, 37)],
            );
            check(
                &format!("{name} the whole disk"),
                &m,
                &[circle([0.0, 0.0], 250.0, 37)],
            );
        }
    }
}

#[test]
fn a_t_pivot_row_is_its_cells_halved() {
    let (nc, ny) = (24, 10);
    let b = orca(nc, ny, true);
    let m = b.mesh(Halo::NEMO_T_PIVOT).unwrap();
    let last = m.nrow() - 1;
    // the file's cells of that row, whole: each is two mesh cells
    let at = |c: usize| {
        let k = ((last + 1) * b.ncol + c) * 4;
        let q: Vec<[f64; 2]> = (0..4).map(|v| [b.x[k + v], b.y[k + v]]).collect();
        shoelace(&q).abs()
    };
    for c in 0..nc {
        let mirror = (nc + 2 - c) % nc;
        let both = m.cell_area(last, c) + m.cell_area(last, mirror);
        assert!(
            (both - at(c)).abs() < 1e-9 * at(c).max(1.0),
            "col {c}: {both} vs {}",
            at(c)
        );
        if c == 1 || c == nc / 2 + 1 {
            // the pivots: collapsed in the file, and in the mesh
            assert_eq!(m.cell_area(last, c), 0.0, "pivot {c}");
        }
    }
}

#[test]
fn halos_left_in_are_refused() {
    let (nc, ny) = (24, 10);
    for t_pivot in [false, true] {
        let b = orca(nc, ny, t_pivot);
        let base = if t_pivot {
            Halo::NEMO_T_PIVOT
        } else {
            Halo::NEMO_F_PIVOT
        };
        // the collapsed first row: its cells share no corners
        let e = b
            .mesh(Halo {
                rows: [0, 1],
                ..base
            })
            .unwrap_err();
        assert!(
            e.contains("don't share a corner") && e.contains("Halo"),
            "{e}"
        );
        // the cyclic columns
        let e = b
            .mesh(Halo {
                cols: [0, 0],
                ..base
            })
            .unwrap_err();
        assert!(e.contains("overlap") && e.contains("Halo"), "{e}");
        let e = b
            .mesh(Halo {
                cols: [0, 1],
                ..base
            })
            .unwrap_err();
        assert!(e.contains("overlap"), "{e}");
        // the north-fold row (at a T pivot it isn't the row that mirrors
        // itself, and the row that is lies under it)
        let e = b
            .mesh(Halo {
                rows: [1, 0],
                ..base
            })
            .unwrap_err();
        let want = if t_pivot {
            "doesn't mirror itself"
        } else {
            "overlap"
        };
        assert!(e.contains(want), "{e}");
    }
    // a T pivot taken as an F pivot: the last row mirrors itself
    let e = orca(nc, ny, true).mesh(Halo::NEMO_F_PIVOT).unwrap_err();
    assert!(e.contains("T-point pivot"), "{e}");
    // and the other way round: halving a row that doesn't mirror itself
    // would drop the top half of each of its cells
    let e = orca(nc, ny, false).mesh(Halo::NEMO_T_PIVOT).unwrap_err();
    assert!(
        e.contains("doesn't mirror itself") && e.contains("F-point pivot"),
        "{e}"
    );
}

/// A sheared, bent grid with nothing symmetric about it.
fn warped() -> Bounds {
    Bounds::from_corners(7, 5, |s, j| {
        let (s, j) = (s as f64, j as f64);
        [
            10.0 + 1.3 * s + 0.2 * j + 0.05 * j * j,
            -4.0 + 0.9 * j + 0.1 * (s * 0.7).sin(),
        ]
    })
}

#[test]
fn any_vertex_order_gives_the_same_mesh() {
    let b = warped();
    let want = b.mesh(Halo::NONE).unwrap();
    // the same corners from_corners takes
    let (nc, nr) = (b.ncol, b.nrow);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nr {
        for i in 0..=nc {
            let (jj, ii, v) = match (j < nr, i < nc) {
                (true, true) => (j, i, 0),
                (true, false) => (j, nc - 1, 1),
                (false, true) => (nr - 1, i, 3),
                (false, false) => (nr - 1, nc - 1, 2),
            };
            let k = (jj * nc + ii) * 4 + v;
            x.push(b.x[k]);
            y.push(b.y[k]);
        }
    }
    let direct = QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap();
    let same = |m: &QuadMesh, name: &str| {
        for r in 0..nr {
            for c in 0..nc {
                assert_eq!(
                    m.corners(r, c),
                    direct.corners(r, c),
                    "{name}: cell ({r}, {c})"
                );
            }
        }
    };
    same(&want, "CF");
    // every rotation and reflection of the four vertices
    for rot in 0..4 {
        for flip in [false, true] {
            let perm: [usize; 4] = std::array::from_fn(|s| {
                if flip {
                    (rot + 4 - s) % 4
                } else {
                    (s + rot) % 4
                }
            });
            let mut p = Bounds {
                ncol: nc,
                nrow: nr,
                x: b.x.clone(),
                y: b.y.clone(),
            };
            for cell in 0..nc * nr {
                for (s, &v) in perm.iter().enumerate() {
                    // stored vertex v holds CF slot s
                    p.x[cell * 4 + v] = b.x[cell * 4 + s];
                    p.y[cell * 4 + v] = b.y[cell * 4 + s];
                }
            }
            same(
                &p.mesh(Halo::NONE).unwrap(),
                &format!("rot {rot} flip {flip}"),
            );
        }
    }
}

#[test]
fn shared_corners_agree_to_rounding_or_are_refused() {
    let b = warped();
    // cell (2, 3)'s copy of its corner (3, 4) (vertex 2), shared with cells
    // (2, 4), (3, 3) and (3, 4)
    let k = (2 * b.ncol + 3) * 4 + 2;
    let mut near = Bounds {
        ncol: b.ncol,
        nrow: b.nrow,
        x: b.x.clone(),
        y: b.y.clone(),
    };
    near.x[k] += 1e-6;
    let m = near.mesh(Halo::NONE).unwrap();
    // the copy of cell (3, 4), whose first vertex it is, is used
    assert_eq!(
        m.corners(2, 3)[2],
        [b.x[(3 * b.ncol + 4) * 4], b.y[(3 * b.ncol + 4) * 4]]
    );
    let mut far = near;
    far.x[k] += 0.1;
    let e = far.mesh(Halo::NONE).unwrap_err();
    assert!(
        e.contains("cells (2, 3) and (2, 4) don't share a corner"),
        "{e}"
    );
}

#[test]
fn longitudes_a_period_apart_are_named() {
    // a 10 degree grid round the globe, its longitudes wrapped to [0, 360)
    // part way through: the seam is inside the mesh
    let b = Bounds::from_cells(12, 3, |i, j| {
        let lon = 300.0 + 10.0 * i as f64;
        let w = if lon >= 360.0 { lon - 360.0 } else { lon };
        let (s, n) = (-15.0 + 10.0 * j as f64, -5.0 + 10.0 * j as f64);
        [[w, s], [w + 10.0, s], [w + 10.0, n], [w, n]]
    });
    let e = b.mesh(Halo::NONE).unwrap_err();
    assert!(e.contains("period apart"), "{e}");
}

#[test]
fn find_overlap_sees_copies_that_keep_orientation() {
    // an annulus whose first two columns are stored again at the end
    let (nc, nr) = (26, 4);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nr {
        for i in 0..=nc {
            let a = ((i % 24) as f64 / 24.0) * TAU;
            let r = 100.0 + 20.0 * j as f64;
            x.push(r * a.cos());
            y.push(r * a.sin());
        }
    }
    let m = QuadMesh::from_corners(nc, nr, x.clone(), y.clone(), CellMap::EqualArea).unwrap();
    let [a, b] = m.find_overlap().unwrap();
    assert_eq!(a.0, b.0);
    assert_eq!((a.1 % 24, b.1 % 24), (a.1 % 24, a.1 % 24), "{a:?} {b:?}");
    // the annulus alone has none
    let keep = |v: &[f64]| {
        (0..=nr)
            .flat_map(|j| v[j * (nc + 1)..j * (nc + 1) + 25].to_vec())
            .collect::<Vec<_>>()
    };
    let m = QuadMesh::from_corners(24, nr, keep(&x), keep(&y), CellMap::EqualArea).unwrap();
    assert_eq!(m.find_overlap(), None);
}

#[test]
fn rings_round_the_pivots() {
    // the fold's ends, (60, 0) and (-60, 0): a T pivot's collapsed cells
    // are there, one of them on the seam when the halo is split one
    // column each side
    let (nc, ny) = (24, 10);
    for t_pivot in [false, true] {
        let b = orca(nc, ny, t_pivot);
        for cols in [[0, 2], [1, 1]] {
            let name = format!("t_pivot {t_pivot}, cols {cols:?}");
            let m = b
                .mesh(Halo {
                    cols,
                    rows: [1, 1],
                    fold_through_centres: t_pivot,
                })
                .unwrap();
            let sn = snaps(&m);
            let mut g = Lcg(7);
            let mut n = 0;
            for k in 0..400 {
                let pole = if k % 2 == 0 { 60.0 } else { -60.0 };
                let c = [pole + 12.0 * (g.u() - 0.5), 12.0 * (g.u() - 0.5)];
                let r = 2.0 + 25.0 * g.u();
                let ext = star(&mut g, c, r, 5 + k % 9, if k % 3 == 0 { &sn } else { &[] });
                let rings = vec![ext];
                if rings[0].len() < 3 || !simple(&rings) {
                    continue;
                }
                check(&format!("{name} pivot case {k}"), &m, &rings);
                n += 1;
            }
            assert!(n > 300, "{name}: {n}");
        }
    }
}

/// The ORCA-like grid as a set of one band in its own plane.
fn orca_set(b: &Bounds, t_pivot: bool) -> MeshSet {
    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }
    let halo = if t_pivot { Halo::NEMO_T_PIVOT } else { Halo::NEMO_F_PIVOT };
    let bands = [Band { rows: 0..b.nrow - 2, to_world: &Same }];
    MeshSet::from_cf_bounds(b.ncol, b.nrow, &b.x, &b.y, halo, &bands, None).unwrap()
}

/// A field on the file's grid into the set's numbering and back, with
/// NEMO's halo filled: every cell but the first row's comes back. The
/// field is each cell's centre, which a halo cell shares with the cell it
/// copies (at a T pivot, the two cells that halve one model cell do too).
#[test]
fn fields_go_to_the_set_and_back_to_the_file() {
    for t_pivot in [false, true] {
        let (nc, ny) = (24, 10);
        let b = orca(nc, ny, t_pivot);
        let s = orca_set(&b, t_pivot);
        assert_eq!((s.size(), s.file_size(), s.halo().fold_through_centres), ([nc, ny - 2], [nc + 2, ny], t_pivot));
        let nx = b.ncol;
        let centre = |j: usize, i: usize| {
            let k = (j * nx + i) * 4;
            [(0..4).map(|v| b.x[k + v]).sum::<f64>() / 4.0, (0..4).map(|v| b.y[k + v]).sum::<f64>() / 4.0]
        };
        let file: Vec<f64> = (0..ny).flat_map(|j| (0..nx).flat_map(move |i| centre(j, i))).collect();
        let field = s.from_file_layout(&file, 2).unwrap();
        assert_eq!(field.len(), nc * (ny - 2) * 2);
        // the set's (row, col) is the file's (row + 1, col)
        assert_eq!(&field[..2], &file[nx * 2..nx * 2 + 2]);
        let mut back = s.to_file_layout(&field, 2, f64::NAN).unwrap();
        // the halo masked: the first row, the last two columns, the last row
        let halo = |j: usize, i: usize| j == 0 || j == ny - 1 || i >= nc;
        for j in 0..ny {
            for i in 0..nx {
                let k = (j * nx + i) * 2;
                if halo(j, i) {
                    assert!(back[k].is_nan() && back[k + 1].is_nan(), "({j}, {i}) masked");
                } else {
                    assert!((back[k] - file[k]).abs() < 1e-9 && (back[k + 1] - file[k + 1]).abs() < 1e-9, "({j}, {i}): {:?} vs {:?}", &back[k..k + 2], &file[k..k + 2]);
                }
            }
        }
        s.fill_halo(&mut back, 2).unwrap();
        let worst = (2 * nx..file.len()).map(|k| (back[k] - file[k]).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-9, "t_pivot {t_pivot}: the halo filled to {worst:e}");
        assert!(back[..2 * nx].iter().all(|v| v.is_nan()), "the first row is left");
    }
}

/// At a T pivot, the two cells of the set that halve one model cell give
/// it their mean by area, in both of the file's copies of it; a half that
/// is the fill, or NaN, is left out.
#[test]
fn a_t_pivot_cell_takes_its_halves_mean_by_area() {
    let (nc, ny) = (24, 10);
    let b = orca(nc, ny, true);
    let s = orca_set(&b, true);
    let (nr, nx) = (ny - 2, nc + 2);
    let last = nr - 1;
    // each cell's value its column; one half missing in a pair
    let mut field: Vec<f64> = (0..nc * nr).map(|k| (k % nc) as f64).collect();
    field[last * nc + 5] = -1.0;
    let back = s.to_file_layout(&field, 1, -1.0).unwrap();
    let area = |c: usize| {
        let (p, r, col) = s.locate_cell(last, c).unwrap();
        p.mesh().cell_area(r, col)
    };
    for c in 0..nc {
        let m = (nc + 2 - c) % nc;
        let got = back[(last + 1) * nx + c];
        let want = if m == c {
            c as f64
        } else if c == 5 || m == 5 {
            // only the other half's
            if c == 5 { m as f64 } else { c as f64 }
        } else {
            (area(c) * c as f64 + area(m) * m as f64) / (area(c) + area(m))
        };
        assert!((got - want).abs() < 1e-12, "col {c} (mirror {m}): {got} vs {want}");
        assert_eq!(got.to_bits(), back[(last + 1) * nx + m].to_bits(), "col {c}: both copies");
    }
    // the rows below keep their own
    assert!((0..nc).all(|c| back[nr * nx - nx + c] == field[(nr - 2) * nc + c]));
}

/// The helpers refuse fields of the wrong size, and only NEMO's halos are
/// filled.
#[test]
fn file_layout_refusals() {
    let b = orca(24, 10, false);
    let s = orca_set(&b, false);
    assert!(s.from_file_layout(&[0.0; 10], 1).unwrap_err().contains("26 x 10"));
    assert!(s.to_file_layout(&[0.0f32; 10], 1, 0.0).unwrap_err().contains("24 x 8"));
    let mut wrong = vec![0.0; 10];
    assert!(s.fill_halo(&mut wrong, 1).is_err());
}
