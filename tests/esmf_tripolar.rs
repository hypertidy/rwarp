//! ESMF on a tripolar grid: conservative weights from a [`QuadMesh`] with a
//! seam and a fold, against ESMF's (stage 4 of
//! design/curvilinear-frontend.md, #37).
//!
//! The source is the analytic tripolar cap of `tests/fixtures/esmf/gen.py`
//! (`tripolar.grid.txt`): confocal ellipses about two grid poles, 48 x 16
//! cells, given by its corners in north-polar Lambert azimuthal equal-area
//! on a sphere. Its last row line folds onto itself (an F-point pivot) and
//! column line 48 is column line 0, a seam that runs into one pole. In that
//! plane it is one sheet, as rwarp holds a tripolar grid
//! ([`rwarp::mesh`], "Seams and folds"). Two destinations, each with a
//! Cartesian and a spherical ESMF map:
//!
//! - `tripolar_laea`: 300 km cells in the same plane.
//! - `tripolar_lonlat`: 6 x 3 degree lon/lat cells from 54 N to the pole,
//!   whose top row are triangles with a corner on the pole.
//!
//! Each destination cell's outline goes into the mesh's index space with
//! [`QuadMesh::cover_ring`]; the coverage of a source cell is the fraction
//! of its area the outline covers, and `S = coverage * source area /
//! destination area` (ESMF's `destarea` normalisation). Planar areas in
//! the plane are areas on the sphere, since the projection is equal-area.
//!
//! `cargo test --test esmf_tripolar -- --nocapture` prints the measured
//! differences. No CRS is involved: the projection is written out here.

mod esmf_common;

use controlledburn::CoverageWindow;
use esmf_common::{diff, fixture_path, read_fixture, smooth, Fixture, Ours};
use rwarp::mesh::{CellMap, Halo, QuadMesh};

/// The fixtures' sphere (NEMO's).
const RADIUS: f64 = 6371229.0;

/// North-polar Lambert azimuthal equal-area on the sphere, as in `gen.py`.
fn laea(lon: f64, lat: f64) -> [f64; 2] {
    let rho = 2.0 * RADIUS * ((90.0 - lat).to_radians() / 2.0).sin();
    let lam = lon.to_radians();
    [rho * lam.sin(), -rho * lam.cos()]
}

fn laea_inv(p: [f64; 2]) -> (f64, f64) {
    let rho = p[0].hypot(p[1]);
    (
        p[0].atan2(-p[1]).to_degrees(),
        90.0 - 2.0 * (rho / (2.0 * RADIUS)).asin().to_degrees(),
    )
}

/// The source corners: `(ncol, nrow, x, y)`, row-major over the
/// `(nrow + 1) x (ncol + 1)` corner lattice.
fn read_grid() -> (usize, usize, Vec<f64>, Vec<f64>) {
    let path = fixture_path("tripolar.grid.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let dims: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let (ncol, nrow) = (dims[0], dims[1]);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for l in lines {
        let mut f = l.split_whitespace().map(|v| v.parse::<f64>().unwrap());
        x.push(f.next().unwrap());
        y.push(f.next().unwrap());
    }
    assert_eq!(x.len(), (ncol + 1) * (nrow + 1), "truncated grid");
    (ncol, nrow, x, y)
}

fn source() -> QuadMesh {
    let (ncol, nrow, x, y) = read_grid();
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap()
}

/// The smooth field at each source cell's corner mean, in ESMF's order.
fn field(m: &QuadMesh) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.ncol() * m.nrow());
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let q = m.corners(r, c);
            let mean = [0, 1].map(|k| q.iter().map(|p| p[k]).sum::<f64>() / 4.0);
            let (lon, lat) = laea_inv(mean);
            v.push(smooth(lon, lat));
        }
    }
    v
}

/// Destination cell `(row, col)`'s outline in the plane, counter-clockwise,
/// each side in `densify` pieces joined straight in the destination grid's
/// coordinates (lon/lat or the plane) and then taken to the plane.
fn outline(fx: &Fixture, row: usize, col: usize, densify: usize) -> Vec<[f64; 2]> {
    let lonlat = match fx.header("dst_kind") {
        "lonlat" => true,
        "laea" => false,
        k => panic!("dst_kind {k}"),
    };
    let g: [f64; 6] = fx.nums("dst_gt");
    let (c, r) = (col as f64, row as f64);
    let corners = [[c, r], [c + 1.0, r], [c + 1.0, r + 1.0], [c, r + 1.0]];
    let mut ring: Vec<[f64; 2]> = Vec::with_capacity(4 * densify + 1);
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        for s in 0..densify {
            let t = s as f64 / densify as f64;
            let (px, py) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
            let (x, y) = (g[0] + px * g[1] + py * g[2], g[3] + px * g[4] + py * g[5]);
            let p = if lonlat { laea(x, y) } else { [x, y] };
            // the pole, twice at the top of a lon/lat grid's last row
            if ring.last() != Some(&p) {
                ring.push(p);
            }
        }
    }
    ring.push(ring[0]);
    ring
}

fn shoelace(ring: &[[f64; 2]]) -> f64 {
    0.5 * ring
        .windows(2)
        .map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1])
        .sum::<f64>()
}

/// rwarp's map from the mesh to the fixture's destination, in ESMF's terms
/// (areas in the plane's square metres).
fn ours(m: &QuadMesh, fx: &Fixture, densify: usize) -> Ours {
    let [dw, dh] = fx.nums::<2>("dst_size").map(|v| v as usize);
    let mut win = CoverageWindow::default();
    let mut o = Ours {
        s: Default::default(),
        frac_b: Vec::with_capacity(dw * dh),
        area_b: Vec::with_capacity(dw * dh),
    };
    for r in 0..dh {
        for c in 0..dw {
            let ring = outline(fx, r, c, densify);
            let area = shoelace(&ring);
            assert!(
                area > 0.0,
                "destination cell ({r}, {c}) is not counter-clockwise"
            );
            let mut covered = 0.0;
            if m.cover_ring(&ring, &mut win).unwrap() {
                let cov = win.coverage();
                for jr in 0..win.nrow() {
                    for jc in 0..win.ncol() {
                        let (sr, sc) = (win.row0() + jr as i64, win.col0() + jc as i64);
                        let v = cov[jr * win.ncol() + jc];
                        if v == 0.0
                            || sr < 0
                            || sc < 0
                            || sr as usize >= m.nrow()
                            || sc as usize >= m.ncol()
                        {
                            continue;
                        }
                        let (sr, sc) = (sr as usize, sc as usize);
                        let a = v * m.cell_area(sr, sc);
                        covered += a;
                        let key = ((r * dw + c + 1) as u64, (sr * m.ncol() + sc + 1) as u64);
                        o.s.insert(key, a / area);
                    }
                }
            }
            o.frac_b.push(covered / area);
            o.area_b.push(area);
        }
    }
    o
}

/// max |S ours - S ESMF| over destination cells that both cover completely.
fn s_inside(fx: &Fixture, o: &Ours) -> f64 {
    let full =
        |k: u64| fx.frac_b[k as usize - 1] > 1.0 - 1e-9 && o.frac_b[k as usize - 1] > 1.0 - 1e-9;
    let mut d: f64 = 0.0;
    for (k, &e) in fx.s.iter().filter(|(k, _)| full(k.0)) {
        d = d.max((o.s.get(k).copied().unwrap_or(0.0) - e).abs());
    }
    for (k, &r) in o.s.iter().filter(|(k, _)| full(k.0)) {
        d = d.max((fx.s.get(k).copied().unwrap_or(0.0) - r).abs());
    }
    d
}

fn max_rel(a: &[f64], b: &[f64], scale: f64) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| ((a / scale - b) / b).abs())
        .fold(0.0, f64::max)
}

const CASES: [&str; 2] = ["tripolar_laea", "tripolar_lonlat"];

/// Cartesian ESMF in the plane has rwarp's cells: source cells straight
/// between their corners, destination cells too (densify 1). Weights,
/// fractions and areas agree to rounding, through the seam, the fold, the
/// two grid poles and the geographic pole.
#[test]
fn cartesian_esmf_matches_on_a_tripolar_grid() {
    let m = source();
    let src = field(&m);
    for name in CASES {
        let fx = read_fixture(&format!("{name}.cart.txt"));
        let o = ours(&m, &fx, 1);
        let d = diff(&fx, &o, &src);
        let area = max_rel(&o.area_b, &fx.area_b, 1.0);
        eprintln!("{name:16} cart: {d:?}, max relative area difference {area:.2e}");
        assert!(
            d.esmf_rows < 1e-9,
            "{name}: ESMF row sums off by {:e}",
            d.esmf_rows
        );
        assert!(d.s < 1e-9, "{name}: S differs by {:e}", d.s);
        assert!(d.frac_b < 1e-9, "{name}: frac_b differs by {:e}", d.frac_b);
        assert!(area < 1e-12, "{name}: area differs by {area:e}");
        assert!(d.field < 1e-9, "{name}: field differs by {:e}", d.field);
        let (seam, fold) = crossings(&m, &o);
        eprintln!("{name:16} destination cells across the seam {seam}, across the fold {fold}");
        assert!(
            seam > 0 && fold > 0,
            "{name}: the case should cross the seam and the fold"
        );
    }
}

/// Destination cells with weights on both sides of the seam (columns 0 and
/// ncol - 1 of one row), and on both sides of the fold away from the poles
/// (last-row cells i and ncol - 1 - i).
fn crossings(m: &QuadMesh, o: &Ours) -> (usize, usize) {
    let (ncol, top) = (m.ncol() as u64, m.nrow() as u64 - 1);
    let mut by_dst: std::collections::HashMap<u64, Vec<(u64, u64)>> = Default::default();
    for &(d, s) in o.s.keys() {
        by_dst
            .entry(d)
            .or_default()
            .push(((s - 1) / ncol, (s - 1) % ncol));
    }
    let (mut seam, mut fold) = (0, 0);
    for cells in by_dst.values() {
        let has = |rc: (u64, u64)| cells.contains(&rc);
        seam += cells.iter().any(|&(r, c)| c == 0 && has((r, ncol - 1))) as usize;
        fold += cells
            .iter()
            .any(|&(r, c)| r == top && (2..ncol / 2 - 2).contains(&c) && has((top, ncol - 1 - c)))
            as usize;
    }
    (seam, fold)
}

/// The NEMO layout of the same grid, as CF bounds with a land row first,
/// the two cyclic columns repeated last and the fold row mirrored last,
/// read with [`Halo::NEMO_F_PIVOT`], is the same mesh to the bit.
#[test]
fn the_nemo_layout_is_the_same_mesh() {
    let m = source();
    let (ncol, nrow) = (m.ncol(), m.nrow());
    let (nc, nr) = (ncol + 2, nrow + 2);
    let (mut xb, mut yb) = (vec![0.0; nc * nr * 4], vec![0.0; nc * nr * 4]);
    for jf in 0..nr {
        for i in 0..nc {
            let ic = i % ncol;
            let v = match jf {
                // land: collapsed onto the first corner below
                0 => [m.corners(0, ic)[0]; 4],
                _ if jf <= nrow => m.corners(jf - 1, ic),
                // the fold row: cell i is row nrow - 1's mirror cell, turned
                _ => {
                    let q = m.corners(nrow - 1, ncol - 1 - ic);
                    [q[2], q[3], q[0], q[1]]
                }
            };
            for (k, p) in v.iter().enumerate() {
                xb[(jf * nc + i) * 4 + k] = p[0];
                yb[(jf * nc + i) * 4 + k] = p[1];
            }
        }
    }
    let e = QuadMesh::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, CellMap::EqualArea).unwrap_err();
    eprintln!("without the layout: {e}");
    let h =
        QuadMesh::from_cf_bounds(nc, nr, &xb, &yb, Halo::NEMO_F_PIVOT, CellMap::EqualArea).unwrap();
    assert_eq!((h.ncol(), h.nrow()), (ncol, nrow));
    for r in 0..nrow {
        for c in 0..ncol {
            assert_eq!(h.corners(r, c), m.corners(r, c), "cell ({r}, {c})");
        }
    }
    let fx = read_fixture("tripolar_laea.cart.txt");
    assert_eq!(ours(&h, &fx, 1).s, ours(&m, &fx, 1).s);
}

/// Spherical ESMF: the same corners in lon/lat, joined by great circles on
/// both sides. rwarp's source cells are straight in the plane, and so are
/// its destination cells for `tripolar_laea`; for `tripolar_lonlat` they
/// are the lon/lat boxes, with sides joined straight in the plane at
/// densify 1 and following the parallels at densify 8. The differences are
/// recorded as ceilings, about 1.5 times what was measured (ESMF 8.9.1),
/// as `[S, frac_b, field, area_b]`: numbers to shrink, not failures.
///
/// They are the cell definition, not arithmetic (the Cartesian test above
/// matches to rounding). Source cells are 13 to 363 km across (the square
/// root of their area; median 183 km), destination cells 300 km or 6 by 3
/// degrees. The largest differences are on the source's outer row, whose
/// edges are the longest and furthest from the pole, so their great
/// circles bow furthest from the straight line in the plane; over
/// destination cells both cover completely the S difference is 1.7e-3 at
/// most at densify 1. On the same grids with cells half the size (not
/// committed: `gen.py` with `ncol=96, nrow=32` and the destination steps
/// halved) area_b differences fall to a quarter (second order in cell
/// size), the field's to about 0.3 of these, and S's to a half: a shift of
/// an edge by h^2 is a share h of a cell of side h. As in `tests/esmf.rs`,
/// densify 8 is further from ESMF than densify 1: ESMF joins the lon/lat
/// corners by great circles, which bow poleward of the parallels that
/// densify 8 follows.
#[test]
fn spherical_esmf_differences_on_a_tripolar_grid_are_recorded() {
    #[rustfmt::skip]
    let ceilings: [(&str, usize, [f64; 4]); 3] = [
        ("tripolar_laea",   1, [4.0e-3, 4.0e-3, 1.3e-4, 4.2e-4]),
        ("tripolar_lonlat", 1, [3.0e-3, 3.0e-3, 1.6e-4, 1.3e-3]),
        ("tripolar_lonlat", 8, [1.7e-2, 1.7e-2, 1.25e-3, 2.7e-3]),
    ];
    let m = source();
    let src = field(&m);
    for (name, densify, [cs, cf, cv, ca]) in ceilings {
        let fx = read_fixture(&format!("{name}.sph.txt"));
        let o = ours(&m, &fx, densify);
        let d = diff(&fx, &o, &src);
        let area = max_rel(&o.area_b, &fx.area_b, RADIUS * RADIUS);
        let inside = s_inside(&fx, &o);
        eprintln!("{name:16} sph densify {densify}: {d:?}, S inside {inside:.2e}, max relative area difference {area:.2e}");
        assert!(
            d.esmf_rows < 1e-9,
            "{name}: ESMF row sums off by {:e}",
            d.esmf_rows
        );
        assert!(
            d.s <= cs,
            "{name} densify {densify}: S differs by {:e} > {cs:e}",
            d.s
        );
        assert!(
            d.frac_b <= cf,
            "{name} densify {densify}: frac_b differs by {:e} > {cf:e}",
            d.frac_b
        );
        assert!(
            d.field <= cv,
            "{name} densify {densify}: field differs by {:e} > {cv:e}",
            d.field
        );
        assert!(
            area <= ca,
            "{name} densify {densify}: area_b differs by {area:e} > {ca:e}"
        );
    }
}
