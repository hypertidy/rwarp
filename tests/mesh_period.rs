//! A declared period of world x (`QuadMesh::with_period`,
//! `Rectilinear::with_period`): longitude, whose seam's two walls are 360
//! apart in the plane. Polygons are burned at each whole period at which
//! they overlap the grid, points are found at any, and a mesh source
//! (`MeshTransformer`) takes outlines across its seam and round a pole.
//!
//! Generator 5 of `design/curvilinear-frontend.md`: a regular global grid
//! declared periodic, its columns rotated by `k`, against controlledburn's
//! regular burn shifted by `k`.

use std::collections::HashMap;
use std::f64::consts::PI;

use controlledburn::{BurnOptions, BurnResult, Coord, CoverageWindow, Geometry, GridSpec, Polygon};
use rwarp::mesh::{
    unwrap_bounds, unwrap_grid, CellMap, Halo, MeshTransformer, QuadMesh, Rectilinear,
};
use rwarp::warp::{warp_to, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{
    exact_weights, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights,
};
use rwarp::CrsTransform;

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

/// 10 degree cells, 36 x 18, row 0 at the north pole.
const D: f64 = 10.0;
const NC: usize = 36;
const NR: usize = 18;

/// Corners of the global grid with its columns rotated by `k`: column 0
/// from `-180 + k D`, column line `NC` one period on. With `curve` the
/// interior lines are bent, smoothly and periodically in longitude, so
/// cells aren't rectangles.
fn global(k: usize, curve: bool) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=NR {
        for i in 0..=NC {
            let lon = -180.0 + (i + k) as f64 * D;
            let lat = 90.0 - j as f64 * D;
            let (dx, dy) = if curve && j > 0 && j < NR {
                let t = lon.to_radians();
                (
                    2.0 * (2.0 * t).sin() * lat.to_radians().cos(),
                    1.5 * (3.0 * t).cos() * (lat / 90.0 * PI).sin(),
                )
            } else {
                (0.0, 0.0)
            };
            x.push(lon + dx);
            y.push(lat + dy);
        }
    }
    (x, y)
}

fn mesh(k: usize, curve: bool) -> QuadMesh {
    let (x, y) = global(k, curve);
    QuadMesh::from_corners(NC, NR, x, y, CellMap::EqualArea)
        .unwrap()
        .with_period(360.0)
        .unwrap()
}

fn rect(k: usize) -> Rectilinear {
    let x = (0..=NC).map(|i| -180.0 + (i + k) as f64 * D).collect();
    let y = (0..=NR).map(|j| 90.0 - j as f64 * D).collect();
    Rectilinear::new(x, y).unwrap().with_period(360.0).unwrap()
}

fn ring(pts: &[[f64; 2]]) -> Vec<Coord> {
    let mut r: Vec<Coord> = pts.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
    r.push(r[0]);
    r
}

/// A star-shaped polygon about `c`, radius `r / 2` to `r`, with a square
/// hole of half-width `r / 5`.
fn star(rng: &mut Lcg, c: [f64; 2], r: f64) -> Vec<Vec<[f64; 2]>> {
    let n = 7 + (rng.u() * 9.0) as usize;
    let outer: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = (i as f64 + 0.8 * rng.u()) / n as f64 * 2.0 * PI;
            let rr = r * (0.5 + 0.5 * rng.u());
            [c[0] + rr * a.cos(), c[1] + rr * a.sin()]
        })
        .collect();
    let h = r / 5.0;
    let hole = vec![
        [c[0] - h, c[1] - h],
        [c[0] - h, c[1] + h],
        [c[0] + h, c[1] + h],
        [c[0] + h, c[1] - h],
    ];
    vec![outer, hole]
}

/// A cap round the south pole, closed along it (as Natural Earth's
/// Antarctica is): a wavy parallel from -180 to 180, then down to -90 and
/// back along it, starting at `lon0`.
fn cap(lon0: f64) -> Vec<Vec<[f64; 2]>> {
    let mut r: Vec<[f64; 2]> = (0..=72)
        .map(|i| {
            let lon = lon0 + 5.0 * i as f64;
            [lon, -62.0 + 3.0 * (3.0 * lon.to_radians()).sin()]
        })
        .collect();
    r.extend([[lon0 + 360.0, -90.0], [lon0, -90.0]]);
    vec![r]
}

/// Planar area of a polygon given as rings, exterior first.
fn area(rings: &[Vec<[f64; 2]>]) -> f64 {
    let a = |r: &[[f64; 2]]| {
        let n = r.len();
        0.5 * (0..n)
            .map(|i| r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1])
            .sum::<f64>()
            .abs()
    };
    a(&rings[0]) - rings[1..].iter().map(|r| a(r)).sum::<f64>()
}

fn geom(rings: &[Vec<[f64; 2]>]) -> Geometry {
    Geometry::Polygon(Polygon {
        rings: rings.iter().map(|r| ring(r)).collect(),
    })
}

/// Random polygons, many across a seam at longitude `seam` (in any of its
/// conventions), each moved by a whole number of periods from -1 to 1;
/// and the cap.
fn polygons(seed: u64, seam: f64) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for i in 0..60 {
        let lon = if i % 2 == 0 {
            seam + 30.0 * (rng.u() - 0.5)
        } else {
            -180.0 + 360.0 * rng.u()
        };
        let lat = -60.0 + 120.0 * rng.u();
        let r = 5.0 + 20.0 * rng.u();
        let mut p = star(&mut rng, [lon, lat], r);
        let s = 360.0 * ((rng.u() * 3.0).floor() - 1.0);
        for r in &mut p {
            for v in r.iter_mut() {
                v[0] += s;
            }
        }
        out.push(p);
    }
    out.push(cap(-180.0));
    out
}

/// `(id, row, col) -> fraction`, runs as 1.
fn cells(res: &BurnResult) -> HashMap<(i32, i32, i32), f64> {
    let mut m = HashMap::new();
    for e in &res.edges {
        *m.entry((e.id, e.row, e.col)).or_insert(0.0) += e.fraction as f64;
    }
    for r in &res.runs {
        for c in r.col_start..r.col_end {
            *m.entry((r.id, r.row, c)).or_insert(0.0) += 1.0;
        }
    }
    m
}

fn max_diff<K: std::hash::Hash + Eq + Copy>(a: &HashMap<K, f64>, b: &HashMap<K, f64>) -> f64 {
    let one = |p: &HashMap<K, f64>, q: &HashMap<K, f64>| {
        p.iter()
            .map(|(k, v)| (v - q.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max)
    };
    one(a, b).max(one(b, a))
}

/// Cells of mesh `k` renumbered as the cells of mesh 0: column `c` there
/// is column `(c + k) % NC` here.
fn unrotate(m: &HashMap<(i32, i32, i32), f64>, k: usize) -> HashMap<(i32, i32, i32), f64> {
    m.iter()
        .map(|(&(id, r, c), &f)| ((id, r, (c + k as i32) % NC as i32), f))
        .collect()
}

/// Each polygon's covered area on `m` against its own.
fn worst_area(polys: &[Vec<Vec<[f64; 2]>>], res: &HashMap<(i32, i32, i32), f64>, m: &QuadMesh) -> f64 {
    let mut got = vec![0.0; polys.len()];
    for (&(id, r, c), &f) in res {
        got[id as usize] += f * m.cell_area(r as usize, c as usize);
    }
    polys
        .iter()
        .zip(&got)
        .map(|(p, g)| (g - area(p)).abs() / area(p))
        .fold(0.0, f64::max)
}

/// Generator 5. The regular grid's columns rotated by `k` and declared
/// periodic, as a `QuadMesh` and as a `Rectilinear`: polygons across the
/// seam, in any of their conventions, against controlledburn's burn on a
/// regular grid five periods wide, folded back onto one period. Fractions
/// are `f32`, so equal to about 1e-7.
#[test]
fn generator_5_rotated_columns_match_the_regular_burn() {
    for k in [0, 1, 7, 18, 35] {
        let seam = -180.0 + k as f64 * D;
        let polys = polygons(5 + k as u64, seam);
        let geoms: Vec<Geometry> = polys.iter().map(|p| geom(p)).collect();
        let m = mesh(k, false);
        let ours = cells(&m.burn(&geoms, BurnOptions::coverage()).unwrap());
        let theirs = cells(&rect(k).burn(&geoms, BurnOptions::coverage()).unwrap());
        let wide = GridSpec::new(seam - 720.0, -90.0, seam + 1080.0, 90.0, 5 * NC as u32, NR as u32);
        let regular = cells(&controlledburn::burn(&geoms, &wide, BurnOptions::coverage()).unwrap());
        let mut folded = HashMap::new();
        for ((id, r, c), f) in regular {
            *folded.entry((id, r, c % NC as i32)).or_insert(0.0) += f;
        }
        // polygons whose cells are at both ends of the mesh
        let across = (0..polys.len() as i32)
            .filter(|&id| {
                ours.keys().any(|&(i, _, c)| i == id && c == 0)
                    && ours.keys().any(|&(i, _, c)| i == id && c == NC as i32 - 1)
            })
            .count();
        let (d, dr, a) = (max_diff(&ours, &folded), max_diff(&theirs, &folded), worst_area(&polys, &ours, &m));
        eprintln!("k = {k:2}: {} cells, {across} polygons across the seam; mesh {d:.1e}, rectilinear {dr:.1e}, area {a:.1e}", ours.len());
        assert!(across >= 15, "only {across} polygons cross the seam");
        assert!(d < 2e-6 && dr < 2e-6, "mesh {d:e}, rectilinear {dr:e}");
        assert!(a < 1e-6, "a polygon's covered area is off by {a:e}");
    }
}

/// On a curved mesh too, where the seam is makes no difference: rotated by
/// `k`, the same polygons cover the same cells, renumbered.
#[test]
fn a_curved_mesh_covers_the_same_wherever_its_seam_is() {
    let polys = polygons(11, -110.0);
    let geoms: Vec<Geometry> = polys.iter().map(|p| geom(p)).collect();
    let m0 = mesh(0, true);
    let base = cells(&m0.burn(&geoms, BurnOptions::coverage()).unwrap());
    assert!(worst_area(&polys, &base, &m0) < 1e-6);
    for k in [7, 20, 35] {
        let m = mesh(k, true);
        let res = cells(&m.burn(&geoms, BurnOptions::coverage()).unwrap());
        let d = max_diff(&unrotate(&res, k), &base);
        let a = worst_area(&polys, &res, &m);
        eprintln!("k = {k:2}: {} cells; against k = 0 {d:.1e}, area {a:.1e}", res.len());
        assert!(d < 2e-6 && a < 1e-6, "{d:e}, area {a:e}");
    }
}

/// A polygon a whole number of periods away is the same polygon, and so
/// is a point; `burn_par` is `burn`.
#[test]
fn a_polygon_burns_the_same_at_any_whole_period() {
    let m = mesh(7, true);
    let polys = polygons(3, -110.0);
    let at = |s: f64| -> Vec<Geometry> {
        polys
            .iter()
            .map(|p| {
                let moved: Vec<Vec<[f64; 2]>> = p
                    .iter()
                    .map(|r| r.iter().map(|v| [v[0] + s, v[1]]).collect())
                    .collect();
                geom(&moved)
            })
            .collect()
    };
    let base = cells(&m.burn(&at(0.0), BurnOptions::coverage()).unwrap());
    for s in [-720.0, -360.0, 360.0, 1080.0] {
        let d = max_diff(&cells(&m.burn(&at(s), BurnOptions::coverage()).unwrap()), &base);
        assert!(d < 2e-6, "moved by {s}: {d:e}");
    }
    let par = m.burn_par(&at(0.0), BurnOptions::coverage(), 4).unwrap();
    assert_eq!(cells(&par), base);

    let pts: Vec<Coord> = (0..50)
        .map(|i| Coord { x: -180.0 + 7.3 * i as f64, y: -80.0 + 3.1 * i as f64 })
        .collect();
    let burn_points = |s: f64| {
        let g = Geometry::MultiPoint(pts.iter().map(|c| Coord { x: c.x + s, y: c.y }).collect());
        let mut p: Vec<(i32, i32)> = m
            .burn(&[g], BurnOptions::coverage())
            .unwrap()
            .points
            .iter()
            .map(|p| (p.row, p.col))
            .collect();
        p.sort();
        p
    };
    let p0 = burn_points(0.0);
    assert_eq!(p0.len(), pts.len());
    assert_eq!(burn_points(360.0), p0);
    assert_eq!(burn_points(-720.0), p0);
}

/// `cover_ring` across the seam: its window spans the pieces at both
/// ends, and holds the burn's fractions.
#[test]
fn cover_ring_across_the_seam_is_the_burn() {
    let m = mesh(7, true);
    let g = rect(7);
    let mut rng = Lcg(17);
    let mut win = CoverageWindow::default();
    let (mut wide, mut worst) = (0, 0.0f64);
    for i in 0..40 {
        let c = [-110.0 + 20.0 * (rng.u() - 0.5) + 360.0 * (i % 3) as f64, -60.0 + 120.0 * rng.u()];
        let r = 3.0 + 9.0 * rng.u();
        let outer = star(&mut rng, c, r).swap_remove(0);
        let burn = |res: BurnResult| {
            cells(&res)
                .into_iter()
                .map(|((_, r, c), f)| ((0, r, c), f))
                .collect::<HashMap<_, _>>()
        };
        let window = |win: &CoverageWindow| {
            let mut out = HashMap::new();
            for (k, &f) in win.coverage().iter().enumerate() {
                let (r, c) = (win.row0() + (k / win.ncol()) as i64, win.col0() + (k % win.ncol()) as i64);
                if f != 0.0 && (0..NC as i64).contains(&c) && (0..NR as i64).contains(&r) {
                    out.insert((0, r as i32, c as i32), f);
                }
            }
            out
        };
        let pg = [geom(std::slice::from_ref(&outer))];
        assert!(m.cover_ring(&outer, &mut win).unwrap());
        wide += (win.ncol() == NC) as usize;
        let d = max_diff(&window(&win), &burn(m.burn(&pg, BurnOptions::coverage()).unwrap()));
        assert!(g.cover_ring(&outer, &mut win));
        let dr = max_diff(&window(&win), &burn(g.burn(&pg, BurnOptions::coverage()).unwrap()));
        worst = worst.max(d).max(dr);
    }
    eprintln!("{wide} of 40 rings span the seam; window against burn {worst:.1e}");
    assert!(wide >= 15, "only {wide} rings span the seam");
    assert!(worst < 2e-6, "{worst:e}");
}

/// Points at any whole period: `locate`, `locate_all`, `to_index` and
/// `Rectilinear::cell`. A point on the seam's far wall is on column 0's
/// near wall one period back, which column 0 owns.
#[test]
fn points_are_found_at_any_whole_period() {
    let m = mesh(7, true);
    let g = rect(7);
    let plain = QuadMesh::from_corners(NC, NR, global(7, true).0, global(7, true).1, CellMap::EqualArea).unwrap();
    let mut rng = Lcg(23);
    for _ in 0..200 {
        let p = [-110.0 + 360.0 * rng.u(), -85.0 + 170.0 * rng.u()];
        let cell = m.locate(p, None).unwrap();
        let idx = m.to_index(p, None).unwrap();
        for s in [-720.0, -360.0, 360.0, 720.0] {
            let q = [p[0] + s, p[1]];
            assert_eq!(m.locate(q, None), Some(cell));
            assert_eq!(m.locate_all(q), vec![cell]);
            let iq = m.to_index(q, Some(cell)).unwrap();
            assert!((iq[0] - idx[0]).abs() < 1e-9 && (iq[1] - idx[1]).abs() < 1e-9);
            if s.abs() > 360.0 {
                assert_eq!(plain.locate(q, None), None);
            }
            assert_eq!(g.cell(q), g.cell(p));
        }
    }
    // the regular grid's seam, at 110 W
    let flat = mesh(7, false);
    assert_eq!(flat.locate([250.0, 5.0], None), Some((8, 0)));
    assert_eq!(flat.locate([-470.0, 5.0], None), Some((8, 0)));
    assert_eq!(g.cell([250.0, 5.0]), Some((8, 0)));
}

/// Longitudes stored with a jump, as corners, centres and CF bounds, are
/// refused as they are and give the continuous mesh once unwrapped. A
/// cyclic halo (NEMO's: the last column first, the first column last) is
/// dropped with `Halo`; left in, it overlaps the cells it copies one
/// period on, and `with_period` refuses it.
#[test]
fn longitudes_that_jump_unwrap_to_one_sheet() {
    let wrap = |v: f64| (v + 180.0).rem_euclid(360.0) - 180.0;
    let (x, y) = global(7, false);
    let reference = mesh(7, false);

    let mut jumped: Vec<f64> = x.iter().map(|&v| wrap(v)).collect();
    let e = QuadMesh::from_corners(NC, NR, jumped.clone(), y.clone(), CellMap::EqualArea).unwrap_err();
    assert!(e.contains("unwrap_grid"), "{e}");
    unwrap_grid(NC + 1, NR + 1, &mut jumped, 360.0).unwrap();
    assert_eq!(jumped, x);

    // centres, through from_centres
    let centre = |a: &[f64], r: usize, c: usize| {
        0.25 * (a[r * (NC + 1) + c] + a[r * (NC + 1) + c + 1] + a[(r + 1) * (NC + 1) + c] + a[(r + 1) * (NC + 1) + c + 1])
    };
    let (cx, cy): (Vec<f64>, Vec<f64>) = (0..NR * NC)
        .map(|k| (centre(&x, k / NC, k % NC), centre(&y, k / NC, k % NC)))
        .unzip();
    let mut cj: Vec<f64> = cx.iter().map(|&v| wrap(v)).collect();
    unwrap_grid(NC, NR, &mut cj, 360.0).unwrap();
    let a = QuadMesh::from_centres(NC, NR, &cj, &cy, CellMap::EqualArea).unwrap();
    let b = QuadMesh::from_centres(NC, NR, &cx, &cy, CellMap::EqualArea).unwrap();
    for r in 0..NR {
        for c in 0..NC {
            assert_eq!(a.corners(r, c), b.corners(r, c));
        }
    }

    // CF bounds in [0, 360), NEMO's cyclic halo: file column 0 is grid
    // column NC - 1, file columns 1 to NC are the grid, file column NC + 1
    // is grid column 0
    let fc = NC + 2;
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    for r in 0..NR {
        for f in 0..fc {
            let c = (f + NC - 1) % NC;
            for (j, i) in [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)] {
                xb.push(x[j * (NC + 1) + i].rem_euclid(360.0));
                yb.push(y[j * (NC + 1) + i]);
            }
        }
    }
    let halo = Halo { cols: [1, 1], ..Halo::NONE };
    let e = QuadMesh::from_cf_bounds(fc, NR, &xb, &yb, halo, CellMap::EqualArea).unwrap_err();
    assert!(e.contains("unwrap_bounds"), "{e}");
    unwrap_bounds(fc, NR, &mut xb, 360.0).unwrap();
    let cf = QuadMesh::from_cf_bounds(fc, NR, &xb, &yb, halo, CellMap::EqualArea)
        .unwrap()
        .with_period(360.0)
        .unwrap();
    assert!(cf.find_overlap().is_none());
    // the same cells, a whole number of periods away
    let turn = cf.corners(0, 0)[0][0] - reference.corners(0, 0)[0][0];
    assert_eq!(turn.rem_euclid(360.0), 0.0);
    for r in 0..NR {
        for c in 0..NC {
            let (p, q) = (cf.corners(r, c), reference.corners(r, c));
            for v in 0..4 {
                assert_eq!([p[v][0] - turn, p[v][1]], q[v], "cell ({r}, {c})");
            }
        }
    }
    let kept = QuadMesh::from_cf_bounds(fc, NR, &xb, &yb, Halo::NONE, CellMap::EqualArea).unwrap();
    let e = kept.with_period(360.0).unwrap_err();
    assert!(e.contains("whole number of periods"), "{e}");
}

/// What `with_period` refuses, and what it keeps.
#[test]
fn a_period_is_checked() {
    for p in [0.0, -360.0, f64::NAN, f64::INFINITY] {
        assert!(mesh(0, false).with_period(p).is_err(), "period {p}");
        assert!(rect(0).with_period(p).is_err(), "period {p}");
    }
    // 37 columns of 10 degrees go round more than once
    let (n, x, y) = {
        let n = NC + 1;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for j in 0..=NR {
            for i in 0..=n {
                x.push(-180.0 + i as f64 * D);
                y.push(90.0 - j as f64 * D);
            }
        }
        (n, x, y)
    };
    let e = QuadMesh::from_corners(n, NR, x, y, CellMap::EqualArea)
        .unwrap()
        .with_period(360.0)
        .unwrap_err();
    assert!(e.contains("more than once"), "{e}");
    let wide = Rectilinear::new((0..=n).map(|i| i as f64 * D).collect(), vec![0.0, 10.0]).unwrap();
    assert!(wide.with_period(360.0).is_err());

    // a cell half a period wide, as round a pole: two columns of 180
    let halves = QuadMesh::from_corners(2, 1, vec![-180.0, 0.0, 180.0, -180.0, 0.0, 180.0], vec![90.0, 90.0, 90.0, 0.0, 0.0, 0.0], CellMap::EqualArea).unwrap();
    let e = halves.with_period(360.0).unwrap_err();
    assert!(e.contains("round a") && e.contains("pole"), "{e}");
    assert!(Rectilinear::new(vec![-180.0, 0.0, 180.0], vec![0.0, 10.0]).unwrap().with_period(360.0).is_err());

    // a regional grid from 120 to 300 takes polygons from -180 to 180
    let sq = |x0: f64| vec![vec![[x0, 0.0], [x0 + 20.0, 0.0], [x0 + 20.0, 20.0], [x0, 20.0]]];
    let regional = Rectilinear::new((0..=18).map(|i| 120.0 + i as f64 * D).collect(), vec![20.0, 10.0, 0.0])
        .unwrap()
        .with_period(360.0)
        .unwrap();
    let res = cells(&regional.burn(&[geom(&sq(-80.0))], BurnOptions::coverage()).unwrap());
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    assert_eq!(keys, vec![(0, 0, 16), (0, 0, 17), (0, 1, 16), (0, 1, 17)]);
    assert_eq!(regional.period(), Some(360.0));

    let m = mesh(7, true);
    assert_eq!(m.period(), Some(360.0));
    assert_eq!(m.window(3, 2, 10, 5).unwrap().period(), Some(360.0));
    assert_eq!(QuadMesh::from_corners(NC, NR, global(0, false).0, global(0, false).1, CellMap::EqualArea).unwrap().period(), None);

    // a polygon twenty periods wide
    let long = vec![vec![[0.0, 0.0], [7200.0, 0.0], [7200.0, 1.0], [0.0, 1.0]]];
    let e = m.burn(&[geom(&long)], BurnOptions::coverage()).unwrap_err();
    assert!(e.contains("periods"), "{e}");
}

// ---------------------------------------------------------------------------
// MeshTransformer
// ---------------------------------------------------------------------------

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// `(dst_row, dst_col, src_row, src_col) -> frac`, with mesh `k`'s source
/// columns renumbered as mesh 0's.
fn entries(w: &Weights, k: usize) -> HashMap<(u32, u32, u32, u32), f64> {
    (0..w.len())
        .map(|i| {
            (
                (w.dst_row[i], w.dst_col[i], w.src_row[i], (w.src_col[i] + k as u32) % NC as u32),
                w.frac[i],
            )
        })
        .collect()
}

/// Each source cell's `frac` summed, by mesh 0's numbering.
fn shared(w: &Weights, k: usize) -> Vec<f64> {
    let mut s = vec![0.0; NC * NR];
    for (&(_, _, r, c), f) in &entries(w, k) {
        s[r as usize * NC + c as usize] += f;
    }
    s
}

/// A destination grid in longitude and latitude, 7.5 degree pixels, from
/// -180 or from 0: pixels straddle the seam of each rotated mesh, and
/// every outline crossing it is measured at both ends. Weights are mesh
/// 0's, renumbered, and each cell is shared out in full.
#[test]
fn a_mesh_in_longitude_takes_outlines_across_its_seam() {
    let (dw, dh) = (48, 24);
    for x0 in [-180.0, 0.0, -176.25] {
        let gt = [x0, 7.5, 0.0, 90.0, 0.0, -7.5];
        let weights = |k: usize| {
            let t = MeshTransformer::with_backend(mesh(k, true), gt, Box::new(Same), Box::new(Same)).unwrap();
            exact_weights(&t, &t.source_window(), dw, dh, &WeightOptions::new(2))
        };
        let base = weights(0);
        let worst = shared(&base, 0).iter().map(|s| (s - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-12, "from {x0}: a cell's frac sums to 1 +- {worst:e}");
        for k in [7, 20] {
            let w = weights(k);
            let d = max_diff(&entries(&w, k), &entries(&base, 0));
            let area = w.dst_area.iter().zip(&base.dst_area).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            // destination pixels with cells at both ends of the mesh
            let mut ends: HashMap<(u32, u32), [bool; 2]> = HashMap::new();
            for i in 0..w.len() {
                let e = ends.entry((w.dst_row[i], w.dst_col[i])).or_default();
                e[0] |= w.src_col[i] == 0;
                e[1] |= w.src_col[i] == NC as u32 - 1;
            }
            let across = ends.values().filter(|e| e[0] && e[1]).count();
            eprintln!("from {x0:7}, k = {k:2}: {} entries, {across} pixels across the seam; frac {d:.1e}, area {area:.1e}", w.len());
            assert!(across >= dh, "only {across} pixels across the seam");
            assert!(d < 1e-12 && area < 1e-9, "frac {d:e}, area {area:e}");
        }
    }
}

const R: f64 = 6371229.0;

/// North-polar Lambert azimuthal equal-area on a sphere: `true` from
/// longitude and latitude, `false` back (longitude in -180 to 180).
struct Laea(bool);
impl CrsTransform for Laea {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some(if self.0 {
            let rho = 2.0 * R * ((90.0 - y).to_radians() / 2.0).sin();
            (rho * x.to_radians().sin(), -rho * x.to_radians().cos())
        } else {
            let rho = x.hypot(y);
            (x.atan2(-y).to_degrees(), 90.0 - 2.0 * (rho / (2.0 * R)).asin().to_degrees())
        })
    }
}

/// 25 x 25 pixels of 400 km round the north pole, which is at the centre
/// of pixel (12, 12); every cell north of 50 N is inside.
const POLAR: [f64; 6] = [-5.0e6, 4.0e5, 0.0, 5.0e6, 0.0, -4.0e5];
const PW: usize = 25;

fn polar(k: usize) -> MeshTransformer {
    MeshTransformer::with_backend(mesh(k, true), POLAR, Box::new(Laea(false)), Box::new(Laea(true))).unwrap()
}

/// A polar destination: outlines cross every rotated mesh's seam, and the
/// pole's outline goes round it and is closed along latitude 90. The
/// weights are mesh 0's, renumbered; the cells north of 50 N are shared
/// out in full; the pole's pixel takes all of the top row's cells, and
/// covers its outline's area.
#[test]
fn a_polar_destination_crosses_the_seam_and_the_pole() {
    for densify in [1, 4] {
        let opts = WeightOptions::new(densify);
        let w0 = {
            let t = polar(0);
            exact_weights(&t, &t.source_window(), PW, PW, &opts)
        };
        let inside = shared(&w0, 0)[..4 * NC].iter().map(|s| (s - 1.0).abs()).fold(0.0, f64::max);
        let pole: Vec<usize> = (0..w0.len()).filter(|&i| (w0.dst_row[i], w0.dst_col[i]) == (12, 12)).collect();
        let top = pole.iter().filter(|&&i| w0.src_row[i] == 0).count();
        let covered: f64 = pole.iter().map(|&i| w0.frac[i] * w0.src_area(i)).sum();
        let outline = w0.dst_area[12 * PW + 12];
        eprintln!(
            "densify {densify}: {} entries; cells north of 50 N shared out to {inside:.1e}; pole pixel: {top} top-row cells, {covered:.3} of {outline:.3} square degrees",
            w0.len()
        );
        assert!(inside < 1e-12, "{inside:e}");
        assert_eq!(top, NC);
        assert!((covered - outline).abs() < 1e-9 * outline, "{covered} vs {outline}");
        for k in [7, 20] {
            let t = polar(k);
            let w = exact_weights(&t, &t.source_window(), PW, PW, &opts);
            let d = max_diff(&entries(&w, k), &entries(&w0, 0));
            eprintln!("  k = {k:2}: frac {d:.1e} from k = 0");
            assert!(d < 1e-12, "{d:e}");
        }
    }
}

/// `warp()`'s exact `Average` over a periodic mesh source is
/// `apply(FracArea)`, bit for bit, across the seam and round the pole.
#[test]
fn warp_exact_average_across_the_seam_is_apply_frac_area() {
    let t = polar(7);
    let data: Vec<f64> = (0..NC * NR)
        .map(|p| 10.0 + ((p % NC) as f64 * 0.4).sin() * ((p / NC) as f64 * 0.3).cos())
        .collect();
    let buf = SourceBuffer { data: &data[..], ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    for densify in [1, 3] {
        let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify });
        let warped = warp_to::<f64, f64>(&t, &buf, PW, PW, &opts);
        let w = exact_weights(&t, &GridWindow::from(&buf), PW, PW, &WeightOptions::warp_compatible(densify));
        let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
        let differ = warped.iter().zip(&applied).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let computed = warped.iter().filter(|v| (5.0..15.0).contains(*v)).count();
        eprintln!("densify {densify}: {computed} pixels computed, {differ} differ");
        assert_eq!(differ, 0);
        assert_eq!(computed, PW * PW);
    }
}

/// With the CRS backend: the mesh in longitude and latitude on a sphere,
/// the destination in north-polar LAEA on it. The same weights as with the
/// projection written out, to the backend's rounding.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn a_crs_backend_gives_the_same_weights() {
    let laea = format!("+proj=laea +lat_0=90 +lon_0=0 +R={R} +units=m +no_defs");
    let lonlat = format!("+proj=longlat +R={R} +no_defs");
    let with_crs = MeshTransformer::new(mesh(7, true), &lonlat, POLAR, &laea).unwrap();
    let written = polar(7);
    let opts = WeightOptions::new(4);
    let a = exact_weights(&with_crs, &with_crs.source_window(), PW, PW, &opts);
    let b = exact_weights(&written, &written.source_window(), PW, PW, &opts);
    let d = max_diff(&entries(&a, 0), &entries(&b, 0));
    let area = a
        .dst_area
        .iter()
        .zip(&b.dst_area)
        .map(|(x, y)| ((x - y) / y).abs())
        .fold(0.0, f64::max);
    eprintln!("{} entries; CRS backend vs written out: frac {d:.1e}, area {area:.1e}", a.len());
    assert_eq!(a.len(), b.len());
    assert!(d < 1e-9 && area < 1e-9, "frac {d:e}, area {area:e}");
}

/// The hint for longitudes that jump names the helper, in degrees and in
/// radians; and coordinates far beyond the grid neither overflow nor
/// take forever.
#[test]
fn jumps_are_named_and_far_rings_are_refused() {
    use std::f64::consts::TAU;
    for (p, step) in [(360.0, D), (TAU, D.to_radians())] {
        let x: Vec<f64> = (0..=4).map(|i| (2.0 * p / 5.0 + step * i as f64 + p / 2.0).rem_euclid(p) - p / 2.0).collect();
        let x = [x.clone(), x].concat();
        let y = [[step; 5], [0.0; 5]].concat();
        let e = QuadMesh::from_corners(4, 1, x, y, CellMap::EqualArea).unwrap_err();
        assert!(e.contains("unwrap_grid"), "period {p}: {e}");
    }
    let m = mesh(0, false);
    let far = vec![vec![[-1e300, 0.0], [1e300, 0.0], [1e300, 1.0], [-1e300, 1.0]]];
    assert!(m.burn(&[geom(&far)], BurnOptions::coverage()).is_err());
    let beyond = vec![vec![[1e300, 0.0], [1e300 + 1e290, 0.0], [1e300, 1.0]]];
    let res = m.burn(&[geom(&beyond)], BurnOptions::coverage()).unwrap();
    assert!(res.runs.is_empty() && res.edges.is_empty());
    assert_eq!(m.locate([1e300, 0.0], None), None);
    assert!(!rect(0).cover_ring(&far[0], &mut CoverageWindow::default()));
}
