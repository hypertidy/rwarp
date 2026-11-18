//! `QuadMesh::burn_from` and `Rectilinear::burn_from`: polygons and
//! points in another CRS, their edges straight there, densified by
//! bigcurve into the grid's world (curvilinear stage 5b,
//! `design/curvilinear-frontend.md`).
//!
//! Oracles (#38): the area burned converges on the closed form at the
//! tolerance's rate while the points grow as its inverse square root; a
//! fine tolerance is a fine fixed densify. Also: an identity CRS is
//! `burn`; the far pole is cut away; a cut at 180 degrees is no cut; points;
//! a mesh with a period unwraps rings across its seam and refuses one
//! round its pole; `Rectilinear` is the same grid as a `QuadMesh`;
//! threads; refusals.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use controlledburn::{BurnMode, BurnResult, Coord, Geometry, Polygon};
use rwarp::mesh::{CellMap, QuadMesh, Rectilinear};
use rwarp::CrsTransform;

const R: f64 = 6371229.0;
const NCOL: usize = 48;
const NROW: usize = 16;

/// `tests/mesh_transformer.rs`'s tripolar cap: confocal ellipses about
/// grid poles at `(+-a, 0)`, a seam into one grid pole and a fold between
/// the two, turned and shifted, in metres of a north polar LAEA. It
/// reaches about 62 N.
fn tripolar() -> QuadMesh {
    let (a, mu, turn, shift) = (1.5e6, 1.475f64, 20f64.to_radians(), (9e4, -5e4));
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=NROW {
        let m = mu * (NROW - j) as f64 / NROW as f64;
        for i in 0..=NCOL {
            let i = i % NCOL;
            let k = i.min(NCOL - i);
            let t = std::f64::consts::TAU * k as f64 / NCOL as f64;
            let sign = if i > NCOL / 2 { -1.0 } else { 1.0 };
            let (ex, ey) = (a * m.cosh() * t.cos(), a * m.sinh() * t.sin() * sign);
            x.push(ex * turn.cos() - ey * turn.sin() + shift.0);
            y.push(ex * turn.sin() + ey * turn.cos() + shift.1);
        }
    }
    QuadMesh::from_corners(NCOL, NROW, x, y, CellMap::EqualArea).unwrap()
}

/// North polar LAEA on a sphere of radius `R`: longitude and latitude to
/// metres (`Laea(true)`), and back.
struct Laea(bool);

/// `Laea(true)`, counting the points it takes.
struct Counted<'a>(&'a AtomicUsize);

impl CrsTransform for Counted<'_> {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Some(laea(x, y))
    }
}

fn laea(lon: f64, lat: f64) -> (f64, f64) {
    let rho = 2.0 * R * ((90.0 - lat).to_radians() / 2.0).sin();
    let (s, c) = lon.to_radians().sin_cos();
    (rho * s, -rho * c)
}

impl CrsTransform for Laea {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some(if self.0 {
            laea(x, y)
        } else {
            (x.atan2(-y).to_degrees(), 90.0 - 2.0 * (x.hypot(y) / (2.0 * R)).asin().to_degrees())
        })
    }
}

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

fn polygon(rings: &[&[(f64, f64)]]) -> Geometry {
    Geometry::Polygon(Polygon::new(rings.iter().map(|r| r.iter().map(|&(x, y)| Coord { x, y }).collect()).collect()))
}

/// A box in longitude and latitude, anticlockwise.
fn lonlat_box(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
}

/// Geometry `id`'s covered world area on `m`.
fn area(r: &BurnResult, m: &QuadMesh, id: i32) -> f64 {
    let runs: f64 = r
        .runs
        .iter()
        .filter(|g| g.id == id)
        .flat_map(|g| (g.col_start..g.col_end).map(move |c| (g.row as usize, c as usize)))
        .map(|(row, c)| m.cell_area(row, c))
        .sum();
    let edges: f64 = r.edges.iter().filter(|e| e.id == id).map(|e| e.fraction as f64 * m.cell_area(e.row as usize, e.col as usize)).sum();
    runs + edges
}

/// `(id, row, col) -> fraction`, 1 for a run's cells.
fn cells(r: &BurnResult) -> HashMap<(i32, i32, i32), f64> {
    let mut m = HashMap::new();
    for g in &r.runs {
        for c in g.col_start..g.col_end {
            m.insert((g.id, g.row, c), 1.0);
        }
    }
    for e in &r.edges {
        m.insert((e.id, e.row, e.col), e.fraction as f64);
    }
    m
}

fn max_diff(a: &HashMap<(i32, i32, i32), f64>, b: &HashMap<(i32, i32, i32), f64>) -> f64 {
    a.iter()
        .map(|(k, v)| (v - b.get(k).copied().unwrap_or(0.0)).abs())
        .chain(b.iter().filter(|(k, _)| !a.contains_key(k)).map(|(_, v)| v.abs()))
        .fold(0.0, f64::max)
}

/// The cap north of 70 N, written as a lon/lat box (the 180 meridian
/// there and back, the pole a line).
fn cap70() -> Geometry {
    polygon(&[&lonlat_box(-180.0, 180.0, 70.0, 90.0)])
}

/// On the sphere, LAEA keeps area: the cap's is `2 pi R^2 (1 - sin 70)`.
/// Its parallel comes out a circle, and the chords inside it take a
/// little off: between a sixth and two thirds of the tolerance times the
/// circle's length (each chord is off by between a quarter and the whole
/// tolerance, and a chord's sliver is two thirds of it times its length).
/// The points grow as the tolerance to the power -1/2, over four decades
/// (1028 to 113804 points from 100 m to 1 cm). A circle bisects evenly,
/// so from one tolerance to the next they double or quadruple. Above
/// about 300 m the first cut, at the mesh's tallest cell, holds the
/// chords closer than the tolerance asks.
#[test]
fn the_cap_converges_on_its_area() {
    let m = tripolar();
    let want = std::f64::consts::TAU * R * R * (1.0 - 70f64.to_radians().sin());
    let circle = std::f64::consts::TAU * 2.0 * R * 10f64.to_radians().sin();
    let mut points = Vec::new();
    for tol in [1e2, 1e1, 1.0, 1e-1, 1e-2] {
        let count = AtomicUsize::new(0);
        let r = m.burn_from(&[cap70()], &Counted(&count), &Laea(false), tol, BurnMode::Coverage).unwrap();
        let n = count.load(Ordering::Relaxed);
        let short = (want - area(&r, &m, 0)) / (tol * circle);
        eprintln!("tolerance {tol} m: {n} points, short by {short:.3} of the tolerance times the circle");
        assert!((1.0 / 6.0 - 0.02..0.67).contains(&short), "tolerance {tol}: {short}");
        points.push(n as f64);
    }
    let rate = (points[4] / points[0]).log10() / 4.0;
    eprintln!("points grow as the tolerance to the power -{rate:.3}");
    assert!((0.45..0.55).contains(&rate), "{rate}");
}

/// A fine tolerance is a fine fixed densify: the box's edges cut into
/// 4096 pieces in longitude and latitude, each end projected, and burned
/// with `burn`.
#[test]
fn a_fine_tolerance_is_a_fine_densify() {
    let m = tripolar();
    let b = lonlat_box(20.0, 110.0, 66.0, 84.0);
    let fine: Vec<Coord> = b
        .windows(2)
        .flat_map(|e| {
            (0..4096).map(move |k| {
                let t = k as f64 / 4096.0;
                let (x, y) = laea(e[0].0 + t * (e[1].0 - e[0].0), e[0].1 + t * (e[1].1 - e[0].1));
                Coord { x, y }
            })
        })
        .chain(std::iter::once({
            let (x, y) = laea(b[0].0, b[0].1);
            Coord { x, y }
        }))
        .collect();
    let want = m.burn(&[Geometry::Polygon(Polygon::new(vec![fine]))], BurnMode::Coverage).unwrap();
    let got = m.burn_from(&[polygon(&[&b])], &Laea(true), &Laea(false), 0.01, BurnMode::Coverage).unwrap();
    let d = max_diff(&cells(&got), &cells(&want));
    let (a, w) = (area(&got, &m, 0), area(&want, &m, 0));
    // closed form: R^2 dlon (sin 84 - sin 66)
    let exact = R * R * 90f64.to_radians() * (84f64.to_radians().sin() - 66f64.to_radians().sin());
    eprintln!("tolerance 1 cm vs densify 4096: fractions to {d:.1e}, areas {:.1e} and {:.1e} from the closed form", (a - exact) / exact, (w - exact) / exact);
    assert!(d < 1e-6, "{d:e}");
    assert!(((a - exact) / exact).abs() < 1e-6);
}

/// With the mesh's own CRS on both sides, `burn_from` is `burn` (the
/// first cut adds points along straight edges, so to rounding).
#[test]
fn an_identity_crs_is_burn() {
    let m = tripolar();
    let (x, y) = laea(30.0, 75.0);
    let ring = [(x, y), (x + 2.4e6, y + 3e5), (x + 9e5, y + 2.2e6), (x - 6e5, y + 1.1e6), (x, y)];
    let g = [polygon(&[&ring])];
    let want = cells(&m.burn(&g, BurnMode::Coverage).unwrap());
    let got = cells(&m.burn_from(&g, &Same, &Same, 1.0, BurnMode::Coverage).unwrap());
    assert!(want.len() > 20);
    assert!(max_diff(&got, &want) < 1e-6);
}

/// The whole world in longitude and latitude covers every cell in full:
/// it is cut to the mesh's latitudes first, and without the cut the
/// south would come out round the edge of the LAEA's disk. Antarctica
/// gets nothing. A box across 180 written past it is the same box split
/// at it.
#[test]
fn the_far_pole_is_cut_away_and_180_is_no_cut() {
    let m = tripolar();
    let world = polygon(&[&lonlat_box(-180.0, 180.0, -90.0, 90.0)]);
    let south = polygon(&[&lonlat_box(-180.0, 180.0, -90.0, -60.0)]);
    let r = m.burn_from(&[world, south], &Laea(true), &Laea(false), 1.0, BurnMode::Coverage).unwrap();
    let full: f64 = (0..NROW).flat_map(|r| (0..NCOL).map(move |c| (r, c))).map(|(r, c)| m.cell_area(r, c)).sum();
    let covered = area(&r, &m, 0);
    eprintln!("whole world: {:.1e} of the mesh's area off, {} edge cells", (covered - full) / full, r.edges.iter().filter(|e| e.id == 0).count());
    assert!(((covered - full) / full).abs() < 1e-7);
    assert!(r.edges.iter().filter(|e| e.id == 0).all(|e| e.fraction > 0.99999));
    assert!(r.runs.iter().all(|g| g.id == 0) && r.edges.iter().all(|e| e.id == 0));
    // across 180, written past it and split at it
    let across = polygon(&[&lonlat_box(150.0, 230.0, 68.0, 80.0)]);
    let split = Geometry::MultiPolygon(vec![
        Polygon::new(vec![lonlat_box(150.0, 180.0, 68.0, 80.0).iter().map(|&(x, y)| Coord { x, y }).collect()]),
        Polygon::new(vec![lonlat_box(-180.0, -130.0, 68.0, 80.0).iter().map(|&(x, y)| Coord { x, y }).collect()]),
    ]);
    let r = m.burn_from(&[across, split], &Laea(true), &Laea(false), 0.1, BurnMode::Coverage).unwrap();
    let (a, b) = (area(&r, &m, 0), area(&r, &m, 1));
    let exact = R * R * 80f64.to_radians() * (80f64.to_radians().sin() - 68f64.to_radians().sin());
    eprintln!("across 180: {:.1e}, split at it: {:.1e} from the closed form", (a - exact) / exact, (b - exact) / exact);
    assert!(((a - exact) / exact).abs() < 1e-6 && ((b - exact) / exact).abs() < 1e-6);
}

/// Points land in the cell their projection does; one in the south, or
/// beyond the mesh's latitudes, lands nowhere.
#[test]
fn points_in_another_crs() {
    let m = tripolar();
    let at = [(10.0, 80.0), (-120.0, 72.5), (170.0, 66.0), (0.0, -80.0), (45.0, 20.0)];
    let geoms = vec![
        Geometry::MultiPoint(at.iter().map(|&(x, y)| Coord { x, y }).collect()),
        Geometry::Point(Coord { x: 0.0, y: -80.0 }),
    ];
    let r = m.burn_from(&geoms, &Laea(true), &Laea(false), 1.0, BurnMode::Coverage).unwrap();
    let want: Vec<(i32, i32)> = at[..3]
        .iter()
        .map(|&(lon, lat)| {
            let (x, y) = laea(lon, lat);
            let (row, col) = m.locate([x, y], None).unwrap();
            (row as i32, col as i32)
        })
        .collect();
    let got: Vec<(i32, i32)> = r.points.iter().map(|p| (p.row, p.col)).collect();
    assert_eq!(got, want);
    assert!(r.points.iter().all(|p| p.id == 0));
    // a projection that folds the south onto the north: what it sends
    // into the mesh from beyond the mesh's latitudes isn't taken
    struct Folded;
    impl CrsTransform for Folded {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some(laea(x, y.abs()))
        }
    }
    let south = [Geometry::Point(Coord { x: 10.0, y: -80.0 }), polygon(&[&lonlat_box(0.0, 40.0, -85.0, -70.0)])];
    let r = m.burn_from(&south, &Folded, &Laea(false), 1.0, BurnMode::Coverage).unwrap();
    assert!(r.points.is_empty() && r.runs.is_empty() && r.edges.is_empty());
    let north = [Geometry::Point(Coord { x: 10.0, y: 80.0 }), polygon(&[&lonlat_box(0.0, 40.0, 70.0, 85.0)])];
    let r = m.burn_from(&north, &Folded, &Laea(false), 1.0, BurnMode::Coverage).unwrap();
    assert!(r.points.len() == 1 && !r.runs.is_empty());
}

/// A global mesh in longitude and latitude, 10 degree cells, with a
/// declared period, and polygons in the north polar LAEA. One across the
/// antimeridian comes out of the LAEA jumping 360 degrees, and is
/// unwrapped: it burns as the same polygon turned 180 degrees about the
/// pole, 18 columns over. One round the pole is refused.
#[test]
fn a_mesh_with_a_period_unwraps_rings() {
    let (nc, nr) = (36, 18);
    let x: Vec<f64> = (0..=nr).flat_map(|_| (0..=nc).map(|i| -180.0 + 10.0 * i as f64)).collect();
    let y: Vec<f64> = (0..=nr).flat_map(|j| std::iter::repeat_n(90.0 - 10.0 * j as f64, nc + 1)).collect();
    let m = QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap().with_period(360.0).unwrap();
    // in LAEA metres, south of the pole and across 180 (negative y is
    // longitude 0, positive y is 180)
    let p = [(-1.5e6, 2.0e6), (1.0e6, 1.6e6), (1.4e6, 3.1e6), (-0.6e6, 3.4e6), (-1.5e6, 2.0e6)];
    let q: Vec<(f64, f64)> = p.iter().map(|&(x, y)| (-x, -y)).collect();
    let r = m.burn_from(&[polygon(&[&p]), polygon(&[&q])], &Laea(false), &Laea(true), 1e-6, BurnMode::Coverage).unwrap();
    let (a, b) = (cells(&r), cells(&r));
    let pa: HashMap<(i32, i32), f64> = a.iter().filter(|(k, _)| k.0 == 0).map(|(k, v)| ((k.1, (k.2 + 18) % 36), *v)).collect();
    let pb: HashMap<(i32, i32), f64> = b.iter().filter(|(k, _)| k.0 == 1).map(|(k, v)| ((k.1, k.2), *v)).collect();
    let cols: std::collections::HashSet<i32> = a.keys().filter(|k| k.0 == 0).map(|k| k.2).collect();
    assert!(cols.contains(&0) && cols.contains(&35), "it reaches both sides of the seam: {cols:?}");
    assert_eq!(pa.len(), pb.len());
    let d = pa.iter().map(|(k, v)| (v - pb[k]).abs()).fold(0.0, f64::max);
    eprintln!("across the seam vs turned: {} cells, to {d:.1e}", pa.len());
    assert!(d < 1e-6, "{d:e}");
    // a circle round the pole doesn't close in longitude
    let round: Vec<(f64, f64)> = (0..=64).map(|k| {
        let t = std::f64::consts::TAU * (k % 64) as f64 / 64.0;
        (1e6 * t.cos(), 1e6 * t.sin())
    }).collect();
    let e = m.burn_from(&[polygon(&[&round])], &Laea(false), &Laea(true), 1.0, BurnMode::Coverage).unwrap_err();
    assert!(e.contains("geometry 0") && e.contains("round a pole"), "{e}");
}

/// A `Rectilinear` grid is the same grid as a `QuadMesh` of its
/// corners: the same burn from another CRS, to rounding.
#[test]
fn rectilinear_is_the_same_grid() {
    let (nc, nr) = (30, 30);
    let xs: Vec<f64> = (0..=nc).map(|i| -3e6 + 2e5 * i as f64).collect();
    let ys: Vec<f64> = (0..=nr).map(|j| 3e6 - 2e5 * j as f64).collect();
    let rect = Rectilinear::new(xs.clone(), ys.clone()).unwrap();
    let x: Vec<f64> = (0..=nr).flat_map(|_| xs.iter().copied()).collect();
    let y: Vec<f64> = ys.iter().flat_map(|&v| std::iter::repeat_n(v, nc + 1)).collect();
    let quad = QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap();
    let g = [cap70(), polygon(&[&lonlat_box(-40.0, 75.0, 60.0, 78.0)])];
    let a = cells(&rect.burn_from(&g, &Laea(true), &Laea(false), 1.0, BurnMode::Coverage).unwrap());
    let b = cells(&quad.burn_from(&g, &Laea(true), &Laea(false), 1.0, BurnMode::Coverage).unwrap());
    assert!(a.len() > 200);
    assert!(max_diff(&a, &b) < 1e-6, "{:e}", max_diff(&a, &b));
}

/// The burn on threads: the same records.
#[test]
fn burn_from_par_is_burn_from() {
    let m = tripolar();
    let g: Vec<Geometry> = (0..24)
        .map(|k| {
            let x = -180.0 + 15.0 * k as f64;
            polygon(&[&lonlat_box(x, x + 11.0, 64.0 + (k % 5) as f64, 75.0 + (k % 7) as f64)])
        })
        .chain([cap70()])
        .collect();
    let one = m.burn_from(&g, &Laea(true), &Laea(false), 1e-3, BurnMode::Coverage).unwrap();
    for threads in [0, 4] {
        let par = m.burn_from_par(&g, &Laea(true), &Laea(false), 1e-3, BurnMode::Coverage, threads).unwrap();
        assert_eq!(par, one, "{threads} threads");
    }
}

/// What is refused: lines, a tolerance that isn't positive, modes other
/// than coverage, a polygon point that doesn't go to the world, a mesh
/// none of whose cells go back.
#[test]
fn refusals() {
    let m = tripolar();
    let line = Geometry::LineString(vec![Coord { x: 0.0, y: 70.0 }, Coord { x: 10.0, y: 80.0 }]);
    let e = m.burn_from(&[cap70(), line], &Laea(true), &Laea(false), 1.0, BurnMode::Coverage).unwrap_err();
    assert!(e.contains("geometry 1") && e.contains("lines"), "{e}");
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = m.burn_from(&[cap70()], &Laea(true), &Laea(false), tol, BurnMode::Coverage).unwrap_err();
        assert!(e.contains("tolerance"), "{e}");
    }
    assert!(m.burn_from(&[cap70()], &Laea(true), &Laea(false), 1.0, BurnMode::Approx).is_err());
    struct NorthOnly;
    impl CrsTransform for NorthOnly {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            (y > 72.0).then(|| laea(x, y))
        }
    }
    let e = m.burn_from(&[cap70(), polygon(&[&lonlat_box(0.0, 30.0, 66.0, 80.0)])], &NorthOnly, &Laea(false), 1.0, BurnMode::Coverage).unwrap_err();
    assert!(e.contains("geometry 0") && e.contains("doesn't go"), "{e}");
    struct Nowhere;
    impl CrsTransform for Nowhere {
        fn convert(&self, _: f64, _: f64) -> Option<(f64, f64)> {
            None
        }
    }
    let e = m.burn_from(&[cap70()], &Laea(true), &Nowhere, 1.0, BurnMode::Coverage).unwrap_err();
    assert!(e.contains("no cell"), "{e}");
}
