//! A mesh of polygons with a grid whose cells are in its world
//! coordinates on the other side (conservative stage 5c,
//! `design/conservative.md` section 7): `PolyMesh::weights_onto` and
//! `weights_to` through a `MeshTransformer` (a `QuadMesh`), a
//! `GenImgGeolocTransformer` (geolocation arrays) or a `SetTransformer` (a
//! `MeshSet` in several worlds).
//!
//! Oracles: each face's fractions of the cells against the cells' own
//! burn of the face as a polygon (`QuadMesh::burn`, `MeshSet::burn`,
//! controlledburn's walk, a different code); a mesh in longitude and
//! latitude with a period, and geolocation arrays with `lon_wrap`, against
//! the same cells as a regular grid; the faces tiling the cells; the two
//! directions one table.

use std::collections::HashMap;

use controlledburn::{BurnOptions, BurnResult, Coord, Geometry, Polygon};
use rwarp::geoloc_transform::{GenImgGeolocTransformer, GeolocArrays, GeolocOptions, PixelGeo};
use rwarp::mesh::{Band, CellMap, Halo, MeshSet, MeshTransformer, PolyMesh, QuadMesh, SetTransformer, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::weights::{lonlat_measure, GridWindow, WeightOptions, Weights};
use rwarp::CrsTransform;

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// Quads, triangles and hexagons (two quads' worth) in rows of three
/// kinds, on a lattice of `nx` by `ny` boxes `dx` by `dy` from `(x0, y0)`,
/// interior nodes moved by up to `jitter` of a box.
fn mixed(x0: f64, y0: f64, dx: f64, dy: f64, nx: usize, ny: usize, jitter: f64) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    let n = |i: usize, k: usize| k * (nx + 1) + i;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..=nx {
            let inner = i > 0 && i < nx && k > 0 && k < ny;
            let (fi, fk) = (i as f64, k as f64);
            let (jx, jy) = if inner { ((1.7 * fi + 2.3 * fk).sin(), (1.3 * fi - 0.7 * fk).cos()) } else { (0.0, 0.0) };
            x.push(x0 + dx * (fi + jitter * jx));
            y.push(y0 + dy * (fk + jitter * jy));
        }
    }
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        let mut i = 0;
        while i < nx {
            let (a, b, c, d) = (n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1));
            let faces: Vec<Vec<usize>> = match k % 3 {
                1 if (i + k) % 2 == 0 => vec![vec![a, b, c], vec![a, c, d]],
                1 => vec![vec![a, b, d], vec![b, c, d]],
                2 if i + 1 < nx => {
                    i += 1;
                    vec![vec![a, b, n(i + 1, k), n(i + 1, k + 1), c, d]]
                }
                _ => vec![vec![a, b, c, d]],
            };
            for f in faces {
                nodes.extend(f);
                start.push(nodes.len());
            }
            i += 1;
        }
    }
    (x, y, start, nodes)
}

/// Face `f` as a polygon for a burn, which takes it as written: with the
/// mesh's period, each node within half a period of the one before.
fn polygon(m: &PolyMesh, f: usize) -> Geometry {
    let mut ring: Vec<Coord> = Vec::new();
    for &n in m.face(f) {
        let [mut x, y] = m.node(n);
        if let (Some(p), Some(prev)) = (m.period(), ring.last()) {
            x -= p * ((x - prev.x) / p).round();
        }
        ring.push(Coord::new(x, y));
    }
    ring.push(ring[0]);
    Geometry::Polygon(Polygon::new(vec![ring]))
}

/// A burn's fractions, keyed by (id, row, col).
fn burned(r: &BurnResult) -> HashMap<(u32, u32, u32), f64> {
    let mut m = HashMap::new();
    for e in &r.edges {
        *m.entry((e.id as u32, e.row as u32, e.col as u32)).or_insert(0.0) += e.fraction as f64;
    }
    for run in &r.runs {
        for c in run.col_start..run.col_end {
            *m.entry((run.id as u32, run.row as u32, c as u32)).or_insert(0.0) += 1.0;
        }
    }
    m
}

/// Onto's entries keyed by (face, row, col).
fn onto_fracs(w: &Weights) -> HashMap<(u32, u32, u32), f64> {
    (0..w.len()).map(|k| ((w.dst_col[k], w.src_row[k], w.src_col[k]), w.frac[k])).collect()
}

/// The worst difference of two tables, an entry in one only counting whole.
fn max_diff(a: &HashMap<(u32, u32, u32), f64>, b: &HashMap<(u32, u32, u32), f64>) -> f64 {
    a.iter()
        .map(|(k, v)| (v - b.get(k).copied().unwrap_or(0.0)).abs())
        .chain(b.iter().filter(|(k, _)| !a.contains_key(k)).map(|(_, v)| v.abs()))
        .fold(0.0, f64::max)
}

/// Onto's and to's overlap measures, entry for entry.
fn transposed(onto: &Weights, to: &Weights) -> f64 {
    let a: HashMap<_, _> = (0..onto.len()).map(|k| ((onto.dst_col[k], onto.src_row[k], onto.src_col[k]), onto.frac[k] * onto.src_area(k))).collect();
    let b: HashMap<_, _> = (0..to.len()).map(|k| ((to.src_col[k], to.dst_row[k], to.dst_col[k]), to.frac[k] * to.src_area(k))).collect();
    assert_eq!(a.len(), b.len());
    a.iter().map(|(k, v)| (v - b[k]).abs() / v.abs().max(1.0)).fold(0.0, f64::max)
}

// ---- a curvilinear mesh ------------------------------------------------------------

/// 30 by 24 cells of a curved lattice.
fn curved() -> QuadMesh {
    let (nc, nr) = (30, 24);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nr {
        for i in 0..=nc {
            let (fi, fj) = (i as f64, j as f64);
            x.push(fi + 0.3 * (0.4 * fj).sin() + 0.02 * fj);
            y.push(fj + 0.25 * (0.3 * fi).cos() - 0.03 * fi);
        }
    }
    QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea).unwrap()
}

/// Twice the signed area of a ring (shoelace).
fn area2(p: &[[f64; 2]]) -> f64 {
    (0..p.len()).map(|i| p[i][0] * p[(i + 1) % p.len()][1] - p[(i + 1) % p.len()][0] * p[i][1]).sum()
}

/// `subject` clipped to the convex `clip` (Sutherland and Hodgman): the
/// area of their overlap, for any simple subject.
fn overlap(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> f64 {
    let mut c = clip.to_vec();
    if area2(&c) < 0.0 {
        c.reverse();
    }
    let mut out = subject.to_vec();
    for i in 0..c.len() {
        let (a, b) = (c[i], c[(i + 1) % c.len()]);
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let input = std::mem::take(&mut out);
        for j in 0..input.len() {
            let (p, q) = (input[j], input[(j + 1) % input.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            return 0.0;
        }
    }
    area2(&out).abs() / 2.0
}

/// The mixed mesh onto a curvilinear `QuadMesh` in the same plane: each
/// face's fractions are its overlap with each cell, clipped here, over
/// the cell's area, to 1e-12 (and the cells' burn of the faces gives the
/// same to its `f32`, but for two slivers of 3e-7 at a corner it leaves
/// out); to is onto turned round.
#[test]
fn onto_a_curvilinear_mesh_is_its_overlaps() {
    let q = curved();
    let (x, y, start, nodes) = mixed(-1.3, 2.2, 1.7, 1.4, 16, 12, 0.25);
    let m = PolyMesh::new(x, y, start, nodes).unwrap();
    let t = MeshTransformer::with_backend(q.clone(), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let grid = t.source_window();
    for opts in [WeightOptions::new(1), WeightOptions::new(3).with_refine_tolerance(1e-6)] {
        let onto = m.weights_onto(&t, &grid, &opts).unwrap();
        let geoms: Vec<Geometry> = (0..m.nface()).map(|f| polygon(&m, f)).collect();
        let burn = burned(&q.burn(&geoms, BurnOptions::coverage()).unwrap());
        let d = max_diff(&onto_fracs(&onto), &burn);
        assert!(d < 4e-7, "{d:e}");
        // every face against every cell its box meets
        let mut clipped = HashMap::new();
        for f in 0..m.nface() {
            let ring: Vec<[f64; 2]> = m.face(f).iter().map(|&n| m.node(n)).collect();
            for r in 0..24 {
                for c in 0..30 {
                    let a = overlap(&ring, &q.corners(r, c)) / q.cell_area(r, c);
                    if a > 0.0 {
                        clipped.insert((f as u32, r as u32, c as u32), a);
                    }
                }
            }
        }
        let d = max_diff(&onto_fracs(&onto), &clipped);
        eprintln!("{opts:?}: fractions to {d:.1e} of the clipped overlaps");
        assert!(d < 1e-12, "{d:e}");
        // cells are measured in the plane, and the faces too
        assert!((0..onto.len()).all(|k| onto.src_cell_area[k] == q.cell_area(onto.src_row[k] as usize, onto.src_col[k] as usize)));
        for f in 0..m.nface() {
            let v = m.face(f);
            let a = (0..v.len()).map(|i| {
                let (p, q) = (m.node(v[i]), m.node(v[(i + 1) % v.len()]));
                p[0] * q[1] - q[0] * p[1]
            });
            let a = a.sum::<f64>().abs() / 2.0;
            assert!((onto.dst_area[f] - a).abs() < 1e-12 * a, "face {f}");
        }
        let to = m.weights_to(&t, &grid, &opts).unwrap();
        assert!(transposed(&onto, &to) < 1e-14);
        assert_eq!(to.dst_area.len(), 30 * 24);
        assert!((0..30 * 24).all(|k| to.dst_area[k] == q.cell_area(k / 30, k % 30)));
    }
}

// ---- in longitude and latitude, with a period ---------------------------------------

/// A 5 degree global lattice from -180 as a mesh with a period.
fn lonlat_mesh() -> QuadMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=36 {
        for i in 0..=72 {
            x.push(-180.0 + 5.0 * i as f64);
            y.push(90.0 - 5.0 * j as f64);
        }
    }
    QuadMesh::from_corners(72, 36, x, y, CellMap::EqualArea).unwrap().with_period(360.0).unwrap()
}

/// Boxes 7 by 4 degrees across 180 and a cap round the north pole, in
/// longitude and latitude with a period.
fn seam_boxes() -> PolyMesh {
    let (mut x, mut y, mut start, mut nodes) = mixed(160.0, -21.0, 7.0, 4.0, 7, 9, 0.2);
    for v in &mut x {
        if *v > 180.0 {
            *v -= 360.0;
        }
    }
    let base = x.len();
    for k in 0..12 {
        x.push(-180.0 + 30.0 * k as f64);
        y.push(77.0);
        nodes.push(base + k);
    }
    start.push(nodes.len());
    PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap()
}

/// Across 180 and round the pole, the mesh with a period gives the
/// fractions of the same cells as a regular grid (planar in longitude and
/// latitude both), to rounding.
#[test]
fn a_lonlat_mesh_with_a_period_is_the_regular_grid() {
    let m = seam_boxes();
    let q = MeshTransformer::with_backend(lonlat_mesh(), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let gt = [-180.0, 5.0, 0.0, 90.0, 0.0, -5.0];
    let g = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([72, 36]);
    let grid = GridWindow::new(72, 36);
    for opts in [WeightOptions::new(2), WeightOptions::new(1).with_refine_tolerance(1e-4)] {
        let a = m.weights_onto(&q, &grid, &opts).unwrap();
        let b = m.weights_onto(&g, &grid, &opts).unwrap();
        let d = max_diff(&onto_fracs(&a), &onto_fracs(&b));
        assert!(d < 1e-12, "{d:e}");
        // areas: square degrees, and pixels of 25 square degrees
        let area = a.dst_area.iter().zip(&b.dst_area).map(|(p, q)| (p / 25.0 - q).abs()).fold(0.0, f64::max);
        assert!(area < 1e-12, "{area:e}");
        // the cap's cells, north of 77 N, partly and wholly
        let cap = m.nface() - 1;
        assert!(a.dst_col.iter().zip(&a.frac).filter(|(&f, &v)| f as usize == cap && v == 1.0).count() >= 72 * 2);
    }
}

/// Geolocation arrays with `lon_wrap`, sampled at the same corners
/// (stored from -180, so the cells across 180 jump), give the mesh's
/// weights.
#[test]
fn geolocation_arrays_are_their_mesh() {
    let m = seam_boxes();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=36 {
        for i in 0..=72 {
            let lon = -180.0 + 5.0 * i as f64;
            x.push(if lon >= 180.0 { lon - 360.0 } else { lon });
            y.push(90.0 - 5.0 * j as f64);
        }
    }
    let g = GeolocArrays::new(x, y, 73, 37, GeolocOptions::default().with_lon_wrap(true)).unwrap();
    assert!(g.world_cells().is_ok());
    let geo = GenImgGeolocTransformer::with_backend(PixelGeo::Geoloc(g), PixelGeo::affine(WORLD_GT).unwrap(), Box::new(Same), Box::new(Same));
    let q = MeshTransformer::with_backend(lonlat_mesh(), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let grid = GridWindow::new(72, 36);
    let opts = WeightOptions::new(2);
    let (a, b) = (m.weights_onto(&geo, &grid, &opts).unwrap(), m.weights_onto(&q, &grid, &opts).unwrap());
    let d = max_diff(&onto_fracs(&a), &onto_fracs(&b));
    assert!(d < 1e-12, "{d:e}");
    let to = m.weights_to(&geo, &grid, &opts).unwrap();
    assert!(transposed(&a, &to) < 1e-14);
}

// ---- a set in two worlds -------------------------------------------------------------

const R: f64 = 6371229.0;

/// Longitude and latitude to south-polar (`0`) or north-polar (`1`)
/// LAEA, or back with `.1`.
struct World(usize, bool);

impl CrsTransform for World {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let s = if self.0 == 0 { -1.0 } else { 1.0 };
        Some(if !self.1 {
            let rho = 2.0 * R * ((90.0 - s * y).to_radians() / 2.0).sin();
            (rho * x.to_radians().sin(), -s * rho * x.to_radians().cos())
        } else {
            let rho = x.hypot(y);
            (x.atan2(-s * y).to_degrees(), s * (90.0 - 2.0 * (rho / (2.0 * R)).min(1.0).asin().to_degrees()))
        })
    }
}

/// 36 columns of 10 degrees, rows of 6 degrees from 78 S to 78 N, split
/// at the equator into a south-polar and a north-polar band.
fn set() -> MeshSet {
    let (nc, nr) = (36, 26);
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    for j in 0..nr {
        for i in 0..nc {
            for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)] {
                xb.push(-180.0 + 10.0 * ii as f64);
                yb.push(-78.0 + 6.0 * jj as f64);
            }
        }
    }
    let (s, n) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..13, to_world: &s }, Band { rows: 13..26, to_world: &n }];
    MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap()
}

/// Boxes 8 by 5 degrees all round from 84 S to 84 N, every other one
/// written the other way round, and a cap round each pole on their
/// nodes, with a period. The caps are `BOXES` and `BOXES + 1`.
const BOXES: usize = 45 * 34;

fn boxes_all_round() -> PolyMesh {
    let (nx, ny) = (45, 34);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..nx {
            x.push(-177.0 + 8.0 * i as f64);
            y.push(-84.0 + 5.0 * k as f64 + 0.7 * (i as f64 * 0.9).sin() * (k > 0 && k < ny) as u8 as f64);
        }
    }
    let n = |i: usize, k: usize| k * nx + i % nx;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        for i in 0..nx {
            let mut f = [n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)];
            if (i + k) % 2 == 1 {
                f.reverse();
            }
            nodes.extend(f);
            start.push(nodes.len());
        }
    }
    nodes.extend((0..nx).map(|i| n(i, ny)));
    start.push(nodes.len());
    nodes.extend((0..nx).rev().map(|i| n(i, 0)));
    start.push(nodes.len());
    for v in &mut x {
        if *v >= 180.0 {
            *v -= 360.0;
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap()
}

/// Onto a set in two polar LAEAs: each face's fractions are the set's
/// burn of it (densified into each band's world to the same tolerance,
/// differently cut), to 1e-6 (8e-7: the burn's `f32` and the slivers it
/// leaves out, as on a curvilinear mesh); every cell of the set is covered
/// once over, by faces written either way round (a face near one pole,
/// inside out round the other in the far world, is left out there, the
/// caps round the poles too); to is onto turned round.
#[test]
fn onto_a_set_is_its_burn() {
    let s = set();
    let m = boxes_all_round();
    let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> =
        vec![(Box::new(World(0, false)), Box::new(World(0, true))), (Box::new(World(1, false)), Box::new(World(1, true)))];
    let t = SetTransformer::with_backend(&s, WORLD_GT, crs).unwrap();
    let grid = t.source_window();
    let opts = WeightOptions::new(1).with_refine_tolerance(1.0);
    let onto = m.weights_onto(&t, &grid, &opts).unwrap();
    // the caps don't close as polygons in longitude and latitude
    let geoms: Vec<Geometry> = (0..BOXES).map(|f| polygon(&m, f)).collect();
    let (ws, wn) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..13, to_world: &ws }, Band { rows: 13..26, to_world: &wn }];
    let burn = burned(&s.burn(&geoms, &bands, 1.0, BurnOptions::coverage()).unwrap());
    let mut ours = onto_fracs(&onto);
    assert!(ours.keys().all(|k| (k.0 as usize) < BOXES), "a cap beyond 84 reaches the set");
    ours.retain(|k, _| (k.0 as usize) < BOXES);
    let d = max_diff(&ours, &burn);
    eprintln!("fractions to {d:.1e} of the set's burn");
    assert!(d < 1e-6, "{d:e}");
    let mut per_cell = vec![0.0; 36 * 26];
    for k in 0..onto.len() {
        per_cell[onto.src_row[k] as usize * 36 + onto.src_col[k] as usize] += onto.frac[k];
    }
    let tiles = per_cell.iter().map(|s| (s - 1.0).abs()).fold(0.0, f64::max);
    assert!(tiles < 1e-12, "{tiles:e}");
    let to = m.weights_to(&t, &grid, &opts).unwrap();
    assert!(transposed(&onto, &to) < 1e-12);
}

/// With world cells the measure is planar in their world.
#[test]
fn a_measure_with_world_cells_is_refused() {
    let m = seam_boxes();
    let q = MeshTransformer::with_backend(lonlat_mesh(), WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let opts = WeightOptions::new(1).with_measure(lonlat_measure([-180.0, 5.0, 0.0, 90.0, 0.0, -5.0]).unwrap());
    let e = m.weights_onto(&q, &GridWindow::new(72, 36), &opts).unwrap_err();
    assert!(e.contains("Planar"), "{e}");
}


/// Every value of a table, bit for bit.
fn bits(w: &Weights) -> Vec<u64> {
    let mut v: Vec<u64> = [w.dst_off, w.dst_size].iter().flatten().map(|&x| x as u64).collect();
    for ix in [&w.dst_row, &w.dst_col, &w.src_row, &w.src_col] {
        v.push(ix.len() as u64);
        v.extend(ix.iter().map(|&x| x as u64));
    }
    for f in [&w.frac, &w.dst_area, &w.src_cell_area] {
        v.push(f.len() as u64);
        v.extend(f.iter().map(|x| x.to_bits()));
    }
    v
}

/// On threads, the serial table bit for bit, both ways: onto a curvilinear
/// mesh, geolocation arrays across 180, a set in two worlds.
#[test]
fn threads_are_the_serial_table() {
    let (x, y, start, nodes) = mixed(-1.3, 2.2, 1.7, 1.4, 16, 12, 0.25);
    let curved_faces = PolyMesh::new(x, y, start, nodes).unwrap();
    let quad = || MeshTransformer::with_backend(curved(), WORLD_GT, Box::new(Same), Box::new(Same));
    let seam = seam_boxes();
    let geo = || {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for j in 0..=36 {
            for i in 0..=72 {
                let lon = -180.0 + 5.0 * i as f64;
                x.push(if lon >= 180.0 { lon - 360.0 } else { lon });
                y.push(90.0 - 5.0 * j as f64);
            }
        }
        let g = GeolocArrays::new(x, y, 73, 37, GeolocOptions::default().with_lon_wrap(true))?;
        Ok::<_, String>(GenImgGeolocTransformer::with_backend(PixelGeo::Geoloc(g), PixelGeo::affine(WORLD_GT)?, Box::new(Same), Box::new(Same)))
    };
    let s = set();
    let all_round = boxes_all_round();
    let on_set = || {
        let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> =
            vec![(Box::new(World(0, false)), Box::new(World(0, true))), (Box::new(World(1, false)), Box::new(World(1, true)))];
        SetTransformer::with_backend(&s, WORLD_GT, crs)
    };
    let refined = WeightOptions::new(1).with_refine_tolerance(1e-6);
    for threads in [2, 3] {
        let g = quad().unwrap().source_window();
        let (a, b) = (curved_faces.weights_onto(&quad().unwrap(), &g, &refined), curved_faces.weights_onto_par(quad, &g, &refined, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "curved onto, {threads} threads");
        let (a, b) = (curved_faces.weights_to(&quad().unwrap(), &g, &refined), curved_faces.weights_to_par(quad, &g, &refined, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "curved to, {threads} threads");
        let (g, o) = (GridWindow::new(72, 36), WeightOptions::new(2));
        let (a, b) = (seam.weights_onto(&geo().unwrap(), &g, &o), seam.weights_onto_par(geo, &g, &o, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "geolocation onto, {threads} threads");
        let (a, b) = (seam.weights_to(&geo().unwrap(), &g, &o), seam.weights_to_par(geo, &g, &o, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "geolocation to, {threads} threads");
        let (g, o) = (on_set().unwrap().source_window(), WeightOptions::new(1).with_refine_tolerance(10.0));
        let (a, b) = (all_round.weights_onto(&on_set().unwrap(), &g, &o), all_round.weights_onto_par(on_set, &g, &o, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "set onto, {threads} threads");
        let (a, b) = (all_round.weights_to(&on_set().unwrap(), &g, &o), all_round.weights_to_par(on_set, &g, &o, threads));
        assert!(bits(&a.unwrap()) == bits(&b.unwrap()), "set to, {threads} threads");
    }
}
