//! Conservative weights between a mesh of polygons and a grid, both ways
//! (conservative stage 5, `design/conservative.md` section 7):
//! `PolyMesh::weights_onto` and `PolyMesh::weights_to`.
//!
//! Oracles: a mesh of the grid's own cells is the identity, and of its
//! cells cut on a diagonal halves; boxes in longitude and latitude
//! against the closed form of their overlaps on the sphere, across the
//! seam and round the poles; boxes in an equal-area projection against
//! their area on the sphere, converging at bigcurve's rate; and, on a
//! mixed mesh, the faces tiling the grid (every covered cell's overlaps
//! sum to it) and the two directions being one table.

use std::cell::Cell;
use std::collections::HashMap;
use std::f64::consts::PI;

use rwarp::mesh::{CellMap, MeshTransformer, PolyMesh, QuadMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer, SourceWrap};
use rwarp::warp::SourceBuffer;
use rwarp::weights::{lonlat_measure, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// From the mesh's coordinates to the pixels of a grid `gt` in the same
/// CRS.
fn same(gt: [f64; 6]) -> GenImgProjTransformer {
    GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap()
}

/// [`same`] for a grid in longitude and latitude, `size` pixels, its
/// longitudes wrapped round `center` as GDAL's are.
fn lonlat(gt: [f64; 6], size: [usize; 2], center: f64) -> GenImgProjTransformer {
    same(gt).with_center_long(CenterLong::At(center)).with_source_size(size)
}

/// A lattice of `nx` by `ny` boxes from `(x0, y0)`, `dx` by `dy` each
/// (y up), as nodes, and its faces: node `(i, k)` is `k * (nx + 1) + i`.
fn lattice(x0: f64, y0: f64, dx: f64, dy: f64, nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..=nx {
            x.push(x0 + dx * i as f64);
            y.push(y0 + dy * k as f64);
        }
    }
    (x, y)
}

/// The boxes of a lattice as quads, anticlockwise with y up.
fn quads(nx: usize, ny: usize) -> (Vec<usize>, Vec<usize>) {
    let n = |i: usize, k: usize| k * (nx + 1) + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        for i in 0..nx {
            nodes.extend([n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)]);
            start.push(nodes.len());
        }
    }
    (start, nodes)
}

/// Twice the signed area of face `f` (shoelace).
fn shoelace(m: &PolyMesh, f: usize) -> f64 {
    let v = m.face(f);
    (0..v.len())
        .map(|i| {
            let (a, b) = (m.node(v[i]), m.node(v[(i + 1) % v.len()]));
            a[0] * b[1] - b[0] * a[1]
        })
        .sum()
}

/// Entries keyed by (face, grid row, grid col): onto's overlap measures.
fn onto_overlaps(w: &Weights) -> HashMap<(u32, u32, u32), f64> {
    (0..w.len()).map(|k| ((w.dst_col[k], w.src_row[k], w.src_col[k]), w.frac[k] * w.src_area(k))).collect()
}

/// The same from weights_to.
fn to_overlaps(w: &Weights) -> HashMap<(u32, u32, u32), f64> {
    (0..w.len()).map(|k| ((w.src_col[k], w.dst_row[k], w.dst_col[k]), w.frac[k] * w.src_area(k))).collect()
}

/// Per grid cell of an `ncol`-wide grid, the sum of onto's `frac` over
/// faces.
fn frac_per_cell(w: &Weights, ncol: usize, ncell: usize) -> Vec<f64> {
    let mut s = vec![0.0; ncell];
    for k in 0..w.len() {
        s[w.src_row[k] as usize * ncol + w.src_col[k] as usize] += w.frac[k];
    }
    s
}

// ---- the grid's own cells ------------------------------------------------------

const GT: [f64; 6] = [100.0, 10.0, 0.0, 50.0, 0.0, -5.0];

/// A mesh of the cells of `GT`'s 6 by 4 grid is the identity both ways:
/// one entry per face, 1, on its own cell.
#[test]
fn the_grid_s_own_cells_are_the_identity() {
    let (x, y) = lattice(100.0, 30.0, 10.0, 5.0, 6, 4);
    let (start, nodes) = quads(6, 4);
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let (t, grid) = (same(GT), GridWindow::new(6, 4));
    for d in [1, 3] {
        let onto = mesh.weights_onto(&t, &grid, &WeightOptions::new(d)).unwrap();
        assert_eq!(onto.len(), 24);
        for k in 0..24 {
            // face (k, i) of the lattice, y up, is grid cell (3 - k, i)
            let f = onto.dst_col[k] as usize;
            assert_eq!((onto.src_row[k], onto.src_col[k]), ((3 - f / 6) as u32, (f % 6) as u32));
            assert!((onto.frac[k] - 1.0).abs() < 1e-15, "{}", onto.frac[k]);
        }
        assert!(onto.dst_area.iter().all(|a| (a - 1.0).abs() < 1e-15));
        let to = mesh.weights_to(&t, &grid, &WeightOptions::new(d)).unwrap();
        assert_eq!(to.len(), 24);
        for k in 0..24 {
            let cell = to.dst_row[k] as usize * 6 + to.dst_col[k] as usize;
            assert_eq!(cell, k, "grouped by cell, row-major");
            assert_eq!(to.src_col[k] as usize, (3 - k / 6) * 6 + k % 6);
            assert!((to.frac[k] - 1.0).abs() < 1e-15 && (to.src_cell_area[k] - 1.0).abs() < 1e-15);
        }
        assert_eq!(to.dst_area, vec![1.0; 24]);
    }
}

/// Each cell cut on a diagonal: each triangle is half its cell.
#[test]
fn triangles_halve_their_cells() {
    let (x, y) = lattice(100.0, 30.0, 10.0, 5.0, 6, 4);
    let n = |i: usize, k: usize| k * 7 + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..4 {
        for i in 0..6 {
            let (a, b, c, d) = (n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1));
            let tris = if (i + k) % 2 == 0 { [[a, b, c], [a, c, d]] } else { [[a, b, d], [b, c, d]] };
            for t in tris {
                nodes.extend(t);
                start.push(nodes.len());
            }
        }
    }
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let onto = mesh.weights_onto(&same(GT), &GridWindow::new(6, 4), &WeightOptions::new(1)).unwrap();
    assert_eq!(onto.len(), 48);
    assert!(onto.frac.iter().all(|f| (f - 0.5).abs() < 1e-15));
    assert!(onto.dst_area.iter().all(|a| (a - 0.5).abs() < 1e-15));
    let to = mesh.weights_to(&same(GT), &GridWindow::new(6, 4), &WeightOptions::new(1)).unwrap();
    assert!(to.frac.iter().all(|f| (f - 1.0).abs() < 1e-15));
    assert!(to.dst_covered().iter().all(|c| (c - 1.0).abs() < 1e-15));
}

// ---- a mixed mesh, jittered --------------------------------------------------------

/// Quads, triangles and hexagons (two quads' worth) in rows of three
/// kinds, on a lattice of `nx` by `ny` boxes 1.2 by 0.9 from (9, 11.3),
/// interior nodes moved by up to a quarter of a box.
fn mixed(nx: usize, ny: usize) -> PolyMesh {
    let (mut x, mut y) = lattice(9.0, 11.3, 1.2, 0.9, nx, ny);
    for k in 1..ny {
        for i in 1..nx {
            let (fi, fk) = (i as f64, k as f64);
            x[k * (nx + 1) + i] += 0.3 * (1.7 * fi + 2.3 * fk).sin();
            y[k * (nx + 1) + i] += 0.225 * (1.3 * fi - 0.7 * fk).cos();
        }
    }
    let n = |i: usize, k: usize| k * (nx + 1) + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        let mut i = 0;
        while i < nx {
            let (a, b, c, d) = (n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1));
            match k % 3 {
                0 => {
                    nodes.extend([a, b, c, d]);
                    start.push(nodes.len());
                }
                1 => {
                    let tris = if (i + k) % 2 == 0 { [[a, b, c], [a, c, d]] } else { [[a, b, d], [b, c, d]] };
                    for t in tris {
                        nodes.extend(t);
                        start.push(nodes.len());
                    }
                }
                _ if i + 1 < nx => {
                    nodes.extend([a, b, n(i + 2, k), n(i + 2, k + 1), c, d]);
                    start.push(nodes.len());
                    i += 1;
                }
                _ => {
                    nodes.extend([a, b, c, d]);
                    start.push(nodes.len());
                }
            }
            i += 1;
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap()
}

/// Half-unit cells over [10, 22] x [10, 20]: the mixed mesh, over [9,
/// 21] x [11.3, 19.4], hangs over its left edge.
const MIXED_GT: [f64; 6] = [10.0, 0.5, 0.0, 20.0, 0.0, -0.5];
const MIXED: (usize, usize) = (24, 20);

/// The faces tile the grid: every cell inside the mesh's rectangle has
/// overlaps summing to it, and every face inside the grid to the face,
/// its area its shoelace area. Onto and to are one table.
#[test]
fn a_mixed_mesh_tiles_the_grid_both_ways() {
    let mesh = mixed(10, 9);
    let (t, grid) = (same(MIXED_GT), GridWindow::new(MIXED.0, MIXED.1));
    for opts in [WeightOptions::new(1), WeightOptions::new(3)] {
        let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
        let per_cell = frac_per_cell(&onto, MIXED.0, MIXED.0 * MIXED.1);
        let mut inside = 0;
        for (cell, s) in per_cell.iter().enumerate() {
            let (r, c) = ((cell / MIXED.0) as f64, (cell % MIXED.0) as f64);
            let (x0, y1) = (10.0 + 0.5 * c, 20.0 - 0.5 * r);
            if x0 >= 9.0 && x0 + 0.5 <= 21.0 && y1 - 0.5 >= 11.3 && y1 <= 19.4 {
                inside += 1;
                assert!((s - 1.0).abs() < 1e-13, "cell {cell}: {s}");
            }
        }
        assert_eq!(inside, 22 * 15);
        for (f, covered) in onto.dst_covered().into_iter().enumerate() {
            // the shoelace in world units; a pixel is 0.25 of them
            let area = shoelace(&mesh, f).abs() / 2.0 / 0.25;
            assert!((onto.dst_area[f] - area).abs() < 1e-12 * area, "face {f}");
            if mesh.face(f).iter().all(|&n| mesh.node(n)[0] >= 10.0) {
                assert!((covered - area).abs() < 1e-12 * area, "face {f}");
            } else {
                assert!(covered < area * (1.0 - 1e-6), "face {f} hangs over the edge");
            }
        }
        let to = mesh.weights_to(&t, &grid, &opts).unwrap();
        let (a, b) = (onto_overlaps(&onto), to_overlaps(&to));
        assert_eq!(a.len(), b.len());
        for (k, v) in &a {
            assert!((v - b[k]).abs() < 1e-14, "{k:?}: {v} {}", b[k]);
        }
        // grouped by cell, row-major, faces in order within a cell
        let key = |k: usize| (to.dst_row[k], to.dst_col[k], to.src_col[k]);
        assert!((1..to.len()).all(|k| key(k - 1) < key(k)));
    }
}

/// Applying the weights: the integral of a field on the faces is kept
/// onto the grid (destination area normalisation) and back, and a
/// constant stays constant (fraction normalisation).
#[test]
fn apply_keeps_the_integral() {
    let mesh = mixed(10, 9);
    let (t, grid) = (same(MIXED_GT), GridWindow::new(MIXED.0, MIXED.1));
    let opts = WeightOptions::new(2);
    let to = mesh.weights_to(&t, &grid, &opts).unwrap();
    let nf = mesh.nface();
    let field: Vec<f64> = (0..nf).map(|f| 1.0 + (f as f64 * 0.37).sin()).collect();
    let faces = SourceBuffer { data: &field, ncol: nf, nrow: 1, nbands: 1, col_off: 0, row_off: 0 };
    let on_grid: Vec<f64> = to.apply(&faces, Normalization::DestArea, &ApplyOptions::new());
    // the faces' integral inside the grid
    let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
    let inside: f64 = onto.dst_covered().iter().zip(&field).map(|(a, v)| a * v).sum();
    let grid_sum: f64 = on_grid.iter().zip(&to.dst_area).map(|(v, a)| v * a).sum();
    assert!((grid_sum - inside).abs() < 1e-12 * inside, "{grid_sum} {inside}");
    // a constant on the faces is that constant wherever they are
    let ones = vec![2.5; nf];
    let faces = SourceBuffer { data: &ones, ..faces };
    let c: Vec<f64> = to.apply(&faces, Normalization::FracArea, &ApplyOptions::new());
    let covered = to.dst_covered();
    for (k, v) in c.iter().enumerate() {
        if covered[k] > 0.0 {
            assert!((v - 2.5).abs() < 1e-14, "cell {k}: {v}");
        }
    }
    // and back onto the faces, the grid's integral under them
    let cells: Vec<f64> = (0..MIXED.0 * MIXED.1).map(|k| (k as f64 * 0.11).cos()).collect();
    let src = SourceBuffer { data: &cells, ncol: MIXED.0, nrow: MIXED.1, nbands: 1, col_off: 0, row_off: 0 };
    let on_faces: Vec<f64> = onto.apply(&src, Normalization::DestArea, &ApplyOptions::new());
    let face_sum: f64 = on_faces.iter().zip(&onto.dst_area).map(|(v, a)| v * a).sum();
    let under: f64 = frac_per_cell(&onto, MIXED.0, cells.len()).iter().zip(&cells).map(|(f, v)| f * v).sum();
    assert!((face_sum - under).abs() < 1e-12 * under.abs(), "{face_sum} {under}");
}

// ---- longitude and latitude, on the sphere -----------------------------------------

/// Steradians per pixel of a grid `d` degrees square, under the
/// spherical measure (whose areas are in pixels at the equator).
fn sr(d: f64) -> f64 {
    d.to_radians() * d.to_radians()
}

/// The area on the unit sphere of the box [lon0, lon1] x [lat0, lat1].
fn box_sr(lon: [f64; 2], lat: [f64; 2]) -> f64 {
    (lon[1] - lon[0]).to_radians() * (lat[1].to_radians().sin() - lat[0].to_radians().sin())
}

/// Boxes 2.5 by 2 degrees from (101.3, -41.7) over 1 degree cells from
/// (100, -10): each overlap is the closed form of the two boxes' overlap
/// on the sphere, and each face its own area.
#[test]
fn boxes_on_the_sphere_are_exact() {
    let gt = [100.0, 1.0, 0.0, -10.0, 0.0, -1.0];
    let (x, y) = lattice(101.3, -41.7, 2.5, 2.0, 20, 15);
    let (start, nodes) = quads(20, 15);
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let opts = WeightOptions::new(1).with_measure(lonlat_measure(gt).unwrap());
    let onto = mesh.weights_onto(&same(gt), &GridWindow::new(60, 40), &opts).unwrap();
    let mut worst: f64 = 0.0;
    for k in 0..onto.len() {
        let f = onto.dst_col[k] as usize;
        let (i, j) = (f % 20, f / 20);
        let (r, c) = (onto.src_row[k] as f64, onto.src_col[k] as f64);
        let lon = [(101.3 + 2.5 * i as f64).max(100.0 + c), (101.3 + 2.5 * (i + 1) as f64).min(101.0 + c)];
        let lat = [(-41.7 + 2.0 * j as f64).max(-11.0 - r), (-41.7 + 2.0 * (j + 1) as f64).min(-10.0 - r)];
        let got = onto.frac[k] * onto.src_area(k) * sr(1.0);
        worst = worst.max((got - box_sr(lon, lat)).abs());
    }
    assert!(worst < 1e-16, "{worst:e}");
    for f in 0..mesh.nface() {
        let (i, j) = ((f % 20) as f64, (f / 20) as f64);
        let want = box_sr([101.3 + 2.5 * i, 103.8 + 2.5 * i], [-41.7 + 2.0 * j, -39.7 + 2.0 * j]);
        assert!((onto.dst_area[f] * sr(1.0) - want).abs() < 1e-16, "face {f}");
    }
    // every cell under the boxes, wholly
    let per_cell = frac_per_cell(&onto, 60, 2400);
    for (cell, s) in per_cell.iter().enumerate() {
        let (r, c) = (cell / 60, cell % 60);
        if (2..=50).contains(&c) && (2..=30).contains(&r) {
            assert!((s - 1.0).abs() < 1e-14, "cell {cell}: {s}");
        }
    }
}

/// A window of the grid gives the whole grid's entries in it, both ways,
/// under the spherical measure; onto the mesh each face keeps its whole
/// area, and from it each cell its own.
#[test]
fn a_window_is_the_grid_s_entries_in_it() {
    let gt = [100.0, 1.0, 0.0, -10.0, 0.0, -1.0];
    let (x, y) = lattice(101.3, -41.7, 2.5, 2.0, 20, 15);
    let (start, nodes) = quads(20, 15);
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let opts = WeightOptions::new(2).with_measure(lonlat_measure(gt).unwrap());
    let (t, whole) = (same(gt), GridWindow::new(60, 40));
    let part = GridWindow::new(17, 11).with_offset(9, 13);
    let inside = |r: u32, c: u32| (13..24).contains(&r) && (9..26).contains(&c);
    let (a, b) = (mesh.weights_onto(&t, &whole, &opts).unwrap(), mesh.weights_onto(&t, &part, &opts).unwrap());
    let ea: HashMap<_, _> = onto_overlaps(&a).into_iter().filter(|(k, _)| inside(k.1, k.2)).collect();
    let eb = onto_overlaps(&b);
    assert_eq!(ea.len(), eb.len());
    assert!(ea.iter().all(|(k, v)| (v - eb[k]).abs() < 1e-15), "onto");
    assert!(a.dst_area.iter().zip(&b.dst_area).all(|(x, y)| (x - y).abs() < 1e-14 * x));
    let (a, b) = (mesh.weights_to(&t, &whole, &opts).unwrap(), mesh.weights_to(&t, &part, &opts).unwrap());
    let ea: HashMap<_, _> = to_overlaps(&a).into_iter().filter(|(k, _)| inside(k.1, k.2)).collect();
    let eb = to_overlaps(&b);
    assert_eq!(ea.len(), eb.len());
    assert!(ea.iter().all(|(k, v)| (v - eb[k]).abs() < 1e-15), "to");
    assert_eq!((b.dst_off, b.dst_size), ([9, 13], [17, 11]));
    for (k, area) in b.dst_area.iter().enumerate() {
        let r = (13 + k / 17) as f64;
        assert!((area * sr(1.0) - box_sr([0.0, 1.0], [-11.0 - r, -10.0 - r])).abs() < 1e-17, "cell {k}");
    }
}

/// A 2 degree global grid from -180: a box across 180, a cap round the
/// north pole and three nodes round the south, in a mesh with a period.
fn seam_and_poles() -> PolyMesh {
    let mut x = vec![176.0, -176.0, -176.0, 176.0];
    let mut y = vec![10.0, 10.0, 20.0, 20.0];
    let mut nodes = vec![0, 1, 2, 3];
    let mut start = vec![0, 4];
    for k in 0..36 {
        x.push(-180.0 + 10.0 * k as f64);
        y.push(80.0);
        nodes.push(x.len() - 1);
    }
    start.push(nodes.len());
    for lon in [0.0, 120.0, 240.0] {
        x.push(lon);
        y.push(-60.0);
        nodes.push(x.len() - 1);
    }
    start.push(nodes.len());
    PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap()
}

/// Across the seam each face's overlaps are its closed form, at both ends
/// of the grid; round a pole the cap is closed along it. The grid stored
/// from 0 gives the same, its columns turned 180.
#[test]
fn across_the_seam_and_round_the_poles() {
    let gt = [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0];
    let t = lonlat(gt, [180, 90], 0.0);
    assert_eq!(t.source_wrap(), Some(SourceWrap::new(180.0, [0.0, 90.0])));
    let mesh = seam_and_poles();
    let opts = WeightOptions::new(4).with_measure(lonlat_measure(gt).unwrap());
    let grid = GridWindow::new(180, 90);
    let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
    let want = [box_sr([176.0, 184.0], [10.0, 20.0]), 2.0 * PI * (1.0 - 80f64.to_radians().sin()), 2.0 * PI * (1.0 - 60f64.to_radians().sin())];
    // the south cap's 2700 cells sum to 2e-14 of it
    let covered = onto.dst_covered();
    for (f, want) in want.into_iter().enumerate() {
        assert!((onto.dst_area[f] * sr(2.0) - want).abs() < 1e-14 * want, "face {f}: {} {want}", onto.dst_area[f] * sr(2.0));
        assert!((covered[f] * sr(2.0) - want).abs() < 1e-13 * want, "face {f}");
    }
    let mut seam: Vec<(u32, u32)> = (0..onto.len()).filter(|&k| onto.dst_col[k] == 0).map(|k| (onto.src_row[k], onto.src_col[k])).collect();
    seam.sort();
    let mut expect: Vec<(u32, u32)> = (35..40).flat_map(|r| [0, 1, 178, 179].map(|c| (r, c))).collect();
    expect.sort();
    assert_eq!(seam, expect);
    for k in 0..onto.len() {
        match onto.dst_col[k] {
            0 => assert!((onto.frac[k] - 1.0).abs() < 1e-14),
            // north of 80 is rows 0 to 4, wholly
            1 => assert!(onto.src_row[k] < 5 && (onto.frac[k] - 1.0).abs() < 1e-14),
            _ => assert!(onto.src_row[k] >= 75 && (onto.frac[k] - 1.0).abs() < 1e-14),
        }
    }
    // each face is wholly on the grid, so it all goes to it
    let to = mesh.weights_to(&t, &grid, &opts).unwrap();
    let mut per_face = [0.0; 3];
    for k in 0..to.len() {
        per_face[to.src_col[k] as usize] += to.frac[k];
    }
    assert!(per_face.iter().all(|s| (s - 1.0).abs() < 1e-14), "{per_face:?}");
    // the same data from 0 to 360
    let shifted = mesh.weights_onto(&lonlat([0.0, 2.0, 0.0, 90.0, 0.0, -2.0], [180, 90], 180.0), &grid, &opts).unwrap();
    let key = |w: &Weights, k: usize, s: u32| (w.dst_col[k], w.src_row[k], (w.src_col[k] + s) % 180);
    let a: HashMap<_, _> = (0..onto.len()).map(|k| (key(&onto, k, 90), onto.frac[k])).collect();
    assert_eq!(a.len(), shifted.len());
    for k in 0..shifted.len() {
        assert!((a[&key(&shifted, k, 0)] - shifted.frac[k]).abs() < 1e-14);
    }
    // without the period the box goes the long way round, 352 degrees
    let (x, y): (Vec<f64>, Vec<f64>) = (0..4).map(|n| (mesh.node(n)[0], mesh.node(n)[1])).unzip();
    let plain = PolyMesh::new(x, y, vec![0, 4], vec![0, 1, 2, 3]).unwrap();
    let long = plain.weights_onto(&t, &grid, &opts).unwrap();
    assert!((long.dst_area[0] * sr(2.0) - box_sr([-176.0, 176.0], [10.0, 20.0])).abs() < 1e-14);
}

// ---- refined edges, in an equal-area projection --------------------------------------

/// North polar LAEA on the sphere of radius `R`, written out: from
/// longitude and latitude to metres.
struct Laea;

const R: f64 = 6371229.0;

impl CrsTransform for Laea {
    fn convert(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
        let rho = 2.0 * R * ((90.0 - lat).to_radians() / 2.0).sin();
        Some((rho * lon.to_radians().sin(), -rho * lon.to_radians().cos()))
    }
}

/// Counts the points it moves.
struct Counting<'a, T> {
    inner: T,
    n: &'a Cell<usize>,
}

impl<T: Transformer> Transformer for Counting<'_, T> {
    fn transform(&self, d: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        self.n.set(self.n.get() + x.len());
        self.inner.transform(d, x, y)
    }
}

/// 100 km cells over the pole, 60 by 60.
const LAEA_GT: [f64; 6] = [-3.0e6, 1.0e5, 0.0, 3.0e6, 0.0, -1.0e5];

/// Boxes 15 by 5 degrees from 65 N to 85 N all round, in longitude and
/// latitude with a period, their nodes from -180 (so a row's last box
/// crosses 180 to its first).
fn ring_boxes() -> PolyMesh {
    let (nx, ny) = (24, 4);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..nx {
            x.push(-180.0 + 15.0 * i as f64);
            y.push(65.0 + 5.0 * k as f64);
        }
    }
    let n = |i: usize, k: usize| k * nx + i % nx;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        for i in 0..nx {
            nodes.extend([n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)]);
            start.push(nodes.len());
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap()
}

/// Each box's area in LAEA is its area on the sphere, in pixels.
fn ring_box_area(f: usize) -> f64 {
    let k = (f / 24) as f64;
    R * R * box_sr([0.0, 15.0], [65.0 + 5.0 * k, 70.0 + 5.0 * k]) / 1e10
}

/// Whether cell `cell` of the 60 by 60 grid is wholly between 65 N and
/// 85 N (a pixel inside, in case a box's edge is a tolerance out).
fn inside_ring(cell: usize) -> bool {
    let (r, c) = ((cell / 60) as f64, (cell % 60) as f64);
    let (x0, x1, y0, y1) = (c - 30.0, c - 29.0, 29.0 - r, 30.0 - r);
    let near = x0.max(0.0).min(x1).hypot(y0.max(0.0).min(y1));
    let far = [x0.hypot(y0), x0.hypot(y1), x1.hypot(y0), x1.hypot(y1)].into_iter().fold(0.0, f64::max);
    let rho = |lat: f64| 2.0 * R * ((90.0 - lat).to_radians() / 2.0).sin() / 1e5;
    near > rho(85.0) + 1.0 && far < rho(65.0) - 1.0
}

/// Refined to a tolerance in pixels, the boxes' parallels (circles in
/// LAEA) are within it: each box's area converges on its area on the
/// sphere at the tolerance's rate (1.35 to 1.58 times it, from 1e-1 to
/// 1e-6 pixels), the points growing as its inverse square root (from
/// 1e-3, where the boxes' corners no longer dominate the count). The
/// faces tile the grid at every tolerance, across the mesh's seam too.
#[test]
fn refined_edges_converge_on_the_sphere() {
    let mesh = ring_boxes();
    let grid = GridWindow::new(60, 60);
    let n = Cell::new(0);
    let inner = GenImgProjTransformer::with_backend(LAEA_GT, WORLD_GT, Box::new(Laea), Box::new(Same)).unwrap();
    let t = Counting { inner, n: &n };
    let mut prev = f64::INFINITY;
    let mut points = Vec::new();
    for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        n.set(0);
        let onto = mesh.weights_onto(&t, &grid, &WeightOptions::new(1).with_refine_tolerance(tol)).unwrap();
        let err = (0..mesh.nface()).map(|f| (onto.dst_area[f] - ring_box_area(f)).abs()).fold(0.0, f64::max);
        assert!(err < 2.0 * tol && err < prev / 5.0, "tolerance {tol}: {err:e}");
        prev = err;
        let per_cell = frac_per_cell(&onto, 60, 3600);
        let mut full = 0;
        for (cell, s) in per_cell.iter().enumerate() {
            if inside_ring(cell) {
                full += 1;
                assert!((s - 1.0).abs() < 1e-13, "tolerance {tol}, cell {cell}: {s}");
            }
        }
        assert_eq!(full, 1956);
        eprintln!("tolerance {tol:e}: worst box {err:.2e} pixels, {} points", n.get());
        points.push(n.get() as f64);
    }
    let rate = (points[5] / points[2]).log10() / 3.0;
    eprintln!("points per decade of tolerance from 1e-3: 10^{rate:.3}");
    assert!((0.45..0.55).contains(&rate), "{rate}");
}

/// A fine tolerance gives what a fine densify gives.
#[test]
fn a_fine_tolerance_is_a_fine_densify() {
    let mesh = ring_boxes();
    let grid = GridWindow::new(60, 60);
    let t = GenImgProjTransformer::with_backend(LAEA_GT, WORLD_GT, Box::new(Laea), Box::new(Same)).unwrap();
    let a = mesh.weights_onto(&t, &grid, &WeightOptions::new(1).with_refine_tolerance(1e-7)).unwrap();
    let b = mesh.weights_onto(&t, &grid, &WeightOptions::new(4096)).unwrap();
    let (ea, eb) = (onto_overlaps(&a), onto_overlaps(&b));
    let d = ea.iter().map(|(k, v)| (v - eb.get(k).copied().unwrap_or(0.0)).abs()).fold(0.0, f64::max);
    let area = (0..mesh.nface()).map(|f| (a.dst_area[f] - b.dst_area[f]).abs()).fold(0.0, f64::max);
    let exact = (0..mesh.nface()).map(|f| (a.dst_area[f] - ring_box_area(f)).abs()).fold(0.0, f64::max);
    eprintln!("tolerance 1e-7 against densify 4096: overlaps to {d:.1e}, areas to {area:.1e}; areas from the sphere's to {exact:.1e}");
    assert!(d < 1e-6 && area < 1e-6, "{d:e} {area:e}");
}

// ---- failures and refusals -----------------------------------------------------------

/// The identity onto unit cells, failing for x strictly between 3.25 and
/// 3.75.
struct Gap;

impl Transformer for Gap {
    fn transform(&self, d: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let gap: Vec<bool> = x.iter().map(|x| *x > 3.25 && *x < 3.75).collect();
        let mut ok = same([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]).transform(d, x, y);
        for (k, g) in gap.into_iter().enumerate() {
            if g {
                (x[k], y[k], ok[k]) = (f64::NAN, f64::NAN, false);
            }
        }
        ok
    }
}

/// A point of a face's outline that fails fails the face: none at
/// densify 1, the edges' midpoints at densify 2 or when refined. The
/// other faces are as without the gap.
#[test]
fn a_point_that_fails_fails_its_face() {
    let (x, y) = lattice(0.0, -3.0, 1.0, 1.0, 6, 3);
    let (start, nodes) = quads(6, 3);
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let grid = GridWindow::new(6, 3);
    let failed = |w: &Weights| (0..18).filter(|&f| w.dst_area[f].is_nan()).map(|f| f % 6).collect::<Vec<_>>();
    assert!(failed(&mesh.weights_onto(&Gap, &grid, &WeightOptions::new(1)).unwrap()).is_empty());
    for opts in [WeightOptions::new(2), WeightOptions::new(1).with_refine_tolerance(1e-3)] {
        let w = mesh.weights_onto(&Gap, &grid, &opts).unwrap();
        assert_eq!(failed(&w), vec![3; 3]);
        assert!(w.dst_col.iter().all(|&f| f % 6 != 3));
        assert_eq!(w.len(), 15);
        assert!(w.frac.iter().all(|&f| f == 1.0));
        let to = mesh.weights_to(&Gap, &grid, &opts).unwrap();
        assert_eq!(to.len(), 15);
        assert!(to.dst_col.iter().all(|&c| c != 3));
    }
}

/// A tolerance must be positive. (A grid with world cells is taken: the
/// same cells as a `QuadMesh` are the identity; `tests/poly_mesh_cells.rs`
/// has the rest.)
#[test]
fn refusals() {
    let (x, y) = lattice(0.0, -3.0, 1.0, 1.0, 6, 3);
    let (start, nodes) = quads(6, 3);
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap();
    let grid = GridWindow::new(6, 3);
    let (qx, qy) = lattice(0.0, -3.0, 1.0, 1.0, 6, 3);
    let q = QuadMesh::from_corners(6, 3, qx, qy, CellMap::EqualArea).unwrap();
    let mt = MeshTransformer::with_backend(q, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    let w = mesh.weights_onto(&mt, &grid, &WeightOptions::new(1)).unwrap();
    assert_eq!(w.len(), 18);
    assert!((0..18).all(|k| w.src_row[k] * 6 + w.src_col[k] == w.dst_col[k] && (w.frac[k] - 1.0).abs() < 1e-15));
    let t = same([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
    for tol in [0.0, -1.0, f64::NAN] {
        let e = mesh.weights_onto(&t, &grid, &WeightOptions::new(1).with_refine_tolerance(tol)).unwrap_err();
        assert!(e.contains("positive"), "{e}");
    }
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

/// On threads, both ways, the serial table bit for bit: a grid in pixels,
/// one in longitude and latitude across the seam and round the poles
/// under the spherical measure, refined edges in LAEA, and faces that
/// fail.
#[test]
fn threads_are_the_serial_table() {
    let mixed_t = || Ok::<_, String>(same(MIXED_GT));
    let ll_gt = [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0];
    let ll = || Ok::<_, String>(lonlat(ll_gt, [180, 90], 0.0));
    let laea = || GenImgProjTransformer::with_backend(LAEA_GT, WORLD_GT, Box::new(Laea), Box::new(Same));
    let mesh = mixed(10, 9);
    let seam = seam_and_poles();
    let ring = ring_boxes();
    let (x, y) = lattice(0.0, -3.0, 1.0, 1.0, 6, 3);
    let (start, nodes) = quads(6, 3);
    let gap = PolyMesh::new(x, y, start, nodes).unwrap();
    let cases: Vec<(&PolyMesh, GridWindow, WeightOptions, usize)> = vec![
        (&mesh, GridWindow::new(MIXED.0, MIXED.1), WeightOptions::new(2), 0),
        (&seam, GridWindow::new(180, 90), WeightOptions::new(4).with_measure(lonlat_measure(ll_gt).unwrap()), 1),
        (&ring, GridWindow::new(60, 60), WeightOptions::new(1).with_refine_tolerance(1e-4), 2),
        (&gap, GridWindow::new(6, 3), WeightOptions::new(1).with_refine_tolerance(1e-3), 3),
    ];
    for (m, grid, opts, t) in cases {
        let (onto, to) = match t {
            0 => (m.weights_onto(&same(MIXED_GT), &grid, &opts), m.weights_to(&same(MIXED_GT), &grid, &opts)),
            1 => (m.weights_onto(&ll().unwrap(), &grid, &opts), m.weights_to(&ll().unwrap(), &grid, &opts)),
            2 => (m.weights_onto(&laea().unwrap(), &grid, &opts), m.weights_to(&laea().unwrap(), &grid, &opts)),
            _ => (m.weights_onto(&Gap, &grid, &opts), m.weights_to(&Gap, &grid, &opts)),
        };
        let (onto, to) = (bits(&onto.unwrap()), bits(&to.unwrap()));
        for threads in [1, 2, 3, 7] {
            let (a, b) = match t {
                0 => (m.weights_onto_par(mixed_t, &grid, &opts, threads), m.weights_to_par(mixed_t, &grid, &opts, threads)),
                1 => (m.weights_onto_par(ll, &grid, &opts, threads), m.weights_to_par(ll, &grid, &opts, threads)),
                2 => (m.weights_onto_par(laea, &grid, &opts, threads), m.weights_to_par(laea, &grid, &opts, threads)),
                _ => (m.weights_onto_par(|| Ok(&Gap), &grid, &opts, threads), m.weights_to_par(|| Ok(&Gap), &grid, &opts, threads)),
            };
            assert!(bits(&a.unwrap()) == onto, "case {t}, {threads} threads: onto");
            assert!(bits(&b.unwrap()) == to, "case {t}, {threads} threads: to");
        }
    }
    // the first transformer's error, and the options' checks, come back
    let e = mesh.weights_onto_par(|| Err::<GenImgProjTransformer, _>("no".to_string()), &GridWindow::new(2, 2), &WeightOptions::new(1), 4);
    assert_eq!(e.unwrap_err(), "no");
    let e = mesh.weights_to_par(mixed_t, &GridWindow::new(2, 2), &WeightOptions::new(1).with_refine_tolerance(-1.0), 4);
    assert!(e.unwrap_err().contains("positive"));
}
