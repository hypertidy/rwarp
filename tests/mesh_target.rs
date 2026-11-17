//! `MeshTarget`: a `QuadMesh` as a warp target, for `exact_weights` and
//! `warp()` onto a curvilinear grid (`rwarp::mesh`).
//!
//! The target is a tripolar cap like `tests/mesh_transformer.rs`'s, in
//! north-polar LAEA, with a seam into one grid pole and a fold between the
//! two. Each cell is measured on its own as a target pixel, so neither
//! matters; what does is that the cells' outlines go to the source exactly
//! and tile. No CRS is needed except in the last test.

use std::collections::HashMap;

use rwarp::mesh::{CellMap, MeshTarget, MeshTransformer, QuadMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::{warp_to, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{
    exact_weights, lonlat_measure, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights,
};
use rwarp::{CrsTransform, Transformer};

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

const NCOL: usize = 48;
const NROW: usize = 16;

/// Confocal ellipses about grid poles at `(+-a, 0)`, from the outer
/// ellipse (row line 0) to the segment between the poles (the last row
/// line, folded), column line `ncol` on column line 0; turned and
/// shifted, in metres of north-polar LAEA.
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

/// A second mesh over the cap: a lattice of 30 x 25 cells, bent and
/// turned, in the same plane.
fn bent() -> QuadMesh {
    let (nc, nr) = (30, 25);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nr {
        for i in 0..=nc {
            let (u, v) = (i as f64 / nc as f64 - 0.5, j as f64 / nr as f64 - 0.5);
            let (px, py) = (8.2e6 * u + 4e5 * (3.0 * v).sin(), 7.6e6 * v + 3e5 * (2.5 * u).cos());
            let t = 0.3f64;
            x.push(px * t.cos() - py * t.sin());
            y.push(px * t.sin() + py * t.cos());
        }
    }
    QuadMesh::from_corners(nc, nr, x, y, CellMap::Bilinear).unwrap()
}

/// 150 km pixels over the whole cap, north up, in the cap's plane.
const GT: [f64; 6] = [-3.9e6, 1.5e5, 0.0, 3.9e6, 0.0, -1.5e5];
const GW: usize = 52;
const GH: usize = 52;
const PIXEL: f64 = 1.5e5 * 1.5e5;

fn grid_to_cap() -> MeshTarget<GenImgProjTransformer> {
    let inner = GenImgProjTransformer::with_backend(GT, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
    MeshTarget::new(tripolar(), inner)
}

type Key = (u32, u32, u32, u32);

/// `(dst_row, dst_col, src_row, src_col) -> overlap`, the overlap's
/// measure times `scale`.
fn overlaps(w: &Weights, scale: f64) -> HashMap<Key, f64> {
    (0..w.len())
        .map(|k| ((w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]), w.frac[k] * w.src_area(k) * scale))
        .collect()
}

/// Keys with source and destination swapped.
fn transposed(m: &HashMap<Key, f64>) -> HashMap<Key, f64> {
    m.iter().map(|(&(a, b, c, d), &v)| ((c, d, a, b), v)).collect()
}

/// max |a - b| over the union of entries, a missing entry being 0.
fn max_diff(a: &HashMap<Key, f64>, b: &HashMap<Key, f64>) -> f64 {
    let one = |p: &HashMap<Key, f64>, q: &HashMap<Key, f64>| {
        p.iter()
            .map(|(k, v)| (v - q.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max)
    };
    one(a, b).max(one(b, a))
}

/// In one plane, the overlaps of the grid's pixels with the mesh's cells
/// are the same whichever is the target: `MeshTarget` (grid to mesh)
/// against `MeshTransformer` (mesh to grid), transposed. Each target
/// cell's `dst_area` is the cell's area, and each is covered in full.
#[test]
fn overlaps_are_the_mesh_sources_transposed() {
    let t = grid_to_cap();
    let m = t.target_mesh();
    let src = MeshTransformer::with_backend(tripolar(), GT, Box::new(Same), Box::new(Same)).unwrap();
    for densify in [1, 3] {
        let opts = WeightOptions::new(densify);
        let onto = exact_weights(&t, &GridWindow::new(GW, GH), NCOL, NROW, &opts);
        let from = exact_weights(&src, &src.source_window(), GW, GH, &opts);
        let d = max_diff(&overlaps(&onto, PIXEL), &transposed(&overlaps(&from, 1.0))) / PIXEL;
        let mut area = 0.0f64;
        let mut cover = 0.0f64;
        let covered = onto.dst_covered();
        for r in 0..NROW {
            for c in 0..NCOL {
                let a = m.cell_area(r, c);
                if a == 0.0 {
                    continue;
                }
                let k = r * NCOL + c;
                area = area.max((onto.dst_area[k] * PIXEL - a).abs() / a);
                cover = cover.max((covered[k] / onto.dst_area[k] - 1.0).abs());
            }
        }
        eprintln!("densify {densify}: {} entries; transposed overlaps {d:.1e} of a pixel, cell areas {area:.1e}, cover {cover:.1e}", onto.len());
        assert!(onto.len() > 2000);
        assert!(d < 1e-12, "overlaps differ by {d:e} of a pixel");
        assert!(area < 1e-12 && cover < 1e-12, "areas {area:e}, cover {cover:e}");
    }
}

/// From one mesh onto another, in one plane: the source's cells are
/// walked (`MeshTransformer` with `WORLD_GT` inside `MeshTarget`), and the
/// overlaps are the same either way round.
#[test]
fn mesh_to_mesh_overlaps_are_symmetric() {
    let (a, b) = (tripolar(), bent());
    let between = |src: QuadMesh, dst: QuadMesh| {
        let (sw, sh, dw, dh) = (src.ncol(), src.nrow(), dst.ncol(), dst.nrow());
        let inner = MeshTransformer::with_backend(src, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
        let t = MeshTarget::new(dst, inner);
        assert!(t.source_cells().is_some());
        exact_weights(&t, &GridWindow::new(sw, sh), dw, dh, &WeightOptions::new(2))
    };
    let ab = between(a.clone(), b.clone());
    let ba = between(b.clone(), a.clone());
    let scale = a.cell_area(0, 0);
    let d = max_diff(&overlaps(&ab, 1.0), &transposed(&overlaps(&ba, 1.0))) / scale;
    // every cell of the cap is inside the bent mesh: its own area, covered
    let covered = ba.dst_covered();
    let worst = (0..NROW * NCOL)
        .filter(|&k| a.cell_area(k / NCOL, k % NCOL) > 0.0)
        .map(|k| (covered[k] / a.cell_area(k / NCOL, k % NCOL) - 1.0).abs())
        .fold(0.0, f64::max);
    eprintln!("{} and {} entries; symmetric to {d:.1e} of a cell; cap cells covered to {worst:.1e}", ab.len(), ba.len());
    assert!(ab.len() > 1000 && ab.len() == ba.len());
    assert!(d < 1e-12 && worst < 1e-12, "{d:e}, {worst:e}");
}

const R: f64 = 6371229.0;

/// North-polar LAEA on a sphere: `true` from longitude and latitude,
/// `false` back (longitude in -180 to 180).
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

/// A global 2.5 degree grid in longitude and latitude, from `x0`.
fn lonlat_gt(x0: f64) -> [f64; 6] {
    [x0, 2.5, 0.0, 90.0, 0.0, -2.5]
}
const LW: usize = 144;
const LH: usize = 72;

/// The cap as a target from a global grid in longitude and latitude, its
/// longitudes wrapped round the grid's centre.
fn lonlat_to_cap(x0: f64) -> MeshTarget<GenImgProjTransformer> {
    let inner = GenImgProjTransformer::with_backend(lonlat_gt(x0), WORLD_GT, Box::new(Laea(false)), Box::new(Laea(true)))
        .unwrap()
        .with_center_long(CenterLong::At(x0 + 180.0))
        .with_source_size([LW, LH]);
    MeshTarget::new(tripolar(), inner)
}

/// From a global grid in longitude and latitude, with areas on the sphere
/// (`lonlat_measure`): cells across the grid's seam and the cell over the
/// pole are measured whole, every cell is covered in full, and where the
/// grid's seam is makes no difference. The cells' areas on the sphere
/// approach their planar LAEA areas (true areas) as `densify` grows:
/// their sides are straight in LAEA, and straight in longitude and
/// latitude between the points that go to the source.
#[test]
fn onto_a_tripolar_grid_from_longitude_and_latitude() {
    let m = tripolar();
    let steradian = (2.5f64.to_radians()).powi(2);
    let mut last = f64::INFINITY;
    for densify in [2, 8, 32] {
        let t = lonlat_to_cap(-180.0);
        let opts = WeightOptions::new(densify).with_measure(lonlat_measure(lonlat_gt(-180.0)).unwrap());
        let w = exact_weights(&t, &GridWindow::new(LW, LH), NCOL, NROW, &opts);
        let covered = w.dst_covered();
        let cover = (0..NROW * NCOL)
            .filter(|&k| m.cell_area(k / NCOL, k % NCOL) > 0.0)
            .map(|k| (covered[k] / w.dst_area[k] - 1.0).abs())
            .fold(0.0, f64::max);
        // cells across the grid's seam, and the one over the pole
        let mut cols: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for i in 0..w.len() {
            cols.entry((w.dst_row[i], w.dst_col[i])).or_default().push(w.src_col[i]);
        }
        let across = cols.values().filter(|c| c.contains(&0) && c.contains(&(LW as u32 - 1))).count();
        let pole: Vec<_> = (0..w.len()).filter(|&i| w.src_row[i] == 0).map(|i| (w.dst_row[i], w.dst_col[i])).collect();
        let mut pole_cells = pole.clone();
        pole_cells.sort();
        pole_cells.dedup();
        let pole_cols = (0..w.len()).filter(|&i| w.src_row[i] == 0).map(|i| w.src_col[i]).collect::<std::collections::HashSet<_>>().len();
        // the cap's area on the sphere, against its LAEA area
        let sphere: f64 = w.dst_area.iter().sum::<f64>() * steradian * R * R;
        let plane: f64 = (0..NROW).flat_map(|r| (0..NCOL).map(move |c| (r, c))).map(|(r, c)| m.cell_area(r, c)).sum();
        let rel = (sphere - plane).abs() / plane;
        eprintln!("densify {densify:2}: {} entries; covered to {cover:.1e}; {across} cells across the seam; source row 0 in {} cells, all {pole_cols} columns; area on the sphere vs LAEA {rel:.1e}", w.len(), pole_cells.len());
        assert!(cover < 1e-12, "{cover:e}");
        assert!(across >= 3, "only {across} cells across the seam");
        assert_eq!(pole_cols, LW, "the pole's cell takes all of source row 0");
        assert!(rel < last / 8.0, "{rel:e} after {last:e}");
        last = rel;

        // the grid's seam elsewhere: the same weights, columns renumbered
        if densify == 8 {
            let t2 = lonlat_to_cap(20.0);
            let opts2 = opts.with_measure(lonlat_measure(lonlat_gt(20.0)).unwrap());
            let w2 = exact_weights(&t2, &GridWindow::new(LW, LH), NCOL, NROW, &opts2);
            let shift = (200.0 / 2.5) as u32; // column 0 from 20 E is column 80 from 180 W
            let renumbered: HashMap<Key, f64> = overlaps(&w2, 1.0)
                .into_iter()
                .map(|((a, b, r, c), v)| ((a, b, r, (c + shift) % LW as u32), v))
                .collect();
            let d = max_diff(&renumbered, &overlaps(&w, 1.0));
            eprintln!("  seam at 20 E: overlaps {d:.1e} of a pixel from the seam at 180");
            assert!(d < 1e-10, "{d:e}");
        }
    }
    assert!(last < 1e-5, "{last:e}");
}

/// `warp()`'s exact `Average` onto the mesh is `apply(FracArea)` of
/// warp-compatible weights, bit for bit, across the source's seam and
/// round the pole.
#[test]
fn warp_exact_average_onto_a_mesh_is_apply_frac_area() {
    let t = lonlat_to_cap(-180.0);
    let data: Vec<f64> = (0..LW * LH)
        .map(|p| 10.0 + ((p % LW) as f64 * 0.2).sin() * ((p / LW) as f64 * 0.15).cos())
        .collect();
    let buf = SourceBuffer { data: &data[..], ncol: LW, nrow: LH, nbands: 1, col_off: 0, row_off: 0 };
    for densify in [1, 4] {
        let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify });
        let warped = warp_to::<f64, f64>(&t, &buf, NCOL, NROW, &opts);
        let w = exact_weights(&t, &GridWindow::from(&buf), NCOL, NROW, &WeightOptions::warp_compatible(densify));
        let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
        let differ = warped.iter().zip(&applied).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let computed = warped.iter().filter(|v| (5.0..15.0).contains(*v)).count();
        eprintln!("densify {densify}: {computed} cells computed, {differ} differ");
        assert_eq!(differ, 0);
        // every cell with area (the fold's pivot cells have none)
        assert!(computed >= NCOL * NROW - 2, "only {computed} cells computed");
    }
}

/// Point kernels sample the source at each cell's centre: `Bilinear` of a
/// linear field gives the field there. Target pixels go to the source and
/// back.
#[test]
fn point_kernels_sample_cell_centres() {
    let t = grid_to_cap();
    let m = t.target_mesh();
    let f = |x: f64, y: f64| 3.0 + 2e-6 * x - 1e-6 * y;
    let data: Vec<f64> = (0..GW * GH)
        .map(|p| {
            let (c, r) = ((p % GW) as f64 + 0.5, (p / GW) as f64 + 0.5);
            f(GT[0] + c * GT[1], GT[3] + r * GT[5])
        })
        .collect();
    let buf = SourceBuffer { data: &data[..], ncol: GW, nrow: GH, nbands: 1, col_off: 0, row_off: 0 };
    let out = warp_to::<f64, f64>(&t, &buf, NCOL, NROW, &WarpOptions::new(ResampleAlg::Bilinear));
    let mut worst = 0.0f64;
    for r in 0..NROW {
        for c in 0..NCOL {
            let p = m.to_world([c as f64 + 0.5, r as f64 + 0.5]).unwrap();
            worst = worst.max((out[r * NCOL + c] - f(p[0], p[1])).abs());
        }
    }
    eprintln!("bilinear of a linear field at the cell centres: {worst:.1e}");
    assert!(worst < 1e-12, "{worst:e}");

    let (mut n, mut back) = (0, 0.0f64);
    for r in 0..NROW {
        for c in 0..NCOL {
            if m.cell_area(r, c) == 0.0 {
                continue;
            }
            let (px, py) = (c as f64 + 0.3, r as f64 + 0.6);
            let (mut x, mut y) = (vec![px], vec![py]);
            assert!(t.transform(true, &mut x, &mut y)[0]);
            let want = m.to_world([px, py]).unwrap();
            assert!((GT[0] + x[0] * GT[1] - want[0]).abs() < 1e-6 && (GT[3] + y[0] * GT[5] - want[1]).abs() < 1e-6);
            assert!(t.transform(false, &mut x, &mut y)[0]);
            back = back.max((x[0] - px).abs()).max((y[0] - py).abs());
            n += 1;
        }
    }
    eprintln!("{n} target pixels to the source and back: {back:.1e}");
    assert!(back < 1e-9, "{back:e}");
    // off the mesh
    let (mut x, mut y) = (vec![-1.0, NCOL as f64 + 1.0], vec![0.5, 0.5]);
    assert_eq!(t.transform(true, &mut x, &mut y), vec![false, false]);
    assert!(x.iter().all(|v| v.is_infinite()));
}

/// Weights for blocks of the target mesh (`with_dst_offset`) are the
/// whole mesh's weights.
#[test]
fn target_windows_tile_the_mesh() {
    let t = lonlat_to_cap(-180.0);
    let opts = WeightOptions::new(4).with_measure(lonlat_measure(lonlat_gt(-180.0)).unwrap());
    let src = GridWindow::new(LW, LH);
    let whole = overlaps(&exact_weights(&t, &src, NCOL, NROW, &opts), 1.0);
    let mut tiled = HashMap::new();
    for (c0, c1) in [(0, 20), (20, NCOL)] {
        for (r0, r1) in [(0, 7), (7, NROW)] {
            let w = exact_weights(&t, &src, c1 - c0, r1 - r0, &opts.with_dst_offset(c0, r0));
            assert!(w.dst_col.iter().all(|&c| (c0..c1).contains(&(c as usize))));
            tiled.extend(overlaps(&w, 1.0));
        }
    }
    let d = max_diff(&whole, &tiled);
    eprintln!("{} entries whole, {} tiled: {d:.1e}", whole.len(), tiled.len());
    assert_eq!(whole.len(), tiled.len());
    assert!(d == 0.0, "{d:e}");
}

/// With the CRS backend: the same weights as with the projection written
/// out, and `from_grid` takes GDAL's centre longitude from the source.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn a_crs_backend_gives_the_same_weights() {
    let laea = format!("+proj=laea +lat_0=90 +lon_0=0 +R={R} +units=m +no_defs");
    let lonlat = format!("+proj=longlat +R={R} +no_defs");
    let gt = lonlat_gt(-180.0);
    let with_crs = MeshTarget::from_grid(&lonlat, gt, [LW, LH], tripolar(), &laea).unwrap();
    assert_eq!(with_crs.inner().center_long(), Some(0.0));
    let written = lonlat_to_cap(-180.0);
    let opts = WeightOptions::new(4).with_measure(lonlat_measure(gt).unwrap());
    let a = exact_weights(&with_crs, &GridWindow::new(LW, LH), NCOL, NROW, &opts);
    let b = exact_weights(&written, &GridWindow::new(LW, LH), NCOL, NROW, &opts);
    let d = max_diff(&overlaps(&a, 1.0), &overlaps(&b, 1.0));
    let area = a.dst_area.iter().zip(&b.dst_area).filter(|(_, y)| **y > 0.0).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max);
    // PROJ puts a few points on the source's seam at the other end, which
    // leaves slivers of no measure: compared by value, a missing entry 0
    eprintln!("{} and {} entries; CRS backend vs written out: overlaps {d:.1e}, areas {area:.1e}", a.len(), b.len());
    assert!(d < 1e-9 && area < 1e-9, "{d:e}, {area:e}");
}
