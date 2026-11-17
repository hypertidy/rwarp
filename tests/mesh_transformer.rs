//! `MeshTransformer`: a `QuadMesh` as a warp source, for `exact_weights`
//! and `warp()`'s exact footprint (`rwarp::mesh`).
//!
//! The mesh is a tripolar cap like `tests/fixtures/esmf/gen.py`'s, with a
//! seam into one grid pole and a fold between the two, so the walk crosses
//! both. No CRS is needed except in the last test.

use std::collections::HashMap;

use rwarp::mesh::{CellMap, MeshTransformer, QuadMesh};
use rwarp::warp::{warp_to, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{
    exact_weights, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights,
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
/// line, folded: corner `i` on corner `ncol - i`), column line `ncol` on
/// column line 0; turned and shifted, in metres.
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

/// 150 km pixels over the whole cap, north up.
const GT: [f64; 6] = [-3.9e6, 1.5e5, 0.0, 3.9e6, 0.0, -1.5e5];
const DW: usize = 52;
const DH: usize = 52;

fn transformer(gt: [f64; 6]) -> MeshTransformer {
    MeshTransformer::with_backend(tripolar(), gt, Box::new(Same), Box::new(Same)).unwrap()
}

/// `(dst_row, dst_col, src_row, src_col) -> frac`
fn entries(w: &Weights) -> HashMap<(u32, u32, u32, u32), f64> {
    (0..w.len())
        .map(|k| {
            (
                (w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]),
                w.frac[k],
            )
        })
        .collect()
}

/// max |a - b| over the union of entries, a missing entry being 0.
fn max_diff(a: &HashMap<(u32, u32, u32, u32), f64>, b: &HashMap<(u32, u32, u32, u32), f64>) -> f64 {
    let one = |p: &HashMap<_, f64>, q: &HashMap<_, f64>| {
        p.iter()
            .map(|(k, v)| (v - q.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max)
    };
    one(a, b).max(one(b, a))
}

/// With a destination over the whole mesh, every cell's area is shared out
/// in full: each source cell's `frac` sums to 1, across the seam, the fold
/// and the poles, and the destination cells' covered areas sum to the
/// mesh's area.
#[test]
fn every_cell_is_shared_out_in_full() {
    let t = transformer(GT);
    let m = t.source_mesh();
    for densify in [1, 3] {
        let w = exact_weights(&t, &t.source_window(), DW, DH, &WeightOptions::new(densify));
        assert!(
            w.frac.iter().all(|&f| f > 0.0 && f <= 1.0 + 1e-12),
            "frac out of range"
        );
        let mut sums = vec![0.0; NCOL * NROW];
        for k in 0..w.len() {
            sums[w.src_row[k] as usize * NCOL + w.src_col[k] as usize] += w.frac[k];
        }
        let worst = sums.iter().map(|s| (s - 1.0).abs()).fold(0.0, f64::max);
        let total: f64 = (0..NROW)
            .flat_map(|r| (0..NCOL).map(move |c| (r, c)))
            .map(|(r, c)| m.cell_area(r, c))
            .sum();
        let covered: f64 = w.dst_covered().iter().sum();
        let rel = (covered - total).abs() / total;
        eprintln!("densify {densify}: {} entries, worst |sum frac - 1| {worst:.1e}, covered area {rel:.1e}", w.len());
        assert!(
            worst < 1e-12,
            "densify {densify}: a cell's frac sums to 1 +- {worst:e}"
        );
        assert!(
            rel < 1e-12,
            "densify {densify}: covered area off by {rel:e}"
        );
    }
}

/// `warp()` with the exact footprint and `Average` over a mesh source is
/// `apply(FracArea)` of warp-compatible weights, bit for bit, as for
/// other sources (`tests/weights.rs`).
#[test]
fn warp_exact_average_is_apply_frac_area() {
    let t = transformer(GT);
    let data: Vec<f64> = (0..NCOL * NROW)
        .map(|p| 10.0 + ((p % NCOL) as f64 * 0.4).sin() * ((p / NCOL) as f64 * 0.3).cos())
        .collect();
    let buf = SourceBuffer {
        data: &data[..],
        ncol: NCOL,
        nrow: NROW,
        nbands: 1,
        col_off: 0,
        row_off: 0,
    };
    for densify in [1, 2] {
        let opts =
            WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify });
        let warped = warp_to::<f64, f64>(&t, &buf, DW, DH, &opts);
        let w = exact_weights(
            &t,
            &GridWindow::from(&buf),
            DW,
            DH,
            &WeightOptions::warp_compatible(densify),
        );
        let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
        let differ = warped
            .iter()
            .zip(&applied)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let computed = warped.iter().filter(|v| (5.0..15.0).contains(*v)).count();
        eprintln!("densify {densify}: {computed} pixels computed, {differ} differ");
        assert_eq!(differ, 0);
        assert!(computed > 1000, "only {computed} pixels computed");
    }
}

/// Weights over source windows, 2 x 2, are the whole mesh's weights:
/// each window is a mesh of its own (`QuadMesh::window`), and a cell's
/// coverage doesn't depend on the cells around it.
#[test]
fn source_windows_tile_the_mesh() {
    let t = transformer(GT);
    let whole = entries(&exact_weights(
        &t,
        &t.source_window(),
        DW,
        DH,
        &WeightOptions::new(2),
    ));
    let mut tiled = HashMap::new();
    for (c0, c1) in [(0, 20), (20, NCOL)] {
        for (r0, r1) in [(0, 9), (9, NROW)] {
            let src = GridWindow::new(c1 - c0, r1 - r0).with_offset(c0, r0);
            let w = exact_weights(&t, &src, DW, DH, &WeightOptions::new(2));
            assert!(w.src_col.iter().all(|&c| (c0..c1).contains(&(c as usize))));
            assert!(w.src_row.iter().all(|&r| (r0..r1).contains(&(r as usize))));
            tiled.extend(entries(&w));
        }
    }
    let d = max_diff(&whole, &tiled);
    eprintln!(
        "{} entries whole, {} tiled, max frac difference {d:.1e}",
        whole.len(),
        tiled.len()
    );
    assert!(d < 1e-12, "windows differ from the whole mesh by {d:e}");
}

/// A destination stored south up has the same weights, rows reversed:
/// outlines of either orientation are covered alike.
#[test]
fn either_row_order_gives_the_same_weights() {
    let up = [GT[0], GT[1], 0.0, GT[3] + DH as f64 * GT[5], 0.0, -GT[5]];
    let north = entries(&exact_weights(
        &transformer(GT),
        &GridWindow::new(NCOL, NROW),
        DW,
        DH,
        &WeightOptions::new(1),
    ));
    let south: HashMap<_, _> = entries(&exact_weights(
        &transformer(up),
        &GridWindow::new(NCOL, NROW),
        DW,
        DH,
        &WeightOptions::new(1),
    ))
    .into_iter()
    .map(|((r, c, sr, sc), f)| ((DH as u32 - 1 - r, c, sr, sc), f))
    .collect();
    let d = max_diff(&north, &south);
    eprintln!("north up vs south up: max frac difference {d:.1e}");
    assert!(d < 1e-12);
}

/// Destination pixels to the mesh's index space and back.
#[test]
fn transform_maps_pixels_to_cells_and_back() {
    let t = transformer(GT);
    let m = t.source_mesh();
    let (mut n_in, mut n_out) = (0, 0);
    for k in 0..400 {
        let (px, py) = (0.3 + (k % 20) as f64 * 2.6, 0.7 + (k / 20) as f64 * 2.55);
        let (mut x, mut y) = (vec![px], vec![py]);
        if !t.transform(true, &mut x, &mut y)[0] {
            n_out += 1;
            assert!(m
                .to_index([GT[0] + px * GT[1], GT[3] + py * GT[5]], None)
                .is_none());
            continue;
        }
        n_in += 1;
        let p = m.to_world([x[0], y[0]]).unwrap();
        let want = [GT[0] + px * GT[1], GT[3] + py * GT[5]];
        assert!(
            (p[0] - want[0]).abs() < 1e-6 && (p[1] - want[1]).abs() < 1e-6,
            "{p:?} vs {want:?}"
        );
        assert!(t.transform(false, &mut x, &mut y)[0]);
        assert!(
            (x[0] - px).abs() < 1e-9 && (y[0] - py).abs() < 1e-9,
            "({}, {}) vs ({px}, {py})",
            x[0],
            y[0]
        );
    }
    eprintln!("{n_in} points on the mesh, {n_out} off it");
    assert!(n_in > 200 && n_out > 20);
}

/// With the CRS backend: the mesh in north-polar LAEA on a sphere, the
/// destination in lon/lat on it. The same weights as with the projection
/// written out, to the backend's rounding.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn a_crs_backend_gives_the_same_weights() {
    const R: f64 = 6371229.0;
    struct Laea(bool);
    impl CrsTransform for Laea {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some(if self.0 {
                let rho = 2.0 * R * ((90.0 - y).to_radians() / 2.0).sin();
                (rho * x.to_radians().sin(), -rho * x.to_radians().cos())
            } else {
                let rho = x.hypot(y);
                (
                    x.atan2(-y).to_degrees(),
                    90.0 - 2.0 * (rho / (2.0 * R)).asin().to_degrees(),
                )
            })
        }
    }
    let gt = [0.0, 6.0, 0.0, 90.0, 0.0, -3.0];
    let (dw, dh) = (60, 12);
    let laea = format!("+proj=laea +lat_0=90 +lon_0=0 +R={R} +units=m +no_defs");
    let lonlat = format!("+proj=longlat +R={R} +no_defs");
    let with_crs = MeshTransformer::new(tripolar(), &laea, gt, &lonlat).unwrap();
    let written =
        MeshTransformer::with_backend(tripolar(), gt, Box::new(Laea(true)), Box::new(Laea(false)))
            .unwrap();
    let opts = WeightOptions::new(4);
    let a = exact_weights(&with_crs, &with_crs.source_window(), dw, dh, &opts);
    let b = exact_weights(&written, &written.source_window(), dw, dh, &opts);
    let d = max_diff(&entries(&a), &entries(&b));
    let area = a
        .dst_area
        .iter()
        .zip(&b.dst_area)
        .map(|(x, y)| ((x - y) / y).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "{} entries; CRS backend vs written out: frac {d:.1e}, area {area:.1e}",
        a.len()
    );
    assert!(a.len() > 2000);
    assert!(d < 1e-10 && area < 1e-10, "frac {d:e}, area {area:e}");
}
