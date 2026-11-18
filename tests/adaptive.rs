//! `Footprint::Adaptive` and `WeightOptions::refine_tolerance`: the exact
//! footprint's edges bisected by bigcurve where the transform bends them
//! (curvilinear stage 5a, `design/curvilinear-frontend.md`).
//!
//! Oracles (#38): the error falls with the tolerance while the points
//! added grow as its inverse square root; a fine tolerance gives what a
//! fine fixed `densify` gives. Also: nothing changes where the transform
//! is affine; tiles, threads and `warp()` agree with the weights bit for
//! bit; a point that fails fails its pixel; a mesh source refines in its
//! world coordinates. The seam and the pole are in `tests/antimeridian.rs`,
//! a whole ocean grid in pieces in `tests/mesh_set.rs`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use rwarp::mesh::{CellMap, MeshTransformer, QuadMesh};
use rwarp::transform::{SourceCells, SourceWrap};
use rwarp::warp::{warp_to, warp_to_par, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, exact_weights_par, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

/// Destination pixels `step` units square from `(x0, y0)` down and right,
/// into a polar source grid: column by angle (80 over a quarter turn), row
/// by radius (half a unit). Straight destination edges bend in the source,
/// most near the origin.
#[derive(Clone, Copy)]
struct Polar {
    x0: f64,
    y0: f64,
    step: f64,
}

/// 12 by 12 pixels from (4, 28) on an 80 by 80 source: every edge bends.
const POLAR: Polar = Polar { x0: 4.0, y0: 28.0, step: 2.0 };
const SRC: (usize, usize) = (80, 80);
const DST: (usize, usize) = (12, 12);

impl Transformer for Polar {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let dth = std::f64::consts::FRAC_PI_2 / 80.0;
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            let (wx, wy) = (self.x0 + self.step * *x, self.y0 - self.step * *y);
            *x = wy.atan2(wx) / dth;
            *y = wx.hypot(wy) / 0.5;
        }
        vec![true; x.len()]
    }
}

/// A transformer that counts the points it moves.
struct Counting<'a, T> {
    inner: T,
    n: &'a Cell<usize>,
}

impl<T: Transformer> Transformer for Counting<'_, T> {
    fn transform(&self, d: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        self.n.set(self.n.get() + x.len());
        self.inner.transform(d, x, y)
    }
    fn source_wrap(&self) -> Option<SourceWrap> {
        self.inner.source_wrap()
    }
    fn source_cells(&self) -> Option<&dyn SourceCells> {
        self.inner.source_cells()
    }
}

fn weights_in(t: &impl Transformer, src: (usize, usize), dst: (usize, usize), opts: &WeightOptions) -> Weights {
    exact_weights(t, &GridWindow::new(src.0, src.1), dst.0, dst.1, opts)
}

fn weights(t: &impl Transformer, opts: &WeightOptions) -> Weights {
    weights_in(t, SRC, DST, opts)
}

/// The weights and the points the transformer moved for them.
fn counted<T: Transformer + Copy>(t: T, src: (usize, usize), dst: (usize, usize), opts: &WeightOptions) -> (Weights, usize) {
    let n = Cell::new(0);
    let w = weights_in(&Counting { inner: t, n: &n }, src, dst, opts);
    (w, n.get())
}

fn adaptive(tolerance: f64) -> WeightOptions {
    WeightOptions::new(1).with_refine_tolerance(tolerance)
}

type Key = (u32, u32, u32, u32);

fn entries(w: &Weights) -> HashMap<Key, f64> {
    (0..w.len()).map(|k| ((w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]), w.frac[k])).collect()
}

/// The largest difference between two tables' overlaps, pair by pair (a
/// pair in one only counts whole).
fn max_diff(a: &Weights, b: &Weights) -> f64 {
    let (a, b) = (entries(a), entries(b));
    a.iter()
        .map(|(k, v)| (v - b.get(k).copied().unwrap_or(0.0)).abs())
        .chain(b.iter().filter(|(k, _)| !a.contains_key(k)).map(|(_, v)| v.abs()))
        .fold(0.0, f64::max)
}

/// The same table bit for bit, NaN areas included.
fn same_weights(a: &Weights, b: &Weights) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    (a.dst_row == b.dst_row && a.dst_col == b.dst_col && a.src_row == b.src_row && a.src_col == b.src_col)
        && bits(&a.frac) == bits(&b.frac)
        && bits(&a.dst_area) == bits(&b.dst_area)
        && bits(&a.src_cell_area) == bits(&b.src_cell_area)
}

/// The error falls with the tolerance, about one to one (the overlaps
/// are fractions of a source pixel, the tolerance is in source pixels),
/// and the points grow as its inverse square root: a hundredfold for a
/// tolerance 1e4 times finer. The first three of each (points, error):
/// (1761, 1.0e-2), (5777, 8.9e-4), (19785, 8.0e-5).
#[test]
fn the_error_follows_the_tolerance_at_the_square_root_rate() {
    let reference = weights(&POLAR, &WeightOptions::new(4096));
    let mut points = Vec::new();
    for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        let (w, n) = counted(POLAR, SRC, DST, &adaptive(tol));
        let err = max_diff(&w, &reference);
        eprintln!("tolerance {tol:.0e}: {n} points, error {err:.1e}");
        assert!(err < 1.5 * tol && err > 0.05 * tol, "tolerance {tol:e}: error {err:e}");
        points.push(n as f64);
    }
    let rate = (points[4] / points[0]).log10() / 4.0;
    eprintln!("points grow as the tolerance to the power -{rate:.3}");
    assert!((0.45..0.55).contains(&rate), "{rate}");
    for p in points.windows(2) {
        assert!(p[1] > 2.5 * p[0] && p[1] < 4.0 * p[0], "{p:?}");
    }
}

/// A fine tolerance gives what a fine fixed densify gives: overlaps to
/// 7.8e-8, outline areas to 1.7e-8 relative (an overlap only one has is
/// counted whole).
#[test]
fn a_fine_tolerance_is_a_fine_densify() {
    let fine = weights(&POLAR, &WeightOptions::new(4096));
    let a = weights(&POLAR, &adaptive(1e-7));
    let d = max_diff(&a, &fine);
    let area = a.dst_area.iter().zip(&fine.dst_area).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max);
    eprintln!("tolerance 1e-7 vs densify 4096: overlaps to {d:.1e}, outline areas to {area:.1e}");
    assert!(d < 2e-7 && area < 5e-8, "{d:e} {area:e}");
}

/// Where only some edges bend, only they get points: 48 by 4 pixels a
/// unit square along the x axis from 0.5, bending most at the first. At a
/// tolerance of 1e-4 the error is densify 64's, from under half its
/// points (13313 against 27713).
#[test]
fn points_go_where_the_edges_bend() {
    let p = Polar { x0: 0.5, y0: 4.5, step: 1.0 };
    let (src, dst) = ((80, 110), (48, 4));
    let reference = weights_in(&p, src, dst, &WeightOptions::new(4096));
    let (a, na) = counted(p, src, dst, &adaptive(1e-4));
    let (f, nf) = counted(p, src, dst, &WeightOptions::new(64));
    let (ea, ef) = (max_diff(&a, &reference), max_diff(&f, &reference));
    eprintln!("tolerance 1e-4: {na} points, error {ea:.1e}; densify 64: {nf} points, error {ef:.1e}");
    assert!(ea < 1.2 * ef, "{ea:e} vs {ef:e}");
    assert!((na as f64) < 0.6 * nf as f64, "{na} vs {nf}");
}

/// A rotation and scale straight into source pixels.
#[derive(Clone, Copy)]
struct Rotated;

impl Transformer for Rotated {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (s, c) = 0.5f64.sin_cos();
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            let (u, v) = (*x * 2.7, *y * 2.7);
            *x = 30.0 + c * u - s * v;
            *y = 5.0 + s * u + c * v;
        }
        vec![true; x.len()]
    }
}

/// Where the transform is affine, nothing is added: the same weights, bit
/// for bit, as the fixed densify they start from.
#[test]
fn nothing_is_added_where_the_transform_is_affine() {
    for densify in [1, 3] {
        let fixed = weights(&Rotated, &WeightOptions::new(densify));
        assert!(fixed.len() > 1000);
        for tol in [1e-2, 1e-6] {
            let a = weights(&Rotated, &WeightOptions::new(densify).with_refine_tolerance(tol));
            assert!(same_weights(&a, &fixed), "densify {densify}, tolerance {tol:e}");
        }
    }
}

/// Tiles of the destination, and threads, give the whole window's
/// weights bit for bit: a segment's points depend only on its ends.
#[test]
fn tiles_and_threads_are_the_whole() {
    let opts = WeightOptions::new(2).with_refine_tolerance(1e-4);
    let whole = weights(&POLAR, &opts);
    let mut tiled: HashMap<Key, u64> = HashMap::new();
    let mut areas = vec![f64::NAN; DST.0 * DST.1];
    for (c0, c1) in [(0, 5), (5, 12)] {
        for (r0, r1) in [(0, 7), (7, 12)] {
            let w = exact_weights(&POLAR, &GridWindow::new(SRC.0, SRC.1), c1 - c0, r1 - r0, &opts.with_dst_offset(c0, r0));
            for (k, v) in entries(&w) {
                assert!(tiled.insert(k, v.to_bits()).is_none());
            }
            for r in 0..r1 - r0 {
                for c in 0..c1 - c0 {
                    areas[(r0 + r) * DST.0 + c0 + c] = w.dst_area[r * (c1 - c0) + c];
                }
            }
        }
    }
    let whole_bits: HashMap<Key, u64> = entries(&whole).into_iter().map(|(k, v)| (k, v.to_bits())).collect();
    assert_eq!(whole_bits, tiled);
    assert!(whole.dst_area.iter().zip(&areas).all(|(a, b)| a.to_bits() == b.to_bits()));
    for threads in [0, 3] {
        let par = exact_weights_par(|| Ok(POLAR), &GridWindow::new(SRC.0, SRC.1), DST.0, DST.1, &opts, threads).unwrap();
        assert!(same_weights(&par, &whole), "{threads} threads");
    }
}

/// A smooth field.
fn field(ncol: usize, nrow: usize) -> Vec<f64> {
    (0..ncol * nrow).map(|k| 10.0 + ((k % ncol) as f64 * 0.21).sin() * ((k / ncol) as f64 * 0.13).cos()).collect()
}

/// `warp()`'s `Average` with the adaptive footprint is `apply(FracArea)`
/// of the weights with the same tolerance, warp-compatible, bit for bit;
/// on threads too.
#[test]
fn warp_is_apply_of_the_weights() {
    let data = field(SRC.0, SRC.1);
    let buf = SourceBuffer { data: &data, ncol: SRC.0, nrow: SRC.1, nbands: 1, col_off: 0, row_off: 0 };
    for (densify, tol) in [(1, 1e-3), (2, 1e-5)] {
        let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Adaptive { densify, tolerance: tol });
        let warped = warp_to::<f64, f64>(&POLAR, &buf, DST.0, DST.1, &opts);
        let w = weights(&POLAR, &WeightOptions::warp_compatible(densify).with_refine_tolerance(tol));
        let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
        assert!(warped.iter().all(|v| (8.0..12.0).contains(v)));
        assert!(warped.iter().zip(&applied).all(|(a, b)| a.to_bits() == b.to_bits()), "densify {densify}, tolerance {tol:e}");
        let par = warp_to_par::<f64, f64, _>(|| Ok(POLAR), &buf, DST.0, DST.1, &opts, 3).unwrap();
        assert!(warped.iter().zip(&par).all(|(a, b)| a.to_bits() == b.to_bits()));
        // and it isn't the unrefined footprint's
        let plain = warp_to::<f64, f64>(&POLAR, &buf, DST.0, DST.1, &opts.with_footprint(Footprint::Exact { densify }));
        assert!(warped.iter().zip(&plain).any(|(a, b)| a != b));
    }
}

/// `POLAR`, failing between destination x 3.25 and 3.75.
#[derive(Clone, Copy)]
struct Gap;

impl Transformer for Gap {
    fn transform(&self, d: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let gap: Vec<bool> = x.iter().map(|x| *x > 3.25 && *x < 3.75).collect();
        let mut ok = POLAR.transform(d, x, y);
        for (k, g) in gap.into_iter().enumerate() {
            if g {
                (x[k], y[k], ok[k]) = (f64::NAN, f64::NAN, false);
            }
        }
        ok
    }
}

/// A new point that fails fails its pixel, as a failed lattice point
/// does: the midpoint at x 3.5 fails column 3 and nothing else. Without
/// refinement no point lands there.
#[test]
fn a_failed_point_fails_its_pixel() {
    let failed = |w: &Weights| (0..DST.0 * DST.1).filter(|&k| w.dst_area[k].is_nan()).map(|k| k % DST.0).collect::<Vec<_>>();
    assert!(failed(&weights(&Gap, &WeightOptions::new(1))).is_empty());
    let a = weights(&Gap, &adaptive(1e-3));
    assert_eq!(failed(&a), vec![3; DST.1]);
    assert!(a.dst_col.iter().all(|&c| c != 3));
    // the others are as without the gap
    let (ea, eb) = (entries(&a), entries(&weights(&POLAR, &adaptive(1e-3))));
    assert!(eb.iter().all(|(k, v)| k.1 == 3 || ea[k].to_bits() == v.to_bits()));
}

// ---- a mesh source, refined in its world coordinates ------------------------------

const NCOL: usize = 48;
const NROW: usize = 16;

/// `tests/mesh_transformer.rs`'s tripolar cap: confocal ellipses about
/// grid poles at `(+-a, 0)`, a seam into one grid pole and a fold between
/// the two, turned and shifted, in metres of a north polar LAEA.
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

/// North polar LAEA on a sphere, written out: longitude and latitude to
/// metres (`true`) and back. Counts the points it takes to metres.
struct Laea(bool);

const R: f64 = 6371229.0;

static TO_LAEA: AtomicUsize = AtomicUsize::new(0);

impl CrsTransform for Laea {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some(if self.0 {
            TO_LAEA.fetch_add(1, Ordering::Relaxed);
            let rho = 2.0 * R * ((90.0 - y).to_radians() / 2.0).sin();
            (rho * x.to_radians().sin(), -rho * x.to_radians().cos())
        } else {
            let rho = x.hypot(y);
            (x.atan2(-y).to_degrees(), 90.0 - 2.0 * (rho / (2.0 * R)).asin().to_degrees())
        })
    }
}

/// The cap from 6 by 3 degree pixels in longitude and latitude, north of
/// 54 N: their parallels are arcs of circles in the mesh's LAEA, their
/// meridians straight.
fn mesh_lonlat() -> MeshTransformer {
    MeshTransformer::with_backend(tripolar(), [0.0, 6.0, 0.0, 90.0, 0.0, -3.0], Box::new(Laea(true)), Box::new(Laea(false)))
        .unwrap()
}

const MESH_DST: (usize, usize) = (60, 12);

/// A mesh source's outlines are refined in its world coordinates, the
/// tolerance in metres: fractions of its cells to 5.5e-4 at 100 m and
/// 9e-6 at 1 m, against densify 1024. (An arc of a circle bisects evenly,
/// so its points are a fixed densify's: the worst overlap at 1 m is
/// densify 64's, bit for bit.) `warp()` is `apply` of the weights, on
/// threads too.
#[test]
fn a_mesh_source_refines_in_its_world() {
    let t = mesh_lonlat();
    let w = |o: &WeightOptions| exact_weights(&t, &t.source_window(), MESH_DST.0, MESH_DST.1, o);
    let reference = w(&WeightOptions::new(1024));
    let mut errs = Vec::new();
    for tol in [1e2, 1.0] {
        let a = w(&adaptive(tol));
        let err = max_diff(&a, &reference);
        eprintln!("tolerance {tol} m: error {err:.1e}");
        assert!(a.dst_area.iter().all(|a| a.is_finite()));
        errs.push(err);
    }
    assert!(errs[0] < 1e-3 && errs[1] < 2e-5, "{errs:?}");
    assert!(max_diff(&w(&WeightOptions::new(1)), &reference) > 1e-2);

    let data = field(NCOL, NROW);
    let buf = SourceBuffer { data: &data, ncol: NCOL, nrow: NROW, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Adaptive { densify: 1, tolerance: 10.0 });
    let warped = warp_to::<f64, f64>(&t, &buf, MESH_DST.0, MESH_DST.1, &opts);
    let wc = WeightOptions::warp_compatible(1).with_refine_tolerance(10.0);
    let applied: Vec<f64> = exact_weights(&t, &GridWindow::from(&buf), MESH_DST.0, MESH_DST.1, &wc)
        .apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
    let computed = warped.iter().filter(|v| (8.0..12.0).contains(*v)).count();
    assert!(computed > 600, "{computed}");
    assert!(warped.iter().zip(&applied).all(|(a, b)| a.to_bits() == b.to_bits()));
    let par = warp_to_par::<f64, f64, _>(|| Ok(mesh_lonlat()), &buf, MESH_DST.0, MESH_DST.1, &opts, 3).unwrap();
    assert!(warped.iter().zip(&par).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// Only meridians are straight in the LAEA: the points land on parallels.
/// With the destination edges straight in the world (the same CRS) a mesh
/// source gets nothing new.
#[test]
fn straight_world_edges_get_nothing_in_a_mesh() {
    let gt = [-3.9e6, 1.5e5, 0.0, 3.9e6, 0.0, -1.5e5];
    let t = MeshTransformer::with_backend(tripolar(), gt, Box::new(Same), Box::new(Same)).unwrap();
    let w = |o: &WeightOptions| exact_weights(&t, &t.source_window(), 52, 52, o);
    let fixed = w(&WeightOptions::new(2));
    assert!(fixed.len() > 1000);
    assert!(same_weights(&w(&WeightOptions::new(2).with_refine_tolerance(1.0)), &fixed));
    // the CRS change does: points on the parallels only
    let n0 = TO_LAEA.load(Ordering::Relaxed);
    let _ = exact_weights(&mesh_lonlat(), &t.source_window(), MESH_DST.0, MESH_DST.1, &adaptive(1.0));
    assert!(TO_LAEA.load(Ordering::Relaxed) - n0 > 10 * 61 * 13);
}

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

#[test]
#[should_panic(expected = "must be positive")]
fn a_tolerance_that_is_not_positive_is_refused() {
    weights(&POLAR, &adaptive(0.0));
}
