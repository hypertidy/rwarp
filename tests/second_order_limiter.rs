//! The second-order limiter (`Limiter::BarthJespersen` in
//! `ApplyOptions::limiter`, conservative stage 6d), on sharp fronts and
//! smooth fields:
//!
//! - without it, values next to a front overshoot the values they are
//!   drawn from; with it, each destination value (`FracArea`) is within
//!   the values of its sources and their neighbours, for a grid's cells,
//!   a grid with world cells, and a mesh's faces, band by band, cells
//!   without data left out;
//! - the integral is conserved as without it;
//! - inside a window a linear field is untouched, to rounding; on the
//!   window's edge, where a plane is one-sided, it is limited;
//! - on a smooth field it costs some accuracy (flattening extrema), and
//!   stays well ahead of first order.

use std::collections::HashMap;

use controlledburn::{CoverageWindow, Measure};
use rwarp::mesh::{CellMap, MeshTransformer, PolyMesh, QuadMesh};
use rwarp::warp::{Nodata, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, ApplyOptions, GridWindow, Limiter, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

/// A rotation and scale straight into the source's (or grid's) pixels.
#[derive(Clone, Copy)]
struct Rotated {
    angle: f64,
    scale: f64,
    origin: [f64; 2],
}

impl Rotated {
    fn at(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let (u, v) = (p[0] * self.scale, p[1] * self.scale);
        [self.origin[0] + c * u - s * v, self.origin[1] + s * u + c * v]
    }
}

impl Transformer for Rotated {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            [*x, *y] = self.at([*x, *y]);
        }
        vec![true; x.len()]
    }
}

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// The source window: 40 x 32 cells at (5, 3) in the full grid.
fn src_window() -> GridWindow {
    GridWindow::new(40, 32).with_offset(5, 3)
}

/// Finer destination cells, inside the source.
fn finer() -> Rotated {
    Rotated { angle: 0.4, scale: 0.83, origin: [16.0, 4.0] }
}

/// Coarser destination cells, over the source's edge.
fn over_the_edge() -> Rotated {
    Rotated { angle: 0.4, scale: 1.37, origin: [-6.0, -9.0] }
}

/// Rows of 0.25 degrees going south from 50 S, over the full source grid.
fn sphere() -> Measure {
    Measure::cos_latitude((-50.0f64).to_radians(), (-0.25f64).to_radians())
}

/// Two fronts: a step across a line, and a disc of 2 on it.
fn fronts(p: [f64; 2]) -> f64 {
    let step = if 0.6 * p[0] + 0.8 * p[1] > 22.0 { 1.0 } else { 0.0 };
    let disc = if (p[0] - 25.0).hypot(p[1] - 18.0) < 6.0 { 2.0 } else { 0.0 };
    step + disc
}

/// The other band's: a step the other way, and a strip.
fn other_fronts(p: [f64; 2]) -> f64 {
    let step = if 0.9 * p[0] - 0.45 * p[1] > 12.0 { -1.5 } else { 0.5 };
    let strip = if (p[1] - 20.0).abs() < 2.5 { 4.0 } else { 0.0 };
    step + strip
}

/// A field `f` at each cell's middle over the window.
fn field(src: &GridWindow, f: impl Fn([f64; 2]) -> f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(src.ncol * src.nrow);
    for r in 0..src.nrow {
        for c in 0..src.ncol {
            v.push(f([(src.col_off + c) as f64 + 0.5, (src.row_off + r) as f64 + 0.5]));
        }
    }
    v
}

fn buffer<'a>(src: &GridWindow, data: &'a [f64], nbands: usize) -> SourceBuffer<'a, f64> {
    SourceBuffer { data, ncol: src.ncol, nrow: src.nrow, nbands, col_off: src.col_off, row_off: src.row_off }
}

fn limited() -> ApplyOptions {
    ApplyOptions::new().with_limiter(Limiter::BarthJespersen)
}

fn dst_cell(w: &Weights, k: usize) -> usize {
    (w.dst_row[k] as usize - w.dst_off[1]) * w.dst_size[0] + (w.dst_col[k] as usize - w.dst_off[0])
}

/// Each destination cell's bounds: the lowest and highest of `value(p)`
/// (`None` without data) over its sources and, for each, the window cells
/// `round(p)` gives.
fn bounds(w: &Weights, src_of: impl Fn(usize) -> usize, round: impl Fn(usize) -> Vec<usize>, value: impl Fn(usize) -> Option<f64>) -> Vec<(f64, f64)> {
    let mut b = vec![(f64::INFINITY, f64::NEG_INFINITY); w.dst_size[0] * w.dst_size[1]];
    for k in 0..w.len() {
        let p = src_of(k);
        if value(p).is_none() {
            continue;
        }
        let cell = dst_cell(w, k);
        for q in round(p).into_iter().chain([p]) {
            if let Some(v) = value(q) {
                b[cell] = (b[cell].0.min(v), b[cell].1.max(v));
            }
        }
    }
    b
}

/// A grid source's entry `k` to its window cell.
fn window_cell<'a>(w: &'a Weights, src: &GridWindow) -> impl Fn(usize) -> usize + 'a {
    let (ro, co, ncol) = (src.row_off, src.col_off, src.ncol);
    move |k| (w.src_row[k] as usize - ro) * ncol + (w.src_col[k] as usize - co)
}

/// The eight cells round window cell `p`, in the window.
fn round_in(src: &GridWindow) -> impl Fn(usize) -> Vec<usize> {
    let (ncol, nrow) = (src.ncol as i64, src.nrow as i64);
    move |p| {
        let (r, c) = ((p as i64) / ncol, (p as i64) % ncol);
        let mut v = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (r + dr, c + dc);
                if (dr, dc) != (0, 0) && r >= 0 && c >= 0 && r < nrow && c < ncol {
                    v.push((r * ncol + c) as usize);
                }
            }
        }
        v
    }
}

/// How far the values `v` go past their bounds, at most, and over how
/// many cells.
fn overshoot(v: &[f64], b: &[(f64, f64)]) -> (f64, usize) {
    let (mut worst, mut n) = (0.0f64, 0);
    for (&x, &(lo, hi)) in v.iter().zip(b) {
        if lo <= hi && x.is_finite() {
            worst = worst.max(lo - x).max(x - hi);
            n += 1;
        }
    }
    (worst, n)
}

#[test]
fn it_is_off_unless_asked_for() {
    assert_eq!(ApplyOptions::new().limiter, Limiter::Off);
    assert_eq!(ApplyOptions::from(&WarpOptions::default()).limiter, Limiter::Off);
    assert_eq!(limited().limiter, Limiter::BarthJespersen);
}

#[test]
fn fronts_don_t_overshoot() {
    let src = src_window();
    let data = field(&src, fronts);
    for measure in [Measure::Planar, sphere()] {
        for (name, t, dw, dh) in [("finer", finer(), 30, 24), ("over the edge", over_the_edge(), 40, 36)] {
            let w = exact_weights(&t, &src, dw, dh, &WeightOptions::new(2).with_measure(measure).with_moments(true));
            let b = bounds(&w, window_cell(&w, &src), round_in(&src), |p| Some(data[p]));
            let buf = buffer(&src, &data, 1);
            let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
            let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &limited());
            let ((a, n), (z, _)) = (overshoot(&free, &b), overshoot(&held, &b));
            eprintln!("{measure:?} {name}: over {n} cells, without the limiter past the bounds by {a:.3}, with it {z:.1e}");
            assert!(n > 500 && a > 0.05, "{n} {a}");
            assert!(z < 1e-13, "{z}");
            // and so within the field's range
            assert!(held.iter().filter(|v| v.is_finite()).all(|&v| (-1e-13..=3.0 + 1e-13).contains(&v)));
        }
    }
}

#[test]
fn the_integral_is_conserved() {
    // the destination over the source's edge cuts its edge cells, and a
    // few cells have no data
    let src = src_window();
    let t = over_the_edge();
    for measure in [Measure::Planar, sphere()] {
        let w = exact_weights(&t, &src, 40, 36, &WeightOptions::new(2).with_measure(measure).with_moments(true));
        let mut data = field(&src, fronts);
        for k in [17usize, 400, 401, 441, 900] {
            data[k] = -9999.0;
        }
        let opts = limited().with_src_nodata(Nodata::Value(-9999.0));
        let v: Vec<f64> = w.apply_second_order(&buffer(&src, &data, 1), Normalization::DestArea, &opts);
        let free: Vec<f64> = w.apply_second_order(&buffer(&src, &data, 1), Normalization::DestArea, &opts.clone().with_limiter(Limiter::Off));
        assert_ne!(v, free, "the limiter did nothing");
        let mut got = 0.0;
        for (k, &x) in v.iter().enumerate() {
            if x.is_finite() && x != -9999.0 && w.dst_area[k].is_finite() {
                got += x * w.dst_area[k];
            }
        }
        let mut want = 0.0;
        for k in 0..w.len() {
            let p = window_cell(&w, &src)(k);
            if data[p] != -9999.0 {
                want += data[p] * w.frac[k] * w.src_area(k);
            }
        }
        eprintln!("{measure:?}: integral {got} vs {want}");
        assert!((got - want).abs() < 1e-12 * want.abs(), "{measure:?}: {got} vs {want}");
    }
}

/// The centroid of a whole cell under `measure`.
fn cell_centroid(measure: Measure, r: i64, c: i64) -> [f64; 2] {
    let (x, y) = (c as f64, r as f64);
    let (xs, ys) = ([x, x + 1.0, x + 1.0, x], [y, y, y + 1.0, y + 1.0]);
    let mut w = CoverageWindow::with_measure(measure);
    w.set_moments(true);
    w.reset(c, r, 1, 1);
    w.add_ring(&xs, &ys, 1.0);
    let cov = w.resolve()[0];
    [x + w.moment_x()[0] / cov, y + w.moment_y()[0] / cov]
}

#[test]
fn a_linear_field_is_untouched_inside() {
    let src = src_window();
    let t = over_the_edge();
    let lin = |p: [f64; 2]| 3.0 + 0.7 * p[0] - 1.3 * p[1];
    for measure in [Measure::Planar, sphere()] {
        let w = exact_weights(&t, &src, 40, 36, &WeightOptions::new(2).with_measure(measure).with_moments(true));
        let mut data = Vec::new();
        for r in 0..src.nrow as i64 {
            for c in 0..src.ncol as i64 {
                data.push(lin(cell_centroid(measure, src.row_off as i64 + r, src.col_off as i64 + c)));
            }
        }
        let buf = buffer(&src, &data, 1);
        let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
        let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &limited());
        // destination cells whose sources all have their eight neighbours
        let mut inside = vec![true; free.len()];
        let edge = |k: usize| {
            let (r, c) = (w.src_row[k] as usize - src.row_off, w.src_col[k] as usize - src.col_off);
            r == 0 || c == 0 || r + 1 == src.nrow || c + 1 == src.ncol
        };
        for k in 0..w.len() {
            if edge(k) {
                inside[dst_cell(&w, k)] = false;
            }
        }
        let n = inside.iter().zip(&free).filter(|(i, v)| **i && v.is_finite()).count();
        // the same but for slivers' gradient terms, which the limiter drops
        let gap = (0..free.len()).filter(|&k| inside[k]).map(|k| (free[k] - held[k]).abs()).fold(0.0, f64::max);
        let moved = (0..free.len()).filter(|&k| inside[k] && free[k].to_bits() != held[k].to_bits()).count();
        let limited = (0..free.len()).filter(|&k| !inside[k] && (free[k] - held[k]).abs() > 1e-6).count();
        eprintln!("{measure:?}: {n} cells inside the same to {gap:.1e} ({moved} not bit for bit); {limited} on the edge limited");
        assert!(n > 500 && gap < 1e-13 && limited > 10, "{n} {gap} {limited}");
    }
}

/// The mean of `sin(x / a) cos(y / b)` over a unit cell, in closed form.
fn sine_mean(a: f64, b: f64, x: f64, y: f64) -> f64 {
    a * ((x / a).cos() - ((x + 1.0) / a).cos()) * b * (((y + 1.0) / b).sin() - (y / b).sin())
}

/// The mean of `f` over destination cell `(r, c)`, a parallelogram under
/// `t`, by 12 x 12 Gauss-Legendre points.
fn dst_mean(t: &Rotated, r: usize, c: usize, f: impl Fn([f64; 2]) -> f64) -> f64 {
    const G: [(f64, f64); 6] = [
        (0.1252334085114689, 0.2491470458134028),
        (0.3678314989981802, 0.2334925365383548),
        (0.5873179542866175, 0.2031674267230659),
        (0.7699026741943047, 0.1600783285433462),
        (0.9041172563704749, 0.1069393259953184),
        (0.9815606342467192, 0.0471753363865118),
    ];
    let mut s = 0.0;
    for &(u, wu) in &G {
        for u in [u, -u] {
            for &(v, wv) in &G {
                for v in [v, -v] {
                    s += wu * wv * f(t.at([c as f64 + 0.5 * (1.0 + u), r as f64 + 0.5 * (1.0 + v)]));
                }
            }
        }
    }
    s / 4.0
}

/// The destination cells whose sources the table covers wholly and whose
/// sources' neighbours are all in the window, as in `second_order.rs`:
/// elsewhere a plane is centred off its cell's middle, and the method is
/// first order there, limited or not.
fn interior(w: &Weights, src: &GridWindow) -> Vec<bool> {
    let mut covered = HashMap::<(u32, u32), f64>::new();
    for k in 0..w.len() {
        *covered.entry((w.src_row[k], w.src_col[k])).or_default() += w.frac[k];
    }
    let mut ok = vec![true; w.dst_size[0] * w.dst_size[1]];
    for k in 0..w.len() {
        let (r, c) = (w.src_row[k] as usize - src.row_off, w.src_col[k] as usize - src.col_off);
        let full = (covered[&(w.src_row[k], w.src_col[k])] - 1.0).abs() < 1e-12;
        if !full || r == 0 || c == 0 || r + 1 == src.nrow || c + 1 == src.ncol {
            ok[dst_cell(w, k)] = false;
        }
    }
    ok
}

/// The limiter flattens planes at a smooth field's extrema too: it costs
/// some accuracy there, the more the less the field is resolved, and it
/// stays far ahead of first order.
#[test]
fn a_smooth_field_loses_a_little() {
    let src = src_window();
    let t = finer();
    let w = exact_weights(&t, &src, 30, 24, &WeightOptions::new(1).with_moments(true));
    let inside = interior(&w, &src);
    // the wavelengths from 75 cells to 19
    let mut last = f64::NAN;
    for (a, b) in [(12.0, 8.0), (6.0, 4.0), (3.0, 2.0)] {
        let mut data = Vec::new();
        for r in 0..src.nrow {
            for c in 0..src.ncol {
                data.push(sine_mean(a, b, (src.col_off + c) as f64, (src.row_off + r) as f64));
            }
        }
        let buf = buffer(&src, &data, 1);
        let first: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
        let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
        let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &limited());
        let (mut e, mut n) = ([0.0f64; 3], 0.0);
        for r in 0..24 {
            for c in 0..30 {
                if !inside[r * 30 + c] {
                    continue;
                }
                let want = dst_mean(&t, r, c, |p| (p[0] / a).sin() * (p[1] / b).cos());
                let k = r * 30 + c;
                for (e, v) in e.iter_mut().zip([&first, &free, &held]) {
                    *e += (v[k] - want).powi(2);
                }
                n += 1.0;
            }
        }
        let [e1, e2, el] = e.map(|e| (e / n).sqrt());
        eprintln!("sine at {a}: rms first order {e1:.2e}, second {e2:.2e}, second limited {el:.2e}");
        // dearer than without it, at most twice, and far ahead of first order
        assert!(n > 400.0 && e2 < el && el < 2.0 * e2 && el < 0.15 * e1, "{a}: {e1} {e2} {el}");
        // halving the wavelength is doubling the cells: second order's
        // error grows four times, first order's twice, the limited one's
        // at least three
        if last.is_finite() {
            assert!(el / last > 3.0, "{a}: x{}", el / last);
        }
        last = el;
    }
}

#[test]
fn bands_and_cells_without_data() {
    // two bands with different fronts, and a strip of cells without data
    // across both
    let src = src_window();
    let t = finer();
    let w = exact_weights(&t, &src, 30, 24, &WeightOptions::new(2).with_moments(true));
    let (a, b) = (field(&src, fronts), field(&src, other_fronts));
    let gone = |p: usize| (p % src.ncol + 2 * (p / src.ncol)) % 9 == 0;
    let mut data = Vec::new();
    for p in 0..a.len() {
        if gone(p) {
            data.extend([-9999.0, -9999.0]);
        } else {
            data.extend([a[p], b[p]]);
        }
    }
    let opts = limited().with_src_nodata(Nodata::Value(-9999.0));
    let buf = buffer(&src, &data, 2);
    let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &opts);
    let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &opts.clone().with_limiter(Limiter::Off));
    for (band, v) in [&a, &b].into_iter().enumerate() {
        let bd = bounds(&w, window_cell(&w, &src), round_in(&src), |p| (!gone(p)).then_some(v[p]));
        let pick = |o: &[f64]| o.iter().skip(band).step_by(2).map(|&x| if x == -9999.0 { f64::NAN } else { x }).collect::<Vec<_>>();
        let ((x, n), (z, _)) = (overshoot(&pick(&free), &bd), overshoot(&pick(&held), &bd));
        eprintln!("band {band}: over {n} cells, without the limiter past the bounds by {x:.3}, with it {z:.1e}");
        assert!(n > 500 && x > 0.05 && z < 1e-13, "band {band}: {n} {x} {z}");
    }
}

/// A window leaves slivers of rounding in a table (2 to 4% of the entries
/// here, under 1e-12 of their cell, centroids up to 45 cells away). The
/// limiter leaves them out, and gives them their cell's value: so a
/// destination cell with nothing else, which `FracArea` makes all
/// sliver, stays in bounds. One such cell, added to a real table.
#[test]
fn a_sliver_takes_its_cell_s_value() {
    let src = src_window();
    let mut w = exact_weights(&over_the_edge(), &src, 40, 36, &WeightOptions::new(2).with_moments(true));
    // a linear field, whose planes the limiter leaves as they are inside
    let data = field(&src, |p| 3.0 + 0.7 * p[0] - 1.3 * p[1]);
    let buf = buffer(&src, &data, 1);
    let k = (0..w.len())
        .find(|&k| {
            let p = window_cell(&w, &src)(k);
            let (r, c) = (p / src.ncol, p % src.ncol);
            w.frac[k] > 0.5 && r > 0 && c > 0 && r + 1 < src.nrow && c + 1 < src.ncol
        })
        .unwrap();
    let (sr, sc) = (w.src_row[k], w.src_col[k]);
    let own = data[window_cell(&w, &src)(k)];
    // a destination cell with no entries, and a sliver of that source cell
    // in it, centred 30 cells off
    let covered = w.dst_covered();
    let empty = (0..covered.len()).find(|&c| covered[c] == 0.0 && w.dst_area[c].is_finite()).unwrap();
    let (dr, dc) = ((w.dst_off[1] + empty / w.dst_size[0]) as u32, (w.dst_off[0] + empty % w.dst_size[0]) as u32);
    let f = 1e-17;
    w.dst_row.push(dr);
    w.dst_col.push(dc);
    w.src_row.push(sr);
    w.src_col.push(sc);
    w.frac.push(f);
    w.moment.push([f * 30.5, f * 0.5]);
    let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &limited());
    eprintln!("a sliver alone: {} without the limiter, {} with it, its cell's value {own}", free[empty], held[empty]);
    assert!((free[empty] - own).abs() > 10.0, "{}", free[empty]);
    assert_eq!(held[empty], own);
}

// ---- a grid with world cells, and a mesh's faces ------------------------------

/// A curvilinear grid by its corners, as in `second_order_world.rs`.
fn curved(ncol: usize, nrow: usize) -> QuadMesh {
    let curve = |s: f64, t: f64| {
        [140.0 + 10.0 * s + 0.6 * (1.3 + 2.5 * t).sin() * s, -32.0 - 8.0 * t + 0.5 * (0.7 + 3.0 * s).sin() + 0.3 * s * t]
    };
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = curve(i as f64 / ncol as f64, j as f64 / nrow as f64);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap()
}

#[test]
fn world_cells_don_t_overshoot() {
    // the curved grid's cells as the source, onto a regular grid finer
    // than them; the fronts in the grid's indices
    let t = MeshTransformer::with_backend(curved(40, 30), [141.0, 0.2, 0.0, -33.0, 0.0, -0.2], Box::new(Same), Box::new(Same)).unwrap();
    let src = t.source_window();
    let w = exact_weights(&t, &src, 40, 30, &WeightOptions::new(2).with_moments(true));
    assert!(!w.src_cell_area.is_empty());
    let data = field(&src, |p| fronts([p[0] * 1.2 - 3.0, p[1] * 1.1 + 2.0]));
    let b = bounds(&w, window_cell(&w, &src), round_in(&src), |p| Some(data[p]));
    let buf = buffer(&src, &data, 1);
    let free: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let held: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &limited());
    let ((a, n), (z, _)) = (overshoot(&free, &b), overshoot(&held, &b));
    eprintln!("world cells: over {n} cells, without the limiter past the bounds by {a:.3}, with it {z:.1e}");
    assert!(n > 500 && a > 0.05 && z < 1e-13, "{n} {a} {z}");
    // conserved
    let v: Vec<f64> = w.apply_second_order(&buf, Normalization::DestArea, &limited());
    let got: f64 = v.iter().zip(&w.dst_area).filter(|(x, _)| x.is_finite()).map(|(x, a)| x * a).sum();
    let want: f64 = (0..w.len()).map(|k| data[window_cell(&w, &src)(k)] * w.frac[k] * w.src_area(k)).sum();
    assert!((got - want).abs() < 1e-12 * want.abs(), "{got} vs {want}");
}

/// A lattice of `nx` by `ny` unit boxes, its inner nodes moved a little,
/// every third box cut into two triangles, as in `second_order_faces.rs`.
fn mixed(nx: usize, ny: usize) -> PolyMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for k in 0..=ny {
        for i in 0..=nx {
            let inner = i > 0 && i < nx && k > 0 && k < ny;
            let j = if inner { 0.15 * (1.7 * i as f64 + 2.3 * k as f64).sin() } else { 0.0 };
            let l = if inner { 0.15 * (0.9 * i as f64 - 1.9 * k as f64).cos() } else { 0.0 };
            x.push(i as f64 + j);
            y.push(k as f64 + l);
        }
    }
    let n = |i: usize, k: usize| k * (nx + 1) + i;
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..ny {
        for i in 0..nx {
            let f = [n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)];
            if (i + 2 * k) % 3 == 0 {
                nodes.extend([f[0], f[1], f[2]]);
                start.push(nodes.len());
                nodes.extend([f[0], f[2], f[3]]);
            } else {
                nodes.extend(f);
            }
            start.push(nodes.len());
        }
    }
    PolyMesh::new(x, y, start, nodes).unwrap()
}

/// The faces sharing a node with each face.
fn neighbours(m: &PolyMesh) -> Vec<Vec<usize>> {
    let mut at: HashMap<usize, Vec<usize>> = HashMap::new();
    for f in 0..m.nface() {
        for &n in m.face(f) {
            at.entry(n).or_default().push(f);
        }
    }
    (0..m.nface()).map(|f| m.face(f).iter().flat_map(|n| at[n].iter().copied()).filter(|&g| g != f).collect()).collect()
}

#[test]
fn faces_don_t_overshoot() {
    // a mesh of unit faces, each over about three grid cells a side
    let mesh = mixed(14, 12);
    let t = Rotated { angle: 0.35, scale: 3.1, origin: [10.0, 2.0] };
    let grid = GridWindow::new(48, 44).with_offset(1, 1);
    let w = mesh.weights_to(&t, &grid, &WeightOptions::new(2).with_moments(true)).unwrap();
    // the fronts at each face's nodes' mean, in grid pixels
    let data: Vec<f64> = (0..mesh.nface())
        .map(|f| {
            let v = mesh.face(f);
            let m = v.iter().fold([0.0, 0.0], |s, &n| [s[0] + mesh.node(n)[0], s[1] + mesh.node(n)[1]]);
            let p = t.at([m[0] / v.len() as f64, m[1] / v.len() as f64]);
            fronts([p[0] * 0.8, p[1] * 0.8])
        })
        .collect();
    let nbrs = neighbours(&mesh);
    // the limiter's neighbours are some of these: the bounds are no tighter
    let b = bounds(&w, |k| w.src_col[k] as usize, |f| nbrs[f].clone(), |f| Some(data[f]));
    let buf = SourceBuffer { data: &data[..], ncol: data.len(), nrow: 1, nbands: 1, col_off: 0, row_off: 0 };
    let free: Vec<f64> = mesh.apply_second_order(&w, &buf, Normalization::FracArea, &ApplyOptions::new());
    let held: Vec<f64> = mesh.apply_second_order(&w, &buf, Normalization::FracArea, &limited());
    let ((a, n), (z, _)) = (overshoot(&free, &b), overshoot(&held, &b));
    eprintln!("faces: over {n} cells, without the limiter past the bounds by {a:.3}, with it {z:.1e}");
    assert!(n > 500 && a > 0.05 && z < 1e-13, "{n} {a} {z}");
    // conserved
    let v: Vec<f64> = mesh.apply_second_order(&w, &buf, Normalization::DestArea, &limited());
    let got: f64 = v.iter().zip(&w.dst_area).filter(|(x, _)| x.is_finite()).map(|(x, a)| x * a).sum();
    let want: f64 = (0..w.len()).map(|k| data[w.src_col[k] as usize] * w.frac[k] * w.src_area(k)).sum();
    assert!((got - want).abs() < 1e-12 * want.abs(), "{got} vs {want}");
}
