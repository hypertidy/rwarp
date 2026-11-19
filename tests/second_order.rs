//! Second-order conservative remapping (`Weights::apply_second_order`,
//! `Weights::second_order`), against exact answers: fields linear in
//! source pixels come out exactly (planar and on the sphere), the integral
//! is conserved, the folded table is the same method, the error falls as
//! the square of the cell size, and moments survive the seam, threads and
//! tiles.

use controlledburn::{CoverageWindow, Measure};
use rwarp::transform::SourceWrap;
use rwarp::warp::{Nodata, SourceBuffer};
use rwarp::weights::{exact_weights, exact_weights_par, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::Transformer;

/// A rotation and scale straight into source pixels.
struct Rotated {
    angle: f64,
    scale: f64,
    origin: (f64, f64),
}

impl Rotated {
    fn at(&self, x: f64, y: f64) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let (u, v) = (x * self.scale, y * self.scale);
        [self.origin.0 + c * u - s * v, self.origin.1 + s * u + c * v]
    }
}

impl Transformer for Rotated {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            [*x, *y] = self.at(*x, *y);
        }
        vec![true; x.len()]
    }
}

/// The source window: 40 x 32 cells at (5, 3) in the full grid.
fn src_window() -> GridWindow {
    GridWindow::new(40, 32).with_offset(5, 3)
}

fn rotated() -> Rotated {
    Rotated { angle: 0.4, scale: 0.83, origin: (16.0, 4.0) }
}

/// A field given by its value `f(col, row)` at each cell's centroid, over
/// the source window.
fn field(src: &GridWindow, centroid: impl Fn(i64, i64) -> [f64; 2], f: impl Fn([f64; 2]) -> f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(src.ncol * src.nrow);
    for r in 0..src.nrow as i64 {
        for c in 0..src.ncol as i64 {
            v.push(f(centroid(src.row_off as i64 + r, src.col_off as i64 + c)));
        }
    }
    v
}

fn buffer<'a>(src: &GridWindow, data: &'a [f64]) -> SourceBuffer<'a, f64> {
    SourceBuffer { data, ncol: src.ncol, nrow: src.nrow, nbands: 1, col_off: src.col_off, row_off: src.row_off }
}

/// The centroid of a polygon (source pixels, full grid) under `measure`,
/// from controlledburn's window moments.
fn centroid(measure: Measure, xs: &[f64], ys: &[f64]) -> [f64; 2] {
    let mut w = CoverageWindow::with_measure(measure);
    w.set_moments(true);
    assert!(w.reset_to_ring(xs, ys));
    w.add_ring(xs, ys, 1.0);
    w.resolve();
    let (mut a, mut mx, mut my) = (0.0, 0.0, 0.0);
    for k in 0..w.coverage().len() {
        let (i, j) = ((w.col0() + (k % w.ncol()) as i64) as f64, (w.row0() + (k / w.ncol()) as i64) as f64);
        let cell = measure.cell_area(j as i64);
        a += w.coverage()[k] * cell;
        mx += (i * w.coverage()[k] + w.moment_x()[k]) * cell;
        my += (j * w.coverage()[k] + w.moment_y()[k]) * cell;
    }
    [mx / a, my / a]
}

/// The centroid of a whole cell under `measure`.
fn cell_centroid(measure: Measure, r: i64, c: i64) -> [f64; 2] {
    let (x, y) = (c as f64, r as f64);
    centroid(measure, &[x, x + 1.0, x + 1.0, x], &[y, y, y + 1.0, y + 1.0])
}

/// The destination cells whose sources the table covers wholly and whose
/// sources' neighbours are all in the window: where a linear field must
/// come out exactly.
fn interior(w: &Weights, src: &GridWindow) -> Vec<bool> {
    let n = w.dst_size[0] * w.dst_size[1];
    let mut covered = std::collections::HashMap::<(u32, u32), f64>::new();
    for k in 0..w.len() {
        *covered.entry((w.src_row[k], w.src_col[k])).or_default() += w.frac[k];
    }
    let inner = |r: u32, c: u32| {
        let (r, c) = (r as usize, c as usize);
        r > src.row_off && c > src.col_off && r + 1 < src.row_off + src.nrow && c + 1 < src.col_off + src.ncol
    };
    let mut ok = vec![true; n];
    let mut any = vec![false; n];
    for k in 0..w.len() {
        let cell = (w.dst_row[k] as usize - w.dst_off[1]) * w.dst_size[0] + (w.dst_col[k] as usize - w.dst_off[0]);
        any[cell] = true;
        let full = (covered[&(w.src_row[k], w.src_col[k])] - 1.0).abs() < 1e-12;
        if !full || !inner(w.src_row[k], w.src_col[k]) {
            ok[cell] = false;
        }
    }
    ok.iter().zip(&any).map(|(&o, &a)| o && a).collect()
}

/// Rows of 0.25 degrees going south from 50 S, over the full source grid.
fn sphere() -> Measure {
    Measure::cos_latitude((-50.0f64).to_radians(), (-0.25f64).to_radians())
}

#[test]
fn linear_fields_come_out_exactly() {
    let t = rotated();
    let (dw, dh) = (30, 24);
    let f = |p: [f64; 2]| 3.0 + 0.7 * p[0] - 1.3 * p[1];
    for measure in [Measure::Planar, sphere()] {
        let opts = WeightOptions::new(1).with_measure(measure).with_moments(true);
        let w = exact_weights(&t, &src_window(), dw, dh, &opts);
        assert_eq!(w.moment.len(), w.len());
        let data = field(&src_window(), |r, c| cell_centroid(measure, r, c), f);
        let buf = buffer(&src_window(), &data);
        let second: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
        let first: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
        let inside = interior(&w, &src_window());
        let (mut n, mut worst, mut worst1) = (0, 0.0f64, 0.0f64);
        for r in 0..dh {
            for c in 0..dw {
                let k = r * dw + c;
                if !inside[k] {
                    continue;
                }
                let q: Vec<[f64; 2]> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                    .iter()
                    .map(|&(dx, dy)| t.at(c as f64 + dx, r as f64 + dy))
                    .collect();
                let (xs, ys): (Vec<f64>, Vec<f64>) = q.iter().map(|p| (p[0], p[1])).unzip();
                let want = f(centroid(measure, &xs, &ys));
                worst = worst.max((second[k] - want).abs());
                worst1 = worst1.max((first[k] - want).abs());
                n += 1;
            }
        }
        eprintln!("{measure:?}: {n} interior cells, second order off by {worst:.1e}, first {worst1:.1e}");
        assert!(n > 200, "{n}");
        assert!(worst < 1e-11, "second order off by {worst}");
        assert!(worst1 > 0.1, "first order should not be exact: {worst1}");
    }
}

#[test]
fn the_integral_is_conserved() {
    // a destination larger than the source window cuts its edge cells, and
    // a few cells have no data
    let t = Rotated { angle: 0.4, scale: 1.37, origin: (-6.0, -9.0) };
    let (dw, dh) = (40, 36);
    for measure in [Measure::Planar, sphere()] {
        let w = exact_weights(&t, &src_window(), dw, dh, &WeightOptions::new(2).with_measure(measure).with_moments(true));
        let mut data = field(&src_window(), |r, c| [c as f64 + 0.5, r as f64 + 0.5], |p| (p[0] / 4.0).sin() * (p[1] / 3.0).cos() + 2.0);
        for k in [17usize, 400, 401, 441, 900] {
            data[k] = -9999.0;
        }
        let buf = buffer(&src_window(), &data);
        let opts = ApplyOptions::new().with_src_nodata(Nodata::Value(-9999.0));
        let v: Vec<f64> = w.apply_second_order(&buf, Normalization::DestArea, &opts);
        let mut got = 0.0;
        for (k, &x) in v.iter().enumerate() {
            if x.is_finite() && x != -9999.0 && w.dst_area[k].is_finite() {
                got += x * w.dst_area[k];
            }
        }
        let mut want = 0.0;
        for k in 0..w.len() {
            let p = (w.src_row[k] as usize - src_window().row_off) * src_window().ncol + (w.src_col[k] as usize - src_window().col_off);
            if data[p] != -9999.0 {
                want += data[p] * w.frac[k] * w.src_area(k);
            }
        }
        assert!((got - want).abs() < 1e-12 * want.abs(), "{measure:?}: {got} vs {want}");
    }
}

#[test]
fn the_folded_table_is_the_same_method() {
    let t = Rotated { angle: 0.4, scale: 1.37, origin: (-6.0, -9.0) };
    let (dw, dh) = (40, 36);
    for measure in [Measure::Planar, sphere()] {
        let w = exact_weights(&t, &src_window(), dw, dh, &WeightOptions::new(2).with_measure(measure).with_moments(true));
        let mut data = field(&src_window(), |r, c| [c as f64 + 0.5, r as f64 + 0.5], |p| (p[0] / 4.0).sin() * (p[1] / 3.0).cos());
        for mask in [false, true] {
            let mut valid = vec![true; src_window().ncol * src_window().nrow];
            if mask {
                for k in [17usize, 400, 401, 441, 900] {
                    valid[k] = false;
                    data[k] = -9999.0;
                }
            }
            let folded = w.second_order(&src_window(), mask.then_some(&valid[..]), None);
            assert!(folded.moment.is_empty() && folded.frac.iter().any(|&f| f < 0.0));
            let buf = buffer(&src_window(), &data);
            let opts = ApplyOptions::new().with_src_nodata(Nodata::Value(-9999.0));
            for norm in [Normalization::DestArea, Normalization::FracArea] {
                let a: Vec<f64> = w.apply_second_order(&buf, norm, &opts);
                let b: Vec<f64> = folded.apply(&buf, norm, &opts);
                let worst = a.iter().zip(&b).map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() }).fold(0.0, f64::max);
                assert!(worst < 1e-12, "{measure:?} mask {mask} {norm:?}: {worst}");
            }
        }
    }
}

/// The mean of `sin(x / a) cos(y / b)` over a unit cell, in closed form.
fn sine_mean(a: f64, b: f64, x: f64, y: f64) -> f64 {
    a * ((x / a).cos() - ((x + 1.0) / a).cos()) * b * (((y + 1.0) / b).sin() - (y / b).sin())
}

/// A quadratic at length scale `s`: `X^2 + X Y / 2 - 0.8 Y^2` with `X = (x
/// - 20) / s`, `Y = (y - 15) / s`.
fn quadratic(s: f64, x: f64, y: f64) -> f64 {
    let (u, v) = ((x - 20.0) / s, (y - 15.0) / s);
    u * u + 0.5 * u * v - 0.8 * v * v
}

/// Its mean over a unit cell, in closed form.
fn quadratic_mean(s: f64, x: f64, y: f64) -> f64 {
    quadratic(s, x + 0.5, y + 0.5) + (1.0 - 0.8) / (12.0 * s * s)
}

/// The mean of `f` over destination cell `(r, c)`, a parallelogram under
/// `t`, by 12 x 12 Gauss-Legendre points (exact for a quadratic).
fn dst_mean(t: &Rotated, r: usize, c: usize, f: impl Fn(f64, f64) -> f64) -> f64 {
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
                    let p = t.at(c as f64 + 0.5 * (1.0 + u), r as f64 + 0.5 * (1.0 + v));
                    s += wu * wv * f(p[0], p[1]);
                }
            }
        }
    }
    s / 4.0
}

/// RMS errors of first and second order over the interior cells, for a
/// field given by its cell means and its pointwise values.
fn rms_errors(w: &Weights, t: &Rotated, mean: impl Fn(f64, f64) -> f64, f: impl Fn(f64, f64) -> f64) -> (f64, f64) {
    let src = src_window();
    let (dw, dh) = (w.dst_size[0], w.dst_size[1]);
    let inside = interior(w, &src);
    let data = field(&src, |r, c| [c as f64, r as f64], |p| mean(p[0], p[1]));
    let buf = buffer(&src, &data);
    let second: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let first: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
    let (mut e1, mut e2, mut n) = (0.0, 0.0, 0.0);
    for r in 0..dh {
        for c in 0..dw {
            let k = r * dw + c;
            if inside[k] {
                let want = dst_mean(t, r, c, &f);
                e1 += (first[k] - want).powi(2);
                e2 += (second[k] - want).powi(2);
                n += 1.0;
            }
        }
    }
    ((e1 / n).sqrt(), (e2 / n).sqrt())
}

#[test]
fn the_error_is_second_order() {
    let t = rotated();
    let w = exact_weights(&t, &src_window(), 30, 24, &WeightOptions::new(1).with_moments(true));
    // a quadratic's second-order error scales exactly as the square of
    // the cell size over its length scale
    let (_, a) = rms_errors(&w, &t, |x, y| quadratic_mean(10.0, x, y), |x, y| quadratic(10.0, x, y));
    let (_, b) = rms_errors(&w, &t, |x, y| quadratic_mean(20.0, x, y), |x, y| quadratic(20.0, x, y));
    eprintln!("quadratic: second-order rms {a:.3e} at length scale 10, {b:.3e} at 20");
    assert!(a > 1e-6 && (a / b - 4.0).abs() < 1e-6, "x{}", a / b);
    // and on a smooth field it is far below first order
    let (e1, e2) = rms_errors(&w, &t, |x, y| sine_mean(12.0, 8.0, x, y), |x, y| (x / 12.0).sin() * (y / 8.0).cos());
    eprintln!("sine: first-order rms {e1:.3e}, second {e2:.3e}");
    assert!(e2 < 0.05 * e1, "{e1} vs {e2}");
}

/// A global source of 360 x 180 pixels that wraps, and a destination grid
/// on it moved half a pixel west, so its first column straddles the seam,
/// with rows 0.7 of the source's.
struct Seam;

impl Transformer for Seam {
    fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            *x -= 0.5;
            *y = 0.7 * *y + 0.3;
        }
        vec![true; x.len()]
    }

    fn source_wrap(&self) -> Option<SourceWrap> {
        Some(SourceWrap::new(360.0, [0.0, 180.0]))
    }
}

#[test]
fn moments_survive_the_seam() {
    // a field linear in rows: across the seam, it comes out exactly
    let src = GridWindow::new(360, 180);
    let w = exact_weights(&Seam, &src, 360, 250, &WeightOptions::new(1).with_moments(true));
    let f = |y: f64| 4.0 + 0.25 * y;
    let data = field(&src, |r, c| [c as f64 + 0.5, r as f64 + 0.5], |p| f(p[1]));
    let buf = buffer(&src, &data);
    let v: Vec<f64> = w.apply_second_order(&buf, Normalization::FracArea, &ApplyOptions::new());
    let v1: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
    let mut worst1 = 0.0f64;
    for r in 3..245 {
        // the cell across the seam, and one beside it
        for c in [0, 1] {
            let want = f(0.7 * (r as f64 + 0.5) + 0.3);
            assert!((v[r * 360 + c] - want).abs() < 1e-12, "({r}, {c}): {} vs {want}", v[r * 360 + c]);
            worst1 = worst1.max((v1[r * 360 + c] - want).abs());
        }
    }
    assert!(worst1 > 1e-3, "first order should not be exact: {worst1}");
    // the seam cell takes from both ends of the source
    let cols: std::collections::BTreeSet<u32> =
        (0..w.len()).filter(|&k| w.dst_row[k] == 90 && w.dst_col[k] == 0).map(|k| w.src_col[k]).collect();
    assert_eq!(cols.into_iter().collect::<Vec<_>>(), vec![0, 359]);
}

#[test]
fn threads_and_tiles_keep_the_moments() {
    let t = rotated();
    let opts = WeightOptions::new(4).with_moments(true);
    let whole = exact_weights(&t, &src_window(), 30, 24, &opts);
    let par = exact_weights_par(|| Ok(&t), &src_window(), 30, 24, &opts, 4).unwrap();
    assert_eq!(whole, par);
    // row bands concatenate to the whole, moments and all
    let mut bands: Option<Weights> = None;
    for (r0, h) in [(0, 7), (7, 11), (18, 6)] {
        let b = exact_weights(&t, &src_window(), 30, h, &opts.with_dst_offset(0, r0));
        match &mut bands {
            None => bands = Some(b),
            Some(acc) => {
                acc.dst_row.extend(b.dst_row);
                acc.dst_col.extend(b.dst_col);
                acc.src_row.extend(b.src_row);
                acc.src_col.extend(b.src_col);
                acc.frac.extend(b.frac);
                acc.moment.extend(b.moment);
                acc.dst_area.extend(b.dst_area);
                acc.dst_size[1] += h;
            }
        }
    }
    let bands = bands.unwrap();
    assert_eq!(bands.moment, whole.moment);
    assert_eq!(bands.frac, whole.frac);
    // without moments the table is as it was
    let plain = exact_weights(&t, &src_window(), 30, 24, &WeightOptions::new(4));
    assert!(plain.moment.is_empty());
    assert_eq!(plain.frac, whole.frac);
}
