//! `warp_to_par`: `warp_to` on several threads, bit for bit.
//!
//! An analytic transformer that swirls the destination over the source
//! (rotation growing with radius, scale, points that fail far out) takes
//! every method, both footprints, kernel widening, nodata, a cutline and
//! a destination offset; then a CRS backend through the approximate
//! transformer, as gdalwarp warps.

use std::sync::atomic::{AtomicUsize, Ordering};

use rwarp::cutline::{Cutline, CutlineRule};
use rwarp::warp::{
    warp_par, warp_to, warp_to_par, DstNodata, Footprint, KernelScaling, Nodata, ResampleAlg,
    SourceBuffer, WarpOptions,
};
use rwarp::Transformer;

/// Destination pixels round `(50, 40)` to source pixels round `(60, 45)`,
/// scaled by `s` and turned by an angle that grows with the radius; points
/// over 75 destination pixels out fail.
struct Swirl {
    s: f64,
}

impl Transformer for Swirl {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| {
                let (u, v, r) = if dst_to_src {
                    let (u, v) = (*x - 50.0, *y - 40.0);
                    (u, v, u.hypot(v))
                } else {
                    let (u, v) = ((*x - 60.0) / self.s, (*y - 45.0) / self.s);
                    (u, v, u.hypot(v))
                };
                if r > 75.0 {
                    (*x, *y) = (f64::INFINITY, f64::INFINITY);
                    return false;
                }
                let a = 0.35 + 0.004 * r;
                let (sa, ca) = a.sin_cos();
                if dst_to_src {
                    *x = 60.0 + self.s * (u * ca - v * sa);
                    *y = 45.0 + self.s * (u * sa + v * ca);
                } else {
                    *x = 50.0 + u * ca + v * sa;
                    *y = 40.0 - u * sa + v * ca;
                }
                true
            })
            .collect()
    }
}

const ALGS: [ResampleAlg; 13] = [
    ResampleAlg::NearestNeighbour,
    ResampleAlg::Bilinear,
    ResampleAlg::Cubic,
    ResampleAlg::CubicSpline,
    ResampleAlg::Lanczos,
    ResampleAlg::Average,
    ResampleAlg::Mode,
    ResampleAlg::Min,
    ResampleAlg::Max,
    ResampleAlg::Med,
    ResampleAlg::Q1,
    ResampleAlg::Q3,
    ResampleAlg::Rms,
];

/// Two bands over 130 x 100 source pixels, a few nodata.
fn source() -> Vec<f64> {
    (0..130 * 100)
        .flat_map(|p| {
            let (c, r) = ((p % 130) as f64, (p / 130) as f64);
            let a = 10.0 + (c * 0.17).sin() * (r * 0.11).cos() + ((p * 7919) % 13) as f64;
            let b = if p % 97 == 0 { -9999.0 } else { c * 0.5 - r * 0.25 };
            [a, b]
        })
        .collect()
}

fn same(a: &[f64], b: &[f64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// Every method, both footprints, point and widened kernels, on 0, 2, 3
/// and 7 threads: `warp_to`'s values bit for bit.
#[test]
fn every_method_is_warp_to() {
    let data = source();
    let src = SourceBuffer { data: &data, ncol: 130, nrow: 100, nbands: 2, col_off: 0, row_off: 0 };
    let (dw, dh) = (100, 80);
    for s in [0.8, 1.7] {
        let t = Swirl { s };
        for alg in ALGS {
            for footprint in [Footprint::GdalBox, Footprint::Exact { densify: 2 }] {
                if !alg.is_area() && footprint != Footprint::GdalBox {
                    continue;
                }
                for scaling in [KernelScaling::Point, KernelScaling::WidenWhenDownsampling] {
                    let opts = WarpOptions::new(alg)
                        .with_footprint(footprint)
                        .with_scaling(scaling)
                        .with_src_nodata(Nodata::Value(-9999.0))
                        .with_dst_nodata(DstNodata::Value(-1.0));
                    let one = warp_to::<f64, f64>(&t, &src, dw, dh, &opts);
                    let computed = one.iter().filter(|&&v| v != -1.0).count();
                    assert!(computed > dw * dh, "{alg:?}: {computed}");
                    for threads in [0, 2, 3, 7] {
                        let par = warp_to_par::<f64, f64, _>(|| Ok(Swirl { s }), &src, dw, dh, &opts, threads).unwrap();
                        assert_eq!(same(&one, &par), 0, "{alg:?}, {footprint:?}, {scaling:?}, scale {s}, {threads} threads");
                    }
                }
            }
        }
    }
}

/// A cutline with a blended edge, a destination window at an offset, a
/// source window at an offset: the same again; a borrowed transformer
/// serves every thread.
#[test]
fn cutlines_and_windows_are_warp_to() {
    let data = source();
    let ring = vec![[20.0, 10.0], [110.0, 25.0], [95.0, 90.0], [30.0, 70.0], [20.0, 10.0]];
    let t = Swirl { s: 1.2 };
    for rule in [CutlineRule::Centre, CutlineRule::Exact] {
        let mut cut = Cutline::new(vec![vec![ring.clone()]]);
        cut.rule = rule;
        cut.blend_dist = if rule == CutlineRule::Centre { 6.0 } else { 0.0 };
        for alg in [ResampleAlg::Bilinear, ResampleAlg::Average, ResampleAlg::Mode] {
            let opts = WarpOptions::new(alg).with_cutline(cut.clone()).with_dst_offset(7, 11);
            // the source's rows 10 to 90, columns 5 to 125
            let sub: Vec<f64> = (10..90)
                .flat_map(|r| (5..125).flat_map(move |c| [r * 130 + c, r * 130 + c]))
                .enumerate()
                .map(|(i, p)| data[p * 2 + i % 2])
                .collect();
            for src in [
                SourceBuffer { data: &data, ncol: 130, nrow: 100, nbands: 2, col_off: 0, row_off: 0 },
                SourceBuffer { data: &sub, ncol: 120, nrow: 80, nbands: 2, col_off: 5, row_off: 10 },
            ] {
                let one = warp_to::<f64, f64>(&t, &src, 70, 55, &opts);
                for threads in [2, 5] {
                    let par = warp_par(|| Ok(&t), &src, 70, 55, &opts, threads).unwrap();
                    assert_eq!(same(&one, &par), 0, "{rule:?}, {alg:?}, {threads} threads");
                }
            }
        }
    }
}

/// `transformer()` is called once on each thread that works, the calling
/// thread's first, and not again on one thread; its error comes back.
#[test]
fn a_transformer_a_thread() {
    let data = source();
    let src = SourceBuffer { data: &data, ncol: 130, nrow: 100, nbands: 2, col_off: 0, row_off: 0 };
    let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify: 2 });
    let calls = AtomicUsize::new(0);
    let make = || {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(Swirl { s: 1.0 })
    };
    for (threads, most) in [(1, 1), (4, 4), (64, 64)] {
        calls.store(0, Ordering::Relaxed);
        warp_to_par::<f64, f64, _>(make, &src, 100, 80, &opts, threads).unwrap();
        let n = calls.load(Ordering::Relaxed);
        assert!((1..=most).contains(&n), "{n} calls on {threads} threads");
    }
    // the calling thread's fails: nothing is warped
    let r = warp_to_par::<f64, f64, Swirl>(|| Err("no transformer".into()), &src, 100, 80, &opts, 4);
    assert_eq!(r.unwrap_err(), "no transformer");
    // another thread's fails: the error, once the others have finished
    let first = AtomicUsize::new(0);
    let r = warp_to_par::<f64, f64, Swirl>(
        || if first.fetch_add(1, Ordering::Relaxed) == 0 { Ok(Swirl { s: 1.0 }) } else { Err("later".into()) },
        &src,
        100,
        80,
        &opts,
        4,
    );
    assert_eq!(r.unwrap_err(), "later");
}

/// With a CRS backend through the approximate transformer, as gdalwarp
/// warps: a global longitude/latitude grid to Antarctic polar
/// stereographic, each thread with its own libproj (or proj4rs)
/// transformer.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn through_a_crs_backend() {
    use rwarp::approx::ApproxTransformer;
    use rwarp::transform::GenImgProjTransformer;
    let (sw, sh) = (360, 180);
    let data: Vec<f64> = (0..sw * sh).map(|p| ((p % sw) as f64 * 0.05).sin() * 10.0 + (p / sw) as f64).collect();
    let src = SourceBuffer { data: &data, ncol: sw, nrow: sh, nbands: 1, col_off: 0, row_off: 0 };
    let src_gt = [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0];
    let dst_gt = [-4e6, 25e3, 0.0, 4e6, 0.0, -25e3];
    let make = || {
        GenImgProjTransformer::new(
            "+proj=longlat +datum=WGS84",
            src_gt,
            "+proj=stere +lat_0=-90 +lat_ts=-71 +datum=WGS84",
            dst_gt,
        )
        .map(|t| ApproxTransformer::new(t.with_source_size([sw, sh]), 0.125))
    };
    let t = make().unwrap();
    for alg in [ResampleAlg::Bilinear, ResampleAlg::Average, ResampleAlg::Mode] {
        let opts = WarpOptions::new(alg).with_scaling(KernelScaling::WidenWhenDownsampling);
        let one = warp_to::<f64, f64>(&t, &src, 320, 320, &opts);
        let par = warp_to_par::<f64, f64, _>(make, &src, 320, 320, &opts, 6).unwrap();
        assert_eq!(same(&one, &par), 0, "{alg:?}");
        assert!(one.iter().filter(|&&v| v != 0.0).count() > 320 * 320 / 2);
    }
}
