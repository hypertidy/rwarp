//! `exact_weights_par`: `exact_weights` on several threads, entry for entry
//! and bit for bit.
//!
//! An analytic transformer that swirls the destination over the source
//! (points far out fail), under the planar and the cos-latitude measure,
//! with destination and source windows at offsets; a mesh source with
//! world cells; then a CRS backend.

use rwarp::mesh::{CellMap, MeshTransformer, QuadMesh};
use rwarp::weights::{exact_weights, exact_weights_par, lonlat_measure, GridWindow, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

/// Destination pixels round `(50, 40)` to source pixels round `(60, 45)`,
/// turned by an angle that grows with the radius; points over 75
/// destination pixels out fail.
struct Swirl;

impl Transformer for Swirl {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| {
                let (u, v) = if dst_to_src { (*x - 50.0, *y - 40.0) } else { ((*x - 60.0) / 1.3, (*y - 45.0) / 1.3) };
                let r = u.hypot(v);
                if r > 75.0 {
                    (*x, *y) = (f64::INFINITY, f64::INFINITY);
                    return false;
                }
                let (sa, ca) = (0.35 + 0.004 * r).sin_cos();
                (*x, *y) = if dst_to_src {
                    (60.0 + 1.3 * (u * ca - v * sa), 45.0 + 1.3 * (u * sa + v * ca))
                } else {
                    (50.0 + u * ca + v * sa, 40.0 - u * sa + v * ca)
                };
                true
            })
            .collect()
    }
}

/// The same table bit for bit, `NaN` areas included.
fn same(a: &Weights, b: &Weights) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    (a.dst_row == b.dst_row && a.dst_col == b.dst_col && a.src_row == b.src_row && a.src_col == b.src_col)
        && bits(&a.frac) == bits(&b.frac)
        && bits(&a.dst_area) == bits(&b.dst_area)
        && bits(&a.src_cell_area) == bits(&b.src_cell_area)
        && (a.dst_off, a.dst_size, a.measure) == (b.dst_off, b.dst_size, b.measure)
}

/// Both measures, densify 1 and 3, a destination window at an offset and a
/// source window at one, on 0, 2, 3 and 7 threads.
#[test]
fn source_pixels_are_exact_weights() {
    let lonlat = lonlat_measure([-180.0, 1.0, 0.0, 85.0, 0.0, -1.0]).unwrap();
    for measure in [controlledburn::Measure::Planar, lonlat] {
        for densify in [1, 3] {
            for (src, off) in [(GridWindow::new(130, 100), (0, 0)), (GridWindow::new(90, 70).with_offset(20, 15), (6, 9))] {
                let opts = WeightOptions::new(densify).with_measure(measure).with_dst_offset(off.0, off.1);
                let one = exact_weights(&Swirl, &src, 130, 110, &opts);
                assert!(one.len() > 8000 && one.dst_area.iter().any(|a| a.is_nan()), "{}", one.len());
                for threads in [0, 2, 3, 7] {
                    let par = exact_weights_par(|| Ok(Swirl), &src, 130, 110, &opts, threads).unwrap();
                    assert!(same(&one, &par), "{measure:?}, densify {densify}, {off:?}, {threads} threads");
                }
                // a borrowed transformer serves every thread
                assert!(same(&one, &exact_weights_par(|| Ok(&Swirl), &src, 130, 110, &opts, 4).unwrap()));
            }
        }
    }
}

/// A mesh source, its cells in world coordinates: the world areas too.
#[test]
fn world_cells_are_exact_weights() {
    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }
    // a full annulus with a seam on the x axis
    let (ncol, nrow) = (48, 8);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let a = ((i % ncol) as f64 / ncol as f64) * std::f64::consts::TAU;
            let r = 100.0 + 12.0 * j as f64;
            x.push(r * a.cos());
            y.push(r * a.sin());
        }
    }
    let mesh = QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap();
    let gt = [-200.0, 5.0, 0.0, 200.0, 0.0, -5.0];
    let make = || MeshTransformer::with_backend(mesh.clone(), gt, Box::new(Same), Box::new(Same));
    let t = make().unwrap();
    let src = t.source_window();
    for densify in [1, 4] {
        let opts = WeightOptions::new(densify);
        let one = exact_weights(&t, &src, 80, 80, &opts);
        assert_eq!(one.src_cell_area.len(), one.len());
        for threads in [3, 8] {
            assert!(same(&one, &exact_weights_par(make, &src, 80, 80, &opts, threads).unwrap()), "{threads} threads");
        }
    }
}

/// `transformer()`'s error comes back, from the calling thread's or a
/// later one's.
#[test]
fn transformer_errors_come_back() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let src = GridWindow::new(130, 100);
    let opts = WeightOptions::new(1);
    let r = exact_weights_par::<Swirl>(|| Err("no transformer".into()), &src, 100, 80, &opts, 4);
    assert_eq!(r.unwrap_err(), "no transformer");
    let calls = AtomicUsize::new(0);
    let r = exact_weights_par(
        || if calls.fetch_add(1, Ordering::Relaxed) == 0 { Ok(Swirl) } else { Err("later".to_string()) },
        &src,
        100,
        80,
        &opts,
        4,
    );
    assert_eq!(r.unwrap_err(), "later");
}

/// World cells under another measure panic on the calling thread, as for
/// `exact_weights`.
#[test]
#[should_panic(expected = "Measure::Planar")]
fn world_cells_need_the_planar_measure() {
    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }
    let mesh = QuadMesh::from_corners(1, 1, vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0], CellMap::EqualArea).unwrap();
    let make = || MeshTransformer::with_backend(mesh.clone(), [0.0, 0.1, 0.0, 1.0, 0.0, -0.1], Box::new(Same), Box::new(Same));
    let opts = WeightOptions::new(1).with_measure(lonlat_measure([-180.0, 1.0, 0.0, 90.0, 0.0, -1.0]).unwrap());
    let _ = exact_weights_par(make, &GridWindow::new(1, 1), 10, 10, &opts, 4);
}

/// Through a CRS backend: a global longitude/latitude grid to Antarctic
/// polar stereographic, each thread with its own libproj (or proj4rs)
/// transformer, under the cos-latitude measure.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn through_a_crs_backend() {
    use rwarp::transform::GenImgProjTransformer;
    let src_gt = [-180.0, 1.0, 0.0, 90.0, 0.0, -1.0];
    let dst_gt = [-4e6, 50e3, 0.0, 4e6, 0.0, -50e3];
    let make = || {
        GenImgProjTransformer::new("+proj=longlat +datum=WGS84", src_gt, "+proj=stere +lat_0=-90 +lat_ts=-71 +datum=WGS84", dst_gt)
            .map(|t| t.with_source_size([360, 180]))
    };
    let opts = WeightOptions::new(2).with_measure(lonlat_measure(src_gt).unwrap());
    let src = GridWindow::new(360, 180);
    let one = exact_weights(&make().unwrap(), &src, 160, 160, &opts);
    assert!(one.len() > 160 * 160);
    assert!(same(&one, &exact_weights_par(make, &src, 160, 160, &opts, 6).unwrap()));
}
