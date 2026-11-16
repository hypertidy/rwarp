//! The exact footprint over a geolocation source, measured in the source's
//! world coordinates (stage 3 of `design/curvilinear-frontend.md`, #36):
//! each destination pixel's outline is walked onto the source window's
//! cells under the equal-area map, so weights are fractions of each
//! cell's world area.
//!
//! Oracles: each weight against an independent clip of the destination
//! pixel by the source cell; for a pixel inside the source, the sum of
//! weight times cell area equals the outline's area (generator 8 of the
//! design, which the source-pixel measure fails); an affine grid given as
//! geolocation arrays gives the affine path's weights; the same cells
//! given at another step, convention or window give the same weights.
//!
//! The CRS step is the identity (no backend needed) except in the last
//! test.

use std::collections::HashMap;

use rwarp::approx::ApproxTransformer;
use rwarp::geoloc_transform::{GenImgGeolocTransformer, GeolocArrays, GeolocConvention, GeolocOptions, PixelGeo};
use rwarp::mesh::{CellMap, QuadMesh};
use rwarp::warp::{warp, Footprint, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

struct Id;

impl CrsTransform for Id {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// The source raster: `W` by `H` pixels.
const W: usize = 40;
const H: usize = 30;

/// A polar sector: columns along the angle, rows along the radius. Its
/// cells are trapezoids of different areas.
fn sector(i: f64, j: f64) -> [f64; 2] {
    let a = (20.0 + 1.2 * i).to_radians();
    let r = 1000.0 + 40.0 * j;
    [r * a.cos(), r * a.sin()]
}

/// Whether a world point is inside the sector with a margin of 1.5 cells.
fn deep_in_sector(p: [f64; 2]) -> bool {
    let r = p[0].hypot(p[1]);
    let a = p[1].atan2(p[0]).to_degrees();
    (1000.0 + 40.0 * 1.5..1000.0 + 40.0 * (H as f64 - 1.5)).contains(&r)
        && (20.0 + 1.2 * 1.5..20.0 + 1.2 * (W as f64 - 1.5)).contains(&a)
}

/// Samples of `f` at array positions `(i, j)`, `nx` by `ny`.
fn arrays(f: impl Fn(f64, f64) -> [f64; 2], nx: usize, ny: usize, opts: GeolocOptions) -> GeolocArrays {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..ny {
        for i in 0..nx {
            let p = f(i as f64, j as f64);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    GeolocArrays::new(x, y, nx, ny, opts).unwrap()
}

/// The sector sampled at every raster pixel corner (top-left convention).
fn sector_arrays() -> GeolocArrays {
    arrays(sector, W + 1, H + 1, GeolocOptions::default())
}

/// 40 x 40 destination pixels of 35 world units, slightly rotated, over
/// the sector and beyond its edges.
const N: usize = 40;
const DST_GT: [f64; 6] = [600.0, 35.0, 4.0, 2050.0, 3.0, -35.0];

fn dst_corner(gt: [f64; 6], c: f64, r: f64) -> [f64; 2] {
    [gt[0] + c * gt[1] + r * gt[2], gt[3] + c * gt[4] + r * gt[5]]
}

/// Destination pixel `(r, c)` as a world polygon.
fn dst_pixel(r: usize, c: usize) -> Vec<[f64; 2]> {
    let (r, c) = (r as f64, c as f64);
    vec![dst_corner(DST_GT, c, r), dst_corner(DST_GT, c + 1.0, r), dst_corner(DST_GT, c + 1.0, r + 1.0), dst_corner(DST_GT, c, r + 1.0)]
}

fn geoloc(g: GeolocArrays) -> GenImgGeolocTransformer {
    geoloc_to(g, DST_GT)
}

fn geoloc_to(g: GeolocArrays, gt: [f64; 6]) -> GenImgGeolocTransformer {
    GenImgGeolocTransformer::with_backend(PixelGeo::Geoloc(g), PixelGeo::affine(gt).unwrap(), Box::new(Id), Box::new(Id))
}

/// A transformer without [`Transformer::source_cells`]: the exact
/// footprint in source pixel space, as before.
struct PixelsOnly<'a, T>(&'a T);

impl<T: Transformer> Transformer for PixelsOnly<'_, T> {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        self.0.transform(dst_to_src, x, y)
    }
}

fn area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    0.5 * (0..n).map(|k| p[k][0] * p[(k + 1) % n][1] - p[(k + 1) % n][0] * p[k][1]).sum::<f64>()
}

/// Sutherland-Hodgman: `subject` clipped by the convex polygon `by`.
fn clip(subject: &[[f64; 2]], by: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut by = by.to_vec();
    if area(&by) < 0.0 {
        by.reverse();
    }
    let side = |a: [f64; 2], b: [f64; 2], p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let mut out = subject.to_vec();
    for k in 0..by.len() {
        let (a, b) = (by[k], by[(k + 1) % by.len()]);
        let input = std::mem::take(&mut out);
        for m in 0..input.len() {
            let (p, q) = (input[m], input[(m + 1) % input.len()]);
            let (sp, sq) = (side(a, b, p), side(a, b, q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

type Table = HashMap<(u32, u32, u32, u32), f64>;

fn table(w: &Weights) -> Table {
    let mut m = HashMap::new();
    for k in 0..w.len() {
        let key = (w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]);
        assert!(m.insert(key, w.frac[k]).is_none(), "{key:?} twice");
    }
    m
}

fn max_diff(a: &Table, b: &Table) -> f64 {
    let mut worst: f64 = 0.0;
    for (k, x) in a {
        worst = worst.max((x - b.get(k).copied().unwrap_or(0.0)).abs());
    }
    for (k, y) in b {
        if !a.contains_key(k) {
            worst = worst.max(y.abs());
        }
    }
    worst
}

/// Per destination cell of the window, the sum of weight times source
/// cell area.
fn covered(w: &Weights) -> Vec<f64> {
    let mut s = vec![0.0; w.dst_size[0] * w.dst_size[1]];
    for k in 0..w.len() {
        let (r, c) = (w.dst_row[k] as usize - w.dst_off[1], w.dst_col[k] as usize - w.dst_off[0]);
        s[r * w.dst_size[0] + c] += w.frac[k] * w.src_area(k);
    }
    s
}

/// The same weights, bit for bit (`NaN` areas included).
fn same(a: &Weights, b: &Weights) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    (&a.dst_row, &a.dst_col, &a.src_row, &a.src_col) == (&b.dst_row, &b.dst_col, &b.src_row, &b.src_col)
        && bits(&a.frac) == bits(&b.frac)
        && bits(&a.dst_area) == bits(&b.dst_area)
        && bits(&a.src_cell_area) == bits(&b.src_cell_area)
        && (a.dst_off, a.dst_size, a.measure) == (b.dst_off, b.dst_size, b.measure)
}

#[test]
fn every_weight_is_the_clipped_area_over_the_cell_area() {
    let t = geoloc(sector_arrays());
    assert!(t.source_cells().is_some());
    let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &WeightOptions::new(2));
    assert_eq!(w.src_cell_area.len(), w.len());
    let got = table(&w);
    let mut worst: f64 = 0.0;
    let mut pairs = 0;
    for r in 0..N {
        for c in 0..N {
            let p = dst_pixel(r, c);
            for sr in 0..H {
                for sc in 0..W {
                    let (i, j) = (sc as f64, sr as f64);
                    let q = [sector(i, j), sector(i + 1.0, j), sector(i + 1.0, j + 1.0), sector(i, j + 1.0)];
                    let f = area(&clip(&p, &q)).abs() / area(&q).abs();
                    let g = got.get(&(r as u32, c as u32, sr as u32, sc as u32)).copied().unwrap_or(0.0);
                    if f > 0.0 || g > 0.0 {
                        pairs += 1;
                    }
                    worst = worst.max((f - g).abs());
                }
            }
        }
    }
    eprintln!("{} entries, {pairs} overlapping pairs by clipping, max |weight - clip| {worst:.1e}", w.len());
    assert!(worst < 1e-12, "{worst}");
    // the cell areas are the cells' world areas
    for k in 0..w.len() {
        let (i, j) = (w.src_col[k] as f64, w.src_row[k] as f64);
        let q = [sector(i, j), sector(i + 1.0, j), sector(i + 1.0, j + 1.0), sector(i, j + 1.0)];
        assert!((w.src_area(k) - area(&q).abs()).abs() < 1e-9 * w.src_area(k));
    }
}

#[test]
fn weights_times_cell_areas_sum_to_the_outline_area() {
    // generator 8: for a pixel inside the source, and the source-pixel
    // measure fails it
    let t = geoloc(sector_arrays());
    let opts = WeightOptions::new(2);
    let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &opts);
    let mesh = t.source_cells().unwrap().mesh(0, 0, W, H).unwrap();
    let old = exact_weights(&PixelsOnly(&t), &GridWindow::new(W, H), N, N, &opts);
    assert!(old.src_cell_area.is_empty());
    let (sum, mut old_sum) = (covered(&w), vec![0.0; N * N]);
    for k in 0..old.len() {
        old_sum[old.dst_row[k] as usize * N + old.dst_col[k] as usize] +=
            old.frac[k] * mesh.cell_area(old.src_row[k] as usize, old.src_col[k] as usize);
    }
    let (mut inside, mut worst, mut worst_old) = (0, 0.0f64, 0.0f64);
    for r in 0..N {
        for c in 0..N {
            let p = dst_pixel(r, c);
            let a = area(&p).abs();
            let k = r * N + c;
            assert!((w.dst_area[k] - a).abs() < 1e-12 * a, "({r}, {c}): outline area {} vs {a}", w.dst_area[k]);
            if p.iter().all(|&q| deep_in_sector(q)) {
                inside += 1;
                worst = worst.max((sum[k] - a).abs() / a);
                worst_old = worst_old.max((old_sum[k] - a).abs() / a);
            }
        }
    }
    eprintln!("{inside} pixels inside: max relative error {worst:.1e} in world cells, {worst_old:.1e} in source pixels");
    assert!(inside > 300);
    assert!(worst < 1e-12, "{worst}");
    assert!(worst_old > 1e-4, "{worst_old}");
}

#[test]
fn source_cells_inside_the_destination_are_shared_out_whole() {
    let t = geoloc(sector_arrays());
    let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &WeightOptions::new(2));
    let mut sum = vec![0.0; W * H];
    for k in 0..w.len() {
        sum[w.src_row[k] as usize * W + w.src_col[k] as usize] += w.frac[k];
    }
    // destination pixel coordinates of a world point
    let det = DST_GT[1] * DST_GT[5] - DST_GT[2] * DST_GT[4];
    let to_dst = |p: [f64; 2]| {
        let (dx, dy) = (p[0] - DST_GT[0], p[1] - DST_GT[3]);
        [(DST_GT[5] * dx - DST_GT[2] * dy) / det, (-DST_GT[4] * dx + DST_GT[1] * dy) / det]
    };
    let mut whole = 0;
    let mut worst: f64 = 0.0;
    for sr in 0..H {
        for sc in 0..W {
            let (i, j) = (sc as f64, sr as f64);
            let q = [sector(i, j), sector(i + 1.0, j), sector(i + 1.0, j + 1.0), sector(i, j + 1.0)];
            if q.iter().map(|&p| to_dst(p)).all(|d| d.iter().all(|&v| v > 0.5 && v < N as f64 - 0.5)) {
                whole += 1;
                worst = worst.max((sum[sr * W + sc] - 1.0).abs());
            }
        }
    }
    eprintln!("{whole} source cells inside the destination: max |sum - 1| {worst:.1e}");
    assert!(whole > 300);
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn an_affine_grid_as_geolocation_arrays_gives_the_affine_weights() {
    // a rotated, sheared grid: its cells are parallelograms, for which the
    // equal-area map is the affine map
    let gt = [700.0, 30.0, 7.0, 2000.0, -5.0, -40.0];
    let affine = |i: f64, j: f64| [gt[0] + i * gt[1] + j * gt[2], gt[3] + i * gt[4] + j * gt[5]];
    let det = (gt[1] * gt[5] - gt[2] * gt[4]).abs();
    let t = geoloc(arrays(affine, W + 1, H + 1, GeolocOptions::default()));
    assert!(t.source_cells().is_some());
    let a = GenImgGeolocTransformer::with_backend(PixelGeo::affine(gt).unwrap(), PixelGeo::affine(DST_GT).unwrap(), Box::new(Id), Box::new(Id));
    assert!(a.source_cells().is_none());
    for densify in [1, 3] {
        let opts = WeightOptions::new(densify);
        let wg = exact_weights(&t, &GridWindow::new(W, H), N, N, &opts);
        let wa = exact_weights(&a, &GridWindow::new(W, H), N, N, &opts);
        let d = max_diff(&table(&wg), &table(&wa));
        eprintln!("densify {densify}: {} and {} entries, max difference {d:.1e}", wg.len(), wa.len());
        assert!(d < 1e-12, "{d}");
        for k in 0..wg.len() {
            assert!((wg.src_area(k) - det).abs() < 1e-9 * det);
        }
        for (x, y) in wg.dst_area.iter().zip(&wa.dst_area) {
            assert!((x - y * det).abs() < 1e-9 * x, "{x} vs {}", y * det);
        }
    }
}

#[test]
fn subsampled_arrays_give_the_weights_of_their_full_resolution_cells() {
    // the sector at every second pixel corner, then the same surface at
    // every corner: the step-2 cells are the step-2 bilinear model's,
    // straight-sided within each square of samples
    let half = |i: f64, j: f64| sector(2.0 * i, 2.0 * j);
    let g2 = arrays(half, W / 2 + 1, H / 2 + 1, GeolocOptions::default().with_pixel(0.0, 2.0).with_line(0.0, 2.0));
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=H {
        for i in 0..=W {
            let (a, b) = g2.pixel_to_geo(i as f64, j as f64).unwrap();
            x.push(a);
            y.push(b);
        }
    }
    let g1 = GeolocArrays::new(x, y, W + 1, H + 1, GeolocOptions::default()).unwrap();
    let opts = WeightOptions::new(2);
    let w2 = exact_weights(&geoloc(g2), &GridWindow::new(W, H), N, N, &opts);
    let w1 = exact_weights(&geoloc(g1), &GridWindow::new(W, H), N, N, &opts);
    let d = max_diff(&table(&w2), &table(&w1));
    eprintln!("step 2 vs step 1: {} entries, max difference {d:.1e}", w2.len());
    assert!(d < 1e-12, "{d}");
}

#[test]
fn centre_samples_make_the_cells_from_centres_makes() {
    // samples at pixel centres; the cells' corners are the means of four
    // centres and extrapolate half a cell beyond the outermost ones
    let centre = |i: f64, j: f64| sector(i + 0.5, j + 0.5);
    let g = arrays(centre, W, H, GeolocOptions::default().with_convention(GeolocConvention::PixelCenter));
    let mesh = g.window_mesh(0, 0, W, H).unwrap();
    let (mut cx, mut cy) = (Vec::new(), Vec::new());
    for j in 0..H {
        for i in 0..W {
            let p = centre(i as f64, j as f64);
            cx.push(p[0]);
            cy.push(p[1]);
        }
    }
    let want = QuadMesh::from_centres(W, H, &cx, &cy, CellMap::EqualArea).unwrap();
    let mut worst: f64 = 0.0;
    for r in 0..H {
        for c in 0..W {
            for (a, b) in mesh.corners(r, c).iter().zip(want.corners(r, c)) {
                worst = worst.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs());
            }
        }
    }
    assert!(worst < 1e-9, "{worst}");
    // and the area invariant holds for it
    let t = geoloc(g);
    let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &WeightOptions::new(2));
    let sum = covered(&w);
    let mut inside = 0;
    for r in 0..N {
        for c in 0..N {
            let p = dst_pixel(r, c);
            if p.iter().all(|&q| deep_in_sector(q)) {
                inside += 1;
                let a = area(&p).abs();
                assert!((sum[r * N + c] - a).abs() < 1e-12 * a, "({r}, {c}): {} vs {a}", sum[r * N + c]);
            }
        }
    }
    assert!(inside > 300);
}

#[test]
fn source_windows_split_the_weights_exactly() {
    let t = geoloc(sector_arrays());
    let opts = WeightOptions::new(2);
    let whole = exact_weights(&t, &GridWindow::new(W, H), N, N, &opts);
    let mut parts: Table = HashMap::new();
    for (col_off, row_off, ncol, nrow) in [(0, 0, W, 13), (0, 13, 25, H - 13), (25, 13, W - 25, H - 13)] {
        let w = exact_weights(&t, &GridWindow::new(ncol, nrow).with_offset(col_off, row_off), N, N, &opts);
        assert_eq!(w.dst_area, whole.dst_area, "the outline's area does not depend on the window");
        for (k, v) in table(&w) {
            let (sr, sc) = (k.2 as usize, k.3 as usize);
            assert!((row_off..row_off + nrow).contains(&sr) && (col_off..col_off + ncol).contains(&sc));
            assert!(parts.insert(k, v).is_none());
        }
    }
    let d = max_diff(&table(&whole), &parts);
    eprintln!("windows vs whole: max difference {d:.1e}");
    assert!(d < 1e-12, "{d}");
}

#[test]
fn arrays_without_whole_cells_fall_back_to_source_pixels() {
    let opts = WeightOptions::new(2);
    let check = |g: GeolocArrays, why: &str| {
        let reason = g.world_cells().expect_err("no world cells").to_string();
        assert!(reason.contains(why), "{reason}");
        assert!(g.window_mesh(0, 0, W, H).is_err());
        let t = geoloc(g);
        assert!(t.source_cells().is_none());
        let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &opts);
        let old = exact_weights(&PixelsOnly(&t), &GridWindow::new(W, H), N, N, &opts);
        assert!(w.src_cell_area.is_empty());
        assert!(same(&w, &old), "{why}: not the source-pixel weights");
    };
    // a missing sample (NaN, or equal to the X nodata value)
    let mut x: Vec<f64> = (0..=H).flat_map(|j| (0..=W).map(move |i| sector(i as f64, j as f64)[0])).collect();
    let y: Vec<f64> = (0..=H).flat_map(|j| (0..=W).map(move |i| sector(i as f64, j as f64)[1])).collect();
    x[9 * (W + 1) + 7] = f64::NAN;
    check(GeolocArrays::new(x.clone(), y.clone(), W + 1, H + 1, GeolocOptions::default()).unwrap(), "missing");
    x[9 * (W + 1) + 7] = -999.0;
    check(GeolocArrays::new(x, y, W + 1, H + 1, GeolocOptions::default().with_nodata_x(Some(-999.0))).unwrap(), "missing");
    // a fold: a column of samples moved back over its neighbour
    let fold = |i: f64, j: f64| if i == 12.0 { sector(10.5, j) } else { sector(i, j) };
    check(arrays(fold, W + 1, H + 1, GeolocOptions::default()), "both orientations");
    // longitudes across the antimeridian
    let lonlat = |i: f64, j: f64| {
        let lon = 170.0 + 0.5 * i;
        [if lon > 180.0 { lon - 360.0 } else { lon }, -60.0 + 0.5 * j]
    };
    check(arrays(lonlat, W + 1, H + 1, GeolocOptions::default().with_lon_wrap(true)), "antimeridian");
}

#[test]
fn warp_reduces_the_same_weights_by_area() {
    let t = geoloc(sector_arrays());
    let opts = WarpOptions::new(ResampleAlg::Average)
        .with_footprint(Footprint::Exact { densify: 2 })
        .with_src_nodata(Nodata::None);
    // a field that varies by cell, and a constant one
    let mut data = Vec::with_capacity(W * H * 2);
    for r in 0..H {
        for c in 0..W {
            data.push(((r * 7 + c * 13) % 17) as f64 + 0.25 * c as f64);
            data.push(42.0);
        }
    }
    let buf = SourceBuffer { data: &data, ncol: W, nrow: H, nbands: 2, col_off: 0, row_off: 0 };
    let out: Vec<f64> = warp(&t, &buf, N, N, &opts);
    let w = exact_weights(&t, &GridWindow::new(W, H), N, N, &WeightOptions::warp_compatible(2));
    let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::new());
    let (mut set, mut same) = (0, 0);
    for k in 0..N * N {
        let (a, b) = (out[2 * k], applied[2 * k]);
        if a.is_finite() && a != 0.0 {
            set += 1;
            assert!((out[2 * k + 1] - 42.0).abs() < 1e-12, "constant field gives {}", out[2 * k + 1]);
        }
        if a.to_bits() == b.to_bits() && out[2 * k + 1].to_bits() == applied[2 * k + 1].to_bits() {
            same += 1;
        }
    }
    eprintln!("{set} destination pixels with values; {same} of {} the same bits as Weights::apply", N * N);
    assert!(set > 400);
    assert_eq!(same, N * N, "warp() and Weights::apply differ");
    // weighting by area, not by fraction: the area-weighted mean of the
    // first band over a pixel's entries
    let mut num = vec![0.0; N * N];
    let mut den = vec![0.0; N * N];
    for k in 0..w.len() {
        let d = w.dst_row[k] as usize * N + w.dst_col[k] as usize;
        let s = w.src_row[k] as usize * W + w.src_col[k] as usize;
        num[d] += w.frac[k] * w.src_area(k) * data[2 * s];
        den[d] += w.frac[k] * w.src_area(k);
    }
    for k in 0..N * N {
        if den[k] > 0.0 {
            assert!((out[2 * k] - num[k] / den[k]).abs() < 1e-9, "pixel {k}: {} vs {}", out[2 * k], num[k] / den[k]);
        }
    }
}

#[test]
fn the_approximate_transformer_keeps_the_world_cells() {
    let t = geoloc(sector_arrays());
    let opts = WeightOptions::new(2);
    let exact = exact_weights(&t, &GridWindow::new(W, H), N, N, &opts);
    let approx = ApproxTransformer::new(geoloc(sector_arrays()), 0.125);
    assert!(approx.source_cells().is_some());
    assert!(same(&exact_weights(&approx, &GridWindow::new(W, H), N, N, &opts), &exact));
}

#[test]
fn geolocation_arrays_alone_are_measured_in_world_cells() {
    // as a transformer on their own, destination pixels are world
    // coordinates: unit squares, here 30 by 30 of them deep in the sector
    let g = sector_arrays();
    assert!(g.source_cells().is_some());
    let opts = WeightOptions::new(1).with_dst_offset(1200, 1200);
    let w = exact_weights(&g, &GridWindow::new(W, H), 30, 30, &opts);
    let sum = covered(&w);
    for (k, s) in sum.iter().enumerate() {
        assert!((s - 1.0).abs() < 1e-12 && (w.dst_area[k] - 1.0).abs() < 1e-12, "cell {k}: {s}");
    }
}

#[test]
#[should_panic(expected = "Measure::Planar")]
fn a_spherical_measure_is_refused_for_world_cells() {
    let t = geoloc(sector_arrays());
    let m = rwarp::weights::lonlat_measure([-180.0, 1.0, 0.0, 90.0, 0.0, -1.0]).unwrap();
    exact_weights(&t, &GridWindow::new(W, H), N, N, &WeightOptions::new(1).with_measure(m));
}

/// A curved longitude/latitude grid to a south polar stereographic grid:
/// destination pixel edges are curves in the source's coordinates.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn a_curvilinear_lonlat_source_to_a_polar_grid() {
    let lonlat = |i: f64, j: f64| [-80.0 + 1.0 * i + 0.2 * j, -78.0 + 0.4 * j + 2.0 * (i / 9.0).sin()];
    let g = arrays(lonlat, W + 1, H + 1, GeolocOptions::default());
    let gt = [-1.3e6, 25e3, 0.0, 1.7e6, 0.0, -25e3];
    let t = GenImgGeolocTransformer::new(PixelGeo::Geoloc(g), "EPSG:4326", PixelGeo::affine(gt).unwrap(), "EPSG:3031").unwrap();
    let mesh = t.source_cells().unwrap().mesh(0, 0, W, H).unwrap();
    let (nc, nr) = (60, 50);
    for densify in [1, 8] {
        let opts = WeightOptions::new(densify);
        let w = exact_weights(&t, &GridWindow::new(W, H), nc, nr, &opts);
        let old = exact_weights(&PixelsOnly(&t), &GridWindow::new(W, H), nc, nr, &opts);
        let sum = covered(&w);
        let mut old_sum = vec![0.0; nc * nr];
        for k in 0..old.len() {
            old_sum[old.dst_row[k] as usize * nc + old.dst_col[k] as usize] +=
                old.frac[k] * mesh.cell_area(old.src_row[k] as usize, old.src_col[k] as usize);
        }
        // pixels whose cells, and their neighbours, are all in the source:
        // every entry away from the source's edge, and the weights sum to
        // more than a cell
        let mut deep = vec![true; nc * nr];
        let mut any = vec![false; nc * nr];
        for k in 0..w.len() {
            let d = w.dst_row[k] as usize * nc + w.dst_col[k] as usize;
            any[d] = true;
            let (r, c) = (w.src_row[k] as usize, w.src_col[k] as usize);
            if r == 0 || c == 0 || r + 1 >= H || c + 1 >= W {
                deep[d] = false;
            }
        }
        let (mut inside, mut worst, mut worst_old) = (0, 0.0f64, 0.0f64);
        for k in 0..nc * nr {
            if any[k] && deep[k] {
                inside += 1;
                let a = w.dst_area[k];
                worst = worst.max((sum[k] - a).abs() / a);
                worst_old = worst_old.max((old_sum[k] - a).abs() / a);
            }
        }
        eprintln!("densify {densify}: {inside} pixels inside: max relative error {worst:.1e} in world cells, {worst_old:.1e} in source pixels");
        assert!(inside > 200);
        assert!(worst < 1e-11, "{worst}");
        assert!(worst_old > 1e-5, "{worst_old}");
    }
}
