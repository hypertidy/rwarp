//! Whole-raster, multi-chunk parity with gdalwarp, and the exact retry.
//!
//! The fixtures in `tests/fixtures/chunks/` are gdalwarp output with a
//! small warp memory limit, written by `gen.py` there (which records the
//! GDAL version), with the chunks GDAL warped taken from its debug log.
//! Each case is planned with [`plan_chunks`] and warped with
//! [`warp_chunks`], through the approximate transformer gdalwarp uses.

use std::path::PathBuf;

use rwarp::plan::{plan_chunks, warp_chunks, ChunkOptions, ScaleOrigin, WarpPlan};
use rwarp::warp::{warp, DataType, KernelScaling, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::Transformer;

/// The source field `gen.py` writes.
fn value(c: usize, r: usize) -> f64 {
    ((c * 7919 + r * 104729) % 1000) as f64 / 10.0 + c as f64 * 0.5 - r as f64 * 0.25
}

#[allow(dead_code)]
struct Case {
    name: String,
    alg: ResampleAlg,
    src_gt: [f64; 6],
    src_size: [usize; 2],
    dst_crs: String,
    dst_gt: [f64; 6],
    dst_size: [usize; 2],
    wm: f64,
    nodata: bool,
    block: Option<[usize; 2]>,
    special: Vec<[f64; 2]>,
    /// src off, src size, dst off, dst size, GDAL's scale
    chunks: Vec<([usize; 2], [usize; 2], [usize; 2], [usize; 2], [f64; 2])>,
    vals: Vec<f64>,
}

fn nums<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

#[allow(dead_code)]
fn read_case(name: &str) -> Case {
    let path: PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "chunks", &format!("{name}.txt")].iter().collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut c = Case {
        name: name.into(),
        alg: ResampleAlg::NearestNeighbour,
        src_gt: [0.0; 6],
        src_size: [0; 2],
        dst_crs: String::new(),
        dst_gt: [0.0; 6],
        dst_size: [0; 2],
        wm: 0.0,
        nodata: false,
        block: None,
        special: vec![],
        chunks: vec![],
        vals: vec![],
    };
    let mut lines = text.lines();
    for l in lines.by_ref() {
        let Some(h) = l.strip_prefix("# ") else {
            let d: Vec<usize> = nums(l);
            c.dst_size = [d[0], d[1]];
            break;
        };
        let (k, v) = h.split_once(' ').unwrap();
        match k {
            "alg" => {
                c.alg = match v {
                    "near" => ResampleAlg::NearestNeighbour,
                    "bilinear" => ResampleAlg::Bilinear,
                    "cubic" => ResampleAlg::Cubic,
                    "lanczos" => ResampleAlg::Lanczos,
                    _ => panic!("alg {v}"),
                }
            }
            "src_gt" => c.src_gt = nums::<f64>(v).try_into().unwrap(),
            "src_size" => c.src_size = nums::<usize>(v).try_into().unwrap(),
            "dst_crs" => c.dst_crs = v.into(),
            "dst_gt" => c.dst_gt = nums::<f64>(v).try_into().unwrap(),
            "wm" => c.wm = v.parse().unwrap(),
            "nodata" => c.nodata = v == "True",
            "block" => c.block = (v != "-").then(|| nums::<usize>(v).try_into().unwrap()),
            "special" => c.special.push(nums::<f64>(v).try_into().unwrap()),
            "chunk" => {
                let f: Vec<f64> = nums(v);
                let u = |i: usize| [f[i] as usize, f[i + 1] as usize];
                c.chunks.push((u(0), u(2), u(4), u(6), [f[8], f[9]]));
            }
            _ => {}
        }
    }
    c.vals = lines.flat_map(|l| l.split_whitespace()).map(|v| v.parse().unwrap()).collect();
    assert_eq!(c.vals.len(), c.dst_size[0] * c.dst_size[1], "{name}: value count");
    c
}

#[cfg(any(feature = "proj", feature = "proj4rs"))]
mod gdal {
    use super::*;
    use rwarp::approx::ApproxTransformer;
    use rwarp::transform::GenImgProjTransformer;

    const NODATA: f64 = -9999.0;
    const CASES: [&str; 7] = [
        "merc_lanczos",
        "merc_bilinear",
        "merc_bilinear_tiled",
        "stere_near",
        "stere_cubic",
        "laea_near",
        "laea_bilinear",
    ];

    fn transformer(c: &Case) -> ApproxTransformer<GenImgProjTransformer> {
        let exact = GenImgProjTransformer::new("EPSG:4326", c.src_gt, &c.dst_crs, c.dst_gt).unwrap();
        ApproxTransformer::new(exact, 0.125)
    }

    fn options(c: &Case) -> ChunkOptions {
        ChunkOptions::gdalwarp(c.alg)
            .with_memory_limit(c.wm)
            .with_pixel_cost(DataType::Float64, 1, c.nodata, c.nodata)
            .with_special_points(c.special.clone())
            .with_dst_block_size(c.block)
    }

    fn warp_opts(c: &Case) -> WarpOptions {
        let o = WarpOptions::new(c.alg);
        if c.nodata {
            o.with_src_nodata(Nodata::Value(NODATA))
        } else {
            o
        }
    }

    fn source(c: &Case) -> Vec<f64> {
        let [w, h] = c.src_size;
        (0..h).flat_map(|r| (0..w).map(move |col| value(col, r))).collect()
    }

    fn run(c: &Case, plan: &WarpPlan) -> Vec<f64> {
        let t = transformer(c);
        let data = source(c);
        let [w, h] = c.src_size;
        let buf = SourceBuffer { data: &data, ncol: w, nrow: h, nbands: 1, col_off: 0, row_off: 0 };
        warp_chunks(&t, plan, &buf, &warp_opts(c))
    }

    /// (pixels differing beyond 1e-6 relative, max absolute difference)
    fn compare(got: &[f64], want: &[f64]) -> (usize, f64) {
        let mut n = 0;
        let mut max = 0.0f64;
        for (g, w) in got.iter().zip(want) {
            let d = (g - w).abs();
            if d > 1e-6 * w.abs().max(1.0) {
                n += 1;
                max = max.max(d);
            }
        }
        (n, max)
    }

    #[test]
    fn chunk_lists_match_gdalwarp() {
        for name in CASES {
            let c = read_case(name);
            let plan = plan_chunks(&transformer(&c), c.src_size, c.dst_size, &options(&c));
            let got: Vec<_> = plan.live().map(|k| (k.src_off, k.src_size, k.dst_off, k.dst_size)).collect();
            let want: Vec<_> = c.chunks.iter().map(|k| (k.0, k.1, k.2, k.3)).collect();
            assert_eq!(got, want, "{name}: chunks");
            assert!(want.len() > 4, "{name}: {} chunks, want many", want.len());
            // GDAL logs scales with %f
            for (k, w) in plan.live().zip(&c.chunks) {
                for i in 0..2 {
                    assert!((k.scale[i] - w.4[i]).abs() <= 5e-7, "{name}: scale {:?} vs {:?}", k.scale, w.4);
                }
            }
            // the chunks tile the target
            let area: usize = plan.chunks.iter().map(|k| k.dst_size[0] * k.dst_size[1]).sum();
            assert_eq!(area, c.dst_size[0] * c.dst_size[1], "{name}: coverage");
        }
    }

    #[test]
    fn whole_raster_warps_match_gdalwarp() {
        let mut report = Vec::new();
        for name in CASES {
            let c = read_case(name);
            let plan = plan_chunks(&transformer(&c), c.src_size, c.dst_size, &options(&c));
            let (n, max) = compare(&run(&c, &plan), &c.vals);
            report.push(format!("{name}: {n} of {} differ (max {max:e}), {} chunks", c.vals.len(), plan.live().count()));
            eprintln!("{}", report.last().unwrap());
            assert_eq!(n, 0, "{}", report.last().unwrap());
        }
    }

    /// rwarp's default estimates each chunk's kernel scale over the chunk;
    /// GDAL estimates it at the output's top-left. Under Mercator the y
    /// scale changes with latitude, so the widened lanczos differs.
    #[test]
    fn scale_at_chunk_differs_where_gdal_omits_the_offset() {
        let c = read_case("merc_lanczos");
        let t = transformer(&c);
        let ours = plan_chunks(&t, c.src_size, c.dst_size, &options(&c).with_scale_origin(ScaleOrigin::Chunk));
        let gdal = plan_chunks(&t, c.src_size, c.dst_size, &options(&c));
        let windows = |p: &WarpPlan| p.chunks.iter().map(|k| (k.dst_off, k.src_off, k.src_size)).collect::<Vec<_>>();
        assert_eq!(windows(&ours), windows(&gdal), "same chunks");
        assert!(ours.live().zip(gdal.live()).any(|(a, b)| a.scale != b.scale), "some scale differs");
        let (n, _) = compare(&run(&c, &ours), &c.vals);
        eprintln!("merc_lanczos with ScaleOrigin::Chunk: {n} of {} differ from gdalwarp", c.vals.len());
        assert!(n > 0);
    }

    #[test]
    fn source_window_matches_plan() {
        let c = read_case("stere_near");
        let t = transformer(&c);
        let planner = rwarp::plan::ChunkPlanner::new(&t, c.src_size, c.dst_size, options(&c));
        let plan = planner.plan();
        for k in plan.live() {
            let w = planner.source_window(k.dst_off, k.dst_size).unwrap();
            assert_eq!([w.xoff as usize, w.yoff as usize], k.src_off);
            assert_eq!([w.xsize as usize, w.ysize as usize], k.src_size);
            assert_eq!(w.fill_ratio, k.fill_ratio);
        }
    }
}

/// An affine map from target to source pixels, `src = dst * f`, with an
/// optional "approximate" error: calls with more than five points are off
/// by `err` in x, as an interpolating transformer can be; five or fewer
/// are exact (as for GDAL's and rwarp's approximate transformers).
struct Scaled {
    f: f64,
    err: f64,
}

impl Transformer for Scaled {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let n = x.len();
        let (f, e) = if dst_to_src { (self.f, if n > 5 { self.err } else { 0.0 }) } else { (1.0 / self.f, 0.0) };
        for (x, y) in x.iter_mut().zip(y.iter_mut()) {
            *x = *x * f + e;
            *y *= f;
        }
        vec![true; n]
    }
}

/// GDAL's retry: a point the approximate transformer puts less than one
/// pixel off the source window is transformed again on its own, which is
/// exact, so the edge pixel keeps its value.
#[test]
fn points_just_off_the_window_are_retried_alone() {
    let data: Vec<f64> = (0..100).map(|v| v as f64 + 1.0).collect();
    let buf = SourceBuffer { data: &data, ncol: 10, nrow: 10, nbands: 1, col_off: 0, row_off: 0 };
    for alg in [ResampleAlg::NearestNeighbour, ResampleAlg::Bilinear, ResampleAlg::Cubic] {
        let o = WarpOptions::new(alg).with_scaling(KernelScaling::Point);
        let exact: Vec<f64> = warp(&Scaled { f: 1.0, err: 0.0 }, &buf, 10, 10, &o);
        // pushed 0.75 px right: the last column lands past x = 10
        let approx: Vec<f64> = warp(&Scaled { f: 1.0, err: 0.75 }, &buf, 10, 10, &o);
        for r in 0..10 {
            assert_eq!(approx[r * 10 + 9], exact[r * 10 + 9], "{alg:?} row {r}");
            assert_ne!(approx[r * 10 + 9], 0.0, "{alg:?} row {r}: filled");
        }
        // pushed 1.5 px: off by more than a pixel, not retried
        let far: Vec<f64> = warp(&Scaled { f: 1.0, err: 1.5 }, &buf, 10, 10, &o);
        assert!((0..10).all(|r| far[r * 10 + 9] == 0.0), "{alg:?}: beyond one pixel stays off");
    }
}

/// A plan as values: the memory limit splits, chunks tile the target, and
/// warping chunk by chunk equals one warp when the kernel scale is fixed.
#[test]
fn plan_tiles_target_and_warps_like_one_pass() {
    let t = Scaled { f: 2.0, err: 0.0 };
    let opts = ChunkOptions::new(ResampleAlg::Bilinear)
        .with_memory_limit(100_000.0)
        .with_pixel_cost(DataType::Float64, 1, false, false);
    let plan = plan_chunks(&t, [200, 160], [100, 80], &opts);
    assert!(plan.chunks.len() >= 4, "{} chunks", plan.chunks.len());
    let mut seen = vec![0u8; 100 * 80];
    for k in &plan.chunks {
        assert_eq!(k.scale, [0.5, 0.5]);
        let mem = 8.0 * (k.src_size[0] * k.src_size[1] + k.dst_size[0] * k.dst_size[1]) as f64;
        assert!(mem <= 100_000.0, "{k:?}");
        for r in k.dst_off[1]..k.dst_off[1] + k.dst_size[1] {
            for col in k.dst_off[0]..k.dst_off[0] + k.dst_size[0] {
                seen[r * 100 + col] += 1;
            }
        }
    }
    assert!(seen.iter().all(|&s| s == 1), "chunks tile the target once");
    // sorted top to bottom, then left to right
    assert!(plan.chunks.windows(2).all(|w| (w[0].dst_off[1], w[0].dst_off[0]) < (w[1].dst_off[1], w[1].dst_off[0])));

    let data: Vec<f64> = (0..200 * 160).map(|i| value(i % 200, i / 200)).collect();
    let buf = SourceBuffer { data: &data, ncol: 200, nrow: 160, nbands: 1, col_off: 0, row_off: 0 };
    let wo = WarpOptions::new(ResampleAlg::Bilinear);
    let chunked: Vec<f64> = warp_chunks(&t, &plan, &buf, &wo);
    let whole: Vec<f64> = warp(&t, &buf, 100, 80, &wo.with_scaling(KernelScaling::Fixed { x: 0.5, y: 0.5 }));
    assert_eq!(chunked, whole);
    assert_eq!(plan.src_region(), Some(([0, 0], [200, 160])));
}

/// A target that misses the source plans one empty chunk, filled with the
/// destination nodata.
#[test]
fn empty_chunks_get_the_fill() {
    struct Away;
    impl Transformer for Away {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            x.iter_mut().for_each(|v| *v += 1000.0);
            y.iter_mut().for_each(|v| *v += 1000.0);
            vec![true; x.len()]
        }
    }
    let plan = plan_chunks(&Away, [10, 10], [8, 6], &ChunkOptions::new(ResampleAlg::Cubic));
    assert_eq!(plan.chunks.len(), 1);
    assert!(plan.chunks[0].is_empty());
    assert_eq!(plan.src_region(), None);
    let data = vec![1.0f32; 100];
    let buf = SourceBuffer { data: &data, ncol: 10, nrow: 10, nbands: 1, col_off: 0, row_off: 0 };
    let o = WarpOptions::new(ResampleAlg::Cubic).with_dst_nodata(rwarp::warp::DstNodata::Value(-1.0));
    let out: Vec<f32> = warp_chunks(&Away, &plan, &buf, &o);
    assert_eq!(out, vec![-1.0; 48]);
}

/// A low fill ratio splits a large target (GDAL's fill ratio heuristic):
/// here most of each window's padding falls outside a narrow source.
#[test]
fn fill_ratio_heuristic_can_be_turned_off() {
    let t = Scaled { f: 0.05, err: 0.0 };
    let base = ChunkOptions::new(ResampleAlg::Lanczos).with_scaling(KernelScaling::Point);
    let split = plan_chunks(&t, [12, 12], [240, 240], &base);
    let whole = plan_chunks(&t, [12, 12], [240, 240], &base.clone().with_fill_ratio_split(0.0, 100));
    assert_eq!(whole.chunks.len(), 1);
    assert!(split.chunks.len() > 1, "{:?}", split.chunks);
}

#[cfg(feature = "serde")]
#[test]
fn plans_round_trip_through_serde() {
    let t = Scaled { f: 2.0, err: 0.0 };
    let opts = ChunkOptions::new(ResampleAlg::Bilinear).with_memory_limit(100_000.0);
    let plan = plan_chunks(&t, [400, 320], [200, 160], &opts);
    let json = serde_json::to_string(&plan).unwrap();
    let back: WarpPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(back, plan);
    assert!(json.contains("\"fill_ratio\""));
}

/// The antimeridian split (0.2's `collect_chunk_list`): a target whose
/// source x jumps across the whole source width is split at the jump into
/// compact windows either side and a gap of a few columns; GDAL's planning
/// reads the whole width.
#[test]
fn split_antimeridian_reads_compact_windows() {
    // source column = target column + 100, wrapped at 200
    struct Wrap;
    impl Transformer for Wrap {
        fn transform(&self, _: bool, x: &mut [f64], _: &mut [f64]) -> Vec<bool> {
            x.iter_mut().for_each(|v| *v = (*v + 100.0).rem_euclid(200.0));
            vec![true; x.len()]
        }
    }
    let base = ChunkOptions::new(ResampleAlg::NearestNeighbour);
    let gdal = plan_chunks(&Wrap, [200, 150], [200, 150], &base);
    let split = plan_chunks(&Wrap, [200, 150], [200, 150], &base.with_split_antimeridian(true));
    assert_eq!(gdal.src_pixels(), 200 * 150);
    assert!(split.chunks.len() >= 2);
    // compact either side; only the columns at the jump read the whole width
    for k in &split.chunks {
        assert!(k.dst_size[0] <= 2 || k.src_size[0] < 100, "{k:?}");
    }
    let area: usize = split.chunks.iter().map(|k| k.dst_size[0] * k.dst_size[1]).sum();
    assert_eq!(area, 200 * 150);
}
