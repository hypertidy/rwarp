//! GDAL's backmap geolocation inverse ([`GeolocInverse::Backmap`]) against
//! gdalwarp, and where it differs from rwarp's quad index.
//!
//! The fixtures in `tests/fixtures/geoloc/` are written by `gen.py` there
//! (which records the GDAL version): curvilinear lon/lat sources warped by
//! gdalwarp through its geolocation transformer. Each case rebuilds the
//! geolocation arrays from the same formulas (checked against a hash of
//! GDAL's arrays) and compares:
//!
//! - **coordinates**: GDAL's transformer, destination pixel centres to
//!   source pixel/line: success exactly where GDAL succeeds, and the same
//!   position to 1e-6 pixels (5e-10 measured);
//! - **estimate**: `GDAL_GEOLOC_USE_MAX_ACCURACY=NO` against
//!   [`GeolocInverse::BackmapEstimate`], which checks the float32 backmap
//!   itself, hole filling included;
//! - **warps**: nearest neighbour pixel for pixel, bilinear and cubic to
//!   1e-9 relative (identical, measured), planned and warped in chunks as
//!   gdalwarp does, with `-et 0` and, for some cases, through the
//!   approximate transformer at gdalwarp's default `-et 0.125`.
//!
//! The grids follow wxylib/fixtures' curvilinear set, which wxy-warp's
//! port of the backmap (`geoloc.rs`) was checked against: a curved swath
//! by cell centres and by cell corners, to EPSG:4326 (fine and coarse) and
//! EPSG:3031, and a polar ring crossing the antimeridian; plus a swath
//! across the antimeridian, one with holes in its arrays and source nodata,
//! and one with overlapping scans.
//!
//! Runs with either CRS backend.

use std::path::PathBuf;

use rwarp::geoloc_transform::{
    GenImgGeolocTransformer, GeolocArrays, GeolocConvention, GeolocInverse, GeolocOptions, PixelGeo,
};
use rwarp::approx::ApproxTransformer;
use rwarp::plan::{plan_chunks, warp_chunks, ChunkOptions};
use rwarp::warp::{DataType, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::Transformer;

const NODATA: f64 = -9999.0;
const SRC_CRS: &str = "+proj=longlat +datum=WGS84 +no_defs";

/// Coordinate tolerance against GDAL, in source pixels. Through the
/// identity (EPSG:4326 targets) rwarp agrees with GDAL to rounding; through
/// a projection the CRS step differs in its last digits between PROJ
/// versions and proj4rs.
const TOL: f64 = 1e-6;

fn value(c: usize, r: usize) -> f64 {
    ((c * 7919 + r * 104729) % 1000) as f64 / 10.0 + c as f64 * 0.5 - r as f64 * 0.25
}

fn src_nodata(c: usize, r: usize) -> bool {
    (c * 3 + r * 5) % 17 == 0
}

fn geoloc_missing(c: usize, r: usize) -> bool {
    ((20..24).contains(&r) && (30..42).contains(&c)) || (r * 7 + c * 13) % 37 == 0 || (c == 0 && r < 10)
}

fn wrap(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// `grid_point` in gen.py, operation for operation.
fn grid_point(kind: &str, nx: usize, ny: usize, c: usize, r: usize, shift: f64) -> (f64, f64) {
    let i = c as f64 + shift;
    let j = r as f64 + shift;
    let (nxf, nyf) = (nx as f64, ny as f64);
    match kind {
        "swath" | "holes" => {
            let h = nyf / 2.0;
            let t = i / nxf;
            (
                140.0 + 0.275 * i + 0.075 * j + 0.0025 * ((j - h) * (j - h)),
                -40.0 - 0.225 * j + 0.05 * i + 6.0 * (t * (1.0 - t)),
            )
        }
        "polar" => (wrap(-180.0 + 360.0 * (i + 0.5) / nxf + 0.2 * j), -60.0 - 25.0 * (j + 0.5) / nyf),
        "dateline" => {
            let h = nyf / 2.0;
            (wrap(168.0 + 0.35 * i + 0.06 * j + 0.0015 * ((j - h) * (j - h))), -45.0 - 0.25 * j + 0.04 * i)
        }
        "bowtie" => {
            let e = (i - nxf / 2.0) / (nxf / 2.0);
            let o = 0.5 + 1.0 * (e * e);
            let k = (r / 6) as f64;
            (150.0 + 0.3 * i + 0.02 * j, -50.0 - 0.3 * j + 0.3 * (o * k))
        }
        _ => panic!("grid {kind}"),
    }
}

fn fnv1a(vals: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in vals {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

struct Case {
    name: String,
    grid: String,
    size: [usize; 2],
    shift: f64,
    convention: GeolocConvention,
    raster: [usize; 2],
    geoloc_nodata: Option<f64>,
    src_nodata: bool,
    fnv1a: u64,
    dst_crs: String,
    dst_gt: [f64; 6],
    dst_size: [usize; 2],
    sections: Vec<(String, Vec<f64>)>,
}

fn nums<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

fn read_case(name: &str) -> Case {
    let path: PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "geoloc", &format!("{name}.txt")].iter().collect();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut c = Case {
        name: name.into(),
        grid: String::new(),
        size: [0; 2],
        shift: 0.0,
        convention: GeolocConvention::TopLeftCorner,
        raster: [0; 2],
        geoloc_nodata: None,
        src_nodata: false,
        fnv1a: 0,
        dst_crs: String::new(),
        dst_gt: [0.0; 6],
        dst_size: [0; 2],
        sections: vec![],
    };
    for l in text.lines() {
        if let Some(h) = l.strip_prefix("# ") {
            let (k, v) = h.split_once(' ').unwrap();
            match k {
                "grid" => c.grid = v.into(),
                "size" => c.size = nums::<usize>(v).try_into().unwrap(),
                "shift" => c.shift = v.parse().unwrap(),
                "convention" => {
                    c.convention = match v {
                        "PIXEL_CENTER" => GeolocConvention::PixelCenter,
                        "TOP_LEFT_CORNER" => GeolocConvention::TopLeftCorner,
                        _ => panic!("convention {v}"),
                    }
                }
                "raster" => c.raster = nums::<usize>(v).try_into().unwrap(),
                "geoloc_nodata" => c.geoloc_nodata = (v != "-").then(|| v.parse().unwrap()),
                "src_nodata" => c.src_nodata = v == "True",
                "fnv1a" => c.fnv1a = v.parse().unwrap(),
                "dst_crs" => c.dst_crs = v.into(),
                "dst_gt" => c.dst_gt = nums::<f64>(v).try_into().unwrap(),
                "dst_size" => c.dst_size = nums::<usize>(v).try_into().unwrap(),
                _ => {}
            }
        } else if let Some(s) = l.strip_prefix("@ ") {
            c.sections.push((s.into(), vec![]));
        } else {
            c.sections.last_mut().unwrap().1.extend(nums::<f64>(l));
        }
    }
    c
}

impl Case {
    fn section(&self, name: &str) -> Option<&[f64]> {
        self.sections.iter().find(|s| s.0 == name).map(|s| &s.1[..])
    }

    fn arrays(&self) -> (Vec<f64>, Vec<f64>) {
        let [nx, ny] = self.size;
        let (mut lon, mut lat) = (Vec::new(), Vec::new());
        for r in 0..ny {
            for c in 0..nx {
                let (mut x, mut y) = grid_point(&self.grid, nx, ny, c, r, self.shift);
                if let Some(nd) = self.geoloc_nodata {
                    if geoloc_missing(c, r) {
                        (x, y) = (nd, nd);
                    }
                }
                lon.push(x);
                lat.push(y);
            }
        }
        let all: Vec<f64> = lon.iter().chain(&lat).copied().collect();
        assert_eq!(fnv1a(&all), self.fnv1a, "{}: arrays differ from gen.py's", self.name);
        (lon, lat)
    }

    fn geoloc(&self, inverse: GeolocInverse) -> GeolocArrays {
        let (lon, lat) = self.arrays();
        // GDAL wraps longitudes for a geographic SRS with every longitude in
        // [-180, 180], which holds for every case.
        let opts = GeolocOptions::default()
            .with_convention(self.convention)
            .with_nodata_x(self.geoloc_nodata)
            .with_lon_wrap(true)
            .with_inverse(inverse);
        GeolocArrays::new(lon, lat, self.size[0], self.size[1], opts).unwrap()
    }

    fn transformer(&self, inverse: GeolocInverse) -> GenImgGeolocTransformer {
        GenImgGeolocTransformer::new(
            PixelGeo::Geoloc(self.geoloc(inverse)),
            SRC_CRS,
            PixelGeo::affine(self.dst_gt).unwrap(),
            &self.dst_crs,
        )
        .unwrap()
    }

    fn centres(&self) -> (Vec<f64>, Vec<f64>) {
        let [w, h] = self.dst_size;
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for r in 0..h {
            for c in 0..w {
                x.push(c as f64 + 0.5);
                y.push(r as f64 + 0.5);
            }
        }
        (x, y)
    }

    /// The source field `gen.py` warps for `alg`.
    fn source(&self, alg: &str) -> Vec<f64> {
        let [w, h] = self.raster;
        (0..h)
            .flat_map(|r| {
                (0..w).map(move |c| {
                    if self.src_nodata && src_nodata(c, r) {
                        NODATA
                    } else if alg == "near" {
                        (c + 1000 * r) as f64
                    } else {
                        value(c, r)
                    }
                })
            })
            .collect()
    }

    /// The warp `gen.py` records as section `section`: `<alg>` exact,
    /// `<alg>_et` through the approximate transformer at gdalwarp's default
    /// 0.125 pixels. Planned and warped in chunks as gdalwarp does.
    fn warp(&self, inverse: GeolocInverse, section: &str) -> Vec<f64> {
        let (alg, et) = match section.strip_suffix("_et") {
            Some(a) => (a, true),
            None => (section, false),
        };
        let ra = match alg {
            "near" => ResampleAlg::NearestNeighbour,
            "bilinear" => ResampleAlg::Bilinear,
            "cubic" => ResampleAlg::Cubic,
            _ => panic!("alg {alg}"),
        };
        let opts = ChunkOptions::gdalwarp(ra).with_pixel_cost(DataType::Float64, 1, true, true);
        let data = self.source(alg);
        let [w, h] = self.raster;
        let buf = SourceBuffer { data: &data, ncol: w, nrow: h, nbands: 1, col_off: 0, row_off: 0 };
        let wo = WarpOptions::new(ra).with_src_nodata(Nodata::Value(NODATA));
        let t = self.transformer(inverse);
        if et {
            let t = ApproxTransformer::new(t, 0.125);
            let plan = plan_chunks(&t, self.raster, self.dst_size, &opts);
            warp_chunks(&t, &plan, &buf, &wo)
        } else {
            let plan = plan_chunks(&t, self.raster, self.dst_size, &opts);
            warp_chunks(&t, &plan, &buf, &wo)
        }
    }
}

const CASES: [&str; 9] = [
    "swath_pc_4326",
    "swath_pc_4326_coarse",
    "swath_pc_3031",
    "swath_tlc_4326",
    "swath_tlc_3031",
    "polar_pc_3031",
    "dateline_pc_laea",
    "holes_pc_4326",
    "bowtie_tlc_4326",
];

/// (points where success differs, max coordinate difference where both
/// succeed, points where both succeed)
fn compare_coords(got: &[f64], got_ok: &[bool], want: &[f64]) -> (Vec<usize>, f64, usize) {
    let mut bad = Vec::new();
    let mut max: f64 = 0.0;
    let mut both = 0;
    for k in 0..got_ok.len() {
        let wx = want[2 * k];
        match (got_ok[k], !wx.is_nan()) {
            (true, true) => {
                both += 1;
                max = max.max((got[2 * k] - wx).abs()).max((got[2 * k + 1] - want[2 * k + 1]).abs());
            }
            (false, false) => {}
            _ => bad.push(k),
        }
    }
    (bad, max, both)
}

fn transform_centres(c: &Case, inverse: GeolocInverse) -> (Vec<f64>, Vec<bool>) {
    let t = c.transformer(inverse);
    let (mut x, mut y) = c.centres();
    let ok = t.transform(true, &mut x, &mut y);
    (x.iter().zip(&y).flat_map(|(a, b)| [*a, *b]).collect(), ok)
}

#[test]
fn backmap_coordinates_match_gdal() {
    let mut report = Vec::new();
    for name in CASES {
        let c = read_case(name);
        let (got, ok) = transform_centres(&c, GeolocInverse::Backmap);
        let (bad, max, both) = compare_coords(&got, &ok, c.section("coords").unwrap());
        let line = format!("{name}: {both} points, success differs at {}, max difference {max:e}", bad.len());
        eprintln!("{line}");
        report.push((line, bad.is_empty() && max < TOL && both > 100));
    }
    let failed: Vec<_> = report.iter().filter(|r| !r.1).map(|r| &r.0).collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn backmap_estimate_matches_gdal_without_max_accuracy() {
    let mut report = Vec::new();
    for name in CASES {
        let c = read_case(name);
        let Some(want) = c.section("approx") else { continue };
        let (got, ok) = transform_centres(&c, GeolocInverse::BackmapEstimate);
        let (bad, max, both) = compare_coords(&got, &ok, want);
        let line = format!("{name}: {both} points, success differs at {}, max difference {max:e}", bad.len());
        eprintln!("{line}");
        report.push((line, bad.is_empty() && max < TOL && both > 100));
    }
    assert!(report.len() >= 5);
    let failed: Vec<_> = report.iter().filter(|r| !r.1).map(|r| &r.0).collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

/// (pixels differing beyond `rel` relative, max absolute difference)
fn compare(got: &[f64], want: &[f64], rel: f64) -> (usize, f64) {
    let mut n = 0;
    let mut max = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        if d > rel * w.abs().max(1.0) {
            n += 1;
            max = max.max(d);
        }
    }
    (n, max)
}

#[test]
fn backmap_warps_match_gdalwarp() {
    let mut report = Vec::new();
    for name in CASES {
        let c = read_case(name);
        for alg in ["near", "bilinear", "cubic", "near_et", "bilinear_et"] {
            let Some(want) = c.section(alg) else { continue };
            let got = c.warp(GeolocInverse::Backmap, alg);
            // Nearest neighbour: identical. Interpolated: to rounding.
            let (n, max) = compare(&got, want, if alg.starts_with("near") { 0.0 } else { 1e-9 });
            let filled = want.iter().filter(|v| **v != NODATA).count();
            let line = format!("{name} {alg}: {n} of {} differ (max {max:e}), {filled} filled", want.len());
            eprintln!("{line}");
            report.push((line, n == 0));
        }
    }
    let failed: Vec<_> = report.iter().filter(|r| !r.1).map(|r| &r.0).collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

/// Where the quad index and the backmap differ, on the same cases, and why.
///
/// Counted over target pixel centres: the inverse succeeds with one method
/// only (inside the source raster, or outside it, where no kernel reads),
/// or with both at different positions; and pixels of the nearest-neighbour
/// warp. Measured (EPSG:4326 cases exactly; the projected ones with libproj
/// 9 and proj4rs alike):
///
/// - Grids without folds or seams agree wherever both succeed, and their
///   warps are identical. GDAL succeeds at more points outside the raster:
///   its search reaches extrapolated squares more than one beyond the top
///   and left edges, the quad index stops one ring out.
/// - `holes_pc_4326`: one point lands in two squares 0.4 pixels apart, both
///   in the raster's last line, so the nearest pixel is the same. Next to a
///   missing sample GDAL's `PixelLineToXY` extrapolates beyond the last
///   line along the line only, which folds the extrapolated square back
///   over the grid, where the two methods find different squares.
/// - `bowtie_tlc_4326`: where scans overlap, the quad index takes the
///   earlier scan (row-major order) and GDAL the later one (its backmap
///   estimate lands there first): 22 points, 19 warp pixels, each one row
///   apart.
/// - `polar_pc_3031`: the arrays wrap 360 degrees with no square across the
///   seam between the last and first column. Both methods fill the seam
///   with extrapolated squares, which overlap; and GDAL's backmap, averaged
///   across the antimeridian, puts its estimate near the last column for
///   points whose square is in the first few, beyond its search radius, so
///   GDAL leaves a line of empty pixels along the antimeridian that the
///   quad index fills.
#[test]
fn quad_index_and_backmap_differ_only_where_expected() {
    let mut report = Vec::new();
    for name in CASES {
        let c = read_case(name);
        let [w, h] = c.raster;
        let inside = |x: f64, y: f64| x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64;
        let (q, q_ok) = transform_centres(&c, GeolocInverse::QuadIndex);
        let (b, b_ok) = transform_centres(&c, GeolocInverse::Backmap);
        let (mut only_q, mut only_b, mut only_inside, mut apart, mut max) = (0, 0, 0, 0, 0.0f64);
        for k in 0..q_ok.len() {
            let (qx, qy, bx, by) = (q[2 * k], q[2 * k + 1], b[2 * k], b[2 * k + 1]);
            match (q_ok[k], b_ok[k]) {
                (true, false) => {
                    only_q += 1;
                    only_inside += inside(qx, qy) as usize;
                }
                (false, true) => {
                    only_b += 1;
                    only_inside += inside(bx, by) as usize;
                }
                (true, true) => {
                    let d = (qx - bx).abs().max((qy - by).abs());
                    if d > 1e-9 {
                        apart += 1;
                    }
                    max = max.max(d);
                }
                _ => {}
            }
        }
        let near_q = c.warp(GeolocInverse::QuadIndex, "near");
        let near_b = c.warp(GeolocInverse::Backmap, "near");
        let px = near_q.iter().zip(&near_b).filter(|(a, b)| a != b).count();
        let line = format!(
            "{name}: quad index only {only_q}, backmap only {only_b} ({only_inside} of these inside the raster), \
             both but apart {apart} (max {max:.3} px); nearest warp: {px} of {} pixels differ",
            near_q.len()
        );
        eprintln!("{line}");
        let expected = match name {
            "holes_pc_4326" => only_inside == 0 && apart == 1 && px == 0,
            "bowtie_tlc_4326" => only_inside == 0 && apart == 22 && px == 19,
            "polar_pc_3031" => only_inside > 0 && px > 0,
            _ => only_inside == 0 && apart == 0 && px == 0,
        };
        report.push((line, expected));
    }
    let failed: Vec<_> = report.iter().filter(|r| !r.1).map(|r| &r.0).collect();
    assert!(failed.is_empty(), "{failed:#?}");
}
