//! Geolocation-array warps checked against GDAL.
//!
//! The fixture (`tests/fixtures/geoloc_swath.rs`) is a synthetic curved
//! longitude/latitude swath south of Tasmania, warped by GDAL to a local
//! LAEA grid that overhangs it on every side, once per georeferencing
//! convention. Regenerate it with `tests/fixtures/make_geoloc_swath.py`.
//!
//! Two checks per convention:
//!
//! - **Coordinates.** For every target pixel centre, rwarp's destination to
//!   source transform succeeds exactly where GDAL's does, and agrees on the
//!   source pixel/line.
//! - **Values.** A nearest-neighbour warp of a raster whose values encode
//!   the source pixel (col + 1000 * row) matches `gdalwarp -geoloc -r near
//!   -et 0`, pixel for pixel.
//!
//! Runs with either CRS backend.

#[path = "fixtures/geoloc_swath.rs"]
mod fx;

use rwarp::geoloc_transform::{
    GenImgGeolocTransformer, GeolocArrays, GeolocConvention, GeolocOptions, PixelGeo,
};
use rwarp::warp::{warp, KernelScaling, Nodata, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::Transformer;

const SRC_CRS: &str = "+proj=longlat +datum=WGS84 +no_defs";

/// Coordinate tolerance against GDAL, in pixels. The fixture's PROJ and the
/// one rwarp links (or proj4rs) differ in the last digits of the CRS step,
/// by up to a few 1e-8 pixels across versions; the geolocation arithmetic
/// itself agrees to about 1e-13.
const TOL: f64 = 1e-6;

/// Strictly inside the raster, clear of its edges by more than `TOL`. Points
/// on or very near an edge may succeed on one side and not the other when
/// CRS results differ in their last digits, so they are not compared.
fn well_inside(px: f64, py: f64, ncol: usize, nrow: usize) -> bool {
    px > TOL && px < ncol as f64 - TOL && py > TOL && py < nrow as f64 - TOL
}

fn transformer(convention: GeolocConvention) -> GenImgGeolocTransformer {
    let opts = GeolocOptions {
        convention,
        // GDAL sets this for a geographic SRS with longitudes in [-180, 180].
        lon_wrap: true,
        ..Default::default()
    };
    let g = GeolocArrays::new(fx::LON.to_vec(), fx::LAT.to_vec(), fx::NCOL, fx::NROW, opts)
        .expect("geolocation arrays");
    GenImgGeolocTransformer::new(
        PixelGeo::Geoloc(g),
        SRC_CRS,
        PixelGeo::affine(fx::DST_GT).unwrap(),
        fx::DST_CRS,
    )
    .expect("transformer")
}

fn check_coordinates(convention: GeolocConvention, gdal_x: &[f64], gdal_y: &[f64]) {
    let t = transformer(convention);
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for r in 0..fx::DST_NROW {
        for c in 0..fx::DST_NCOL {
            x.push(c as f64 + 0.5);
            y.push(r as f64 + 0.5);
        }
    }
    let ok = t.transform(true, &mut x, &mut y);

    // Outside the raster, GDAL's domain depends on its backmap's bounding
    // box, so either side may succeed there with a coordinate no kernel
    // uses. Well inside the raster, success must agree exactly.
    let inside = |px: f64, py: f64| well_inside(px, py, fx::NCOL, fx::NROW);
    let mut max_err: f64 = 0.0;
    let mut both = 0;
    let mut mismatched = Vec::new();
    for k in 0..x.len() {
        let gdal_ok = !gdal_x[k].is_nan();
        match (ok[k], gdal_ok) {
            (true, true) => {
                both += 1;
                max_err = max_err.max((x[k] - gdal_x[k]).abs()).max((y[k] - gdal_y[k]).abs());
            }
            (true, false) if inside(x[k], y[k]) => mismatched.push((k, "rwarp only", x[k], y[k])),
            (false, true) if inside(gdal_x[k], gdal_y[k]) => {
                mismatched.push((k, "GDAL only", gdal_x[k], gdal_y[k]))
            }
            _ => {}
        }
    }
    assert!(
        mismatched.is_empty(),
        "{convention:?}: success differs from GDAL inside the raster at {} points, first: {:?}",
        mismatched.len(),
        mismatched.first()
    );
    assert!(both > 800, "{convention:?}: only {both} points to compare");
    assert!(max_err < TOL, "{convention:?}: max coordinate difference {max_err} pixels");
}

fn check_nearest(convention: GeolocConvention, gdal_near: &[i32]) {
    let t = transformer(convention);
    let src: Vec<i32> = (0..fx::NROW)
        .flat_map(|r| (0..fx::NCOL).map(move |c| (c + 1000 * r) as i32))
        .collect();
    let buf = SourceBuffer { data: &src[..], ncol: fx::NCOL, nrow: fx::NROW, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions {
        alg: ResampleAlg::NearestNeighbour,
        scaling: KernelScaling::Point,
        src_nodata: Nodata::Value(fx::NODATA as f64),
        ..Default::default()
    };
    let out = warp(&t, &buf, fx::DST_NCOL, fx::DST_NROW, &opts);
    let diffs: Vec<(usize, i32, i32)> = out
        .iter()
        .zip(gdal_near)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(k, (a, b))| (k, *a, *b))
        .collect();
    assert!(
        diffs.is_empty(),
        "{convention:?}: {} of {} pixels differ from gdalwarp, first: {:?}",
        diffs.len(),
        out.len(),
        diffs.first()
    );
}

#[test]
fn pixel_center_coordinates_match_gdal() {
    check_coordinates(GeolocConvention::PixelCenter, &fx::PC_SRC_X, &fx::PC_SRC_Y);
}

#[test]
fn top_left_corner_coordinates_match_gdal() {
    check_coordinates(GeolocConvention::TopLeftCorner, &fx::TLC_SRC_X, &fx::TLC_SRC_Y);
}

#[test]
fn pixel_center_nearest_matches_gdalwarp() {
    check_nearest(GeolocConvention::PixelCenter, &fx::PC_NEAR);
}

#[test]
fn top_left_corner_nearest_matches_gdalwarp() {
    check_nearest(GeolocConvention::TopLeftCorner, &fx::TLC_NEAR);
}

// ---------------------------------------------------------------------------
// The other direction: an affine source warped onto a curvilinear target
// (GDAL's DST_METHOD=GEOLOC_ARRAY), PIXEL_CENTER.
// ---------------------------------------------------------------------------

fn dst_geoloc_transformer() -> GenImgGeolocTransformer {
    let opts = GeolocOptions {
        convention: GeolocConvention::PixelCenter,
        lon_wrap: true,
        ..Default::default()
    };
    let g = GeolocArrays::new(fx::LON.to_vec(), fx::LAT.to_vec(), fx::NCOL, fx::NROW, opts)
        .expect("geolocation arrays");
    GenImgGeolocTransformer::new(
        PixelGeo::affine(fx::DST_GT).unwrap(),
        fx::DST_CRS,
        PixelGeo::Geoloc(g),
        SRC_CRS,
    )
    .expect("transformer")
}

fn centres(ncol: usize, nrow: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for r in 0..nrow {
        for c in 0..ncol {
            x.push(c as f64 + 0.5);
            y.push(r as f64 + 0.5);
        }
    }
    (x, y)
}

#[test]
fn curvilinear_target_forward_matches_gdal() {
    // Target (swath) pixel centres to the affine source: forward geolocation.
    let t = dst_geoloc_transformer();
    let (mut x, mut y) = centres(fx::NCOL, fx::NROW);
    let ok = t.transform(true, &mut x, &mut y);
    for k in 0..x.len() {
        assert!(ok[k] && !fx::DSTGEO_D2S_X[k].is_nan(), "point {k}");
        assert!((x[k] - fx::DSTGEO_D2S_X[k]).abs() < TOL, "point {k}");
        assert!((y[k] - fx::DSTGEO_D2S_Y[k]).abs() < TOL, "point {k}");
    }
}

#[test]
fn curvilinear_target_inverse_matches_gdal() {
    // Source pixel centres to the swath: inverse geolocation.
    let t = dst_geoloc_transformer();
    let (mut x, mut y) = centres(fx::DST_NCOL, fx::DST_NROW);
    let ok = t.transform(false, &mut x, &mut y);
    let inside = |px: f64, py: f64| well_inside(px, py, fx::NCOL, fx::NROW);
    let mut both = 0;
    for k in 0..x.len() {
        let gdal_ok = !fx::DSTGEO_S2D_X[k].is_nan();
        match (ok[k], gdal_ok) {
            (true, true) => {
                both += 1;
                assert!((x[k] - fx::DSTGEO_S2D_X[k]).abs() < TOL, "point {k}");
                assert!((y[k] - fx::DSTGEO_S2D_Y[k]).abs() < TOL, "point {k}");
            }
            (true, false) => assert!(!inside(x[k], y[k]), "rwarp only, inside: point {k}"),
            (false, true) => assert!(
                !inside(fx::DSTGEO_S2D_X[k], fx::DSTGEO_S2D_Y[k]),
                "GDAL only, inside: point {k}"
            ),
            (false, false) => {}
        }
    }
    assert!(both > 600, "only {both} points to compare");
}

#[test]
fn curvilinear_target_nearest_matches_gdalwarp() {
    let t = dst_geoloc_transformer();
    let src: Vec<i32> = (0..fx::DST_NROW)
        .flat_map(|r| (0..fx::DST_NCOL).map(move |c| (c + 1000 * r) as i32))
        .collect();
    let buf = SourceBuffer { data: &src[..], ncol: fx::DST_NCOL, nrow: fx::DST_NROW, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions {
        alg: ResampleAlg::NearestNeighbour,
        scaling: KernelScaling::Point,
        src_nodata: Nodata::Value(fx::NODATA as f64),
        ..Default::default()
    };
    let out = warp(&t, &buf, fx::NCOL, fx::NROW, &opts);
    let diffs: Vec<(usize, i32, i32)> = out
        .iter()
        .zip(fx::DSTGEO_NEAR.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(k, (a, b))| (k, *a, *b))
        .collect();
    assert!(diffs.is_empty(), "{} pixels differ, first: {:?}", diffs.len(), diffs.first());
}
