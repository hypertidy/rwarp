//! Warp a curvilinear source (2D longitude and latitude arrays, as for a
//! satellite swath or an ocean model's native grid) onto a regular grid.
//!
//!     cargo run --example curvilinear_swath
//!
//! The swath is synthetic: a curved, rotated longitude/latitude grid south
//! of Tasmania, with values that encode the source pixel (col + 1000 * row).

use rwarp::geoloc_transform::{
    GenImgGeolocTransformer, GeolocArrays, GeolocConvention, GeolocOptions, PixelGeo,
};
use rwarp::warp::{warp, Nodata, ResampleAlg, SourceBuffer, WarpOptions};

fn main() -> Result<(), String> {
    let (ncol, nrow) = (24, 18);
    let mut lon = Vec::with_capacity(ncol * nrow);
    let mut lat = Vec::with_capacity(ncol * nrow);
    for j in 0..nrow {
        for i in 0..ncol {
            let (i, j) = (i as f64, j as f64);
            lon.push(142.0 + 0.30 * i + 0.08 * j + 0.004 * j * j);
            lat.push(-42.0 - 0.22 * j + 0.05 * i - 0.002 * i * i);
        }
    }

    // Coordinates are cell centres, as CF coordinate variables are. GDAL's
    // default convention is TopLeftCorner, so say so explicitly.
    let opts = GeolocOptions::default()
        .with_convention(GeolocConvention::PixelCenter)
        .with_lon_wrap(true);
    let swath = GeolocArrays::new(lon, lat, ncol, nrow, opts)?;

    // Target: a 40 x 32 grid of 10 km pixels in a local LAEA.
    let dst_crs = "+proj=laea +lat_0=-45 +lon_0=146 +datum=WGS84 +units=m +no_defs";
    let dst_gt = [-420000.0, 10000.0, 0.0, 330000.0, 0.0, -10000.0];
    let t = GenImgGeolocTransformer::new(
        PixelGeo::Geoloc(swath),
        "+proj=longlat +datum=WGS84 +no_defs",
        PixelGeo::affine(dst_gt)?,
        dst_crs,
    )?;

    let data: Vec<i32> = (0..nrow)
        .flat_map(|r| (0..ncol).map(move |c| (c + 1000 * r) as i32))
        .collect();
    let src = SourceBuffer { data: &data, ncol, nrow, nbands: 1, col_off: 0, row_off: 0 };
    let wopts = WarpOptions::new(ResampleAlg::NearestNeighbour)
        .with_src_nodata(Nodata::Value(-1.0));
    let out = warp(&t, &src, 40, 32, &wopts);

    let covered = out.iter().filter(|&&v| v != -1).count();
    println!("{covered} of {} target pixels fall on the swath", out.len());
    // The centre row, as source (col, row) pairs.
    let row: Vec<String> = out[16 * 40..17 * 40]
        .iter()
        .map(|&v| if v < 0 { ".".to_string() } else { format!("{},{}", v % 1000, v / 1000) })
        .collect();
    println!("centre row: {}", row.join(" "));
    Ok(())
}
