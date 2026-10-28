//! Plan and warp one 256 x 256 tile in a local Lambert azimuthal equal-area
//! projection, from a Web Mercator tile pyramid.
//!
//!     cargo run --example tile_laea
//!     cargo run --example tile_laea --no-default-features --features proj4rs
//!
//! The source pixels here are synthetic (each holds its column number), so
//! the example runs without network access; a real application fetches the
//! source window from tiles or a COG instead.

use rwarp::approx::ApproxTransformer;
use rwarp::source_window::compute_source_window;
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{warp, Nodata, ResampleAlg, SourceBuffer, WarpOptions};

const WEBMERC_HALF: f64 = 20037508.342789244;
const LAEA_TAS: &str = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";

fn main() -> Result<(), String> {
    // Source: Web Mercator at zoom 8, as one virtual raster.
    let n = 256 * (1 << 8);
    let px = 2.0 * WEBMERC_HALF / n as f64;
    let src_gt = [-WEBMERC_HALF, px, 0.0, WEBMERC_HALF, 0.0, -px];
    let src_size = [n, n];

    // Target: a 256 x 256 tile, 400 km across, centred on Tasmania.
    let extent = 400_000.0;
    let dst_gt = [-extent / 2.0, extent / 256.0, 0.0, extent / 2.0, 0.0, -extent / 256.0];

    // 1. Transformer: target pixel -> source pixel, approximated as gdalwarp does.
    let exact = GenImgProjTransformer::new("EPSG:3857", src_gt, LAEA_TAS, dst_gt)?;
    let t = ApproxTransformer::new(exact, 0.125);

    // 2. Plan: which source pixels does the tile need? No pixels are read.
    let win = compute_source_window(&t, [0, 0], [256, 256], src_size, 1)
        .ok_or("tile does not map onto the source")?;
    println!(
        "source window: offset ({}, {}), size {} x {} pixels",
        win.xoff, win.yoff, win.xsize, win.ysize
    );

    // 3. Read: fetch that window (here, synthesise it).
    let (w, h) = (win.xsize as usize, win.ysize as usize);
    let data: Vec<f32> = (0..w * h).map(|i| (win.xoff as usize + i % w) as f32).collect();
    let src = SourceBuffer {
        data: &data,
        ncol: w,
        nrow: h,
        nbands: 1,
        col_off: win.xoff as usize,
        row_off: win.yoff as usize,
    };

    // 4. Warp.
    let opts = WarpOptions { alg: ResampleAlg::Bilinear, src_nodata: Nodata::Value(f64::NAN), ..Default::default() };
    let out = warp(&t, &src, 256, 256, &opts);
    let valid = out.iter().filter(|v| !v.is_nan()).count();
    println!("warped {} of {} target pixels", valid, out.len());
    Ok(())
}
