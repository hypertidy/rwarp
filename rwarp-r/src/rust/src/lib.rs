use extendr_api::prelude::*;
use std::ffi::{CStr, CString, c_char};
use proj_sys as ffi;
use rwarp::approx::ApproxTransformer;
use rwarp::plan::{ChunkOptions, ChunkPlanner};
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::{
    BandNodata, DstNodata, KernelScaling, ModeTies, Nodata, ResampleAlg, Sample, SourceBuffer,
    WarpOptions,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn ctx_error(ctx: *mut ffi::PJ_CONTEXT) -> Option<String> {
    let code = unsafe { ffi::proj_context_errno(ctx) };
    if code == 0 {
        return None;
    }
    let msg = unsafe { ffi::proj_errno_string(code) };
    if msg.is_null() {
        Some(format!("PROJ error code {}", code))
    } else {
        Some(unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() })
    }
}

fn cstr_to_option(p: *const c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
    }
}

fn na_string() -> Robj {
    r!(extendr_api::NA_STRING)
}

fn transformer(src_crs: &str, src_gt: &[f64], dst_crs: &str, dst_gt: &[f64]) -> GenImgProjTransformer {
    let src_gt6: [f64; 6] = src_gt[..6].try_into().expect("src_gt must have 6 elements");
    let dst_gt6: [f64; 6] = dst_gt[..6].try_into().expect("dst_gt must have 6 elements");
    GenImgProjTransformer::new(src_crs, src_gt6, dst_crs, dst_gt6)
        .unwrap_or_else(|e| throw_r_error(&e))
}

/// The transformer as gdalwarp builds it: with the source size (so source
/// longitudes wrap at GDAL's CENTER_LONG) and the target extent of the full
/// destination grid (gdalwarp's -te).
fn transformer_gdal(
    src_crs: &str, src_gt: &[f64], src_size: [usize; 2],
    dst_crs: &str, dst_gt: &[f64], dst_full: [usize; 2],
) -> GenImgProjTransformer {
    let g = dst_gt;
    let (w, h) = (dst_full[0] as f64, dst_full[1] as f64);
    let xs = [g[0], g[0] + w * g[1], g[0] + h * g[2], g[0] + w * g[1] + h * g[2]];
    let ys = [g[3], g[3] + w * g[4], g[3] + h * g[5], g[3] + w * g[4] + h * g[5]];
    let fmin = |v: &[f64; 4]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let fmax = |v: &[f64; 4]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    transformer(src_crs, src_gt, dst_crs, dst_gt)
        .with_source_size(src_size)
        .with_target_extent([fmin(&xs), fmin(&ys), fmax(&xs), fmax(&ys)])
}

fn inv_gt_apply(g: &[f64], x: f64, y: f64) -> [f64; 2] {
    let det = g[1] * g[5] - g[2] * g[4];
    let (dx, dy) = (x - g[0], y - g[3]);
    [(g[5] * dx - g[2] * dy) / det, (-g[4] * dx + g[1] * dy) / det]
}

/// The source CRS's geographic CRS, as WKT, via PROJ.
fn geodetic_wkt(crs: &str) -> Option<String> {
    let cdef = CString::new(rwarp_crs_string(crs)).ok()?;
    unsafe {
        let ctx = ffi::proj_context_create();
        if ctx.is_null() { return None; }
        let pj = ffi::proj_create(ctx, cdef.as_ptr());
        let mut out = None;
        if !pj.is_null() {
            let geod = ffi::proj_crs_get_geodetic_crs(ctx, pj);
            if !geod.is_null() {
                out = cstr_to_option(ffi::proj_as_wkt(ctx, geod, ffi::PJ_WKT_TYPE_PJ_WKT2_2019, std::ptr::null()));
                ffi::proj_destroy(geod);
            }
            ffi::proj_destroy(pj);
        }
        ffi::proj_context_destroy(ctx);
        out
    }
}

fn rwarp_crs_string(s: &str) -> String {
    if s.starts_with("+proj=") && !s.contains("+type=crs") { format!("{} +type=crs", s) } else { s.to_string() }
}

/// GDAL's special points (GDALTransformLonLatToDestGenImgProjTransformer):
/// longitude 0 at latitudes -89.9999 and 89.9999, as target pixels. A
/// target window holding one is grid-sampled when planned, so a pole inside
/// it is not missed. For a source geographic in degrees GDAL (3.6 and later)
/// takes the two points as target coordinates and applies only the target's
/// inverse geotransform; otherwise they are reprojected from the source's
/// geographic CRS.
fn special_points(src_crs: &str, dst_crs: &str, dst_gt: &[f64]) -> Vec<[f64; 2]> {
    let pts = [(0.0, -89.9999), (0.0, 89.9999)];
    if rwarp::crs::is_geographic_degrees(src_crs).unwrap_or(false) {
        return pts.iter().map(|&(x, y)| inv_gt_apply(dst_gt, x, y)).collect();
    }
    let Some(geod) = geodetic_wkt(src_crs) else { return Vec::new() };
    let Ok(b) = rwarp::crs::ProjBackend::new(&geod, dst_crs) else { return Vec::new() };
    use rwarp::crs::CrsTransform;
    pts.iter()
        .filter_map(|&(x, y)| b.convert(x, y))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| inv_gt_apply(dst_gt, x, y))
        .collect()
}

fn resample_alg(resample: &str) -> ResampleAlg {
    match resample {
        "near" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "cubicspline" => ResampleAlg::CubicSpline,
        "lanczos" => ResampleAlg::Lanczos,
        "average" => ResampleAlg::Average,
        "mode" => ResampleAlg::Mode,
        "min" => ResampleAlg::Min,
        "max" => ResampleAlg::Max,
        "med" => ResampleAlg::Med,
        "rms" => ResampleAlg::Rms,
        _ => throw_r_error(&format!("unsupported resample method: {}", resample)),
    }
}


fn kernel_scaling(scaling: &str, scale: &[f64]) -> KernelScaling {
    match scaling {
        "widen" => KernelScaling::WidenWhenDownsampling,
        "point" => KernelScaling::Point,
        "fixed" => {
            if scale.len() != 2 || !scale.iter().all(|v| v.is_finite() && *v > 0.0) {
                throw_r_error("scaling = \"fixed\" needs scale = c(x, y), both positive");
            }
            KernelScaling::Fixed { x: scale[0], y: scale[1] }
        }
        _ => throw_r_error(&format!("unsupported scaling: {}", scaling)),
    }
}

fn mode_ties(ties: &str) -> ModeTies {
    match ties {
        "first" => ModeTies::First,
        "min" => ModeTies::Min,
        "max" => ModeTies::Max,
        _ => throw_r_error(&format!("unsupported mode_ties: {}", ties)),
    }
}

fn band_nodata(rule: &str) -> BandNodata {
    match rule {
        "partial" => BandNodata::Partial,
        "unified" => BandNodata::Unified,
        "per_band" => BandNodata::PerBand,
        _ => throw_r_error(&format!("unsupported band_nodata: {}", rule)),
    }
}

// ---------------------------------------------------------------------------
// CRS utilities
// ---------------------------------------------------------------------------

/// Describe a CRS string in all available output formats.
///
/// Calls PROJ directly (via proj-sys) to expose the full range of
/// representations for any coordinate reference system string: EPSG codes,
/// WKT variants, PROJ strings, PROJJSON.  Useful for debugging why a
/// particular CRS string is accepted or rejected by the warp pipeline.
///
/// @param definition A CRS string: EPSG code ("EPSG:4326"), PROJ string
///   ("+proj=laea +datum=WGS84 +type=crs"), WKT, etc.
///
/// @return A named list with elements: input, id, description, definition,
///   wkt2_2019, wkt2_2019_simplified, wkt1_gdal, wkt1_esri, proj5, proj4,
///   projjson, error.  Elements that could not be produced are NA_character_.
///
/// @examples
/// d <- crs_describe("EPSG:4326")
/// cat(d$wkt2_2019)
///
/// # Bare PROJ string without +type=crs is treated as an *operation*, not a CRS,
/// # so it has no WKT1 form
/// crs_describe("+proj=laea")$wkt1_gdal           # NA
/// crs_describe("+proj=laea +type=crs")$wkt1_gdal # WKT
/// @export
#[extendr]
fn crs_describe(definition: &str) -> Robj {
    let cdef = match CString::new(definition) {
        Ok(s) => s,
        Err(e) => throw_r_error(&format!("invalid definition string: {}", e)),
    };
    unsafe {
        let ctx = ffi::proj_context_create();
        if ctx.is_null() {
            throw_r_error("proj_context_create() returned NULL");
        }
        let pj = ffi::proj_create(ctx, cdef.as_ptr());
        let error = if pj.is_null() {
            ctx_error(ctx)
                .unwrap_or_else(|| "proj_create returned NULL (unknown error)".to_string())
        } else {
            String::new()
        };
        let (id, desc, defn) = if !pj.is_null() {
            let info = ffi::proj_pj_info(pj);
            (cstr_to_option(info.id), cstr_to_option(info.description), cstr_to_option(info.definition))
        } else {
            (None, None, None)
        };
        let null_opts: *const *const c_char = std::ptr::null();
        let mk = |wkt_type: ffi::PJ_WKT_TYPE| {
            if pj.is_null() { None }
            else { cstr_to_option(ffi::proj_as_wkt(ctx, pj, wkt_type, null_opts)) }
        };
        let wkt2_2019            = mk(ffi::PJ_WKT_TYPE_PJ_WKT2_2019);
        let wkt2_2019_simplified = mk(ffi::PJ_WKT_TYPE_PJ_WKT2_2019_SIMPLIFIED);
        let wkt1_gdal            = mk(ffi::PJ_WKT_TYPE_PJ_WKT1_GDAL);
        let wkt1_esri            = mk(ffi::PJ_WKT_TYPE_PJ_WKT1_ESRI);
        let proj5 = if pj.is_null() { None } else {
            cstr_to_option(ffi::proj_as_proj_string(ctx, pj, ffi::PJ_PROJ_STRING_TYPE_PJ_PROJ_5, null_opts))
        };
        let proj4 = if pj.is_null() { None } else {
            cstr_to_option(ffi::proj_as_proj_string(ctx, pj, ffi::PJ_PROJ_STRING_TYPE_PJ_PROJ_4, null_opts))
        };
        let projjson = if pj.is_null() { None } else {
            cstr_to_option(ffi::proj_as_projjson(ctx, pj, null_opts))
        };
        if !pj.is_null() { ffi::proj_destroy(pj); }
        ffi::proj_context_destroy(ctx);
        let opt_str = |x: Option<String>| -> Robj {
            match x { Some(s) => r!(s), None => na_string() }
        };
        let error_robj: Robj = if error.is_empty() { na_string() } else { r!(error) };
        let pairs: Vec<(String, Robj)> = vec![
            ("input".into(),                r!(definition)),
            ("id".into(),                   opt_str(id)),
            ("description".into(),          opt_str(desc)),
            ("definition".into(),           opt_str(defn)),
            ("wkt2_2019".into(),            opt_str(wkt2_2019)),
            ("wkt2_2019_simplified".into(), opt_str(wkt2_2019_simplified)),
            ("wkt1_gdal".into(),            opt_str(wkt1_gdal)),
            ("wkt1_esri".into(),            opt_str(wkt1_esri)),
            ("proj5".into(),                opt_str(proj5)),
            ("proj4".into(),                opt_str(proj4)),
            ("projjson".into(),             opt_str(projjson)),
            ("error".into(),                error_robj),
        ];
        List::from_pairs(pairs).into()
    }
}

// ---------------------------------------------------------------------------
// Coordinate transform
// ---------------------------------------------------------------------------

/// Map destination pixel centres to fractional source pixel coordinates.
///
/// Implements GDAL's GenImgProjTransform:
///   dst pixel -> dst geo (geotransform) -> src geo (PROJ) -> src pixel (inverse gt).
/// The returned source pixel coordinates are 0-based and fractional; NA means
/// the point falls outside the PROJ domain or behind the horizon.
///
/// @param src_crs CRS of the source raster (WKT, EPSG code, or PROJ string
///   with `+type=crs`).
/// @param src_gt Geotransform of the source raster: 6-element numeric vector
///   `c(xmin, xres, 0, ymax, 0, -yres)` (north-up convention).
/// @param dst_crs CRS of the destination raster.
/// @param dst_gt Geotransform of the destination raster.
/// @param dst_dim Integer vector `c(ncol, nrow)` of the destination grid.
///
/// @return A list with elements `src_x` and `src_y`: numeric vectors of
///   length `prod(dst_dim)` giving fractional 0-based source pixel coordinates
///   in column-major (row-major on-wire) order.
///
/// @seealso [warp_plan()] to compute a full chunk plan,
///   [warp_resample()] to execute a warp.
///
/// @examples
/// \dontrun{
/// src_gt <- c(147.0, 0.0001, 0, -35.0, 0, -0.0001)
/// dst_gt <- c(147.0, 0.001, 0, -35.0, 0, -0.001)
/// coords <- pixel_transform(
///   src_crs = "EPSG:4326", src_gt = src_gt,
///   dst_crs = "EPSG:4326", dst_gt = dst_gt,
///   dst_dim = c(100L, 100L)
/// )
/// range(coords$src_x, na.rm = TRUE)
/// }
/// @export
#[extendr]
fn pixel_transform(
    src_crs: &str, src_gt: &[f64],
    dst_crs: &str, dst_gt: &[f64],
    dst_dim: &[i32],
) -> List {
    let t = transformer(src_crs, src_gt, dst_crs, dst_gt);
    let dim = [dst_dim[0] as usize, dst_dim[1] as usize];
    let (src_x, src_y) = rwarp::transform::transform_grid(&t, &dim);
    list!(src_x = src_x, src_y = src_y)
}

// ---------------------------------------------------------------------------
// Chunk planning
// ---------------------------------------------------------------------------

/// Plan the source windows required to fill a destination raster window.
///
/// Wraps `rwarp::plan::ChunkPlanner`.  Returns a list of chunk descriptors,
/// each containing the source window (col/row offset + size) and fill ratio.
/// Chunks that read no source are dropped.
///
/// @param src_crs CRS of the source raster.
/// @param src_gt Geotransform of the source raster (6-element numeric).
/// @param src_dim Integer `c(ncol, nrow)` of the source raster.
/// @param dst_crs CRS of the destination raster.
/// @param dst_gt Geotransform of the destination raster (6-element numeric).
/// @param dst_off Integer `c(col_off, row_off)` of the destination window
///   origin (0-based). Default `c(0L, 0L)`.
/// @param dst_size Integer `c(ncol, nrow)` of the destination window.
///   Default uses the full grid implied by `dst_gt`.
/// @param resample_padding Extra pixels of source context for interpolating
///   kernels (e.g. 1 for bilinear). Default 1.
/// @param min_fill_ratio A chunk with fill above 0 and below this fraction is
///   split. Default 0.0 (never split on fill).
/// @param min_dst_size Only split a chunk with a side longer than this
///   (pixels). Default 1.
///
/// @return A list of chunk lists.  Each top-level element corresponds to one
///   destination sub-window and contains:
///   \describe{
///     \item{dst_xoff, dst_yoff}{Destination window offset (0-based).}
///     \item{dst_xsize, dst_ysize}{Destination window size.}
///     \item{src_xoff, src_yoff}{Source window offset (0-based).}
///     \item{src_xsize, src_ysize}{Source window size.}
///     \item{fill_ratio}{Area of the source window over the area of the
///       window before clamping to the source raster.}
///   }
///   Returns `NULL` if no valid chunks could be found.
///
/// @examples
/// \dontrun{
/// plan <- warp_plan(
///   src_crs = "EPSG:4326",
///   src_gt  = c(-180, 0.01, 0, 90, 0, -0.01),
///   src_dim = c(36000L, 18000L),
///   dst_crs = "EPSG:3857",
///   dst_gt  = c(-20037508, 156543, 0, 20037508, 0, -156543),
///   dst_size = c(256L, 256L)
/// )
/// length(plan)
/// plan[[1]]$fill_ratio
/// }
/// @export
#[extendr]
fn warp_plan(
    src_crs: &str, src_gt: &[f64], src_dim: &[i32],
    dst_crs: &str, dst_gt: &[f64],
    dst_off: &[i32], dst_size: &[i32],
    resample_padding: i32,
    min_fill_ratio: f64,
    min_dst_size: i32,
    resample: &str,
    scaling: &str,
    scale: &[f64],
    split_antimeridian: bool,
) -> Robj {
    let src_size = [src_dim[0] as usize, src_dim[1] as usize];
    let off = [dst_off[0] as usize, dst_off[1] as usize];
    let size = [dst_size[0] as usize, dst_size[1] as usize];
    let full = [off[0] + size[0], off[1] + size[1]];
    let t = transformer_gdal(src_crs, src_gt, src_size, dst_crs, dst_gt, full);
    let opts = ChunkOptions::new(resample_alg(resample))
        .with_scaling(kernel_scaling(scaling, scale))
        .with_special_points(special_points(src_crs, dst_crs, dst_gt))
        .with_source_extra(if resample_padding < 0 { None } else { Some(resample_padding as usize) })
        .with_fill_ratio_split(min_fill_ratio, min_dst_size as usize)
        .with_split_antimeridian(split_antimeridian);
    let planner = ChunkPlanner::new(&t, src_size, [off[0] + size[0], off[1] + size[1]], opts);
    let plan = planner.plan_window(off, size);
    let chunks: Vec<Robj> = plan.live().map(|c| {
        list!(
            dst_xoff   = c.dst_off[0] as i32,
            dst_yoff   = c.dst_off[1] as i32,
            dst_xsize  = c.dst_size[0] as i32,
            dst_ysize  = c.dst_size[1] as i32,
            src_xoff   = c.src_off[0] as i32,
            src_yoff   = c.src_off[1] as i32,
            src_xsize  = c.src_size[0] as i32,
            src_ysize  = c.src_size[1] as i32,
            fill_ratio = c.fill_ratio,
            scale_x    = c.scale[0],
            scale_y    = c.scale[1]
        ).into()
    }).collect();
    if chunks.is_empty() {
        return r!(NULL);
    }
    let inner: Robj = List::from_values(chunks).into();
    List::from_values(vec![inner]).into()
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Warp and resample a source pixel buffer into a destination grid.
///
/// Core warp kernel.  Reads a pre-fetched source window (as returned by
/// e.g. `gdalraster::GDALRaster$read()`) and reprojects it into the
/// destination grid using the ApproxTransformer (adaptive scanline
/// interpolation, matching GDAL's default warp quality).
///
/// The source buffer must be a **row-major integer vector** of length
/// `src_ncol * src_nrow`.  Use `as.integer(as.vector(t(m)))` to convert from
/// a column-major R matrix `m`.
///
/// @param src_crs CRS of the source raster.
/// @param src_gt Geotransform of the source raster (6-element numeric).
/// @param dst_crs CRS of the destination raster.
/// @param dst_gt Geotransform of the destination raster (6-element numeric).
/// @param dst_dim Integer `c(ncol, nrow)` of the destination window.
/// @param src_pixels Integer vector (row-major) of source pixel values.
/// @param src_ncol Number of columns in the source buffer.
/// @param src_nrow Number of rows in the source buffer.
/// @param src_col_off Column offset of the source buffer within the full source
///   raster (0-based).
/// @param src_row_off Row offset of the source buffer within the full source
///   raster (0-based).
/// @param nodata No-data value written to destination pixels with no valid
///   source.  Note: the current implementation may write `i32::MIN`
///   (`NA` in R) instead of this value for some out-of-bounds pixels.
/// @param max_error Maximum pixel-coordinate error for the approx transformer
///   (default 0.125, matching GDAL).
/// @param resample Resampling method: "near" (default), "bilinear", "cubic"
///   or "lanczos".
///
/// @return Integer vector of length `prod(dst_dim)` in row-major order.
///   Convert to a display matrix with
///   `matrix(result, nrow = dst_dim[2], ncol = dst_dim[1], byrow = TRUE)`.
///
/// @seealso [warp_plan()] to compute source windows before calling this.
///
/// @examples
/// \dontrun{
/// ## Reproject a small GHRSST tile to EPSG:3412 (Antarctic polar stereo)
/// library(gdalraster)
/// library(vaster)
///
/// url <- paste0(
///   "/vsicurl/https://data.source.coop/ausantarctic/ghrsst-mur-v2/",
///   "2025/11/26/20251126090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1",
///   "_analysed_sst.tif"
/// )
/// ds <- new(GDALRaster, url, read_only = TRUE)
///
/// src_crs <- ds$getProjectionRef()
/// src_gt  <- ds$getGeoTransform()
/// src_dim <- c(ds$getRasterXSize(), ds$getRasterYSize())
///
/// ## 40 x 40 destination tile at Null Island, 25 km resolution, EPSG:3412
/// dst_crs <- "EPSG:3412"
/// center  <- reproj::reproj_xy(cbind(0, -60), target = dst_crs, source = "EPSG:4326")
/// dst_ex  <- c(center[1] - 500000, center[1] + 500000,
///              center[2] - 500000, center[2] + 500000)
/// dst_gt  <- vaster::extent_dim_to_gt(dst_ex, c(40L, 40L))
/// dst_dim <- c(40L, 40L)
///
/// plan <- warp_plan(
///   src_crs = src_crs, src_gt = src_gt, src_dim = as.integer(src_dim),
///   dst_crs = dst_crs, dst_gt = dst_gt,
///   dst_off = c(0L, 0L), dst_size = dst_dim,
///   resample_padding = 1L, min_fill_ratio = 0.0, min_dst_size = 1L
/// )
///
/// chunk <- plan[[1]][[1]]
/// src_window <- ds$read(1L,
///   xoff = chunk$src_xoff, yoff = chunk$src_yoff,
///   xsize = chunk$src_xsize, ysize = chunk$src_ysize,
///   out_xsize = chunk$src_xsize, out_ysize = chunk$src_ysize
/// )
/// src_buf <- as.integer(src_window)
///
/// result <- warp_resample(
///   src_crs = src_crs, src_gt = src_gt,
///   dst_crs = dst_crs, dst_gt = dst_gt,
///   dst_dim = dst_dim,
///   src_pixels  = src_buf,
///   src_ncol    = chunk$src_xsize,
///   src_nrow    = chunk$src_ysize,
///   src_col_off = chunk$src_xoff,
///   src_row_off = chunk$src_yoff,
///   nodata = -32768L
/// )
/// image(matrix(result, nrow = dst_dim[2], ncol = dst_dim[1], byrow = TRUE))
/// }
/// @export
#[extendr]
fn warp_resample(
    src_crs: &str, src_gt: &[f64],
    dst_crs: &str, dst_gt: &[f64],
    dst_dim: &[i32],
    src_pixels: &[i32],
    src_ncol: i32, src_nrow: i32,
    src_col_off: i32, src_row_off: i32,
    nodata: i32,
    max_error: f64,
    resample: &str,
) -> Vec<i32> {
    let t = ApproxTransformer::new(transformer(src_crs, src_gt, dst_crs, dst_gt), max_error);
    rwarp::warp::warp_resample(
        &t,
        src_pixels,
        src_ncol as usize, src_nrow as usize,
        src_col_off as usize, src_row_off as usize,
        dst_dim[0] as usize, dst_dim[1] as usize,
        nodata,
        resample_alg(resample),
    )
}


// ---------------------------------------------------------------------------
// Typed chunk loop: warp_exec()
// ---------------------------------------------------------------------------

/// The R vector types a read function can return, and the sample type each
/// maps to: integer -> i32 (NA is i32::MIN), double -> f64, raw -> u8.
trait RSample: Sample {
    fn from_robj(r: &Robj) -> Option<Vec<Self>>;
    /// An R nodata value (NA arrives as NaN) as this type's nodata, or None.
    /// `dst` is true for the destination, where a double NA means NaN fill.
    fn nodata(v: f64, dst: bool) -> Option<f64>;
    fn to_robj(v: Vec<Self>) -> Robj;
}

impl RSample for i32 {
    fn from_robj(r: &Robj) -> Option<Vec<Self>> { r.as_integer_slice().map(|s| s.to_vec()) }
    fn nodata(v: f64, _dst: bool) -> Option<f64> {
        if v.is_nan() { Some(i32::MIN as f64) } else { Some(v) }
    }
    fn to_robj(v: Vec<Self>) -> Robj { r!(v) }
}

impl RSample for f64 {
    fn from_robj(r: &Robj) -> Option<Vec<Self>> { r.as_real_slice().map(|s| s.to_vec()) }
    fn nodata(v: f64, dst: bool) -> Option<f64> {
        // NaN is always nodata in a source; as a destination it is the fill.
        if v.is_nan() { if dst { Some(f64::NAN) } else { None } } else { Some(v) }
    }
    fn to_robj(v: Vec<Self>) -> Robj { r!(v) }
}

impl RSample for u8 {
    fn from_robj(r: &Robj) -> Option<Vec<Self>> { r.as_raw_slice().map(|s| s.to_vec()) }
    fn nodata(v: f64, _dst: bool) -> Option<f64> { if v.is_nan() { None } else { Some(v) } }
    fn to_robj(v: Vec<Self>) -> Robj { Raw::from_bytes(&v).into() }
}

fn nodata_of<T: RSample>(v: &[f64], dst: bool) -> Vec<Option<f64>> {
    v.iter().map(|&x| T::nodata(x, dst)).collect()
}

fn src_nodata_opt<T: RSample>(v: &[f64]) -> Nodata {
    let nd = nodata_of::<T>(v, false);
    match nd.len() {
        0 => Nodata::None,
        1 => nd[0].map_or(Nodata::None, Nodata::Value),
        _ if nd.iter().all(|x| x.is_none()) => Nodata::None,
        // PerBand has no "none for this band"; a NaN never matches, so it stands in.
        _ => Nodata::PerBand(nd.iter().map(|x| x.unwrap_or(f64::NAN)).collect()),
    }
}

fn dst_nodata_opt<T: RSample>(v: &[f64]) -> DstNodata {
    let nd = nodata_of::<T>(v, true);
    match nd.len() {
        0 => DstNodata::None,
        1 => nd[0].map_or(DstNodata::None, DstNodata::Value),
        _ if nd.iter().all(|x| x.is_none()) => DstNodata::None,
        _ => DstNodata::PerBand(nd.iter().map(|x| x.unwrap_or(0.0)).collect()),
    }
}

struct ExecArgs<'a> {
    chunks: &'a [i32],   // 8 per chunk: dst xoff yoff xsize ysize, src xoff yoff xsize ysize
    scales: &'a [f64],   // 2 per chunk
    read_fn: &'a Function,
    t: &'a ApproxTransformer<GenImgProjTransformer>,
    dst_off: [usize; 2],
    dst_size: [usize; 2],
    nb: usize,
    verbose: bool,
}

fn read_chunk<T: RSample>(a: &ExecArgs, k: usize) -> Vec<T> {
    let c = &a.chunks[k * 8..k * 8 + 8];
    let got = a.read_fn
        .call(pairlist!(c[4], c[5], c[6], c[7]))
        .unwrap_or_else(|e| throw_r_error(&format!("read_fn failed: {}", e)));
    let v = T::from_robj(&got).unwrap_or_else(|| {
        throw_r_error("read_fn must return the same type (integer, double or raw) for every chunk")
    });
    let n = c[6] as usize * c[7] as usize * a.nb;
    if v.len() != n {
        throw_r_error(&format!(
            "read_fn returned {} values for a {}x{} window of {} band(s); expected {}",
            v.len(), c[6], c[7], a.nb, n
        ));
    }
    v
}

/// Band-sequential (as R reads bands) to pixel-interleaved (as rwarp warps).
fn interleave<T: Copy>(v: &[T], npix: usize, nb: usize) -> Vec<T> {
    if nb == 1 { return v.to_vec(); }
    let mut out = Vec::with_capacity(v.len());
    for p in 0..npix { for b in 0..nb { out.push(v[b * npix + p]); } }
    out
}

fn exec_typed<T: RSample>(a: &ExecArgs, first: Vec<T>, base: &WarpOptions) -> Vec<T> {
    let nb = a.nb;
    let [w, h] = a.dst_size;
    // The fill, from a warp that sees no source (as rwarp::plan::warp_chunks).
    let empty: SourceBuffer<'_, T> = SourceBuffer { data: &[], ncol: 0, nrow: 0, nbands: nb, col_off: 0, row_off: 0 };
    let mut o = base.clone();
    o.alg = ResampleAlg::NearestNeighbour;
    o.scaling = KernelScaling::Point;
    let fill: Vec<T> = rwarp::warp::warp_to(a.t, &empty, 1, 1, &o);
    let mut out: Vec<T> = Vec::with_capacity(w * h * nb);
    for _ in 0..w * h { out.extend_from_slice(&fill); }

    let nchunk = a.chunks.len() / 8;
    let mut first = Some(first);
    for k in 0..nchunk {
        let c = &a.chunks[k * 8..k * 8 + 8];
        let (sw, sh) = (c[6] as usize, c[7] as usize);
        if sw == 0 || sh == 0 { continue; }
        let raw = match first.take() { Some(v) => v, None => read_chunk::<T>(a, k) };
        let data = interleave(&raw, sw * sh, nb);
        let buf = SourceBuffer { data: &data, ncol: sw, nrow: sh, nbands: nb, col_off: c[4] as usize, row_off: c[5] as usize };
        let mut co = base.clone();
        co.scaling = KernelScaling::Fixed { x: a.scales[k * 2], y: a.scales[k * 2 + 1] };
        co.dst_col_off = c[0] as usize;
        co.dst_row_off = c[1] as usize;
        let (cw, ch) = (c[2] as usize, c[3] as usize);
        if a.verbose {
            rprintln!("chunk {}/{}: src [{},{}] {}x{} -> dst [{},{}] {}x{}, scale {:.3} {:.3}",
                k + 1, nchunk, c[4], c[5], sw, sh, c[0], c[1], cw, ch, a.scales[k * 2], a.scales[k * 2 + 1]);
        }
        let tile: Vec<T> = rwarp::warp::warp(a.t, &buf, cw, ch, &co);
        let (dx, dy) = (c[0] as usize - a.dst_off[0], c[1] as usize - a.dst_off[1]);
        // only the pixels the chunk wrote: the two chunks of an antimeridian
        // gap share a target window (rwarp::plan::merge_tile)
        rwarp::plan::merge_tile(&mut out, &tile, &fill, w, [dx, dy], cw, nb);
    }
    out
}

/// Pixel-interleaved, row-major to an R matrix (one band) or array
/// c(nrow, ncol, nbands), column-major.
fn to_r_array<T: RSample>(v: Vec<T>, w: usize, h: usize, nb: usize) -> Robj {
    let mut cm: Vec<T> = Vec::with_capacity(v.len());
    for b in 0..nb { for c in 0..w { for r in 0..h { cm.push(v[(r * w + c) * nb + b]); } } }
    let mut obj = T::to_robj(cm);
    let dim: Vec<i32> = if nb == 1 { vec![h as i32, w as i32] } else { vec![h as i32, w as i32, nb as i32] };
    obj.set_attrib("dim", dim).unwrap();
    obj
}

/// Bytes, pixel-interleaved and row-major, packed as a nativeRaster:
/// R | G<<8 | B<<16 | A<<24 per pixel, which is also row-major. One band is
/// grey, three are RGB, four are RGBA. Without an alpha band, a pixel whose
/// bands all equal the destination nodata is transparent.
fn to_nara(v: &[u8], w: usize, h: usize, nb: usize, dst_nd: &[Option<f64>]) -> Robj {
    let nd: Vec<Option<u8>> = (0..nb)
        .map(|b| dst_nd.get(if dst_nd.len() == 1 { 0 } else { b }).copied().flatten())
        .map(|x| x.filter(|v| (0.0..=255.0).contains(v)).map(|v| v as u8))
        .collect();
    let masked = nb != 4 && nd.iter().all(|x| x.is_some());
    let mut px: Vec<i32> = Vec::with_capacity(w * h);
    for p in 0..w * h {
        let s = &v[p * nb..(p + 1) * nb];
        let (r, g, b, a) = match nb {
            1 => (s[0], s[0], s[0], 255u8),
            3 => (s[0], s[1], s[2], 255u8),
            4 => (s[0], s[1], s[2], s[3]),
            _ => throw_r_error("output = \"nara\" needs 1, 3 or 4 bands"),
        };
        let a = if masked && s.iter().zip(&nd).all(|(x, n)| Some(*x) == *n) { 0 } else { a };
        px.push((r as u32 | (g as u32) << 8 | (b as u32) << 16 | (a as u32) << 24) as i32);
    }
    let mut obj: Robj = r!(px);
    obj.set_attrib("dim", vec![h as i32, w as i32]).unwrap();
    obj.set_attrib("channels", 4i32).unwrap();
    obj.set_class(&["nativeRaster"]).unwrap();
    obj
}

/// Internal: the chunk loop behind the R function warp_exec().
/// @noRd
#[extendr]
fn warp_exec_rs(
    chunks: &[i32], scales: &[f64],
    read_fn: Function,
    src_crs: &str, src_gt: &[f64], src_dim: &[i32],
    dst_crs: &str, dst_gt: &[f64],
    dst_off: &[i32], dst_size: &[i32],
    nbands: i32,
    max_error: f64,
    resample: &str,
    mode_ties_: &str,
    src_nodata: &[f64], dst_nodata: &[f64],
    band_nodata_: &str,
    mask_band: i32,
    nara: bool,
    verbose: bool,
) -> Robj {
    if chunks.len() % 8 != 0 || scales.len() != chunks.len() / 4 {
        throw_r_error("malformed plan: 8 integers and 2 scales per chunk");
    }
    if nbands < 1 { throw_r_error("nbands must be at least 1"); }
    let nb = nbands as usize;
    let t = if src_dim.len() == 2 {
        let full = [(dst_off[0] + dst_size[0]) as usize, (dst_off[1] + dst_size[1]) as usize];
        ApproxTransformer::new(transformer_gdal(
            src_crs, src_gt, [src_dim[0] as usize, src_dim[1] as usize], dst_crs, dst_gt, full), max_error)
    } else {
        ApproxTransformer::new(transformer(src_crs, src_gt, dst_crs, dst_gt), max_error)
    };
    let a = ExecArgs {
        chunks, scales, read_fn: &read_fn, t: &t,
        dst_off: [dst_off[0] as usize, dst_off[1] as usize],
        dst_size: [dst_size[0] as usize, dst_size[1] as usize],
        nb, verbose,
    };
    let mut base = WarpOptions::default();
    base.alg = resample_alg(resample);
    base.mode_ties = mode_ties(mode_ties_);
    base.band_nodata = band_nodata(band_nodata_);
    if mask_band > 0 {
        if mask_band as usize > nb { throw_r_error("mask_band is past the last band"); }
        base.mask_band = Some(mask_band as usize - 1);
    }
    let [w, h] = a.dst_size;

    // The first non-empty chunk's read decides the type.
    let k0 = (0..chunks.len() / 8).find(|&k| chunks[k * 8 + 6] > 0 && chunks[k * 8 + 7] > 0);
    let probe: Robj = match k0 {
        Some(k) => {
            let c = &chunks[k * 8..k * 8 + 8];
            read_fn.call(pairlist!(c[4], c[5], c[6], c[7]))
                .unwrap_or_else(|e| throw_r_error(&format!("read_fn failed: {}", e)))
        }
        // Nothing to read: the type comes from read_fn on an empty window.
        None => read_fn.call(pairlist!(0i32, 0i32, 0i32, 0i32))
            .unwrap_or_else(|e| throw_r_error(&format!("read_fn failed: {}", e))),
    };
    // Run the typed loop; the probe read is reused for the first chunk.
    macro_rules! run {
        ($t:ty) => {{
            base.src_nodata = src_nodata_opt::<$t>(src_nodata);
            base.dst_nodata = dst_nodata_opt::<$t>(dst_nodata);
            let first: Vec<$t> = match k0 {
                Some(k) => {
                    let v = <$t>::from_robj(&probe).unwrap();
                    let c = &chunks[k * 8..k * 8 + 8];
                    let n = c[6] as usize * c[7] as usize * nb;
                    if v.len() != n {
                        throw_r_error(&format!(
                            "read_fn returned {} values for a {}x{} window of {} band(s); expected {}",
                            v.len(), c[6], c[7], nb, n));
                    }
                    v
                }
                None => Vec::new(),
            };
            exec_typed::<$t>(&a, first, &base)
        }};
    }
    match probe.rtype() {
        Rtype::Integers => {
            let v = run!(i32);
            if nara { throw_r_error("output = \"nara\" needs raw (byte) data from read_fn"); }
            to_r_array(v, w, h, nb)
        }
        Rtype::Doubles => {
            let v = run!(f64);
            if nara { throw_r_error("output = \"nara\" needs raw (byte) data from read_fn"); }
            to_r_array(v, w, h, nb)
        }
        Rtype::Raw => {
            let v = run!(u8);
            if nara {
                to_nara(&v, w, h, nb, &nodata_of::<u8>(dst_nodata, true))
            } else {
                to_r_array(v, w, h, nb)
            }
        }
        _ => throw_r_error("read_fn must return an integer, double or raw vector"),
    }
}

// ---------------------------------------------------------------------------
// extendr boilerplate
// ---------------------------------------------------------------------------

extendr_module! {
    mod rwarp;
    fn crs_describe;
    fn pixel_transform;
    fn warp_plan;
    fn warp_resample;
    fn warp_exec_rs;
}
