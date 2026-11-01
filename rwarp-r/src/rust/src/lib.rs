use extendr_api::prelude::*;
use std::ffi::{CStr, CString, c_char};
use proj_sys as ffi;
use rwarp::approx::ApproxTransformer;
use rwarp::plan::{ChunkOptions, ChunkPlanner};
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::ResampleAlg;

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

fn resample_alg(resample: &str) -> ResampleAlg {
    match resample {
        "near" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "lanczos" => ResampleAlg::Lanczos,
        _ => throw_r_error(&format!("unsupported resample method: {}", resample)),
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
) -> Robj {
    let t = transformer(src_crs, src_gt, dst_crs, dst_gt);
    let src_size = [src_dim[0] as usize, src_dim[1] as usize];
    let off = [dst_off[0] as usize, dst_off[1] as usize];
    let size = [dst_size[0] as usize, dst_size[1] as usize];
    let opts = ChunkOptions::default()
        .with_source_extra(Some(resample_padding as usize))
        .with_fill_ratio_split(min_fill_ratio, min_dst_size as usize);
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
            fill_ratio = c.fill_ratio
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
// extendr boilerplate
// ---------------------------------------------------------------------------

extendr_module! {
    mod rwarp;
    fn crs_describe;
    fn pixel_transform;
    fn warp_plan;
    fn warp_resample;
}
