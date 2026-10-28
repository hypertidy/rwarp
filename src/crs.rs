//! CRS-to-CRS coordinate transform backends.
//!
//! `GenImgProjTransformer` needs exactly one thing from a projection library:
//! map a geo coordinate in one CRS to a geo coordinate in another. This module
//! isolates that behind [`CrsTransform`] so the rest of the pipeline (approx
//! transformer, source-window planning, resampling kernels) is independent of
//! which projection library is linked.
//!
//! Two backends are provided, selected by Cargo feature:
//!
//! | feature   | backend                        | needs            | wasm |
//! | --------- | ------------------------------ | ---------------- | ---- |
//! | `proj`    | `proj` crate over libproj      | native libproj   | no   |
//! | `proj4rs` | `proj4rs` pure-Rust proj4      | nothing          | yes  |
//!
//! Both may be enabled; `proj` is preferred when present. Callers can also
//! bypass both by implementing [`CrsTransform`] themselves and using
//! [`crate::transform::GenImgProjTransformer::with_backend`] -- for example a
//! wasm module that forwards to a JavaScript PROJ build, or a lookup table.
//!
//! ## Conventions
//!
//! A `CrsTransform` has its direction fixed at construction (source CRS and
//! target CRS are given once). Geographic coordinates are always longitude,
//! latitude in **degrees**, matching the `proj` crate's `new_known_crs`
//! behaviour (axis order normalised to easting/northing). The `proj4rs`
//! backend converts to and from radians internally so both backends agree.

/// Map a coordinate from one CRS to another. Direction is fixed at
/// construction. Returns `None` when the point cannot be transformed
/// (outside the projection domain, NaN, etc.).
///
/// No `Send`/`Sync` bound: the `proj` backend wraps raw libproj handles,
/// which are neither (PROJ contexts are per-thread). Build one transformer
/// per thread. The `proj4rs` backend is plain data and is `Send + Sync`.
pub trait CrsTransform {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)>;
}

/// Add `+type=crs` to a bare proj string so PROJ treats it as a CRS rather
/// than an operation. Not needed for `proj4rs`, which ignores unknown keys.
#[cfg_attr(not(feature = "proj"), allow(dead_code))]
pub(crate) fn ensure_crs(s: &str) -> std::borrow::Cow<'_, str> {
    if s.starts_with("+proj=") && !s.contains("+type=crs") {
        std::borrow::Cow::Owned(format!("{s} +type=crs"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// Build the `(dst -> src, src -> dst)` pair of transforms with the default
/// backend for this build.
pub fn crs_pair(
    src_crs: &str,
    dst_crs: &str,
) -> Result<(Box<dyn CrsTransform>, Box<dyn CrsTransform>), String> {
    #[cfg(feature = "proj")]
    {
        let d2s = ProjBackend::new(dst_crs, src_crs)?;
        let s2d = ProjBackend::new(src_crs, dst_crs)?;
        return Ok((Box::new(d2s), Box::new(s2d)));
    }
    #[cfg(all(not(feature = "proj"), feature = "proj4rs"))]
    {
        let d2s = Proj4rsBackend::new(dst_crs, src_crs)?;
        let s2d = Proj4rsBackend::new(src_crs, dst_crs)?;
        return Ok((Box::new(d2s), Box::new(s2d)));
    }
    #[cfg(not(any(feature = "proj", feature = "proj4rs")))]
    {
        let _ = (src_crs, dst_crs);
        Err("rwarp built without a CRS backend: enable the `proj` or `proj4rs` feature, \
             or construct GenImgProjTransformer::with_backend"
            .to_string())
    }
}

/// Whether `crs` is geographic with degrees as its angular unit, with the
/// backend compiled in: the test GDAL makes before wrapping a source's
/// longitudes around its centre (`IsGeographic()` and `GetAngularUnits()`
/// within 1e-9 of a degree, in `InsertCenterLong`, gdaltransformer.cpp).
///
/// With libproj, a bound CRS counts by its base CRS and a compound CRS by
/// its horizontal part, as in GDAL. With proj4rs, `+proj=longlat` (and
/// `latlong`, `lonlat`, `latlon`) is geographic, always in degrees.
pub fn is_geographic_degrees(crs: &str) -> Result<bool, String> {
    #[cfg(feature = "proj")]
    {
        return proj_geographic_degrees(crs);
    }
    #[cfg(all(not(feature = "proj"), feature = "proj4rs"))]
    {
        let p = proj4rs::proj::Proj::from_user_string(crs)
            .map_err(|e| format!("proj4rs cannot parse {crs:?}: {e}"))?;
        return Ok(p.is_latlong());
    }
    #[cfg(not(any(feature = "proj", feature = "proj4rs")))]
    {
        let _ = crs;
        Err("rwarp built without a CRS backend".to_string())
    }
}

/// GDAL's `SRS_UA_DEGREE_CONV`.
#[cfg_attr(not(feature = "proj"), allow(dead_code))]
const DEGREE_TO_RADIAN: f64 = 0.0174532925199433;

#[cfg(feature = "proj")]
fn proj_geographic_degrees(crs: &str) -> Result<bool, String> {
    use proj_sys::*;
    use std::ffi::CString;

    let def = CString::new(ensure_crs(crs).as_ref()).map_err(|e| format!("{crs:?}: {e}"))?;
    // SAFETY: every object created here is destroyed before return, with
    // the context last; the pointers are checked for null before use.
    unsafe {
        let ctx = proj_context_create();
        if ctx.is_null() {
            return Err("proj_context_create failed".to_string());
        }
        let mut owned: Vec<*mut PJ> = Vec::new();
        let result = (|| {
            let pj = proj_create(ctx, def.as_ptr());
            if pj.is_null() {
                return Err(format!("PROJ cannot parse {crs:?}"));
            }
            owned.push(pj);
            // OGRSpatialReference::IsGeographic: a bound CRS by its base, a
            // compound CRS by its first (horizontal) component, itself
            // perhaps bound.
            let mut crs_pj = pj;
            if proj_get_type(crs_pj) == PJ_TYPE_PJ_TYPE_BOUND_CRS {
                let base = proj_get_source_crs(ctx, crs_pj);
                if base.is_null() {
                    return Ok(false);
                }
                owned.push(base);
                crs_pj = base;
            }
            if proj_get_type(crs_pj) == PJ_TYPE_PJ_TYPE_COMPOUND_CRS {
                let horiz = proj_crs_get_sub_crs(ctx, crs_pj, 0);
                if horiz.is_null() {
                    return Ok(false);
                }
                owned.push(horiz);
                crs_pj = horiz;
                if proj_get_type(crs_pj) == PJ_TYPE_PJ_TYPE_BOUND_CRS {
                    let base = proj_get_source_crs(ctx, crs_pj);
                    if base.is_null() {
                        return Ok(false);
                    }
                    owned.push(base);
                    crs_pj = base;
                }
            }
            let t = proj_get_type(crs_pj);
            if t != PJ_TYPE_PJ_TYPE_GEOGRAPHIC_2D_CRS && t != PJ_TYPE_PJ_TYPE_GEOGRAPHIC_3D_CRS {
                return Ok(false);
            }
            // OGRSpatialReference::GetAngularUnits: the first axis of the
            // geodetic CRS's ellipsoidal coordinate system, else degrees.
            let mut factor = DEGREE_TO_RADIAN;
            let geod = proj_crs_get_geodetic_crs(ctx, crs_pj);
            if !geod.is_null() {
                owned.push(geod);
                let cs = proj_crs_get_coordinate_system(ctx, geod);
                if !cs.is_null() {
                    owned.push(cs);
                    if proj_cs_get_type(ctx, cs) == PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ELLIPSOIDAL {
                        let mut f = 0.0;
                        let ok = proj_cs_get_axis_info(
                            ctx,
                            cs,
                            0,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            &mut f,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                        );
                        if ok != 0 {
                            factor = f;
                        }
                    }
                }
            }
            Ok((factor - DEGREE_TO_RADIAN).abs() <= 1e-9)
        })();
        for p in owned.into_iter().rev() {
            proj_destroy(p);
        }
        proj_context_destroy(ctx);
        result
    }
}

// ---------------------------------------------------------------------------
// proj (libproj) backend
// ---------------------------------------------------------------------------

/// [`CrsTransform`] over the `proj` crate (libproj).
#[cfg(feature = "proj")]
pub struct ProjBackend {
    inner: proj::Proj,
}

#[cfg(feature = "proj")]
impl ProjBackend {
    pub fn new(from_crs: &str, to_crs: &str) -> Result<Self, String> {
        let from = ensure_crs(from_crs);
        let to = ensure_crs(to_crs);
        let inner = proj::Proj::new_known_crs(&from, &to, None)
            .map_err(|e| format!("PROJ {from_crs} -> {to_crs} failed: {e}"))?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "proj")]
impl CrsTransform for ProjBackend {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.inner.convert((x, y)).ok()
    }
}

// ---------------------------------------------------------------------------
// proj4rs (pure Rust) backend
// ---------------------------------------------------------------------------

/// [`CrsTransform`] over `proj4rs`. Accepts proj strings (`+proj=...`) and
/// `EPSG:NNNN` codes (via `crs-definitions`). No WKT, no proj.db, no grids.
#[cfg(feature = "proj4rs")]
pub struct Proj4rsBackend {
    from: proj4rs::proj::Proj,
    to: proj4rs::proj::Proj,
    from_is_latlong: bool,
    to_is_latlong: bool,
}

#[cfg(feature = "proj4rs")]
impl Proj4rsBackend {
    pub fn new(from_crs: &str, to_crs: &str) -> Result<Self, String> {
        let from = proj4rs::proj::Proj::from_user_string(from_crs)
            .map_err(|e| format!("proj4rs cannot parse {from_crs:?}: {e}"))?;
        let to = proj4rs::proj::Proj::from_user_string(to_crs)
            .map_err(|e| format!("proj4rs cannot parse {to_crs:?}: {e}"))?;
        let from_is_latlong = from.is_latlong();
        let to_is_latlong = to.is_latlong();
        Ok(Self { from, to, from_is_latlong, to_is_latlong })
    }
}

#[cfg(feature = "proj4rs")]
impl CrsTransform for Proj4rsBackend {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let mut p = if self.from_is_latlong {
            (x.to_radians(), y.to_radians(), 0.0)
        } else {
            (x, y, 0.0)
        };
        proj4rs::transform::transform(&self.from, &self.to, &mut p).ok()?;
        let (mut ox, mut oy) = (p.0, p.1);
        if self.to_is_latlong {
            ox = ox.to_degrees();
            oy = oy.to_degrees();
        }
        if ox.is_finite() && oy.is_finite() {
            Some((ox, oy))
        } else {
            None
        }
    }
}

#[cfg(all(test, feature = "proj4rs"))]
mod tests {
    use super::*;

    const HOBART: (f64, f64) = (147.3257, -42.8826);

    #[test]
    fn epsg_4326_to_3857_matches_known_value() {
        let t = Proj4rsBackend::new("EPSG:4326", "EPSG:3857").unwrap();
        let (x, y) = t.convert(HOBART.0, HOBART.1).unwrap();
        // Reference: analytic spherical Mercator, R = 6378137
        assert!((x - 16400221.9).abs() < 1.0, "x = {x}");
        assert!((y - -5294119.4).abs() < 1.0, "y = {y}");
    }

    #[test]
    fn laea_roundtrip_via_3857() {
        // A local LAEA centred on Tasmania: the slippy-map target CRS.
        let laea = "+proj=laea +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";
        let fwd = Proj4rsBackend::new("EPSG:3857", laea).unwrap();
        let inv = Proj4rsBackend::new(laea, "EPSG:3857").unwrap();
        let src = Proj4rsBackend::new("EPSG:4326", "EPSG:3857").unwrap();

        let (mx, my) = src.convert(HOBART.0, HOBART.1).unwrap();
        let (lx, ly) = fwd.convert(mx, my).unwrap();
        // Hobart is ~27 km east, ~98 km south of the LAEA origin.
        assert!((lx - 26_700.0).abs() < 500.0, "laea x = {lx}");
        assert!((ly - -98_000.0).abs() < 500.0, "laea y = {ly}");

        let (bx, by) = inv.convert(lx, ly).unwrap();
        assert!((bx - mx).abs() < 1e-3 && (by - my).abs() < 1e-3);
    }

    #[test]
    fn ortho_matches_proj_and_roundtrips() {
        // Reference: PROJ 9.8.1 via pyproj, same definition.
        let ortho = "+proj=ortho +lat_0=-42 +lon_0=147 +datum=WGS84 +units=m +no_defs";
        let fwd = Proj4rsBackend::new("EPSG:4326", ortho).unwrap();
        let inv = Proj4rsBackend::new(ortho, "EPSG:4326").unwrap();
        let (x, y) = fwd.convert(HOBART.0, HOBART.1).unwrap();
        assert!((x - 26608.2567).abs() < 1e-3, "ortho x = {x}");
        assert!((y - -98087.5790).abs() < 1e-3, "ortho y = {y}");
        let (lon, lat) = inv.convert(x, y).unwrap();
        assert!((lon - HOBART.0).abs() < 1e-9 && (lat - HOBART.1).abs() < 1e-9);
        // The far side of the globe is not visible.
        let far = fwd.convert(-33.0, 42.0);
        assert!(far.is_none(), "far side = {far:?}");
    }

    #[test]
    fn failure_is_none_not_panic() {
        let t = Proj4rsBackend::new("EPSG:4326", "EPSG:3857").unwrap();
        // Mercator is undefined at the pole.
        assert!(t.convert(0.0, 90.0).is_none() || t.convert(0.0, 90.0).unwrap().1.is_finite());
        assert!(t.convert(f64::NAN, 0.0).is_none());
    }

    #[test]
    fn bare_proj_string_gets_type_crs() {
        assert_eq!(ensure_crs("+proj=laea +lat_0=-42"), "+proj=laea +lat_0=-42 +type=crs");
        assert_eq!(ensure_crs("EPSG:3857"), "EPSG:3857");
    }
}
