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
