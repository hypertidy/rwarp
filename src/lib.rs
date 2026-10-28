#![doc = concat!(
    "> ⚠️ **v", env!("CARGO_PKG_VERSION"), " — early development release.** ",
    "This crate closely follows GDAL's warp algorithms but is an independent ",
    "implementation — it is not affiliated with or endorsed by the GDAL project. ",
    "The logic is validated against GDAL output in the [cogcache](https://github.com/rgdal-dev/cogcache) ",
    "package but has not been formally verified or tested in production. ",
    "This crate will change significantly. Feedback and testing are *very* welcome.\n\n",
    "---",
)]

//! # rwarp
//!
//! A Rust implementation of the GDAL warp pipeline: coordinate transforms,
//! approximate transforms, source window planning, and resampling kernels.
//!
//! rwarp is **pure Rust with no runtime GDAL dependency**. Coordinate
//! reference system transforms go through the [`crs::CrsTransform`] trait,
//! with two backends selected by Cargo feature: `proj` (default; the `proj`
//! crate over libproj, full PROJ) or `proj4rs` (pure Rust, proj-string and
//! EPSG-code parametric, builds for `wasm32-unknown-unknown`). Grid
//! arithmetic uses [`vaster`]. The algorithms are independently
//! implemented with reference to GDAL's source (commit `a3b7b01d3e`,
//! GDAL 3.13.0dev), not translated from it.
//!
//! ## Design philosophy
//!
//! GDAL's `GDALWarpOperation` bundles planning, I/O, coordinate mapping,
//! and resampling into a single pipeline. It's not easy for casual users
//! to determine what will be read and to plan out a pipeline. rwarp separates
//! these concerns:
//!
//! ```text
//! target grid spec
//!       │
//!       ▼
//! [plan]  plan_chunks / compute_source_window
//!       │   → which chunks, which source pixels does each need,
//!       │     at which kernel scale (gdalwarp's chunking)?
//!       │   → the plan is inspectable, serializable, resumable
//!       │
//!       ▼
//! [read]  (caller: fetch bytes from COG / Zarr / parquet byte-ref store)
//!       │
//!       ▼
//! [warp]  warp
//!         → nearest / bilinear / cubic / cubicspline / lanczos kernels,
//!           average / mode / min / max / med / q1 / q3 / rms area methods
//! ```
//!
//! Before touching any pixels you can inspect
//! how many source blocks each target tile needs, detect antimeridian or
//! pole problems, compare target grid specs, and estimate total I/O cost.
//!
//! ## Modules
//!
//! | Module | Responsibility | GDAL equivalent |
//! |--------|---------------|-----------------|
//! | [`transform`] | `Transformer` trait, `GenImgProjTransformer` | `GDALGenImgProjTransform` |
//! | [`gcp_transform`] | `GcpTransformer` (polynomial order 1–3) | `GDALGCPTransform` |
//! | [`geoloc_transform`] | `GeolocArrays`, `GenImgGeolocTransformer` (curvilinear source and/or target) | `GDALGeoLocTransform`, `GDALGenImgProjTransform` with geolocation arrays |
//! | [`approx`] | `ApproxTransformer` (adaptive interpolation) | `GDALApproxTransform` |
//! | [`source_window`] | `compute_source_window` (one window, quick) | `ComputeSourceWindow` |
//! | [`plan`] | `plan_chunks`, `ChunkPlanner`, `warp_chunks`: gdalwarp's chunk list, source windows and per-chunk kernel scale as values | `GDALWarpOperation::CollectChunkList`, `ComputeSourceWindow` |
//! | [`cutline`] | `Cutline`: polygon limiting the source pixels used, as a per-pixel density | `GDALWarpCutlineMaskerEx`, `TransformCutlineToSource` |
//! | [`warp`] | Resampling kernels (one-pass generic `Sample` kernel with GDAL's downsampling widening, area-weighted average and mode, and a per-band `i32` point-kernel reference) | `gdalwarpkernel.cpp` |
//!
//! ## Validation status
//!
//! It is early days for this tool, we have done some validation but much more is needed.
//!
//! Components that have been validated against GDAL output on real data:
//!
//! | Component | Test data | Result |
//! |-----------|-----------|--------|
//! | `GenImgProjTransformer` | Sentinel-2 B04| 65536/65536 pixels identical to `pixel_extract` |
//! | `ApproxTransformer` | Same | bit-identical to GDAL's `GDALApproxTransform` |
//! | `ComputeSourceWindow` | GEBCO global, Fiji LCC | matches GDAL source windows |
//! | Nearest-neighbour kernel | Sentinel-2, IBCSO Mawson | 65536/65536 identical to `gdalwarp -r near` |
//! | Bilinear/cubic/lanczos | GEBCO at ~1:1 ratio | 65536/65536 identical to GDAL |
//! | Kernel widening, scale estimate | synthetic 64×64 Float32, 1.5:1 to 4:1, anisotropic, with nodata (`tests/kernels.rs`) | within 1e-6 relative of `gdalwarp` (GDAL 3.14.0dev-739fcf03bf) |
//! | Point kernels at any ratio | same, against `gdalwarp -wo XSCALE=1 -wo YSCALE=1` | within 1e-6 relative |
//! | Average (GDAL box) | same, plus upsampling, overhanging and 30°-rotated destinations | within 1e-6 relative of `gdalwarp -r average` |
//! | Average (exact footprint) | axis-aligned cases vs the GDAL box; 45° diamond vs the analytic mean | within 1e-5 (f32 coverage fractions) |
//! | Mode, all `MODE_TIES` | synthetic categorical, Byte/Int16/Int32/Float32, 4:1 and 2.78:1, nodata, rotated | identical to `gdalwarp -r mode` (26 cases); ties at first occurrence follow GDAL's Byte path for every type |
//! | Nodata: point kernels, `UNIFIED_SRC_NODATA` unset/YES/NO over three bands, average thresholds and excluded values | synthetic Float32 single- and three-band (near, bilinear, cubic, lanczos, average, mode) | within 1e-6 relative of `gdalwarp` (30 cases) |
//! | Cubic spline; min, max, med, q1, q3, rms; Int64, UInt64, Float16 | synthetic Float32/Int16/Int64/UInt64/Float16, point and widened, nodata, rotated, three bands | within 1e-6 relative (integers exact, Float16 within one step) of `gdalwarp` (50 cases, plus 18 more mode cases) |
//! | Integer rounding/clamping, nodata avoidance, `[0, n)` source bounds | synthetic Byte/Int16/Int32/Float32 with .5 means, cubic overshoot, means equal to nodata, edge-straddling grids | identical to `gdalwarp` (12 cases) |
//! | Antimeridian split-read | GEBCO Fiji tile \[5,3\] | 70× source pixel reduction vs naive read |
//! | Chunk planning, whole-raster chunked warp (`plan_chunks`, `warp_chunks`) | lon/lat sources to Mercator, polar stereographic and LAEA under `-wm 100000` or the default, MEM and tiled targets, near/bilinear/cubic/lanczos (`tests/chunks.rs`) | chunk lists and every output pixel identical to `gdalwarp` (GDAL 3.14.0dev-739fcf03bf), 16 cases of 1 to 28 chunks, with both CRS backends |
//! | Longitude wrapping of a geographic source (`CENTER_LONG`) | sources across 180 degrees stored as 170..190, -190..-170, 140..200 and 0..360, and one wider than 360 under a `-te` across 180, to LAEA, Mercator, polar stereographic and lon/lat targets across 180 (`tests/chunks.rs`) | centre longitude as GDAL's; exact transformer within 1e-12 px of GDAL's (libproj); chunk lists and every output pixel identical to `gdalwarp` with both CRS backends, 11 cases |
//! | `GeolocArrays` / `GenImgGeolocTransformer` | synthetic curvilinear swath, both georeferencing conventions, curvilinear source and curvilinear target (`tests/geoloc.rs`) | coordinates within 1e-6 px of GDAL 3.8.4's transformer (CRS step varies slightly by PROJ version); nearest warp identical to `gdalwarp -et 0` |
//! | Cutline (`-cutline`, `CUTLINE_ALL_TOUCHED`, `-cblend`) | synthetic Float32, ties through pixel centres, holes, multipolygons, nodata, every method, rotated and LAEA to lon/lat (`tests/cutline.rs`) | within 1e-6 relative of `gdalwarp` (67 cases); differs only at centre ties on a diagonal (rounding, 5 pixels) and where GDAL's all-touched line walker burns corner-touched pixels or uses tolerances (3 cases) |
//!
//! **Downsampled interpolated warps:** when the output is coarser than the
//! source, GDAL widens the bilinear, cubic and lanczos kernels to cover the
//! source area of each output pixel (antialiasing). rwarp does the same by
//! default ([`warp::KernelScaling::WidenWhenDownsampling`]), estimating the
//! scale per call as GDAL does per chunk; [`warp::KernelScaling::Point`]
//! keeps fixed support, matching `gdalwarp -wo XSCALE=1 -wo YSCALE=1`.
//! Overview pre-selection is still worthwhile for I/O: the planning layer
//! knows the ratio and can read a coarser level. See the [`warp`] module
//! documentation.
//!
//! ## Extensibility: n-D chunks and vector-quantity warps
//!
//! The warp kernel operates on one 2D spatial slice at a time. N-dimensional
//! chunks — `(band, y, x)`, `(time, y, x)`, `(y, x, rgb)` — are handled by
//! the caller iterating over outer axes and calling `warp_resample` per slice.
//! The `Transformer` is constructed once and reused; only the spatial axes
//! are involved.
//!
//! The [`Transformer`] trait is intentionally minimal: it maps coordinate
//! arrays without knowing what the data values are. This makes it the right
//! seam for future transformer plugins:
//!
//! - **Curvilinear grids** (ROMS, ocean model output): pixel → geo via 2D
//!   lookup table rather than affine + PROJ. Implements `Transformer`,
//!   composes with `ApproxTransformer`, `compute_source_window`, and
//!   `warp_resample` unchanged. Implemented in [`geoloc_transform`], for a
//!   curvilinear source, a curvilinear target, or both.
//!
//! - **Displacement-field transformers**: pixel offsets from a vector field
//!   (DEM orthorectification, InSAR pixel offsets) applied on top of a base
//!   `GenImgProjTransformer`.
//!
//! - **Vector-quantity-preserving warps**: scalar resampling is correct for
//!   temperature, elevation, reflectance. Velocity fields `(u, v)` stored in
//!   index-grid coordinates (as ROMS does) require the vectors to be rotated
//!   by the local Jacobian of the transform at each point — otherwise
//!   reprojected arrows point in the wrong direction, silently. This is the
//!   same problem as a north arrow on a projected map: correct only at the
//!   calibration point, wrong everywhere else. A future `warp_vector()` kernel
//!   would call [`Transformer::jacobian`] per pixel to rotate `(u, v)` into
//!   the output frame. See the cogcache design docs for the full discussion.
//!
//! ## GDAL source lineage
//!
//! GDAL source commit used for analysis: `a3b7b01d3e` (GDAL 3.13.0dev,
//! 2026-02-23); kernel widening and the lanczos and nodata rules in `warp.rs`
//! were read at `739fcf03bf` (GDAL 3.14.0dev). GDAL is MIT-licensed. This crate is an independent
//! implementation, not a port. We
//! think it's also very useful as a way to investigate and carry forward the value
//! of the existing GDAL C++ implementation which has years of battle-tested logic
//! within.
//!
//! | rwarp | GDAL source file | Lines |
//! |-------|-----------------|-------|
//! | `transform.rs` | `gdaltransformer.cpp`, `ogr/ogrct.cpp` | ~L2800, ~L3100, ~L3500; `InsertCenterLong` ~L1423 and source wrap ~L2444 (at `739fcf03bf`) |
//! | `gcp_transform.rs` | `alg/gdal_crs.c` | GCP polynomial fit |
//! | `geoloc_transform.rs` | `alg/gdalgeoloc.cpp` | `PixelLineToXY`, `Transform`, `GDALInverseBilinearInterpolation` |
//! | `approx.rs` | `gdaltransformer.cpp` | ~L4113, ~L4374 |
//! | `source_window.rs` | `gdalwarpoperation.cpp` | ~L1456, ~L2656, ~L2751 |
//! | `plan.rs` | `gdalwarpoperation.cpp` | ~L977-1670, ~L2640-3390 (at `739fcf03bf`) |
//! | `cutline.rs` | `alg/gdalcutline.cpp`, `alg/llrasterize.cpp`, `apps/gdalwarp_lib.cpp` | `GDALWarpCutlineMaskerEx`, `BlendMaskGenerator`, `GDALdllImageFilledPolygon`, `TransformCutlineToSource` (at `739fcf03bf`) |
//! | `warp.rs` | `gdalwarpkernel.cpp` | ~L3084, ~L3262, ~L3655, ~L5510; widening ~L1041, ~L1092-1319, ~L4027, ~L4188, ~L4840; average ~L6845-7030, ~L7577, ~L7960; mode ~L7147-7235, ~L8115-8170; rounding/nodata/bounds ~L1587, ~L1613, ~L1856, ~L5251 (at `739fcf03bf`, 3.14.0dev) |

pub mod crs;
pub mod transform;
pub mod gcp_transform;
pub mod geoloc_transform;
pub mod approx;
pub mod source_window;
pub mod plan;
pub mod cutline;
pub mod warp;

// Re-export the Transformer trait at the crate root so modules can
// `use crate::Transformer` without reaching into transform::
pub use transform::Transformer;
pub use crs::CrsTransform;
