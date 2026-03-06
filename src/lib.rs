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
//! rwarp is **pure Rust with no runtime GDAL dependency**. It uses the
//! [`proj`] crate for coordinate reference system transforms and
//! [`vaster`] for grid arithmetic. The algorithms are independently
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
//! [plan]  compute_source_window / collect_chunk_list
//!       │   → which source pixels does each target chunk need?
//!       │   → the plan is inspectable, serializable, resumable
//!       │
//!       ▼
//! [read]  (caller: fetch bytes from COG / Zarr / parquet byte-ref store)
//!       │
//!       ▼
//! [warp]  warp_resample
//!         → nearest / bilinear / cubic / lanczos kernel
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
//! | [`approx`] | `ApproxTransformer` (adaptive interpolation) | `GDALApproxTransform` |
//! | [`source_window`] | `ComputeSourceWindow`, `CollectChunkList` | `GDALWarpOperation` planning |
//! | [`warp`] | Resampling kernels | `gdalwarpkernel.cpp` |
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
//! | Antimeridian split-read | GEBCO Fiji tile \[5,3\] | 70× source pixel reduction vs naive read |
//!
//! **Known limitation — downsampled interpolated warps:** when the
//! source/destination pixel ratio is significantly greater than 1 (coarser
//! output than source), GDAL scales the resampling kernel to cover the
//! appropriate source area (antialiasing). rwarp's kernels use the textbook
//! fixed width. At ~1:1 ratio results are bit-identical; at 4.5:1 ratio
//! bilinear outputs differ noticeably. The practical solution is overview
//! pre-selection: the planning layer knows the source/destination ratio and
//! can select the appropriate overview level before calling the kernel. See
//! the [`warp`] module documentation for details.
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
//!   `warp_resample` unchanged.
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
//! 2026-02-23). GDAL is MIT-licensed. This crate is an independent
//! implementation, not a port. We
//! think it's also very useful as a way to investigate and carry forward the value
//! of the existing GDAL C++ implementation which has years of battle-tested logic
//! within.
//!
//! | rwarp | GDAL source file | Lines |
//! |-------|-----------------|-------|
//! | `transform.rs` | `gdaltransformer.cpp` | ~L2800, ~L3100, ~L3500 |
//! | `gcp_transform.rs` | `alg/gdal_crs.c` | GCP polynomial fit |
//! | `approx.rs` | `gdaltransformer.cpp` | ~L4113, ~L4374 |
//! | `source_window.rs` | `gdalwarpoperation.cpp` | ~L1456, ~L2656, ~L2751 |
//! | `warp.rs` | `gdalwarpkernel.cpp` | ~L3084, ~L3262, ~L3655, ~L5510 |

pub mod transform;
pub mod gcp_transform;
pub mod approx;
pub mod source_window;
pub mod warp;

// Re-export the Transformer trait at the crate root so modules can
// `use crate::Transformer` without reaching into transform::
pub use transform::Transformer;
