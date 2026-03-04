//! rwarp: GDAL warp pipeline in Rust
//!
//! Coordinate transforms, approximate transforms, source window computation,
//! and resampling kernels.  Independent of any I/O layer or R bindings.
//!
//! ## Modules
//!
//! - `transform` — `Transformer` trait, `GenImgProjTransformer` (GDAL `GDALGenImgProjTransform`)
//! - `approx` — `ApproxTransformer` (GDAL `GDALApproxTransform`)
//! - `source_window` — `ComputeSourceWindow`, chunk planning (GDAL `GDALWarpOperation`)
//! - `warp` — Resampling kernels: nearest, bilinear, cubic, lanczos

pub mod transform;
pub mod approx;
pub mod source_window;
pub mod warp;
