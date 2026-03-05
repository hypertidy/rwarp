//! rwarp: GDAL warp pipeline in Rust
//!
//! Coordinate transforms, approximate transforms, source window computation,
//! and resampling kernels.  Independent of any I/O layer or R bindings.
//!
//! ## Modules
//!
//! - `transform` — `Transformer` trait, `GenImgProjTransformer` (GDAL `GDALGenImgProjTransform`)
//! - `gcp_transform` — `GcpTransformer` (GDAL `GDALGCPTransform`, polynomial order 1–3)
//! - `approx` — `ApproxTransformer` (GDAL `GDALApproxTransform`)
//! - `source_window` — `ComputeSourceWindow`, chunk planning (GDAL `GDALWarpOperation`)
//! - `warp` — Resampling kernels: nearest, bilinear, cubic, lanczos

pub mod transform;
pub mod gcp_transform;
pub mod approx;
pub mod source_window;
pub mod warp;

// Re-export the Transformer trait at the crate root so modules can
// `use crate::Transformer` without reaching into transform::
pub use transform::Transformer;
