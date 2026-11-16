# Notes on rwarp Rust API compatibility

`src/rust/src/lib.rs` builds against rwarp 0.4.1. It uses
`transform::GenImgProjTransformer`, `transform::transform_grid`,
`approx::ApproxTransformer`, `plan::ChunkPlanner`, `plan::ChunkOptions`,
`plan::merge_tile`, `crs::ProjBackend`, `crs::is_geographic_degrees`,
`warp::warp`, `warp::warp_to` and the deprecated `warp::warp_resample`. None of
these changed in rwarp 0.4, whose changes are in the mesh and weights API
(controlledburn's types), which the package doesn't use.

## proj-sys version

`crs_describe()` calls `proj-sys` directly, for `proj_as_wkt`,
`proj_as_proj_string` and `proj_as_projjson`.  Only one `proj-sys` can be in the
dependency graph, so its version must be the one rwarp uses (0.27).  rwarp turns
on `proj`'s `pkg_config` feature, so rwarp links the system PROJ that
`pkg-config` finds, the same one GDAL should be using.
