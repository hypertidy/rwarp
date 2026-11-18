# Notes on rwarp Rust API compatibility

`src/rust/src/lib.rs` builds against rwarp 0.5.0. It uses
`transform::GenImgProjTransformer`, `transform::transform_grid`,
`approx::ApproxTransformer`, `plan::ChunkPlanner`, `plan::ChunkOptions`,
`plan::merge_tile`, `crs::ProjBackend`, `crs::is_geographic_degrees`,
`warp::warp` (for `warp_exec()` and `warp_buffer()`), `warp::warp_to` and the
deprecated `warp::warp_resample` (for the deprecated R `warp_resample()`). None of
these changed in rwarp 0.4 or 0.5. 0.4's changes are in the mesh and weights API
(controlledburn's types); 0.5's are in the mesh module, the exact footprint and
new functions on several threads, with `warp()`'s rows reorganised for the
threads and its values the same bit for bit. The package uses none of them, and
its tests pass unchanged on 0.5.0.

## proj-sys version

`crs_describe()` calls `proj-sys` directly, for `proj_as_wkt`,
`proj_as_proj_string` and `proj_as_projjson`.  Only one `proj-sys` can be in the
dependency graph, so its version must be the one rwarp uses (0.27).  rwarp turns
on `proj`'s `pkg_config` feature, so rwarp links the system PROJ that
`pkg-config` finds, the same one GDAL should be using.
