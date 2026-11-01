# Notes on rwarp Rust API compatibility

`src/rust/src/lib.rs` is written against rwarp 0.3.1: `transform::GenImgProjTransformer`,
`transform::transform_grid`, `approx::ApproxTransformer`, `plan::ChunkPlanner`
and `warp::warp_resample`.

## proj-sys version

`crs_describe()` calls `proj-sys` directly, for `proj_as_wkt`,
`proj_as_proj_string` and `proj_as_projjson`.  Only one `proj-sys` can be in the
dependency graph, so its version must be the one rwarp uses (0.27).  rwarp turns
on `proj`'s `pkg_config` feature, so rwarp links the system PROJ that
`pkg-config` finds, the same one GDAL should be using.
