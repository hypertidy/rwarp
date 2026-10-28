# API documentation

- **This version** (built from `main` with this site):
  [api/rwarp/](api/rwarp/index.html)
- **Released versions**: [docs.rs/rwarp](https://docs.rs/rwarp)

The crate documentation goes into more detail than this guide on each
module, including the GDAL source each part follows, with file and line
references.

| Module | Contents |
|---|---|
| `transform` | `Transformer` trait, `GenImgProjTransformer`, scanline helpers |
| `approx` | `ApproxTransformer` |
| `gcp_transform` | `GcpTransformer` |
| `geoloc_transform` | `GeolocArrays`, `GenImgGeolocTransformer`, `PixelGeo` |
| `crs` | `CrsTransform` trait and the `proj` / `proj4rs` backends |
| `source_window` | `compute_source_window`, `collect_chunk_list` |
| `warp` | `warp`, `WarpOptions`, `SourceBuffer`, `Sample`, kernels |
