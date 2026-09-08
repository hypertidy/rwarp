# rwarp

Rust implementation of the GDAL warp pipeline: coordinate transforms,
approximate transforms, source window planning, and resampling kernels.

See the [crate documentation](https://docs.rs/rwarp) for full details.

## CRS backends

The CRS-to-CRS step is behind the `CrsTransform` trait (`rwarp::crs`), selected
by Cargo feature:

- `proj` (default): the `proj` crate over libproj. Full PROJ: WKT, proj.db,
  datum grids. Needs a native libproj.
- `proj4rs`: pure Rust. Proj strings and `EPSG:NNNN` codes only. No C
  dependencies; builds for `wasm32-unknown-unknown`.

```sh
cargo test                                          # libproj backend
cargo test --no-default-features --features proj4rs # pure Rust
cargo build --no-default-features --features proj4rs --target wasm32-unknown-unknown
```

`GenImgProjTransformer::with_backend` accepts any `CrsTransform` pair, so a
caller can supply its own (for example a JavaScript PROJ build reached from
wasm) without either feature.
