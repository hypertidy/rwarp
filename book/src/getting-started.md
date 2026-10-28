# Getting started

## Add the crate

```toml
[dependencies]
rwarp = "0.1"
```

The latest release on crates.io is older than this documentation; to use the
current API, depend on the repository:

```toml
[dependencies]
rwarp = { git = "https://github.com/hypertidy/rwarp" }
```

## Choose a CRS backend

| Feature | Backend | Accepts | Needs |
|---|---|---|---|
| `proj` (default) | libproj, through the `proj` crate | anything PROJ does: WKT, EPSG codes, proj strings, datum grids | a native libproj (or the crate's bundled build) |
| `proj4rs` | pure Rust | proj strings and `EPSG:NNNN` codes | nothing; builds for `wasm32-unknown-unknown` |

```sh
cargo test                                            # libproj
cargo test --no-default-features --features proj4rs   # pure Rust
cargo build --no-default-features --features proj4rs --target wasm32-unknown-unknown
```

Or bring your own: `GenImgProjTransformer::with_backend` takes any pair of
`CrsTransform` implementations (one each way), for example a JavaScript
PROJ build reached from WebAssembly.

## A first warp

This plans and warps one 256 x 256 tile in a local Lambert azimuthal
equal-area projection from a Web Mercator tile pyramid. It is
`examples/tile_laea.rs`; run it with `cargo run --example tile_laea`.

```rust,ignore
{{#include ../../examples/tile_laea.rs:12:}}
```

The four steps are the whole pipeline:

1. **Transformer.** Maps target pixels to source pixels. Wrapping it in
   `ApproxTransformer` with a 0.125 pixel error, as gdalwarp does by
   default, transforms a few points per scanline exactly and interpolates
   the rest.
2. **Plan.** `compute_source_window` finds the source pixels the tile
   needs, without reading any.
3. **Read.** Fetch that window however your data is stored.
4. **Warp.** `warp` resamples into the target, with options for the
   method, kernel scaling, nodata and more (see
   [Resampling](guide/resampling.md)).
