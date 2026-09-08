# rwarp-wasm

Browser shim over [rwarp](..): plan and warp one slippy-map tile from a
Web Mercator (or any) source grid into an arbitrary target CRS, entirely
client-side. Pure Rust via the `proj4rs` backend; no libproj, no server.

## Build

    cargo install wasm-pack        # once
    wasm-pack build --target web --release
    # -> pkg/rwarp_wasm.js, pkg/rwarp_wasm_bg.wasm

Native tests of the core (no wasm toolchain needed):

    cargo test

## Demo

`www/index.html` is a Leaflet map in a local LAEA whose tiles are warped
from OpenStreetMap in a Web Worker. Serve the crate directory (wasm needs
http, not file://) after building:

    python3 -m http.server 8000
    # open http://localhost:8000/www/

## API (JavaScript)

    const w = new Warper(srcCrs, srcGt, dstCrs, dstGt, maxError);
    const win = w.source_window(dstW, dstH, srcW, srcH, padding);
    // -> Int32Array [xoff, yoff, xsize, ysize] in source pixels, or undefined
    const rgba = w.warp_rgba(srcRgba, srcW, srcH, xoff, yoff, dstW, dstH, "nearest");
    // -> Uint8Array, dstW*dstH*4, unmapped pixels transparent

Geotransforms are GDAL order `[x0, dx, rx, y0, ry, dy]`. CRS strings are
proj strings (`+proj=laea +lat_0=-42 +lon_0=147 ...`) or `EPSG:NNNN`.
