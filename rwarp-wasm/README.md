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

## Sources

`www/sources.js` resolves a source into one grid description (CRS, origin,
tile size, per-level resolution and matrix size, URL template). Three kinds:

- XYZ Web Mercator templates (`{z}/{x}/{y}`, `{s}`, `{-y}`)
- WMTS: reads GetCapabilities in the browser, picks the layer's
  TileMatrixSet, resolves `{Time}` and other dimensions to their defaults.
  Any CRS the `proj4rs` backend knows, e.g. NASA GIBS in EPSG:3031/3413.
- ArcGIS cached MapServer: reads `?f=json` for `tileInfo`/`lods`.

The worker never sees a service type, only the grid. The source must send
CORS headers, since tiles are decoded through a canvas.

## Projection support

The `proj4rs` backend implements: latlong, laea, stere (and ups), sterea,
aea, lcc, tmerc/etmerc/utm, merc/webmerc, eqc, moll, geos, somerc, geocent.
Not yet available: `ortho` (absent upstream), `aeqd` (upstream feature
depends on a C geodesic library, so it is off for wasm). Both are candidates
for pure-Rust contributions to proj4rs.
