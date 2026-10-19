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

### Bringing your own projections

`proj4rs` parses a subset of what proj4js or PROJ do. When it declines a
CRS, the page can lend rwarp its own projection library:

    const w = Warper.withTransforms(srcGt, dstGt, dstToSrc, srcToDst, maxError);

where `dstToSrc(x, y, out)` maps a destination geo coordinate to a source
geo coordinate by writing `[x, y]` into `out` (a `Float64Array` of length 2)
and returning `true`, or returns `false` for a point it cannot transform;
`srcToDst` is the inverse. For example, with proj4js:

    const d2s = proj4(dstCrs, srcCrs), s2d = proj4(srcCrs, dstCrs);
    const call = (t) => (x, y, out) => {
      try { const r = t.forward([x, y]); out[0] = r[0]; out[1] = r[1]; return isFinite(r[0]) && isFinite(r[1]); }
      catch { return false; }
    };
    const w = Warper.withTransforms(srcGt, dstGt, call(d2s), call(s2d), 0.125);

Everything downstream - the approximate transformer, the source window, the
kernels - is unchanged; the callback is made once per point the approximate
transformer actually evaluates, a few thousand per tile, so the cost is
small. `warp_rgba` and `warp_f32` work as before.

## Sources

`www/sources.js` resolves a source into one grid description (CRS, origin,
tile size, per-level resolution and matrix size, URL template). Three kinds:

- XYZ Web Mercator templates (`{z}/{x}/{y}`, `{s}`, `{-y}`)
- WMTS: reads GetCapabilities in the browser, picks the layer's
  TileMatrixSet, resolves `{Time}` and other dimensions to their defaults.
  Any CRS the `proj4rs` backend knows, e.g. NASA GIBS in EPSG:3031/3413.
- ArcGIS cached MapServer: reads `?f=json` for `tileInfo`/`lods`.

- COG: any `.tif` URL. geotiff.js (ESM from jsdelivr) does range reads and
  decoding; overviews become levels; the worker reads a pixel window at the
  chosen overview, colourises it (rescale min/max + 256-entry colormap,
  nodata/NaN transparent; 3-band u8 passes through as RGB), then warps the
  RGBA. Colormaps in `www/colormap.js` are polynomial fits, no assets.
  The file's CRS comes from GeoKeys (EPSG codes only; override in the panel
  for user-defined CRSs). Single-band data is warped as Float32 values
  (`Warper.warp_f32`, nodata and NaN respected) and colourised afterwards;
  RGB composites are warped as u8.

The worker never sees a service type, only the grid. The source must send
CORS headers, since tiles are decoded through a canvas.

## Projection support

The `proj4rs` backend implements: latlong, laea, stere (and ups), sterea,
aea, lcc, tmerc/etmerc/utm, merc/webmerc, eqc, moll, geos, somerc, geocent,
ortho. Not yet available: `aeqd` (upstream feature depends on a C geodesic
library, so it is off for wasm), a candidate for a pure-Rust contribution to
proj4rs.
