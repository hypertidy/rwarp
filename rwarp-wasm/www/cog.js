// Cloud-optimised GeoTIFF as a source. Overviews become levels; the worker
// reads a pixel window at the chosen level straight through geotiff.js
// (range requests, block cache, all codecs), colourises it to RGBA, and
// hands that to the warper like any other source.
//
// Used from both the page (resolve) and the worker (read), so keep it free
// of DOM access.

import { fromUrl } from "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";
import { colorizeSingle, colorizeRgb, percentiles } from "./colormap.js";

const open = new Map(); // url -> Promise<GeoTIFF>
export function openCog(url) {
  if (!open.has(url)) open.set(url, fromUrl(url, { cache: true, blockSize: 65536, allowFullFile: false }));
  return open.get(url);
}

function crsFromGeoKeys(gk) {
  const p = gk.ProjectedCSTypeGeoKey, g = gk.GeographicTypeGeoKey;
  if (p && p !== 32767) return `EPSG:${p}`;
  if (g && g !== 32767) return `EPSG:${g}`;
  return null;
}

/// Build the grid description for a COG. `opts.crs` overrides the file's
/// CRS (needed when the GeoKeys are user-defined); `opts.style` sets
/// colormap/min/max/bands, with "auto" min/max taken from the smallest
/// overview's 2-98 percentiles.
export async function cogSource(url, opts = {}) {
  const tiff = await openCog(url);
  const n = await tiff.getImageCount();
  const img0 = await tiff.getImage(0);
  const gk = img0.getGeoKeys() || {};
  const crs = opts.crs || crsFromGeoKeys(gk);
  if (!crs) throw new Error("COG has a user-defined CRS; give an explicit source CRS");
  const [ox, oy] = img0.getOrigin();
  const [rx0] = img0.getResolution();
  const nodata = img0.getGDALNoData();
  const bands = img0.getSamplesPerPixel();
  const format = img0.getSampleFormat();      // 1 uint, 2 int, 3 float
  const bits = img0.getBitsPerSample();
  const geographic = !!(gk.GeographicTypeGeoKey && !gk.ProjectedCSTypeGeoKey) || /^EPSG:4326$/.test(crs);

  const levels = [];
  for (let i = 0; i < n; i++) {
    const im = await tiff.getImage(i);
    const [rx] = im.getResolution(img0);
    const tw = im.getTileWidth(), th = im.getTileHeight();
    levels.push({
      id: String(i), res: Math.abs(rx), width: im.getWidth(), height: im.getHeight(),
      mw: Math.ceil(im.getWidth() / tw), mh: Math.ceil(im.getHeight() / th), tileW: tw, tileH: th,
    });
  }
  levels.sort((a, b) => b.res - a.res);

  // Style: RGB pass-through for 3/4-band u8, otherwise single band + colormap.
  const style = Object.assign({ colormap: "viridis", min: "auto", max: "auto", bands: null }, opts.style || {});
  const isU8 = format === 1 && bits === 8;
  if (!style.bands) style.bands = (bands >= 3) ? [0, 1, 2] : [0];
  if (style.min === "auto" || style.max === "auto") {
    if (isU8 && style.bands.length === 3) {
      style.min = style.min === "auto" ? 0 : style.min;
      style.max = style.max === "auto" ? 255 : style.max;
    } else {
      // Smallest overview, whole image, capped: percentiles of the first styled band.
      const small = await tiff.getImage(n - 1);
      const w = Math.min(small.getWidth(), 1024), h = Math.min(small.getHeight(), 1024);
      const [arr] = await small.readRasters({ window: [0, 0, w, h], samples: [style.bands[0]] });
      const [lo, hi] = percentiles(arr, nodata);
      style.min = style.min === "auto" ? lo : style.min;
      style.max = style.max === "auto" ? hi : style.max;
    }
  }

  return {
    kind: "cog", name: `${url.split("/").pop()} (${crs}, ${bands} band${bands > 1 ? "s" : ""}, ${fmtName(format, bits)})`,
    url, crs, origin: [ox, oy], tileW: levels[0].tileW, tileH: levels[0].tileH, levels,
    ground: geographic ? "geographic" : (crs === "EPSG:3857" ? "mercator" : "flat"),
    nodata: nodata === undefined ? null : nodata, style, dtype: fmtName(format, bits), nbands: bands,
    baseRes: Math.abs(rx0),
  };
}

function fmtName(format, bits) {
  return (format === 3 ? "float" : format === 2 ? "int" : "uint") + bits;
}

/// Worker side: read `[x0, y0, x1, y1]` (pixels, level `lv`) and return RGBA.
export async function readCogWindowRgba(source, lv, x0, y0, x1, y1) {
  const tiff = await openCog(source.url);
  const im = await tiff.getImage(Number(lv.id));
  const w = x1 - x0, h = y1 - y0;
  const { style, nodata } = source;
  const rasters = await im.readRasters({ window: [x0, y0, x1, y1], samples: style.bands });
  const n = w * h;
  if (style.bands.length >= 3) return colorizeRgb(rasters, n, style, nodata);
  return colorizeSingle(rasters[0], n, style, nodata);
}
