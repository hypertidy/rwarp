// Tile-warping worker. Owns the wasm module; the page only sends tile
// requests and receives ImageData back.
//
// Message in:  { id, source, level, dstCrs, dstGt, dstW, dstH, alg }
//   source: grid object from sources.js; level: index into source.levels
// Message out: { id, image: ImageData | null, info: string }

import init, { Warper } from "../pkg/rwarp_wasm.js";
import { readCogWindowRgba, readCogWindowF32 } from "./cog.js";
import { colorizeSingle } from "./colormap.js";

const MAX_SOURCE_TILES = 64;   // refuse tiles that would need more than this
const MAX_COG_PIXELS = 4 << 20; // and COG windows larger than this
const CACHE_MAX = 400;         // decoded source tiles kept in memory

const ready = init();
const cache = new Map();       // url -> Uint8ClampedArray

const SUBDOMAINS = ["a", "b", "c"];
function tileUrl(template, z, x, y, mh) {
  return template
    .replace("{s}", SUBDOMAINS[(x + y) % SUBDOMAINS.length])
    .split("{TileMatrix}").join(z).split("{TileCol}").join(x).split("{TileRow}").join(y)
    .replace("{z}", z).replace("{x}", x).replace("{y}", y)
    .replace("{-y}", mh - 1 - y);
}

async function fetchTileRgba(url, w, h) {
  const hit = cache.get(url);
  if (hit) { cache.delete(url); cache.set(url, hit); return hit; }
  const resp = await fetch(url, { mode: "cors" });
  if (!resp.ok) throw new Error(`${resp.status} ${url}`);
  const blob = await resp.blob();
  const bmp = await createImageBitmap(blob);
  const cv = new OffscreenCanvas(w, h);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const data = ctx.getImageData(0, 0, w, h).data;
  cache.set(url, data);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  return data;
}

async function warpTile(msg) {
  const { source, level, dstCrs, dstGt, dstW, dstH, alg } = msg;
  const lv = source.levels[level];
  const tw = lv.tileW || source.tileW, th = lv.tileH || source.tileH;
  const srcGt = [source.origin[0], lv.res, 0, source.origin[1], 0, -lv.res];
  const srcW = lv.width || lv.mw * tw, srcH = lv.height || lv.mh * th;

  const warper = new Warper(source.crs, Float64Array.from(srcGt), dstCrs, Float64Array.from(dstGt), 0.125);
  try {
    const win = warper.source_window(dstW, dstH, srcW, srcH, 1);
    if (!win) return { image: null, info: "no source window" };
    const [xoff, yoff, xsize, ysize] = win;

    if (source.kind === "cog") {
      // Read the window straight from the file at this overview level.
      const x0 = Math.max(0, xoff), y0 = Math.max(0, yoff);
      const x1 = Math.min(lv.width, xoff + xsize), y1 = Math.min(lv.height, yoff + ysize);
      if (x1 <= x0 || y1 <= y0) return { image: null, info: "window outside source" };
      if ((x1 - x0) * (y1 - y0) > MAX_COG_PIXELS) {
        return { image: null, info: `window ${x1 - x0}x${y1 - y0} px too large, skipping` };
      }
      const t0 = performance.now();
      let out, t1;
      if (source.style.bands.length >= 3) {
        // RGB: colour is the data; warp the u8 composite.
        const rgba = await readCogWindowRgba(source, lv, x0, y0, x1, y1);
        t1 = performance.now();
        out = warper.warp_rgba(rgba, x1 - x0, y1 - y0, x0, y0, dstW, dstH, alg);
      } else {
        // Single band: warp the values, then colourise the result, so the
        // kernel interpolates data rather than colours and nodata stays exact.
        const vals = await readCogWindowF32(source, lv, x0, y0, x1, y1);
        t1 = performance.now();
        const nd = source.nodata === null ? NaN : source.nodata;
        const warped = warper.warp_f32(vals, x1 - x0, y1 - y0, x0, y0, dstW, dstH, nd, alg);
        out = colorizeSingle(warped, dstW * dstH, source.style, source.nodata);
      }
      const ms = (performance.now() - t1).toFixed(1);
      const image = new ImageData(new Uint8ClampedArray(out.buffer, out.byteOffset, out.length), dstW, dstH);
      return { image, info: `ovr${lv.id} ${x1 - x0}x${y1 - y0} px, read ${(t1 - t0).toFixed(0)} ms, warp ${ms} ms` };
    }

    const tx0 = Math.max(0, Math.floor(xoff / tw));
    const ty0 = Math.max(0, Math.floor(yoff / th));
    const tx1 = Math.min(lv.mw - 1, Math.floor((xoff + xsize - 1) / tw));
    const ty1 = Math.min(lv.mh - 1, Math.floor((yoff + ysize - 1) / th));
    const nx = tx1 - tx0 + 1, ny = ty1 - ty0 + 1;
    if (nx <= 0 || ny <= 0) return { image: null, info: "window outside source" };
    if (nx * ny > MAX_SOURCE_TILES) {
      return { image: null, info: `needs ${nx * ny} source tiles, skipping` };
    }

    const urls = [];
    for (let ty = ty0; ty <= ty1; ty++)
      for (let tx = tx0; tx <= tx1; tx++)
        urls.push({ tx, ty, url: tileUrl(source.template, lv.id, tx, ty, lv.mh) });
    const tiles = await Promise.all(urls.map(u => fetchTileRgba(u.url, tw, th).catch(() => null)));
    const got = tiles.filter(Boolean).length;

    const W = nx * tw, H = ny * th;
    const buf = new Uint8Array(W * H * 4);
    tiles.forEach((t, i) => {
      if (!t) return;
      const ox = (urls[i].tx - tx0) * tw, oy = (urls[i].ty - ty0) * th;
      for (let r = 0; r < th; r++) {
        buf.set(t.subarray(r * tw * 4, (r + 1) * tw * 4), ((oy + r) * W + ox) * 4);
      }
    });

    const t0 = performance.now();
    const out = warper.warp_rgba(buf, W, H, tx0 * tw, ty0 * th, dstW, dstH, alg);
    const ms = (performance.now() - t0).toFixed(1);
    const image = new ImageData(new Uint8ClampedArray(out.buffer, out.byteOffset, out.length), dstW, dstH);
    const miss = got < urls.length ? ` (${urls.length - got} missing)` : "";
    return { image, info: `L${lv.id} ${nx}x${ny} src tiles${miss}, warp ${ms} ms` };
  } finally {
    warper.free();
  }
}

self.onmessage = async (e) => {
  await ready;
  const msg = e.data;
  try {
    const { image, info } = await warpTile(msg);
    self.postMessage({ id: msg.id, image, info }, image ? [image.data.buffer] : []);
  } catch (err) {
    self.postMessage({ id: msg.id, image: null, info: String(err) });
  }
};
