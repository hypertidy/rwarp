// Tile-warping worker. Owns the wasm module; the page only sends tile
// requests and receives ImageData back.
//
// Message in:  { id, srcTemplate, srcCrs, dstCrs, dstGt, dstW, dstH, alg, srcZoom }
// Message out: { id, image: ImageData | null, info: string }

import init, { Warper } from "../pkg/rwarp_wasm.js";

const WEBMERC_HALF = 20037508.342789244;
const MAX_SOURCE_TILES = 64;   // refuse tiles that would need more than this
const CACHE_MAX = 400;         // decoded source tiles kept in memory

const ready = init();
const cache = new Map();       // url -> Uint8ClampedArray (256*256*4)

const SUBDOMAINS = ["a", "b", "c"];
function tileUrl(template, z, x, y) {
  return template
    .replace("{s}", SUBDOMAINS[(x + y) % SUBDOMAINS.length])
    .replace("{z}", z).replace("{x}", x).replace("{y}", y)
    .replace("{-y}", Math.pow(2, z) - 1 - y);
}

function srcGeotransform(z) {
  const n = 256 * Math.pow(2, z);
  const px = (2 * WEBMERC_HALF) / n;
  return { gt: [-WEBMERC_HALF, px, 0, WEBMERC_HALF, 0, -px], n };
}

async function fetchTileRgba(url) {
  const hit = cache.get(url);
  if (hit) { cache.delete(url); cache.set(url, hit); return hit; }
  const resp = await fetch(url, { mode: "cors" });
  if (!resp.ok) throw new Error(`${resp.status} ${url}`);
  const blob = await resp.blob();
  const bmp = await createImageBitmap(blob);
  const cv = new OffscreenCanvas(256, 256);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, 256, 256);
  bmp.close();
  const data = ctx.getImageData(0, 0, 256, 256).data;
  cache.set(url, data);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  return data;
}

async function warpTile(msg) {
  const { srcTemplate, srcCrs, dstCrs, dstGt, dstW, dstH, alg, srcZoom } = msg;
  const { gt: srcGt, n } = srcGeotransform(srcZoom);

  const warper = new Warper(srcCrs, Float64Array.from(srcGt), dstCrs, Float64Array.from(dstGt), 0.125);
  try {
    const win = warper.source_window(dstW, dstH, n, n, 1);
    if (!win) return { image: null, info: "no source window" };
    const [xoff, yoff, xsize, ysize] = win;

    // Which source tiles cover the window?
    const ntiles = n / 256;
    const tx0 = Math.max(0, Math.floor(xoff / 256));
    const ty0 = Math.max(0, Math.floor(yoff / 256));
    const tx1 = Math.min(ntiles - 1, Math.floor((xoff + xsize - 1) / 256));
    const ty1 = Math.min(ntiles - 1, Math.floor((yoff + ysize - 1) / 256));
    const nx = tx1 - tx0 + 1, ny = ty1 - ty0 + 1;
    if (nx <= 0 || ny <= 0) return { image: null, info: "window outside source" };
    if (nx * ny > MAX_SOURCE_TILES) {
      return { image: null, info: `needs ${nx * ny} source tiles, skipping` };
    }

    // Fetch, then mosaic into one RGBA buffer.
    const urls = [];
    for (let ty = ty0; ty <= ty1; ty++)
      for (let tx = tx0; tx <= tx1; tx++)
        urls.push({ tx, ty, url: tileUrl(srcTemplate, srcZoom, tx, ty) });
    const tiles = await Promise.all(urls.map(u => fetchTileRgba(u.url).catch(() => null)));

    const W = nx * 256, H = ny * 256;
    const buf = new Uint8Array(W * H * 4);
    tiles.forEach((t, i) => {
      if (!t) return;
      const ox = (urls[i].tx - tx0) * 256, oy = (urls[i].ty - ty0) * 256;
      for (let r = 0; r < 256; r++) {
        buf.set(t.subarray(r * 1024, (r + 1) * 1024), ((oy + r) * W + ox) * 4);
      }
    });

    const t0 = performance.now();
    const out = warper.warp_rgba(buf, W, H, tx0 * 256, ty0 * 256, dstW, dstH, alg);
    const ms = (performance.now() - t0).toFixed(1);
    const image = new ImageData(new Uint8ClampedArray(out.buffer, out.byteOffset, out.length), dstW, dstH);
    return { image, info: `z${srcZoom} ${nx}x${ny} src tiles, warp ${ms} ms` };
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
