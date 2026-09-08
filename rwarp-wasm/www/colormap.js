// Colour mapping for single-band and multi-band numeric rasters, following
// the rio-tiler / titiler / deck.gl-raster convention: rescale (min, max)
// to 0..1, then a 256-entry colormap LUT; nodata and NaN become alpha 0.
//
// Viridis-family maps use Matt Zucker's degree-6 polynomial fits
// (https://www.shadertoy.com/view/WlfXRN), so no colormap assets are needed.

function poly(c0, c1, c2, c3, c4, c5, c6) {
  return (t) => {
    const r = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      r[i] = c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i])))));
    }
    return r;
  };
}

const FUNCS = {
  viridis: poly(
    [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
    [0.1050930431085774, 1.404613529898575, 1.384590162594685],
    [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
    [-4.634230498983486, -5.799100973351585, -19.33244095627987],
    [6.228269936347081, 14.17993336680509, 56.69055260068105],
    [4.776384997670288, -13.74514537774601, -65.35303263337234],
    [-5.435455855934631, 4.645852612178535, 26.3124352495832]),
  magma: poly(
    [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
    [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
    [8.353717279216625, -3.577719514958484, 0.3144679030132573],
    [-27.66873308576866, 14.26473078096533, -13.68929312895104],
    [52.17613981234068, -27.94360607168351, 12.94416944238394],
    [-50.76852536473588, 29.04658282127291, 4.23415299384598],
    [18.65570506591883, -11.48977351997711, -5.601961508734096]),
  plasma: poly(
    [0.05873234392399702, 0.02333670892565664, 0.5433401826748754],
    [2.176514634195958, 0.2383834171260182, 0.7539604599784036],
    [-2.689460476458034, -7.455851135738909, 3.110799939717086],
    [6.130348345893603, 42.3461881477227, -28.51885465332158],
    [-11.10743619062271, -82.66631109428045, 60.13984767418263],
    [10.02306557647065, 71.41361770095349, -54.07218655560067],
    [-3.658713842777788, -22.93153465461149, 18.19190778539828]),
  inferno: poly(
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172]),
  gray: (t) => [t, t, t],
  // Diverging blue-white-red.
  rdbu: (t) => t < 0.5
    ? [0.13 + 0.87 * (t * 2), 0.4 + 0.6 * (t * 2), 0.67 + 0.33 * (t * 2)]
    : [1.0, 1.0 - 0.6 * ((t - 0.5) * 2), 1.0 - 0.8 * ((t - 0.5) * 2)],
  // Simple hypsometric tint for elevation.
  terrain: (t) => {
    const stops = [[0.20, 0.60, 0.70], [0.36, 0.72, 0.36], [0.85, 0.85, 0.55], [0.60, 0.45, 0.30], [0.95, 0.95, 0.95]];
    const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    return stops[i].map((a, k) => a + (stops[i + 1][k] - a) * f);
  },
};

export const COLORMAPS = Object.keys(FUNCS);

const lutCache = new Map();
export function lut(name) {
  if (lutCache.has(name)) return lutCache.get(name);
  const f = FUNCS[name] || FUNCS.viridis;
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = f(i / 255);
    out[i * 3] = Math.max(0, Math.min(255, Math.round(r * 255)));
    out[i * 3 + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
    out[i * 3 + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
  }
  lutCache.set(name, out);
  return out;
}

/// Single band -> RGBA via rescale + colormap. `nodata` may be null.
export function colorizeSingle(band, n, style, nodata) {
  const { min, max, colormap } = style;
  const L = lut(colormap);
  const out = new Uint8Array(n * 4);
  const scale = 255 / (max - min || 1);
  for (let i = 0; i < n; i++) {
    const v = band[i];
    if (v !== v || (nodata !== null && v === nodata)) continue; // NaN or nodata: transparent
    let k = (v - min) * scale;
    k = k < 0 ? 0 : k > 255 ? 255 : k | 0;
    out[i * 4] = L[k * 3]; out[i * 4 + 1] = L[k * 3 + 1]; out[i * 4 + 2] = L[k * 3 + 2]; out[i * 4 + 3] = 255;
  }
  return out;
}

/// Three bands -> RGBA with a shared linear rescale. u8 inputs with
/// min=0,max=255 pass through unchanged.
export function colorizeRgb(bands, n, style, nodata) {
  const { min, max } = style;
  const out = new Uint8Array(n * 4);
  const scale = 255 / (max - min || 1);
  const [r, g, b] = bands;
  for (let i = 0; i < n; i++) {
    const rv = r[i], gv = g[i], bv = b[i];
    if (rv !== rv || (nodata !== null && rv === nodata && gv === nodata && bv === nodata)) continue;
    let k;
    k = (rv - min) * scale; out[i * 4] = k < 0 ? 0 : k > 255 ? 255 : k;
    k = (gv - min) * scale; out[i * 4 + 1] = k < 0 ? 0 : k > 255 ? 255 : k;
    k = (bv - min) * scale; out[i * 4 + 2] = k < 0 ? 0 : k > 255 ? 255 : k;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/// 2nd/98th percentile of finite, non-nodata values (for an "auto" stretch).
export function percentiles(arr, nodata, lo = 0.02, hi = 0.98) {
  const vals = [];
  const step = Math.max(1, Math.floor(arr.length / 200000));
  for (let i = 0; i < arr.length; i += step) {
    const v = arr[i];
    if (v === v && (nodata === null || v !== nodata)) vals.push(v);
  }
  if (!vals.length) return [0, 1];
  vals.sort((a, b) => a - b);
  return [vals[Math.floor(lo * (vals.length - 1))], vals[Math.floor(hi * (vals.length - 1))]];
}
