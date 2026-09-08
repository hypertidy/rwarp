// Source tile grids. Every source resolves to the same plain object, which
// is what the worker consumes:
//
//   {
//     name, crs,                  // crs: "EPSG:NNNN" or proj string
//     origin: [x, y],             // top-left corner of the grid, CRS units
//     tileW, tileH,               // tile size in pixels
//     levels: [{ id, res, mw, mh }],  // id as used in the URL; res = CRS units per pixel
//     template,                   // URL with {z}|{TileMatrix} {x}|{TileCol} {y}|{TileRow} {s} {-y}
//     ground: "mercator" | "flat" // how res relates to ground metres (level picking)
//   }
//
// Web Mercator XYZ is one instance. WMTS and ArcGIS grids are read from the
// service itself, so any WMTS (polar or otherwise) works without hard-coding.

const WEBMERC_HALF = 20037508.342789244;

export function xyzWebMercator(template, maxZoom = 19, name = "XYZ") {
  const levels = [];
  for (let z = 0; z <= maxZoom; z++) {
    const n = Math.pow(2, z);
    levels.push({ id: String(z), res: (2 * WEBMERC_HALF) / (256 * n), mw: n, mh: n });
  }
  return {
    name, crs: "EPSG:3857", origin: [-WEBMERC_HALF, WEBMERC_HALF],
    tileW: 256, tileH: 256, levels, template, ground: "mercator",
  };
}

// --- WMTS ------------------------------------------------------------------

function txt(el, local) {
  const n = el.getElementsByTagNameNS("*", local)[0];
  return n ? n.textContent.trim() : null;
}

/// Read a WMTS 1.0.0 capabilities document and build the grid for `layerId`.
/// With no `layerId`, a single-layer document (the ArcGIS Server case) is
/// used as-is; a multi-layer one fails with the list of identifiers.
export async function wmtsSource(capabilitiesUrl, layerId, opts = {}) {
  const resp = await fetch(capabilitiesUrl);
  if (!resp.ok) throw new Error(`capabilities ${resp.status}`);
  const xml = new DOMParser().parseFromString(await resp.text(), "text/xml");
  const layers = [...xml.getElementsByTagNameNS("*", "Layer")];
  const ids = layers.map(l => txt(l, "Identifier"));
  let layer;
  if (layerId) {
    layer = layers.find(l => txt(l, "Identifier") === layerId);
    if (!layer) throw new Error(`layer ${layerId} not in capabilities; available: ${ids.join(", ")}`);
  } else if (layers.length === 1) {
    layer = layers[0];
    layerId = ids[0];
  } else {
    throw new Error(`capabilities has ${layers.length} layers, give one: ${ids.join(", ")}`);
  }

  const tmsName = opts.tileMatrixSet
    || txt(layer.getElementsByTagNameNS("*", "TileMatrixSetLink")[0], "TileMatrixSet");
  const format = txt(layer, "Format") || "image/png";
  const style = txt(layer.getElementsByTagNameNS("*", "Style")[0] || layer, "Identifier") || "default";

  // Dimensions (Time, etc.): substitute defaults unless overridden.
  const dims = {};
  for (const d of layer.getElementsByTagNameNS("*", "Dimension")) {
    const id = txt(d, "Identifier");
    dims[id] = (opts.dimensions && opts.dimensions[id]) || txt(d, "Default") || "default";
  }

  // Tile URL: RESTful ResourceURL if present, else KVP GetTile.
  let template = null;
  for (const r of layer.getElementsByTagNameNS("*", "ResourceURL")) {
    if (r.getAttribute("resourceType") === "tile") { template = r.getAttribute("template"); break; }
  }
  if (template) {
    template = template.replace("{TileMatrixSet}", tmsName).replace("{Style}", style);
    for (const [k, v] of Object.entries(dims)) template = template.split(`{${k}}`).join(v);
  } else {
    const base = capabilitiesUrl.split("?")[0].replace(/\/1\.0\.0\/WMTSCapabilities\.xml$/, "");
    const extra = Object.entries(dims).map(([k, v]) => `&${k.toUpperCase()}=${encodeURIComponent(v)}`).join("");
    template = `${base}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=${layerId}&STYLE=${style}` +
      `&TILEMATRIXSET=${tmsName}&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=${encodeURIComponent(format)}${extra}`;
  }

  const tms = [...xml.getElementsByTagNameNS("*", "TileMatrixSet")]
    .find(t => t.parentNode.localName === "Contents" && txt(t, "Identifier") === tmsName);
  if (!tms) throw new Error(`TileMatrixSet ${tmsName} not found`);
  const crsText = txt(tms, "SupportedCRS") || "";
  const code = (crsText.match(/(\d+)\s*$/) || [])[1];
  const crs = code ? `EPSG:${code}` : crsText;
  const geographic = code === "4326";
  // WMTS: pixel = 0.28 mm; resolution = scale * 0.00028 in CRS units.
  const unitsPerMetre = geographic ? 1 / 111319.49079327358 : 1;

  let origin = null;
  const levels = [];
  for (const tm of tms.getElementsByTagNameNS("*", "TileMatrix")) {
    let [a, b] = txt(tm, "TopLeftCorner").split(/\s+/).map(Number);
    if (geographic) [a, b] = [b, a]; // WMTS gives lat lon for 4326
    origin = origin || [a, b];
    levels.push({
      id: txt(tm, "Identifier"),
      res: Number(txt(tm, "ScaleDenominator")) * 0.00028 * unitsPerMetre,
      mw: Number(txt(tm, "MatrixWidth")), mh: Number(txt(tm, "MatrixHeight")),
      tileW: Number(txt(tm, "TileWidth")), tileH: Number(txt(tm, "TileHeight")),
    });
  }
  levels.sort((p, q) => q.res - p.res);
  return {
    name: `${layerId} (${tmsName}, ${crs})`, crs, origin,
    tileW: levels[0].tileW, tileH: levels[0].tileH, levels, template,
    ground: code === "3857" ? "mercator" : "flat",
  };
}

// --- ArcGIS REST tile cache ------------------------------------------------

export async function arcgisSource(serviceUrl) {
  const base = serviceUrl.replace(/\/+$/, "");
  const resp = await fetch(`${base}?f=json`);
  if (!resp.ok) throw new Error(`service ${resp.status}`);
  const j = await resp.json();
  if (!j.tileInfo) throw new Error("service has no tileInfo (not a cached map service)");
  const ti = j.tileInfo;
  const sr = ti.spatialReference || {};
  const code = sr.latestWkid || sr.wkid;
  const crs = code === 102100 ? "EPSG:3857" : `EPSG:${code}`;
  const ext = j.fullExtent;
  const levels = ti.lods.map(l => ({
    id: String(l.level), res: l.resolution,
    mw: Math.ceil((ext.xmax - ti.origin.x) / (l.resolution * ti.cols)),
    mh: Math.ceil((ti.origin.y - ext.ymin) / (l.resolution * ti.rows)),
  }));
  return {
    name: `${j.mapName || "ArcGIS"} (${crs})`, crs, origin: [ti.origin.x, ti.origin.y],
    tileW: ti.cols, tileH: ti.rows, levels, template: `${base}/tile/{z}/{y}/{x}`,
    ground: crs === "EPSG:3857" ? "mercator" : "flat",
  };
}

// --- Presets ---------------------------------------------------------------

/// Guess the source type from the URL shape: a WMTS capabilities document,
/// an XYZ template (has {z}), or an ArcGIS MapServer (anything else).
export function specFromUrl(url, arg = "") {
  if (/WMTSCapabilities\.xml/i.test(url) || /SERVICE=WMTS/i.test(url)) return { type: "wmts", url, layer: arg };
  if (/\{z\}|\{TileMatrix\}/.test(url)) return { type: "xyz", url, maxZoom: Number(arg) || 19 };
  return { type: "arcgis", url: url.replace(/\/WMTS.*$/, "") };
}

export const PRESETS = {
  esri_imagery_wmts: { type: "wmts", url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml", layer: "" },
  osm: { type: "xyz", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", maxZoom: 19 },
  carto_light: { type: "xyz", url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", maxZoom: 19 },
  gibs_3031_bluemarble: {
    type: "wmts", url: "https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/1.0.0/WMTSCapabilities.xml",
    layer: "BlueMarble_NextGeneration",
  },
  gibs_3031_modis: {
    type: "wmts", url: "https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/1.0.0/WMTSCapabilities.xml",
    layer: "MODIS_Terra_CorrectedReflectance_TrueColor",
  },
  gibs_3413_bluemarble: {
    type: "wmts", url: "https://gibs.earthdata.nasa.gov/wmts/epsg3413/best/1.0.0/WMTSCapabilities.xml",
    layer: "BlueMarble_NextGeneration",
  },
  esri_antarctic: { type: "arcgis", url: "https://services.arcgisonline.com/arcgis/rest/services/Polar/Antarctic_Imagery/MapServer" },
  esri_arctic: { type: "arcgis", url: "https://services.arcgisonline.com/arcgis/rest/services/Polar/Arctic_Imagery/MapServer" },
};

export async function resolveSource(spec) {
  switch (spec.type) {
    case "xyz": return xyzWebMercator(spec.url, spec.maxZoom, spec.url);
    case "wmts": return wmtsSource(spec.url, spec.layer, { dimensions: spec.dimensions });
    case "arcgis": return arcgisSource(spec.url);
    default: throw new Error(`unknown source type ${spec.type}`);
  }
}
