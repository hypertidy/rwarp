//! Cutlines: restrict a warp to the source pixels inside a polygon, as
//! gdalwarp's `-cutline` does, or weight each source pixel by how much of it
//! the polygon covers.
//!
//! A [`Cutline`] is polygon geometry in **source pixel/line coordinates** of
//! the full source image (column `x` spans `[x, x + 1)`, line `y` grows
//! down), which is where gdalwarp puts it before warping
//! (`TransformCutlineToSource`, apps/gdalwarp_lib.cpp). Put one on
//! [`WarpOptions::cutline`](crate::warp::WarpOptions::cutline); [`warp`](crate::warp::warp)
//! turns it into a per-source-pixel density over the source buffer only
//! (GDAL's `pafUnifiedSrcDensity`, from `GDALWarpCutlineMaskerEx`,
//! alg/gdalcutline.cpp) and the kernels use that density as described in the
//! [`warp`](crate::warp) module documentation.
//!
//! [`Cutline::from_geotransform`] takes geometry in the source CRS and
//! [`Cutline::from_transformer`] geometry in any space a [`Transformer`]
//! maps to source pixels, densified as gdalwarp densifies.
//!
//! ## Rules
//!
//! - [`CutlineRule::Centre`] (GDAL's default): a source pixel is in when its
//!   centre is inside, with GDAL's tie rules (`GDALdllImageFilledPolygon`,
//!   alg/llrasterize.cpp): a centre on a left edge is out, on a right edge
//!   in, on a top edge in, on a bottom edge out, and a hole whose top edge
//!   runs along a line of centres does not cut that line. The burn is
//!   `controlledburn`'s centre rule ([`controlledburn::BurnMode::Approx`])
//!   on the window turned through 180 degrees, plus one pass for horizontal
//!   edges along centre lines. On a diagonal edge through centres GDAL's
//!   answer is a rounding error and the two can differ.
//! - [`CutlineRule::AllTouched`] (`CUTLINE_ALL_TOUCHED=YES`): a source pixel
//!   is in when the polygon covers any of it, by more than 1e-9 of its area
//!   (exact coverage from [`controlledburn::burn`] in coverage mode). GDAL burns
//!   the centre rule plus every pixel a boundary line passes through
//!   (`GDALdllImageLineAllTouched`); the two differ only for a boundary
//!   that runs within GDAL's tolerances of a pixel wall (see the GDAL
//!   parity chapter of the book).
//! - [`CutlineRule::Exact`]: the density of a source pixel is the fraction of
//!   its area inside the polygon. Not a GDAL mode: area methods weight each
//!   pixel by it, point kernels weight their taps by it.
//!
//! [`Cutline::blend_dist`] is GDAL's `CUTLINE_BLEND_DIST` for the first two
//! rules: within that many source pixels of the boundary the density ramps
//! from 1 inside to 0 outside, 0.5 on the line.

use controlledburn::{burn, BurnOptions, Coord, Geometry, GridSpec, Polygon};

use crate::Transformer;

/// A ring of `[x, y]` points, closed or not (the last point joins the
/// first either way).
pub type Ring = Vec<[f64; 2]>;

/// How a [`Cutline`] decides each source pixel's density. See the module
/// documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CutlineRule {
    /// In when the pixel centre is inside (gdalwarp's default).
    #[default]
    Centre,
    /// In when the polygon covers any part of the pixel
    /// (`-wo CUTLINE_ALL_TOUCHED=YES`).
    AllTouched,
    /// The covered fraction of the pixel, as a weight. Not a GDAL mode.
    Exact,
}

/// Polygon geometry in source pixel/line coordinates that limits which
/// source pixels a warp uses (gdalwarp `-cutline`). See the module
/// documentation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Cutline {
    /// Polygons, each an exterior ring followed by its holes, in source
    /// pixel/line coordinates of the full source image. Several polygons
    /// are a multipolygon: their union.
    pub polygons: Vec<Vec<Ring>>,
    /// How pixels on the boundary count.
    pub rule: CutlineRule,
    /// GDAL's `CUTLINE_BLEND_DIST` (`-cblend`), in source pixels: the
    /// density ramps linearly from 1 at this distance inside the boundary
    /// to 0 at this distance outside. Zero for a hard edge. Ignored by
    /// [`CutlineRule::Exact`].
    pub blend_dist: f64,
}

/// GDAL's `SRC_DENSITY_THRESHOLD` (gdalwarpkernel.cpp): below it a source
/// pixel is not used by the point kernels.
pub(crate) const SRC_DENSITY_THRESHOLD: f64 = 1e-9;
/// GDAL's `BAND_DENSITY_THRESHOLD`: the area methods use a source pixel
/// whose density is above it.
pub(crate) const BAND_DENSITY_THRESHOLD: f64 = 1e-10;
/// Coverage within this of 0 or 1 counts as 0 or 1.
const COVER_EPS: f64 = 1e-9;

impl Cutline {
    /// A cutline with the centre rule and no blending, from polygons in
    /// source pixel/line coordinates.
    pub fn new(polygons: Vec<Vec<Ring>>) -> Self {
        Cutline { polygons, rule: CutlineRule::Centre, blend_dist: 0.0 }
    }

    /// The same cutline with another rule.
    pub fn with_rule(mut self, rule: CutlineRule) -> Self {
        self.rule = rule;
        self
    }

    /// The same cutline with a blend distance in source pixels.
    pub fn with_blend_dist(mut self, blend_dist: f64) -> Self {
        self.blend_dist = blend_dist;
        self
    }

    /// Polygons in the source CRS, mapped to source pixel/line through the
    /// inverse of the source geotransform: gdalwarp's path when the cutline
    /// is in the source raster's CRS (no densification is needed, as the
    /// map is affine).
    pub fn from_geotransform(polygons: Vec<Vec<Ring>>, src_gt: [f64; 6]) -> Result<Self, String> {
        let inv = vaster::inv_geotransform(&src_gt).ok_or_else(|| "source geotransform not invertible".to_string())?;
        let polygons = polygons
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|r| {
                        r.into_iter()
                            .map(|[x, y]| [inv[0] + x * inv[1] + y * inv[2], inv[3] + x * inv[4] + y * inv[5]])
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Ok(Cutline::new(polygons))
    }

    /// Polygons in the destination space of `transformer` (whatever its
    /// `transform(true, ..)` maps from), mapped to source pixel/line.
    ///
    /// For a cutline in a CRS, build a
    /// [`GenImgProjTransformer`](crate::transform::GenImgProjTransformer)
    /// from that CRS to the source with the identity geotransform
    /// `[0, 1, 0, 0, 0, 1]` on the cutline side, so its "pixels" are CRS
    /// coordinates; this is how gdalwarp builds its cutline transformer.
    ///
    /// Densifies as gdalwarp does by default (`GDALWARP_DENSIFY_CUTLINE=YES`):
    /// if a transformed edge is longer than one source pixel, the input is
    /// segmentized at `L / P` (the longest input edge `L` over its longest
    /// transformed edge `P`, in pixels), with OGR's rule for intermediate
    /// points, and transformed again. gdalwarp skips this step when
    /// `-cblend` is given; this helper does not know about blending and
    /// always densifies. Fails if any point fails to transform.
    pub fn from_transformer(polygons: Vec<Vec<Ring>>, transformer: &impl Transformer) -> Result<Self, String> {
        let first = transform_polygons(&polygons, transformer)?;
        let max_px = max_segment(&first);
        if max_px <= 1.0 {
            return Ok(Cutline::new(first));
        }
        let max_in = max_segment(&polygons);
        let seg = max_in / max_px;
        if !(seg > 0.0 && seg.is_finite()) {
            return Ok(Cutline::new(first));
        }
        let dense: Vec<Vec<Ring>> =
            polygons.iter().map(|p| p.iter().map(|r| segmentize(r, seg)).collect()).collect();
        Ok(Cutline::new(transform_polygons(&dense, transformer)?))
    }

    /// Bounding box of every ring, `[xmin, ymin, xmax, ymax]` in source
    /// pixel/line, or `None` if there are no points.
    pub fn envelope(&self) -> Option<[f64; 4]> {
        let mut e = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in self.polygons.iter().flatten().flatten() {
            e[0] = e[0].min(p[0]);
            e[1] = e[1].min(p[1]);
            e[2] = e[2].max(p[0]);
            e[3] = e[3].max(p[1]);
        }
        (e[0] <= e[2] && e[1] <= e[3]).then_some(e)
    }

    /// The density of each source pixel in the window `ncol` x `nrow` at
    /// `(col_off, row_off)` of the full source image, row-major: what the
    /// warp uses. 0 outside, 1 inside, fractions on a blend ramp or (for
    /// [`CutlineRule::Exact`]) on the boundary.
    ///
    /// Only the window is rasterized: the centre rule is
    /// `controlledburn::burn` in approx mode over the window's grid, the
    /// others the same in coverage mode, and blending visits only the
    /// pixels within the blend distance of each edge.
    pub fn density(&self, col_off: usize, row_off: usize, ncol: usize, nrow: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.density_into(col_off, row_off, ncol, nrow, &mut out);
        out
    }

    /// [`Cutline::density`] into a reused buffer.
    pub fn density_into(&self, col_off: usize, row_off: usize, ncol: usize, nrow: usize, out: &mut Vec<f32>) {
        out.clear();
        out.resize(ncol * nrow, 0.0);
        if ncol == 0 || nrow == 0 {
            return;
        }
        let bd = if self.rule == CutlineRule::Exact { 0.0 } else { self.blend_dist.max(0.0) };
        let Some(env) = self.envelope() else { return };
        let (x0, y0) = (col_off as f64, row_off as f64);
        let (x1, y1) = (x0 + ncol as f64, y0 + nrow as f64);
        // Far from the window: everything out (GDALWarpCutlineMaskerEx).
        if env[2] + bd < x0 || env[0] - bd > x1 || env[3] + bd < y0 || env[1] - bd > y1 {
            return;
        }
        let win = Win { col_off, row_off, ncol, nrow };
        match self.rule {
            CutlineRule::Centre => self.burn_centre(&win, out),
            CutlineRule::AllTouched | CutlineRule::Exact => self.burn_coverage(&win, out),
        }
        if bd > 0.0 {
            self.blend(&win, bd, out);
        }
    }

    /// GDAL's centre rule over the window.
    ///
    /// `controlledburn`'s approx mode takes a row whose centre line meets
    /// an edge's top end and not its bottom end (y up), and a centre on an
    /// edge as left of it. GDAL takes the line range `[top, bottom)` (y
    /// down) and a centre on an entering edge as out, on a leaving edge as
    /// in. Rotating the geometry and the window by 180 degrees, `(x, y) ->
    /// (-x, -y)`, maps one onto the other exactly, for any polygon.
    fn burn_centre(&self, w: &Win, out: &mut [f32]) {
        let Some(res) = self.burn_window(w, BurnOptions::approx()) else { return };
        for run in &res.runs {
            let (row, a, b) = w.run(run);
            out[row * w.ncol + a..row * w.ncol + b].fill(1.0);
        }
        for p in &self.polygons {
            gdal_horizontal_edges(p, w, out);
        }
    }

    /// Coverage over the window: any part (all touched) or the fraction
    /// (exact), from `controlledburn::burn` in coverage mode, which is
    /// sparse (interior runs and boundary cells), so the cost is the
    /// perimeter plus filling the output.
    fn burn_coverage(&self, w: &Win, out: &mut [f32]) {
        let Some(res) = self.burn_window(w, BurnOptions::coverage()) else { return };
        for run in &res.runs {
            let (row, a, b) = w.run(run);
            out[row * w.ncol + a..row * w.ncol + b].fill(1.0);
        }
        // Polygons of a multipolygon may share a boundary cell: their
        // fractions add.
        let all_touched = self.rule == CutlineRule::AllTouched;
        for e in &res.edges {
            let k = e.row as usize * w.ncol + (w.ncol - 1 - e.col as usize);
            out[k] = (out[k] + e.fraction).min(1.0);
        }
        for e in &res.edges {
            let k = e.row as usize * w.ncol + (w.ncol - 1 - e.col as usize);
            let c = out[k] as f64;
            out[k] = if c <= COVER_EPS {
                0.0
            } else if all_touched || c >= 1.0 - COVER_EPS {
                1.0
            } else {
                out[k]
            };
        }
    }

    /// `controlledburn::burn` over the window, with the geometry and the
    /// window rotated by 180 degrees (see [`Cutline::burn_centre`]): the
    /// grid's rows are the window's rows, its columns the window's columns
    /// right to left.
    fn burn_window(&self, w: &Win, opts: BurnOptions) -> Option<controlledburn::BurnResult> {
        let geoms: Vec<Geometry> = self
            .polygons
            .iter()
            .map(|p| {
                Geometry::Polygon(Polygon::new(
                    p.iter().map(|r| r.iter().map(|&[x, y]| Coord::new(-x, -y)).collect()).collect(),
                ))
            })
            .collect();
        let (c0, r0) = (w.col_off as f64, w.row_off as f64);
        let grid = GridSpec::new(-(c0 + w.ncol as f64), -(r0 + w.nrow as f64), -c0, -r0, w.ncol as u32, w.nrow as u32);
        burn(&geoms, &grid, opts).ok()
    }

    /// GDAL's blend (`BlendMaskGenerator`, alg/gdalcutline.cpp): with `d`
    /// the distance from a pixel centre to the nearest ring edge, a pixel
    /// in the burned mask is scaled by `0.5 + 0.5 d / bd` and one outside
    /// gets `0.5 - 0.5 d / bd`, for `d <= bd`; pixels further out stay 0,
    /// further in stay 1.
    fn blend(&self, w: &Win, bd: f64, out: &mut [f32]) {
        let (c0, r0) = (w.col_off as f64, w.row_off as f64);
        // Each edge, bucketed by the window rows whose centre line is
        // within bd of it.
        let segs: Vec<([f64; 2], [f64; 2])> = self
            .polygons
            .iter()
            .flatten()
            .flat_map(|r| (0..r.len()).map(move |k| (r[k], r[(k + 1) % r.len()])))
            .collect();
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); w.nrow];
        for (s, (a, b)) in segs.iter().enumerate() {
            let ylo = a[1].min(b[1]) - bd - r0 - 0.5;
            let yhi = a[1].max(b[1]) + bd - r0 - 0.5;
            let j1 = yhi.floor().min(w.nrow as f64 - 1.0);
            let j0 = ylo.ceil().max(0.0);
            if j1 < j0 {
                continue;
            }
            for row in &mut rows[j0 as usize..=j1 as usize] {
                row.push(s as u32);
            }
        }
        // One row of distances at a time.
        let mut dist = vec![f64::INFINITY; w.ncol];
        let (mut lo, mut hi) = (usize::MAX, 0usize);
        for (j, row) in rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            let cy = r0 + j as f64 + 0.5;
            for &s in row {
                let (a, b) = segs[s as usize];
                let Some((xl, xr)) = capsule_row(a, b, bd, cy) else { continue };
                let i0 = (xl - c0 - 0.5).ceil().max(0.0);
                let i1 = (xr - c0 - 0.5).floor().min(w.ncol as f64 - 1.0);
                if i1 < i0 {
                    continue;
                }
                let (i0, i1) = (i0 as usize, i1 as usize);
                lo = lo.min(i0);
                hi = hi.max(i1);
                for (i, d) in dist.iter_mut().enumerate().take(i1 + 1).skip(i0) {
                    *d = d.min(seg_dist([c0 + i as f64 + 0.5, cy], a, b));
                }
            }
            if lo > hi {
                continue;
            }
            let orow = &mut out[j * w.ncol..(j + 1) * w.ncol];
            for (o, d) in orow[lo..=hi].iter_mut().zip(&mut dist[lo..=hi]) {
                if *d <= bd {
                    let ratio = if *o == 0.0 { 0.5 - (*d / bd) * 0.5 } else { 0.5 + (*d / bd) * 0.5 };
                    // The burned mask is 0 or 1; GDAL multiplies a density
                    // of 1 by the ratio in either case.
                    *o = ratio as f32;
                }
                *d = f64::INFINITY;
            }
            (lo, hi) = (usize::MAX, 0);
        }
    }
}

/// A source window.
struct Win {
    col_off: usize,
    row_off: usize,
    ncol: usize,
    nrow: usize,
}

impl Win {
    /// A run of the rotated grid as (row, first column, end column) of the
    /// window.
    fn run(&self, r: &controlledburn::GridRun) -> (usize, usize, usize) {
        (r.row as usize, self.ncol - r.col_end as usize, self.ncol - r.col_start as usize)
    }
}

/// GDAL's fill of horizontal edges lying on a centre line
/// (`GDALdllImageFilledPolygon`, "do not skip bottom horizontal segments").
/// GDAL orders every ring clockwise (`GDALCollectRingsFromGeometry`, in
/// pixel/line with the y-up formula) and fills, on that centre line, the
/// span of each horizontal edge running towards -x. In that order those are
/// the edges with the polygon below them and the hole edges with the hole
/// below them; the scanline rule already has the first, so in effect a
/// hole's top edge on a centre line does not cut that line.
fn gdal_horizontal_edges(poly: &[Ring], w: &Win, out: &mut [f32]) {
    let (c0, r0) = (w.col_off as f64, w.row_off as f64);
    for ring in poly {
        let n = ring.len();
        if n < 2 {
            continue;
        }
        let area: f64 = (0..n)
            .map(|k| {
                let (p, q) = (ring[k], ring[(k + 1) % n]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum();
        let clockwise = area < 0.0;
        for k in 0..n {
            // Edge from `p` to `q` in clockwise order.
            let (mut p, mut q) = (ring[k], ring[(k + 1) % n]);
            if !clockwise {
                std::mem::swap(&mut p, &mut q);
            }
            let y = p[1] - r0;
            if q[1] - r0 != y || p[0] <= q[0] {
                continue;
            }
            let row = y - 0.5;
            if row.fract() != 0.0 || row < 0.0 || row >= w.nrow as f64 {
                continue;
            }
            let x1 = (q[0] - c0 + 0.5).floor();
            let x2 = (p[0] - c0 + 0.5).floor();
            if x1 > w.ncol as f64 - 1.0 || x2 <= 0.0 {
                continue;
            }
            let (a, b) = (x1.max(0.0) as usize, x2.min(w.ncol as f64) as usize);
            let row = row as usize;
            out[row * w.ncol + a..row * w.ncol + b].fill(1.0);
        }
    }
}

/// Distance from `p` to segment `a`-`b`.
fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let (ex, ey) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    (ex * ex + ey * ey).sqrt()
}

/// The x range where the line `y = cy` meets the set of points within `r`
/// of segment `a`-`b` (a convex capsule), slightly widened; `None` if it
/// misses.
fn capsule_row(a: [f64; 2], b: [f64; 2], r: f64, cy: f64) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in [a, b] {
        let h = r * r - (cy - c[1]) * (cy - c[1]);
        if h >= 0.0 {
            let s = h.sqrt();
            lo = lo.min(c[0] - s);
            hi = hi.max(c[0] + s);
        }
    }
    // The band between the end discs: |perpendicular| <= r and the
    // projection within the segment, both linear in x on the row.
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = (dx * dx + dy * dy).sqrt();
    if l > 0.0 {
        let (ux, uy) = (dx / l, dy / l);
        // t(x) = (x - ax) ux + (cy - ay) uy in [0, l];
        // p(x) = (x - ax) uy - (cy - ay) ux in [-r, r].
        let (mut x0, mut x1) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut clip = |slope: f64, c: f64, vmin: f64, vmax: f64| {
            // vmin <= slope * (x - ax) + c <= vmax
            if slope.abs() < 1e-300 {
                if c < vmin || c > vmax {
                    x0 = f64::INFINITY;
                    x1 = f64::NEG_INFINITY;
                }
            } else {
                let (u, v) = ((vmin - c) / slope + a[0], (vmax - c) / slope + a[0]);
                x0 = x0.max(u.min(v));
                x1 = x1.min(u.max(v));
            }
        };
        clip(ux, (cy - a[1]) * uy, 0.0, l);
        clip(uy, -(cy - a[1]) * ux, -r, r);
        if x0 <= x1 {
            lo = lo.min(x0);
            hi = hi.max(x1);
        }
    }
    (lo <= hi).then(|| {
        let pad = 1e-9 * (1.0 + lo.abs().max(hi.abs()));
        (lo - pad, hi + pad)
    })
}

fn transform_polygons(polygons: &[Vec<Ring>], transformer: &impl Transformer) -> Result<Vec<Vec<Ring>>, String> {
    let mut out = Vec::with_capacity(polygons.len());
    for p in polygons {
        let mut rings = Vec::with_capacity(p.len());
        for r in p {
            let mut x: Vec<f64> = r.iter().map(|q| q[0]).collect();
            let mut y: Vec<f64> = r.iter().map(|q| q[1]).collect();
            let ok = transformer.transform(true, &mut x, &mut y);
            if ok.iter().any(|&o| !o) {
                return Err("cutline transformation failed".to_string());
            }
            rings.push(x.into_iter().zip(y).map(|(x, y)| [x, y]).collect());
        }
        out.push(rings);
    }
    Ok(out)
}

/// Longest edge (consecutive points, as gdalwarp's
/// `GetMaximumSegmentLength`).
fn max_segment(polygons: &[Vec<Ring>]) -> f64 {
    let mut m: f64 = 0.0;
    for r in polygons.iter().flatten() {
        for w in r.windows(2) {
            let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
            m = m.max(dx * dx + dy * dy);
        }
    }
    m.sqrt()
}

/// OGR's `OGRSimpleCurve::segmentize`: an edge longer than `max` (beyond
/// a relative 1e-5 in squared length) gets `floor(len / max - 0.01)`
/// evenly spaced intermediate points.
fn segmentize(r: &Ring, max: f64) -> Ring {
    let mut out = Vec::with_capacity(r.len());
    let m2 = max * max;
    for (k, &p) in r.iter().enumerate() {
        out.push(p);
        let Some(&q) = r.get(k + 1) else { break };
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let d2 = dx * dx + dy * dy;
        if d2 - m2 > 1e-5 * m2 {
            let n = ((d2 / m2).sqrt() - 1e-2).floor().max(0.0) as usize;
            let (sx, sy) = (dx / (n as f64 + 1.0), dy / (n as f64 + 1.0));
            for j in 1..=n {
                out.push([p[0] + j as f64 * sx, p[1] + j as f64 * sy]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Ring {
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    }

    fn grid(d: &[f32], ncol: usize) -> Vec<String> {
        d.chunks(ncol).map(|r| r.iter().map(|&v| if v > 0.0 { '1' } else { '0' }).collect()).collect()
    }

    #[test]
    fn centre_rule_ties_follow_gdal() {
        // Edges through centres: left and bottom out, right and top in.
        let c = Cutline::new(vec![vec![rect(1.5, 1.5, 4.5, 3.5)]]);
        let d = c.density(0, 0, 6, 5);
        assert_eq!(grid(&d, 6), ["000000", "001110", "001110", "000000", "000000"]);
        // Either ring orientation.
        let mut r = rect(1.5, 1.5, 4.5, 3.5);
        r.reverse();
        assert_eq!(Cutline::new(vec![vec![r]]).density(0, 0, 6, 5), d);
    }

    #[test]
    fn hole_top_edge_on_centre_line_is_kept() {
        let c = Cutline::new(vec![vec![rect(0.0, 0.0, 6.0, 6.0), rect(1.5, 2.5, 4.5, 4.5)]]);
        let d = c.density(0, 0, 6, 6);
        // The hole's top edge (line 2.5) does not cut row 2 and its bottom
        // edge (4.5) is outside the hole by the scanline rule. Across, the
        // hole is the polygon's [left, right) rule turned round: column 1
        // (centre on its left edge) stays, column 4 (on its right) goes.
        // Checked against GDAL's rasterizer.
        assert_eq!(grid(&d, 6), ["111111", "111111", "111111", "110001", "111111", "111111"]);
    }

    #[test]
    fn window_offset_is_respected() {
        let c = Cutline::new(vec![vec![rect(10.2, 20.2, 12.8, 21.8)]]);
        let d = c.density(9, 19, 5, 4);
        assert_eq!(grid(&d, 5), ["00000", "01110", "01110", "00000"]);
    }

    #[test]
    fn exact_and_all_touched() {
        let c = Cutline::new(vec![vec![rect(0.5, 0.0, 2.0, 1.0)]]).with_rule(CutlineRule::Exact);
        assert_eq!(c.density(0, 0, 3, 1), [0.5, 1.0, 0.0]);
        let c = c.with_rule(CutlineRule::AllTouched);
        assert_eq!(c.density(0, 0, 3, 1), [1.0, 1.0, 0.0]);
    }

    #[test]
    fn blend_ramp() {
        // Vertical edge at x = 5; pixel centres at distance 0.5, 1.5, ...
        let c = Cutline::new(vec![vec![rect(5.0, -10.0, 20.0, 10.0)]]).with_blend_dist(2.0);
        let d = c.density(0, 0, 10, 1);
        let want = [0.0, 0.0, 0.0, 0.125, 0.375, 0.625, 0.875, 1.0, 1.0, 1.0];
        for (a, b) in d.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{d:?}");
        }
    }

    #[test]
    fn capsule_row_contains_every_close_point() {
        let (a, b) = ([1.3, 2.1], [7.9, -3.4]);
        for j in -60..60 {
            let cy = j as f64 * 0.17;
            let range = capsule_row(a, b, 1.7, cy);
            for i in -100..150 {
                let x = i as f64 * 0.093;
                let inside = seg_dist([x, cy], a, b) <= 1.7;
                let covered = range.is_some_and(|(l, h)| x >= l && x <= h);
                assert!(!inside || covered, "{x} {cy}");
            }
        }
    }

    #[test]
    fn segmentize_matches_ogr() {
        let r = vec![[0.0, 0.0], [10.0, 0.0]];
        assert_eq!(segmentize(&r, 2.5).len(), 5);
        // 10 / 3 = 3.33: three intermediate points.
        assert_eq!(segmentize(&r, 3.0).len(), 5);
        assert_eq!(segmentize(&r, 10.0).len(), 2);
    }
}
