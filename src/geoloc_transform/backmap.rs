//! GDAL's backmap inverse for geolocation arrays
//! ([`GeolocInverse::Backmap`](super::GeolocInverse::Backmap)).
//!
//! Follows GDAL 3.14 (`alg/gdalgeoloc.cpp` and `alg/rasterfill.cpp` at
//! `739fcf03bf`; the logic is unchanged since 3.12) in its arithmetic
//! order, including the float32 storage of the backmap, so that the
//! inverse gives GDAL's answers point for point:
//!
//! | rwarp | GDAL |
//! |-------|------|
//! | [`Backmap::new`] | `GDALGeoLoc::LoadGeolocFinish` (extent) and `GDALGeoLoc::GenerateBackMap` |
//! | [`Backmap::inverse`] | `GDALGeoLoc::Transform`, `bDstToSrc = TRUE` |
//! | `fill_nodata` | `GDALFillNodata`, inverse distance, max distance 3, one smoothing pass |
//! | `multi_filter` | `GDALMultiFilter` and `GDALFilterLine` |
//! | `Ring::holds` | `OGRLinearRing::isPointInRing` or `isPointOnRingBoundary`, envelope tested |
//!
//! Only the in-memory backmap is ported (`GDALGeoLocCArrayAccessors`). GDAL
//! switches to temporary GTiff datasets for arrays of 24 megapixels or more
//! (`GDAL_GEOLOC_USE_TEMP_DATASETS`); that path runs the same arithmetic on
//! float32 bands and is not checked here.
//!
//! The ported code reaches the geolocation arrays only through the parent
//! module's `pixel_line_to_xy`, `pixel_line_to_xy_int`, `get` and
//! `is_missing`, which follow GDAL's `PixelLineToXY` and nodata test.

use super::{inverse_bilinear, GeolocArrays};

/// GDAL's marker for a backmap cell with no source position.
const INVALID_BMXY: f32 = -10.0;
/// GDAL's block size for walking the arrays and the backmap.
const TILE_SIZE: usize = 256;

/// GDAL's backmap: a regular grid over the geolocation arrays' extent, in
/// their coordinates, holding for each node the raster pixel/line there.
#[derive(Debug, Clone)]
pub(super) struct Backmap {
    /// North-up geotransform of the backmap.
    gt: [f64; 6],
    w: i32,
    h: i32,
    x: Vec<f32>,
    y: Vec<f32>,
}

/// GDAL's ring `(x0, y0), (x2, y2), (x3, y3), (x1, y1), (x0, y0)` for the
/// square with corners 0 = (i, j), 1 = (i, j + 1), 2 = (i + 1, j),
/// 3 = (i + 1, j + 1).
struct Ring([(f64, f64); 5]);

impl Ring {
    fn new(q: &Square) -> Self {
        let [x0, x1, x2, x3] = q.x;
        let [y0, y1, y2, y3] = q.y;
        Ring([(x0, y0), (x2, y2), (x3, y3), (x1, y1), (x0, y0)])
    }

    /// `OGRSimpleCurve::getEnvelope` then the containment test in
    /// `isPointInRing` and `isPointOnRingBoundary`.
    fn in_envelope(&self, tx: f64, ty: f64) -> bool {
        let p = &self.0;
        let (mut x0, mut x1, mut y0, mut y1) = (p[0].0, p[0].0, p[0].1, p[0].1);
        for &(x, y) in &p[1..] {
            if x < x0 {
                x0 = x;
            }
            if x > x1 {
                x1 = x;
            }
            if y < y0 {
                y0 = y;
            }
            if y > y1 {
                y1 = y;
            }
        }
        tx >= x0 && tx <= x1 && ty >= y0 && ty <= y1
    }

    /// `OGRLinearRing::isPointInRing`, without the envelope test.
    fn contains(&self, tx: f64, ty: f64) -> bool {
        let p = &self.0;
        let mut crossings = 0;
        let (mut prev_x, mut prev_y) = (p[0].0 - tx, p[0].1 - ty);
        for &(x, y) in &p[1..] {
            let (x1, y1) = (x - tx, y - ty);
            let (x2, y2) = (prev_x, prev_y);
            if (y1 > 0.0 && y2 <= 0.0) || (y2 > 0.0 && y1 <= 0.0) {
                let cut = (x1 * y2 - x2 * y1) / (y2 - y1);
                if 0.0 < cut {
                    crossings += 1;
                }
            }
            prev_x = x1;
            prev_y = y1;
        }
        crossings % 2 != 0
    }

    /// `OGRLinearRing::isPointOnRingBoundary`, without the envelope test:
    /// an exact zero cross product, so in practice only points on
    /// axis-aligned edges or at vertices.
    fn on_boundary(&self, tx: f64, ty: f64) -> bool {
        let p = &self.0;
        let (mut prev_x, mut prev_y) = (tx - p[0].0, ty - p[0].1);
        for i in 1..p.len() {
            let (dx1, dy1) = (tx - p[i].0, ty - p[i].1);
            let (dx2, dy2) = (prev_x, prev_y);
            if dx1 * dy2 - dx2 * dy1 == 0.0 && !(dx1 == dx2 && dy1 == dy2) {
                let sx = p[i].0 - p[i - 1].0;
                let sy = p[i].1 - p[i - 1].1;
                let cross = dx2 * sx + dy2 * sy;
                if cross >= 0.0 && cross <= sx * sx + sy * sy {
                    return true;
                }
            }
            prev_x = dx1;
            prev_y = dy1;
        }
        false
    }

    /// `isPointInRing(p) || isPointOnRingBoundary(p)`, each testing the
    /// envelope first (their default).
    fn holds(&self, tx: f64, ty: f64) -> bool {
        self.in_envelope(tx, ty) && (self.contains(tx, ty) || self.on_boundary(tx, ty))
    }
}

/// Four corners of one grid square, in GDAL's naming: 0 = (i, j),
/// 1 = (i, j + 1), 2 = (i + 1, j), 3 = (i + 1, j + 1).
#[derive(Clone, Copy)]
struct Square {
    x: [f64; 4],
    y: [f64; 4],
}

impl Square {
    /// `GDALGeoLoc::ExtractSquare` through the integer `PixelLineToXY`.
    fn at(g: &GeolocArrays, i: i64, j: i64) -> Option<Self> {
        let (x0, y0) = g.pixel_line_to_xy_int(i, j)?;
        let (x2, y2) = g.pixel_line_to_xy_int(i + 1, j)?;
        let (x1, y1) = g.pixel_line_to_xy_int(i, j + 1)?;
        let (x3, y3) = g.pixel_line_to_xy_int(i + 1, j + 1)?;
        Some(Square { x: [x0, x1, x2, x3], y: [y0, y1, y2, y3] })
    }

    /// GDAL's antimeridian test. If the square straddles it, its positive
    /// longitudes are moved below -180 (the first of the two tries GDAL
    /// makes) and the result is true.
    fn straddle(&mut self, lon_wrap: bool) -> bool {
        let [x0, x1, x2, x3] = self.x;
        if lon_wrap
            && x0.abs() > 170.0
            && x1.abs() > 170.0
            && x2.abs() > 170.0
            && x3.abs() > 170.0
            && ((x1 - x0).abs() > 180.0 || (x2 - x0).abs() > 180.0 || (x3 - x0).abs() > 180.0)
        {
            for v in self.x.iter_mut() {
                if *v > 0.0 {
                    *v -= 360.0;
                }
            }
            return true;
        }
        false
    }

    fn shift_east(&mut self) {
        for v in self.x.iter_mut() {
            *v += 360.0;
        }
    }

    /// `GDALInverseBilinearInterpolation` from corner 0 at `(i, j)`.
    fn invert(&self, gx: f64, gy: f64, i: i64, j: i64) -> (f64, f64) {
        let (mut fi, mut fj) = (i as f64, j as f64);
        let (x, y) = (&self.x, &self.y);
        inverse_bilinear(gx, gy, x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], &mut fi, &mut fj);
        (fi, fj)
    }
}

/// The extent `GDALGeoLoc::LoadGeolocFinish` computes: every valid sample,
/// widened by the arrays' outer edges (the extrapolated edge for
/// `TOP_LEFT_CORNER`, half a cell beyond the outermost samples for
/// `PIXEL_CENTER`).
fn extent(g: &GeolocArrays) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x) = (f64::MAX, -f64::MAX);
    let (mut min_y, mut max_y) = (f64::MAX, -f64::MAX);
    let mut update = |x: f64, y: f64| {
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    };
    // GDAL walks 256 x 256 blocks; the extent does not depend on order.
    for iy in 0..g.nrow {
        for ix in 0..g.ncol {
            let (x, y) = g.get(ix, iy);
            if !g.is_missing(x) {
                update(g.unshift_geo_x(x), y);
            }
        }
    }
    let (nx, ny) = (g.ncol as f64, g.nrow as f64);
    let mut edge = |px: f64, py: f64| {
        if let Some((mut x, y)) = g.pixel_line_to_xy(px, py) {
            if g.opts.lon_wrap {
                x = x.clamp(-180.0, 180.0);
            }
            update(x, y);
        }
    };
    if g.convention_offset() == 0.0 {
        for ix in 0..=g.ncol {
            edge(ix as f64, ny);
        }
        for iy in 0..=g.nrow {
            edge(nx, iy as f64);
        }
    } else {
        for ix in 0..=g.ncol {
            edge(ix as f64, -0.5);
        }
        for ix in 0..=g.ncol {
            edge(ix as f64, ny - 1.0 + 0.5);
        }
        for iy in 0..=g.nrow {
            edge(-0.5, iy as f64);
        }
        for iy in 0..=g.nrow {
            edge(nx - 1.0 + 0.5, iy as f64);
        }
    }
    (min_x, max_x, min_y, max_y)
}

/// `START_ITER_PER_BLOCK` helper of `GenerateBackMap`: the first and
/// (exclusive, padded) last sample position of each 256-sample block along
/// an axis of `n` samples, accumulated in steps of `step` from `-step`.
fn block_ranges(n: usize, step: f64) -> Vec<(f64, f64)> {
    let nblocks = n.div_ceil(TILE_SIZE);
    let mut se = vec![(0.0, 0.0); nblocks + 1];
    let mut last: i64 = -1;
    let mut d = -step;
    while d <= n as f64 + 2.0 * step {
        let b = (d / TILE_SIZE as f64) as i64;
        if b != last {
            if last >= 0 {
                se[last as usize].1 = d + step / 10.0;
            }
            se[b as usize].0 = d;
            last = b;
        }
        d += step;
    }
    let b = (d / TILE_SIZE as f64) as usize;
    se[b].1 = d + step / 10.0;
    se
}

impl Backmap {
    /// Build the backmap for `g` with GDAL's oversampling factor
    /// (`GEOLOC_BACKMAP_OVERSAMPLE_FACTOR`, default 1.3).
    pub(super) fn new(g: &GeolocArrays, oversample: f64) -> Result<Self, String> {
        let (ext_min_x, ext_max_x, ext_min_y, ext_max_y) = extent(g);
        let nx = g.ncol as i32;
        let ny = g.nrow as i32;
        let target = g.ncol as f64 * g.nrow as f64 * oversample;
        let size = ((ext_max_x - ext_min_x) * (ext_max_y - ext_min_y) / target).sqrt();
        // GDAL tests only for zero; NaN (no valid sample) fails later on the
        // size checks, which a NaN never passes.
        if size == 0.0 {
            return Err("invalid pixel size for the geolocation backmap".to_string());
        }
        let min_x = ext_min_x - size / 2.0;
        let max_x = ext_max_x + size / 2.0;
        let max_y = ext_max_y + size / 2.0;
        let min_y = ext_min_y - size / 2.0;
        let bw = ((max_x - min_x) / size).ceil();
        let bh = ((max_y - min_y) / size).ceil();
        if !(bw > 0.0 && bw + 2.0 < i32::MAX as f64) || !(bh > 0.0 && bh + 2.0 < i32::MAX as f64) {
            return Err(format!("geolocation backmap size overflow: {bw} x {bh}"));
        }
        let (mut bw, mut bh) = (bw as i32, bh as i32);
        let psx = (max_x - min_x) / bw as f64;
        let psy = (max_y - min_y) / bh as f64;
        // One more for the right and bottom extension of TOP_LEFT_CORNER.
        bw += 1;
        bh += 1;
        let n = (bw as usize)
            .checked_mul(bh as usize)
            .ok_or_else(|| format!("geolocation backmap size overflow: {bw} x {bh}"))?;
        let mut bm = Backmap {
            gt: [min_x, psx, 0.0, max_y, 0.0, -psy],
            w: bw,
            h: bh,
            x: vec![0f32; n],
            y: vec![0f32; n],
        };
        let mut wt = vec![0f32; n];

        let off = g.convention_offset();
        let o = g.opts;
        let lon_wrap = o.lon_wrap;
        let at = |ix: i32, iy: i32| (iy as usize) * (bw as usize) + ix as usize;

        // GDAL's UpdateBackmap lambda.
        let update = |bx: &mut [f32], by: &mut [f32], wt: &mut [f32], ibx: i32, iby: i32, dx: f64, dy: f64, w: f64| {
            let k = at(ibx, iby);
            let ux = bx[k] + (w * ((dx + off) * o.pixel_step + o.pixel_offset)) as f32;
            let uy = by[k] + (w * ((dy + off) * o.line_step + o.line_offset)) as f32;
            let uw = wt[k] + w as f32;
            if uw > 0.0 {
                let fx = ux / uw;
                let fy = uy / uw;
                let gpx = (f64::from(fx) - o.pixel_offset) / o.pixel_step - off;
                let gpy = (f64::from(fy) - o.line_offset) / o.line_step - off;
                let ixa = (gpx.max(0.0) as i32).min(nx - 1);
                let iya = (gpy.max(0.0) as i32).min(ny - 1);
                let (glx, gly) = g.get(ixa as usize, iya as usize);
                // C's static_cast<unsigned> of a double above -1 truncates
                // toward zero; samples start at -1 / oversample.
                let ix = dx as u32;
                let iy = dy as u32;
                let near = || {
                    let (sx, sy) = g.get(ix as usize, iy as usize);
                    (glx - sx).abs() <= 2.0 * psx && (gly - sy).abs() <= 2.0 * psy
                };
                if !g.is_missing(glx) && ((ix >= (nx - 1) as u32 || iy >= (ny - 1) as u32) || near()) {
                    bx[k] = ux;
                    by[k] = uy;
                    wt[k] = uw;
                }
            }
        };

        // Walk the arrays' pixel space densely enough to hit every backmap
        // node of an affine grid, forward-projecting each sample.
        let step = 1.0 / oversample;
        let ys = block_ranges(g.nrow, step);
        let xs = block_ranges(g.ncol, step);
        let (nyb, nxb) = (g.nrow.div_ceil(TILE_SIZE), g.ncol.div_ceil(TILE_SIZE));
        for yb in ys.iter().take(nyb) {
            for xb in xs.iter().take(nxb) {
                let mut dy = yb.0;
                while dy < yb.1 {
                    let mut dx = xb.0;
                    while dx < xb.1 {
                        bm.sample(g, dx, dy, &mut wt, &update, lon_wrap, off);
                        dx += step;
                    }
                    dy += step;
                }
            }
        }

        // Average the weighted sums; untouched nodes are invalid.
        for k in 0..n {
            let w = wt[k];
            if w > 0.0 {
                bm.x[k] /= w;
                bm.y[k] /= w;
            } else {
                bm.x[k] = INVALID_BMXY;
                bm.y[k] = INVALID_BMXY;
            }
        }
        drop(wt);

        let (w, h) = (bw as usize, bh as usize);
        fill_nodata(&mut bm.x, w, h, 3.0, 1);
        fill_nodata(&mut bm.y, w, h, 3.0, 1);

        // GDAL's last pass: along each row, bridge a gap between two valid
        // nodes whose positions differ by at most 2 pixels in both axes.
        // (GDAL walks blocks, carrying each row's state across them.)
        for iy in 0..h {
            let mut last: i64 = -1;
            let mut last_x = 0f32;
            for ix in 0..w {
                let v = bm.x[iy * w + ix];
                if v == INVALID_BMXY {
                    continue;
                }
                if last != -1 && ix as i64 > last + 1 && (v - last_x).abs() <= 2.0 {
                    let vy = bm.y[iy * w + ix];
                    let last_y = bm.y[iy * w + last as usize];
                    if (vy - last_y).abs() <= 2.0 {
                        for inner in (last + 1) as usize..ix {
                            let alpha = (inner as i64 - last) as f32 / (ix as i64 - last) as f32;
                            bm.x[iy * w + inner] = (1.0 - alpha) * last_x + alpha * v;
                            bm.y[iy * w + inner] = (1.0 - alpha) * last_y + alpha * vy;
                        }
                    }
                }
                last = ix as i64;
                last_x = v;
            }
        }
        Ok(bm)
    }

    /// One forward-projected sample of `GenerateBackMap`: the exact inverse
    /// at the backmap node to its top left when one of the four squares
    /// around the sample holds that node, otherwise a bilinear splat of the
    /// sample's own position onto the four nodes around it.
    #[allow(clippy::too_many_arguments)]
    fn sample<F>(&mut self, g: &GeolocArrays, dx: f64, dy: f64, wt: &mut [f32], update: &F, lon_wrap: bool, off: f64)
    where
        F: Fn(&mut [f32], &mut [f32], &mut [f32], i32, i32, f64, f64, f64),
    {
        let Some((glx, gly)) = g.pixel_line_to_xy(dx, dy) else {
            return;
        };
        let (bw, bh) = (self.w, self.h);
        let at = |ix: i32, iy: i32| (iy as usize) * (bw as usize) + ix as usize;
        let (min_x, psx, max_y, psy) = (self.gt[0], self.gt[1], self.gt[3], -self.gt[5]);
        let o = g.opts;
        let dbx = (glx - min_x) / psx;
        let dby = (max_y - gly) / psy;
        let ibx = dbx.floor() as i32;
        let iby = dby.floor() as i32;

        if ibx >= 0 && ibx < bw && iby >= 0 && iby < bh {
            let mut gx = min_x + ibx as f64 * psx;
            let gy = max_y - iby as f64 * psy;
            let mut found = false;
            let outer = if lon_wrap && gx.abs() >= 180.0 { 2 } else { 1 };
            for it in 0..outer {
                if it == 1 && gx >= 180.0 {
                    gx -= 360.0;
                } else if it == 1 && gx <= -180.0 {
                    gx += 360.0;
                }
                let i0 = dx.floor() as i64;
                let j0 = dy.floor() as i64;
                let mut sx = -1;
                while !found && sx <= 0 {
                    let mut sy = -1;
                    while !found && sy <= 0 {
                        let (pixel, line) = (i0 + sx, j0 + sy);
                        // GDAL breaks out of the inner loop on failure.
                        let Some(mut q) = Square::at(g, pixel, line) else {
                            break;
                        };
                        let iters = if q.straddle(lon_wrap) { 2 } else { 1 };
                        // GDAL does not stop after a match here: a square
                        // straddling the antimeridian may match twice, and
                        // the second wins.
                        for k in 0..iters {
                            if k == 1 {
                                q.shift_east();
                            }
                            if Ring::new(&q).holds(gx, gy) {
                                found = true;
                                let (vx, vy) = q.invert(gx, gy, pixel, line);
                                let vx = (vx + off) * o.pixel_step + o.pixel_offset;
                                let vy = (vy + off) * o.line_step + o.line_offset;
                                let k = at(ibx, iby);
                                self.x[k] = vx as f32;
                                self.y[k] = vy as f32;
                                wt[k] = 1.0;
                            }
                        }
                        sy += 1;
                    }
                    sx += 1;
                }
            }
            if found {
                return;
            }
        }

        // Non-nominal cases: nodata, holes, the arrays' edges.
        if ibx < -1 || iby < -1 || ibx > bw || iby > bh {
            return;
        }
        let fx = dbx - ibx as f64;
        let fy = dby - iby as f64;
        let (bx, by) = (&mut self.x, &mut self.y);
        if ibx >= 0 && iby >= 0 && ibx < bw && iby < bh && wt[at(ibx, iby)] != 1.0 {
            update(bx, by, wt, ibx, iby, dx, dy, (1.0 - fx) * (1.0 - fy));
        }
        if iby >= 0 && ibx + 1 < bw && iby < bh && wt[at(ibx + 1, iby)] != 1.0 {
            update(bx, by, wt, ibx + 1, iby, dx, dy, fx * (1.0 - fy));
        }
        if ibx + 1 < bw && iby + 1 < bh && wt[at(ibx + 1, iby + 1)] != 1.0 {
            update(bx, by, wt, ibx + 1, iby + 1, dx, dy, fx * fy);
        }
        if ibx >= 0 && ibx < bw && iby + 1 < bh && wt[at(ibx, iby + 1)] != 1.0 {
            update(bx, by, wt, ibx, iby + 1, dx, dy, (1.0 - fx) * fy);
        }
    }

    fn node(&self, ix: i32, iy: i32) -> (f32, f32) {
        let k = (iy as usize) * (self.w as usize) + ix as usize;
        (self.x[k], self.y[k])
    }

    /// Size of the backmap, `(width, height)`, and its geotransform.
    pub(super) fn grid(&self) -> ((usize, usize), [f64; 6]) {
        ((self.w as usize, self.h as usize), self.gt)
    }

    /// Georeferenced coordinates to raster pixel/line, as
    /// `GDALGeoLoc::Transform` with `bDstToSrc = TRUE`: the backmap's
    /// bilinear estimate, then (if `refine`) the exact inverse in the first
    /// square holding the point in a spiral search of radius 3 around the
    /// estimate (5 near the poles of a wrapped geographic grid).
    ///
    /// Without `refine` this is GDAL's `GDAL_GEOLOC_USE_MAX_ACCURACY=NO`.
    pub(super) fn inverse(&self, g: &GeolocArrays, geo_x: f64, geo_y: f64, refine: bool) -> Option<(f64, f64)> {
        let o = g.opts;
        let off = g.convention_offset();
        let dbx = (geo_x - self.gt[0]) / self.gt[1];
        let dby = (geo_y - self.gt[3]) / self.gt[5];
        if !(dbx >= 0.0 && dby >= 0.0 && dbx + 1.0 < self.w as f64 && dby + 1.0 < self.h as f64) {
            return None;
        }
        let (ibx, iby) = (dbx as i32, dby as i32);
        let (x00, y00) = self.node(ibx, iby);
        if x00 == INVALID_BMXY {
            return None;
        }
        let (x10, y10) = self.node(ibx + 1, iby);
        let (x01, y01) = self.node(ibx, iby + 1);
        let (x11, y11) = self.node(ibx + 1, iby + 1);
        let (fx, fy) = (dbx - ibx as f64, dby - iby as f64);
        let d = f64::from;
        let (px, py) = if x10 != INVALID_BMXY && x01 != INVALID_BMXY && x11 != INVALID_BMXY {
            (
                (1.0 - fy) * (d(x00) + fx * d(x10 - x00)) + fy * (d(x01) + fx * d(x11 - x01)),
                (1.0 - fy) * (d(y00) + fx * d(y10 - y00)) + fy * (d(y01) + fx * d(y11 - y01)),
            )
        } else if x10 != INVALID_BMXY {
            (d(x00) + fx * d(x10 - x00), d(y00) + fx * d(y10 - y00))
        } else if x01 != INVALID_BMXY {
            (d(x00) + fy * d(x01 - x00), d(y00) + fy * d(y01 - y00))
        } else {
            (d(x00), d(y00))
        };
        if !refine {
            return Some((px, py));
        }
        let gpx = (px - o.pixel_offset) / o.pixel_step - off;
        let gpy = (py - o.line_offset) / o.line_step - off;

        // GDAL's radius, "rather empirical", tuned on a Sentinel-5P product
        // over the north pole and GDAL issue #5823.
        let radius: i64 = if o.lon_wrap && geo_y.abs() >= 85.0 { 5 } else { 3 };
        let (npx, npy) = (gpx.floor() as i64, gpy.floor() as i64);
        let (nx, ny) = (g.ncol as i64, g.nrow as i64);
        for r in 0..=radius {
            let count = if r == 0 { 1 } else { 8 * r };
            for it in 0..count {
                let sx = if r == 0 {
                    0
                } else if it < 2 * r {
                    -r + it
                } else if it < 4 * r {
                    r
                } else if it < 6 * r {
                    r - (it - 4 * r)
                } else {
                    -r
                };
                let sy = if r == 0 {
                    0
                } else if it < 2 * r {
                    r
                } else if it < 4 * r {
                    r - (it - 2 * r)
                } else if it < 6 * r {
                    -r
                } else {
                    -r + (it - 6 * r)
                };
                if npx >= nx - sx || npy >= ny - sy {
                    continue;
                }
                let (ix, iy) = (npx + sx, npy + sy);
                // GDAL's test is `||`, not `&&`: squares further than one
                // beyond the top or left edge are searched (extrapolated)
                // as long as the other index is not.
                if ix >= -1 || iy >= -1 {
                    let Some(mut q) = Square::at(g, ix, iy) else {
                        continue;
                    };
                    let iters = if q.straddle(o.lon_wrap) { 2 } else { 1 };
                    for k in 0..iters {
                        if k == 1 {
                            q.shift_east();
                        }
                        if Ring::new(&q).holds(geo_x, geo_y) {
                            let (vx, vy) = q.invert(geo_x, geo_y, ix, iy);
                            return Some((
                                (vx + off) * o.pixel_step + o.pixel_offset,
                                (vy + off) * o.line_step + o.line_offset,
                            ));
                        }
                    }
                }
            }
        }
        None
    }
}

/// `GDALFillNodata` on a float32 band whose nodata is [`INVALID_BMXY`]:
/// inverse distance weighting over a four-quadrant search to
/// `max_dist` pixels, no `NODATA` option, then `smoothing` passes of
/// `GDALMultiFilter` over the filled pixels.
fn fill_nodata(band: &mut [f32], w: usize, h: usize, max_dist: f64, smoothing: usize) {
    let max_steps = max_dist.floor() as i32;
    let none: u32 = if w > 65533 || h > 65533 { 4_000_002 } else { 65535 };
    let valid: Vec<bool> = band.iter().map(|&v| v != INVALID_BMXY).collect();

    // Top-down pass: per column, the last valid row (within max_dist) and
    // its value, kept for the bottom-up pass.
    let mut y_work = vec![0u32; w * h];
    let mut v_work = vec![0f32; w * h];
    let mut last_y = vec![none; w];
    let mut this_y = vec![0u32; w];
    let mut last_v = vec![0f32; w];
    let mut this_v = vec![0f32; w];
    for iy in 0..h {
        for ix in 0..w {
            if valid[iy * w + ix] {
                this_v[ix] = band[iy * w + ix];
                this_y[ix] = iy as u32;
            } else if iy as f64 <= max_dist + last_y[ix] as f64 {
                this_v[ix] = last_v[ix];
                this_y[ix] = last_y[ix];
            } else {
                this_y[ix] = none;
            }
        }
        y_work[iy * w..(iy + 1) * w].copy_from_slice(&this_y);
        v_work[iy * w..(iy + 1) * w].copy_from_slice(&this_v);
        std::mem::swap(&mut this_v, &mut last_v);
        std::mem::swap(&mut this_y, &mut last_y);
    }

    // Bottom-up pass, interpolating each invalid pixel from the nearest
    // value in each quadrant.
    last_y.iter_mut().for_each(|v| *v = none);
    let mut filt = vec![false; w * h];
    for iy in (0..h).rev() {
        let row = iy * w;
        let mut scan: Vec<f32> = band[row..row + w].to_vec();
        for ix in 0..w {
            if valid[row + ix] {
                this_v[ix] = scan[ix];
                this_y[ix] = iy as u32;
            } else if last_y[ix].wrapping_sub(iy as u32) as f64 <= max_dist {
                // GUInt32 arithmetic in GDAL.
                this_v[ix] = last_v[ix];
                this_y[ix] = last_y[ix];
            } else {
                this_y[ix] = none;
            }
        }
        let top_y = &y_work[row..row + w];
        let top_v = &v_work[row..row + w];
        for ix in 0..w {
            if valid[row + ix] {
                continue;
            }
            let mut steps = max_steps;
            // Top left, bottom left, top right, bottom right.
            let mut dist = [max_dist + 1.0; 4];
            let mut val = [0f32; 4];
            let check = |d: &mut f64, v: &mut f32, tx: usize, ty: u32, tv: f32| {
                if ty != none {
                    let dx = tx as f64 - ix as f64;
                    let dy = ty as f64 - iy as f64;
                    let sq = dx * dx + dy * dy;
                    if sq < *d * *d {
                        *d = sq.sqrt();
                        *v = tv;
                    }
                }
            };
            let mut step = 0;
            while step <= steps {
                let left = (ix as i32 - step).max(0) as usize;
                let right = (ix as i32 + step).min(w as i32 - 1) as usize;
                check(&mut dist[0], &mut val[0], left, top_y[left], top_v[left]);
                check(&mut dist[1], &mut val[1], left, last_y[left], last_v[left]);
                if step != 0 {
                    check(&mut dist[2], &mut val[2], right, top_y[right], top_v[right]);
                    check(&mut dist[3], &mut val[3], right, last_y[right], last_v[right]);
                    if step & 0x3 == 0 {
                        steps = dist[0].max(dist[1]).max(dist[2].max(dist[3])).floor() as i32;
                    }
                }
                step += 1;
            }
            let (mut wsum, mut vsum) = (0.0f64, 0.0f64);
            let mut any = false;
            for q in 0..4 {
                if dist[q] <= max_dist {
                    any = true;
                    let wq = 1.0 / dist[q];
                    wsum += wq;
                    vsum += f64::from(val[q]) * wq;
                }
            }
            if any {
                filt[row + ix] = true;
                scan[ix] = if wsum > 0.0 { (vsum / wsum) as f32 } else { 0.0 };
            }
        }
        band[row..row + w].copy_from_slice(&scan);
        std::mem::swap(&mut this_v, &mut last_v);
        std::mem::swap(&mut this_y, &mut last_y);
    }

    if smoothing > 0 {
        multi_filter(band, w, h, &filt, smoothing);
    }
}

/// `GDALMultiFilter`: `iterations` 3 x 3 averaging passes over the filled
/// pixels (`filt`), in GDAL's rolling buffer of `iterations + 2` lines and
/// three passes. Contributors are the pixels that are valid (not
/// [`INVALID_BMXY`]) when their line is loaded, as GDAL re-reads the
/// nodata mask of the updated band.
fn multi_filter(band: &mut [f32], w: usize, h: usize, filt: &[bool], iterations: usize) {
    let lines = iterations + 2;
    let mut tmask = vec![false; w * lines];
    let mut fmask = vec![false; w * lines];
    let mut pass = vec![0f32; 3 * w * lines];
    let plane = w * lines;
    let mut counter = 0usize;
    for new_line in 0..h + iterations {
        counter = (counter + 1) % 3;
        let slast = (counter % 3) * plane;
        let last = ((counter + 1) % 3) * plane;
        let this = ((counter + 2) % 3) * plane;
        let off = new_line % lines;
        if new_line < h {
            for ix in 0..w {
                let v = band[new_line * w + ix];
                tmask[off * w + ix] = v != INVALID_BMXY;
                fmask[off * w + ix] = filt[new_line * w + ix];
                pass[this + off * w + ix] = v;
            }
        }
        let lo = new_line as i64 - iterations as i64;
        let mut f = new_line as i64 - 1;
        while f >= lo {
            let fl = f;
            f -= 1;
            let this_off = fl.rem_euclid(lines as i64) as usize;
            if fl >= 0 {
                let (src, dst) = (last + this_off * w, this + this_off * w);
                pass.copy_within(src..src + w, dst);
            }
            // GDAL skips the first and last lines.
            if fl < 1 || fl >= h as i64 - 1 {
                continue;
            }
            let last_off = ((fl - 1) % lines as i64) as usize;
            let next_off = ((fl + 1) % lines as i64) as usize;
            for ix in 0..w {
                let out = this + this_off * w + ix;
                if !fmask[this_off * w + ix] {
                    pass[out] = pass[last + this_off * w + ix];
                    continue;
                }
                let mut sum = 0f32;
                let mut wsum = 0f32;
                let rows = [
                    (slast + last_off * w, last_off),
                    (last + this_off * w, this_off),
                    (this + next_off * w, next_off),
                ];
                for (line, m) in rows {
                    if ix > 0 && tmask[m * w + ix - 1] {
                        sum += pass[line + ix - 1];
                        wsum += 1.0;
                    }
                    if tmask[m * w + ix] {
                        sum += pass[line + ix];
                        wsum += 1.0;
                    }
                    if ix < w - 1 && tmask[m * w + ix + 1] {
                        sum += pass[line + ix + 1];
                        wsum += 1.0;
                    }
                }
                pass[out] = sum / wsum;
            }
        }
        if new_line >= iterations {
            let save = new_line - iterations;
            let o = this + (save % lines) * w;
            band[save * w..(save + 1) * w].copy_from_slice(&pass[o..o + w]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_matches_ogr_on_edges_and_vertices() {
        // Unit square in GDAL's corner order: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1).
        let q = Square { x: [0.0, 0.0, 1.0, 1.0], y: [0.0, 1.0, 0.0, 1.0] };
        let r = Ring::new(&q);
        assert!(r.holds(0.5, 0.5));
        assert!(r.holds(0.0, 0.5), "left edge");
        assert!(r.holds(1.0, 1.0), "vertex");
        assert!(!r.holds(1.0 + 1e-12, 0.5));
        assert!(!r.holds(f64::NAN, 0.5));
    }

    #[test]
    fn fill_nodata_fills_a_hole_by_inverse_distance() {
        // A 5 x 5 ramp (value = column) with its centre missing. The
        // nearest value per quadrant: 2 above and 2 below (distance 1), 3
        // to the upper right (1) and lower right (sqrt 2), as GDAL's
        // quadrants include the centre column on the left only. Inverse
        // distance gives 2.4605; the 3 x 3 smoothing pass then averages it
        // with its eight neighbours: (18 - 2 + 2.4605) / 9.
        let w = 5;
        let mut band: Vec<f32> = (0..25).map(|k| (k % w) as f32).collect();
        band[12] = INVALID_BMXY;
        fill_nodata(&mut band, w, w, 3.0, 1);
        let idw = (2.0 + 2.0 + 3.0 + 3.0 / 2f64.sqrt()) / (3.0 + 1.0 / 2f64.sqrt());
        let want = (16.0 + idw as f32) / 9.0;
        assert!((band[12] - want).abs() < 1e-5, "{} vs {want}", band[12]);
        // A pixel beyond the search distance stays invalid.
        let mut band = vec![INVALID_BMXY; 81];
        band[0] = 1.0;
        fill_nodata(&mut band, 9, 9, 3.0, 1);
        assert_eq!(band[80], INVALID_BMXY);
        assert_ne!(band[1], INVALID_BMXY);
    }
}
