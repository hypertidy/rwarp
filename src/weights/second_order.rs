//! Second-order conservative remapping on top of the first-order table:
//! ESMF's `conserve_2nd` (Kritsikis et al. 2017), from weights recorded
//! with [`WeightOptions::moments`](super::WeightOptions::moments).
//!
//! Each source cell's value `f` becomes a plane over the cell,
//! `f + g . (x - c)`, and each overlap takes the plane's mean over it,
//! which is the plane at the overlap's centroid. So a destination cell's
//! value is
//!
//! ```text
//! sum over its sources of S (f + g . d),  d = (overlap centroid) - c
//! ```
//!
//! with `S` the first-order weight. As ESMF does:
//!
//! - **`c` is the centroid of the part of the source cell that the table
//!   covers**: the measure-weighted mean of its overlaps' centroids. The
//!   gradient term then integrates to zero over that part, so whatever the
//!   gradient, the mapped integral is the first order's: conservative
//!   exactly, under [`Normalization::DestArea`], even where a destination
//!   grid's edge or a mask cuts a source cell (where the plane is then
//!   centred off the cell's middle).
//! - **`g` comes from the cell's neighbours** (the eight round it in the
//!   source window, less any without data) by Green's theorem round the
//!   polygon of their centres, taken anticlockwise round `c`. It is exact
//!   for fields linear in source pixels. It is zero (first order for that
//!   cell) where there are fewer than three neighbours, `c` is outside
//!   their polygon (by more than 1e-14, ESMF's tolerance), or the polygon
//!   has no area: at a corner of the window, say.
//! - **A neighbour's centre** is its centroid under the table's measure:
//!   the middle of the cell for [`Measure::Planar`] (ESMF's corner mean),
//!   and for [`Measure::CosLatitude`] a little towards the equator, so
//!   fields linear in source pixels (in longitude and latitude) come out
//!   exactly there too.
//!
//! One difference from ESMF: an overlap's centre here is its centroid,
//! from the moments controlledburn's window gives exactly. ESMF's is the
//! mean of the overlap polygon's vertices (`_calc_centroid_2D_2D_cart` in
//! ESMCI_SM.C), which is the centroid for a rectangle, a triangle or a
//! parallelogram, but not for the general polygons a projected grid cuts
//! from a lon/lat one. So for a grid onto a grid the two give the same
//! weights, and elsewhere this one is exact for fields linear in source
//! pixels and ESMF's is not. Given ESMF's vertex means instead, it gives
//! ESMF's weights to rounding (`tests/esmf_second_order.rs`).
//!
//! ESMF's spherical method works in three dimensions on the sphere;
//! this works in source pixels, under the source's measure.
//!
//! [`Weights::apply_second_order`] estimates the gradients from the field
//! at hand, so a neighbour without data drops out of them as a cell
//! without data drops out of first-order [`Weights::apply`].
//! [`Weights::second_order`] folds them into one table, as ESMF's weight
//! files hold it, for a fixed set of cells with data.

use super::{ApplyOptions, GridWindow, Normalization, Weights};
use crate::warp::{Interleaved, Output, Res, Sample, SourceBuffer};
use controlledburn::{CoverageWindow, Measure};

/// ESMF's tolerance for the source centre being inside the neighbours'
/// polygon (`is_pnt_in_polygon`).
const INSIDE_TOL: f64 = 1e-14;

/// A neighbour of a source cell, in the gradient's polygon.
#[derive(Debug, Clone, Copy)]
struct Nbr {
    /// Its centre in source pixels (full grid).
    at: [f64; 2],
    /// The angle from the source cell's centre to it.
    angle: f64,
    /// Its coefficient in the gradient.
    grad: [f64; 2],
    /// Its index: a buffer pixel, or a window cell.
    p: usize,
}

/// The gradient's coefficients by Green's theorem round the neighbours'
/// centres, ESMF's `_set_grad_info_using_greens_2D_2D_cart`: with the
/// neighbours sorted anticlockwise round `centre`, each edge's outward
/// normal (times its length) is shared by its two ends, and `g = sum of
/// nbr.grad f_nbr - src f_src`. `None` (a zero gradient) for fewer than
/// three neighbours, a centre outside their polygon, or a polygon of no
/// area. Returns `src`, the source cell's own coefficient.
fn green(centre: [f64; 2], nbrs: &mut [Nbr]) -> Option<[f64; 2]> {
    if nbrs.len() < 3 {
        return None;
    }
    for n in nbrs.iter_mut() {
        n.angle = (n.at[1] - centre[1]).atan2(n.at[0] - centre[0]);
        n.grad = [0.0; 2];
    }
    nbrs.sort_by(|a, b| a.angle.total_cmp(&b.angle));
    let m = nbrs.len();
    // inside: no edge has the centre more than the tolerance to its right
    for i in 0..m {
        let (a, b) = (nbrs[i].at, nbrs[(i + 1) % m].at);
        let cross = (b[0] - a[0]) * (centre[1] - a[1]) - (b[1] - a[1]) * (centre[0] - a[0]);
        if cross <= -INSIDE_TOL {
            return None;
        }
    }
    let mut area = 0.0;
    for i in 0..m {
        let (a, b) = (nbrs[i].at, nbrs[(i + 1) % m].at);
        area += (a[0] - centre[0]) * (b[1] - centre[1]) - (b[0] - centre[0]) * (a[1] - centre[1]);
    }
    let area = 0.5 * area;
    if area == 0.0 {
        return None;
    }
    let mut src = [0.0; 2];
    let mut prev = m - 1;
    for i in 0..m {
        let e = [nbrs[i].at[0] - nbrs[prev].at[0], nbrs[i].at[1] - nbrs[prev].at[1]];
        let out = [e[1], -e[0]];
        for k in 0..2 {
            nbrs[i].grad[k] += out[k];
            nbrs[prev].grad[k] += out[k];
            src[k] += out[k];
        }
        prev = i;
    }
    let half = 0.5 / area;
    for n in nbrs.iter_mut() {
        n.grad = [n.grad[0] * half, n.grad[1] * half];
    }
    Some([src[0] / area, src[1] / area])
}

/// The centroid of a whole cell in row `row` under `measure`, from its
/// top-left corner: `(0.5, 0.5)` for [`Measure::Planar`].
fn cell_centroid(measure: Measure, row: i64) -> [f64; 2] {
    if let Measure::Planar = measure {
        return [0.5, 0.5];
    }
    let mut w = CoverageWindow::with_measure(measure);
    w.set_moments(true);
    w.reset(0, row, 1, 1);
    let r = row as f64;
    w.add_ring(&[0.0, 1.0, 1.0, 0.0], &[r, r, r + 1.0, r + 1.0], 1.0);
    let cov = w.resolve()[0];
    [w.moment_x()[0] / cov, w.moment_y()[0] / cov]
}

/// Cell centroids by row, worked out once each.
struct Centroids {
    measure: Measure,
    rows: std::collections::HashMap<i64, [f64; 2]>,
}

impl Centroids {
    fn new(measure: Measure) -> Self {
        Centroids { measure, rows: Default::default() }
    }

    /// The centroid of cell `(row, col)` (full grid) in source pixels.
    fn at(&mut self, row: i64, col: i64) -> [f64; 2] {
        let m = self.measure;
        let c = *self.rows.entry(row).or_insert_with(|| cell_centroid(m, row));
        [col as f64 + c[0], row as f64 + c[1]]
    }
}

/// The table's source cells, each once, with the centre of the part the
/// table covers, and each entry's cell and its overlap's first moment about
/// that centre.
struct Sources {
    /// Index (buffer pixel or window cell) of each source cell, ascending.
    p: Vec<usize>,
    /// Its full-grid `[row, col]`.
    rc: Vec<[i64; 2]>,
    /// The covered part's centroid, in full-grid source pixels.
    centre: Vec<[f64; 2]>,
    /// Each entry's source cell (into the above), `u32::MAX` for an entry
    /// outside the window.
    of: Vec<u32>,
    /// Each entry's overlap's first moment about its cell's centre, over
    /// the cell's measure: `frac` times (overlap centroid less the centre),
    /// as `moment - frac * centre`, so an overlap of next to no area has
    /// next to none, whatever rounding does to its centroid.
    dm: Vec<[f64; 2]>,
}

impl Weights {
    /// The source cells of the table within a window, by `index` (an
    /// entry to its index there, or `None` outside it).
    fn sources(&self, index: impl Fn(usize) -> Option<usize>) -> Sources {
        assert_eq!(
            self.moment.len(),
            self.len(),
            "second order needs each entry's moments: WeightOptions::with_moments(true)"
        );
        let mut by: Vec<(usize, u32)> = (0..self.len()).filter_map(|k| index(k).map(|p| (p, k as u32))).collect();
        by.sort_unstable();
        let mut s = Sources {
            p: Vec::new(),
            rc: Vec::new(),
            centre: Vec::new(),
            of: vec![u32::MAX; self.len()],
            dm: vec![[0.0; 2]; self.len()],
        };
        let mut i = 0;
        while i < by.len() {
            let p = by[i].0;
            let mut j = i;
            let (mut f, mut mx, mut my) = (0.0, 0.0, 0.0);
            while j < by.len() && by[j].0 == p {
                let k = by[j].1 as usize;
                f += self.frac[k];
                mx += self.moment[k][0];
                my += self.moment[k][1];
                j += 1;
            }
            let k0 = by[i].1 as usize;
            let (r, c) = (self.src_row[k0] as i64, self.src_col[k0] as i64);
            // the covered part's centroid, from the cell's corner
            let local = if f > 0.0 { [mx / f, my / f] } else { [0.5, 0.5] };
            let n = s.p.len() as u32;
            for &(_, k) in &by[i..j] {
                let k = k as usize;
                s.of[k] = n;
                let f = self.frac[k];
                s.dm[k] = [self.moment[k][0] - f * local[0], self.moment[k][1] - f * local[1]];
            }
            s.p.push(p);
            s.rc.push([r, c]);
            s.centre.push([c as f64 + local[0], r as f64 + local[1]]);
            i = j;
        }
        s
    }

    /// Second-order conservative remapping of the source buffer `src`
    /// (ESMF's `conserve_2nd`; see the module docs): [`Weights::apply`]
    /// with each source value replaced by its cell's plane at the
    /// overlap's centroid. The gradients come from `src` itself, band by
    /// band: neighbours outside the buffer or without data (as
    /// [`ApplyOptions`] says) are left out of them, and a cell without data
    /// is left out of its destinations, as in [`Weights::apply`].
    ///
    /// Under [`Normalization::DestArea`] the integral is conserved exactly
    /// (to rounding), as by the first order: the gradient terms integrate to
    /// zero over each source cell's covered part.
    ///
    /// # Panics
    ///
    /// As [`Weights::apply`], and if the table has no moments
    /// ([`WeightOptions::with_moments`](super::WeightOptions::with_moments)).
    pub fn apply_second_order<S: Sample, D: Sample>(
        &self,
        src: &SourceBuffer<'_, S>,
        norm: Normalization,
        opts: &ApplyOptions,
    ) -> Vec<D> {
        let nb = src.nbands;
        assert_eq!(src.data.len(), src.ncol * src.nrow * nb, "source buffer size mismatch");
        assert!(opts.mask_band.is_none_or(|a| a < nb), "mask_band out of range");
        let ncell = self.dst_size[0] * self.dst_size[1];
        assert_eq!(self.dst_area.len(), ncell, "dst_area size mismatch");
        let out_spec = Output::<D>::from_nodata(S::DATA_TYPE, nb, &opts.src_nodata, &opts.dst_nodata);
        let img = Interleaved {
            src: src.data,
            ncol: src.ncol,
            nrow: src.nrow,
            nb,
            mask: opts.mask_band,
            nodata: out_spec.src_nodata::<S>(),
            rule: opts.band_nodata,
            cut: None,
        };
        let (ro, co) = (src.row_off as i64, src.col_off as i64);
        let buf = |r: i64, c: i64| -> Option<usize> {
            let (r, c) = (r - ro, c - co);
            (r >= 0 && c >= 0 && (r as usize) < src.nrow && (c as usize) < src.ncol)
                .then(|| r as usize * src.ncol + c as usize)
        };
        let s = self.sources(|k| buf(self.src_row[k] as i64, self.src_col[k] as i64));
        // whether pixel p has data in band b
        let ok = |p: usize, b: usize| {
            let pix = img.px(p);
            img.pixel_ok(pix) && img.band_ok(pix, b)
        };
        // each source cell's gradient, band by band
        let mut cen = Centroids::new(self.measure);
        let mut grad = vec![[0.0f64; 2]; s.p.len() * nb];
        let mut nbrs: Vec<Nbr> = Vec::with_capacity(8);
        for (i, &[r, c]) in s.rc.iter().enumerate() {
            for b in 0..nb {
                if !ok(s.p[i], b) {
                    continue;
                }
                nbrs.clear();
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        if (dr, dc) == (0, 0) {
                            continue;
                        }
                        if let Some(p) = buf(r + dr, c + dc).filter(|&p| ok(p, b)) {
                            nbrs.push(Nbr { at: cen.at(r + dr, c + dc), angle: 0.0, grad: [0.0; 2], p });
                        }
                    }
                }
                if let Some(own) = green(s.centre[i], &mut nbrs) {
                    let f = |p: usize| img.px(p)[b].to_f64();
                    let mut g = [-own[0] * f(s.p[i]), -own[1] * f(s.p[i])];
                    for n in &nbrs {
                        g[0] += n.grad[0] * f(n.p);
                        g[1] += n.grad[1] * f(n.p);
                    }
                    grad[i * nb + b] = g;
                }
            }
        }
        let mut out = out_spec.filled(ncell);
        let order = self.grouped();
        let (mut sum, mut wsum) = (vec![0.0f64; nb], vec![0.0f64; nb]);
        let mut start = 0;
        while start < order.len() {
            let cell = self.cell(order[start]);
            let mut end = start + 1;
            while end < order.len() && self.cell(order[end]) == cell {
                end += 1;
            }
            sum.fill(0.0);
            wsum.fill(0.0);
            for &k in &order[start..end] {
                let i = s.of[k];
                if i == u32::MAX {
                    continue;
                }
                let (i, w, a, dm) = (i as usize, self.overlap(k), self.src_area(k), s.dm[k]);
                for b in 0..nb {
                    if ok(s.p[i], b) {
                        let g = grad[i * nb + b];
                        sum[b] += w * img.px(s.p[i])[b].to_f64() + a * (g[0] * dm[0] + g[1] * dm[1]);
                        wsum[b] += w;
                    }
                }
            }
            for b in 0..nb {
                let r = if wsum[b] > 0.0 {
                    let denom = match norm {
                        Normalization::FracArea => wsum[b],
                        Normalization::DestArea => self.dst_area[cell],
                    };
                    Res::<S>::Computed(sum[b] / denom)
                } else {
                    Res::None
                };
                if let Some(v) = out_spec.result(r, b) {
                    out[cell * nb + b] = v;
                }
            }
            start = end;
        }
        out
    }

    /// The second-order table folded into one, as ESMF's `conserve_2nd`
    /// weight files hold it: each destination cell's entries are its
    /// sources' and their neighbours', with `frac` such that
    /// [`Weights::normalised`] gives ESMF's `S` (for
    /// [`Normalization::DestArea`], its `destarea`), and [`Weights::apply`]
    /// gives [`Weights::apply_second_order`] for a field with data in every
    /// cell `valid` marks. Some `frac` are negative.
    ///
    /// `src` is the source window the gradients may draw on, and `valid`
    /// (row-major over it, `None` for every cell) the cells with data: a
    /// cell without is no one's neighbour and has no entries, as a masked
    /// cell in ESMF. Entries come grouped by destination cell, as the
    /// table's, and within a cell in source scan order, one per source
    /// cell. For a field with other cells without data, use
    /// [`Weights::apply_second_order`]: [`Weights::apply`] would drop their
    /// entries and leave the gradient terms unbalanced.
    ///
    /// # Panics
    ///
    /// If the table has no moments, or `valid` isn't the window's size.
    pub fn second_order(&self, src: &GridWindow, valid: Option<&[bool]>) -> Weights {
        let n = src.ncol * src.nrow;
        if let Some(v) = valid {
            assert_eq!(v.len(), n, "valid must have one value per cell of the source window");
        }
        let (ro, co) = (src.row_off as i64, src.col_off as i64);
        let at = |r: i64, c: i64| -> Option<usize> {
            let (r, c) = (r - ro, c - co);
            (r >= 0 && c >= 0 && (r as usize) < src.nrow && (c as usize) < src.ncol)
                .then(|| r as usize * src.ncol + c as usize)
                .filter(|&p| valid.is_none_or(|v| v[p]))
        };
        let s = self.sources(|k| at(self.src_row[k] as i64, self.src_col[k] as i64));
        // each source cell's coefficients: (window cell, coefficient) for
        // its neighbours, then its own
        let mut cen = Centroids::new(self.measure);
        let (mut start, mut coef): (Vec<usize>, Vec<(usize, [f64; 2])>) = (vec![0], Vec::new());
        let mut nbrs: Vec<Nbr> = Vec::with_capacity(8);
        for (i, &[r, c]) in s.rc.iter().enumerate() {
            nbrs.clear();
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if (dr, dc) == (0, 0) {
                        continue;
                    }
                    if let Some(p) = at(r + dr, c + dc) {
                        nbrs.push(Nbr { at: cen.at(r + dr, c + dc), angle: 0.0, grad: [0.0; 2], p });
                    }
                }
            }
            if let Some(own) = green(s.centre[i], &mut nbrs) {
                coef.extend(nbrs.iter().map(|n| (n.p, n.grad)));
                coef.push((s.p[i], [-own[0], -own[1]]));
            }
            start.push(coef.len());
        }
        let cell_area = |p: usize| self.measure.cell_area(ro + (p / src.ncol) as i64);
        let mut w = Weights {
            dst_off: self.dst_off,
            dst_size: self.dst_size,
            dst_area: self.dst_area.clone(),
            measure: self.measure,
            ..Weights::default()
        };
        let order = self.grouped();
        let mut acc: Vec<(usize, f64)> = Vec::new();
        let mut g = 0;
        while g < order.len() {
            let cell = self.cell(order[g]);
            let mut end = g + 1;
            while end < order.len() && self.cell(order[end]) == cell {
                end += 1;
            }
            acc.clear();
            for &k in &order[g..end] {
                let i = s.of[k];
                if i == u32::MAX {
                    continue;
                }
                let (i, f, dm) = (i as usize, self.frac[k], s.dm[k]);
                let a = cell_area(s.p[i]);
                acc.push((s.p[i], f));
                for &(p, c) in &coef[start[i]..start[i + 1]] {
                    acc.push((p, (a / cell_area(p)) * (c[0] * dm[0] + c[1] * dm[1])));
                }
            }
            acc.sort_by_key(|e| e.0);
            acc.dedup_by(|b, a| {
                if a.0 == b.0 {
                    a.1 += b.1;
                    true
                } else {
                    false
                }
            });
            let (dr, dc) = (self.dst_row[order[g]], self.dst_col[order[g]]);
            for &(p, f) in &acc {
                w.dst_row.push(dr);
                w.dst_col.push(dc);
                w.src_row.push((src.row_off + p / src.ncol) as u32);
                w.src_col.push((src.col_off + p % src.ncol) as u32);
                w.frac.push(f);
            }
            g = end;
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(r: f64, c: f64) -> Vec<Nbr> {
        let mut v = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                if (dr, dc) != (0, 0) {
                    let at = [c + dc as f64 + 0.5, r + dr as f64 + 0.5];
                    v.push(Nbr { at, angle: 0.0, grad: [0.0; 2], p: v.len() });
                }
            }
        }
        v
    }

    #[test]
    fn green_is_exact_for_a_plane_and_sobel_on_a_grid() {
        // f = 3 + 2 x - 5 y at the neighbours' centres
        let f = |p: [f64; 2]| 3.0 + 2.0 * p[0] - 5.0 * p[1];
        for centre in [[4.5, 7.5], [4.2, 7.9], [4.97, 7.03]] {
            let mut n = ring(7.0, 4.0);
            let own = green(centre, &mut n).unwrap();
            let mut g = [-own[0] * f(centre), -own[1] * f(centre)];
            for v in &n {
                g[0] += v.grad[0] * f(v.at);
                g[1] += v.grad[1] * f(v.at);
            }
            assert!((g[0] - 2.0).abs() < 1e-13 && (g[1] + 5.0).abs() < 1e-13, "{g:?}");
            // the neighbours' coefficients sum to the cell's own
            let sum = n.iter().fold([0.0; 2], |s, v| [s[0] + v.grad[0], s[1] + v.grad[1]]);
            assert!((sum[0] - own[0]).abs() < 1e-15 && (sum[1] - own[1]).abs() < 1e-15);
        }
        // on a regular grid the x coefficients are Sobel's: 1, 2, 1 over 8
        let mut n = ring(7.0, 4.0);
        green([4.5, 7.5], &mut n).unwrap();
        for v in &n {
            let dx = v.at[0] - 4.5;
            let want = dx.signum() * if v.at[1] == 7.5 { 0.25 } else { 0.125 };
            let want = if dx == 0.0 { 0.0 } else { want };
            assert!((v.grad[0] - want).abs() < 1e-15, "{:?}: {}", v.at, v.grad[0]);
        }
    }

    #[test]
    fn green_gives_up_where_the_neighbours_dont_surround_the_centre() {
        // a corner cell: three neighbours, the centre outside them
        let mut n: Vec<Nbr> = ring(0.0, 0.0).into_iter().filter(|v| v.at[0] > 0.0 && v.at[1] > 0.0).collect();
        assert_eq!(n.len(), 3);
        assert!(green([0.5, 0.5], &mut n).is_none());
        // an edge cell: five, the centre on their polygon's edge, which counts
        let mut n: Vec<Nbr> = ring(5.0, 0.0).into_iter().filter(|v| v.at[0] > 0.0).collect();
        assert_eq!(n.len(), 5);
        assert!(green([0.5, 5.5], &mut n).is_some());
        // ... and just outside it, which doesn't
        assert!(green([0.5 - 1e-13, 5.5], &mut n).is_none());
        // two neighbours
        let mut n: Vec<Nbr> = ring(5.0, 5.0).into_iter().take(2).collect();
        assert!(green([5.5, 5.5], &mut n).is_none());
    }

    #[test]
    fn cell_centroids_move_towards_the_equator() {
        assert_eq!(cell_centroid(Measure::Planar, 3), [0.5, 0.5]);
        // rows going south from 60 S: the centroid is above the middle
        let m = Measure::cos_latitude((-60.0f64).to_radians(), (-1.0f64).to_radians());
        let c = cell_centroid(m, 4);
        assert!((c[0] - 0.5).abs() < 1e-15 && c[1] < 0.5 && c[1] > 0.49, "{c:?}");
    }
}
