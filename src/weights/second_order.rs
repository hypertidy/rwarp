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
//! **A source with world cells** (geolocation arrays, a
//! [`QuadMesh`](crate::mesh::QuadMesh) through
//! [`MeshTransformer`](crate::mesh::MeshTransformer), a
//! [`MeshSet`](crate::mesh::MeshSet) through
//! [`SetTransformer`](crate::mesh::SetTransformer)) takes the same method
//! in its cells' index space under the walk's equal-area map, where its
//! overlaps are measured. There each cell is a unit square holding its
//! world area evenly, so the moments are exact centroids, a whole cell's
//! is its middle, and the neighbours are the eight round it in the full
//! grid, whichever piece, band or world their cells are in. It conserves
//! exactly, as above, and it is exact for fields linear in the grid's
//! indices. It is not exact for fields linear in world coordinates: the
//! equal-area map is affine only on each quarter of a cell, so a plane in
//! index space is off a world plane by about how far the cells are from
//! parallelograms. On a smooth grid that is second order too.
//! [`Weights::second_order`] needs the window's world cell areas for such
//! a source ([`source_cell_areas`](super::source_cell_areas)).
//!
//! **A mesh's faces as sources** ([`PolyMesh::weights_to`]) have no index
//! space, so their planes are in the grid's, where their overlaps are
//! measured (its pixels, or its cells' index space), through
//! [`PolyMesh::apply_second_order`] and [`PolyMesh::second_order`]. Each
//! overlap's moments are about its grid cell's corner. A face's centre is
//! the centroid of the part the table covers, and its neighbours are the
//! faces sharing a node with it (ESMF's), each centred the same way. A
//! face whose covered part isn't one piece in the grid's cells (cut by its
//! seam) is first order, and a neighbour whose cells aren't next to the
//! face's (across the seam) is left out.
//!
//! [`Weights::apply_second_order`] estimates the gradients from the field
//! at hand, so a neighbour without data drops out of them as a cell
//! without data drops out of first-order [`Weights::apply`].
//! [`Weights::second_order`] folds them into one table, as ESMF's weight
//! files hold it, for a fixed set of cells with data.

use super::{ApplyOptions, GridWindow, Normalization, Weights};
use crate::mesh::PolyMesh;
use crate::warp::{Interleaved, Output, Res, Sample, SourceBuffer};
use controlledburn::{CoverageWindow, Measure};

/// ESMF's tolerance for the source centre being inside the neighbours'
/// polygon (`is_pnt_in_polygon`).
const INSIDE_TOL: f64 = 1e-14;

/// An overlap of less than this much of its face is rounding (a window
/// leaves slivers of 1e-17 or so in cells the face doesn't reach): left out
/// of where the face is in the grid's cells.
const SLIVER: f64 = 1e-12;

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

/// A buffer's pixel for full-grid `(row, col)`, `None` outside it.
fn buffer_index<S>(src: &SourceBuffer<'_, S>) -> impl Fn(i64, i64) -> Option<usize> {
    let (ro, co, nrow, ncol) = (src.row_off as i64, src.col_off as i64, src.nrow, src.ncol);
    move |r: i64, c: i64| -> Option<usize> {
        let (r, c) = (r - ro, c - co);
        (r >= 0 && c >= 0 && (r as usize) < nrow && (c as usize) < ncol).then(|| r as usize * ncol + c as usize)
    }
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
pub(crate) struct Sources {
    /// Index (buffer pixel or window cell) of each source cell, ascending.
    p: Vec<usize>,
    /// Its full-grid `[row, col]`.
    rc: Vec<[i64; 2]>,
    /// The covered part's centroid, in full-grid source pixels.
    centre: Vec<[f64; 2]>,
    /// The first of its entries.
    k0: Vec<u32>,
    /// Each entry's source cell (into the above), `u32::MAX` for an entry
    /// outside the window.
    of: Vec<u32>,
    /// Each entry's overlap's first moment about its cell's centre, over
    /// the cell's measure: `frac` times (overlap centroid less the centre),
    /// as `moment - frac * centre`, so an overlap of next to no area has
    /// next to none, whatever rounding does to its centroid.
    dm: Vec<[f64; 2]>,
}

/// A source cell's candidate neighbour for its gradient: its index (as
/// [`Sources::p`]) and its centre.
type Cand = (usize, [f64; 2]);

/// Each source cell's candidate neighbours, into the vector (empty to
/// start with): those without data are left out of them later.
type Stencil<'a> = dyn FnMut(usize, &mut Vec<Cand>) + 'a;

impl Weights {
    /// The source cells of the table within a window, by `index` (an
    /// entry to its index there, or `None` outside it), each entry's
    /// moments about `origin(k)` (in full-grid pixels): its source cell's
    /// corner for a grid source, its grid cell's for a face.
    fn sources(&self, index: impl Fn(usize) -> Option<usize>, origin: impl Fn(usize) -> [f64; 2]) -> Sources {
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
            k0: Vec::new(),
            of: vec![u32::MAX; self.len()],
            dm: vec![[0.0; 2]; self.len()],
        };
        let mut i = 0;
        while i < by.len() {
            let p = by[i].0;
            let k0 = by[i].1 as usize;
            let o = origin(k0);
            // each entry's origin from the first's: none for a grid source
            let off = |k: usize| {
                let ok = origin(k);
                [ok[0] - o[0], ok[1] - o[1]]
            };
            let mut j = i;
            let (mut f, mut mx, mut my) = (0.0, 0.0, 0.0);
            while j < by.len() && by[j].0 == p {
                let k = by[j].1 as usize;
                let d = off(k);
                f += self.frac[k];
                mx += self.moment[k][0] + self.frac[k] * d[0];
                my += self.moment[k][1] + self.frac[k] * d[1];
                j += 1;
            }
            let (r, c) = (self.src_row[k0] as i64, self.src_col[k0] as i64);
            // the covered part's centroid, from the first entry's origin
            let local = if f > 0.0 { [mx / f, my / f] } else { [0.5, 0.5] };
            let n = s.p.len() as u32;
            for &(_, k) in &by[i..j] {
                let k = k as usize;
                s.of[k] = n;
                let (f, d) = (self.frac[k], off(k));
                s.dm[k] = [self.moment[k][0] - f * (local[0] - d[0]), self.moment[k][1] - f * (local[1] - d[1])];
            }
            s.p.push(p);
            s.rc.push([r, c]);
            s.centre.push([o[0] + local[0], o[1] + local[1]]);
            s.k0.push(k0 as u32);
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
    /// For a table from a mesh's faces ([`PolyMesh::weights_to`]), use
    /// [`PolyMesh::apply_second_order`]: here the sources are a grid's
    /// cells, and its neighbours the cells round each.
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
        let buf = buffer_index(src);
        let s = self.sources(|k| buf(self.src_row[k] as i64, self.src_col[k] as i64), |k| {
            [self.src_col[k] as f64, self.src_row[k] as f64]
        });
        let mut cen = Centroids::new(self.measure);
        self.apply_planes(src, norm, opts, &s, &mut |i, out| {
            let [r, c] = s.rc[i];
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if (dr, dc) == (0, 0) {
                        continue;
                    }
                    if let Some(p) = buf(r + dr, c + dc) {
                        out.push((p, cen.at(r + dr, c + dc)));
                    }
                }
            }
        })
    }

    /// [`Weights::apply_second_order`] for the source cells `s` (indexed
    /// by buffer pixel), each one's candidate neighbours from `stencil`
    /// (by buffer pixel too).
    pub(crate) fn apply_planes<S: Sample, D: Sample>(
        &self,
        src: &SourceBuffer<'_, S>,
        norm: Normalization,
        opts: &ApplyOptions,
        s: &Sources,
        stencil: &mut Stencil<'_>,
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
        // whether pixel p has data in band b
        let ok = |p: usize, b: usize| {
            let pix = img.px(p);
            img.pixel_ok(pix) && img.band_ok(pix, b)
        };
        // each source cell's gradient, band by band
        let mut grad = vec![[0.0f64; 2]; s.p.len() * nb];
        let mut cands: Vec<Cand> = Vec::with_capacity(12);
        let mut nbrs: Vec<Nbr> = Vec::with_capacity(12);
        for i in 0..s.p.len() {
            cands.clear();
            stencil(i, &mut cands);
            for b in 0..nb {
                if !ok(s.p[i], b) {
                    continue;
                }
                nbrs.clear();
                nbrs.extend(cands.iter().filter(|c| ok(c.0, b)).map(|&(p, at)| Nbr { at, angle: 0.0, grad: [0.0; 2], p }));
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
    /// `areas` is for a table from a source with world cells (one with
    /// [`Weights::src_cell_area`]): each window cell's world area,
    /// row-major, from [`source_cell_areas`](super::source_cell_areas) for
    /// the same window. The folded table then has
    /// [`Weights::src_cell_area`] too. A cell whose area isn't positive
    /// (one in none of a set's pieces, or of no area) counts as without
    /// data; for [`Weights::apply_second_order`] to give the same, mark it
    /// so there too. `None` for a source measured in its pixels.
    ///
    /// For a table from a mesh's faces ([`PolyMesh::weights_to`]), use
    /// [`PolyMesh::second_order`].
    ///
    /// # Panics
    ///
    /// If the table has no moments, if `valid` or `areas` isn't the
    /// window's size, or if `areas` is given for a table from a source
    /// measured in its pixels or missing for one with world cells.
    pub fn second_order(&self, src: &GridWindow, valid: Option<&[bool]>, areas: Option<&[f64]>) -> Weights {
        let n = src.ncol * src.nrow;
        if let Some(v) = valid {
            assert_eq!(v.len(), n, "valid must have one value per cell of the source window");
        }
        let world = !self.src_cell_area.is_empty();
        match areas {
            Some(a) => {
                // an empty table can't say what its source was
                assert!(world || self.is_empty(), "areas are for a table from a source with world cells");
                assert_eq!(a.len(), n, "areas must have one value per cell of the source window");
            }
            None => assert!(
                !world,
                "a table from a source with world cells needs the window's cell areas (weights::source_cell_areas)"
            ),
        }
        let (ro, co) = (src.row_off as i64, src.col_off as i64);
        let at = |r: i64, c: i64| -> Option<usize> {
            let (r, c) = (r - ro, c - co);
            (r >= 0 && c >= 0 && (r as usize) < src.nrow && (c as usize) < src.ncol)
                .then(|| r as usize * src.ncol + c as usize)
                .filter(|&p| valid.is_none_or(|v| v[p]))
                .filter(|&p| areas.is_none_or(|a| a[p] > 0.0))
        };
        let s = self.sources(|k| at(self.src_row[k] as i64, self.src_col[k] as i64), |k| {
            [self.src_col[k] as f64, self.src_row[k] as f64]
        });
        let mut cen = Centroids::new(self.measure);
        let mut stencil = |i: usize, out: &mut Vec<Cand>| {
            let [r, c] = s.rc[i];
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if (dr, dc) == (0, 0) {
                        continue;
                    }
                    if let Some(p) = at(r + dr, c + dc) {
                        out.push((p, cen.at(r + dr, c + dc)));
                    }
                }
            }
        };
        let cell_area = |p: usize| match areas {
            Some(a) => a[p],
            None => self.measure.cell_area(ro + (p / src.ncol) as i64),
        };
        let place = |p: usize| ((src.row_off + p / src.ncol) as u32, (src.col_off + p % src.ncol) as u32);
        self.fold_planes(&s, &mut stencil, &cell_area, &place, world)
    }

    /// [`Weights::second_order`] for the source cells `s` (by window
    /// cell, or face), each one's candidate neighbours from `stencil`,
    /// each window cell's measure `cell_area(p)` and full-grid `(row,
    /// col)` `place(p)`; with `areas` each entry's source cell measure in
    /// [`Weights::src_cell_area`].
    pub(crate) fn fold_planes(
        &self,
        s: &Sources,
        stencil: &mut Stencil<'_>,
        cell_area: &dyn Fn(usize) -> f64,
        place: &dyn Fn(usize) -> (u32, u32),
        areas: bool,
    ) -> Weights {
        // each source cell's coefficients: (window cell, coefficient) for
        // its neighbours, then its own
        let (mut start, mut coef): (Vec<usize>, Vec<(usize, [f64; 2])>) = (vec![0], Vec::new());
        let mut cands: Vec<Cand> = Vec::with_capacity(12);
        let mut nbrs: Vec<Nbr> = Vec::with_capacity(12);
        for i in 0..s.p.len() {
            cands.clear();
            stencil(i, &mut cands);
            nbrs.clear();
            nbrs.extend(cands.iter().map(|&(p, at)| Nbr { at, angle: 0.0, grad: [0.0; 2], p }));
            if let Some(own) = green(s.centre[i], &mut nbrs) {
                coef.extend(nbrs.iter().map(|n| (n.p, n.grad)));
                coef.push((s.p[i], [-own[0], -own[1]]));
            }
            start.push(coef.len());
        }
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
                let (sr, sc) = place(p);
                w.dst_row.push(dr);
                w.dst_col.push(dc);
                w.src_row.push(sr);
                w.src_col.push(sc);
                w.frac.push(f);
                if areas {
                    w.src_cell_area.push(cell_area(p));
                }
            }
            g = end;
        }
        w
    }
}

/// A mesh's faces as a table's sources ([`PolyMesh::weights_to`]): each
/// face's candidate neighbours are the faces sharing a node with it
/// (ESMF's), centred on the parts of them the table covers. Only faces
/// whose covered part is one piece in the grid's cells (8-connected,
/// slivers aside) take part, and a neighbour only if its cells' bounding
/// box is next to the face's: a face the grid's seam cuts, or one on its
/// far side, has no centre there that means anything.
struct FaceStencil<'a> {
    mesh: &'a PolyMesh,
    /// The sources at each node: `at_node[start[n]..start[n + 1]]`.
    start: Vec<usize>,
    at_node: Vec<u32>,
    /// Each source's covered part is one piece.
    placed: Vec<bool>,
    /// Its cells' bounding box: first and last row, first and last column.
    bbox: Vec<[u32; 4]>,
}

impl<'a> FaceStencil<'a> {
    fn new(mesh: &'a PolyMesh, w: &Weights, s: &Sources) -> Self {
        let n = s.p.len();
        // each source's cells, (row, col), sorted
        let mut cells: Vec<(u32, u32, u32)> = (0..w.len())
            .filter(|&k| s.of[k] != u32::MAX && w.frac[k] > SLIVER)
            .map(|k| (s.of[k], w.dst_row[k], w.dst_col[k]))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let (mut placed, mut bbox) = (vec![false; n], vec![[0u32; 4]; n]);
        let mut a = 0;
        while a < cells.len() {
            let i = cells[a].0 as usize;
            let mut b = a;
            while b < cells.len() && cells[b].0 as usize == i {
                b += 1;
            }
            let rc: Vec<(u32, u32)> = cells[a..b].iter().map(|t| (t.1, t.2)).collect();
            placed[i] = connected(&rc);
            let (c0, c1) = rc.iter().fold((u32::MAX, 0), |(lo, hi), t| (lo.min(t.1), hi.max(t.1)));
            bbox[i] = [rc[0].0, rc[rc.len() - 1].0, c0, c1];
            a = b;
        }
        let face = |i: usize| s.rc[i][1] as usize;
        let mut start = vec![0usize; mesh.nnode() + 1];
        for i in 0..n {
            for &v in mesh.face(face(i)) {
                start[v + 1] += 1;
            }
        }
        for v in 0..mesh.nnode() {
            start[v + 1] += start[v];
        }
        let mut fill = start.clone();
        let mut at_node = vec![0u32; start[mesh.nnode()]];
        for i in 0..n {
            for &v in mesh.face(face(i)) {
                at_node[fill[v]] = i as u32;
                fill[v] += 1;
            }
        }
        FaceStencil { mesh, start, at_node, placed, bbox }
    }

    /// Source `i`'s candidates, by its face's nodes.
    fn cands(&self, s: &Sources, i: usize, out: &mut Vec<Cand>) {
        if !self.placed[i] {
            return;
        }
        let b = self.bbox[i];
        let next = |c: [u32; 4]| {
            c[0] as i64 <= b[1] as i64 + 1
                && c[1] as i64 >= b[0] as i64 - 1
                && c[2] as i64 <= b[3] as i64 + 1
                && c[3] as i64 >= b[2] as i64 - 1
        };
        for &v in self.mesh.face(s.rc[i][1] as usize) {
            for &j in &self.at_node[self.start[v]..self.start[v + 1]] {
                let j = j as usize;
                if j != i && self.placed[j] && next(self.bbox[j]) {
                    out.push((s.p[j], s.centre[j]));
                }
            }
        }
        out.sort_unstable_by_key(|c| c.0);
        out.dedup_by_key(|c| c.0);
    }
}

/// Whether the cells `rc` (sorted, distinct) are one piece, each reached
/// from the first through cells sharing a side or a corner.
fn connected(rc: &[(u32, u32)]) -> bool {
    let mut seen = vec![false; rc.len()];
    let mut todo = vec![0usize];
    seen[0] = true;
    let mut reached = 1;
    while let Some(i) = todo.pop() {
        let (r, c) = (rc[i].0 as i64, rc[i].1 as i64);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, c) = (r + dr, c + dc);
                if r < 0 || c < 0 || r > u32::MAX as i64 || c > u32::MAX as i64 {
                    continue;
                }
                if let Ok(j) = rc.binary_search(&(r as u32, c as u32)) {
                    if !seen[j] {
                        seen[j] = true;
                        reached += 1;
                        todo.push(j);
                    }
                }
            }
        }
    }
    reached == rc.len()
}

impl Weights {
    /// Panics unless the table could be one from
    /// [`PolyMesh::weights_to`] on `mesh`.
    fn check_faces(&self, mesh: &PolyMesh) {
        assert!(
            self.src_row.iter().all(|&r| r == 0) && self.src_col.iter().all(|&c| (c as usize) < mesh.nface()),
            "not a table from this mesh's faces (PolyMesh::weights_to): a source isn't one of its faces"
        );
        assert_eq!(
            self.src_cell_area.len(),
            self.len(),
            "not a table from a mesh's faces (PolyMesh::weights_to): it has no face measures"
        );
    }

    /// [`PolyMesh::apply_second_order`].
    pub(crate) fn apply_faces_second_order<S: Sample, D: Sample>(
        &self,
        mesh: &PolyMesh,
        src: &SourceBuffer<'_, S>,
        norm: Normalization,
        opts: &ApplyOptions,
    ) -> Vec<D> {
        self.check_faces(mesh);
        let buf = buffer_index(src);
        let s = self.sources(|k| buf(0, self.src_col[k] as i64), |k| [self.dst_col[k] as f64, self.dst_row[k] as f64]);
        let faces = FaceStencil::new(mesh, self, &s);
        self.apply_planes(src, norm, opts, &s, &mut |i, out| faces.cands(&s, i, out))
    }

    /// [`PolyMesh::second_order`].
    pub(crate) fn faces_second_order(&self, mesh: &PolyMesh, valid: Option<&[bool]>) -> Weights {
        self.check_faces(mesh);
        let nface = mesh.nface();
        if let Some(v) = valid {
            assert_eq!(v.len(), nface, "valid must have one value per face");
        }
        let s = self.sources(
            |k| Some(self.src_col[k] as usize).filter(|&f| valid.is_none_or(|v| v[f])),
            |k| [self.dst_col[k] as f64, self.dst_row[k] as f64],
        );
        let faces = FaceStencil::new(mesh, self, &s);
        let mut area = vec![f64::NAN; nface];
        for (i, &k) in s.k0.iter().enumerate() {
            area[s.p[i]] = self.src_cell_area[k as usize];
        }
        self.fold_planes(&s, &mut |i, out| faces.cands(&s, i, out), &|p| area[p], &|p| (0, p as u32), true)
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
    fn cells_are_one_piece_through_sides_and_corners() {
        assert!(connected(&[(4, 4)]));
        assert!(connected(&[(0, 0), (0, 1), (1, 2), (2, 1)]));
        assert!(!connected(&[(0, 0), (0, 1), (0, 3)]));
        assert!(!connected(&[(0, 0), (2, 0)]));
    }

    #[test]
    fn a_sliver_of_rounding_doesn_t_split_a_face() {
        // one face over cells (0, 0) and (0, 1), and a sliver in (0, 5)
        let mesh = PolyMesh::new(vec![0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0], vec![0, 3], vec![0, 1, 2]).unwrap();
        let mut w = Weights {
            dst_row: vec![0; 3],
            dst_col: vec![0, 1, 5],
            src_row: vec![0; 3],
            src_col: vec![0; 3],
            frac: vec![0.5, 0.5, 1e-17],
            src_cell_area: vec![2.0; 3],
            moment: vec![[0.25, 0.25], [0.25, 0.25], [5e-18, 5e-18]],
            dst_area: vec![1.0; 6],
            dst_off: [0, 0],
            dst_size: [6, 1],
            measure: Measure::Planar,
        };
        let placed = |w: &Weights| {
            let s = w.sources(|k| Some(w.src_col[k] as usize), |k| [w.dst_col[k] as f64, w.dst_row[k] as f64]);
            let f = FaceStencil::new(&mesh, w, &s);
            (f.placed[0], f.bbox[0])
        };
        assert_eq!(placed(&w), (true, [0, 0, 0, 1]));
        // a real overlap there would
        w.frac[2] = 1e-6;
        assert_eq!(placed(&w), (false, [0, 0, 0, 5]));
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
