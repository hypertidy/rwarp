//! Conservative remapping weights from the exact footprint, as values.
//!
//! [`exact_weights`] measures each destination pixel's outline against the
//! source pixels exactly, as [`warp`](crate::warp::warp) does with
//! [`Footprint::Exact`](crate::warp::Footprint::Exact), and returns the
//! overlaps instead of reducing them: one entry per overlapping
//! (destination cell, source cell) pair, with the overlap's area as a
//! fraction of the source cell. That is the sparse matrix of first-order
//! conservative remapping (ESMF's `conserve`), computed once and applied to
//! any number of fields with [`Weights::apply`].
//!
//! # What the numbers are
//!
//! - **Cells.** Destination cells have edges straight in destination pixel
//!   space, each split into `densify` segments whose end points go through
//!   the transformer exactly. Source cells are unit squares in source pixel
//!   space.
//! - **Measure.** By default areas are planar in source pixel space: for a
//!   source whose pixel-to-world map is affine (any regular grid in its own
//!   CRS) that is world area in the source CRS, up to one constant. With
//!   [`WeightOptions::with_measure`] and [`lonlat_measure`], a regular
//!   longitude/latitude source measures area on the sphere instead
//!   (controlledburn's [`Measure::CosLatitude`]): exact for the cells as
//!   defined here, whose edges are straight in source pixel space.
//! - **`frac`** is the overlap's measure over the source cell's (1 for the
//!   planar measure), unrounded `f64`. Because neighbouring destination outlines are built
//!   from the same transformed points, the destination cells tile source
//!   pixel space: the `frac` of a source cell sums, over destination cells,
//!   to 1 where the destination grid covers it completely and to the
//!   covered fraction where its edge cuts it, up to `f64` rounding (about
//!   1e-15; see `tests/weights.rs`).
//! - **`dst_area`** is the measure of each destination cell's outline
//!   (controlledburn's [`Measure::ring_area`]), whether or not
//!   the outline lies inside the source window. `NaN` where a point of
//!   the outline failed to transform. A destination cell wholly inside the
//!   source has `dst_area` equal to the sum of its `frac` times the source
//!   cell's measure ([`Weights::src_area`], 1 for the planar measure).
//!
//! # Normalisation
//!
//! ESMF's weight `S` for an entry follows from `frac` and `dst_area` (see
//! [`Weights::normalised`]):
//!
//! With `a` the source cell's measure ([`Weights::src_area`]; 1 for the
//! planar measure):
//!
//! - [`Normalization::FracArea`]: `frac a / (sum of frac a over the
//!   destination cell)`. The mean over the part of the cell that has source; what
//!   [`warp`](crate::warp::warp) computes. ESMF `fracarea`, xESMF
//!   `conserve_normed`.
//! - [`Normalization::DestArea`]: `frac a / dst_area`. Partly covered cells
//!   are scaled down by their coverage, so `sum_i value_i * dst_area_i` is
//!   `sum_j frac_j a_j src_j`: the integral is conserved. ESMF `destarea`,
//!   xESMF `conserve`.
//!
//! # ESMF and SCRIP weight files
//!
//! Writing files is wxylib's job; the columns map directly. With full-grid
//! widths `dst_ncol_full` and `src_ncol_full`, ESMF's 1-based `row` is
//! `dst_row * dst_ncol_full + dst_col + 1`, `col` is `src_row *
//! src_ncol_full + src_col + 1` ([`Weights::esmf_row_col`]), and `S` is
//! [`Weights::normalised`]. ESMF's `area_b` corresponds to `dst_area`
//! (in source pixels, or under the spherical measure source pixels at the
//! equator, not steradians) and `frac_b` to [`Weights::dst_covered`] over
//! `dst_area`.
//!
//! # Relation to `warp()`
//!
//! [`warp`](crate::warp::warp) with `Footprint::Exact` and
//! `ResampleAlg::Average` uses the same lattice and the same per-pixel
//! coverage step, and reduces the weights directly instead of storing
//! them. It also snaps coverage within 1e-6 of 1 to 1, drops coverage at or
//! below 1e-6 and rounds the rest to `f32`, for compatibility with rwarp
//! 0.2. [`WeightOptions::warp_compatible`] does the same here, and then
//! [`Weights::apply`] with [`Normalization::FracArea`] reproduces `warp()`
//! bit for bit (tested in `tests/weights.rs`). The defaults keep every
//! positive overlap, unrounded.
//!
//! # Tiling
//!
//! With a destination offset ([`WeightOptions::with_dst_offset`]), the
//! weights of a destination window are those of the whole grid restricted
//! to it, because the window's lattice is the whole grid's lattice. Row
//! bands concatenate to the whole grid's weights exactly. Column offsets
//! do too when `densify` is a power of two (lattice points `off + k /
//! densify` are then exact); otherwise points on a tile can differ from
//! the whole grid's in the last bit.
//!
//! # Limits
//!
//! No antimeridian handling: a destination cell whose outline crosses the
//! source's wrap column folds across the whole source width, and its
//! weights are wrong. The spherical measure takes cell edges straight in
//! source pixel space (rhumb lines on a longitude/latitude source), not
//! great circles as ESMF does; see `design/conservative.md` and
//! `tests/esmf.rs` for the size of the difference.

pub use controlledburn::Measure;

use crate::transform::Transformer;
use crate::warp::{AreaMean, AreaReducer, BandNodata, Cover, DstNodata, Interleaved, Lattice, Nodata, Output, Res, Sample, SourceBuffer, WarpOptions};

/// A window of a grid, with no data: `ncol` by `nrow` cells at
/// `(col_off, row_off)` in the full grid. The source side of
/// [`exact_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GridWindow {
    pub ncol: usize,
    pub nrow: usize,
    pub col_off: usize,
    pub row_off: usize,
}

impl GridWindow {
    /// A whole grid (offset zero).
    pub fn new(ncol: usize, nrow: usize) -> Self {
        GridWindow { ncol, nrow, col_off: 0, row_off: 0 }
    }

    /// The window at `(col_off, row_off)` in the full grid.
    pub fn with_offset(mut self, col_off: usize, row_off: usize) -> Self {
        self.col_off = col_off;
        self.row_off = row_off;
        self
    }
}

impl<T> From<&SourceBuffer<'_, T>> for GridWindow {
    fn from(b: &SourceBuffer<'_, T>) -> Self {
        GridWindow { ncol: b.ncol, nrow: b.nrow, col_off: b.col_off, row_off: b.row_off }
    }
}

/// The spherical measure for a source on a regular longitude/latitude grid
/// (degrees) with geotransform `src_gt`, over its full pixel space: rows are
/// parallels, so the sphere's area element is a function of the row. `None`
/// for a rotated or degenerate geotransform.
///
/// Areas come out per unit of source column, in units of a source pixel at
/// the equator; the constant does not matter to the weights.
pub fn lonlat_measure(src_gt: [f64; 6]) -> Option<Measure> {
    (src_gt[2] == 0.0 && src_gt[4] == 0.0 && src_gt[1] != 0.0 && src_gt[5] != 0.0)
        .then(|| Measure::cos_latitude(src_gt[3].to_radians(), src_gt[5].to_radians()))
}

/// `measure` with its row 0 moved to row `rows` of the grid it was given for.
fn shift_rows(measure: Measure, rows: f64) -> Measure {
    match measure {
        Measure::CosLatitude { phi0, dphi } => Measure::cos_latitude(phi0 + dphi * rows, dphi),
        m => m,
    }
}

/// Options for [`exact_weights`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct WeightOptions {
    /// Segments per destination pixel edge (at least 1), as in
    /// [`Footprint::Exact`](crate::warp::Footprint::Exact).
    pub densify: usize,
    /// Offset of the destination window in the full destination grid.
    pub dst_col_off: usize,
    pub dst_row_off: usize,
    /// Overlaps at or below this are dropped. Default 0: every positive
    /// overlap is kept.
    pub drop_tolerance: f64,
    /// Overlaps within this of 1 become exactly 1. Default 0: none are
    /// snapped.
    pub snap_tolerance: f64,
    /// Round overlaps that are not snapped to `f32` (and back). Default
    /// false.
    pub round_to_f32: bool,
    /// How area is measured, over the full source grid's pixel space.
    /// Default [`Measure::Planar`]; see [`lonlat_measure`].
    pub measure: Measure,
}

impl Default for WeightOptions {
    fn default() -> Self {
        WeightOptions::new(1)
    }
}

impl WeightOptions {
    /// Unrounded weights with `densify` segments per edge, nothing dropped.
    pub fn new(densify: usize) -> Self {
        WeightOptions {
            densify, dst_col_off: 0, dst_row_off: 0,
            drop_tolerance: 0.0, snap_tolerance: 0.0, round_to_f32: false,
            measure: Measure::Planar,
        }
    }

    /// The weights [`warp`](crate::warp::warp) uses for
    /// `Footprint::Exact { densify }`: within 1e-6 of 1 is 1, at or below
    /// 1e-6 is dropped, the rest rounded to `f32`.
    pub fn warp_compatible(densify: usize) -> Self {
        WeightOptions { drop_tolerance: 1e-6, snap_tolerance: 1e-6, round_to_f32: true, ..WeightOptions::new(densify) }
    }

    pub fn with_densify(mut self, densify: usize) -> Self {
        self.densify = densify;
        self
    }

    /// Offset of the destination window in the full grid; see the module
    /// docs on tiling.
    pub fn with_dst_offset(mut self, dst_col_off: usize, dst_row_off: usize) -> Self {
        self.dst_col_off = dst_col_off;
        self.dst_row_off = dst_row_off;
        self
    }

    pub fn with_drop_tolerance(mut self, tol: f64) -> Self {
        self.drop_tolerance = tol;
        self
    }

    pub fn with_snap_tolerance(mut self, tol: f64) -> Self {
        self.snap_tolerance = tol;
        self
    }

    pub fn with_f32_rounding(mut self, round: bool) -> Self {
        self.round_to_f32 = round;
        self
    }

    /// Measure area with `measure`, given over the full source grid's pixel
    /// space (see [`lonlat_measure`]).
    pub fn with_measure(mut self, measure: Measure) -> Self {
        self.measure = measure;
        self
    }

    /// The stored weight for a raw coverage `f`, or `None` to drop it.
    #[inline]
    fn weight(&self, f: f64) -> Option<f64> {
        if self.snap_tolerance > 0.0 && f >= 1.0 - self.snap_tolerance {
            return Some(1.0);
        }
        if f <= self.drop_tolerance || f.is_nan() {
            return None;
        }
        Some(if self.round_to_f32 { f as f32 as f64 } else { f })
    }
}

/// How [`Weights::normalised`] and [`Weights::apply`] turn overlaps into
/// destination values; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Normalization {
    /// Over the covered (and, in `apply`, valid) part of the destination
    /// cell: the area-weighted mean. Matches `warp()`.
    #[default]
    FracArea,
    /// Over the whole destination cell (`dst_area`): conserves the
    /// integral.
    DestArea,
}

/// The weights of one destination cell, for [`exact_weights_with`].
/// Entries are in source scan order (row, then column).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CellWeights<'a> {
    /// Full-grid indices of the destination cell.
    pub dst_row: u32,
    pub dst_col: u32,
    /// Its outline's measure; `NaN` if it failed.
    pub dst_area: f64,
    /// Full-grid source indices and overlaps (fractions of the source
    /// cell's measure), one per entry.
    pub src_row: &'a [u32],
    pub src_col: &'a [u32],
    pub frac: &'a [f64],
}

/// Overlap of each destination cell with each source cell, from the exact
/// footprint; see the module docs.
///
/// Entries are grouped by destination cell, in destination row-major order,
/// and within a cell in source scan order.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Weights {
    /// Full-grid indices (offsets applied), one entry per overlapping pair.
    pub dst_row: Vec<u32>,
    pub dst_col: Vec<u32>,
    pub src_row: Vec<u32>,
    pub src_col: Vec<u32>,
    /// Measure of the overlap as a fraction of the source cell's.
    pub frac: Vec<f64>,
    /// Per destination cell of the window, row-major: its outline's
    /// measure, `NaN` where it failed to transform.
    pub dst_area: Vec<f64>,
    /// The destination window: offset `[col, row]` in the full grid and
    /// size `[ncol, nrow]`. `dst_area` has `ncol * nrow` values.
    pub dst_off: [usize; 2],
    pub dst_size: [usize; 2],
    /// The measure over the full source grid ([`WeightOptions::measure`]).
    pub measure: Measure,
}

/// Options for [`Weights::apply`]: nodata and validity, as in
/// [`WarpOptions`].
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct ApplyOptions {
    pub src_nodata: Nodata,
    pub dst_nodata: DstNodata,
    pub band_nodata: BandNodata,
    /// A band whose zero marks the whole source pixel invalid.
    pub mask_band: Option<usize>,
}

impl ApplyOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_src_nodata(mut self, nodata: Nodata) -> Self {
        self.src_nodata = nodata;
        self
    }

    pub fn with_dst_nodata(mut self, nodata: DstNodata) -> Self {
        self.dst_nodata = nodata;
        self
    }

    pub fn with_band_nodata(mut self, rule: BandNodata) -> Self {
        self.band_nodata = rule;
        self
    }

    pub fn with_mask_band(mut self, band: Option<usize>) -> Self {
        self.mask_band = band;
        self
    }
}

impl From<&WarpOptions> for ApplyOptions {
    /// The nodata and validity settings of a warp.
    fn from(o: &WarpOptions) -> Self {
        ApplyOptions {
            src_nodata: o.src_nodata.clone(),
            dst_nodata: o.dst_nodata.clone(),
            band_nodata: o.band_nodata,
            mask_band: o.mask_band,
        }
    }
}

fn to_u32(v: usize, what: &str) -> u32 {
    u32::try_from(v).unwrap_or_else(|_| panic!("{what} {v} does not fit in u32"))
}

/// Exact-footprint weights of a `dst_ncol` by `dst_nrow` destination window
/// (at the offset in `opts`) against the source window `src`, calling `f`
/// once per destination cell, in row-major order, with its entries. Cells
/// with no entries are passed too (with `dst_area`). Nothing is stored
/// beyond one cell, so this streams weights for problems too large to hold.
///
/// `transformer` maps destination pixels to full-grid source pixels, as
/// for [`warp`](crate::warp::warp).
pub fn exact_weights_with(
    transformer: &impl Transformer,
    src: &GridWindow,
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WeightOptions,
    mut f: impl FnMut(CellWeights<'_>),
) {
    let (ncol, nrow) = (src.ncol as i64, src.nrow as i64);
    to_u32(src.col_off + src.ncol, "source column");
    to_u32(src.row_off + src.nrow, "source row");
    to_u32(opts.dst_col_off + dst_ncol, "destination column");
    to_u32(opts.dst_row_off + dst_nrow, "destination row");
    let src_off = (src.col_off as f64, src.row_off as f64);
    let (mut sr, mut sc, mut fr) = (Vec::new(), Vec::new(), Vec::new());
    if dst_ncol == 0 || dst_nrow == 0 { return; }
    let mut lat = Lattice::new(transformer, dst_ncol, opts.densify.max(1), (opts.dst_col_off, opts.dst_row_off));
    // The window works in source window pixels.
    let measure = shift_rows(opts.measure, src.row_off as f64);
    lat.win.set_measure(measure);
    for row in 0..dst_nrow {
        lat.row(transformer, row);
        let dst_row = (opts.dst_row_off + row) as u32;
        for c in 0..dst_ncol {
            sr.clear();
            sc.clear();
            fr.clear();
            let cover = lat.cover(c, src_off, (ncol, nrow));
            let dst_area = match cover {
                Cover::Failed => f64::NAN,
                _ => measure.ring_area(&lat.rx, &lat.ry).abs(),
            };
            if let Cover::Window { c0, r0, w } = cover {
                for (k, &v) in lat.win.coverage().iter().enumerate() {
                    if let Some(v) = opts.weight(v) {
                        sr.push((src.row_off as i64 + r0 + (k / w) as i64) as u32);
                        sc.push((src.col_off as i64 + c0 + (k % w) as i64) as u32);
                        fr.push(v);
                    }
                }
            }
            f(CellWeights {
                dst_row, dst_col: (opts.dst_col_off + c) as u32, dst_area,
                src_row: &sr, src_col: &sc, frac: &fr,
            });
        }
    }
}

/// Exact-footprint weights of a `dst_ncol` by `dst_nrow` destination window
/// (at the offset in `opts`) against the source window `src`; see the
/// module docs. [`exact_weights_with`] streams them instead.
///
/// `transformer` maps destination pixels to full-grid source pixels, as
/// for [`warp`](crate::warp::warp). Only source cells inside `src` get
/// entries.
pub fn exact_weights(
    transformer: &impl Transformer,
    src: &GridWindow,
    dst_ncol: usize,
    dst_nrow: usize,
    opts: &WeightOptions,
) -> Weights {
    let mut w = Weights {
        dst_off: [opts.dst_col_off, opts.dst_row_off],
        dst_size: [dst_ncol, dst_nrow],
        dst_area: Vec::with_capacity(dst_ncol * dst_nrow),
        measure: opts.measure,
        ..Weights::default()
    };
    exact_weights_with(transformer, src, dst_ncol, dst_nrow, opts, |cell| {
        let n = cell.frac.len();
        w.dst_row.extend(std::iter::repeat_n(cell.dst_row, n));
        w.dst_col.extend(std::iter::repeat_n(cell.dst_col, n));
        w.src_row.extend_from_slice(cell.src_row);
        w.src_col.extend_from_slice(cell.src_col);
        w.frac.extend_from_slice(cell.frac);
        w.dst_area.push(cell.dst_area);
    });
    w
}

/// A destination reducer for [`Normalization::DestArea`]: the sum of
/// weighted valid values over the cell's area.
struct AreaSum {
    sum: Vec<f64>,
    any: Vec<bool>,
    area: f64,
}

impl<T: Sample> AreaReducer<T> for AreaSum {
    fn reset(&mut self) {
        self.sum.fill(0.0);
        self.any.fill(false);
    }

    #[inline]
    fn add(&mut self, pix: &[T], pixel_ok: bool, band_ok: &[bool], w: f64) {
        if !pixel_ok { return; }
        for (((v, &ok), s), any) in pix.iter().zip(band_ok).zip(&mut self.sum).zip(&mut self.any) {
            if ok {
                *s += w * v.to_f64();
                *any = true;
            }
        }
    }

    fn write(&self, out: &mut [Res<T>]) {
        for ((o, &s), &any) in out.iter_mut().zip(&self.sum).zip(&self.any) {
            *o = if any { Res::Computed(s / self.area) } else { Res::None };
        }
    }
}

impl Weights {
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.frac.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frac.is_empty()
    }

    /// Index of an entry's destination cell in the window, row-major.
    #[inline]
    fn cell(&self, k: usize) -> usize {
        let (r, c) = (self.dst_row[k] as usize, self.dst_col[k] as usize);
        assert!(
            r >= self.dst_off[1] && c >= self.dst_off[0]
                && r - self.dst_off[1] < self.dst_size[1] && c - self.dst_off[0] < self.dst_size[0],
            "entry {k}: destination ({r}, {c}) outside the window"
        );
        (r - self.dst_off[1]) * self.dst_size[0] + (c - self.dst_off[0])
    }

    /// Entry order grouped by destination cell, keeping the order within
    /// each cell: the identity for weights from [`exact_weights`].
    fn grouped(&self) -> Vec<usize> {
        let n = self.len();
        let mut seen = vec![false; self.dst_size[0] * self.dst_size[1]];
        let mut prev = usize::MAX;
        let mut ok = true;
        for k in 0..n {
            let c = self.cell(k);
            if c != prev {
                if seen[c] { ok = false; break; }
                seen[c] = true;
                prev = c;
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        if !ok { order.sort_by_key(|&k| self.cell(k)); }
        order
    }

    /// The measure of entry `k`'s source cell: 1 for the planar measure.
    #[inline]
    pub fn src_area(&self, k: usize) -> f64 {
        self.measure.cell_area(self.src_row[k] as i64)
    }

    /// The measure of entry `k`'s overlap, `frac * src_area`.
    #[inline]
    fn overlap(&self, k: usize) -> f64 {
        self.frac[k] * self.src_area(k)
    }

    /// Sum of the overlaps' measure per destination cell of the window,
    /// row-major: the covered part of `dst_area` (ESMF's `frac_b *
    /// area_b`). The sum of `frac` for the planar measure.
    pub fn dst_covered(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.dst_size[0] * self.dst_size[1]];
        for k in 0..self.len() { s[self.cell(k)] += self.overlap(k); }
        s
    }

    /// ESMF's `S` for each entry under a normalisation: the overlap's
    /// measure over `dst_area` ([`Normalization::DestArea`]) or over the
    /// covered measure of its destination cell ([`Normalization::FracArea`]).
    /// Nodata is not considered; [`Weights::apply`] renormalises over valid
    /// source cells.
    pub fn normalised(&self, norm: Normalization) -> Vec<f64> {
        let denom = match norm {
            Normalization::FracArea => self.dst_covered(),
            Normalization::DestArea => self.dst_area.clone(),
        };
        (0..self.len()).map(|k| self.overlap(k) / denom[self.cell(k)]).collect()
    }

    /// ESMF/SCRIP's 1-based flattened indices, `(row, col)`: `row` for the
    /// destination (`dst_row * dst_ncol_full + dst_col + 1`), `col` for the
    /// source, given the full grids' widths.
    pub fn esmf_row_col(&self, dst_ncol_full: usize, src_ncol_full: usize) -> (Vec<u64>, Vec<u64>) {
        let flat = |r: &[u32], c: &[u32], n: usize| -> Vec<u64> {
            r.iter().zip(c).map(|(&r, &c)| r as u64 * n as u64 + c as u64 + 1).collect()
        };
        (flat(&self.dst_row, &self.dst_col, dst_ncol_full), flat(&self.src_row, &self.src_col, src_ncol_full))
    }

    /// Apply to a pixel-interleaved source of one or more bands, returning
    /// the destination window (`dst_size`) interleaved, as
    /// [`warp_to`](crate::warp::warp_to) does.
    ///
    /// Source cells outside `src`'s window, nodata (per band, under
    /// `opts.band_nodata`) and NaN are skipped. [`Normalization::FracArea`]
    /// gives the weighted mean over the valid entries;
    /// [`Normalization::DestArea`] gives the weighted sum of the valid
    /// entries over `dst_area`, not renormalised. Destination cells with
    /// no valid entry get the destination nodata (or 0). Values are stored
    /// as `warp_to` stores them: the working type of `S`, `D` and the nodata
    /// values, moved off the destination nodata, converted to `D`.
    ///
    /// Each destination cell reduces its entries with `frac >= 1` first,
    /// then the others, each in entry order, with the running mean `warp()`
    /// uses, so weights from [`WeightOptions::warp_compatible`] give
    /// `warp()`'s values bit for bit.
    pub fn apply<S: Sample, D: Sample>(&self, src: &SourceBuffer<'_, S>, norm: Normalization, opts: &ApplyOptions) -> Vec<D> {
        let nb = src.nbands;
        assert_eq!(src.data.len(), src.ncol * src.nrow * nb, "source buffer size mismatch");
        assert!(opts.mask_band.is_none_or(|a| a < nb), "mask_band out of range");
        let ncell = self.dst_size[0] * self.dst_size[1];
        assert_eq!(self.dst_area.len(), ncell, "dst_area size mismatch");
        let out_spec = Output::<D>::from_nodata(S::DATA_TYPE, nb, &opts.src_nodata, &opts.dst_nodata);
        let img = Interleaved {
            src: src.data, ncol: src.ncol, nrow: src.nrow, nb,
            mask: opts.mask_band, nodata: out_spec.src_nodata::<S>(), rule: opts.band_nodata,
            cut: None,
        };
        let mut out = out_spec.filled(ncell);
        let order = self.grouped();
        let band_ok = &mut vec![false; nb];
        let mut res: Vec<Res<S>> = vec![Res::None; nb];
        let mut mean = AreaMean { mean: vec![0.0; nb], total: vec![0.0; nb] };
        let mut sum = AreaSum { sum: vec![0.0; nb], any: vec![false; nb], area: 0.0 };
        let buf_index = |k: usize| -> Option<usize> {
            let (r, c) = (self.src_row[k] as usize, self.src_col[k] as usize);
            let (r, c) = (r.checked_sub(src.row_off)?, c.checked_sub(src.col_off)?);
            (r < src.nrow && c < src.ncol).then_some(r * src.ncol + c)
        };
        let mut start = 0;
        while start < order.len() {
            let cell = self.cell(order[start]);
            let mut end = start + 1;
            while end < order.len() && self.cell(order[end]) == cell { end += 1; }
            let group = &order[start..end];
            start = end;
            let run = |red: &mut dyn FnMut(usize, f64)| {
                for full in [true, false] {
                    for &k in group {
                        if (self.frac[k] >= 1.0) != full { continue; }
                        if let Some(p) = buf_index(k) { red(p, self.overlap(k)); }
                    }
                }
            };
            match norm {
                Normalization::FracArea => {
                    AreaReducer::<S>::reset(&mut mean);
                    run(&mut |p, w| img.reduce(&mut mean, p, w, band_ok));
                    mean.write(&mut res);
                }
                Normalization::DestArea => {
                    sum.area = self.dst_area[cell];
                    AreaReducer::<S>::reset(&mut sum);
                    run(&mut |p, w| img.reduce(&mut sum, p, w, band_ok));
                    sum.write(&mut res);
                }
            }
            for (b, &r) in res.iter().enumerate() {
                if let Some(v) = out_spec.result(r, b) { out[cell * nb + b] = v; }
            }
        }
        out
    }
}
