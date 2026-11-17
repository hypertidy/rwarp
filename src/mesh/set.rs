//! A whole curvilinear grid as rectangular pieces, each a [`QuadMesh`] in
//! its band's world coordinates: [`MeshSet`].

use std::ops::Range;

use super::cf::{cf_corners, Halo};
use super::period::range_of;
use super::{bow_tie, shoelace2, CellMap, MeshTarget, MeshTransformer, QuadMesh};
use crate::crs::CrsTransform;
use crate::transform::Transformer;
use crate::weights::{exact_weights, GridWindow, WeightOptions, Weights};

/// A band of rows of a [`MeshSet`] and the projection it's held in: from
/// the bounds' coordinates (longitude and latitude, say) to the band's
/// world coordinates.
pub struct Band<'a> {
    /// The band's rows of cells, in the set's numbering (the file's rows
    /// less the halo's first rows).
    pub rows: Range<usize>,
    /// From the bounds' coordinates to the band's world coordinates.
    pub to_world: &'a dyn CrsTransform,
}

/// One rectangle of a [`MeshSet`]'s cells, a mesh of its own in its band's
/// world: its cell `(row, col)` is the set's cell `(row + row_off, col +
/// col_off)`.
#[derive(Debug, Clone)]
pub struct Piece {
    col_off: usize,
    row_off: usize,
    world: usize,
    mesh: QuadMesh,
}

impl Piece {
    /// `[col_off, row_off]` of the piece's cells in the set.
    pub fn offset(&self) -> [usize; 2] {
        [self.col_off, self.row_off]
    }

    /// The band, and so the world, the piece is in: an index into the
    /// bands the set was built with.
    pub fn world(&self) -> usize {
        self.world
    }

    /// The piece's cells.
    pub fn mesh(&self) -> &QuadMesh {
        &self.mesh
    }
}

/// A curvilinear grid too big or too damaged for one [`QuadMesh`], such as
/// an ocean model's whole global grid, as rectangular pieces of its cells,
/// each a mesh in the world coordinates of its band of rows.
///
/// - **Bands.** No projection holds a whole tripolar ocean grid with
///   straight-edged cells: north-polar Lambert azimuthal equal-area flattens
///   its far south into bow-ties, and longitude and latitude can't hold the
///   cell over the North Pole. Split the rows into bands, each in a
///   projection that holds it: for an ORCA grid, north-polar and
///   south-polar LAEA on the model's sphere, split at the row nearest the
///   equator. Both are equal-area on the same sphere, so every cell's area
///   is in square metres of it, and weights from different bands add up.
/// - **Cells left out.** Cells that aren't valid in their band's world
///   (bow-ties, cells turned the other way, cells that overlap others) are
///   dropped where `droppable` allows: land, in an ocean grid, whose bounds
///   are often damaged. Any other invalid cell is an error. The cells left
///   are covered by rectangles, each a mesh of its own, so a dropped cell
///   is a hole that no ring or outline is walked through, and every other
///   cell keeps its geometry from the file.
/// - **Seams and folds** are crossed inside a piece that spans them (see
///   the [module docs](crate::mesh#seams-and-folds)); between pieces
///   nothing needs joining, since each walk measures its own cells.
/// - **Weights.** [`MeshSet::weights_to`] gives conservative weights from
///   the set to a grid, [`MeshSet::weights_onto`] from a source onto the
///   set, each as one [`Weights`] in the set's numbering, from
///   [`exact_weights`] on every piece.
///
/// Where two bands meet, each side's cells have edges straight in their
/// own projection, so the two bow apart between corners: a thin gap
/// covered by neither. At the equator it is up to `R dlam^2 / 4` high on
/// each side, `dlam` the cells' width in radians, and takes about `dlam^2
/// / (6 dphi)` of each cell next to it, `dphi` their height: 2.3% for
/// ORCA2's 2 by 0.5 degree cells there. Split where the cells are taller,
/// south of an ORCA grid's equatorial refinement (at 20 S, under 1%).
///
/// ```
/// use rwarp::mesh::{Band, Halo, MeshSet};
/// use rwarp::CrsTransform;
///
/// // a 12 x 6 global grid of 30 degree cells, bounds in lon/lat, with one
/// // corner (row line 3, column line 6) moved 40 degrees west: the cells
/// // round it fold over, and some overlap their neighbours
/// let (nc, nr) = (12, 6);
/// let corner = |j: usize, i: usize| {
///     let (x, y) = (-180.0 + 30.0 * i as f64, -90.0 + 30.0 * j as f64);
///     if (j, i) == (3, 6) { (x - 40.0, y) } else { (x, y) }
/// };
/// let (mut xb, mut yb) = (Vec::new(), Vec::new());
/// for j in 0..nr {
///     for i in 0..nc {
///         for (x, y) in [corner(j, i), corner(j, i + 1), corner(j + 1, i + 1), corner(j + 1, i)] {
///             xb.push(x);
///             yb.push(y);
///         }
///     }
/// }
/// // they're land, which may be left out
/// let land: Vec<bool> = (0..nc * nr).map(|k| (2..4).contains(&(k / nc)) && (3..8).contains(&(k % nc))).collect();
///
/// // one band, in the bounds' own coordinates
/// struct Same;
/// impl CrsTransform for Same {
///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> { Some((x, y)) }
/// }
/// let bands = [Band { rows: 0..nr, to_world: &Same }];
/// let set = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, Some(&land)).unwrap();
/// assert!(set.dropped().contains(&(2, 5)));
/// assert!(set.dropped().iter().all(|&(r, c)| land[r * nc + c]));
/// // rectangles round the hole
/// assert!(set.pieces().len() > 2);
/// assert!(set.locate_cell(2, 5).is_none() && set.locate_cell(0, 5).is_some());
///
/// // the same, with nothing that may be left out
/// let e = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap_err();
/// assert!(e.contains("may not be left out"), "{e}");
/// ```
#[derive(Debug, Clone)]
pub struct MeshSet {
    ncol: usize,
    nrow: usize,
    pieces: Vec<Piece>,
    dropped: Vec<(usize, usize)>,
}

/// Rectangles `(c0, c1, r0, r1)` (half-open) covering the `true` cells of
/// an `ncol` by `nrow` mask, row-major: each row's runs, a run continuing
/// the rectangle above it when it spans the same columns.
fn rectangles(kept: &[bool], ncol: usize, nrow: usize) -> Vec<[usize; 4]> {
    let mut done = Vec::new();
    // open rectangles, by their columns
    let mut open: Vec<[usize; 4]> = Vec::new();
    for r in 0..nrow {
        let row = &kept[r * ncol..(r + 1) * ncol];
        let mut runs = Vec::new();
        let mut c = 0;
        while c < ncol {
            if row[c] {
                let c0 = c;
                while c < ncol && row[c] {
                    c += 1;
                }
                runs.push((c0, c));
            } else {
                c += 1;
            }
        }
        let mut next = Vec::with_capacity(runs.len());
        for (c0, c1) in runs {
            match open.iter().position(|o| o[0] == c0 && o[1] == c1) {
                Some(k) => {
                    let mut o = open.swap_remove(k);
                    o[3] = r + 1;
                    next.push(o);
                }
                None => next.push([c0, c1, r, r + 1]),
            }
        }
        done.append(&mut open);
        open = next;
    }
    done.append(&mut open);
    done.sort_by_key(|o| (o[2], o[0]));
    done
}

impl MeshSet {
    /// A set from each cell's own four vertices (CF bounds, SCRIP corners),
    /// as [`QuadMesh::from_cf_bounds`] takes them, with the rows split into
    /// `bands`, each projected to its own world.
    ///
    /// - The `halo` is dropped first, and the set's cell `(row, col)` is
    ///   the file's cell `(row + halo.rows[0], col + halo.cols[0])`. The
    ///   bands' rows are in that numbering, and must tile it in order.
    /// - Each band's bounds go to its world (`to_world`), and its corners
    ///   are found there as [`QuadMesh::from_cf_bounds`] finds them: vertex
    ///   order, shared corners, and a T pivot's halved last row in the
    ///   last band.
    /// - `droppable`, one value per cell of the file (`ncol * nrow`,
    ///   row-major; for an ocean grid, its land), says which cells may be
    ///   left out. A cell is left out if it's a bow-tie in its band's world,
    ///   turned the other way from most of its band, or overlaps another
    ///   cell of its band (both are left out if both may be). An invalid
    ///   cell that may not be left out is an error, named by its row and
    ///   column in the file.
    ///
    /// The pieces' cell map is [`CellMap::EqualArea`].
    pub fn from_cf_bounds(
        ncol: usize,
        nrow: usize,
        xb: &[f64],
        yb: &[f64],
        halo: Halo,
        bands: &[Band],
        droppable: Option<&[bool]>,
    ) -> Result<MeshSet, String> {
        let n = ncol * nrow;
        if xb.len() != 4 * n || yb.len() != 4 * n {
            return Err(format!(
                "bounds arrays have {} and {} values, expected nrow * ncol * 4 = {}",
                xb.len(),
                yb.len(),
                4 * n
            ));
        }
        if droppable.is_some_and(|d| d.len() != n) {
            return Err(format!(
                "droppable has {} values, expected nrow * ncol = {n}",
                droppable.unwrap().len()
            ));
        }
        let [c0, c1] = halo.cols;
        let [r0, r1] = halo.rows;
        if c0 + c1 >= ncol || r0 + r1 >= nrow {
            return Err(format!(
                "the halo (columns {c0} + {c1}, rows {r0} + {r1}) leaves no cells of {ncol} x {nrow}"
            ));
        }
        let (nc, nr) = (ncol - c0 - c1, nrow - r0 - r1);
        let mut at = 0;
        for (k, b) in bands.iter().enumerate() {
            if b.rows.start != at || b.rows.end <= b.rows.start {
                return Err(format!(
                    "band {k} has rows {:?}: the bands must tile rows 0..{nr} in order",
                    b.rows
                ));
            }
            at = b.rows.end;
        }
        if at != nr {
            return Err(format!(
                "the bands end at row {at}, not at the last row {nr}"
            ));
        }
        // whether the set's cell (row, col) may be left out
        let may_drop = |r: usize, c: usize| droppable.is_some_and(|d| d[(r + r0) * ncol + c + c0]);
        let file = |r: usize, c: usize| (r + r0, c + c0);

        let mut pieces = Vec::new();
        let mut dropped = Vec::new();
        for (k, b) in bands.iter().enumerate() {
            let (rs, re) = (b.rows.start, b.rows.end);
            // the band's bounds, every column of the file, in its world
            let lo = (rs + r0) * ncol * 4;
            let hi = (re + r0) * ncol * 4;
            let (mut bx, mut by) = (Vec::with_capacity(hi - lo), Vec::with_capacity(hi - lo));
            for v in lo..hi {
                match b.to_world.convert(xb[v], yb[v]) {
                    Some((x, y)) if x.is_finite() && y.is_finite() => {
                        bx.push(x);
                        by.push(y);
                    }
                    _ => {
                        let cell = v / 4;
                        return Err(format!(
                            "band {k}: a vertex of cell ({}, {}) of the file, ({}, {}), doesn't go to the band's world",
                            cell / ncol,
                            cell % ncol,
                            xb[v],
                            yb[v]
                        ));
                    }
                }
            }
            let band_halo = Halo {
                cols: halo.cols,
                rows: [0, 0],
                fold_through_centres: halo.fold_through_centres && re == nr,
            };
            let cs = cf_corners(ncol, re - rs, &bx, &by, band_halo).map_err(|e| {
                format!("band {k} (rows {rs}..{re}, file rows counted from the band's first): {e}")
            })?;
            let (h, w) = (re - rs, nc);
            let corner = |j: usize, i: usize| [cs.x[j * (w + 1) + i], cs.y[j * (w + 1) + i]];
            let quad = |r: usize, c: usize| {
                [
                    corner(r, c),
                    corner(r, c + 1),
                    corner(r + 1, c + 1),
                    corner(r + 1, c),
                ]
            };
            // the band's orientation: that of most of its cells with area
            let a2: Vec<f64> = (0..h * w).map(|q| shoelace2(&quad(q / w, q % w))).collect();
            let pos = a2.iter().filter(|&&a| a > 0.0).count();
            let neg = a2.iter().filter(|&&a| a < 0.0).count();
            let sign = if neg > pos { -1.0 } else { 1.0 };
            let mut kept = vec![true; h * w];
            for q in 0..h * w {
                let (r, c) = (q / w, q % w);
                let bow = bow_tie(&quad(r, c));
                if bow || a2[q] * sign < 0.0 {
                    if !may_drop(rs + r, c) {
                        let (fr, fc) = file(rs + r, c);
                        return Err(format!(
                            "cell ({fr}, {fc}) of the file {} in band {k}'s world, and may not be left out",
                            if bow { "is a bow-tie" } else { "is turned the other way from its band" }
                        ));
                    }
                    kept[q] = false;
                }
            }
            // pieces, then cells that overlap: left out, and the pieces made again
            let build = |kept: &[bool]| -> Result<Vec<Piece>, String> {
                rectangles(kept, w, h)
                    .into_iter()
                    .map(|[ca, cb, ra, rb]| {
                        let (pw, ph) = (cb - ca, rb - ra);
                        let (mut x, mut y) = (
                            Vec::with_capacity((pw + 1) * (ph + 1)),
                            Vec::with_capacity((pw + 1) * (ph + 1)),
                        );
                        for j in ra..=rb {
                            for i in ca..=cb {
                                let p = corner(j, i);
                                x.push(p[0]);
                                y.push(p[1]);
                            }
                        }
                        let mesh = QuadMesh::from_corners(pw, ph, x, y, CellMap::EqualArea)
                            .map_err(|e| {
                                format!(
                                    "band {k}, the piece at rows {}..{}, columns {ca}..{cb}: {e}",
                                    rs + ra,
                                    rs + rb
                                )
                            })?;
                        Ok(Piece {
                            col_off: ca,
                            row_off: rs + ra,
                            world: k,
                            mesh,
                        })
                    })
                    .collect()
            };
            let mut band_pieces = build(&kept)?;
            let mut clean = false;
            for _ in 0..8 {
                let pairs = overlaps(&band_pieces);
                if pairs.is_empty() {
                    clean = true;
                    break;
                }
                let mut changed = false;
                for (a, b) in pairs {
                    let (da, db) = (may_drop(a.0, a.1), may_drop(b.0, b.1));
                    if !da && !db {
                        let (fa, fb) = (file(a.0, a.1), file(b.0, b.1));
                        return Err(format!(
                            "cells ({}, {}) and ({}, {}) of the file overlap in band {k}'s world, and neither may be left out",
                            fa.0, fa.1, fb.0, fb.1
                        ));
                    }
                    for (cell, d) in [(a, da), (b, db)] {
                        let q = (cell.0 - rs) * w + cell.1;
                        if d && kept[q] {
                            kept[q] = false;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
                band_pieces = build(&kept)?;
            }
            if !clean {
                return Err(format!(
                    "band {k}: cells still overlap after leaving out eight rounds of them"
                ));
            }
            dropped.extend(
                (0..h * w)
                    .filter(|&q| !kept[q])
                    .map(|q| (rs + q / w, q % w)),
            );
            pieces.extend(band_pieces);
        }
        Ok(MeshSet {
            ncol: nc,
            nrow: nr,
            pieces,
            dropped,
        })
    }

    /// The set's size in cells, `[ncol, nrow]`.
    pub fn size(&self) -> [usize; 2] {
        [self.ncol, self.nrow]
    }

    /// The pieces, by first row and then first column.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// The cells left out, `(row, col)` in the set's numbering, row-major.
    pub fn dropped(&self) -> &[(usize, usize)] {
        &self.dropped
    }

    /// The piece holding cell `(row, col)`, and the cell's position in it;
    /// `None` for a cell left out.
    pub fn locate_cell(&self, row: usize, col: usize) -> Option<(&Piece, usize, usize)> {
        self.pieces.iter().find_map(|p| {
            let [c0, r0] = p.offset();
            (row >= r0 && col >= c0 && row < r0 + p.mesh.nrow() && col < c0 + p.mesh.ncol())
                .then(|| (p, row - r0, col - c0))
        })
    }

    /// Conservative weights from the set's cells to a grid of `dst_ncol`
    /// by `dst_nrow` pixels: [`exact_weights`] through the
    /// [`MeshTransformer`] that `transformer(world, mesh)` makes for each
    /// piece (with the grid's geotransform and the CRS steps between the
    /// piece's world and the grid), merged into one [`Weights`] with
    /// source rows and columns in the set's numbering.
    ///
    /// `frac` is each cell's share of its own world area, so cells from
    /// different bands mix only through `src_area`: give every band an
    /// equal-area projection on one sphere, or the areas are in different
    /// units. A target pixel's `dst_area` is its outline's area in the
    /// world of the piece that covers most of it, and NaN where no piece
    /// reaches.
    ///
    /// Each piece is measured over the target pixels it reaches: the box
    /// of its boundary in target pixels, two pixels wider each way (the
    /// window's whole width for a boundary that spans half of it or more,
    /// as across the target's seam), grown on each side its cover touches
    /// until it touches none (a piece's cells are one connected region, so
    /// that is all of them; a piece round a pole grows to the pole's
    /// row).
    pub fn weights_to(
        &self,
        transformer: impl Fn(usize, QuadMesh) -> Result<MeshTransformer, String>,
        dst_ncol: usize,
        dst_nrow: usize,
        opts: &WeightOptions,
    ) -> Result<Weights, String> {
        let full = [opts.dst_col_off, opts.dst_row_off, dst_ncol, dst_nrow];
        let mut out = Weights {
            dst_off: [full[0], full[1]],
            dst_size: [dst_ncol, dst_nrow],
            dst_area: vec![f64::NAN; dst_ncol * dst_nrow],
            measure: opts.measure,
            ..Weights::default()
        };
        let mut best = vec![f64::NEG_INFINITY; dst_ncol * dst_nrow];
        for p in &self.pieces {
            let t = transformer(p.world, p.mesh.clone())?;
            let mut win = reach(&t, &p.mesh, full, 2);
            let w = loop {
                let w = exact_weights(
                    &t,
                    &t.source_window(),
                    win[2],
                    win[3],
                    &opts.with_dst_offset(win[0], win[1]),
                );
                let sides = touched(&w, win, full);
                if !sides.contains(&true) {
                    break w;
                }
                win = grow(win, full, sides);
            };
            let covered = w.dst_covered();
            for (k, &a) in w.dst_area.iter().enumerate() {
                let (r, c) = (win[1] + k / win[2], win[0] + k % win[2]);
                let at = (r - full[1]) * dst_ncol + (c - full[0]);
                if a.is_finite() && covered[k] > best[at] {
                    out.dst_area[at] = a;
                    best[at] = covered[k];
                }
            }
            out.dst_row.extend_from_slice(&w.dst_row);
            out.dst_col.extend_from_slice(&w.dst_col);
            out.src_row
                .extend(w.src_row.iter().map(|&r| r + p.row_off as u32));
            out.src_col
                .extend(w.src_col.iter().map(|&c| c + p.col_off as u32));
            out.frac.extend_from_slice(&w.frac);
            out.src_cell_area.extend_from_slice(&w.src_cell_area);
        }
        Ok(out)
    }

    /// Conservative weights from a source onto the set's cells:
    /// [`exact_weights`] from the source window `src` through a
    /// [`MeshTarget`] on each piece, with the inner transformer that
    /// `inner(world)` makes (from the piece's world to the source, its
    /// destination geotransform [`WORLD_GT`](super::WORLD_GT)), merged
    /// into one [`Weights`] whose destination is the whole set. Cells left
    /// out have no entries and a `dst_area` of NaN. The destination offset
    /// in `opts` is ignored.
    pub fn weights_onto<T: Transformer>(
        &self,
        inner: impl Fn(usize) -> Result<T, String>,
        src: &GridWindow,
        opts: &WeightOptions,
    ) -> Result<Weights, String> {
        let opts = opts.with_dst_offset(0, 0);
        let mut out = Weights {
            dst_off: [0, 0],
            dst_size: [self.ncol, self.nrow],
            dst_area: vec![f64::NAN; self.ncol * self.nrow],
            measure: opts.measure,
            ..Weights::default()
        };
        for p in &self.pieces {
            let t = MeshTarget::new(p.mesh.clone(), inner(p.world)?);
            let (pw, ph) = (p.mesh.ncol(), p.mesh.nrow());
            let w = exact_weights(&t, src, pw, ph, &opts);
            for r in 0..ph {
                for c in 0..pw {
                    out.dst_area[(r + p.row_off) * self.ncol + c + p.col_off] =
                        w.dst_area[r * pw + c];
                }
            }
            out.dst_row
                .extend(w.dst_row.iter().map(|&r| r + p.row_off as u32));
            out.dst_col
                .extend(w.dst_col.iter().map(|&c| c + p.col_off as u32));
            out.src_row.extend_from_slice(&w.src_row);
            out.src_col.extend_from_slice(&w.src_col);
            out.frac.extend_from_slice(&w.frac);
            out.src_cell_area.extend_from_slice(&w.src_cell_area);
        }
        Ok(out)
    }
}

/// The target window `[col_off, row_off, ncol, nrow]` (full-grid pixels)
/// that a piece `m` reaches under `t`, within `full`, to start from: the
/// box of its boundary's corners and side midpoints in target pixels,
/// `margin` pixels wider each way, and `full`'s whole width if the points
/// span half of it or more. `full` itself when a point fails.
fn reach(t: &MeshTransformer, m: &QuadMesh, full: [usize; 4], margin: usize) -> [usize; 4] {
    let (nc, nr) = (m.ncol() as f64, m.nrow() as f64);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    let steps = |n: f64| {
        (0..=(2.0 * n) as usize)
            .map(|k| k as f64 / 2.0)
            .collect::<Vec<_>>()
    };
    for c in steps(nc) {
        x.extend([c, c]);
        y.extend([0.0, nr]);
    }
    for r in steps(nr) {
        x.extend([0.0, nc]);
        y.extend([r, r]);
    }
    let ok = t.transform(false, &mut x, &mut y);
    if ok.iter().any(|&o| !o) {
        return full;
    }
    let ([x0, x1], [y0, y1]) = (range_of(&x), range_of(&y));
    if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
        return full;
    }
    let [fc, fr, fw, fh] = full.map(|v| v as f64);
    let m = margin as f64;
    let (c0, c1) = if x1 - x0 >= fw / 2.0 {
        (fc, fc + fw)
    } else {
        (
            (x0.floor() - m).clamp(fc, fc + fw),
            (x1.ceil() + m).clamp(fc, fc + fw),
        )
    };
    let r0 = (y0.floor() - m).clamp(fr, fr + fh);
    let r1 = (y1.ceil() + m).clamp(fr, fr + fh);
    if c1 <= c0 || r1 <= r0 {
        // the piece is beside the window: one pixel to measure nothing on
        return [
            c0.min(fc + fw - 1.0) as usize,
            r0.min(fr + fh - 1.0) as usize,
            1,
            1,
        ];
    }
    [
        c0 as usize,
        r0 as usize,
        (c1 - c0) as usize,
        (r1 - r0) as usize,
    ]
}

/// The sides of window `win` (left, right, top, bottom) on whose edge
/// pixels the weights `w` reach, where that edge isn't `full`'s.
fn touched(w: &Weights, win: [usize; 4], full: [usize; 4]) -> [bool; 4] {
    let [c0, r0, cw, rh] = win;
    let (c1, r1) = (c0 + cw, r0 + rh);
    let (fc1, fr1) = (full[0] + full[2], full[1] + full[3]);
    let mut t = [false; 4];
    for k in 0..w.len() {
        let (r, c) = (w.dst_row[k] as usize, w.dst_col[k] as usize);
        t[0] |= c == c0 && c0 > full[0];
        t[1] |= c + 1 == c1 && c1 < fc1;
        t[2] |= r == r0 && r0 > full[1];
        t[3] |= r + 1 == r1 && r1 < fr1;
    }
    t
}

/// `win` grown on its touched `sides` (left, right, top, bottom) by its
/// own size that way, within `full`.
fn grow(win: [usize; 4], full: [usize; 4], sides: [bool; 4]) -> [usize; 4] {
    let [c0, r0, cw, rh] = win;
    let (dc, dr) = (cw.max(2), rh.max(2));
    let nc0 = if sides[0] {
        c0.saturating_sub(dc).max(full[0])
    } else {
        c0
    };
    let nc1 = if sides[1] {
        (c0 + cw + dc).min(full[0] + full[2])
    } else {
        c0 + cw
    };
    let nr0 = if sides[2] {
        r0.saturating_sub(dr).max(full[1])
    } else {
        r0
    };
    let nr1 = if sides[3] {
        (r0 + rh + dr).min(full[1] + full[3])
    } else {
        r0 + rh
    };
    [nc0, nr0, nc1 - nc0, nr1 - nr0]
}

/// Pairs of cells of `pieces` (all in one world) that overlap, as `(row,
/// col)` in the set's numbering: a point inside one (its overlap probe,
/// as [`QuadMesh::find_overlap`] takes) that another owns.
fn overlaps(pieces: &[Piece]) -> Vec<((usize, usize), (usize, usize))> {
    let boxes: Vec<[f64; 4]> = pieces
        .iter()
        .map(|p| {
            let ([x0, x1], [y0, y1]) = (range_of(&p.mesh.x), range_of(&p.mesh.y));
            [x0, y0, x1, y1]
        })
        .collect();
    let mut out = Vec::new();
    for (ia, a) in pieces.iter().enumerate() {
        for k in 0..a.mesh.ncol() * a.mesh.nrow() {
            let Some(p) = a.mesh.probe(k) else { continue };
            let here = (a.row_off + k / a.mesh.ncol(), a.col_off + k % a.mesh.ncol());
            for (ib, b) in pieces.iter().enumerate() {
                let bb = boxes[ib];
                if p[0] < bb[0] || p[0] > bb[2] || p[1] < bb[1] || p[1] > bb[3] {
                    continue;
                }
                for (r, c) in b.mesh.locate_all(p) {
                    let there = (b.row_off + r, b.col_off + c);
                    if ib != ia || there != here {
                        out.push((here, there));
                    }
                }
            }
        }
    }
    out
}
