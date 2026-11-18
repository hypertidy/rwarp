//! A whole curvilinear grid as rectangular pieces, each a [`QuadMesh`] in
//! its band's world coordinates: [`MeshSet`].

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use bigcurve::{from_fn, ForwardBatch, Options, Planar, Resampler, Vertex};
use controlledburn::{
    BurnMode, BurnOptions, BurnResult, Coord, Geometry, GridEdge, GridPoint, GridRun, Note, Polygon,
};

use super::cf::{cf_corners, Halo};
use super::period::range_of;
use super::walk::{burn_plane, map_geoms, vertex_chunks, vertices};
use super::{bow_tie, orient, shoelace2, CellMap, MeshTarget, MeshTransformer, QuadMesh};
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
/// - **Burning.** [`MeshSet::burn`] burns polygons and points given in the
///   bounds' coordinates onto the set's cells, densified into each band's
///   world.
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
    clips: Vec<Clip>,
}

/// What [`MeshSet::burn`] cuts a band's polygons to, in the bounds'
/// coordinates: the y range of the band's kept cells' vertices, widened
/// by `step` each way, `step` being its tallest kept cell's height in y
/// (also the longest edge handed to bigcurve). NaN for a band with no
/// cells kept.
#[derive(Debug, Clone)]
struct Clip {
    rows: Range<usize>,
    y: [f64; 2],
    step: f64,
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
        let mut clips = Vec::with_capacity(bands.len());
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
            // the y range of the kept cells' vertices, as the file has them
            let (mut y0, mut y1, mut tall) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
            for q in (0..h * w).filter(|&q| kept[q]) {
                let (fr, fc) = file(rs + q / w, q % w);
                let v = 4 * (fr * ncol + fc);
                let [lo, hi] = range_of(&yb[v..v + 4]);
                (y0, y1, tall) = (y0.min(lo), y1.max(hi), tall.max(hi - lo));
            }
            if tall <= 0.0 {
                tall = y1 - y0;
            }
            clips.push(if tall > 0.0 && tall.is_finite() {
                Clip {
                    rows: rs..re,
                    y: [y0 - tall, y1 + tall],
                    step: tall,
                }
            } else {
                Clip {
                    rows: rs..re,
                    y: [f64::NAN; 2],
                    step: f64::NAN,
                }
            });
        }
        Ok(MeshSet {
            ncol: nc,
            nrow: nr,
            pieces,
            dropped,
            clips,
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

    /// Burn polygons and points onto the set's cells: controlledburn's
    /// tables, with `(row, col)` in the set's numbering and each edge's
    /// fraction of its cell's own area, in the world of the cell's band.
    /// Coverage mode only; lines are refused, as [`QuadMesh::burn`]
    /// refuses them.
    ///
    /// - **Geometries are in the bounds' coordinates** (longitude and
    ///   latitude, for an ocean grid), and `bands` are the bands the set was
    ///   built with: their rows must match, and each `to_world` takes the
    ///   geometries to its band's world.
    /// - **Edges are straight in the bounds' coordinates.** Before a
    ///   polygon goes to a band, each edge is cut into equal pieces no
    ///   longer than the band's tallest cell is high, and each piece is
    ///   densified by bigcurve (d3-geo's adaptive resampling, planar edges)
    ///   until no chord strays more than `tolerance` from the edge's image,
    ///   in the band's world units (metres, for LAEA). A fraction is then
    ///   good to about `tolerance` times the length of polygon edge in the
    ///   cell, over its area. The work grows as `1 / sqrt(tolerance)`.
    /// - **Each band takes the polygons' part near its own cells**: they are
    ///   cut to the y range of its cells' vertices, widened by its tallest
    ///   cell each way. A projection holds one hemisphere well and the far
    ///   pole not at all (a polar LAEA sends it to its edge), so without
    ///   the cut a polygon round the far pole, Antarctica in a north-polar
    ///   band, or a whole-world box, would come out inside out. The cut
    ///   assumes no cell of the band reaches further than that beyond its
    ///   vertices' range. That holds for an ocean grid in longitude and
    ///   latitude, whose cells bulge far less than its tallest is high. A
    ///   cell at the band's edge whose corners come from its neighbours'
    ///   bounds (garbled land, as on ORCA2) could reach further.
    /// - **Cuts at 180 degrees and along a pole** come out of a polar
    ///   projection with both sides on one line. Each ring is split where
    ///   it meets itself, within `1e-12` of its size, and the line taken
    ///   out before it is walked: a whole-world box becomes the circle of
    ///   its cut parallel, an Antarctica cut at one its coast and a hole.
    /// - **Each piece** burns, as a [`QuadMesh`], only what can reach it.
    ///   A grid of tiles over its box in its world marks where its cells
    ///   are, widened by a tile. A polygon whose box meets no marked tile is
    ///   skipped. In the rest, each run of edges that can't reach the piece
    ///   is cut short, and a ring that can't reach it at all is dropped or
    ///   stood in for. Edges are tested 32 at a time by their box first,
    ///   one by one only near the piece. The walk sees the same crossings
    ///   and the same winding round its interior point, so the result is
    ///   the same, bit for bit. A one-row piece round most of a latitude
    ///   circle has a box the size of its band's disk, yet walks only the
    ///   polygons' edges near it.
    /// - **Merged.** The pieces' records are merged in controlledburn's
    ///   order (by geometry, then row and column). Cells left out are never
    ///   in them. Where bands meet, the gap between their cells (see the
    ///   [type docs](MeshSet)) is burned onto no cell.
    /// - **Points** are found in the first piece (bands in order) whose
    ///   cell holds them, if any.
    ///
    /// [`MeshSet::burn_par`] gives the same result on several threads,
    /// and [`MeshSet::burn_par_with`] projects there too.
    ///
    /// ```
    /// use rwarp::mesh::{Band, Halo, MeshSet};
    /// use rwarp::CrsTransform;
    /// use controlledburn::{BurnMode, Coord, Geometry, Polygon};
    ///
    /// // a global grid of 30 degree cells, bounds in lon/lat, in two
    /// // bands of polar LAEA on the unit sphere split at the equator
    /// let (nc, nr) = (12, 6);
    /// let (mut xb, mut yb) = (Vec::new(), Vec::new());
    /// for j in 0..nr {
    ///     for i in 0..nc {
    ///         for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
    ///             xb.push(-180.0 + 30.0 * (i + di) as f64);
    ///             yb.push(-90.0 + 30.0 * (j + dj) as f64);
    ///         }
    ///     }
    /// }
    /// struct Laea(f64);
    /// impl CrsTransform for Laea {
    ///     fn convert(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
    ///         // self.0: 1 for north-polar, -1 for south-polar
    ///         let rho = 2.0 * ((90.0 - self.0 * lat).to_radians() / 2.0).sin();
    ///         let (s, c) = lon.to_radians().sin_cos();
    ///         Some((rho * s, -self.0 * rho * c))
    ///     }
    /// }
    /// let (south, north) = (Laea(-1.0), Laea(1.0));
    /// let bands = [Band { rows: 0..3, to_world: &south }, Band { rows: 3..6, to_world: &north }];
    /// let set = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap();
    ///
    /// // the whole world: every cell covered in full, by both bands
    /// let ring = [(-180.0, -90.0), (180.0, -90.0), (180.0, 90.0), (-180.0, 90.0), (-180.0, -90.0)];
    /// let world = Geometry::Polygon(Polygon::new(vec![ring.iter().map(|&(x, y)| Coord { x, y }).collect()]));
    /// let r = set.burn(&[world], &bands, 1e-9, BurnMode::Coverage).unwrap();
    /// assert!(r.edges.iter().all(|e| e.fraction > 0.999999));
    /// assert!((r.covered_cells() - (nc * nr) as f64).abs() < 1e-3);
    /// ```
    pub fn burn(
        &self,
        geoms: &[Geometry],
        bands: &[Band],
        tolerance: f64,
        opts: impl Into<BurnOptions>,
    ) -> Result<BurnResult, String> {
        self.burn_with(
            geoms,
            bands,
            tolerance,
            opts.into(),
            true,
            1,
            None::<&Serial>,
        )
    }

    /// [`MeshSet::burn`] on up to `threads` threads (0 for all available),
    /// with the same result record for record.
    ///
    /// - **On the calling thread:** each band's polygons go to its world
    ///   there, one band after the other, since a band's [`CrsTransform`]
    ///   needn't be thread-safe (libproj's aren't). So are the points.
    /// - **On the threads:** each of the band's pieces filters and
    ///   shortens its polygons, the costliest pieces first. The pieces'
    ///   polygons are then walked into their index spaces in contiguous
    ///   chunks of about equal vertex counts, taken in turn. Each piece is
    ///   burned with [`controlledburn::burn_par`].
    ///
    /// With `threads = 1` nothing is spawned, as on wasm, and this is
    /// [`MeshSet::burn`]. On ORCA2, Natural Earth's 10 m countries take
    /// about 0.3 s of projecting before any thread starts (1.6 s at a
    /// tolerance of 0.01 m); [`MeshSet::burn_par_with`] projects on the
    /// threads.
    pub fn burn_par(
        &self,
        geoms: &[Geometry],
        bands: &[Band],
        tolerance: f64,
        opts: impl Into<BurnOptions>,
        threads: usize,
    ) -> Result<BurnResult, String> {
        self.burn_with(
            geoms,
            bands,
            tolerance,
            opts.into(),
            true,
            threads,
            None::<&Serial>,
        )
    }

    /// [`MeshSet::burn_par`], with the polygons projected on the threads
    /// too, each thread through band `k`'s transform built by
    /// `to_world(k)`: the same records as [`MeshSet::burn`].
    ///
    /// - **A transform per thread:** a [`CrsTransform`] needn't be
    ///   thread-safe (libproj's handles aren't), so `to_world` builds one
    ///   on each thread that projects into band `k`, the first time it
    ///   does, as [`MeshSet::weights_onto`] builds a transformer for each
    ///   band. It must give what `bands[k].to_world` gives:
    ///   [`crs_pair`](crate::crs::crs_pair) with the same two CRSs, or a
    ///   copy of a transform of your own.
    /// - **On the threads:** each band's polygons, each polygon of a
    ///   multipolygon on its own and those with the most vertices first,
    ///   are cut to the band, densified into its world and split where
    ///   they meet themselves. They are then filtered, walked and burned
    ///   as [`MeshSet::burn_par`] does.
    /// - **On the calling thread:** the points, through `bands`. With one
    ///   thread, or a band of one polygon, the polygons too, and
    ///   `to_world` isn't called.
    /// - **Errors:** the first a band's polygons give, in their order, as
    ///   [`MeshSet::burn`] gives it, or the first `to_world` returns.
    ///
    /// On ORCA2, in analytic polar LAEAs, Natural Earth's 10 m countries
    /// take 0.18 s on 8 threads at a tolerance of 1 m (0.44 s with
    /// [`MeshSet::burn_par`], 1.1 s on one), and 0.92 s at 0.01 m (2.2 s,
    /// 5.6 s). Building a transform costs: libproj takes about 5 ms for
    /// each, and slows when many threads build at once (32 took 0.2 s),
    /// so on a light burn fewer threads can be faster. Through libproj,
    /// ORCA2 at 1 m took 0.20 s on 16 threads and 0.31 s on 32.
    ///
    /// ```
    /// use rwarp::mesh::{Band, Halo, MeshSet};
    /// use rwarp::CrsTransform;
    /// use controlledburn::{BurnMode, Coord, Geometry, Polygon};
    ///
    /// // north-polar LAEA on the unit sphere, over a global grid of 30
    /// // degree cells north of the equator
    /// struct Laea;
    /// impl CrsTransform for Laea {
    ///     fn convert(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
    ///         let rho = 2.0 * ((90.0 - lat).to_radians() / 2.0).sin();
    ///         let (s, c) = lon.to_radians().sin_cos();
    ///         Some((rho * s, -rho * c))
    ///     }
    /// }
    /// let (nc, nr) = (12, 3);
    /// let (mut xb, mut yb) = (Vec::new(), Vec::new());
    /// for j in 0..nr {
    ///     for i in 0..nc {
    ///         for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
    ///             xb.push(-180.0 + 30.0 * (i + di) as f64);
    ///             yb.push(30.0 * (j + dj) as f64);
    ///         }
    ///     }
    /// }
    /// let bands = [Band { rows: 0..nr, to_world: &Laea }];
    /// let set = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap();
    ///
    /// // boxes 10 degrees on a side, one each 15 degrees of longitude
    /// let boxes: Vec<Geometry> = (0..24)
    ///     .map(|k| {
    ///         let (x, y) = (-180.0 + 15.0 * k as f64, 40.0);
    ///         let r = [(x, y), (x + 10.0, y), (x + 10.0, y + 10.0), (x, y + 10.0), (x, y)];
    ///         Geometry::Polygon(Polygon::new(vec![r.iter().map(|&(x, y)| Coord { x, y }).collect()]))
    ///     })
    ///     .collect();
    /// let one = set.burn(&boxes, &bands, 1e-6, BurnMode::Coverage).unwrap();
    /// let par = set.burn_par_with(&boxes, &bands, 1e-6, BurnMode::Coverage, 4, |_| Ok(Laea)).unwrap();
    /// assert_eq!(par, one);
    /// ```
    pub fn burn_par_with<F, T>(
        &self,
        geoms: &[Geometry],
        bands: &[Band],
        tolerance: f64,
        opts: impl Into<BurnOptions>,
        threads: usize,
        to_world: F,
    ) -> Result<BurnResult, String>
    where
        F: Fn(usize) -> Result<T, String> + Sync,
        T: CrsTransform,
    {
        self.burn_with(
            geoms,
            bands,
            tolerance,
            opts.into(),
            true,
            threads,
            Some(&to_world),
        )
    }

    /// [`MeshSet::burn_par_with`], with each piece's polygons filtered and
    /// their rings shortened by its [`Tiles`] (`trim`), or burned whole;
    /// with no `to_world`, projected on the calling thread.
    #[allow(clippy::too_many_arguments)]
    fn burn_with<F, T>(
        &self,
        geoms: &[Geometry],
        bands: &[Band],
        tolerance: f64,
        opts: BurnOptions,
        trim: bool,
        threads: usize,
        to_world: Option<&F>,
    ) -> Result<BurnResult, String>
    where
        F: Fn(usize) -> Result<T, String> + Sync,
        T: CrsTransform,
    {
        if opts.mode != BurnMode::Coverage {
            return Err("only coverage mode is defined for a grid front end".into());
        }
        if !(tolerance > 0.0 && tolerance.is_finite()) {
            return Err(format!("the tolerance must be positive, not {tolerance}"));
        }
        if bands.len() != self.clips.len()
            || bands.iter().zip(&self.clips).any(|(b, c)| b.rows != c.rows)
        {
            let rows: Vec<_> = self.clips.iter().map(|c| c.rows.clone()).collect();
            return Err(format!(
                "the bands must be those the set was built with, rows {rows:?}"
            ));
        }
        if let Some(k) = geoms
            .iter()
            .position(|g| matches!(g, Geometry::LineString(_) | Geometry::MultiLineString(_)))
        {
            return Err(format!(
                "geometry {k}: lines are not supported by the grid front ends yet"
            ));
        }
        let threads = match threads {
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
            n => n,
        };
        let boxes: Vec<[f64; 4]> = self.pieces.iter().map(|p| mesh_box(&p.mesh)).collect();
        let mut out = BurnResult::default();
        for (k, (band, clip)) in bands.iter().zip(&self.clips).enumerate() {
            let pieces: Vec<usize> = (0..self.pieces.len())
                .filter(|&i| self.pieces[i].world == k)
                .collect();
            if pieces.is_empty() {
                continue;
            }
            // the polygons in the band's world, with their positions and boxes
            let (mut world, mut ids, mut wbox) = (Vec::new(), Vec::new(), Vec::new());
            for (id, parts) in project_band(geoms, k, band, clip, tolerance, threads, to_world)? {
                wbox.push(parts.iter().fold(
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |a, p| {
                        [
                            a[0].min(p.bx[0]),
                            a[1].min(p.bx[1]),
                            a[2].max(p.bx[2]),
                            a[3].max(p.bx[3]),
                        ]
                    },
                ));
                world.push(parts);
                ids.push(id as i32);
            }
            // each piece's polygons, filtered and shortened, the largest
            // pieces first
            let select = |j: usize| {
                let i = pieces[j];
                let p = &self.pieces[i];
                // with no cell of any area, the walk has no point to wind round
                let tiles = p
                    .mesh
                    .interior_point()
                    .filter(|_| trim)
                    .map(|p0| (Tiles::new(&p.mesh), p0));
                let (mut sub, mut sel) = (Vec::new(), Vec::new());
                for (g, parts) in world.iter().enumerate() {
                    if !meets(wbox[g], boxes[i]) {
                        continue;
                    }
                    let polys: Vec<Polygon> = match &tiles {
                        None => parts.iter().map(Part::polygon).collect(),
                        Some((t, p0)) => parts
                            .iter()
                            .filter(|q| meets(q.bx, boxes[i]) && t.meets(q.bx))
                            .filter_map(|q| t.trim(q, *p0))
                            .collect(),
                    };
                    if !polys.is_empty() {
                        sub.push(Geometry::MultiPolygon(polys));
                        sel.push(g);
                    }
                }
                (sub, sel)
            };
            let mut order: Vec<usize> = (0..pieces.len()).collect();
            order.sort_by_key(|&j| {
                let m = &self.pieces[pieces[j]].mesh;
                std::cmp::Reverse(m.ncol() * m.nrow())
            });
            let subs = in_parallel(pieces.len(), &order, threads, select)?;
            // walked into each piece's index space, in chunks, the largest
            // first; one chunk a piece on one thread
            let mut jobs = Vec::new();
            for (j, (sub, _)) in subs.iter().enumerate() {
                if sub.is_empty() {
                    continue;
                }
                let chunks = if threads == 1 {
                    vec![(0, sub.len())]
                } else {
                    vertex_chunks(sub, threads * 8)
                };
                for (a, b) in chunks {
                    let n: usize = sub[a..b].iter().map(vertices).sum();
                    jobs.push((j, a, b, n));
                }
            }
            let mut order: Vec<usize> = (0..jobs.len()).collect();
            order.sort_by_key(|&q| std::cmp::Reverse(jobs[q].3));
            let walk = |q: usize| {
                let (j, a, b, _) = jobs[q];
                let mesh = &self.pieces[pieces[j]].mesh;
                let ring = |r: &[[f64; 2]]| mesh.ring_to_index(r);
                let cell = |c: [f64; 2], h| mesh.locate(c, h);
                map_geoms(&subs[j].0[a..b], a, mesh.nrow(), None, &ring, &cell)
            };
            let mut mapped = in_parallel(jobs.len(), &order, threads, walk)?.into_iter();
            // each piece burned, in order
            for (j, (sub, sel)) in subs.iter().enumerate() {
                if sub.is_empty() {
                    continue;
                }
                let p = &self.pieces[pieces[j]];
                let mut plane = Vec::with_capacity(sub.len());
                while plane.len() < sub.len() {
                    let (pl, _) = mapped.next().ok_or("a piece lost its walked polygons")??;
                    plane.extend(pl);
                }
                let res = burn_plane(p.mesh.ncol(), p.mesh.nrow(), &plane, opts, threads)?;
                let id = |j: i32| ids[sel[j as usize]];
                let (dr, dc) = (p.row_off as i32, p.col_off as i32);
                out.runs.extend(res.runs.iter().map(|r| GridRun {
                    row: r.row + dr,
                    col_start: r.col_start + dc,
                    col_end: r.col_end + dc,
                    id: id(r.id),
                }));
                out.edges.extend(res.edges.iter().map(|e| GridEdge {
                    row: e.row + dr,
                    col: e.col + dc,
                    fraction: e.fraction,
                    id: id(e.id),
                }));
                out.notes.extend(res.notes.into_iter().map(|n| Note {
                    geom_index: id(n.geom_index),
                    message: n.message,
                }));
            }
        }
        // points: in the first piece whose cell holds them
        for (id, g) in geoms.iter().enumerate() {
            let cs = match g {
                Geometry::Point(c) => std::slice::from_ref(c),
                Geometry::MultiPoint(cs) => cs.as_slice(),
                _ => continue,
            };
            for c in cs {
                if let Some((row, col)) = self.locate_point(c, bands, &boxes) {
                    out.points.push(GridPoint {
                        row: row as i32,
                        col: col as i32,
                        id: id as i32,
                    });
                }
            }
        }
        out.runs.sort_by_key(|r| (r.id, r.row, r.col_start));
        out.edges.sort_by_key(|e| (e.id, e.row, e.col));
        out.notes.sort_by_key(|n| n.geom_index);
        out.notes.dedup();
        Ok(out)
    }

    /// The set's cell `(row, col)` that holds point `c` (in the bounds'
    /// coordinates), searching the bands in order: `None` outside every
    /// band's y range, and where no piece holds it.
    fn locate_point(
        &self,
        c: &Coord,
        bands: &[Band],
        boxes: &[[f64; 4]],
    ) -> Option<(usize, usize)> {
        for (k, (band, clip)) in bands.iter().zip(&self.clips).enumerate() {
            if !(c.y >= clip.y[0] && c.y <= clip.y[1] && c.x.is_finite()) {
                continue;
            }
            let Some((x, y)) = band.to_world.convert(c.x, c.y) else {
                continue;
            };
            for (i, p) in self.pieces.iter().enumerate() {
                let b = boxes[i];
                if p.world != k || !(x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
                    continue;
                }
                if let Some((r, col)) = p.mesh.locate([x, y], None) {
                    return Some((r + p.row_off, col + p.col_off));
                }
            }
        }
        None
    }
}

/// The `to_world` of [`MeshSet::burn`] and [`MeshSet::burn_par`]: none,
/// so each band projects on the calling thread.
type Serial = fn(usize) -> Result<Box<dyn CrsTransform>, String>;

/// Each geometry's polygons in band `k`'s world (`band`, `clip`), as
/// [`project`] gives them, with the geometry's position: those with any
/// parts, in order. On up to `threads` threads, each through its own
/// transform from `to_world(k)`, built before its first polygon; with
/// none, one thread or one polygon, on the calling thread through
/// `band.to_world`. Err with the first polygon's (in order) that fails.
fn project_band<F, T>(
    geoms: &[Geometry],
    k: usize,
    band: &Band,
    clip: &Clip,
    tolerance: f64,
    threads: usize,
    to_world: Option<&F>,
) -> Result<Vec<(usize, Vec<Part>)>, String>
where
    F: Fn(usize) -> Result<T, String> + Sync,
    T: CrsTransform,
{
    let items: Vec<(usize, &Polygon)> = geoms
        .iter()
        .enumerate()
        .flat_map(|(id, g)| {
            let ps = match g {
                Geometry::Polygon(p) => std::slice::from_ref(p),
                Geometry::MultiPolygon(ps) => ps.as_slice(),
                _ => &[],
            };
            ps.iter().map(move |p| (id, p))
        })
        .collect();
    let options = Options::new(tolerance);
    let projected: Result<Vec<Vec<Part>>, (usize, [f64; 2])> = match to_world {
        Some(make) if threads > 1 && items.len() > 1 => {
            // the most vertices first
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by_key(|&i| {
                std::cmp::Reverse(items[i].1.rings.iter().map(Vec::len).sum::<usize>())
            });
            let init = || {
                let t = make(k).map_err(|e| format!("band {k}'s transform: {e}"))?;
                let forward = from_fn(move |x: f64, y: f64| t.convert(x, y).map(|(a, b)| [a, b]));
                Ok(Resampler::with_edge(forward, Planar::new(), &options))
            };
            let each = |rs: &mut Resampler<_, Planar>, i: usize| {
                project(std::slice::from_ref(items[i].1), clip.y, clip.step, rs)
            };
            in_parallel_with(items.len(), &order, threads, init, each)?
                .into_iter()
                .enumerate()
                .map(|(i, r)| r.map_err(|v| (i, v)))
                .collect()
        }
        _ => {
            let forward =
                from_fn(|x: f64, y: f64| band.to_world.convert(x, y).map(|(a, b)| [a, b]));
            let mut rs = Resampler::with_edge(forward, Planar::new(), &options);
            items
                .iter()
                .enumerate()
                .map(|(i, (_, p))| {
                    project(std::slice::from_ref(*p), clip.y, clip.step, &mut rs)
                        .map_err(|v| (i, v))
                })
                .collect()
        }
    };
    let projected = projected.map_err(|(i, v)| {
        format!(
            "geometry {}: the point ({}, {}) doesn't go to band {k}'s world",
            items[i].0, v[0], v[1]
        )
    })?;
    // each geometry's parts, its polygons' in order
    let mut out: Vec<(usize, Vec<Part>)> = Vec::new();
    for (&(id, _), parts) in items.iter().zip(projected) {
        match out.last_mut() {
            Some((last, all)) if *last == id => all.extend(parts),
            _ => out.push((id, parts)),
        }
    }
    out.retain(|(_, parts)| !parts.is_empty());
    Ok(out)
}

/// `f` for each of `0..n` on up to `threads` threads, taking the items in
/// turn in the order `order` (the costliest first, say): the results in
/// item order. One thread, or one item, runs on the calling thread.
fn in_parallel<T: Send>(
    n: usize,
    order: &[usize],
    threads: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Result<Vec<T>, String> {
    in_parallel_with(n, order, threads, || Ok(()), |_, i| f(i))
}

/// [`in_parallel`], each thread with state of its own, which needn't be
/// [`Send`]: `init()` builds it on the thread, unless every item is taken
/// by then, and `f(state, i)` does item `i`. A thread builds before it
/// takes an item, so where building is slow the costliest items go to the
/// threads that are ready. Err with an `init` error once the other
/// threads have finished, or if a thread panicked.
fn in_parallel_with<S, T: Send>(
    n: usize,
    order: &[usize],
    threads: usize,
    init: impl Fn() -> Result<S, String> + Sync,
    f: impl Fn(&mut S, usize) -> T + Sync,
) -> Result<Vec<T>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if threads <= 1 || n == 1 {
        let mut state = init()?;
        return Ok((0..n).map(|i| f(&mut state, i)).collect());
    }
    let (next, init, f) = (AtomicUsize::new(0), &init, &f);
    let mut done: Vec<(usize, T)> = Vec::with_capacity(n);
    let mut failed = None;
    let joined = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.min(n))
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    if next.load(Ordering::Relaxed) >= n {
                        return Ok(out);
                    }
                    let mut state = init()?;
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = order.get(k) else {
                            break Ok(out);
                        };
                        out.push((i, f(&mut state, i)));
                    }
                })
            })
            .collect();
        // every thread joined, so the scope has none left to panic over
        hs.into_iter().fold(true, |ok, h| match h.join() {
            Ok(Ok(v)) => {
                done.extend(v);
                ok
            }
            Ok(Err(e)) => {
                failed.get_or_insert(e);
                ok
            }
            Err(_) => false,
        })
    });
    if !joined {
        return Err("a burn thread panicked".into());
    }
    if let Some(e) = failed {
        return Err(e);
    }
    done.sort_by_key(|d| d.0);
    Ok(done.into_iter().map(|d| d.1).collect())
}

/// One polygon in a band's world: open rings, the outer first, the box
/// `[x0, y0, x1, y1]` of the outer ring, and each ring's chunk boxes.
struct Part {
    rings: Vec<Vec<[f64; 2]>>,
    bx: [f64; 4],
    /// For each ring, the box of each [`CHUNK`] segments' ends in turn
    /// (the last chunk shorter): chunk `c` holds segments `c * CHUNK` up to
    /// `(c + 1) * CHUNK`, and its box their vertices, both ends.
    chunks: Vec<Vec<[f64; 4]>>,
}

/// Segments in a ring's chunk, tested against a piece's tiles at once.
const CHUNK: usize = 32;

/// The box `[x0, y0, x1, y1]` of points `ps` (inverted when empty).
fn points_box<'a>(ps: impl IntoIterator<Item = &'a [f64; 2]>) -> [f64; 4] {
    ps.into_iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |a, c| {
            [
                a[0].min(c[0]),
                a[1].min(c[1]),
                a[2].max(c[0]),
                a[3].max(c[1]),
            ]
        },
    )
}

impl Part {
    /// A part of open rings `rings`, the outer first, with its boxes.
    fn new(rings: Vec<Vec<[f64; 2]>>) -> Part {
        let chunks: Vec<Vec<[f64; 4]>> = rings
            .iter()
            .map(|r| {
                let n = r.len();
                (0..n.div_ceil(CHUNK))
                    .map(|c| {
                        let (a, b) = (c * CHUNK, ((c + 1) * CHUNK).min(n));
                        points_box(r[a..b].iter().chain(Some(&r[b % n])))
                    })
                    .collect()
            })
            .collect();
        let bx = points_box(&rings[0]);
        Part { rings, bx, chunks }
    }

    /// The polygon whole, its rings closed as controlledburn takes them.
    fn polygon(&self) -> Polygon {
        Polygon {
            rings: self.rings.iter().map(|r| close(r)).collect(),
        }
    }
}

/// Open ring `r` closed, as controlledburn's coordinates.
fn close(r: &[[f64; 2]]) -> Vec<Coord> {
    r.iter()
        .chain(r.first())
        .map(|p| Coord { x: p[0], y: p[1] })
        .collect()
}

/// Polygons `ps` (in the bounds' coordinates) in a band's world: each
/// ring cut to the band's y range `y`, densified (`step`, `rs`) and split
/// where it meets itself; a part with its ring's turn keeps the ring's
/// role (outer or hole), one with the other turn takes the other. Err
/// with a point that doesn't go to the world.
fn project<T: ForwardBatch>(
    ps: &[Polygon],
    y: [f64; 2],
    step: f64,
    rs: &mut Resampler<T, Planar>,
) -> Result<Vec<Part>, [f64; 2]> {
    let mut parts = Vec::with_capacity(ps.len());
    for p in ps {
        let (mut outer, mut holes) = (Vec::new(), Vec::new());
        for (ri, r) in p.rings.iter().enumerate() {
            let cut = clip_y(r, y);
            if cut.len() < 3 {
                if ri == 0 {
                    break;
                }
                continue;
            }
            let wr = densify(rs, &cut, step)?;
            let turn = signed_area(&wr);
            for (part, a) in split_touching(&wr) {
                if (ri == 0) == ((a > 0.0) == (turn > 0.0)) {
                    outer.push((part, a.abs()));
                } else {
                    holes.push(part);
                }
            }
        }
        parts.extend(assemble(outer, holes));
    }
    Ok(parts)
}

/// The box `[x0, y0, x1, y1]` of a mesh's corners.
fn mesh_box(m: &QuadMesh) -> [f64; 4] {
    let ([x0, x1], [y0, y1]) = (range_of(&m.x), range_of(&m.y));
    [x0, y0, x1, y1]
}

/// Whether boxes `[x0, y0, x1, y1]` meet.
fn meets(a: [f64; 4], b: [f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Ring `r` (closed or not) cut to `lo <= y <= hi`, one side and then the
/// other (Sutherland and Hodgman), open. Where the ring leaves and comes
/// back, the cut runs along the line, there and back in places for a
/// ring that isn't convex: an edge with no area. Empty for a ring with a
/// coordinate that isn't finite, as [`QuadMesh::ring_to_index`] takes
/// it, and for a NaN range.
fn clip_y(r: &[Coord], [lo, hi]: [f64; 2]) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = r.iter().map(|c| [c.x, c.y]).collect();
    while pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
        return Vec::new();
    }
    if pts.iter().all(|p| p[1] >= lo && p[1] <= hi) {
        return pts;
    }
    let pts = clip_side(&pts, |y| y >= lo, lo);
    clip_side(&pts, |y| y <= hi, hi)
}

/// Ring `pts` cut to the side of `y = c` where `inside(y)`.
fn clip_side(pts: &[[f64; 2]], inside: impl Fn(f64) -> bool, c: f64) -> Vec<[f64; 2]> {
    let n = pts.len();
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(n + 4);
    let mut push = |p: [f64; 2]| {
        if out.last() != Some(&p) {
            out.push(p);
        }
    };
    for k in 0..n {
        let (a, b) = (pts[k], pts[(k + 1) % n]);
        let (ia, ib) = (inside(a[1]), inside(b[1]));
        if ia {
            push(a);
        }
        if ia != ib {
            let t = (c - a[1]) / (b[1] - a[1]);
            push([a[0] + t * (b[0] - a[0]), c]);
        }
    }
    while out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

/// Open ring `ring` (in the bounds' coordinates) to an open ring in a
/// band's world: each edge cut into equal pieces no longer than `step`
/// (from its lexicographically smaller end, so an edge shared by two
/// rings is cut the same from both sides), then densified by `rs`. Err
/// with a point that doesn't go to the world.
fn densify<T: ForwardBatch>(
    rs: &mut Resampler<T, Planar>,
    ring: &[[f64; 2]],
    step: f64,
) -> Result<Vec<[f64; 2]>, [f64; 2]> {
    let n = ring.len();
    let mut src = Vec::with_capacity(n + 1);
    for k in 0..n {
        let (a, b) = (ring[k], ring[(k + 1) % n]);
        src.push(a);
        let m = ((b[0] - a[0]).hypot(b[1] - a[1]) / step).ceil();
        if m > 1.0 {
            // at most 2^16 pieces an edge; bigcurve takes it from there
            let m = m.min(65536.0) as usize;
            let rev = (b[0], b[1]) < (a[0], a[1]);
            let (p, q) = if rev { (b, a) } else { (a, b) };
            let cut = (1..m).map(|i| {
                let t = i as f64 / m as f64;
                [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
            });
            if rev {
                src.extend(cut.collect::<Vec<_>>().into_iter().rev());
            } else {
                src.extend(cut);
            }
        }
    }
    src.push(ring[0]);
    let v = rs.project_all(&src);
    if let Some(bad) = v.iter().find(|v| !v.ok) {
        return Err([bad.lon, bad.lat]);
    }
    let pairs: Vec<(Vertex, Vertex)> = v.windows(2).map(|w| (w[0], w[1])).collect();
    let (mut interior, mut ends) = (Vec::new(), Vec::with_capacity(pairs.len()));
    rs.edges(&pairs, &mut interior, &mut ends);
    let mut out = Vec::with_capacity(v.len() + interior.len());
    let mut from = 0;
    for (k, &end) in ends.iter().enumerate() {
        out.push([v[k].x, v[k].y]);
        out.extend(interior[from..end].iter().map(|m| [m.x, m.y]));
        from = end;
    }
    Ok(out)
}

/// The signed area of an open ring (positive anticlockwise), taken about
/// its first point.
fn signed_area(r: &[[f64; 2]]) -> f64 {
    let Some(&o) = r.first() else { return 0.0 };
    let mut a = 0.0;
    for k in 1..r.len().saturating_sub(1) {
        let (p, q) = (r[k], r[k + 1]);
        a += (p[0] - o[0]) * (q[1] - o[1]) - (q[0] - o[0]) * (p[1] - o[1]);
    }
    a / 2.0
}

/// Open ring `r` split into loops where it comes back to a point it has
/// been at (within `1e-12` of its largest coordinate), each with its
/// signed area; loops with no area to speak of (under `1e-12` of the
/// ring's box) are dropped.
///
/// A polygon cut at 180 degrees or closed along a pole comes out of a
/// polar projection with both sides of the cut on one line: a slit in to
/// the pole, or a bridge from a coast to the band's cut. The walk can't
/// tell the two sides apart where the line is also a piece's wall, so
/// the slit is taken out here: what is left is the polygon's own
/// boundary, an Antarctica cut at a parallel as its coast and a hole.
fn split_touching(r: &[[f64; 2]]) -> Vec<(Vec<[f64; 2]>, f64)> {
    let ([x0, x1], [y0, y1]) = (
        range_of(&r.iter().map(|p| p[0]).collect::<Vec<_>>()),
        range_of(&r.iter().map(|p| p[1]).collect::<Vec<_>>()),
    );
    let scale = [x0, x1, y0, y1].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let near = 1e-12 * scale;
    let small = 1e-12 * (x1 - x0) * (y1 - y0);
    if near.is_nan() || near <= 0.0 {
        return Vec::new();
    }
    let key = |p: [f64; 2]| ((p[0] / near).floor() as i64, (p[1] / near).floor() as i64);
    // the loop being built, and where each of its points is by key
    let mut stack: Vec<[f64; 2]> = Vec::with_capacity(r.len());
    let mut at: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut loops = Vec::new();
    for &p in r {
        let (kx, ky) = key(p);
        let mut hit = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in at.get(&(kx + dx, ky + dy)).into_iter().flatten() {
                    let q = stack[i];
                    if (p[0] - q[0]).abs() <= near && (p[1] - q[1]).abs() <= near {
                        hit = hit.max(Some(i));
                    }
                }
            }
        }
        match hit {
            Some(i) => {
                // back at stack[i]: what came after it is a loop
                let tail = stack.split_off(i + 1);
                for q in &tail {
                    if let Some(v) = at.get_mut(&key(*q)) {
                        v.pop();
                    }
                }
                let mut lp = Vec::with_capacity(tail.len() + 1);
                lp.push(stack[i]);
                lp.extend(tail);
                loops.push(lp);
            }
            None => {
                at.entry((kx, ky)).or_default().push(stack.len());
                stack.push(p);
            }
        }
    }
    loops.push(stack);
    loops
        .into_iter()
        .filter(|l| l.len() >= 3)
        .map(|l| {
            let a = signed_area(&l);
            (l, a)
        })
        .filter(|(_, a)| a.abs() > small)
        .collect()
}

/// Polygons from outer rings (with their areas) and holes, each hole with
/// the smallest outer ring round its first point; a hole round no outer
/// ring is dropped. Rings closed, as controlledburn takes them.
fn assemble(outer: Vec<(Vec<[f64; 2]>, f64)>, holes: Vec<Vec<[f64; 2]>>) -> Vec<Part> {
    let mut rings: Vec<Vec<Vec<[f64; 2]>>> = outer.iter().map(|(r, _)| vec![r.clone()]).collect();
    for h in holes {
        let at = if outer.len() == 1 {
            Some(0)
        } else {
            (0..outer.len())
                .filter(|&i| winds(&outer[i].0, h[0]))
                .min_by(|&i, &j| outer[i].1.total_cmp(&outer[j].1))
        };
        if let Some(i) = at {
            rings[i].push(h);
        }
    }
    rings.into_iter().map(Part::new).collect()
}

/// Tiles over a piece's box in its world, marking each tile that a cell's
/// corners' box reaches, widened by a tile each way. A box or a segment
/// that meets no marked tile can't meet the piece, and is a tile or more
/// from every point of it. Held as a summed-area table.
struct Tiles {
    x0: f64,
    y0: f64,
    /// The tiles' side, in world units.
    side: f64,
    nx: usize,
    ny: usize,
    /// Marked tiles below and left of each tile corner, `(nx + 1) * (ny + 1)`.
    sum: Vec<u32>,
}

impl Tiles {
    /// Square tiles, twice as many along the box's longer side as the
    /// piece has cells along its longer side (4 to 512), and one more all
    /// round, so the widening fits.
    fn new(m: &QuadMesh) -> Tiles {
        let [x0, y0, x1, y1] = mesh_box(m);
        let n = (2 * m.ncol().max(m.nrow())).clamp(4, 512);
        let side = ((x1 - x0).max(y1 - y0) / n as f64).max(f64::MIN_POSITIVE);
        let nx = (((x1 - x0) / side).ceil() as usize).clamp(1, n) + 2;
        let ny = (((y1 - y0) / side).ceil() as usize).clamp(1, n) + 2;
        let mut t = Tiles {
            x0: x0 - side,
            y0: y0 - side,
            side,
            nx,
            ny,
            sum: vec![0; (nx + 1) * (ny + 1)],
        };
        let mut mark = vec![false; nx * ny];
        for r in 0..m.nrow() {
            for c in 0..m.ncol() {
                let q = m.corners(r, c);
                let b = q.iter().fold(
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |a, p| {
                        [
                            a[0].min(p[0]),
                            a[1].min(p[1]),
                            a[2].max(p[0]),
                            a[3].max(p[1]),
                        ]
                    },
                );
                let Some([i0, j0, i1, j1]) = t.range(b, 1) else {
                    continue;
                };
                for j in j0..=j1 {
                    mark[j * nx + i0..=j * nx + i1].fill(true);
                }
            }
        }
        let w = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
                t.sum[(j + 1) * w + i + 1] = t.sum[j * w + i + 1] + t.sum[(j + 1) * w + i]
                    - t.sum[j * w + i]
                    + mark[j * nx + i] as u32;
            }
        }
        t
    }

    /// The tiles `[i0, j0, i1, j1]` (inclusive) that box `b` reaches,
    /// widened by `grow` tiles each way, within the grid; `None` when it
    /// misses the grid.
    fn range(&self, b: [f64; 4], grow: i64) -> Option<[usize; 4]> {
        let at = |v: f64, o: f64| ((v - o) / self.side).floor() as i64;
        let (i0, i1) = (at(b[0], self.x0) - grow, at(b[2], self.x0) + grow);
        let (j0, j1) = (at(b[1], self.y0) - grow, at(b[3], self.y0) + grow);
        let (i0, i1) = (i0.max(0), i1.min(self.nx as i64 - 1));
        let (j0, j1) = (j0.max(0), j1.min(self.ny as i64 - 1));
        (i0 <= i1 && j0 <= j1).then_some([i0 as usize, j0 as usize, i1 as usize, j1 as usize])
    }

    /// Whether box `b` meets a marked tile.
    fn meets(&self, b: [f64; 4]) -> bool {
        let Some([i0, j0, i1, j1]) = self.range(b, 0) else {
            return false;
        };
        let w = self.nx + 1;
        let s = &self.sum;
        s[(j1 + 1) * w + i1 + 1] + s[j0 * w + i0] > s[j0 * w + i1 + 1] + s[(j1 + 1) * w + i0]
    }

    /// Whether segment `a, b` can't meet the piece: its box meets no marked
    /// tile.
    fn apart(&self, a: [f64; 2], b: [f64; 2]) -> bool {
        !self.meets([
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[0].max(b[0]),
            a[1].max(b[1]),
        ])
    }

    /// A ring round the whole grid, two tiles out: every side apart.
    fn around(&self) -> Vec<[f64; 2]> {
        let d = 2.0 * self.side;
        let (x1, y1) = (
            self.x0 + self.nx as f64 * self.side,
            self.y0 + self.ny as f64 * self.side,
        );
        let (x0, y0) = (self.x0 - d, self.y0 - d);
        let (x1, y1) = (x1 + d, y1 + d);
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    /// Part `q` as the piece's walk needs it, or `None` when its outer
    /// ring is apart from the piece and doesn't hold it. Each ring's runs
    /// of segments apart from the piece are shortened ([`Tiles::shorten`]);
    /// a ring wholly apart is dropped if it doesn't wind round the piece
    /// (a hole, then, that takes nothing) and is [`Tiles::around`] if it
    /// does. The walk sees the same crossings and the same winding round
    /// `p0`, its interior point, so it gives the same index rings.
    ///
    /// A chunk whose box is apart from the piece has all its segments
    /// apart, so only the segments of chunks near it are tested one by
    /// one.
    fn trim(&self, q: &Part, p0: [f64; 2]) -> Option<Polygon> {
        let mut rings = Vec::with_capacity(q.rings.len());
        for (ri, (r, chunks)) in q.rings.iter().zip(&q.chunks).enumerate() {
            let n = r.len();
            let near: Vec<bool> = chunks.iter().map(|&b| self.meets(b)).collect();
            let kept = if near.contains(&true) {
                self.runs(r, chunks, &near, p0)
            } else {
                None
            };
            match kept {
                Some(kept) => rings.push(close(&kept)),
                None => {
                    // wholly apart: round the piece, or round none of it
                    let w: i64 = (0..chunks.len())
                        .map(|c| {
                            chunk_crossings(r, chunks, c, c * CHUNK, ((c + 1) * CHUNK).min(n), p0)
                        })
                        .sum();
                    if w != 0 {
                        rings.push(close(&self.around()));
                    } else if ri == 0 {
                        return None;
                    }
                }
            }
        }
        Some(Polygon { rings })
    }

    /// Ring `r` (open, with its chunks' boxes, those `near` the piece
    /// marked) with each run of segments apart from the piece shortened;
    /// `None` when every segment is apart. Far chunks are passed whole.
    fn runs(
        &self,
        r: &[[f64; 2]],
        chunks: &[[f64; 4]],
        near: &[bool],
        p0: [f64; 2],
    ) -> Option<Vec<[f64; 2]>> {
        let n = r.len();
        let apart = |k: usize| !near[k / CHUNK] || self.apart(r[k], r[(k + 1) % n]);
        let mut out = Vec::new();
        let mut met = false;
        let mut k = 0;
        while k < n {
            if !apart(k) {
                met = true;
                out.push(r[k]);
                k += 1;
                continue;
            }
            // a run of segments apart from the piece, a..b
            let a = k;
            while k < n {
                if !near[k / CHUNK] {
                    k = ((k / CHUNK + 1) * CHUNK).min(n);
                } else if apart(k) {
                    k += 1;
                } else {
                    break;
                }
            }
            self.shorten(r, chunks, a, k, p0, &mut out);
        }
        met.then_some(out)
    }

    /// Vertices `a..=b` of ring `r` (vertex `n` is vertex 0), joined by
    /// segments apart from the piece: push those a walk round them keeps,
    /// all but `b`. Greedily, each shortcut runs as far as it can while it
    /// is apart from the piece too, and closes a loop of winding 0 round
    /// `p0` with the stretch it replaces. Being apart, the loop winds as
    /// often round every point of the piece, and its crossings of `p0`'s
    /// row are a tile or more from `p0`, so the walk's own winding test
    /// counts them alike.
    ///
    /// A shortcut tries a whole chunk ahead (its crossings are none when
    /// its box is all above or below `p0`'s row) before one vertex at a
    /// time, so a far run costs about a test a chunk.
    fn shorten(
        &self,
        r: &[[f64; 2]],
        chunks: &[[f64; 4]],
        a: usize,
        b: usize,
        p0: [f64; 2],
        out: &mut Vec<[f64; 2]>,
    ) {
        let n = r.len();
        let v = |i: usize| r[i % n];
        // from -> to holds, `w` the crossings of the stretch from..to
        let (mut from, mut to) = (a, a + 1);
        let mut w = crossing(v(a), v(a + 1), p0);
        out.push(v(a));
        // a whole chunk ahead failed: one vertex at a time to its end
        let mut slow = false;
        while to < b {
            let whole = !slow && to % CHUNK == 0 && to + CHUNK <= b;
            let (next, wn) = if whole {
                let c = to / CHUNK;
                (
                    to + CHUNK,
                    w + chunk_crossings(r, chunks, c, to, to + CHUNK, p0),
                )
            } else {
                (to + 1, w + crossing(v(to), v(to + 1), p0))
            };
            if wn + crossing(v(next), v(from), p0) == 0 && self.apart(v(from), v(next)) {
                (to, w) = (next, wn);
            } else if whole {
                slow = true;
                continue;
            } else {
                // from -> to is a shortcut; the next starts at `to`
                out.push(v(to));
                from = to;
                to = from + 1;
                w = crossing(v(from), v(to), p0);
            }
            if to % CHUNK == 0 {
                slow = false;
            }
        }
    }
}

/// The crossings of `p0`'s row by segments `a..b` of ring `r`, all in
/// chunk `c`: none when the chunk's box is all above or below the row.
fn chunk_crossings(
    r: &[[f64; 2]],
    chunks: &[[f64; 4]],
    c: usize,
    a: usize,
    b: usize,
    p0: [f64; 2],
) -> i64 {
    let bx = chunks[c];
    if p0[1] < bx[1] || p0[1] >= bx[3] {
        return 0;
    }
    let n = r.len();
    (a..b).map(|k| crossing(r[k], r[(k + 1) % n], p0)).sum()
}

/// Segment `a -> b`'s part in the winding number of a ring round `p`, as
/// the walk counts it: an upward crossing of `p`'s row with `p` on its
/// left is +1, a downward one with `p` on its right -1.
fn crossing(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> i64 {
    if a[1] <= p[1] {
        i64::from(b[1] > p[1] && orient(a, b, p) > 0.0)
    } else {
        -i64::from(b[1] <= p[1] && orient(a, b, p) < 0.0)
    }
}

/// Whether open ring `r` winds round point `p`.
fn winds(r: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = r.len();
    let mut w = 0i64;
    for k in 0..n {
        let (a, b) = (r[k], r[(k + 1) % n]);
        let side = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
        if a[1] <= p[1] {
            if b[1] > p[1] && side > 0.0 {
                w += 1;
            }
        } else if b[1] <= p[1] && side < 0.0 {
            w -= 1;
        }
    }
    w != 0
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

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 6371229.0;

    /// Longitude and latitude to polar LAEA: north for `1.0`, south for `-1.0`.
    struct Laea(f64);
    impl CrsTransform for Laea {
        fn convert(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
            let rho = 2.0 * R * ((90.0 - self.0 * lat).to_radians() / 2.0).sin();
            let (s, c) = lon.to_radians().sin_cos();
            Some((rho * s, -self.0 * rho * c))
        }
    }

    /// A 36 x 16 grid of 10 degree cells from 80 S to 80 N, with one corner
    /// moved well away inside a block of land: the cells round it are left
    /// out, and the rows beside the hole are pieces of their own.
    fn grid() -> (MeshSet, Laea, Laea) {
        let (nc, nr) = (36, 16);
        let corner = |j: usize, i: usize| {
            let (x, y) = (10.0 * i as f64, -80.0 + 10.0 * j as f64);
            if (j, i) == (12, 10) {
                (x + 25.0, y + 3.0)
            } else {
                (x, y)
            }
        };
        let (mut xb, mut yb) = (Vec::new(), Vec::new());
        for j in 0..nr {
            for i in 0..nc {
                for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)] {
                    let (x, y) = corner(jj, ii);
                    xb.push(x);
                    yb.push(y);
                }
            }
        }
        let land: Vec<bool> = (0..nc * nr)
            .map(|k| (10..14).contains(&(k / nc)) && (6..16).contains(&(k % nc)))
            .collect();
        let (south, north) = (Laea(-1.0), Laea(1.0));
        let bands = [
            Band {
                rows: 0..8,
                to_world: &south,
            },
            Band {
                rows: 8..16,
                to_world: &north,
            },
        ];
        let set =
            MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, Some(&land)).unwrap();
        (set, south, north)
    }

    fn ring(pts: impl IntoIterator<Item = [f64; 2]>) -> Vec<Coord> {
        let mut r: Vec<Coord> = pts
            .into_iter()
            .map(|p| Coord { x: p[0], y: p[1] })
            .collect();
        r.push(r[0]);
        r
    }

    fn lonlat_box(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<Coord> {
        ring([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    }

    /// A wobbly blob of `n` vertices round `(lon, lat)`, `r` degrees across,
    /// anticlockwise (`dir` 1) or clockwise (-1).
    fn blob(lon: f64, lat: f64, r: f64, n: usize, seed: f64, dir: f64) -> Vec<Coord> {
        ring((0..n).map(|k| {
            let a = dir * std::f64::consts::TAU * k as f64 / n as f64;
            let rr = r * (1.0 + 0.3 * (5.0 * a + seed).sin() + 0.1 * (17.0 * a + 2.0 * seed).cos());
            [
                lon + rr * a.cos() / (lat.to_radians().cos().max(0.2)),
                (lat + rr * a.sin()).clamp(-89.0, 89.0),
            ]
        }))
    }

    /// Polygons of every kind the trimming meets: blobs far from and across
    /// the strips, with holes, round a pole, across 180, the whole world.
    fn polygons() -> Vec<Geometry> {
        let poly = |rings: Vec<Vec<Coord>>| Geometry::Polygon(Polygon::new(rings));
        let mut g = vec![
            poly(vec![lonlat_box(-180.0, 180.0, -90.0, 90.0)]),
            poly(vec![lonlat_box(-180.0, 180.0, -90.0, -55.0)]),
            poly(vec![
                lonlat_box(-180.0, 180.0, 50.0, 90.0),
                lonlat_box(-180.0, 180.0, 65.0, 90.0),
            ]),
            Geometry::MultiPolygon(vec![
                Polygon::new(vec![lonlat_box(160.0, 180.0, -20.0, 45.0)]),
                Polygon::new(vec![lonlat_box(-180.0, -150.0, -20.0, 45.0)]),
            ]),
            // a long thin sliver along a parallel, through the hole's rows
            poly(vec![lonlat_box(-170.0, 170.0, 41.5, 43.5)]),
        ];
        let mut s = 1.0f64;
        let mut next = || {
            s = (s * 16807.0) % 2147483647.0;
            s / 2147483647.0
        };
        for k in 0..60 {
            let (lon, lat) = (-180.0 + 360.0 * next(), -75.0 + 150.0 * next());
            let r = 2.0 + 35.0 * next() * next();
            let n = 20 + (next() * 400.0) as usize;
            let outer = blob(lon, lat, r, n, k as f64, 1.0);
            g.push(if k % 3 == 0 {
                poly(vec![
                    outer,
                    blob(lon, lat, 0.4 * r, n / 2 + 3, -(k as f64), -1.0),
                ])
            } else {
                poly(vec![outer])
            });
        }
        g
    }

    /// Filtering each piece's polygons and shortening their rings changes
    /// nothing: the burn is the same, record for record and bit for bit.
    #[test]
    fn trimmed_burn_is_the_whole_burn() {
        let (set, south, north) = grid();
        assert!(set.pieces().iter().any(|p| p.mesh().nrow() == 1), "a strip");
        let bands = [
            Band {
                rows: 0..8,
                to_world: &south,
            },
            Band {
                rows: 8..16,
                to_world: &north,
            },
        ];
        let geoms = polygons();
        let opts = BurnOptions::coverage();
        let serial = None::<&Serial>;
        let whole = set
            .burn_with(&geoms, &bands, 1.0, opts, false, 1, serial)
            .unwrap();
        let trimmed = set
            .burn_with(&geoms, &bands, 1.0, opts, true, 1, serial)
            .unwrap();
        assert!(whole.runs.len() + whole.edges.len() > 1000);
        assert_eq!(whole, trimmed);
    }

    /// On any number of threads, the same burn record for record, with
    /// points too; with chunks small enough that several pieces walk
    /// theirs on several threads.
    #[test]
    fn burn_par_is_burn() {
        let (set, south, north) = grid();
        let bands = [
            Band {
                rows: 0..8,
                to_world: &south,
            },
            Band {
                rows: 8..16,
                to_world: &north,
            },
        ];
        let mut geoms = polygons();
        geoms.push(Geometry::MultiPoint(
            (0..50)
                .map(|k| Coord {
                    x: -175.0 + 7.0 * k as f64,
                    y: -70.0 + 2.9 * k as f64,
                })
                .collect(),
        ));
        let one = set.burn(&geoms, &bands, 1.0, BurnMode::Coverage).unwrap();
        assert!(!one.points.is_empty());
        for (threads, trim) in [
            (0, true),
            (1, true),
            (2, true),
            (3, true),
            (8, true),
            (3, false),
        ] {
            let opts = BurnOptions::coverage();
            let par = set
                .burn_with(&geoms, &bands, 1.0, opts, trim, threads, None::<&Serial>)
                .unwrap();
            assert_eq!(one, par, "{threads} threads, trim {trim}");
        }
        // the walk's errors come back, not a panic
        assert!(set
            .burn_par(&geoms, &bands, f64::NAN, BurnMode::Coverage, 4)
            .is_err());
    }

    /// Projected on the threads, each through a transform of its own, the
    /// same burn record for record; the same first error; `to_world`
    /// called once a band on each thread that projects, and not at all on
    /// one thread.
    #[test]
    fn burn_par_with_is_burn() {
        use std::sync::atomic::AtomicUsize;
        let (set, south, north) = grid();
        let bands = [
            Band {
                rows: 0..8,
                to_world: &south,
            },
            Band {
                rows: 8..16,
                to_world: &north,
            },
        ];
        let mut geoms = polygons();
        geoms.push(Geometry::MultiPoint(
            (0..50)
                .map(|k| Coord {
                    x: -175.0 + 7.0 * k as f64,
                    y: -70.0 + 2.9 * k as f64,
                })
                .collect(),
        ));
        let one = set.burn(&geoms, &bands, 1.0, BurnMode::Coverage).unwrap();
        let calls = AtomicUsize::new(0);
        let laea = |k: usize| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(Laea(if k == 0 { -1.0 } else { 1.0 }))
        };
        for (threads, trim) in [(0, true), (2, true), (3, true), (8, true), (3, false)] {
            calls.store(0, Ordering::Relaxed);
            let opts = BurnOptions::coverage();
            let par = set
                .burn_with(&geoms, &bands, 1.0, opts, trim, threads, Some(&laea))
                .unwrap();
            assert_eq!(one, par, "{threads} threads, trim {trim}");
            let n = calls.load(Ordering::Relaxed);
            if threads > 0 {
                assert!(
                    (2..=2 * threads).contains(&n),
                    "{n} calls, {threads} threads"
                );
            }
        }
        // one thread: the bands' own transforms
        let none = |_: usize| Err::<Laea, String>("not called".into());
        let r = set.burn_par_with(&geoms, &bands, 1.0, BurnMode::Coverage, 1, none);
        assert_eq!(r.unwrap(), one);
        // an error from `to_world`
        let r = set.burn_par_with(&geoms, &bands, 1.0, BurnMode::Coverage, 3, none);
        assert_eq!(r.unwrap_err(), "band 0's transform: not called");
        // points north of 85 N don't go to the north band: the first
        // polygon to have one, as on one thread, though the polygons with
        // the most vertices are projected first
        struct Capped(Laea);
        impl CrsTransform for Capped {
            fn convert(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
                (self.0 .0 < 0.0 || lat <= 85.0)
                    .then(|| self.0.convert(lon, lat))
                    .flatten()
            }
        }
        let (south, north) = (Capped(Laea(-1.0)), Capped(Laea(1.0)));
        let capped = [
            Band {
                rows: 0..8,
                to_world: &south,
            },
            Band {
                rows: 8..16,
                to_world: &north,
            },
        ];
        let geoms = &geoms[1..];
        let serial = set.burn(geoms, &capped, 1.0, BurnMode::Coverage);
        let e = serial.unwrap_err();
        assert!(e.starts_with("geometry 1: "), "{e}");
        for threads in [2, 8] {
            let make = |k: usize| Ok(Capped(Laea(if k == 0 { -1.0 } else { 1.0 })));
            let r = set.burn_par_with(geoms, &capped, 1.0, BurnMode::Coverage, threads, make);
            assert_eq!(r.unwrap_err(), e, "{threads} threads");
        }
    }

    /// Rings far from a strip lose their vertices there: one apart from it
    /// is gone, one round it is the tiles' ring, and one across it keeps
    /// only what is near it and a few shortcuts.
    #[test]
    fn trimming_drops_what_is_far() {
        let (set, _south, north) = grid();
        let p = set
            .pieces()
            .iter()
            .filter(|p| p.world() == 1 && p.mesh().nrow() == 1)
            .max_by_key(|p| p.mesh().ncol())
            .unwrap();
        let tiles = Tiles::new(&p.mesh);
        let p0 = p.mesh.interior_point().unwrap();
        let project = |r: &[Coord]| -> Vec<[f64; 2]> {
            r[..r.len() - 1]
                .iter()
                .map(|c| {
                    let (x, y) = north.convert(c.x, c.y).unwrap();
                    [x, y]
                })
                .collect()
        };
        let part = |r: Vec<[f64; 2]>| Part::new(vec![r]);
        let lat = {
            let [x, y] = p.mesh.corners(0, 0)[0];
            90.0 - 2.0 * (x.hypot(y) / (2.0 * R)).asin().to_degrees()
        };
        // a dense circle well north of the strip: apart, and round nothing
        let circle = |l: f64| project(&ring((0..2000).map(|k| [-180.0 + 0.18 * k as f64, l])));
        assert!(tiles.trim(&part(circle(lat + 25.0)), p0).is_none());
        // a dense circle well south of it: round it
        let round = tiles.trim(&part(circle(lat - 25.0)), p0).unwrap();
        assert_eq!(round.rings[0].len(), 5);
        // a dense ring across it: most of its vertices are far
        let across: Vec<[f64; 2]> = circle(lat + 25.0)
            .into_iter()
            .take(1000)
            .chain(circle(lat - 25.0).into_iter().take(1000).rev())
            .collect();
        let n = across.len();
        let t = tiles.trim(&part(across), p0).unwrap();
        eprintln!("across the strip: {n} vertices, {} kept", t.rings[0].len());
        assert!(t.rings[0].len() < n / 4, "{} of {n}", t.rings[0].len());
    }
}
