//! A [`MeshSet`] as a warp source: [`SetTransformer`].

use vaster::inv_geotransform;

use super::MeshSet;
use crate::crs::{crs_pair, CrsTransform};
use crate::mesh::QuadMesh;
use crate::transform::{SourceCells, SourceSheet, Transformer};
use crate::weights::GridWindow;

/// A source whose pixels are the cells of a [`MeshSet`], as a
/// [`Transformer`] from a destination grid's pixels: for
/// [`warp`](crate::warp::warp) and
/// [`exact_weights`](crate::weights::exact_weights) from a whole ocean
/// grid in pieces.
///
/// Source pixel `(col, row)` is the set's cell `(row, col)`, in the set's
/// numbering (the file's rows and columns less the halo's), so a source
/// buffer holds one value per cell, row-major over
/// [`MeshSet::size`]: [`MeshSet::from_file_layout`] takes a field on the
/// file's grid to it. Each band's world is in its own CRS; the
/// destination is a grid with geotransform `dst_gt` in `dst_crs`.
///
/// - **Area methods.** The transformer reports the set's pieces as the
///   source's cells, in several worlds
///   ([`SourceCells::sheets`]). With
///   [`Footprint::Exact`](crate::warp::Footprint::Exact) each destination
///   pixel's outline goes to every band's world, and is walked onto each
///   of the band's pieces whose box it meets. Their coverage feeds one
///   reduction, so `Average`, `Mode`, `Min` and the others take cells
///   from every piece and band the pixel reaches, each weighted by its
///   world area. A pixel across two bands misses the gap between them
///   (see [`MeshSet`]), and the average is over what it covers.
/// - **Inside out.** A polar projection sends its far pole to its edge,
///   so an outline round that pole comes out round the whole world, turned
///   the other way. Each band's turn is found from small triangles at
///   pixel centres, and an outline turned against it is not measured in
///   that band. So a south-polar destination grid takes nothing from a
///   north-polar band at the South Pole.
/// - **Point kernels** map a destination pixel into the set's index space
///   through the piece whose cell holds it (bands in order), with the
///   piece's own [`CellMap`](crate::mesh::CellMap). They read
///   neighbouring cells across the set's index space, across pieces and
///   bands as across a file's arrays. Points in a cell left out, in the
///   gap between bands or off the grid have no source pixel.
/// - **Cells left out** are in no piece: the area methods never reach
///   them and point kernels never land in them, but a point kernel's
///   neighbourhood reads their values, as GDAL's would read a land cell's.
///   Give them the source's nodata.
///
/// ```
/// use rwarp::mesh::{Band, Halo, MeshSet, SetTransformer};
/// use rwarp::warp::{warp, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
/// use rwarp::CrsTransform;
///
/// // a global grid of 30 degree cells, bounds in lon/lat, in two bands of
/// // polar LAEA on the unit sphere split at the equator
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
/// // lon/lat to a band's world (`.1` false) and back (true); `.0` is 1
/// // for north-polar, -1 for south-polar
/// struct Laea(f64, bool);
/// impl CrsTransform for Laea {
///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
///         let s = self.0;
///         Some(if self.1 {
///             let lat = s * (90.0 - 2.0 * (x.hypot(y) / 2.0).min(1.0).asin().to_degrees());
///             (x.atan2(-s * y).to_degrees(), lat)
///         } else {
///             let rho = 2.0 * ((90.0 - s * y).to_radians() / 2.0).sin();
///             let (a, b) = x.to_radians().sin_cos();
///             (rho * a, -s * rho * b)
///         })
///     }
/// }
/// let (south, north) = (Laea(-1.0, false), Laea(1.0, false));
/// let bands = [Band { rows: 0..3, to_world: &south }, Band { rows: 3..6, to_world: &north }];
/// let set = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap();
///
/// // to a 10 degree lon/lat grid: each band's world to lon/lat and back
/// let gt = [-180.0, 10.0, 0.0, 90.0, 0.0, -10.0];
/// let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> = vec![
///     (Box::new(Laea(-1.0, false)), Box::new(Laea(-1.0, true))),
///     (Box::new(Laea(1.0, false)), Box::new(Laea(1.0, true))),
/// ];
/// let t = SetTransformer::with_backend(&set, gt, crs).unwrap();
///
/// // each cell's value is its row
/// let data: Vec<f64> = (0..nr * nc).map(|k| (k / nc) as f64).collect();
/// let src = SourceBuffer { data: &data, ncol: nc, nrow: nr, nbands: 1, col_off: 0, row_off: 0 };
/// let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify: 8 });
/// let out = warp(&t, &src, 36, 18, &opts);
/// // 20 to 30 N is all row 3, 70 to 80 S all row 0
/// assert!((out[6 * 36 + 5] - 3.0).abs() < 1e-12 && out[16 * 36] == 0.0);
/// // 0 to 10 N, next to the gap at the equator, still row 3
/// assert!((out[8 * 36 + 20] - 3.0).abs() < 1e-12);
/// ```
pub struct SetTransformer<'a> {
    set: &'a MeshSet,
    dst_gt: [f64; 6],
    dst_inv_gt: [f64; 6],
    /// Each band's: destination coordinates to its world, and back.
    crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)>,
    /// Each piece's box `[x0, y0, x1, y1]` in its world.
    boxes: Vec<[f64; 4]>,
    /// Each band's pieces.
    by_band: Vec<Vec<usize>>,
    /// The piece of each of the set's cells, row-major (`u32::MAX` for a
    /// cell left out).
    owner: Vec<u32>,
}

impl<'a> SetTransformer<'a> {
    /// The set, its band `k` in CRS `band_crs[k]`, to the grid `dst_gt` in
    /// `dst_crs`, with the compiled-in CRS backend. Fails if a CRS isn't
    /// understood, the bands aren't the set's, or the geotransform isn't
    /// invertible.
    pub fn new(
        set: &'a MeshSet,
        band_crs: &[&str],
        dst_gt: [f64; 6],
        dst_crs: &str,
    ) -> Result<Self, String> {
        let crs = band_crs
            .iter()
            .map(|c| crs_pair(c, dst_crs))
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_backend(set, dst_gt, crs)
    }

    /// With caller-supplied CRS transforms, one pair a band: destination
    /// coordinates to the band's world, and back. Fails if there isn't a
    /// pair for each band, or the geotransform isn't invertible.
    pub fn with_backend(
        set: &'a MeshSet,
        dst_gt: [f64; 6],
        crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)>,
    ) -> Result<Self, String> {
        let nb = set.band_count();
        if crs.len() != nb {
            return Err(format!(
                "{} CRS transforms for a set of {nb} bands",
                crs.len()
            ));
        }
        let dst_inv_gt =
            inv_geotransform(&dst_gt).ok_or_else(|| "geotransform not invertible".to_string())?;
        let [ncol, nrow] = set.size();
        let mut owner = vec![u32::MAX; ncol * nrow];
        let mut by_band = vec![Vec::new(); nb];
        let mut boxes = Vec::with_capacity(set.pieces().len());
        for (i, p) in set.pieces().iter().enumerate() {
            let [c0, r0] = p.offset();
            let m = p.mesh();
            for r in r0..r0 + m.nrow() {
                owner[r * ncol + c0..r * ncol + c0 + m.ncol()].fill(i as u32);
            }
            by_band[p.world()].push(i);
            boxes.push(corners_box(m));
        }
        Ok(SetTransformer {
            set,
            dst_gt,
            dst_inv_gt,
            crs,
            boxes,
            by_band,
            owner,
        })
    }

    /// The source's cells.
    pub fn source_set(&self) -> &MeshSet {
        self.set
    }

    /// A source window of the whole set, for
    /// [`exact_weights`](crate::weights::exact_weights).
    pub fn source_window(&self) -> GridWindow {
        let [ncol, nrow] = self.set.size();
        GridWindow::new(ncol, nrow)
    }

    /// Destination pixel/line to band `k`'s world.
    fn to_world(&self, k: usize, px: f64, py: f64) -> Option<[f64; 2]> {
        let g = &self.dst_gt;
        let (gx, gy) = (g[0] + px * g[1] + py * g[2], g[3] + px * g[4] + py * g[5]);
        let (wx, wy) = self.crs[k].0.convert(gx, gy)?;
        (wx.is_finite() && wy.is_finite()).then_some([wx, wy])
    }

    /// Destination pixel/line to the set's index space `[col, row]`: in
    /// the first piece (bands in order) whose cell holds it, trying the
    /// piece of the last point first. `last` is that piece and its cell.
    fn dst_to_index(
        &self,
        px: f64,
        py: f64,
        last: &mut Option<(usize, (usize, usize))>,
    ) -> Option<[f64; 2]> {
        let pieces = self.set.pieces();
        let at = |i: usize, w: [f64; 2], hint| {
            let b = self.boxes[i];
            if !(w[0] >= b[0] && w[0] <= b[2] && w[1] >= b[1] && w[1] <= b[3]) {
                return None;
            }
            pieces[i].mesh().to_index(w, hint)
        };
        for (k, band) in self.by_band.iter().enumerate() {
            if band.is_empty() {
                continue;
            }
            let Some(w) = self.to_world(k, px, py) else {
                continue;
            };
            let first = last.filter(|&(i, _)| pieces[i].world() == k);
            let tries = first.map(|(i, h)| (i, Some(h))).into_iter().chain(
                band.iter()
                    .filter(|&&i| Some(i) != first.map(|f| f.0))
                    .map(|&i| (i, None)),
            );
            for (i, hint) in tries {
                if let Some(idx) = at(i, w, hint) {
                    let m = pieces[i].mesh();
                    let cell = (
                        (idx[1] as usize).min(m.nrow() - 1),
                        (idx[0] as usize).min(m.ncol() - 1),
                    );
                    *last = Some((i, cell));
                    let [c0, r0] = pieces[i].offset();
                    return Some([idx[0] + c0 as f64, idx[1] + r0 as f64]);
                }
            }
        }
        None
    }

    /// The set's index space `[col, row]` to destination pixel/line,
    /// through the piece of the cell that holds the point (or, on a cell's
    /// edge, of a neighbour that shares it).
    fn index_to_dst(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let [ncol, nrow] = self.set.size();
        if !(x >= 0.0 && y >= 0.0 && x <= ncol as f64 && y <= nrow as f64) {
            return None;
        }
        // the cell under the point, and the one before it on each axis
        // where the point is on the edge they share
        let (c, r) = ((x as usize).min(ncol - 1), (y as usize).min(nrow - 1));
        let cols = [Some(c), (x == c as f64 && c > 0).then(|| c - 1)];
        let rows = [Some(r), (y == r as f64 && r > 0).then(|| r - 1)];
        for row in rows.into_iter().flatten() {
            for col in cols.into_iter().flatten() {
                let i = self.owner[row * ncol + col];
                if i == u32::MAX {
                    continue;
                }
                let p = &self.set.pieces()[i as usize];
                let [c0, r0] = p.offset();
                let Some(w) = p.mesh().to_world([x - c0 as f64, y - r0 as f64]) else {
                    continue;
                };
                let Some((gx, gy)) = self.crs[p.world()].1.convert(w[0], w[1]) else {
                    continue;
                };
                let g = &self.dst_inv_gt;
                return Some([g[0] + gx * g[1] + gy * g[2], g[3] + gx * g[4] + gy * g[5]]);
            }
        }
        None
    }
}

/// The box `[x0, y0, x1, y1]` of a mesh's corners.
fn corners_box(m: &QuadMesh) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            for p in m.corners(r, c) {
                b = [
                    b[0].min(p[0]),
                    b[1].min(p[1]),
                    b[2].max(p[0]),
                    b[3].max(p[1]),
                ];
            }
        }
    }
    b
}

/// Points that fail go to infinity, with `false`, as in the other
/// transformers.
fn put(x: &mut f64, y: &mut f64, r: Option<[f64; 2]>) -> bool {
    match r {
        Some([a, b]) if a.is_finite() && b.is_finite() => {
            (*x, *y) = (a, b);
            true
        }
        _ => {
            (*x, *y) = (f64::INFINITY, f64::INFINITY);
            false
        }
    }
}

/// The set's pieces as sheets, one world a band.
impl SourceCells for SetTransformer<'_> {
    /// Band 0's world; see [`SourceCells::dst_to_world_in`].
    fn dst_to_world(&self, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        self.dst_to_world_in(0, x, y)
    }

    /// Refused: the cells are in several worlds ([`SourceCells::sheets`]).
    fn mesh(&self, _: usize, _: usize, _: usize, _: usize) -> Result<QuadMesh, String> {
        Err("a MeshSet's cells are in several worlds: see SourceCells::sheets".into())
    }

    fn worlds(&self) -> usize {
        self.crs.len()
    }

    fn dst_to_world_in(&self, world: usize, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| put(x, y, self.to_world(world, *x, *y)))
            .collect()
    }

    /// Each piece's cells in the window, as a mesh of their own in its
    /// band's world.
    fn sheets(
        &self,
        col_off: usize,
        row_off: usize,
        ncol: usize,
        nrow: usize,
    ) -> Result<Vec<SourceSheet>, String> {
        let mut out = Vec::new();
        for p in self.set.pieces() {
            let [c0, r0] = p.offset();
            let m = p.mesh();
            let (a, b) = (c0.max(col_off), (c0 + m.ncol()).min(col_off + ncol));
            let (c, d) = (r0.max(row_off), (r0 + m.nrow()).min(row_off + nrow));
            if a >= b || c >= d {
                continue;
            }
            out.push(SourceSheet {
                world: p.world(),
                col_off: a - col_off,
                row_off: c - row_off,
                mesh: m.window(a - c0, c - r0, b - a, d - c)?,
            });
        }
        Ok(out)
    }
}

/// `dst_to_src`: destination pixel/line to the set's index space `[col,
/// row]`, which is source pixel/line; the reverse through the piece that
/// holds the point. Points off every piece fail.
impl Transformer for SetTransformer<'_> {
    fn source_cells(&self) -> Option<&dyn SourceCells> {
        Some(self)
    }

    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let mut last = None;
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| {
                let r = if dst_to_src {
                    self.dst_to_index(*x, *y, &mut last)
                } else {
                    self.index_to_dst(*x, *y)
                };
                put(x, y, r)
            })
            .collect()
    }
}
