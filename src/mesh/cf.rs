//! Meshes from each cell's own four vertices (CF bounds, SCRIP corners),
//! halos, and the overlap check (decision N3).

use super::{orient, CellMap, QuadMesh};

/// Rows and columns that [`QuadMesh::from_cf_bounds`] drops before
/// building the mesh: an ocean model's halo (copies of other cells), and
/// rows that hold no cells.
///
/// A tripolar grid's seam and fold are walls that meet in the world, which
/// the walk crosses with nothing declared (see the
/// [module docs](crate::mesh#seams-and-folds)); a halo stores the cells
/// beyond them again, and those copies have to go, or they're counted
/// twice. [`QuadMesh::from_cf_bounds`] refuses a mesh whose cells overlap,
/// so a halo left in is an error, not a silent double count.
///
/// NEMO's ORCA grids, as written to CMIP6 files (checked on IPSL's eORCA1
/// and ORCA2):
///
/// - The last two columns repeat the first two (the cyclic halo).
/// - The last row repeats a row below it, mirrored (the north-fold halo).
/// - The first row is the closed southern boundary, land, with bounds that
///   in eORCA1 collapse each cell to a point.
/// - At a T-point pivot (ORCA2) the fold runs through the centres of the
///   second-to-last row, so that row mirrors itself: its cells are the same
///   ground in pairs. At an F-point pivot (ORCA1, eORCA1) the fold is the
///   top side of that row, and nothing more is needed.
///
/// [`Halo::NEMO_F_PIVOT`] and [`Halo::NEMO_T_PIVOT`] say so. With either,
/// cell `(row, col)` of the mesh is cell `(row + 1, col)` of the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Halo {
    /// Columns dropped at the start (`cols[0]`) and at the end (`cols[1]`).
    pub cols: [usize; 2],
    /// Rows dropped at the start (`rows[0]`, from row 0) and at the end
    /// (`rows[1]`).
    pub rows: [usize; 2],
    /// Whether the fold runs through the centres of the last row left, as
    /// at a T-point pivot: each cell of that row is then the same ground as
    /// its mirror. The mesh keeps the half of each below the fold, cut at
    /// the midpoints of the cell's column-line sides, and the two mesh
    /// cells of one model cell are its two halves. Give that row's model
    /// value to both (a halo-exchanged field has it in both already); its
    /// model area is the sum of the two mesh cells' areas.
    pub fold_through_centres: bool,
}

impl Halo {
    /// Nothing dropped.
    pub const NONE: Halo = Halo {
        cols: [0, 0],
        rows: [0, 0],
        fold_through_centres: false,
    };

    /// A NEMO ORCA grid whose north fold pivots on an F point (eORCA1,
    /// ORCA1): the last two columns, the first row, and the last row.
    pub const NEMO_F_PIVOT: Halo = Halo {
        cols: [0, 2],
        rows: [1, 1],
        fold_through_centres: false,
    };

    /// A NEMO ORCA grid whose north fold pivots on a T point (ORCA2): as
    /// [`Halo::NEMO_F_PIVOT`], and the fold runs through the centres of
    /// the last row left.
    pub const NEMO_T_PIVOT: Halo = Halo {
        cols: [0, 2],
        rows: [1, 1],
        fold_through_centres: true,
    };
}

/// The eight orders four vertices can be stored in: `ORDERS[o][s]` is the
/// stored vertex at CF slot `s` (0 at row `j`, column `i`; 1 at `j`,
/// `i + 1`; 2 at `j + 1`, `i + 1`; 3 at `j + 1`, `i`). CF's own order
/// first, then its other rotations, then the reflections.
const ORDERS: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [2, 1, 0, 3],
    [3, 2, 1, 0],
];

/// How far apart two copies of one corner may be, as a share of the larger
/// of the two cells: rounding, and bounds stored as `f32`.
const SHARED: f64 = 1e-3;

impl QuadMesh {
    /// A mesh from each cell's own four vertices, `nrow * ncol * 4` values
    /// for each coordinate, row-major with the vertex last, in any one CRS.
    ///
    /// That is CF's layout for the bounds of two-dimensional coordinates
    /// (`lon_bnds(j, i, 4)`), and SCRIP's (`grid_corner_lon(grid_size, 4)`,
    /// with `grid_dims` giving `ncol`, then `nrow`). Corners in degrees or
    /// radians are fine for a grid that is one sheet in them; a tripolar
    /// grid isn't (the pole is inside a cell), so project its bounds first
    /// (Lambert azimuthal equal-area on the north pole keeps areas true),
    /// see the [module docs](crate::mesh#seams-and-folds).
    ///
    /// The rows and columns in `halo` are dropped first, and the mesh is
    /// built from what's left: its cell `(row, col)` is the file's cell
    /// `(row + halo.rows[0], col + halo.cols[0])`.
    ///
    /// The vertex order is found from the data. CF's is vertex 0 at the
    /// cell's corner `(j, i)`, then `(j, i + 1)`, `(j + 1, i + 1)` and
    /// `(j + 1, i)`; any rotation or reflection of it is taken, the same
    /// for every cell (the first that fits, CF's own first). Neighbours
    /// must agree on the corners they share, to a thousandth of the larger
    /// cell (rounding, `f32` storage). Corner `(j, i)` is taken from cell
    /// `(j, i)`, whose first vertex it is, or on the last row or column
    /// from the cell beside it.
    ///
    /// Errors when the arrays have the wrong length or non-finite values,
    /// when the halo leaves no cells, when neighbours don't share corners
    /// in any vertex order (bounds of cells that don't tile, a halo row of
    /// collapsed cells left in, or longitudes 360 apart across a seam:
    /// [`unwrap_bounds`](super::unwrap_bounds) them first),
    /// on any error of [`QuadMesh::from_corners`], and when two cells
    /// overlap ([`QuadMesh::find_overlap`]), as a halo left in does.
    pub fn from_cf_bounds(
        ncol: usize,
        nrow: usize,
        xb: &[f64],
        yb: &[f64],
        halo: Halo,
        map: CellMap,
    ) -> Result<Self, String> {
        let n = ncol * nrow * 4;
        if xb.len() != n || yb.len() != n {
            return Err(format!(
                "bounds arrays have {} and {} values, expected nrow * ncol * 4 = {n}",
                xb.len(),
                yb.len()
            ));
        }
        if let Some(k) = xb.iter().chain(yb).position(|v| !v.is_finite()) {
            let k = k % n;
            return Err(format!(
                "the bounds of cell ({}, {}) are not finite",
                k / 4 / ncol,
                k / 4 % ncol
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
        // vertex `k` as stored, of the kept cell (j, i)
        let v = |j: usize, i: usize, k: usize| {
            let at = ((j + r0) * ncol + i + c0) * 4 + k;
            [xb[at], yb[at]]
        };
        let extent = |j: usize, i: usize| {
            let p: [[f64; 2]; 4] = std::array::from_fn(|k| v(j, i, k));
            let (mut x0, mut y0, mut x1, mut y1) = (p[0][0], p[0][1], p[0][0], p[0][1]);
            for q in &p[1..] {
                (x0, y0, x1, y1) = (x0.min(q[0]), y0.min(q[1]), x1.max(q[0]), y1.max(q[1]));
            }
            (x1 - x0).max(y1 - y0)
        };
        // the first pair of shared corners that disagree under order `o`
        let mismatch = |o: &[usize; 4]| -> Option<String> {
            let far = |a: [f64; 2], b: [f64; 2], tol: f64| {
                (a[0] - b[0]).abs() > tol || (a[1] - b[1]).abs() > tol
            };
            for j in 0..nr {
                for i in 0..nc {
                    // the next column's slots 0 and 3 are our 1 and 2; the
                    // next row's 0 and 1 are our 3 and 2
                    let pairs = [
                        (i + 1 < nc, j, i + 1, [(1, 0), (2, 3)]),
                        (j + 1 < nr, j + 1, i, [(3, 0), (2, 1)]),
                    ];
                    for (has, jn, in_, slots) in pairs {
                        if !has {
                            continue;
                        }
                        let tol = SHARED * extent(j, i).max(extent(jn, in_));
                        for (s, t) in slots {
                            let (a, b) = (v(j, i, o[s]), v(jn, in_, o[t]));
                            if far(a, b, tol) {
                                let d = [b[0] - a[0], b[1] - a[1]];
                                let period = [d[0].abs(), d[1].abs()].iter().any(|&x| {
                                    (x - 360.0).abs() < 1e-6
                                        || (x - std::f64::consts::TAU).abs() < 1e-9
                                });
                                return Some(format!(
                                    "cells ({}, {}) and ({}, {}) don't share a corner: ({}, {}) and ({}, {}){}",
                                    j + r0,
                                    i + c0,
                                    jn + r0,
                                    in_ + c0,
                                    a[0],
                                    a[1],
                                    b[0],
                                    b[1],
                                    if period {
                                        " are a period apart (longitudes across a seam?): unwrap \
                                         them (`unwrap_bounds`) and declare the period \
                                         (`QuadMesh::with_period`), or give the bounds in a \
                                         projection in which the grid is one sheet"
                                    } else {
                                        ""
                                    }
                                ));
                            }
                        }
                    }
                }
            }
            None
        };
        let mut first = None;
        let mut order = None;
        for o in &ORDERS {
            match mismatch(o) {
                None => {
                    order = Some(*o);
                    break;
                }
                Some(e) => {
                    first.get_or_insert(e);
                }
            }
        }
        let Some(o) = order else {
            return Err(format!(
                "{} (in CF's vertex order; no other order fits either). Drop rows or columns \
                 that aren't cells with `Halo`",
                first.unwrap_or_default()
            ));
        };
        // corner (j, i) from cell (j, i), whose first vertex it is; on the
        // last row and column from the cell beside it
        let w = nc + 1;
        let (mut x, mut y) = (vec![0.0; w * (nr + 1)], vec![0.0; w * (nr + 1)]);
        for j in 0..=nr {
            for i in 0..=nc {
                let p = match (j < nr, i < nc) {
                    (true, true) => v(j, i, o[0]),
                    (true, false) => v(j, nc - 1, o[1]),
                    (false, true) => v(nr - 1, i, o[3]),
                    (false, false) => v(nr - 1, nc - 1, o[2]),
                };
                x[j * w + i] = p[0];
                y[j * w + i] = p[1];
            }
        }
        if halo.fold_through_centres {
            // the fold runs through the last row only if that row mirrors
            // itself: its top corners are its bottom corners, reversed
            let mut bottom: Vec<[f64; 2]> = (0..=nc)
                .map(|i| [x[(nr - 1) * w + i], y[(nr - 1) * w + i]])
                .collect();
            bottom.sort_by(|a, b| a[0].total_cmp(&b[0]));
            for i in 0..=nc {
                let p = [x[nr * w + i], y[nr * w + i]];
                let cell = |c: usize| extent(nr - 1, c.min(nc - 1));
                let tol = SHARED * cell(i.saturating_sub(1)).max(cell(i));
                let lo = bottom.partition_point(|q| q[0] < p[0] - tol);
                let hit = bottom[lo..]
                    .iter()
                    .take_while(|q| q[0] <= p[0] + tol)
                    .any(|q| (q[1] - p[1]).abs() <= tol);
                if !hit {
                    return Err(format!(
                        "`fold_through_centres`, but row {} doesn't mirror itself: its top corner {i} \
                         isn't one of its bottom corners (a fold along the row's top side, as at an \
                         F-point pivot?)",
                        nr - 1 + r0
                    ));
                }
            }
            // the last row cut at its column-line sides' midpoints
            for i in 0..=nc {
                let (lo, hi) = ((nr - 1) * w + i, nr * w + i);
                x[hi] = 0.5 * (x[lo] + x[hi]);
                y[hi] = 0.5 * (y[lo] + y[hi]);
            }
        }
        let mesh = Self::from_corners_with_hint(nc, nr, x, y, map, "unwrap_bounds")?;
        if let Some([a, b]) = mesh.find_overlap() {
            let last = a.0 + 1 == nr && b.0 + 1 == nr;
            return Err(format!(
                "cells ({}, {}) and ({}, {}) overlap{}",
                a.0 + r0,
                a.1 + c0,
                b.0 + r0,
                b.1 + c0,
                if last && !halo.fold_through_centres {
                    ": the last row mirrors itself, as at a T-point pivot (`fold_through_centres`)?"
                } else {
                    ": a row or column repeated, as in an ocean model's halo? Drop it with `Halo`"
                }
            ));
        }
        Ok(mesh)
    }

    /// Two cells that overlap, if any: a point inside one that the other
    /// owns too, or, with a declared period ([`QuadMesh::with_period`]),
    /// owns a whole number of periods away.
    ///
    /// [`QuadMesh::from_corners`] refuses cells of both orientations, which
    /// is how an overlapping swath scan shows; a copy that keeps the
    /// orientation (a row or column stored twice, as in an ocean model's
    /// halo, or a row mirrored by a half-turn in index space, as at a
    /// tripolar fold) passes that test and is found here. Each cell with
    /// area is probed once, at the centroid of the larger triangle of the
    /// diagonal that lies inside it; an overlap that misses every probe
    /// (thinner than about half a cell) isn't found.
    /// [`QuadMesh::from_cf_bounds`] runs this check, and
    /// [`QuadMesh::with_period`] its part across the period.
    pub fn find_overlap(&self) -> Option<[(usize, usize); 2]> {
        self.overlap(true)
    }

    /// [`QuadMesh::find_overlap`]: in the plane only with `planar`, and
    /// across the declared period if there is one.
    pub(crate) fn overlap(&self, planar: bool) -> Option<[(usize, usize); 2]> {
        let at = |m: usize| (m / self.ncol, m % self.ncol);
        for (k, cell) in self.cells.iter().enumerate() {
            if cell.area2 == 0.0 {
                continue;
            }
            let Some(p) = self.probe(k) else {
                continue;
            };
            if planar {
                if let Some(other) = self
                    .buckets()
                    .candidates(p)
                    .find(|&m| m != k && self.contains(m / self.ncol, m % self.ncol, p))
                {
                    return Some([at(k), at(other)]);
                }
            }
            if let Some(per) = self.period {
                // a cell may meet itself one period on, if it's that wide
                for s in per.point_shifts(p[0]).skip(1) {
                    let q = [p[0] + s, p[1]];
                    if let Some(other) = self
                        .buckets()
                        .candidates(q)
                        .find(|&m| self.contains(m / self.ncol, m % self.ncol, q))
                    {
                        return Some([at(k), at(other)]);
                    }
                }
            }
        }
        None
    }

    /// A point inside cell `k` (which has area), for the overlap check:
    /// the centroid of the larger triangle of the diagonal that lies
    /// inside it. `None` if neither diagonal does.
    fn probe(&self, k: usize) -> Option<[f64; 2]> {
        let s = self.sign;
        let q = self.corners(k / self.ncol, k % self.ncol);
        // the diagonal whose two triangles both turn the cell's way
        let tri = |a: usize, b: usize, d: usize| (s * orient(q[a], q[b], q[d]), [q[a], q[b], q[d]]);
        let split = |a: usize| {
            let (t, u) = (
                tri(a, (a + 1) % 4, (a + 2) % 4),
                tri(a, (a + 2) % 4, (a + 3) % 4),
            );
            (t.0 > 0.0 && u.0 > 0.0).then_some(if t.0 >= u.0 { t.1 } else { u.1 })
        };
        let t = split(0).or_else(|| split(1))?;
        Some([
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
        ])
    }
}
