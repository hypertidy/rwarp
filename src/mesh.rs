//! Quadrilateral meshes: curvilinear grids as cells, for coverage.
//!
//! Stage 1 of `design/curvilinear-frontend.md`, and unstable: the API may
//! change while the ring walk (stage 2) is built on it. A [`QuadMesh`] holds the
//! `(nrow + 1) x (ncol + 1)` corners of a curvilinear grid, validates
//! them, and maps points between the grid's own coordinates ("world", in
//! whatever CRS the corners are given in) and its index space: fractional
//! `(col, row)`, with cell `(row, col)` covering `[col, col + 1) x [row,
//! row + 1)` and row 0 the first row of the arrays as stored. That is the
//! space controlledburn burns in (its `design/grid-frontends.md`).
//!
//! Cells are the straight-edged quadrilaterals through their corners. The
//! [`CellMap`] says how a cell's inside maps to the unit square:
//!
//! - [`CellMap::Bilinear`] is GDAL's geolocation model (the inverse is
//!   rwarp's port of `GDALInverseBilinearInterpolation`). Equal areas in
//!   index space are not equal areas in the world, so index-space coverage
//!   fractions are off by up to about 1.5% of a cell on curved grids.
//! - [`CellMap::EqualArea`] splits each cell into four triangles at a
//!   centre point and the unit square into four matching triangles, with
//!   the unit centre chosen so each pair holds the same share of area. The
//!   map is affine per triangle with one area scale per cell, so an area
//!   fraction of a cell in index space is the same fraction of the cell in
//!   the world, exactly. This is the map for coverage and area methods.
//!
//! The cells of the two maps are the same quadrilaterals; only the
//! interior map differs.
//!
//! No CRS is involved: the module builds with `--no-default-features`.
//! The walk that moves whole rings into index space (stage 2) builds on
//! this.
//!
//! # Ownership
//!
//! Every point of the mesh belongs to exactly one cell. Orientation tests
//! on each side are computed once, in one canonical direction, so the two
//! cells that share a side see bitwise the same value. A point on a side
//! belongs to the cell on the side's low-index side: a point on the wall
//! between columns `c - 1` and `c` belongs to column `c`, and a point on
//! the wall between rows `r - 1` and `r` belongs to row `r`, as in a
//! half-open raster. So points on the last column's far wall or the last
//! row's far wall are outside the mesh.
//!
//! # Example
//!
//! ```
//! use rwarp::mesh::{CellMap, QuadMesh};
//!
//! // a 3 x 2 polar sector: columns along the angle, rows along the radius
//! let (ncol, nrow) = (3, 2);
//! let (mut x, mut y) = (Vec::new(), Vec::new());
//! for j in 0..=nrow {
//!     for i in 0..=ncol {
//!         let (r, a) = (10.0 + 5.0 * j as f64, (30.0 * i as f64).to_radians());
//!         x.push(r * a.cos());
//!         y.push(r * a.sin());
//!     }
//! }
//! let mesh = QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap();
//! let p = [12.0, 9.0];
//! let idx = mesh.to_index(p, None).unwrap();
//! let back = mesh.to_world(idx).unwrap();
//! assert!((back[0] - p[0]).abs() < 1e-12 && (back[1] - p[1]).abs() < 1e-12);
//! ```

use std::sync::OnceLock;

use crate::geoloc_transform::inverse_bilinear;

/// How a cell's inside maps to the unit square.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellMap {
    /// GDAL's geolocation model: bilinear in the corners. The default,
    /// for parity with point kernels and `geo_to_pixel`.
    #[default]
    Bilinear,
    /// Four triangles per cell with matching area shares: index-space
    /// area fractions are world-area fractions.
    EqualArea,
}

/// The unit square's corners, in the order of a cell's corners `P00, P10,
/// P11, P01`: `(u, v)` with `u` along columns and `v` along rows.
const UNIT: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Twice the signed area of the triangle `a, b, p` (positive when `p` is
/// to the left of `a -> b` in a y-up plane).
#[inline]
fn orient(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Per-cell data computed at construction.
#[derive(Debug, Clone, Copy)]
struct Cell {
    /// Twice the signed area (shoelace over P00, P10, P11, P01).
    area2: f64,
    /// The world centre of the four-triangle split (in the cell's kernel).
    centre: [f64; 2],
    /// The unit-square centre with matching area shares.
    unit: [f64; 2],
    /// Whether the split is exact (always, for convex cells).
    exact: bool,
}

/// A curvilinear grid as straight-edged quadrilateral cells.
#[derive(Debug, Clone)]
pub struct QuadMesh {
    ncol: usize,
    nrow: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    map: CellMap,
    /// +1 or -1: the orientation of every cell with area.
    sign: f64,
    cells: Vec<Cell>,
    buckets: OnceLock<Buckets>,
}

impl QuadMesh {
    /// A mesh from row-major corner arrays of `(nrow + 1) * (ncol + 1)`
    /// values each: corner `(j, i)` (row line `j`, column line `i`) is at
    /// `x[j * (ncol + 1) + i]`.
    ///
    /// Errors when the arrays have the wrong length or non-finite values,
    /// when a cell is a bow-tie (two opposite sides cross), or when cells
    /// with area do not all have the same orientation (a fold: cells
    /// overlap, as on an overlapping swath scan, and location would be
    /// ambiguous). Zero-area cells (collapsed corners) are allowed; they
    /// contain no point.
    pub fn from_corners(
        ncol: usize,
        nrow: usize,
        x: Vec<f64>,
        y: Vec<f64>,
        map: CellMap,
    ) -> Result<Self, String> {
        if ncol == 0 || nrow == 0 {
            return Err(format!(
                "a mesh needs at least one cell, got {ncol} x {nrow}"
            ));
        }
        let n = (ncol + 1) * (nrow + 1);
        if x.len() != n || y.len() != n {
            return Err(format!(
                "corner arrays have {} and {} values, expected (ncol + 1) * (nrow + 1) = {n}",
                x.len(),
                y.len()
            ));
        }
        if let Some(k) = x.iter().chain(&y).position(|v| !v.is_finite()) {
            return Err(format!("corner value {} is not finite", k % n));
        }
        let mut mesh = QuadMesh {
            ncol,
            nrow,
            x,
            y,
            map,
            sign: 1.0,
            cells: Vec::with_capacity(ncol * nrow),
            buckets: OnceLock::new(),
        };

        let (mut pos, mut neg) = (0usize, 0usize);
        for r in 0..nrow {
            for c in 0..ncol {
                let q = mesh.corners(r, c);
                if bow_tie(&q) {
                    return Err(format!(
                        "cell (row {r}, col {c}) is a bow-tie: two opposite sides cross"
                    ));
                }
                let a2 = shoelace2(&q);
                if a2 > 0.0 {
                    pos += 1;
                } else if a2 < 0.0 {
                    neg += 1;
                }
            }
        }
        if pos > 0 && neg > 0 {
            return Err(format!(
                "cells have both orientations ({pos} one way, {neg} the other): the mesh folds over itself"
            ));
        }
        mesh.sign = if neg > 0 { -1.0 } else { 1.0 };

        for r in 0..nrow {
            for c in 0..ncol {
                let q = mesh.corners(r, c);
                let cell = split(&q, mesh.sign);
                mesh.cells.push(cell);
            }
        }
        Ok(mesh)
    }

    /// A mesh from cell centres (`nrow * ncol` row-major values each):
    /// each interior corner is the mean of the four centres around it, and
    /// the outer corners lie half a cell beyond the outermost centres
    /// (centres extrapolated linearly by one cell first). This is the cell
    /// rwarp's geolocation transformer uses for cell edges with
    /// `GeolocConvention::PixelCenter`.
    pub fn from_centres(
        ncol: usize,
        nrow: usize,
        x: &[f64],
        y: &[f64],
        map: CellMap,
    ) -> Result<Self, String> {
        if ncol < 2 || nrow < 2 {
            return Err(format!(
                "centres must be at least 2 x 2, got {ncol} x {nrow}"
            ));
        }
        if x.len() != ncol * nrow || y.len() != ncol * nrow {
            return Err(format!(
                "centre arrays have {} and {} values, expected {}",
                x.len(),
                y.len(),
                ncol * nrow
            ));
        }
        let corners = |a: &[f64]| -> Vec<f64> {
            // pad by linear extrapolation: (nrow + 2) x (ncol + 2)
            let (pc, pr) = (ncol + 2, nrow + 2);
            let mut p = vec![0.0; pc * pr];
            let at = |r: usize, c: usize| a[r * ncol + c];
            for r in 0..nrow {
                for c in 0..ncol {
                    p[(r + 1) * pc + c + 1] = at(r, c);
                }
                p[(r + 1) * pc] = 2.0 * at(r, 0) - at(r, 1);
                p[(r + 1) * pc + ncol + 1] = 2.0 * at(r, ncol - 1) - at(r, ncol - 2);
            }
            for c in 0..pc {
                p[c] = 2.0 * p[pc + c] - p[2 * pc + c];
                p[(pr - 1) * pc + c] = 2.0 * p[(pr - 2) * pc + c] - p[(pr - 3) * pc + c];
            }
            let mut out = Vec::with_capacity((ncol + 1) * (nrow + 1));
            for j in 0..=nrow {
                for i in 0..=ncol {
                    out.push(
                        0.25 * (p[j * pc + i]
                            + p[j * pc + i + 1]
                            + p[(j + 1) * pc + i]
                            + p[(j + 1) * pc + i + 1]),
                    );
                }
            }
            out
        };
        Self::from_corners(ncol, nrow, corners(x), corners(y), map)
    }

    /// Number of columns of cells.
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Number of rows of cells.
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    /// The cell map in use.
    pub fn cell_map(&self) -> CellMap {
        self.map
    }

    /// The four corners of cell `(row, col)` in world coordinates, in the
    /// order `P00, P10, P11, P01` (unit-square corners `(0, 0), (1, 0),
    /// (1, 1), (0, 1)`).
    pub fn corners(&self, row: usize, col: usize) -> [[f64; 2]; 4] {
        let w = self.ncol + 1;
        let k = |j: usize, i: usize| [self.x[j * w + i], self.y[j * w + i]];
        [
            k(row, col),
            k(row, col + 1),
            k(row + 1, col + 1),
            k(row + 1, col),
        ]
    }

    /// The world area of cell `(row, col)`.
    pub fn cell_area(&self, row: usize, col: usize) -> f64 {
        0.5 * self.cells[row * self.ncol + col].area2.abs()
    }

    /// The equal-area split of cell `(row, col)`: the world centre point
    /// and the unit-square point it maps to.
    pub fn split_centre(&self, row: usize, col: usize) -> ([f64; 2], [f64; 2]) {
        let c = &self.cells[row * self.ncol + col];
        (c.centre, c.unit)
    }

    /// The cells whose equal-area split is not exact: non-convex cells
    /// whose balanced split line misses the cell's kernel. Their
    /// index-space fractions are close to, not equal to, world fractions.
    pub fn inexact_cells(&self) -> Vec<(usize, usize)> {
        (0..self.cells.len())
            .filter(|&k| !self.cells[k].exact && self.cells[k].area2 != 0.0)
            .map(|k| (k / self.ncol, k % self.ncol))
            .collect()
    }

    /// Whether cell `(row, col)` owns the world point `p` (see the module
    /// docs for the rule on shared sides).
    pub fn contains(&self, row: usize, col: usize, p: [f64; 2]) -> bool {
        self.triangle(row, col, p).is_some()
    }

    /// The cell that owns `p`, starting the search at `hint` when given.
    ///
    /// With a hint, the search walks cell to cell towards `p` (O(1) when
    /// `p` is near the hint, as consecutive points of a ring or a
    /// refinement midpoint are); otherwise, or when the walk stalls, a
    /// bucket index over the cells' bounding boxes (built on first use)
    /// finds it.
    pub fn locate(&self, p: [f64; 2], hint: Option<(usize, usize)>) -> Option<(usize, usize)> {
        if !(p[0].is_finite() && p[1].is_finite()) {
            return None;
        }
        if let Some((mut r, mut c)) = hint {
            if r < self.nrow && c < self.ncol {
                for _ in 0..(2 * (self.ncol + self.nrow) + 4) {
                    let Some((dr, dc)) = self.step(r, c, p) else {
                        return Some((r, c));
                    };
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr < 0
                        || nc < 0
                        || nr >= self.nrow as i64
                        || nc >= self.ncol as i64
                        || (dr == 0 && dc == 0)
                    {
                        break;
                    }
                    (r, c) = (nr as usize, nc as usize);
                }
            }
        }
        self.buckets()
            .candidates(p)
            .find(|&k| self.contains(k / self.ncol, k % self.ncol, p))
            .map(|k| (k / self.ncol, k % self.ncol))
    }

    /// Every cell that owns `p`. For a valid mesh this is at most one;
    /// the method exists to check that.
    pub fn locate_all(&self, p: [f64; 2]) -> Vec<(usize, usize)> {
        if !(p[0].is_finite() && p[1].is_finite()) {
            return Vec::new();
        }
        self.buckets()
            .candidates(p)
            .filter(|&k| self.contains(k / self.ncol, k % self.ncol, p))
            .map(|k| (k / self.ncol, k % self.ncol))
            .collect()
    }

    /// World point to index space `[col, row]` (fractional), or `None`
    /// outside the mesh.
    pub fn to_index(&self, p: [f64; 2], hint: Option<(usize, usize)>) -> Option<[f64; 2]> {
        let (r, c) = self.locate(p, hint)?;
        let [u, v] = self.local(r, c, p)?;
        Some([c as f64 + u, r as f64 + v])
    }

    /// Index space `[col, row]` to world, for `0 <= col <= ncol` and
    /// `0 <= row <= nrow`; `None` outside.
    pub fn to_world(&self, idx: [f64; 2]) -> Option<[f64; 2]> {
        let [ci, ri] = idx;
        if !(ci >= 0.0 && ri >= 0.0 && ci <= self.ncol as f64 && ri <= self.nrow as f64) {
            return None;
        }
        let c = (ci.floor() as usize).min(self.ncol - 1);
        let r = (ri.floor() as usize).min(self.nrow - 1);
        Some(self.world(r, c, [ci - c as f64, ri - r as f64]))
    }

    /// The cell-local unit-square position `[u, v]` of world point `p` in
    /// cell `(row, col)`, through the cell map. `None` if the cell has no
    /// area or does not own `p`.
    pub fn local(&self, row: usize, col: usize, p: [f64; 2]) -> Option<[f64; 2]> {
        let k = self.triangle(row, col, p)?;
        let q = self.corners(row, col);
        let cell = &self.cells[row * self.ncol + col];
        match self.map {
            CellMap::EqualArea => {
                let (a, b, x) = (q[k], q[(k + 1) % 4], cell.centre);
                let (ua, ub, ux) = (UNIT[k], UNIT[(k + 1) % 4], cell.unit);
                // barycentric weights of p in (x, a, b)
                let d = orient(x, a, b);
                let wa = orient(b, x, p) / d;
                let wb = orient(x, a, p) / d;
                let wx = 1.0 - wa - wb;
                Some([
                    wx * ux[0] + wa * ua[0] + wb * ub[0],
                    wx * ux[1] + wa * ua[1] + wb * ub[1],
                ])
            }
            _ => {
                // GDAL's corner order: 0 = (i, j), 1 = (i, j + 1), 2 = (i + 1, j), 3 = (i + 1, j + 1)
                let (mut i, mut j) = (0.0, 0.0);
                inverse_bilinear(
                    p[0], p[1], q[0][0], q[0][1], q[3][0], q[3][1], q[1][0], q[1][1], q[2][0],
                    q[2][1], &mut i, &mut j,
                );
                Some([i, j])
            }
        }
    }

    /// World position of the cell-local unit-square point `[u, v]` of
    /// cell `(row, col)`, through the cell map.
    pub fn world(&self, row: usize, col: usize, uv: [f64; 2]) -> [f64; 2] {
        let q = self.corners(row, col);
        let [u, v] = uv;
        match self.map {
            CellMap::EqualArea => {
                let cell = &self.cells[row * self.ncol + col];
                let ux = cell.unit;
                // the unit triangle holding (u, v): by the spokes from ux
                let k = unit_triangle(ux, uv);
                let (ua, ub) = (UNIT[k], UNIT[(k + 1) % 4]);
                let d = orient(ux, ua, ub);
                let wa = orient(ub, ux, uv) / d;
                let wb = orient(ux, ua, uv) / d;
                let wx = 1.0 - wa - wb;
                let (a, b, x) = (q[k], q[(k + 1) % 4], cell.centre);
                [
                    wx * x[0] + wa * a[0] + wb * b[0],
                    wx * x[1] + wa * a[1] + wb * b[1],
                ]
            }
            _ => {
                let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                [
                    w[0] * q[0][0] + w[1] * q[1][0] + w[2] * q[2][0] + w[3] * q[3][0],
                    w[0] * q[0][1] + w[1] * q[1][1] + w[2] * q[2][1] + w[3] * q[3][1],
                ]
            }
        }
    }

    /// The orientation test of side `k` (0: top, 1: right, 2: bottom, 3:
    /// left, in index space) of cell `(row, col)` against `p`, signed so
    /// that positive is inside. Computed in the side's canonical direction
    /// (from the corner with the smaller storage index), so neighbours get
    /// the same value with opposite sign.
    fn side(&self, row: usize, col: usize, k: usize, p: [f64; 2]) -> f64 {
        let w = self.ncol + 1;
        let at = |j: usize, i: usize| [self.x[j * w + i], self.y[j * w + i]];
        // canonical: left to right along a row line, top to bottom along a column line
        let (a, b, flip) = match k {
            0 => (at(row, col), at(row, col + 1), 1.0),
            1 => (at(row, col + 1), at(row + 1, col + 1), 1.0),
            2 => (at(row + 1, col), at(row + 1, col + 1), -1.0),
            _ => (at(row, col), at(row + 1, col), -1.0),
        };
        self.sign * flip * orient(a, b, p)
    }

    /// The fan triangle (0 to 3, by side) of cell `(row, col)` that owns
    /// `p`, if the cell does.
    fn triangle(&self, row: usize, col: usize, p: [f64; 2]) -> Option<usize> {
        let cell = &self.cells[row * self.ncol + col];
        if cell.area2 == 0.0 {
            return None;
        }
        let q = self.corners(row, col);
        let x = cell.centre;
        if p == x {
            return Some(0);
        }
        // the centre is in the cell's kernel, so the rays from it to the
        // corners cut the plane into four wedges, each less than a half
        // turn, one per side; wedge k lies between rays k and k + 1 and owns
        // ray k. Only the wedge's own side is then tested (in a non-convex
        // cell the other sides' half-planes do not contain the cell), and
        // for a point on ray k also side k - 1, so a corner is owned only
        // when both its sides allow it. Low-index sides (top, left) are
        // inclusive.
        let spoke: [f64; 4] = std::array::from_fn(|k| self.sign * orient(x, q[k], p));
        let on_ray = |k: usize| {
            spoke[k] == 0.0
                && (p[0] - x[0]) * (q[k][0] - x[0]) + (p[1] - x[1]) * (q[k][1] - x[1]) > 0.0
        };
        let k = (0..4).find(|&k| (spoke[k] > 0.0 || on_ray(k)) && spoke[(k + 1) % 4] < 0.0)?;
        let allowed = |k: usize| {
            let s = self.side(row, col, k, p);
            s > 0.0 || (s == 0.0 && (k == 0 || k == 3))
        };
        if allowed(k) && (!on_ray(k) || allowed((k + 3) % 4)) {
            Some(k)
        } else {
            None
        }
    }

    /// Which way to walk from cell `(row, col)` towards `p`: `None` when
    /// the cell owns it.
    fn step(&self, row: usize, col: usize, p: [f64; 2]) -> Option<(i64, i64)> {
        if self.contains(row, col, p) {
            return None;
        }
        let s: [f64; 4] = std::array::from_fn(|k| self.side(row, col, k, p));
        let dr = if s[0] < 0.0 {
            -1
        } else if s[2] <= 0.0 {
            1
        } else {
            0
        };
        let dc = if s[3] < 0.0 {
            -1
        } else if s[1] <= 0.0 {
            1
        } else {
            0
        };
        Some((dr, dc))
    }

    fn buckets(&self) -> &Buckets {
        self.buckets.get_or_init(|| Buckets::new(self))
    }
}

/// Twice the signed area of a quadrilateral.
fn shoelace2(q: &[[f64; 2]; 4]) -> f64 {
    let mut s = 0.0;
    for k in 0..4 {
        let (a, b) = (q[k], q[(k + 1) % 4]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    s
}

/// Whether two opposite sides of the quadrilateral cross properly.
fn bow_tie(q: &[[f64; 2]; 4]) -> bool {
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]| {
        let (d1, d2) = (orient(a, b, c), orient(a, b, d));
        let (d3, d4) = (orient(c, d, a), orient(c, d, b));
        d1 * d2 < 0.0 && d3 * d4 < 0.0
    };
    cross(q[0], q[1], q[2], q[3]) || cross(q[1], q[2], q[3], q[0])
}

/// The equal-area split of one cell.
///
/// With `T_k(X)` the signed area of side `k`'s triangle to the centre
/// `X`, the split is balanced when `T_0 + T_2 = A / 2` (then `T_1 + T_3 =
/// A / 2` too, since the four always sum to `A`), and the unit centre is
/// `u = 2 T_3 / A`, `v = 2 T_0 / A`. The corner mean is balanced. A convex
/// cell contains it, so all four triangles are proper. A non-convex cell
/// may not: then the point of the balanced line (through the corner mean)
/// inside the cell's kernel that keeps the four triangles furthest from
/// degenerate is used, and failing that the best kernel point found,
/// flagged inexact.
fn split(q: &[[f64; 2]; 4], sign: f64) -> Cell {
    let area2 = shoelace2(q);
    let mean = [
        0.25 * (q[0][0] + q[1][0] + q[2][0] + q[3][0]),
        0.25 * (q[0][1] + q[1][1] + q[2][1] + q[3][1]),
    ];
    if area2 == 0.0 {
        return Cell {
            area2,
            centre: mean,
            unit: [0.5, 0.5],
            exact: true,
        };
    }
    // signed (twice) areas of the four side triangles, scaled to be
    // positive inside
    let t = |x: [f64; 2], k: usize| sign * orient(q[k], q[(k + 1) % 4], x);
    // t is twice a triangle's area and a twice the cell's, so the share
    // T / A is t / a, and the unit triangle on side 0 (area v / 2) holds
    // that share when v = 2 t / a
    let unit = |x: [f64; 2]| {
        let a = sign * area2;
        [2.0 * t(x, 3) / a, 2.0 * t(x, 0) / a]
    };
    let margin = 1e-12 * (sign * area2);
    let proper = |x: [f64; 2]| (0..4).all(|k| t(x, k) > margin);
    if proper(mean) {
        return Cell {
            area2,
            centre: mean,
            unit: unit(mean),
            exact: true,
        };
    }
    // the balanced line: g(X) = T0(X) + T2(X) - A/2 = 0 is affine in X; its
    // gradient is the sum of the two sides' normals, its direction the
    // perpendicular
    let grad = |k: usize| {
        let (a, b) = (q[k], q[(k + 1) % 4]);
        [-(b[1] - a[1]) * sign, (b[0] - a[0]) * sign]
    };
    let (g0, g2) = (grad(0), grad(2));
    let dir = [-(g0[1] + g2[1]), g0[0] + g2[0]];
    // the interval of t with every T_k(mean + t dir) > margin
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..4 {
        let a0 = t(mean, k) - margin;
        let g = grad(k);
        let b = g[0] * dir[0] + g[1] * dir[1];
        if b == 0.0 {
            if a0 <= 0.0 {
                lo = f64::INFINITY;
            }
        } else if b > 0.0 {
            lo = lo.max(-a0 / b);
        } else {
            hi = hi.min(-a0 / b);
        }
    }
    if lo < hi && lo.is_finite() && hi.is_finite() {
        // the balanced point that keeps the four triangles furthest from
        // degenerate: min_k T_k is concave along the line, so a ternary
        // search finds its maximum
        let worst = |tt: f64| {
            let x = [mean[0] + tt * dir[0], mean[1] + tt * dir[1]];
            (0..4).map(|k| t(x, k)).fold(f64::INFINITY, f64::min)
        };
        let (mut a, mut b) = (lo, hi);
        for _ in 0..200 {
            let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
            if worst(m1) < worst(m2) {
                a = m1;
            } else {
                b = m2;
            }
        }
        let tt = 0.5 * (a + b);
        let x = [mean[0] + tt * dir[0], mean[1] + tt * dir[1]];
        if proper(x) {
            return Cell {
                area2,
                centre: x,
                unit: unit(x),
                exact: true,
            };
        }
    }
    // no balanced kernel point: the best kernel point among corner
    // combinations, flagged inexact
    let mut best = (f64::NEG_INFINITY, mean);
    let n = 12;
    for i in 0..=n {
        for j in 0..=(n - i) {
            for k in 0..=(n - i - j) {
                let l = n - i - j - k;
                let w = [i as f64, j as f64, k as f64, l as f64].map(|v| v / n as f64);
                let x = [
                    w[0] * q[0][0] + w[1] * q[1][0] + w[2] * q[2][0] + w[3] * q[3][0],
                    w[0] * q[0][1] + w[1] * q[1][1] + w[2] * q[2][1] + w[3] * q[3][1],
                ];
                let m = (0..4).map(|k| t(x, k)).fold(f64::INFINITY, f64::min);
                if m > best.0 {
                    best = (m, x);
                }
            }
        }
    }
    Cell {
        area2,
        centre: best.1,
        unit: unit(best.1),
        exact: false,
    }
}

/// The unit-square triangle (by side) that holds `uv`, for the split at
/// unit centre `ux`; points outside the square go to the nearest one.
fn unit_triangle(ux: [f64; 2], uv: [f64; 2]) -> usize {
    let s: [f64; 4] = std::array::from_fn(|k| orient(ux, UNIT[k], uv));
    (0..4)
        .find(|&k| s[k] >= 0.0 && s[(k + 1) % 4] < 0.0)
        .unwrap_or(0)
}

/// A uniform bucket grid over the cells' bounding boxes, compressed
/// (offsets plus one array of cell indices, ascending within a bucket).
#[derive(Debug, Clone)]
struct Buckets {
    x0: f64,
    y0: f64,
    w: f64,
    h: f64,
    nx: usize,
    ny: usize,
    offsets: Vec<u32>,
    cells: Vec<u32>,
}

impl Buckets {
    fn new(mesh: &QuadMesh) -> Self {
        let n = mesh.ncol * mesh.nrow;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let boxes: Vec<[f64; 4]> = (0..n)
            .map(|k| {
                let q = mesh.corners(k / mesh.ncol, k % mesh.ncol);
                let b = q
                    .iter()
                    .fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| {
                        [
                            b[0].min(p[0]),
                            b[1].min(p[1]),
                            b[2].max(p[0]),
                            b[3].max(p[1]),
                        ]
                    });
                x0 = x0.min(b[0]);
                y0 = y0.min(b[1]);
                x1 = x1.max(b[2]);
                y1 = y1.max(b[3]);
                b
            })
            .collect();
        let side = ((n as f64).sqrt().ceil() as usize).clamp(1, 4096);
        let (nx, ny) = (side, side);
        let w = ((x1 - x0) / nx as f64).max(f64::MIN_POSITIVE);
        let h = ((y1 - y0) / ny as f64).max(f64::MIN_POSITIVE);
        let mut b = Buckets {
            x0,
            y0,
            w,
            h,
            nx,
            ny,
            offsets: vec![0; nx * ny + 1],
            cells: Vec::new(),
        };
        let range = |b: &Buckets, bx: &[f64; 4]| {
            let (c0, r0) = b.bucket(bx[0], bx[1]);
            let (c1, r1) = b.bucket(bx[2], bx[3]);
            (c0, r0, c1, r1)
        };
        for bx in &boxes {
            let (c0, r0, c1, r1) = range(&b, bx);
            for r in r0..=r1 {
                for c in c0..=c1 {
                    b.offsets[r * nx + c + 1] += 1;
                }
            }
        }
        for i in 0..nx * ny {
            b.offsets[i + 1] += b.offsets[i];
        }
        let mut fill = b.offsets.clone();
        b.cells = vec![0; *b.offsets.last().unwrap() as usize];
        for (k, bx) in boxes.iter().enumerate() {
            let (c0, r0, c1, r1) = range(&b, bx);
            for r in r0..=r1 {
                for c in c0..=c1 {
                    let slot = &mut fill[r * nx + c];
                    b.cells[*slot as usize] = k as u32;
                    *slot += 1;
                }
            }
        }
        b
    }

    fn bucket(&self, x: f64, y: f64) -> (usize, usize) {
        let c = ((x - self.x0) / self.w)
            .floor()
            .clamp(0.0, (self.nx - 1) as f64) as usize;
        let r = ((y - self.y0) / self.h)
            .floor()
            .clamp(0.0, (self.ny - 1) as f64) as usize;
        (c, r)
    }

    /// Cells whose box may hold `p`, in ascending (row-major) order.
    fn candidates(&self, p: [f64; 2]) -> impl Iterator<Item = usize> + '_ {
        let inside = p[0] >= self.x0
            && p[1] >= self.y0
            && p[0] <= self.x0 + self.w * self.nx as f64
            && p[1] <= self.y0 + self.h * self.ny as f64;
        let (c, r) = self.bucket(p[0], p[1]);
        let k = r * self.nx + c;
        let s = if inside {
            &self.cells[self.offsets[k] as usize..self.offsets[k + 1] as usize]
        } else {
            &[][..]
        };
        s.iter().map(|&k| k as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_mesh(map: CellMap) -> QuadMesh {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for j in 0..=2 {
            for i in 0..=3 {
                x.push(i as f64);
                y.push(-(j as f64));
            }
        }
        QuadMesh::from_corners(3, 2, x, y, map).unwrap()
    }

    #[test]
    fn a_regular_lattice_maps_to_itself() {
        for map in [CellMap::Bilinear, CellMap::EqualArea] {
            let m = square_mesh(map);
            let idx = m.to_index([1.25, -0.5], None).unwrap();
            assert!(
                (idx[0] - 1.25).abs() < 1e-15 && (idx[1] - 0.5).abs() < 1e-15,
                "{map:?} {idx:?}"
            );
            assert_eq!(m.cell_area(1, 2), 1.0);
        }
    }

    #[test]
    fn shared_walls_belong_to_the_low_index_side() {
        let m = square_mesh(CellMap::EqualArea);
        // the wall between columns 0 and 1, and between rows 0 and 1
        assert_eq!(m.locate([1.0, -0.5], None), Some((0, 1)));
        assert_eq!(m.locate([0.5, -1.0], None), Some((1, 0)));
        assert_eq!(m.locate([1.0, -1.0], None), Some((1, 1)));
        // the far walls are outside
        assert_eq!(m.locate([3.0, -0.5], None), None);
        assert_eq!(m.locate([0.5, -2.0], None), None);
        assert_eq!(m.locate([0.0, 0.0], None), Some((0, 0)));
    }

    #[test]
    fn rejects_bow_ties_and_folds() {
        // corners (row line 1) given in the wrong order: a bow-tie
        let bow = QuadMesh::from_corners(
            1,
            1,
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            CellMap::EqualArea,
        );
        assert!(bow.unwrap_err().contains("bow-tie"));
        // two cells, the second mirrored back over the first
        let fold = QuadMesh::from_corners(
            2,
            1,
            vec![0.0, 1.0, 0.5, 0.0, 1.0, 0.5],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            CellMap::EqualArea,
        );
        assert!(fold.unwrap_err().contains("fold"));
    }

    #[test]
    fn a_non_convex_cell_gets_a_balanced_kernel_point() {
        // an arrowhead: the corner mean lies outside the cell's kernel
        let q = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [3.0, 0.5]];
        let m = QuadMesh::from_corners(
            1,
            1,
            vec![q[0][0], q[1][0], q[3][0], q[2][0]],
            vec![q[0][1], q[1][1], q[3][1], q[2][1]],
            CellMap::EqualArea,
        )
        .unwrap();
        let (x, u) = m.split_centre(0, 0);
        let qq = m.corners(0, 0);
        let a = shoelace2(&qq);
        let t: Vec<f64> = (0..4)
            .map(|k| orient(qq[k], qq[(k + 1) % 4], x) / a)
            .collect();
        assert!(t.iter().all(|&v| v > 0.0), "{t:?}");
        assert!(m.inexact_cells().is_empty());
        assert!((t[0] + t[2] - 0.5).abs() < 1e-12, "{t:?}");
        assert!((u[1] - 2.0 * t[0]).abs() < 1e-15 && (u[0] - 2.0 * t[3]).abs() < 1e-15);
        // the corner mean was not usable
        let mean = [2.75, 1.125];
        assert!((0..4).any(|k| orient(qq[k], qq[(k + 1) % 4], mean) / a <= 0.0));
    }
}
