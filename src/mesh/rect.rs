//! The rectilinear front end: a grid given by its column and row edges
//! (each strictly monotone, in either direction), as for CF coordinate
//! bounds or a regular longitude and latitude grid stored south up.
//!
//! The map to index space is piecewise linear on each axis (extended
//! linearly beyond the first and last edges), so it is affine inside each
//! cell. Cutting every ring edge at every wall it crosses then makes the
//! index-space ring exact: controlledburn's fractions are fractions of
//! each cell's world area. Parts outside the grid map outside it, where
//! controlledburn ignores them, so no boundary closure is needed.

use controlledburn::{BurnOptions, BurnResult, CoverageWindow, Geometry};

use super::walk::burn_with;

/// A rectilinear grid by its edges: `x` has `ncol + 1` values, `y` has
/// `nrow + 1`, row 0 between `y[0]` and `y[1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectilinear {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn monotone(v: &[f64]) -> bool {
    v.len() >= 2
        && v.iter().all(|a| a.is_finite())
        && (v.windows(2).all(|w| w[1] > w[0]) || v.windows(2).all(|w| w[1] < w[0]))
}

/// Fractional index of `v` along monotone `edges`, extended linearly
/// beyond the ends.
fn axis(edges: &[f64], v: f64) -> f64 {
    let n = edges.len() - 1;
    let up = edges[n] > edges[0];
    // first edge index i with v before edges[i + 1], in the axis direction
    let (mut lo, mut hi) = (0usize, n - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let past = if up {
            v >= edges[mid + 1]
        } else {
            v <= edges[mid + 1]
        };
        if past {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let (a, b) = (edges[lo], edges[lo + 1]);
    lo as f64 + (v - a) / (b - a)
}

impl Rectilinear {
    /// A grid from its edges. Errors unless each axis has at least two
    /// finite, strictly monotone values.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, String> {
        if !monotone(&x) || !monotone(&y) {
            return Err(
                "rectilinear edges must be finite and strictly monotone, two or more per axis"
                    .into(),
            );
        }
        Ok(Rectilinear { x, y })
    }

    pub fn ncol(&self) -> usize {
        self.x.len() - 1
    }

    pub fn nrow(&self) -> usize {
        self.y.len() - 1
    }

    /// World point to index space `[col, row]` (extended beyond the grid).
    pub fn to_index(&self, p: [f64; 2]) -> [f64; 2] {
        [axis(&self.x, p[0]), axis(&self.y, p[1])]
    }

    /// The cell holding `p`, if inside the grid (low walls inclusive).
    pub fn cell(&self, p: [f64; 2]) -> Option<(usize, usize)> {
        let [c, r] = self.to_index(p);
        (c >= 0.0 && r >= 0.0 && c < self.ncol() as f64 && r < self.nrow() as f64)
            .then_some((r as usize, c as usize))
    }

    /// A world ring (closed implicitly) in index space, with a vertex at
    /// every wall it crosses.
    pub fn ring_to_index(&self, ring: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut pts: Vec<[f64; 2]> = ring.to_vec();
        while pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        if pts.len() < 3 || pts.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
            return Vec::new();
        }
        let n = pts.len();
        let mut out = Vec::with_capacity(4 * n);
        let mut cuts: Vec<(f64, [f64; 2])> = Vec::new();
        for k in 0..n {
            let (a, b) = (pts[k], pts[(k + 1) % n]);
            out.push(self.to_index(a));
            cuts.clear();
            let (ia, ib) = (axis(&self.x, a[0]), axis(&self.x, b[0]));
            if a[0] != b[0] {
                let (lo, hi) = (ia.min(ib), ia.max(ib));
                let first = (lo.floor() as i64 + 1).max(0);
                let last = (hi.ceil() as i64 - 1).min(self.ncol() as i64);
                for i in first..=last {
                    let t = (self.x[i as usize] - a[0]) / (b[0] - a[0]);
                    if t > 0.0 && t < 1.0 {
                        let y = a[1] + t * (b[1] - a[1]);
                        cuts.push((t, [i as f64, axis(&self.y, y)]));
                    }
                }
            }
            let (ja, jb) = (axis(&self.y, a[1]), axis(&self.y, b[1]));
            if a[1] != b[1] {
                let (lo, hi) = (ja.min(jb), ja.max(jb));
                let first = (lo.floor() as i64 + 1).max(0);
                let last = (hi.ceil() as i64 - 1).min(self.nrow() as i64);
                for j in first..=last {
                    let t = (self.y[j as usize] - a[1]) / (b[1] - a[1]);
                    if t > 0.0 && t < 1.0 {
                        let x = a[0] + t * (b[0] - a[0]);
                        cuts.push((t, [axis(&self.x, x), j as f64]));
                    }
                }
            }
            cuts.sort_by(|p, q| p.0.total_cmp(&q.0));
            out.extend(cuts.iter().map(|c| c.1));
        }
        out
    }

    /// Burn world geometries onto the grid, as [`QuadMesh::burn`](super::QuadMesh::burn).
    pub fn burn(
        &self,
        geoms: &[Geometry],
        opts: impl Into<BurnOptions>,
    ) -> Result<BurnResult, String> {
        self.burn_par(geoms, opts, 1)
    }

    /// [`Rectilinear::burn`] on up to `threads` threads (0 for all
    /// available), as [`QuadMesh::burn_par`](super::QuadMesh::burn_par).
    pub fn burn_par(
        &self,
        geoms: &[Geometry],
        opts: impl Into<BurnOptions>,
        threads: usize,
    ) -> Result<BurnResult, String> {
        burn_with(
            self.ncol(),
            self.nrow(),
            geoms,
            opts.into(),
            threads,
            &|r| Ok(self.ring_to_index(r)),
            &|p, _| self.cell(p),
        )
    }

    /// Exact coverage of one small world ring through a reused
    /// [`CoverageWindow`], as [`QuadMesh::cover_ring`](super::QuadMesh::cover_ring).
    /// The window may extend beyond the grid; cells outside it are the
    /// caller's to ignore.
    pub fn cover_ring(&self, ring: &[[f64; 2]], win: &mut CoverageWindow) -> bool {
        let ir = self.ring_to_index(ring);
        if ir.len() < 3 {
            win.reset(0, 0, 0, 0);
            return false;
        }
        let xs: Vec<f64> = ir.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = ir.iter().map(|p| p[1]).collect();
        if !win.reset_to_ring(&xs, &ys) {
            return false;
        }
        win.add_ring(&xs, &ys, 1.0);
        win.resolve();
        true
    }
}
