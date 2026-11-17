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

use super::walk::{burn_with, cover_rings, turn_shifts};
use super::Period;

/// A rectilinear grid by its edges: `x` has `ncol + 1` values, `y` has
/// `nrow + 1`, row 0 between `y[0]` and `y[1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectilinear {
    x: Vec<f64>,
    y: Vec<f64>,
    /// A declared period of x ([`Rectilinear::with_period`]).
    period: Option<Period>,
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
        Ok(Rectilinear { x, y, period: None })
    }

    /// The grid with x periodic with `period` (360 for longitude in
    /// degrees), as [`QuadMesh::with_period`](super::QuadMesh::with_period)
    /// says: polygons are burned at each whole period at which they
    /// overlap the grid, and [`Rectilinear::cell`] finds a point at any
    /// whole period. [`Rectilinear::to_index`] and
    /// [`Rectilinear::ring_to_index`] take points where they are.
    ///
    /// For a global grid whose column edges run once round, such as a
    /// Gaussian grid's from 0 to 360, with polygons from -180 to 180.
    ///
    /// Errors when `period` isn't finite and positive, when a column is
    /// half a period wide or more, or when the grid is wider than the
    /// period by half a column or more: it would go round more than once.
    pub fn with_period(mut self, period: f64) -> Result<Self, String> {
        let [lo, hi] = super::period::range_of(&self.x);
        let p = Period::new(period, [lo, hi])?;
        if let Some(c) = self.x.windows(2).position(|w| (w[1] - w[0]).abs() >= period / 2.0) {
            return Err(format!("column {c} is half the period ({period}) wide or more"));
        }
        let centres = self.x.windows(2).map(|w| 0.5 * (w[0] + w[1]));
        let [c0, c1] = super::period::range_of(&centres.collect::<Vec<_>>());
        if c0 + period < hi || c1 - period > lo {
            return Err(format!(
                "the columns span {lo} to {hi}, more than a period ({period}): the grid would go round more than once"
            ));
        }
        self.period = Some(p);
        Ok(self)
    }

    /// The declared period of x ([`Rectilinear::with_period`]).
    pub fn period(&self) -> Option<f64> {
        self.period.map(|p| p.period)
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

    /// The cell holding `p`, if inside the grid (low walls inclusive);
    /// with a declared period, at any whole period.
    pub fn cell(&self, p: [f64; 2]) -> Option<(usize, usize)> {
        let at = |p: [f64; 2]| {
            let [c, r] = self.to_index(p);
            (c >= 0.0 && r >= 0.0 && c < self.ncol() as f64 && r < self.nrow() as f64)
                .then_some((r as usize, c as usize))
        };
        match self.period {
            None => at(p),
            Some(per) => per.point_shifts(p[0]).find_map(|s| at([p[0] + s, p[1]])),
        }
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
            self.period,
            &|r| Ok(self.ring_to_index(r)),
            &|p, _| self.cell(p),
        )
    }

    /// Exact coverage of one small world ring through a reused
    /// [`CoverageWindow`], as [`QuadMesh::cover_ring`](super::QuadMesh::cover_ring).
    /// The window may extend beyond the grid; cells outside it are the
    /// caller's to ignore. With a declared period the ring is taken at
    /// each whole period at which it overlaps the grid, and the window
    /// spans all of them: across the seam, the grid's full width and a
    /// column or so beyond each end.
    ///
    /// False, with an empty window, when the ring has a coordinate that
    /// isn't finite or, with a period, is at no whole period over the
    /// grid or would be taken at more than 16.
    pub fn cover_ring(&self, ring: &[[f64; 2]], win: &mut CoverageWindow) -> bool {
        let Ok(shifts) = turn_shifts(self.period, ring.iter().map(|p| p[0])) else {
            win.reset(0, 0, 0, 0);
            return false;
        };
        let mut rings = Vec::with_capacity(shifts.len());
        for s in shifts {
            let ir = if s == 0.0 {
                self.ring_to_index(ring)
            } else {
                let moved: Vec<[f64; 2]> = ring.iter().map(|p| [p[0] + s, p[1]]).collect();
                self.ring_to_index(&moved)
            };
            if ir.len() >= 3 {
                rings.push(ir);
            }
        }
        cover_rings(&rings, win)
    }
}
