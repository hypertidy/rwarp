//! Rings onto the mesh: the walk of the curvilinear front end (stage 2).
//!
//! A world ring becomes an index-space ring in three steps (see
//! `design/curvilinear-frontend.md`, section 2):
//!
//! 1. Each segment is cut at every cell side and spoke it crosses. The
//!    cells it passes are found by a flood from the cell of its first
//!    point (or from the boundary sides it crosses), and each cell's four
//!    triangles clip it (Cyrus-Beck, so a vertex, corner or centre on the
//!    segment needs no special case).
//! 2. Each cut point gets its index-space position from the edge it lies
//!    on, at its parameter along the edge's canonical direction: exact
//!    along a side (the map is linear there) and along a spoke, and
//!    bitwise the same for every ring that crosses that edge at that point
//!    in either direction. Ring vertices use the cell's barycentric map.
//! 3. A part of the ring outside the mesh is replaced by the boundary arc
//!    from its exit to its re-entry, the arc for which the excursion and
//!    the arc together do not wind around the mesh. The arc lies on the
//!    outer walls, so it changes no cell.
//!
//! The walk needs [`CellMap::EqualArea`]: under `Bilinear` the preimage of
//! a straight edge is curved, and mapping cut points alone would not be
//! exact for either model.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use controlledburn::{
    BurnMode, BurnOptions, BurnResult, Coord, CoverageWindow, Geometry, GridPoint, GridSpec,
    Polygon,
};

use super::{orient, CellMap, QuadMesh, UNIT};

/// Gaps in segment parameter smaller than this join two pieces (the two
/// sides of a vertex the segment passes through can give `t` an ulp apart).
const T_EPS: f64 = 1e-12;

/// An edge of a cell's four-triangle split.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Edge {
    /// Side `k` (0 top, 1 right, 2 bottom, 3 left, in index space).
    Side { row: usize, col: usize, k: usize },
    /// The spoke from the split centre to corner `j`.
    Spoke { row: usize, col: usize, j: usize },
}

/// One piece of a segment inside one triangle.
#[derive(Debug, Clone, Copy)]
struct Piece {
    t0: f64,
    t1: f64,
    /// Index-space positions of the ends.
    p0: [f64; 2],
    p1: [f64; 2],
}

/// A point of the ring as walked: index-space position when it is in the
/// mesh (or on its boundary), and whether the stretch of ring leading to
/// it lies outside the mesh.
#[derive(Debug, Clone, Copy)]
struct Node {
    idx: Option<[f64; 2]>,
    world: [f64; 2],
    out_before: bool,
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// The winding number of a closed polygon about `p`.
fn winding(poly: &[[f64; 2]], p: [f64; 2]) -> i64 {
    let n = poly.len();
    let mut w = 0;
    for k in 0..n {
        let a = poly[k];
        let b = poly[(k + 1) % n];
        if a[1] <= p[1] {
            if b[1] > p[1] && orient(a, b, p) > 0.0 {
                w += 1;
            }
        } else if b[1] <= p[1] && orient(a, b, p) < 0.0 {
            w -= 1;
        }
    }
    w
}

/// Drop consecutive repeats (and a last point equal to the first).
fn dedup(mut v: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    v.dedup();
    while v.len() > 1 && v.first() == v.last() {
        v.pop();
    }
    v
}

/// Whether segments `a b` and `c d` intersect, ends included.
fn segments_meet(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let d1 = orient(c, d, a);
    let d2 = orient(c, d, b);
    let d3 = orient(a, b, c);
    let d4 = orient(a, b, d);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    let on = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        r[0] >= p[0].min(q[0])
            && r[0] <= p[0].max(q[0])
            && r[1] >= p[1].min(q[1])
            && r[1] <= p[1].max(q[1])
    };
    (d1 == 0.0 && on(c, d, a))
        || (d2 == 0.0 && on(c, d, b))
        || (d3 == 0.0 && on(a, b, c))
        || (d4 == 0.0 && on(a, b, d))
}

impl QuadMesh {
    /// The world corner at row line `j`, column line `i`.
    fn corner(&self, j: usize, i: usize) -> [f64; 2] {
        let w = self.ncol + 1;
        [self.x[j * w + i], self.y[j * w + i]]
    }

    /// An edge's canonical world end points (shared by the two triangles
    /// either side of it).
    fn edge_world(&self, e: Edge) -> ([f64; 2], [f64; 2]) {
        match e {
            Edge::Side { row, col, k } => match k {
                0 => (self.corner(row, col), self.corner(row, col + 1)),
                1 => (self.corner(row, col + 1), self.corner(row + 1, col + 1)),
                2 => (self.corner(row + 1, col), self.corner(row + 1, col + 1)),
                _ => (self.corner(row, col), self.corner(row + 1, col)),
            },
            Edge::Spoke { row, col, j } => {
                let q = self.corners(row, col);
                (self.cells[row * self.ncol + col].centre, q[j])
            }
        }
    }

    /// The index-space point at parameter `s` along an edge's canonical
    /// direction.
    fn edge_index(&self, e: Edge, s: f64) -> [f64; 2] {
        match e {
            Edge::Side { row, col, k } => {
                let (c, r) = (col as f64, row as f64);
                match k {
                    0 => [c + s, r],
                    1 => [c + 1.0, r + s],
                    2 => [c + s, r + 1.0],
                    _ => [c, r + s],
                }
            }
            Edge::Spoke { row, col, j } => {
                let ux = self.cells[row * self.ncol + col].unit;
                let uj = UNIT[j];
                [
                    col as f64 + (ux[0] + s * (uj[0] - ux[0])),
                    row as f64 + (ux[1] + s * (uj[1] - ux[1])),
                ]
            }
        }
    }

    /// The index-space point where segment `a b` crosses edge `e`. The
    /// parameter is computed from the edge's canonical ends, so it does not
    /// depend on the segment's direction.
    fn crossing(&self, e: Edge, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let (e0, e1) = self.edge_world(e);
        let oa = orient(a, b, e0);
        let ob = orient(a, b, e1);
        let s = if oa == ob {
            0.5
        } else {
            (oa / (oa - ob)).clamp(0.0, 1.0)
        };
        self.edge_index(e, s)
    }

    /// Index-space position of `p` through triangle `k` of cell
    /// `(row, col)`, without the ownership test (for points on the
    /// triangle's closure).
    fn tri_index(&self, row: usize, col: usize, k: usize, p: [f64; 2]) -> [f64; 2] {
        let q = self.corners(row, col);
        let cell = &self.cells[row * self.ncol + col];
        let (a, b, x) = (q[k], q[(k + 1) % 4], cell.centre);
        let (ua, ub, ux) = (UNIT[k], UNIT[(k + 1) % 4], cell.unit);
        let d = orient(x, a, b);
        let wa = orient(b, x, p) / d;
        let wb = orient(x, a, p) / d;
        let wx = 1.0 - wa - wb;
        [
            col as f64 + (wx * ux[0] + wa * ua[0] + wb * ub[0]),
            row as f64 + (wx * ux[1] + wa * ua[1] + wb * ub[1]),
        ]
    }

    /// Clip segment `a b` by triangle `k` of cell `(row, col)`. Returns the
    /// parameter interval (possibly empty, `t0 == t1`, for a touch) and the
    /// edges that bound it, `None` where the interval ends at `a` or `b`.
    #[allow(clippy::type_complexity)]
    fn clip(
        &self,
        row: usize,
        col: usize,
        k: usize,
        a: [f64; 2],
        b: [f64; 2],
    ) -> Option<(f64, Option<Edge>, f64, Option<Edge>)> {
        let q = self.corners(row, col);
        let x = self.cells[row * self.ncol + col].centre;
        let k1 = (k + 1) % 4;
        let edges = [
            (Edge::Spoke { row, col, j: k }, q[k1]),
            (Edge::Side { row, col, k }, x),
            (Edge::Spoke { row, col, j: k1 }, q[k]),
        ];
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let (mut e0, mut e1) = (None, None);
        for (e, opp) in edges {
            let (p0, p1) = self.edge_world(e);
            let sigma = orient(p0, p1, opp);
            if sigma == 0.0 {
                return None;
            }
            let sigma = sigma.signum();
            let fa = orient(p0, p1, a);
            let fb = orient(p0, p1, b);
            let (ga, gb) = (sigma * fa, sigma * fb);
            if ga < 0.0 && gb < 0.0 {
                return None;
            }
            if ga < 0.0 {
                let t = fa / (fa - fb);
                if t > t0 {
                    t0 = t;
                    e0 = Some(e);
                }
            } else if gb < 0.0 {
                let t = fa / (fa - fb);
                if t < t1 {
                    t1 = t;
                    e1 = Some(e);
                }
            }
        }
        (t0 <= t1).then_some((t0, e0, t1, e1))
    }

    /// Whether segment `a b` meets the bounding box of cell `(row, col)`.
    fn meets_box(&self, row: usize, col: usize, a: [f64; 2], b: [f64; 2]) -> bool {
        let q = self.corners(row, col);
        let (mut x0, mut x1, mut y0, mut y1) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for p in q {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        a[0].max(b[0]) >= x0 && a[0].min(b[0]) <= x1 && a[1].max(b[1]) >= y0 && a[1].min(b[1]) <= y1
    }

    /// Flood from `seeds` over the cells segment `a b` passes, collecting
    /// its pieces.
    #[allow(clippy::too_many_arguments)]
    fn flood(
        &self,
        a: [f64; 2],
        b: [f64; 2],
        ia: Option<[f64; 2]>,
        ib: Option<[f64; 2]>,
        seeds: &[(usize, usize)],
        seen: &mut HashSet<usize>,
        out: &mut Vec<Piece>,
    ) {
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for &s in seeds {
            if seen.insert(s.0 * self.ncol + s.1) {
                stack.push(s);
            }
        }
        while let Some((r, c)) = stack.pop() {
            let mut touched = false;
            if self.cells[r * self.ncol + c].area2 == 0.0 {
                touched = self.meets_box(r, c, a, b);
            } else {
                for k in 0..4 {
                    let Some((t0, e0, t1, e1)) = self.clip(r, c, k, a, b) else {
                        continue;
                    };
                    touched = true;
                    if t1 > t0 {
                        let p0 = match e0 {
                            Some(e) => self.crossing(e, a, b),
                            None => ia.unwrap_or_else(|| self.tri_index(r, c, k, a)),
                        };
                        let p1 = match e1 {
                            Some(e) => self.crossing(e, a, b),
                            None => ib.unwrap_or_else(|| self.tri_index(r, c, k, b)),
                        };
                        out.push(Piece { t0, t1, p0, p1 });
                    }
                }
            }
            if touched {
                for dr in -1i64..=1 {
                    for dc in -1i64..=1 {
                        let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                        if nr < 0 || nc < 0 || nr >= self.nrow as i64 || nc >= self.ncol as i64 {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        if seen.insert(nr * self.ncol + nc) {
                            stack.push((nr, nc));
                        }
                    }
                }
            }
        }
    }

    /// The boundary cells whose outer side segment `a b` meets.
    fn boundary_seeds(&self, a: [f64; 2], b: [f64; 2]) -> Vec<(usize, usize)> {
        let (nc, nr) = (self.ncol, self.nrow);
        let mut out = Vec::new();
        let mut test = |row: usize, col: usize, k: usize| {
            let (p, q) = self.edge_world(Edge::Side { row, col, k });
            if segments_meet(a, b, p, q) {
                out.push((row, col));
            }
        };
        for c in 0..nc {
            test(0, c, 0);
            test(nr - 1, c, 2);
        }
        for r in 0..nr {
            test(r, 0, 3);
            test(r, nc - 1, 1);
        }
        out
    }

    /// The pieces of segment `a b` inside the mesh, in order along it.
    fn segment_pieces(
        &self,
        a: [f64; 2],
        b: [f64; 2],
        ia: Option<[f64; 2]>,
        ib: Option<[f64; 2]>,
        hint: Option<(usize, usize)>,
    ) -> Vec<Piece> {
        let mut seen = HashSet::new();
        let mut pieces = Vec::new();
        let start: Vec<(usize, usize)> = self.locate(a, hint).into_iter().collect();
        self.flood(a, b, ia, ib, &start, &mut seen, &mut pieces);
        pieces.sort_by(|p, q| p.t0.total_cmp(&q.t0).then(p.t1.total_cmp(&q.t1)));
        // Complete when the pieces join from 0 to 1; otherwise the segment
        // leaves the mesh (or starts outside it), and may come back in
        // through any boundary side.
        let mut reach = 0.0;
        let mut complete = !pieces.is_empty() && pieces[0].t0 <= T_EPS;
        for p in &pieces {
            if p.t0 > reach + T_EPS {
                complete = false;
            }
            reach = f64::max(reach, p.t1);
        }
        if !complete || reach < 1.0 - T_EPS {
            let seeds = self.boundary_seeds(a, b);
            self.flood(a, b, ia, ib, &seeds, &mut seen, &mut pieces);
            pieces.sort_by(|p, q| p.t0.total_cmp(&q.t0).then(p.t1.total_cmp(&q.t1)));
        }
        pieces
    }

    /// Position along the mesh boundary (0 to `2 (ncol + nrow)`, clockwise
    /// in index space from corner `(0, 0)`) of an index point on, or next
    /// to, the outer walls.
    fn beta(&self, p: [f64; 2]) -> f64 {
        let (nc, nr) = (self.ncol as f64, self.nrow as f64);
        let x = p[0].clamp(0.0, nc);
        let y = p[1].clamp(0.0, nr);
        let d = [p[1].abs(), (p[0] - nc).abs(), (p[1] - nr).abs(), p[0].abs()];
        let k = (0..4).min_by(|&i, &j| d[i].total_cmp(&d[j])).unwrap_or(0);
        match k {
            0 => x,
            1 => nc + y,
            2 => nc + nr + (nc - x),
            _ => 2.0 * nc + nr + (nr - y),
        }
    }

    /// Boundary corner number `b` (mod the perimeter) as `(index, world)`.
    fn boundary_corner(&self, b: i64) -> ([f64; 2], [f64; 2]) {
        let (nc, nr) = (self.ncol as i64, self.nrow as i64);
        let per = 2 * (nc + nr);
        let b = b.rem_euclid(per);
        let (i, j) = if b < nc {
            (b, 0)
        } else if b < nc + nr {
            (nc, b - nc)
        } else if b < 2 * nc + nr {
            (nc - (b - nc - nr), nr)
        } else {
            (0, nr - (b - 2 * nc - nr))
        };
        ([i as f64, j as f64], self.corner(j as usize, i as usize))
    }

    /// The boundary corners strictly between positions `from` and `to`,
    /// going forward (increasing position) or backward.
    fn arc(&self, from: f64, to: f64, forward: bool) -> Vec<i64> {
        let per = 2.0 * (self.ncol + self.nrow) as f64;
        let mut out = Vec::new();
        if forward {
            let end = if to <= from { to + per } else { to };
            let mut b = from.floor() as i64 + 1;
            while (b as f64) < end {
                out.push(b);
                b += 1;
            }
        } else {
            let end = if to >= from { to - per } else { to };
            let mut b = from.ceil() as i64 - 1;
            while (b as f64) > end {
                out.push(b);
                b -= 1;
            }
        }
        out
    }

    /// A point strictly inside the mesh, for winding tests.
    fn interior_point(&self) -> Option<[f64; 2]> {
        self.cells.iter().find(|c| c.area2 != 0.0).map(|c| c.centre)
    }

    /// A world ring (closed implicitly; a repeated first point is
    /// ignored) as a ring in the mesh's index space `[col, row]`, row 0
    /// the first stored row. Empty when the ring and the mesh do not
    /// overlap. Parts outside the mesh are replaced by arcs of its
    /// boundary, so the result lies in `[0, ncol] x [0, nrow]` and its
    /// coverage of each cell is the world ring's coverage of that cell as
    /// a fraction of the cell's area.
    ///
    /// Errors unless the mesh uses [`CellMap::EqualArea`].
    pub fn ring_to_index(&self, ring: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
        if self.map != CellMap::EqualArea {
            return Err("the ring walk needs CellMap::EqualArea".into());
        }
        let mut pts: Vec<[f64; 2]> = ring.to_vec();
        while pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        if pts.len() < 3 || pts.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
            return Ok(Vec::new());
        }
        let n = pts.len();
        // ring vertices in index space, walking the locate hint along
        let mut hint = None;
        let mut vidx = Vec::with_capacity(n);
        let mut vcell = Vec::with_capacity(n);
        for &p in &pts {
            let cell = self.locate(p, hint);
            if cell.is_some() {
                hint = cell;
            }
            vcell.push(cell);
            vidx.push(cell.and_then(|(r, c)| {
                self.local(r, c, p)
                    .map(|[u, v]| [c as f64 + u, r as f64 + v])
            }));
        }
        // walk
        let mut nodes: Vec<Node> = Vec::new();
        for k in 0..n {
            let k1 = (k + 1) % n;
            let (a, b) = (pts[k], pts[k1]);
            let pieces = self.segment_pieces(a, b, vidx[k], vidx[k1], vcell[k].or(hint));
            let mut cur = 0.0f64;
            for p in &pieces {
                if p.t1 <= cur + T_EPS {
                    continue;
                }
                if p.t0 > cur + T_EPS {
                    // re-entry after a stretch outside (or a start outside)
                    nodes.push(Node {
                        idx: Some(p.p0),
                        world: lerp(a, b, p.t0),
                        out_before: true,
                    });
                } else if cur == 0.0 && vidx[k].is_none() {
                    // starts on the boundary at a vertex the mesh does not own
                    nodes.push(Node {
                        idx: Some(p.p0),
                        world: a,
                        out_before: false,
                    });
                }
                if p.t1 >= 1.0 - T_EPS {
                    nodes.push(Node {
                        idx: Some(vidx[k1].unwrap_or(p.p1)),
                        world: b,
                        out_before: false,
                    });
                } else {
                    nodes.push(Node {
                        idx: Some(p.p1),
                        world: lerp(a, b, p.t1),
                        out_before: false,
                    });
                }
                cur = p.t1;
            }
            if cur < 1.0 - T_EPS {
                nodes.push(Node {
                    idx: vidx[k1],
                    world: b,
                    out_before: true,
                });
            }
        }
        if nodes.iter().all(|q| q.idx.is_none()) {
            // no contact: the ring contains the whole mesh or none of it
            let inside = self
                .interior_point()
                .is_some_and(|p0| winding(&pts, p0) != 0);
            return Ok(if inside {
                let (nc, nr) = (self.ncol as f64, self.nrow as f64);
                vec![[0.0, 0.0], [nc, 0.0], [nc, nr], [0.0, nr]]
            } else {
                Vec::new()
            });
        }
        let m = nodes.len();
        // the link into node i runs outside the mesh
        let outside = |i: usize| {
            let (q, prev) = (nodes[i % m], nodes[(i + m - 1) % m]);
            q.out_before || q.idx.is_none() || prev.idx.is_none()
        };
        let mut out: Vec<[f64; 2]> = Vec::new();
        let Some(s) = (0..m).find(|&i| nodes[i].idx.is_some() && outside(i)) else {
            out.extend(nodes.iter().filter_map(|q| q.idx));
            return Ok(dedup(out));
        };
        let p0 = self.interior_point().unwrap_or([0.0, 0.0]);
        let mut j = s;
        loop {
            let node = nodes[j % m];
            let idx = node.idx.unwrap_or([0.0, 0.0]);
            out.push(idx);
            if !outside(j + 1) {
                j += 1;
                if j % m == s {
                    break;
                }
                continue;
            }
            // an excursion: follow it to the next node in the mesh
            let mut path = vec![node.world];
            let mut r = j + 1;
            while nodes[r % m].idx.is_none() {
                path.push(nodes[r % m].world);
                r += 1;
            }
            let re = nodes[r % m];
            path.push(re.world);
            let (be, br) = (self.beta(idx), self.beta(re.idx.unwrap_or(idx)));
            let mut best: Option<(i64, Vec<i64>)> = None;
            for forward in [true, false] {
                let corners = self.arc(be, br, forward);
                let mut lp = path.clone();
                lp.extend(corners.iter().rev().map(|&b| self.boundary_corner(b).1));
                let w = winding(&lp, p0).abs();
                if best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                    best = Some((w, corners));
                }
            }
            if let Some((_, corners)) = best {
                out.extend(corners.iter().map(|&b| self.boundary_corner(b).0));
            }
            j = r;
            if j % m == s {
                break;
            }
        }
        Ok(dedup(out))
    }

    /// Burn world geometries onto the mesh: controlledburn's tables with
    /// `(row, col)` of the mesh's cells (row 0 the first stored row) and
    /// edge fractions of each cell's world area. Polygons and points;
    /// lines are refused for now (their lengths need a per-piece rescale
    /// to world units). Coverage mode only.
    ///
    /// Polygon edges are straight in the mesh's world coordinates; for
    /// polygons in another CRS, densify first (bigcurve).
    pub fn burn(
        &self,
        geoms: &[Geometry],
        opts: impl Into<BurnOptions>,
    ) -> Result<BurnResult, String> {
        self.burn_par(geoms, opts, 1)
    }

    /// [`QuadMesh::burn`] on up to `threads` threads (0 for all
    /// available), with the same result record for record. Both halves
    /// run in parallel: walking the rings into the mesh's index space, on
    /// contiguous chunks of the geometries of about equal vertex counts,
    /// and the burn, with [`controlledburn::burn_par`]. One geometry is
    /// never split, and small inputs stay on the calling thread, as does
    /// everything with `threads = 1` (so `burn` needs no threads, as on
    /// wasm).
    pub fn burn_par(
        &self,
        geoms: &[Geometry],
        opts: impl Into<BurnOptions>,
        threads: usize,
    ) -> Result<BurnResult, String> {
        if self.map != CellMap::EqualArea {
            return Err("the ring walk needs CellMap::EqualArea".into());
        }
        burn_with(
            self.ncol,
            self.nrow,
            geoms,
            opts.into(),
            threads,
            &|r| self.ring_to_index(r),
            &|p, hint| self.locate(p, hint),
        )
    }

    /// Exact coverage of one small world ring, through a reused
    /// [`CoverageWindow`] in the mesh's index space: the window is placed
    /// over the ring's cells and holds each one's covered fraction of its
    /// world area. Returns false (and leaves an empty window) when the
    /// ring misses the mesh.
    pub fn cover_ring(&self, ring: &[[f64; 2]], win: &mut CoverageWindow) -> Result<bool, String> {
        let ir = self.ring_to_index(ring)?;
        if ir.len() < 3 {
            win.reset(0, 0, 0, 0);
            return Ok(false);
        }
        let xs: Vec<f64> = ir.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = ir.iter().map(|p| p[1]).collect();
        if !win.reset_to_ring(&xs, &ys) {
            return Ok(false);
        }
        win.add_ring(&xs, &ys, 1.0);
        win.resolve();
        Ok(true)
    }
}

/// A world ring to an index ring `[col, row]`.
pub(crate) type RingFn<'a> = dyn Fn(&[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> + Sync + 'a;
/// A world point (with the last point's cell as a hint) to its `(row, col)`.
pub(crate) type CellFn<'a> =
    dyn Fn([f64; 2], Option<(usize, usize)>) -> Option<(usize, usize)> + Sync + 'a;

/// Map rings with `ring` (world ring to index ring `[col, row]`) and
/// points with `cell`, then burn the index-space geometries with
/// controlledburn on the `ncol` by `nrow` grid. Shared by the curvilinear
/// and rectilinear front ends.
///
/// With `threads` other than 1 (0 for all available), the geometries are
/// mapped on several threads, in contiguous chunks of about equal vertex
/// counts taken in turn, and burned with [`controlledburn::burn_par`]. The result is the
/// same as with one thread, record for record.
pub(crate) fn burn_with(
    ncol: usize,
    nrow: usize,
    geoms: &[Geometry],
    opts: BurnOptions,
    threads: usize,
    ring: &RingFn,
    cell: &CellFn,
) -> Result<BurnResult, String> {
    if opts.mode != BurnMode::Coverage {
        return Err("only coverage mode is defined for a grid front end".into());
    }
    let threads = match threads {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    };
    // up to 8 chunks per thread, taken in turn, so a chunk holding one
    // large geometry doesn't hold up the rest
    // one thread: no chunks, no spawn (so `burn` works without threads, on wasm)
    let chunks = if threads == 1 {
        vec![(0, geoms.len())]
    } else {
        vertex_chunks(geoms, threads * 8)
    };
    let mapped: Vec<Result<Mapped, String>> = if chunks.len() <= 1 {
        vec![map_geoms(geoms, 0, nrow, ring, cell)]
    } else {
        let next = AtomicUsize::new(0);
        let mut done: Vec<(usize, Result<Mapped, String>)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads.min(chunks.len()))
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let k = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(a, b)) = chunks.get(k) else {
                                break out;
                            };
                            out.push((k, map_geoms(&geoms[a..b], a, nrow, ring, cell)));
                        }
                    })
                })
                .collect();
            hs.into_iter()
                .flat_map(|h| {
                    h.join()
                        .unwrap_or_else(|_| vec![(0, Err("a mapping thread panicked".into()))])
                })
                .collect()
        });
        done.sort_by_key(|d| d.0);
        done.into_iter().map(|d| d.1).collect()
    };
    let mut plane = Vec::with_capacity(geoms.len());
    let mut points = Vec::new();
    for m in mapped {
        let (pl, pt) = m?;
        plane.extend(pl);
        points.extend(pt);
    }
    let grid = GridSpec::new(0.0, 0.0, ncol as f64, nrow as f64, ncol as u32, nrow as u32);
    let mut res =
        controlledburn::burn_par(&plane, &grid, opts, threads).map_err(|e| e.to_string())?;
    res.points.extend(points);
    Ok(res)
}

/// Index-space geometries (one per input, in order) and points.
type Mapped = (Vec<Geometry>, Vec<GridPoint>);

/// Contiguous `[a, b)` ranges of `geoms`, at most `n` of them, of
/// about equal vertex counts; one range when a thread isn't worth it.
fn vertex_chunks(geoms: &[Geometry], n: usize) -> Vec<(usize, usize)> {
    // at least this many vertices a chunk
    const MIN_VERTICES: usize = 1 << 12;
    let w: Vec<usize> = geoms.iter().map(vertices).collect();
    let total: usize = w.iter().sum();
    let n = n.min(geoms.len()).min(total / MIN_VERTICES).max(1);
    if n == 1 {
        return vec![(0, geoms.len())];
    }
    let mut out = Vec::with_capacity(n);
    let (mut a, mut acc) = (0, 0);
    for (k, wk) in w.iter().enumerate() {
        acc += wk;
        if acc * n >= total * (out.len() + 1) && out.len() + 1 < n {
            out.push((a, k + 1));
            a = k + 1;
        }
    }
    out.push((a, geoms.len()));
    out
}

fn vertices(g: &Geometry) -> usize {
    let poly = |p: &Polygon| p.rings.iter().map(Vec::len).sum::<usize>();
    match g {
        Geometry::Polygon(p) => poly(p),
        Geometry::MultiPolygon(ps) => ps.iter().map(poly).sum(),
        Geometry::MultiPoint(cs) => cs.len(),
        _ => 1,
    }
}

/// Map `geoms` (whose first has id `id0`) into the index plane.
fn map_geoms(
    geoms: &[Geometry],
    id0: usize,
    nrow: usize,
    ring: &RingFn,
    cell: &CellFn,
) -> Result<Mapped, String> {
    let nr = nrow as f64;
    let to_plane = |r: &[Coord]| -> Result<Vec<Coord>, String> {
        let pts: Vec<[f64; 2]> = r.iter().map(|c| [c.x, c.y]).collect();
        let mut out: Vec<Coord> = ring(&pts)?
            .into_iter()
            .map(|p| Coord {
                x: p[0],
                y: nr - p[1],
            })
            .collect();
        match out.first() {
            Some(&f) if out.len() >= 3 => out.push(f),
            _ => out.clear(),
        }
        Ok(out)
    };
    let polygon = |p: &Polygon| -> Result<Option<Polygon>, String> {
        let mut rings = Vec::with_capacity(p.rings.len());
        for (k, r) in p.rings.iter().enumerate() {
            let ir = to_plane(r)?;
            if ir.is_empty() {
                if k == 0 {
                    return Ok(None);
                }
                continue;
            }
            rings.push(ir);
        }
        Ok(Some(Polygon { rings }))
    };
    let mut plane = Vec::with_capacity(geoms.len());
    let mut points = Vec::new();
    for (k, g) in geoms.iter().enumerate() {
        let id = (id0 + k) as i32;
        let polys: Vec<Polygon> = match g {
            Geometry::Polygon(p) => polygon(p)?.into_iter().collect(),
            Geometry::MultiPolygon(ps) => {
                let mut v = Vec::new();
                for p in ps {
                    v.extend(polygon(p)?);
                }
                v
            }
            Geometry::Point(c) => {
                points.extend(cell([c.x, c.y], None).map(|(r, c)| GridPoint {
                    row: r as i32,
                    col: c as i32,
                    id,
                }));
                Vec::new()
            }
            Geometry::MultiPoint(cs) => {
                let mut hint = None;
                for c in cs {
                    let at = cell([c.x, c.y], hint);
                    if let Some((r, col)) = at {
                        points.push(GridPoint {
                            row: r as i32,
                            col: col as i32,
                            id,
                        });
                        hint = at;
                    }
                }
                Vec::new()
            }
            Geometry::LineString(_) | Geometry::MultiLineString(_) => {
                return Err("lines are not supported by the grid front ends yet".into());
            }
        };
        plane.push(Geometry::MultiPolygon(polys));
    }
    Ok((plane, points))
}
