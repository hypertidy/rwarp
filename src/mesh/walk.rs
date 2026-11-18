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
//! Two outer walls that meet in the world (a seam, a fold) are crossed
//! the same way. Where the ring passes from a cell on one to a cell on
//! the other, its index position jumps between the walls at one world
//! point. That is taken as an exit and a re-entry with nothing outside
//! between, and the arc chosen runs along the walls. A vertex on such a
//! wall is placed in each piece's own triangle, not only in the cell that
//! owns it.
//!
//! A cell with no area (a collapsed column, a T-point pivot of a tripolar
//! grid) makes the same kind of jump inside the mesh: the ring passes from
//! one side of it to another at one world point. The walk links the two
//! straight through the collapsed cell's index square, which weighs
//! nothing, and where that cell is on an outer wall, through it to the
//! wall and then across the walls ([`Link`]).
//!
//! The walk uses the equal-area map whatever the mesh's [`CellMap`] (which
//! is for its point methods): under `Bilinear` the preimage of a straight
//! edge is curved, and mapping cut points alone would not be exact for
//! either model. A bilinear measure would need the ring densified
//! (bigcurve; `design/curvilinear-frontend.md`, section 5).

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use controlledburn::{
    BurnMode, BurnOptions, BurnResult, Coord, CoverageWindow, Geometry, GridPoint, GridSpec,
    Polygon,
};

use super::{orient, reproject, CellMap, Period, QuadMesh, UNIT};
use crate::crs::CrsTransform;

/// A multiplicative hash for cell numbers (as rustc's `FxHasher`): the
/// flood's set of seen cells is small and hot, and needs no protection
/// against chosen keys.
#[derive(Default)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

type CellSet = HashSet<usize, BuildHasherDefault<CellHasher>>;

/// The cells a flood has seen: a short list, moved into a hash set if it
/// grows (a flood along a long segment).
#[derive(Default)]
struct Seen {
    small: Vec<usize>,
    big: CellSet,
}

impl Seen {
    fn clear(&mut self) {
        self.small.clear();
        self.big.clear();
    }

    /// Whether `k` is new (and now seen).
    fn insert(&mut self, k: usize) -> bool {
        if self.big.is_empty() {
            if self.small.contains(&k) {
                return false;
            }
            if self.small.len() < 32 {
                self.small.push(k);
                return true;
            }
            self.big.extend(self.small.drain(..));
        }
        self.big.insert(k)
    }
}

/// Buffers the segments of one ring reuse.
#[derive(Default)]
struct Scratch {
    seen: Seen,
    stack: Vec<(usize, usize)>,
    pieces: Vec<Piece>,
}

/// Gaps in segment parameter smaller than this join two pieces (the two
/// sides of a vertex the segment passes through can give `t` an ulp apart).
const T_EPS: f64 = 1e-12;

/// Index-space distance within which a point is on an outer wall, or in a
/// cell's closure (a vertex placed by its barycentric map lands within
/// rounding of the side it is on).
const WALL_EPS: f64 = 1e-9;

/// Index-space distance beyond which two positions of one world point are
/// on different walls that meet in the world, not the same point.
const JUMP_EPS: f64 = 1e-6;

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

/// Whether two index points are different points, not one to rounding.
fn apart(p: [f64; 2], q: [f64; 2]) -> bool {
    (p[0] - q[0]).abs() > JUMP_EPS || (p[1] - q[1]).abs() > JUMP_EPS
}

/// How the walk goes from the end of one piece to the start of the next
/// when they are one world point at two index points. Nothing with area
/// lies between them.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Link {
    /// The same index point.
    Same,
    /// Straight through one cell with no area whose closure holds both.
    Direct,
    /// Across two outer walls that meet in the world (a seam, a fold).
    Walls,
    /// Through a cell with no area to this point on its outer wall, then
    /// across the walls (`false`); or across the walls first (`true`).
    Via([f64; 2], bool),
}

/// Push the nodes that take the walk across `link` to index point `q` (at
/// world point `w`).
fn push_link(nodes: &mut Vec<Node>, link: Link, q: [f64; 2], w: [f64; 2]) {
    let mut node = |idx: [f64; 2], out_before: bool| {
        nodes.push(Node {
            idx: Some(idx),
            world: w,
            out_before,
        })
    };
    match link {
        Link::Same => {}
        Link::Direct => node(q, false),
        Link::Walls => node(q, true),
        Link::Via(y, walls_first) => {
            node(y, walls_first);
            node(q, !walls_first);
        }
    }
}

fn lost(w: [f64; 2]) -> String {
    format!(
        "the ring crosses cells with no area at ({}, {}) in a way the walk can't follow",
        w[0], w[1]
    )
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
    ///
    /// `q` and `x` are the cell's corners and split centre; each edge's
    /// ends are taken from them in its canonical direction, as
    /// [`QuadMesh::edge_world`] gives them.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn clip(
        &self,
        q: &[[f64; 2]; 4],
        x: [f64; 2],
        row: usize,
        col: usize,
        k: usize,
        a: [f64; 2],
        b: [f64; 2],
    ) -> Option<(f64, Option<Edge>, f64, Option<Edge>)> {
        let k1 = (k + 1) % 4;
        let side = match k {
            0 => (q[0], q[1]),
            1 => (q[1], q[2]),
            2 => (q[3], q[2]),
            _ => (q[0], q[3]),
        };
        let edges = [
            (Edge::Spoke { row, col, j: k }, (x, q[k]), q[k1]),
            (Edge::Side { row, col, k }, side, x),
            (Edge::Spoke { row, col, j: k1 }, (x, q[k1]), q[k]),
        ];
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let (mut e0, mut e1) = (None, None);
        for (e, (p0, p1), opp) in edges {
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

    /// Whether segment `a b` meets the bounding box of corners `q`.
    fn meets_box(q: &[[f64; 2]; 4], a: [f64; 2], b: [f64; 2]) -> bool {
        let (mut x0, mut x1, mut y0, mut y1) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for p in q.iter() {
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
        scratch: &mut Scratch,
    ) {
        let Scratch { seen, stack, pieces: out } = scratch;
        stack.clear();
        for &s in seeds {
            if seen.insert(s.0 * self.ncol + s.1) {
                stack.push(s);
            }
        }
        while let Some((r, c)) = stack.pop() {
            let mut touched = false;
            let q = self.corners(r, c);
            let cell = &self.cells[r * self.ncol + c];
            if cell.area2 == 0.0 {
                touched = Self::meets_box(&q, a, b);
            } else if Self::meets_box(&q, a, b) {
                // (a segment that misses the cell's box misses the cell)
                for k in 0..4 {
                    let Some((t0, e0, t1, e1)) = self.clip(&q, cell.centre, r, c, k, a, b) else {
                        continue;
                    };
                    touched = true;
                    if t1 > t0 {
                        // a vertex's own position, unless it is owned on
                        // the other side of a seam or fold from this cell
                        let p0 = match e0 {
                            Some(e) => self.crossing(e, a, b),
                            None => ia
                                .filter(|&q| self.in_triangle(q, r, c, k))
                                .unwrap_or_else(|| self.tri_index(r, c, k, a)),
                        };
                        let p1 = match e1 {
                            Some(e) => self.crossing(e, a, b),
                            None => ib
                                .filter(|&q| self.in_triangle(q, r, c, k))
                                .unwrap_or_else(|| self.tri_index(r, c, k, b)),
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

    /// The boundary cells whose outer side segment `a b` meets, in the
    /// order of the outer sides ([`QuadMesh::boundary_side`]), found
    /// through the sides' bucket index.
    fn boundary_seeds(&self, a: [f64; 2], b: [f64; 2]) -> Vec<(usize, usize)> {
        let mut near = Vec::new();
        self.boundary_sides_near([a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])], &mut near);
        let mut out = Vec::new();
        for o in near {
            let (row, col, k) = self.boundary_side(o);
            let (p, q) = self.edge_world(Edge::Side { row, col, k });
            if segments_meet(a, b, p, q) {
                out.push((row, col));
            }
        }
        out
    }

    /// The pieces of segment `a b` inside the mesh, in order along it,
    /// left in `scratch.pieces`.
    fn segment_pieces(
        &self,
        a: [f64; 2],
        b: [f64; 2],
        ia: Option<[f64; 2]>,
        ib: Option<[f64; 2]>,
        hint: Option<(usize, usize)>,
        scratch: &mut Scratch,
    ) {
        scratch.seen.clear();
        scratch.pieces.clear();
        // the cell that owns `a`; failing that (`a` outside, or on a corner
        // whose owner by the low-index rule has no area, as next to a
        // tripolar grid's pivot) the cells near it
        let start: Vec<(usize, usize)> = match self.locate_planar(a, hint) {
            Some(s) => vec![s],
            None => self
                .buckets()
                .candidates(a)
                .map(|k| (k / self.ncol, k % self.ncol))
                .collect(),
        };
        self.flood(a, b, ia, ib, &start, scratch);
        let pieces = &mut scratch.pieces;
        pieces.sort_by(|p, q| p.t0.total_cmp(&q.t0).then(p.t1.total_cmp(&q.t1)));
        // Complete when the pieces join from 0 to 1; otherwise the segment
        // leaves the mesh (or starts outside it), and may come back in
        // through any boundary side.
        let mut reach = 0.0;
        let mut complete = !pieces.is_empty() && pieces[0].t0 <= T_EPS;
        for p in pieces.iter() {
            if p.t0 > reach + T_EPS {
                complete = false;
            }
            reach = f64::max(reach, p.t1);
        }
        if !complete || reach < 1.0 - T_EPS {
            let seeds = self.boundary_seeds(a, b);
            self.flood(a, b, ia, ib, &seeds, scratch);
            scratch.pieces.sort_by(|p, q| p.t0.total_cmp(&q.t0).then(p.t1.total_cmp(&q.t1)));
        }
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

    /// Whether two index points of one world point are on outer walls
    /// that meet in the world (a seam, a fold): both on walls, and apart.
    /// Elsewhere the two are the same point, bitwise or to rounding.
    fn glued(&self, p: [f64; 2], q: [f64; 2]) -> bool {
        let (nc, nr) = (self.ncol as f64, self.nrow as f64);
        let on_wall = |p: [f64; 2]| {
            p[0].abs() <= WALL_EPS
                || p[1].abs() <= WALL_EPS
                || (p[0] - nc).abs() <= WALL_EPS
                || (p[1] - nr).abs() <= WALL_EPS
        };
        apart(p, q) && on_wall(p) && on_wall(q)
    }

    /// The cells whose closure holds index point `p`.
    fn cells_at(&self, p: [f64; 2]) -> Vec<(usize, usize)> {
        let span = |v: f64, n: usize| {
            let lo = ((v - WALL_EPS).floor() as i64).clamp(0, n as i64 - 1) as usize;
            let hi = ((v + WALL_EPS).floor() as i64).clamp(0, n as i64 - 1) as usize;
            lo..=hi
        };
        let mut out = Vec::new();
        for r in span(p[1], self.nrow) {
            for c in span(p[0], self.ncol) {
                if Self::in_cell(p, r, c) {
                    out.push((r, c));
                }
            }
        }
        out
    }

    fn flat(&self, (r, c): (usize, usize)) -> bool {
        self.cells[r * self.ncol + c].area2 == 0.0
    }

    /// The index point of world point `w` on an outer side of cell
    /// `(row, col)`, if `w` is on one.
    fn on_outer_side(&self, row: usize, col: usize, w: [f64; 2]) -> Option<[f64; 2]> {
        let outer = [
            row == 0,
            col + 1 == self.ncol,
            row + 1 == self.nrow,
            col == 0,
        ];
        (0..4).filter(|&k| outer[k]).find_map(|k| {
            let e = Edge::Side { row, col, k };
            let (e0, e1) = self.edge_world(e);
            let d = [e1[0] - e0[0], e1[1] - e0[1]];
            let len2 = d[0] * d[0] + d[1] * d[1];
            let s = if len2 > 0.0 {
                (((w[0] - e0[0]) * d[0] + (w[1] - e0[1]) * d[1]) / len2).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let f = lerp(e0, e1, s);
            let tol = 1e-9 * (w[0].abs() + w[1].abs()) + 1e-6 * len2.sqrt();
            ((f[0] - w[0]).abs() <= tol && (f[1] - w[1]).abs() <= tol)
                .then(|| self.edge_index(e, s))
        })
    }

    /// How to go from index point `p` to index point `q`, both at world
    /// point `w` with no area between them, or `None` when the walk can't
    /// tell (a chain of cells with no area).
    fn link(&self, p: [f64; 2], q: [f64; 2], w: [f64; 2]) -> Option<Link> {
        if !apart(p, q) {
            return Some(Link::Same);
        }
        let (at_p, at_q) = (self.cells_at(p), self.cells_at(q));
        let shared = |flat: bool| {
            at_p.iter()
                .any(|&k| self.flat(k) == flat && at_q.contains(&k))
        };
        if shared(true) {
            return Some(Link::Direct);
        }
        if self.glued(p, q) {
            return Some(Link::Walls);
        }
        if shared(false) {
            // a triangle of a degenerate split (an inexact cell)
            return Some(Link::Direct);
        }
        // a cell with no area at the boundary, as at the pivot of a
        // tripolar grid's fold where it meets the seam
        for (ends, walls_first) in [(&at_p, false), (&at_q, true)] {
            for &k in ends.iter().filter(|&&k| self.flat(k)) {
                if let Some(y) = self.on_outer_side(k.0, k.1, w) {
                    let (a, b) = if walls_first { (p, y) } else { (y, q) };
                    if self.glued(a, b) {
                        return Some(Link::Via(y, walls_first));
                    }
                }
            }
        }
        None
    }

    /// Whether index point `p` is in the closure of triangle `k` of cell
    /// `(row, col)` in index space, to rounding. (A cell's closure is not
    /// enough: where two corners are one world point, a vertex there has
    /// an index point at each, one per triangle.)
    fn in_triangle(&self, p: [f64; 2], row: usize, col: usize, k: usize) -> bool {
        let (c, r) = (col as f64, row as f64);
        let at = |u: [f64; 2]| [c + u[0], r + u[1]];
        let ux = self.cells[row * self.ncol + col].unit;
        let (x, a, b) = (at(ux), at(UNIT[k]), at(UNIT[(k + 1) % 4]));
        let s = orient(x, a, b).signum();
        s != 0.0
            && [orient(x, a, p), orient(a, b, p), orient(b, x, p)]
                .iter()
                .all(|&v| s * v >= -WALL_EPS)
    }

    /// Whether index point `p` is in cell `(row, col)`'s closure, to
    /// rounding.
    fn in_cell(p: [f64; 2], row: usize, col: usize) -> bool {
        let (c, r) = (col as f64, row as f64);
        p[0] >= c - WALL_EPS
            && p[0] <= c + 1.0 + WALL_EPS
            && p[1] >= r - WALL_EPS
            && p[1] <= r + 1.0 + WALL_EPS
    }

    /// A point strictly inside the mesh, for winding tests (also
    /// `MeshSet::burn`'s, which shortens rings without changing their
    /// winding round it).
    pub(super) fn interior_point(&self) -> Option<[f64; 2]> {
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
    /// Index positions are under the equal-area map, whatever the mesh's
    /// [`CellMap`]: inside a cell they can differ from
    /// [`QuadMesh::to_index`] on a `Bilinear` mesh (never on a cell side).
    ///
    /// The ring is taken where it is in the plane, with a declared period
    /// too ([`QuadMesh::with_period`]); [`QuadMesh::burn`] and
    /// [`QuadMesh::cover_ring`] walk it at each whole period.
    pub fn ring_to_index(&self, ring: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
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
            let cell = self.locate_planar(p, hint);
            if cell.is_some() {
                hint = cell;
            }
            vcell.push(cell);
            vidx.push(cell.and_then(|(r, c)| {
                self.local_in(CellMap::EqualArea, r, c, p)
                    .map(|[u, v]| [c as f64 + u, r as f64 + v])
            }));
        }
        // walk
        let mut nodes: Vec<Node> = Vec::new();
        let mut scratch = Scratch::default();
        let mut first = None;
        for k in 0..n {
            let k1 = (k + 1) % n;
            let (a, b) = (pts[k], pts[k1]);
            self.segment_pieces(a, b, vidx[k], vidx[k1], vcell[k].or(hint), &mut scratch);
            let mut cur = 0.0f64;
            for p in &scratch.pieces {
                if p.t1 <= cur + T_EPS {
                    continue;
                }
                let w0 = lerp(a, b, p.t0);
                let link = match nodes.last().and_then(|q| q.idx) {
                    Some(q) if p.t0 <= cur + T_EPS => {
                        self.link(q, p.p0, w0).ok_or_else(|| lost(w0))?
                    }
                    _ => Link::Same,
                };
                if p.t0 > cur + T_EPS || link == Link::Walls {
                    // re-entry after a stretch outside (or a start
                    // outside), or across two outer walls that meet in
                    // the world (a seam, a fold): closed along the walls
                    nodes.push(Node {
                        idx: Some(p.p0),
                        world: w0,
                        out_before: true,
                    });
                } else if link != Link::Same {
                    // across a cell with no area
                    push_link(&mut nodes, link, p.p0, w0);
                } else if cur == 0.0 && vidx[k].is_none() {
                    // starts on the boundary at a vertex the mesh does not own
                    nodes.push(Node {
                        idx: Some(p.p0),
                        world: a,
                        out_before: false,
                    });
                    if k == 0 {
                        first = Some((p.p0, true));
                    }
                } else if k == 0 && cur == 0.0 {
                    // the ring's start, left implicit: its last node
                    first = Some((p.p0, false));
                }
                if p.t1 >= 1.0 - T_EPS {
                    // the vertex's own position, unless it is on the other
                    // side of a seam, a fold or a cell with no area from
                    // this piece
                    let v = vidx[k1].filter(|&q| !apart(q, p.p1));
                    nodes.push(Node {
                        idx: Some(v.unwrap_or(p.p1)),
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
        // a ring that starts on a seam, a fold or a cell with no area and
        // comes back to its start from the other side
        if let (Some((q, pushed)), Some(l)) = (first, nodes.last().and_then(|q| q.idx)) {
            let link = self.link(l, q, pts[0]).ok_or_else(|| lost(pts[0]))?;
            if link != Link::Same {
                let mut head = Vec::new();
                push_link(&mut head, link, q, pts[0]);
                if pushed {
                    // the start is nodes[0] already; keep its new link
                    nodes[0].out_before = head.pop().is_some_and(|n| n.out_before);
                }
                nodes.splice(0..0, head);
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
            // leaving and coming back at the same boundary point (a ring
            // that only touches the mesh there): both arcs are the whole
            // boundary, so the empty arc is a candidate too
            let mut arcs = vec![self.arc(be, br, true), self.arc(be, br, false)];
            if be == br {
                arcs.push(Vec::new());
            }
            for corners in arcs {
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
        // fewer than three points: the ring only touches the mesh
        let out = dedup(out);
        Ok(if out.len() < 3 { Vec::new() } else { out })
    }

    /// Burn world geometries onto the mesh: controlledburn's tables with
    /// `(row, col)` of the mesh's cells (row 0 the first stored row) and
    /// edge fractions of each cell's world area. Polygons and points;
    /// lines are refused for now (their lengths need a per-piece rescale
    /// to world units). Coverage mode only.
    ///
    /// Polygon edges are straight in the mesh's world coordinates; for
    /// polygons in another CRS, see [`QuadMesh::burn_from`]. With a declared
    /// period ([`QuadMesh::with_period`]), each polygon is burned at every
    /// whole period at which it overlaps the mesh, and points are found at
    /// any whole period.
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
        burn_with(
            self.ncol,
            self.nrow,
            geoms,
            opts.into(),
            threads,
            self.period,
            &|r| self.ring_to_index(r),
            &|p, hint| self.locate(p, hint),
        )
    }

    /// [`QuadMesh::burn`] for polygons and points in another CRS, with
    /// edges straight there: `to_world` takes them to the mesh's world
    /// coordinates, `from_world` takes the mesh's back
    /// ([`crs_pair`](crate::crs::crs_pair)`(mesh_crs, geoms_crs)` gives
    /// both, in that order).
    ///
    /// - **Densified.** Each polygon edge is cut into equal pieces no
    ///   longer than the mesh's tallest cell is high in the geometries'
    ///   CRS, and each piece densified by bigcurve (d3-geo's adaptive
    ///   resampling, planar edges) until no chord strays more than
    ///   `tolerance` from the edge's image, in world units. A fraction is
    ///   then good to about `tolerance` times the length of polygon edge
    ///   in the cell, over its area, and the points grow as `1 /
    ///   sqrt(tolerance)`. The first cut keeps an edge whose ends land
    ///   together (a parallel all the way round, in a polar projection)
    ///   from being passed over.
    /// - **Cut to the mesh first.** The polygons are cut to the y range of
    ///   the mesh's vertices in their CRS, widened by its tallest cell each
    ///   way, as [`MeshSet::burn`](super::MeshSet::burn) cuts a band's. A
    ///   projection holds the near side well and the far one not at all (a
    ///   polar one sends the far pole to its edge, or to infinity), so a
    ///   polygon round the far pole would come out inside out. Points
    ///   beyond that range, or that don't go to the world, are on no cell.
    /// - **Split where a ring meets itself**, as there: a cut at 180
    ///   degrees comes out of a polar projection with both sides on one
    ///   line.
    /// - **With a declared period** ([`QuadMesh::with_period`]: a mesh in
    ///   longitude and latitude), each ring is unwrapped in the world, each
    ///   point to within half a period of the one before, and then burned
    ///   at every period it reaches. A ring that doesn't close then goes
    ///   round a pole of the world and is refused.
    ///
    /// Errors for a polygon point that doesn't go to the world, a
    /// tolerance that isn't positive, lines, and modes other than
    /// coverage. `to_world` and `from_world` run on the calling thread,
    /// so they needn't be thread-safe.
    ///
    /// ```
    /// use rwarp::mesh::{CellMap, QuadMesh};
    /// use rwarp::CrsTransform;
    /// use controlledburn::{BurnMode, Coord, Geometry, Polygon};
    ///
    /// // north-polar LAEA on the unit sphere, and back
    /// struct Laea(bool);
    /// impl CrsTransform for Laea {
    ///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    ///         Some(if self.0 {
    ///             let rho = 2.0 * ((90.0 - y).to_radians() / 2.0).sin();
    ///             let (s, c) = x.to_radians().sin_cos();
    ///             (rho * s, -rho * c)
    ///         } else {
    ///             (x.atan2(-y).to_degrees(), 90.0 - 2.0 * (x.hypot(y) / 2.0).asin().to_degrees())
    ///         })
    ///     }
    /// }
    /// // a 20 by 20 mesh of the square round the pole out to 1, in LAEA
    /// let n = 20;
    /// let v: Vec<f64> = (0..=n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
    /// let x: Vec<f64> = (0..=n).flat_map(|_| v.iter().copied()).collect();
    /// let y: Vec<f64> = (0..=n).flat_map(|j| std::iter::repeat_n(-v[j], n + 1)).collect();
    /// let mesh = QuadMesh::from_corners(n, n, x, y, CellMap::EqualArea).unwrap();
    ///
    /// // the cap north of 70 N, in longitude and latitude: a circle in
    /// // LAEA, of area 2 pi (1 - sin 70) on the unit sphere
    /// let ring = [(-180.0, 70.0), (180.0, 70.0), (180.0, 90.0), (-180.0, 90.0), (-180.0, 70.0)];
    /// let cap = Geometry::Polygon(Polygon::new(vec![ring.iter().map(|&(x, y)| Coord { x, y }).collect()]));
    /// let r = mesh.burn_from(&[cap], &Laea(true), &Laea(false), 1e-6, BurnMode::Coverage).unwrap();
    /// let area = r.covered_cells() * 0.01; // each cell 0.1 by 0.1
    /// let want = std::f64::consts::TAU * (1.0 - 70f64.to_radians().sin());
    /// assert!((area - want).abs() < 1e-5, "{area} vs {want}");
    /// ```
    pub fn burn_from(
        &self,
        geoms: &[Geometry],
        to_world: &dyn CrsTransform,
        from_world: &dyn CrsTransform,
        tolerance: f64,
        opts: impl Into<BurnOptions>,
    ) -> Result<BurnResult, String> {
        self.burn_from_par(geoms, to_world, from_world, tolerance, opts, 1)
    }

    /// [`QuadMesh::burn_from`] with the burn on up to `threads` threads
    /// (0 for all available), as [`QuadMesh::burn_par`]: the same result
    /// record for record. The projection stays on the calling thread.
    pub fn burn_from_par(
        &self,
        geoms: &[Geometry],
        to_world: &dyn CrsTransform,
        from_world: &dyn CrsTransform,
        tolerance: f64,
        opts: impl Into<BurnOptions>,
        threads: usize,
    ) -> Result<BurnResult, String> {
        let opts = opts.into();
        if opts.mode != BurnMode::Coverage {
            return Err("only coverage mode is defined for a grid front end".into());
        }
        let w = self.ncol + 1;
        let reach = reproject::reach(self.ncol, self.nrow, |j, i| [self.x[j * w + i], self.y[j * w + i]], from_world)?;
        let world = reproject::to_world(geoms, to_world, reach, tolerance, self.period())?;
        self.burn_par(&world, opts, threads)
    }

    /// Exact coverage of one small world ring, through a reused
    /// [`CoverageWindow`] in the mesh's index space: the window is placed
    /// over the ring's cells and holds each one's covered fraction of its
    /// world area. Returns false (and leaves an empty window) when the
    /// ring misses the mesh.
    ///
    /// With a declared period ([`QuadMesh::with_period`]) the ring is
    /// walked at each whole period at which it overlaps the mesh, and the
    /// window spans all the pieces: across the seam, the mesh's full
    /// width.
    pub fn cover_ring(&self, ring: &[[f64; 2]], win: &mut CoverageWindow) -> Result<bool, String> {
        let mut rings = Vec::new();
        for s in turn_shifts(self.period, ring.iter().map(|p| p[0]))? {
            let ir = if s == 0.0 {
                self.ring_to_index(ring)?
            } else {
                let moved: Vec<[f64; 2]> = ring.iter().map(|p| [p[0] + s, p[1]]).collect();
                self.ring_to_index(&moved)?
            };
            if ir.len() >= 3 {
                rings.push(ir);
            }
        }
        Ok(cover_rings(&rings, win))
    }
}

/// The shifts in x, whole periods, at which a ring with x values `xs`
/// overlaps the grid's x range: just 0 without a period, or for a ring
/// with a coordinate that isn't finite. Errors for a ring that would be
/// taken at more than 16 periods.
pub(crate) fn turn_shifts(
    period: Option<Period>,
    xs: impl Iterator<Item = f64>,
) -> Result<Vec<f64>, String> {
    let Some(per) = period else {
        return Ok(vec![0.0]);
    };
    let [lo, hi] = xs.fold([f64::INFINITY, f64::NEG_INFINITY], |[a, b], x| [a.min(x), b.max(x)]);
    if !(lo.is_finite() && hi.is_finite()) {
        return Ok(vec![0.0]);
    }
    let (k0, k1) = per.turns(lo, hi);
    if k1.saturating_sub(k0) >= 16 {
        return Err(format!(
            "a ring from x = {lo} to {hi} would be taken at more than 16 periods of {}",
            per.period
        ));
    }
    Ok((k0..=k1).map(|k| k as f64 * per.period).collect())
}

/// The coverage of index rings `rings` (each with three points or more)
/// in `win`, placed over the cells under all of them. False, with an
/// empty window, when there are none or a point isn't finite.
pub(crate) fn cover_rings(rings: &[Vec<[f64; 2]>], win: &mut CoverageWindow) -> bool {
    let split = |r: &[[f64; 2]]| -> (Vec<f64>, Vec<f64>) { r.iter().map(|p| (p[0], p[1])).unzip() };
    match rings {
        [] => {
            win.reset(0, 0, 0, 0);
            false
        }
        [r] => {
            let (xs, ys) = split(r);
            if !win.reset_to_ring(&xs, &ys) {
                return false;
            }
            win.add_ring(&xs, &ys, 1.0);
            win.resolve();
            true
        }
        _ => {
            let (mut x0, mut x1, mut y0, mut y1) =
                (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            for p in rings.iter().flatten() {
                (x0, x1, y0, y1) = (x0.min(p[0]), x1.max(p[0]), y0.min(p[1]), y1.max(p[1]));
            }
            if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
                win.reset(0, 0, 0, 0);
                return false;
            }
            let (c0, r0) = (x0.floor() as i64, y0.floor() as i64);
            let c1 = (x1.ceil() as i64).max(c0 + 1);
            let r1 = (y1.ceil() as i64).max(r0 + 1);
            win.reset(c0, r0, (c1 - c0) as usize, (r1 - r0) as usize);
            for r in rings {
                let (xs, ys) = split(r);
                win.add_ring(&xs, &ys, 1.0);
            }
            win.resolve();
            true
        }
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
/// and rectilinear front ends. With `period`, each polygon is mapped at
/// every whole period at which its exterior ring overlaps the grid's x
/// range, the copies burned together.
///
/// With `threads` other than 1 (0 for all available), the geometries are
/// mapped on several threads, in contiguous chunks of about equal vertex
/// counts taken in turn, and burned with [`controlledburn::burn_par`]. The result is the
/// same as with one thread, record for record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn burn_with(
    ncol: usize,
    nrow: usize,
    geoms: &[Geometry],
    opts: BurnOptions,
    threads: usize,
    period: Option<Period>,
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
        vec![map_geoms(geoms, 0, nrow, period, ring, cell)]
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
                            out.push((k, map_geoms(&geoms[a..b], a, nrow, period, ring, cell)));
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
    let mut res = burn_plane(ncol, nrow, &plane, opts, threads)?;
    res.points.extend(points);
    Ok(res)
}

/// Burn index-space geometries on the `ncol` by `nrow` grid with
/// controlledburn, on up to `threads` threads (it uses fewer for less
/// work, and the calling thread alone for one).
pub(crate) fn burn_plane(
    ncol: usize,
    nrow: usize,
    plane: &[Geometry],
    opts: BurnOptions,
    threads: usize,
) -> Result<BurnResult, String> {
    let grid = GridSpec::new(0.0, 0.0, ncol as f64, nrow as f64, ncol as u32, nrow as u32);
    controlledburn::burn_par(plane, &grid, opts, threads).map_err(|e| e.to_string())
}

/// Index-space geometries (one per input, in order) and points.
pub(crate) type Mapped = (Vec<Geometry>, Vec<GridPoint>);

/// Contiguous `[a, b)` ranges of `geoms`, at most `n` of them, of
/// about equal vertex counts; one range when a thread isn't worth it.
pub(crate) fn vertex_chunks(geoms: &[Geometry], n: usize) -> Vec<(usize, usize)> {
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

pub(crate) fn vertices(g: &Geometry) -> usize {
    let poly = |p: &Polygon| p.rings.iter().map(Vec::len).sum::<usize>();
    match g {
        Geometry::Polygon(p) => poly(p),
        Geometry::MultiPolygon(ps) => ps.iter().map(poly).sum(),
        Geometry::MultiPoint(cs) => cs.len(),
        _ => 1,
    }
}

/// Map `geoms` (whose first has id `id0`) into the index plane, each
/// polygon at every whole period at which it overlaps the grid, with
/// `period`.
pub(crate) fn map_geoms(
    geoms: &[Geometry],
    id0: usize,
    nrow: usize,
    period: Option<Period>,
    ring: &RingFn,
    cell: &CellFn,
) -> Result<Mapped, String> {
    let nr = nrow as f64;
    // a ring moved by `s` in x (0: as it is)
    let to_plane = |r: &[Coord], s: f64| -> Result<Vec<Coord>, String> {
        let pts: Vec<[f64; 2]> = if s == 0.0 {
            r.iter().map(|c| [c.x, c.y]).collect()
        } else {
            r.iter().map(|c| [c.x + s, c.y]).collect()
        };
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
    let shifted = |p: &Polygon, s: f64| -> Result<Option<Polygon>, String> {
        let mut rings = Vec::with_capacity(p.rings.len());
        for (k, r) in p.rings.iter().enumerate() {
            let ir = to_plane(r, s)?;
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
    // the polygon at each whole period at which its exterior overlaps the
    // grid, or as it is without a period
    let polygon = |p: &Polygon, out: &mut Vec<Polygon>| -> Result<(), String> {
        let Some(per) = period else {
            out.extend(shifted(p, 0.0)?);
            return Ok(());
        };
        let Some(outer) = p.rings.first() else {
            return Ok(());
        };
        for s in turn_shifts(Some(per), outer.iter().map(|c| c.x))? {
            out.extend(shifted(p, s)?);
        }
        Ok(())
    };
    let mut plane = Vec::with_capacity(geoms.len());
    let mut points = Vec::new();
    for (k, g) in geoms.iter().enumerate() {
        let id = (id0 + k) as i32;
        let polys: Vec<Polygon> = match g {
            Geometry::Polygon(p) => {
                let mut v = Vec::with_capacity(1);
                polygon(p, &mut v)?;
                v
            }
            Geometry::MultiPolygon(ps) => {
                let mut v = Vec::new();
                for p in ps {
                    polygon(p, &mut v)?;
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
