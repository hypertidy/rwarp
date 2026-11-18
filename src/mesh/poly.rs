//! Meshes of polygons and their conservative weights with a grid:
//! [`PolyMesh`] (conservative stage 5, `design/conservative.md` section 7).

use bigcurve::{Edge, ForwardBatch, GreatCircle, Options, Resampler, Span, Vertex};
use controlledburn::CoverageWindow;

use crate::transform::Transformer;
use crate::warp::{cover_outline, Cover, OutlineScratch};
use crate::weights::{shift_rows, to_u32, GridWindow, WeightOptions, Weights};

/// Points of the faces transformed at once, at most (a chunk ends after
/// the face that reaches it).
const CHUNK: usize = 1 << 16;

/// A mesh of polygonal faces, triangles, hexagons or any mix, each a
/// ring of the mesh's nodes: ICON's triangles, MPAS's hexagons, a UGRID
/// or ESMF mesh. For conservative weights between the mesh and a grid,
/// both ways: [`PolyMesh::weights_onto`] (grid to mesh) and
/// [`PolyMesh::weights_to`] (mesh to grid).
///
/// - **Faces** have edges straight in the nodes' coordinates (any CRS:
///   the transformer the weights take maps them to the grid), or great
///   circles. A face
///   lists its nodes in order round it, either way round, three or more.
///   Faces that share an edge should share its nodes: the edge is then
///   bitwise the same line from both sides, and the faces tile the grid
///   with no gap or overlap.
/// - **A period** ([`PolyMesh::with_period`], 360 for a mesh in
///   longitude and latitude) makes each edge go the shorter way round in
///   x: a face with nodes at 179 and -179 is 2 degrees wide, not 358. A
///   face round a pole (its edges going once round in x) is closed along
///   the pole by the grid it is measured on. Without a period, edges are
///   taken as written.
/// - **Great-circle edges** ([`PolyMesh::with_great_circles`], an ICON or
///   MPAS mesh's): nodes in longitude and latitude in degrees, each edge
///   the shorter arc of the great circle through its ends, cut at
///   `densify` equal angles (or refined by bigcurve's great-circle model).
///   They need `densify` above 1 or a refinement tolerance to bend: at
///   densify 1 an edge is its chord in the grid's pixels. An edge across
///   the equator is cut there first: in longitude and latitude a great
///   circle's curve turns at the equator, and refinement, which tests
///   each piece's middle, can't see a turn there. A node on a
///   pole joins its two edges along their meridians: in longitude and
///   latitude the face runs along the pole from one to the other, whatever
///   longitude the node is written with (ICON puts nodes on the poles).
///
/// No CRS is involved: the type builds with `--no-default-features`.
///
/// ```
/// use rwarp::mesh::{PolyMesh, WORLD_GT};
/// use rwarp::transform::GenImgProjTransformer;
/// use rwarp::weights::{GridWindow, Normalization, WeightOptions};
/// use rwarp::CrsTransform;
///
/// // a 4 by 4 grid of unit cells over [0, 4] x [0, 4], north up
/// struct Same;
/// impl CrsTransform for Same {
///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> { Some((x, y)) }
/// }
/// let gt = [0.0, 1.0, 0.0, 4.0, 0.0, -1.0];
/// let inner = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
/// let grid = GridWindow::new(4, 4);
///
/// // two triangles: the square [0.5, 3.5] x [0.5, 3.5] cut on a diagonal
/// let (x, y) = (vec![0.5, 3.5, 3.5, 0.5], vec![0.5, 0.5, 3.5, 3.5]);
/// let mesh = PolyMesh::new(x, y, vec![0, 3, 6], vec![0, 1, 2, 0, 2, 3]).unwrap();
///
/// // grid to mesh: each triangle's area is 4.5 cells, all of it covered
/// let onto = mesh.weights_onto(&inner, &grid, &WeightOptions::new(1)).unwrap();
/// assert!(onto.dst_area.iter().all(|a| (a - 4.5).abs() < 1e-12));
/// assert!(onto.dst_covered().iter().all(|a| (a - 4.5).abs() < 1e-12));
///
/// // mesh to grid: the corner cells are a quarter covered, half by each
/// // triangle on the diagonal and all by one off it
/// let to = mesh.weights_to(&inner, &grid, &WeightOptions::new(1)).unwrap();
/// let covered = to.dst_covered();
/// assert!((covered[0] - 0.25).abs() < 1e-12 && (covered[5] - 1.0).abs() < 1e-12);
/// // a constant field on the faces stays constant where they are
/// let s = to.normalised(Normalization::FracArea);
/// let mut sum = vec![0.0; 16];
/// for k in 0..to.len() {
///     sum[to.dst_row[k] as usize * 4 + to.dst_col[k] as usize] += s[k];
/// }
/// assert!(sum.iter().all(|&v| (v - 1.0).abs() < 1e-12));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PolyMesh {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Face `f`'s nodes are `nodes[start[f]..start[f + 1]]`.
    start: Vec<usize>,
    nodes: Vec<usize>,
    period: Option<f64>,
    /// Edges are great circles, nodes longitude and latitude.
    great: bool,
}

impl PolyMesh {
    /// From the nodes' coordinates and the faces as lists of 0-based node
    /// indices, one after another: face `f`'s nodes are
    /// `nodes[start[f]..start[f + 1]]` (compressed rows, ESMF's
    /// `elementConn` with offsets from `numElementConn`).
    ///
    /// Errors when `x` and `y` differ in length, `start` doesn't run up
    /// from 0 to `nodes.len()`, a face has fewer than three nodes, or a
    /// face's node is not a node or not finite.
    pub fn new(x: Vec<f64>, y: Vec<f64>, start: Vec<usize>, nodes: Vec<usize>) -> Result<PolyMesh, String> {
        if x.len() != y.len() {
            return Err(format!("x has {} nodes and y {}", x.len(), y.len()));
        }
        if start.first() != Some(&0) || start.last() != Some(&nodes.len()) {
            return Err(format!(
                "start must run from 0 to the number of face nodes ({}), not from {:?} to {:?}",
                nodes.len(),
                start.first(),
                start.last()
            ));
        }
        for (f, w) in start.windows(2).enumerate() {
            if w[1] < w[0] + 3 {
                return Err(if w[1] < w[0] {
                    format!("start decreases at face {f}")
                } else {
                    format!("face {f} has {} nodes, fewer than three", w[1] - w[0])
                });
            }
            for &n in &nodes[w[0]..w[1]] {
                if n >= x.len() {
                    return Err(format!("face {f}: node {n} is not one of the {} nodes", x.len()));
                }
                if !(x[n].is_finite() && y[n].is_finite()) {
                    return Err(format!("face {f}: node {n} is not finite: ({}, {})", x[n], y[n]));
                }
            }
        }
        Ok(PolyMesh { x, y, start, nodes, period: None, great: false })
    }

    /// From a padded table of `width` node indices per face, face after
    /// face: UGRID's `face_node_connectivity` (with `base` its
    /// `start_index`), MPAS's `verticesOnCell` (`base` 1), ICON's
    /// `vertex_of_cell` transposed (`base` 1, `width` 3). Entries that are
    /// not a node index (below `base`, or `base + x.len()` or more), such
    /// as a fill value, are padding and left out.
    ///
    /// Errors as [`PolyMesh::new`], and when `width` is 0 or doesn't
    /// divide the table.
    pub fn from_padded(x: Vec<f64>, y: Vec<f64>, conn: &[i64], width: usize, base: i64) -> Result<PolyMesh, String> {
        if width == 0 || conn.len() % width != 0 {
            return Err(format!("a table of {} entries is not faces of {width}", conn.len()));
        }
        let n = x.len() as i64;
        let (mut start, mut nodes) = (vec![0], Vec::with_capacity(conn.len()));
        for row in conn.chunks(width) {
            nodes.extend(row.iter().filter(|&&v| v >= base && v - base < n).map(|&v| (v - base) as usize));
            start.push(nodes.len());
        }
        PolyMesh::new(x, y, start, nodes)
    }

    /// The same mesh with x periodic: each edge goes the shorter way round
    /// (see the type's docs). Errors when `period` isn't finite and
    /// positive, or isn't 360 for a mesh with great-circle edges.
    pub fn with_period(mut self, period: f64) -> Result<PolyMesh, String> {
        if !(period.is_finite() && period > 0.0) {
            return Err(format!("the period must be finite and positive, not {period}"));
        }
        if self.great && period != 360.0 {
            return Err(format!("great-circle edges are in degrees of longitude: the period is 360, not {period}"));
        }
        self.period = Some(period);
        Ok(self)
    }

    /// The same mesh with great-circle edges (see the type's docs): its
    /// nodes are longitude and latitude in degrees, and the period of
    /// longitude is 360.
    ///
    /// Errors when a face's node has a latitude beyond 90 or -90, or an
    /// edge joins two antipodes, whose great circle isn't one.
    pub fn with_great_circles(mut self) -> Result<PolyMesh, String> {
        for f in 0..self.nface() {
            let v = self.face(f);
            for (i, &n) in v.iter().enumerate() {
                if self.y[n].abs() > 90.0 {
                    return Err(format!("face {f}: node {n} has latitude {}, beyond a pole", self.y[n]));
                }
                let m = v[(i + 1) % v.len()];
                let (a, b) = (unit(self.node(n)), unit(self.node(m)));
                if dot(a, b) < -1.0 + 1e-12 {
                    return Err(format!("face {f}: the edge from node {n} to node {m} joins antipodes"));
                }
            }
        }
        self.period = Some(360.0);
        self.great = true;
        Ok(self)
    }

    /// The number of faces.
    pub fn nface(&self) -> usize {
        self.start.len() - 1
    }

    /// The number of nodes.
    pub fn nnode(&self) -> usize {
        self.x.len()
    }

    /// Face `f`'s nodes, in order round it.
    pub fn face(&self, f: usize) -> &[usize] {
        &self.nodes[self.start[f]..self.start[f + 1]]
    }

    /// Node `n`'s coordinates.
    pub fn node(&self, n: usize) -> [f64; 2] {
        [self.x[n], self.y[n]]
    }

    /// The period of x, if it has one.
    pub fn period(&self) -> Option<f64> {
        self.period
    }

    /// Whether the edges are great circles.
    pub fn great_circles(&self) -> bool {
        self.great
    }

    /// Conservative weights from a grid onto the mesh: the faces are the
    /// destination cells, the grid's cells the source.
    ///
    /// `inner` maps the mesh's coordinates to the grid's pixels: a
    /// [`Transformer`] whose destination geotransform is
    /// [`WORLD_GT`](super::WORLD_GT) and whose source is the grid, as for
    /// [`MeshTarget`](super::MeshTarget). Only cells inside `grid` (a
    /// window of the full grid) get entries.
    ///
    /// Each face is measured as [`exact_weights`](crate::weights::exact_weights)
    /// measures a destination pixel. Its outline is its edges, each cut
    /// into `opts.densify` equal pieces in the mesh's coordinates (and
    /// refined to `opts.refine_tolerance` by bigcurve, in grid pixels, if
    /// given). The end points go through `inner`, are joined by straight
    /// lines in grid pixel space, and are measured against the grid's
    /// cells there under `opts.measure`, across the seam of a grid in
    /// longitude and round its poles as a destination pixel is
    /// ([`Transformer::source_wrap`]). So:
    ///
    /// - the table is that of [`exact_weights`](crate::weights::exact_weights)
    ///   with face `f` as destination cell `(0, f)`: `dst_size` is
    ///   `[nface, 1]`, `dst_area` each face's measure (`NaN` where a point
    ///   of its outline failed), `frac` the overlap over the grid cell's
    ///   measure, and the drop, snap and rounding options apply to it. The
    ///   destination offset in `opts` is not used;
    /// - [`Weights::apply`] takes the grid's values and gives one per face.
    ///
    /// Faces that share an edge share its points exactly, so where the
    /// faces cover a grid cell, its `frac` sums to 1 over them, up to
    /// rounding. That needs a transformer that takes each point on its
    /// own: an exact one, not an
    /// [`ApproxTransformer`](crate::approx::ApproxTransformer), which
    /// interpolates along what it is given.
    ///
    /// Errors when `inner`'s source has world cells
    /// ([`Transformer::source_cells`]: geolocation arrays, a mesh), which
    /// is not supported yet, or when `opts.refine_tolerance` is not
    /// positive. Panics if a grid or face index doesn't fit in `u32`.
    pub fn weights_onto<T: Transformer>(&self, inner: &T, grid: &GridWindow, opts: &WeightOptions) -> Result<Weights, String> {
        let nface = self.nface();
        check_indices(grid, nface);
        let mut w = Weights {
            dst_off: [0, 0],
            dst_size: [nface, 1],
            dst_area: Vec::with_capacity(nface),
            measure: opts.measure,
            ..Weights::default()
        };
        self.overlaps(inner, grid, opts, |f, area, cells| {
            w.dst_area.push(area);
            for &(pc, pr, v) in cells {
                if let Some(v) = opts.weight(v) {
                    w.dst_row.push(0);
                    w.dst_col.push(f as u32);
                    w.src_row.push((grid.row_off as i64 + pr) as u32);
                    w.src_col.push((grid.col_off as i64 + pc) as u32);
                    w.frac.push(v);
                }
            }
        })?;
        Ok(w)
    }

    /// Conservative weights from the mesh to a grid: the faces are the
    /// source cells, the grid's cells (those of the window `grid`) the
    /// destination.
    ///
    /// The overlaps are [`PolyMesh::weights_onto`]'s, measured the same
    /// way through the same `inner` (from the mesh's coordinates to the
    /// grid's pixels), turned round:
    ///
    /// - entries are grouped by grid cell, row-major over the window, and
    ///   within a cell by face; `src_col` is the face (`src_row` 0);
    /// - `frac` is the overlap over the face's measure, which is in
    ///   [`Weights::src_cell_area`] (each entry's face); the drop, snap
    ///   and rounding options apply to it;
    /// - `dst_off` and `dst_size` are the window's, and `dst_area` each
    ///   cell's measure under `opts.measure` (1 for the planar measure);
    /// - [`Weights::apply`] takes one value per face, as a source buffer
    ///   `nface` wide and 1 high, and gives the grid's.
    ///
    /// Errors and panics as [`PolyMesh::weights_onto`].
    pub fn weights_to<T: Transformer>(&self, inner: &T, grid: &GridWindow, opts: &WeightOptions) -> Result<Weights, String> {
        check_indices(grid, self.nface());
        // (row, col) of the window, face, frac, the face's measure
        let mut e: Vec<(u32, u32, u32, f64, f64)> = Vec::new();
        self.overlaps(inner, grid, opts, |f, area, cells| {
            for &(pc, pr, v) in cells {
                let row = grid.row_off as i64 + pr;
                if let Some(v) = opts.weight(v * opts.measure.cell_area(row) / area) {
                    e.push((pr as u32, pc as u32, f as u32, v, area));
                }
            }
        })?;
        e.sort_by_key(|t| (t.0, t.1));
        let (r0, c0) = (grid.row_off as u32, grid.col_off as u32);
        let dst_area = (0..grid.nrow)
            .flat_map(|r| std::iter::repeat_n(opts.measure.cell_area((grid.row_off + r) as i64), grid.ncol))
            .collect();
        Ok(Weights {
            dst_row: e.iter().map(|t| r0 + t.0).collect(),
            dst_col: e.iter().map(|t| c0 + t.1).collect(),
            src_row: vec![0; e.len()],
            src_col: e.iter().map(|t| t.2).collect(),
            frac: e.iter().map(|t| t.3).collect(),
            src_cell_area: e.iter().map(|t| t.4).collect(),
            dst_area,
            dst_off: [grid.col_off, grid.row_off],
            dst_size: [grid.ncol, grid.nrow],
            measure: opts.measure,
        })
    }

    /// Each face's measure and its coverage of the window's cells, face
    /// by face: `f(face, measure, cells)` with `cells` `(col, row,
    /// coverage)` in window pixels, coverage a fraction of the cell's
    /// measure (zeros included). A face that fails has measure `NaN` and
    /// no cells.
    fn overlaps<T: Transformer>(
        &self,
        inner: &T,
        grid: &GridWindow,
        opts: &WeightOptions,
        mut f: impl FnMut(usize, f64, &[(i64, i64, f64)]),
    ) -> Result<(), String> {
        if inner.source_cells().is_some() {
            return Err("PolyMesh weights are with a grid: a transformer whose source has world cells \
                        (geolocation arrays, a mesh) isn't supported yet"
                .into());
        }
        if let Some(t) = opts.refine_tolerance {
            if t.is_nan() || t <= 0.0 {
                return Err(format!("the refinement tolerance must be positive, not {t}"));
            }
        }
        let d = opts.densify.max(1);
        let edge = MeshEdge { period: self.period, great: self.great };
        let wrap = inner.source_wrap();
        let size = (grid.ncol as i64, grid.nrow as i64);
        let (col_off, row_off) = (grid.col_off as f64, grid.row_off as f64);
        let measure = shift_rows(opts.measure, row_off);
        let mut win = CoverageWindow::with_measure(measure);
        let (mut sx, mut pieces, mut cells) = (Vec::new(), Vec::new(), Vec::new());
        let (mut rx, mut ry) = (Vec::new(), Vec::new());
        let mut pts = Points::default();
        let mut refined = Refined::default();
        let nface = self.nface();
        let mut f0 = 0;
        while f0 < nface {
            pts.clear();
            let mut f1 = f0;
            while f1 < nface && (f1 == f0 || pts.mx.len() < CHUNK) {
                self.face_points(f1, d, &edge, &mut pts);
                f1 += 1;
            }
            pts.transform(inner);
            if let Some(tol) = opts.refine_tolerance {
                refined.run(inner, edge, tol, &pts);
            }
            for (i, face) in (f0..f1).enumerate() {
                let r = pts.at[i]..pts.at[i + 1];
                cells.clear();
                if !pts.ok[r.clone()].iter().all(|&ok| ok) || refined.failed.get(i).copied().unwrap_or(false) {
                    f(face, f64::NAN, &cells);
                    continue;
                }
                rx.clear();
                ry.clear();
                for j in r {
                    rx.push(pts.x[j] - col_off);
                    ry.push(pts.y[j] - row_off);
                    if let Some(s) = refined.segment(j) {
                        rx.extend(s.iter().map(|v| v.x - col_off));
                        ry.extend(s.iter().map(|v| v.y - row_off));
                    }
                }
                let mut scratch = OutlineScratch { win: &mut win, sx: &mut sx, pieces: &mut pieces };
                let cover = cover_outline(&mut rx, &mut ry, wrap, row_off, size, &mut scratch);
                let area = match cover {
                    Cover::Failed => f64::NAN,
                    _ => measure.ring_area(&rx, &ry).abs(),
                };
                match cover {
                    Cover::Window { c0, r0, w } => {
                        for (k, &v) in win.coverage().iter().enumerate() {
                            cells.push((c0 + (k % w) as i64, r0 + (k / w) as i64, v));
                        }
                    }
                    Cover::Pieces => cells.extend_from_slice(&pieces),
                    Cover::Failed | Cover::Outside => {}
                }
                f(face, area, &cells);
            }
            f0 = f1;
        }
        Ok(())
    }

    /// Face `f`'s outline points in the mesh's coordinates, appended to
    /// `pts`: each edge's first node, then its interior points, `d - 1`
    /// of them, at equal steps (or, on a great circle, equal angles) from
    /// the edge's lexicographically smaller end (the other aligned to it
    /// by the period), so a shared edge gets bitwise the same points from
    /// both faces. With great circles, a node on a pole is two points on
    /// it, at the longitudes of the nodes before and after it, and an edge
    /// across the equator is cut there and each piece cut at `d` angles.
    fn face_points(&self, f: usize, d: usize, edge: &MeshEdge, pts: &mut Points) {
        let nodes = self.face(f);
        let m = nodes.len();
        for i in 0..m {
            let (a, b) = (self.node(nodes[i]), self.node(nodes[(i + 1) % m]));
            if self.great && a[1].abs() == 90.0 {
                let before = self.node(nodes[(i + m - 1) % m]);
                let mut any = false;
                for n in [before, b] {
                    if n[1].abs() != 90.0 {
                        pts.mx.push(n[0]);
                        pts.my.push(a[1]);
                        any = true;
                    }
                }
                if !any {
                    pts.mx.push(a[0]);
                    pts.my.push(a[1]);
                }
            } else {
                pts.mx.push(a[0]);
                pts.my.push(a[1]);
            }
            let rev = b[0] < a[0] || (b[0] == a[0] && b[1] < a[1]);
            let (s, e) = if rev { (b, a) } else { (a, b) };
            if self.great {
                // an arc across the equator is cut there, where its curve in
                // longitude and latitude turns: refinement can't see a turn
                // at a piece's middle
                let arc = Arc::new(s, e);
                let pieces = match arc.equator() {
                    Some(c) => vec![Arc::new(s, c), Arc::new(c, e)],
                    None => vec![arc],
                };
                let mut chain = Vec::with_capacity(pieces.len() * d);
                for (j, piece) in pieces.iter().enumerate() {
                    if j > 0 {
                        chain.push(piece.s);
                    }
                    chain.extend((1..d).map(|k| piece.at(k as f64 / d as f64)));
                }
                if rev {
                    chain.reverse();
                }
                for p in chain {
                    pts.mx.push(p[0]);
                    pts.my.push(p[1]);
                }
            } else if d > 1 {
                let (dx, dy) = (edge.align_x(s[0], e[0]) - s[0], e[1] - s[1]);
                for k in 1..d {
                    let t = (if rev { d - k } else { k }) as f64 / d as f64;
                    pts.mx.push(s[0] + dx * t);
                    pts.my.push(s[1] + dy * t);
                }
            }
        }
        pts.at.push(pts.mx.len());
    }
}

/// A point's unit vector, from longitude and latitude in degrees.
fn unit(p: [f64; 2]) -> [f64; 3] {
    let (lon, lat) = (p[0].to_radians(), p[1].to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A longitude difference in [-180, 180], exactly 180 keeping its sign.
fn wrap180(d: f64) -> f64 {
    let w = d - 360.0 * (d / 360.0).round();
    if w.abs() == 180.0 { 180f64.copysign(d) } else { w }
}

/// The shorter great-circle arc from `s` to `e` (longitude and latitude
/// in degrees, not antipodes).
struct Arc {
    s: [f64; 2],
    e: [f64; 2],
    a: [f64; 3],
    b: [f64; 3],
    angle: f64,
    /// A meridian's longitude: the arc's when its ends share one, the
    /// other end's when one is on a pole.
    meridian: Option<f64>,
}

impl Arc {
    fn new(s: [f64; 2], e: [f64; 2]) -> Arc {
        let (a, b) = (unit(s), unit(e));
        let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let angle = dot(c, c).sqrt().atan2(dot(a, b));
        let meridian = match (s[1].abs() == 90.0, e[1].abs() == 90.0) {
            (true, false) => Some(e[0]),
            (false, true) => Some(s[0]),
            (false, false) if wrap180(e[0] - s[0]) == 0.0 => Some(s[0]),
            _ => None,
        };
        Arc { s, e, a, b, angle, meridian }
    }

    /// Where the arc crosses the equator, strictly between its ends (not
    /// on a meridian, which crosses it straight), longitude in the
    /// start's frame.
    fn equator(&self) -> Option<[f64; 2]> {
        if self.meridian.is_some() || self.s[1] * self.e[1] >= 0.0 {
            return None;
        }
        let n = [
            self.a[1] * self.b[2] - self.a[2] * self.b[1],
            self.a[2] * self.b[0] - self.a[0] * self.b[2],
            self.a[0] * self.b[1] - self.a[1] * self.b[0],
        ];
        // the great circle's plane meets the equator's along n x z
        let mut q = [n[1], -n[0]];
        if q[0] * (self.a[0] + self.b[0]) + q[1] * (self.a[1] + self.b[1]) < 0.0 {
            q = [-q[0], -q[1]];
        }
        let lon = q[1].atan2(q[0]).to_degrees();
        Some([self.s[0] + wrap180(lon - self.s[0]), 0.0])
    }

    /// The point a fraction `t` of the way along, its longitude within 180
    /// of the start's (the start's own on a pole), or on a meridian its
    /// longitude, exactly, with latitude in proportion.
    fn at(&self, t: f64) -> [f64; 2] {
        if let Some(lon) = self.meridian {
            return [lon, self.s[1] + (self.e[1] - self.s[1]) * t];
        }
        if self.angle < 1e-15 {
            return self.s;
        }
        let sin = self.angle.sin();
        let (wa, wb) = (((1.0 - t) * self.angle).sin() / sin, (t * self.angle).sin() / sin);
        let v = [0, 1, 2].map(|k| wa * self.a[k] + wb * self.b[k]);
        let h = v[0].hypot(v[1]);
        let lat = v[2].atan2(h).to_degrees();
        if h == 0.0 {
            return [self.s[0], lat];
        }
        [self.s[0] + wrap180(v[1].atan2(v[0]).to_degrees() - self.s[0]), lat]
    }
}

/// Every full-grid index of `grid` and every face fits in `u32`, or panic.
fn check_indices(grid: &GridWindow, nface: usize) {
    to_u32(grid.col_off + grid.ncol, "grid column");
    to_u32(grid.row_off + grid.nrow, "grid row");
    to_u32(nface, "face");
}

/// A chunk of faces' outline points: in the mesh's coordinates (`mx`,
/// `my`), and through the transformer (`x`, `y`, `ok`). The chunk's face
/// `i` has points `at[i]..at[i + 1]`.
#[derive(Default)]
struct Points {
    mx: Vec<f64>,
    my: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    ok: Vec<bool>,
    at: Vec<usize>,
}

impl Points {
    fn clear(&mut self) {
        self.mx.clear();
        self.my.clear();
        self.at.clear();
        self.at.push(0);
    }

    fn transform<T: Transformer>(&mut self, inner: &T) {
        self.x.clone_from(&self.mx);
        self.y.clone_from(&self.my);
        let ok = inner.transform(true, &mut self.x, &mut self.y);
        self.ok.clear();
        self.ok.extend((0..self.x.len()).map(|i| ok[i] && self.x[i].is_finite() && self.y[i].is_finite()));
    }

    /// Point `j` as a bigcurve vertex.
    fn vertex(&self, j: usize) -> Vertex {
        Vertex { lon: self.mx[j], lat: self.my[j], x: self.x[j], y: self.y[j], ok: self.ok[j] }
    }
}

/// The new points of a chunk's segments, from bigcurve: segment `j` runs
/// from point `j` to the next round its face, and its points are
/// `out[ends[k - 1]..ends[k]]` with `k = seg[j]`. `failed[i]` is whether a
/// point face `i` needed failed to transform. Empty without a tolerance.
#[derive(Default)]
struct Refined {
    segs: Vec<(Vertex, Vertex)>,
    seg: Vec<usize>,
    out: Vec<Vertex>,
    ends: Vec<usize>,
    failed: Vec<bool>,
}

impl Refined {
    /// Refine the chunk's segments (those of faces whose points all went)
    /// to `tol`, one forward call per level. If a midpoint fails, each
    /// face is run again on its own to find which: its points are the
    /// same, since a segment's depend only on its ends.
    fn run<T: Transformer>(&mut self, inner: &T, edge: MeshEdge, tol: f64, pts: &Points) {
        let nf = pts.at.len() - 1;
        self.segs.clear();
        self.seg.clear();
        self.seg.resize(pts.mx.len(), usize::MAX);
        let mut faces = Vec::with_capacity(nf + 1);
        faces.push(0);
        for i in 0..nf {
            let r = pts.at[i]..pts.at[i + 1];
            if pts.ok[r.clone()].iter().all(|&ok| ok) {
                for j in r.clone() {
                    let next = if j + 1 == r.end { r.start } else { j + 1 };
                    self.seg[j] = self.segs.len();
                    self.segs.push((pts.vertex(j), pts.vertex(next)));
                }
            }
            faces.push(self.segs.len());
        }
        let opts = Options::new(tol);
        let mut rs = Resampler::with_edge(MeshForward { inner, failed: false }, edge, &opts);
        self.out.clear();
        self.ends.clear();
        rs.edges(&self.segs, &mut self.out, &mut self.ends);
        self.failed.clear();
        self.failed.resize(nf, false);
        if !rs.into_inner().failed {
            return;
        }
        let (mut out, mut ends) = (Vec::new(), Vec::new());
        for i in 0..nf {
            let s = &self.segs[faces[i]..faces[i + 1]];
            if s.is_empty() {
                continue;
            }
            let mut one = Resampler::with_edge(MeshForward { inner, failed: false }, edge, &opts);
            out.clear();
            ends.clear();
            one.edges(s, &mut out, &mut ends);
            self.failed[i] = one.into_inner().failed;
        }
    }

    /// The new points of the segment from point `j`, if refined.
    fn segment(&self, j: usize) -> Option<&[Vertex]> {
        let k = *self.seg.get(j)?;
        if k == usize::MAX {
            return None;
        }
        let a = if k == 0 { 0 } else { self.ends[k - 1] };
        Some(&self.out[a..self.ends[k]])
    }
}

/// The transformer as bigcurve's forward map, from the mesh's
/// coordinates to the grid's pixels, noting whether any point failed.
struct MeshForward<'a, T> {
    inner: &'a T,
    failed: bool,
}

impl<T: Transformer> ForwardBatch for MeshForward<'_, T> {
    fn forward_batch(&mut self, x: &mut [f64], y: &mut [f64], _near: &[[f64; 2]], ok: &mut [bool]) {
        let good = self.inner.transform(true, x, y);
        for i in 0..x.len() {
            ok[i] = good[i] && x[i].is_finite() && y[i].is_finite();
            self.failed |= !ok[i];
        }
    }
}

/// bigcurve's edge model for the faces' segments: planar in the mesh's
/// coordinates, the shorter way round its period if it has one, or great
/// circles (bigcurve's [`GreatCircle`]).
#[derive(Debug, Clone, Copy)]
struct MeshEdge {
    period: Option<f64>,
    great: bool,
}

impl MeshEdge {
    /// `to` moved by whole periods to within half a period of `from`
    /// (exactly half keeps the written direction); `to` itself without a
    /// period, or when it is already there.
    fn align_x(&self, from: f64, to: f64) -> f64 {
        let Some(p) = self.period else { return to };
        let d = to - from;
        let mut w = d - p * (d / p).round();
        if w.abs() == p / 2.0 {
            w = (p / 2.0).copysign(d);
        }
        let k = ((w - d) / p).round();
        if k == 0.0 { to } else { to + k * p }
    }
}

impl Edge for MeshEdge {
    fn mid(&self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        if self.great {
            return GreatCircle.mid(a, b);
        }
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
    }

    fn span(&self, a: [f64; 2], b: [f64; 2], floor: f64) -> Span {
        if self.great {
            return GreatCircle.span(a, b, floor);
        }
        if a == b { Span::Negligible } else { Span::Normal }
    }

    fn floor(&self, tolerance: f64) -> f64 {
        if self.great { GreatCircle.floor(tolerance) } else { 0.0 }
    }

    fn align(&self, from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
        if self.great {
            return GreatCircle.align(from, to);
        }
        [self.align_x(from[0], to[0]), to[1]]
    }

    fn reframe(&self, written: [f64; 2], framed: [f64; 2]) -> [f64; 2] {
        if self.great {
            return GreatCircle.reframe(written, framed);
        }
        match self.period {
            Some(p) => [((written[0] - framed[0]) / p).round() * p, 0.0],
            None => [0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_goes_the_shorter_way() {
        let e = MeshEdge { period: Some(360.0), great: false };
        assert_eq!(e.align_x(-179.0, 179.0), -181.0);
        assert_eq!(e.align_x(179.0, -179.0), 181.0);
        assert_eq!(e.align_x(10.0, 20.0), 20.0);
        assert_eq!(e.align_x(0.0, 180.0), 180.0);
        assert_eq!(e.align_x(0.0, -180.0), -180.0);
        assert_eq!(e.align_x(0.0, 540.0), 180.0);
        assert_eq!(MeshEdge { period: None, great: false }.align_x(-179.0, 179.0), 179.0);
    }

    /// Two faces share an edge, written each way: its interior points are
    /// bitwise the same from both, across the seam too.
    #[test]
    fn a_shared_edge_has_the_same_points_from_both_faces() {
        for (x, period) in [([0.1, 1.7, 0.4, 1.9], None), ([179.3, -178.9, 178.1, -179.6], Some(360.0))] {
            let y = vec![0.2, 0.9, 1.3, -0.7];
            let mut m = PolyMesh::new(x.to_vec(), y, vec![0, 3, 6], vec![0, 1, 2, 1, 0, 3]).unwrap();
            m.period = period;
            let edge = MeshEdge { period, great: false };
            for d in [3, 7] {
                let mut pts = Points::default();
                pts.clear();
                m.face_points(0, d, &edge, &mut pts);
                m.face_points(1, d, &edge, &mut pts);
                // face 0 runs 0 -> 1 first, face 1 runs 1 -> 0 first
                let a: Vec<(u64, u64)> = (1..d).map(|k| (pts.mx[k].to_bits(), pts.my[k].to_bits())).collect();
                let mut b: Vec<(u64, u64)> = (1..d).map(|k| (pts.mx[3 * d + k].to_bits(), pts.my[3 * d + k].to_bits())).collect();
                b.reverse();
                assert_eq!(a, b, "densify {d}, period {period:?}");
            }
        }
    }

    /// Longitude and latitude to a polar plane, which bends every edge.
    struct Polar;

    impl Transformer for Polar {
        fn transform(&self, _: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
            for (x, y) in x.iter_mut().zip(y.iter_mut()) {
                let r = 90.0 - *y;
                (*x, *y) = (r * x.to_radians().sin(), -r * x.to_radians().cos());
            }
            vec![true; x.len()]
        }
    }

    /// Refined, a shared edge across the seam gets bitwise the same new
    /// points from both faces.
    #[test]
    fn a_shared_edge_refines_to_the_same_points() {
        let (x, y) = (vec![170.0, -160.0, 175.0, -170.0], vec![60.0, 75.0, 80.0, 55.0]);
        let m = PolyMesh::new(x, y, vec![0, 3, 6], vec![0, 1, 2, 1, 0, 3]).unwrap().with_period(360.0).unwrap();
        let edge = MeshEdge { period: m.period, great: false };
        let mut pts = Points::default();
        pts.clear();
        m.face_points(0, 1, &edge, &mut pts);
        m.face_points(1, 1, &edge, &mut pts);
        pts.transform(&Polar);
        let mut r = Refined::default();
        r.run(&Polar, edge, 1e-3, &pts);
        let bits = |v: &[Vertex]| v.iter().map(|p| (p.x.to_bits(), p.y.to_bits())).collect::<Vec<_>>();
        // face 0's segment 0 runs 0 -> 1, face 1's first (point 3) 1 -> 0
        let (a, mut b) = (bits(r.segment(0).unwrap()), bits(r.segment(3).unwrap()));
        b.reverse();
        assert!(a.len() > 10, "{}", a.len());
        assert_eq!(a, b);
    }

    /// With great circles: two faces share an edge from a node on the
    /// pole and two faces an edge across the seam, each written both ways.
    /// The interior points, and the pole's point at the edge's longitude,
    /// are bitwise the same from both, fixed and refined.
    #[test]
    fn great_circle_edges_are_the_same_from_both_faces() {
        // nodes: the pole (written at lon 17), A, B, C; D, E across 180
        let x = vec![17.0, 40.0, 95.0, -20.0, 179.0, -176.5, 170.0];
        let y = vec![90.0, 70.0, 75.0, 72.0, 10.0, 22.0, 30.0];
        let m = PolyMesh::new(x, y, vec![0, 3, 6, 9, 12], vec![0, 1, 2, 1, 0, 3, 4, 5, 6, 5, 4, 2]).unwrap();
        let m = m.with_great_circles().unwrap();
        let edge = MeshEdge { period: m.period, great: true };
        let bits = |pts: &Points, r: std::ops::Range<usize>| r.map(|k| (pts.mx[k].to_bits(), pts.my[k].to_bits())).collect::<Vec<_>>();
        for d in [1, 4] {
            let mut pts = Points::default();
            pts.clear();
            for f in 0..4 {
                m.face_points(f, d, &edge, &mut pts);
            }
            // face 0: the pole as (lon B, 90), (lon A, 90), then P -> A's
            // interior; face 1: A, A -> P's interior, then (lon A, 90)
            let a = bits(&pts, 1..d + 1);
            let at1 = pts.at[1];
            let mut b = bits(&pts, at1 + 1..at1 + d + 1);
            b.reverse();
            assert_eq!(a, b, "the pole's edge, densify {d}");
            assert_eq!((pts.mx[1], pts.my[1]), (40.0, 90.0));
            // face 2 runs D -> E first, face 3 E -> D first
            let (at2, at3) = (pts.at[2], pts.at[3]);
            let a = bits(&pts, at2 + 1..at2 + d);
            let mut b = bits(&pts, at3 + 1..at3 + d);
            b.reverse();
            assert_eq!(a, b, "across the seam, densify {d}");
            pts.transform(&Polar);
            let mut r = Refined::default();
            r.run(&Polar, edge, 1e-4, &pts);
            let refined = |j: usize| r.segment(j).unwrap().iter().map(|p| (p.x.to_bits(), p.y.to_bits())).collect::<Vec<_>>();
            if d == 1 {
                // the segment from (lon A, 90) to A in face 0 is point 1's;
                // in face 1, from A to (lon A, 90) is point at1's
                let (a, mut b) = (refined(1), refined(at1));
                b.reverse();
                assert_eq!(a, b, "the pole's edge, refined");
                let (a, mut b) = (refined(at2), refined(at3));
                b.reverse();
                assert!(!a.is_empty());
                assert_eq!(a, b, "across the seam, refined");
            }
        }
    }

    /// An arc's points are at equal angles along it, and its cut at the
    /// equator is on it, between its ends. A shared edge across the
    /// equator gets bitwise the same points, cut included, from both faces.
    #[test]
    fn arcs_are_cut_at_equal_angles_and_at_the_equator() {
        let angle = |p: [f64; 2], q: [f64; 2]| dot(unit(p), unit(q)).clamp(-1.0, 1.0).acos();
        let (s, e) = ([-10.0, -20.0], [30.0, 25.0]);
        let arc = Arc::new(s, e);
        let steps: Vec<f64> = (0..8).map(|k| angle(arc.at(k as f64 / 8.0), arc.at((k + 1) as f64 / 8.0))).collect();
        assert!(steps.iter().all(|a| (a - steps[0]).abs() < 1e-12), "{steps:?}");
        let c = arc.equator().unwrap();
        assert_eq!(c[1], 0.0);
        assert!((angle(s, c) + angle(c, e) - angle(s, e)).abs() < 1e-12);
        assert!(Arc::new([10.0, 5.0], [20.0, 30.0]).equator().is_none());
        assert!(Arc::new([10.0, -5.0], [10.0, 30.0]).equator().is_none(), "a meridian");
        let m = PolyMesh::new(vec![-10.0, 30.0, 0.0, 20.0], vec![-20.0, 25.0, 30.0, -30.0], vec![0, 3, 6], vec![0, 1, 2, 1, 0, 3]);
        let m = m.unwrap().with_great_circles().unwrap();
        let edge = MeshEdge { period: m.period, great: true };
        for d in [1, 3] {
            let mut pts = Points::default();
            pts.clear();
            m.face_points(0, d, &edge, &mut pts);
            m.face_points(1, d, &edge, &mut pts);
            // the edge's 2 d - 1 points after its first node, each way
            let n = 2 * d - 1;
            let a: Vec<_> = (1..1 + n).map(|k| (pts.mx[k].to_bits(), pts.my[k].to_bits())).collect();
            let at1 = pts.at[1];
            let mut b: Vec<_> = (at1 + 1..at1 + 1 + n).map(|k| (pts.mx[k].to_bits(), pts.my[k].to_bits())).collect();
            b.reverse();
            assert_eq!(a, b, "densify {d}");
            assert_eq!(pts.my[d], 0.0, "the cut, densify {d}");
        }
    }

    #[test]
    fn padded_tables_drop_their_fill() {
        let (x, y) = (vec![0.0, 1.0, 1.0, 0.0, 2.0], vec![0.0, 0.0, 1.0, 1.0, 0.5]);
        let m = PolyMesh::from_padded(x.clone(), y.clone(), &[1, 2, 3, 4, 2, 5, 3, -1], 4, 1).unwrap();
        assert_eq!(m.nface(), 2);
        assert_eq!(m.face(0), &[0, 1, 2, 3]);
        assert_eq!(m.face(1), &[1, 4, 2]);
        assert!(PolyMesh::from_padded(x.clone(), y.clone(), &[1, 2, 0, 0], 4, 1).is_err());
        assert!(PolyMesh::from_padded(x, y, &[1, 2, 3], 4, 1).is_err());
    }

    #[test]
    fn bad_meshes_are_refused() {
        let (x, y) = (vec![0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0]);
        assert!(PolyMesh::new(x.clone(), vec![0.0; 2], vec![0, 3], vec![0, 1, 2]).is_err());
        assert!(PolyMesh::new(x.clone(), y.clone(), vec![0, 2], vec![0, 1]).is_err());
        assert!(PolyMesh::new(x.clone(), y.clone(), vec![0, 3], vec![0, 1, 3]).is_err());
        assert!(PolyMesh::new(x.clone(), y.clone(), vec![1, 3], vec![0, 1, 2]).is_err());
        assert!(PolyMesh::new(vec![0.0, f64::NAN, 1.0], y.clone(), vec![0, 3], vec![0, 1, 2]).is_err());
        let m = PolyMesh::new(x, y, vec![0, 3], vec![0, 1, 2]).unwrap();
        assert!(m.clone().with_period(0.0).is_err());
        assert_eq!(m.with_period(360.0).unwrap().period(), Some(360.0));
    }
}
